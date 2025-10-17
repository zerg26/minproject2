"""
SMART HOME BUYING ASSISTANT - STREAMLIT APP
Complete 3-agent system with UI
"""

import streamlit as st
from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tavily import TavilyClient
import os
from datetime import datetime

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Smart Home Buying Assistant",
    page_icon="🏠",
    layout="wide"
)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
if 'final_report' not in st.session_state:
    st.session_state.final_report = None
if 'full_state' not in st.session_state:
    st.session_state.full_state = None

# ============================================
# SIDEBAR - API KEYS
# ============================================

with st.sidebar:
    st.title("Configuration")
    
    groq_key = st.text_input(
        "GROQ API Key",
        type="password",
        help="Get from https://console.groq.com"
    )
    
    tavily_key = st.text_input(
        "Tavily API Key",
        type="password",
        help="Get from https://tavily.com"
    )
    
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key
    
    st.divider()
    
    st.markdown("""
    ### How It Works
    
    **3-Agent System:**
    
    1. **Agent 1** - Searches properties
    2. **Agent 2** - Analyzes & ranks
    3. **Agent 3** - Creates report
    
    **Your Inputs:**
    - Location (e.g., Charlotte, NC)
    - Bedrooms
    - Budget
    - Priorities (schools, safety, etc.)
    """)

# ============================================
# STATE DEFINITION
# ============================================

class CompleteState(TypedDict):
    query: str
    preferences: Optional[Dict]
    results: Optional[list]
    agent1_final: Optional[str]
    user_preferences: Optional[Dict]
    analysis: Optional[str]
    summary: Optional[str]
    final_recommendation: Optional[str]
    report_metadata: Optional[Dict]

# ============================================
# AGENT FUNCTIONS
# ============================================

model_name = "llama-3.3-70b-versatile"

def get_tavily_client():
    """Get Tavily client with error handling"""
    if not os.environ.get("TAVILY_API_KEY"):
        st.error("Please enter Tavily API Key in sidebar!")
        st.stop()
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def get_llm(temp=0.3):
    """Get LLM with error handling"""
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please enter GROQ API Key in sidebar!")
        st.stop()
    return ChatGroq(model=model_name, temperature=temp)


# AGENT 1
def agent1_search(state: CompleteState) -> CompleteState:
    """Agent 1: Search properties"""
    
    query = state["query"][:800]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a real estate assistant. Format the user's query for optimized property search in 400 words or less."),
        ("user", f"Query: {query}")
    ])
    
    chain = prompt | ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=800)
    refined_query = chain.invoke({"query": query}).content
    
    if len(refined_query) > 800:
        refined_query = refined_query[:397].rsplit(" ", 1)[0] + "..."
    
    tavily_client = get_tavily_client()
    search_results = tavily_client.search(
        query=refined_query,
        include_domains=["zillow.com/homedetails"],
        max_results=5
    )
    
    state["preferences"] = {"refined_query": refined_query}
    state["results"] = search_results
    
    return state


def agent1_format(state: CompleteState) -> CompleteState:
    """Agent 1: Format results"""
    
    results = state["results"].get("results", [])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Format houses in readable manner. Include if relevant:
crime rate, price, property type, commute to uptown, school district, features.
Do not include preamble."""),
        ("user", "Search results: {results}")
    ])
    
    chain = prompt | ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
    response = chain.invoke({"results": results})
    
    state["agent1_final"] = response.content
    
    return state


# AGENT 2
def agent2_analyze(state: CompleteState) -> CompleteState:
    """Agent 2: Analyze properties"""
    
    agent1_output = state["agent1_final"]
    
    user_prefs = st.session_state.get('user_prefs', {
        "budget_priority": "High",
        "safety_priority": "High",
        "commute_priority": "Medium",
        "schools_priority": "High"
    })
    
    state["user_preferences"] = user_prefs
    
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a real estate analyst. Analyze each property and rate 1-10 based on user preferences. Consider: price, crime, commute, schools, features."),
        ("user", "Properties:\n{properties}\n\nUser Preferences:\n{preferences}\n\nAnalyze each with rating, strengths, weaknesses.")
    ])
    
    pref_text = "\n".join([f"- {k}: {v}" for k, v in user_prefs.items()])
    
    llm = get_llm(0.3)
    chain = analysis_prompt | llm
    
    response = chain.invoke({
        "properties": agent1_output,
        "preferences": pref_text
    })
    
    state["analysis"] = response.content
    
    return state


def agent2_summarize(state: CompleteState) -> CompleteState:
    """Agent 2: Create summary"""
    
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a real estate advisor. Rank properties from best to worst and provide clear recommendation."),
        ("user", "Analysis:\n{analysis}\n\nRank properties 1-3 and recommend the best.")
    ])
    
    llm = get_llm(0.3)
    chain = summary_prompt | llm
    
    response = chain.invoke({"analysis": state["analysis"]})
    
    state["summary"] = response.content
    
    return state


# AGENT 3
def agent3_generate_report(state: CompleteState) -> CompleteState:
    """Agent 3: Generate final report"""
    
    report_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional real estate advisor. Create a clear, well-formatted final recommendation report."),
        ("user", """Create final report:

ANALYSIS: {analysis}

SUMMARY: {summary}

USER PRIORITIES: {preferences}

---

Format with:

HOME BUYING RECOMMENDATION REPORT
Date: {date}
Location: North Carolina

EXECUTIVE SUMMARY
[3-4 sentences recommending #1 property]

TOP 3 PROPERTIES

Property #1: [Name] - Rating: X/10 
- Price: $XXX,XXX
- STRENGTHS ✓ [list]
- WEAKNESSES ⚠ [list]
- Best for: [buyer type]

[Repeat for #2 and #3]

FINAL RECOMMENDATION
RECOMMENDED: Property #1 - [Name]
[2-3 paragraphs explaining why]
CONFIDENCE: High/Medium/Low

FINANCIAL ESTIMATE
- Price: $XXX,XXX
- Down Payment (20%): $XX,XXX
- Monthly Cost: $X,XXX

NEXT STEPS
1. Schedule showing
2. Get pre-approved
3. Conduct inspection
4. Review HOA docs
5. Make offer

Be specific with details.""")
    ])
    
    prefs_text = "\n".join([f"• {k}: {v}" for k, v in state["user_preferences"].items()])
    current_date = datetime.now().strftime("%B %d, %Y")
    
    llm = get_llm(0.2)
    chain = report_prompt | llm
    
    response = chain.invoke({
        "analysis": state["analysis"],
        "summary": state["summary"],
        "preferences": prefs_text,
        "date": current_date
    })
    
    state["final_recommendation"] = response.content
    state["report_metadata"] = {
        "status": "success",
        "generated_date": current_date,
        "num_properties": 3
    }
    
    return state

# ============================================
# BUILD WORKFLOW
# ============================================

@st.cache_resource
def build_workflow():
    """Build the complete workflow"""
    
    workflow = StateGraph(CompleteState)
    
    workflow.add_node("agent1_search", agent1_search)
    workflow.add_node("agent1_format", agent1_format)
    workflow.add_node("agent2_analyze", agent2_analyze)
    workflow.add_node("agent2_summarize", agent2_summarize)
    workflow.add_node("agent3_report", agent3_generate_report)
    
    workflow.set_entry_point("agent1_search")
    workflow.add_edge("agent1_search", "agent1_format")
    workflow.add_edge("agent1_format", "agent2_analyze")
    workflow.add_edge("agent2_analyze", "agent2_summarize")
    workflow.add_edge("agent2_summarize", "agent3_report")
    workflow.add_edge("agent3_report", END)
    
    return workflow.compile()

# ============================================
# MAIN UI
# ============================================

st.title("Smart Home Buying Assistant")
st.markdown("*Powered by AI - Find your perfect home with intelligent property analysis*")

# Check API keys
if not (os.environ.get("GROQ_API_KEY") and os.environ.get("TAVILY_API_KEY")):
    st.warning("Please enter your API keys in the sidebar to get started!")
    st.stop()

# Input section
st.subheader("What are you looking for?")

col1, col2 = st.columns([3, 1])

with col1:
    user_query = st.text_area(
        "Describe your ideal home:",
        placeholder="e.g., I want a 3-bedroom house in Charlotte, NC under $450k with good schools and low crime rate",
        height=100,
        key="query_input"
    )

st.divider()
st.subheader("What matters most to you?")

col1, col2 = st.columns(2)

with col1:
    budget_priority = st.selectbox(
        "Budget Priority",
        ["High - Must stay under budget", "Medium - Some flexibility", "Low - Budget is flexible"],
        index=0
    )
    
    safety_priority = st.selectbox(
        "Safety/Crime Priority",
        ["Very High - Low crime essential", "High - Prefer low crime", "Medium - Not critical"],
        index=0
    )

with col2:
    schools_priority = st.selectbox(
        "School District Priority",
        ["Very High - Excellent schools essential", "High - Good schools important", "Medium - Not critical"],
        index=1
    )
    
    commute_priority = st.selectbox(
        "Commute Time Priority",
        ["High - Must be under 20 min", "Medium - Prefer under 30 min", "Low - Not critical"],
        index=1
    )

# Store in session state
st.session_state.user_prefs = {
    "budget_priority": budget_priority,
    "safety_priority": safety_priority,
    "commute_priority": commute_priority,
    "schools_priority": schools_priority
}


# Search button
search_clicked = st.button("🔍 Find My Perfect Home", type="primary", disabled=not user_query)

# Run workflow
if search_clicked and user_query:
    
    st.session_state.report_generated = False
    st.session_state.final_report = None
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize state
    initial_state = {
        "query": user_query,
        "preferences": None,
        "results": None,
        "agent1_final": None,
        "user_preferences": None,
        "analysis": None,
        "summary": None,
        "final_recommendation": None,
        "report_metadata": None
    }
    
    try:
        # Build workflow
        app = build_workflow()
        
        # Agent 1
        status_text.text("Agent 1: Searching properties...")
        progress_bar.progress(20)
        
        # Agent 2
        status_text.text("Agent 2: Analyzing properties...")
        progress_bar.progress(50)
        
        # Agent 3
        status_text.text("Agent 3: Generating report...")
        progress_bar.progress(80)
        
        # Run complete workflow
        final_state = app.invoke(initial_state)
        
        progress_bar.progress(100)
        status_text.text("Complete!")
        
        # Store in session
        st.session_state.final_report = final_state["final_recommendation"]
        st.session_state.full_state = final_state
        st.session_state.report_generated = True
        
        # Clear progress after 1 sec
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        progress_bar.empty()
        status_text.empty()

# Display report
if st.session_state.report_generated and st.session_state.final_report:
    
    st.divider()
    
    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("Your Personalized Recommendation Report")
    
    with col2:
        # Download button
        st.download_button(
            label="Download Report",
            data=st.session_state.final_report,
            file_name=f"home_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    with col3:
        if st.button("New Search"):
            st.session_state.clear()
            st.rerun()
    
    # Display report in nice container
    with st.container():
        st.markdown(st.session_state.final_report)
    
    # Show intermediate outputs in expandable sections
    st.divider()
    st.subheader("Behind the Scenes")
    
    with st.expander("Agent 1 Output - Property Search"):
        st.text(st.session_state.full_state.get("agent1_final", ""))
    
    with st.expander("Agent 2 Output - Analysis"):
        st.text(st.session_state.full_state.get("analysis", ""))
    
    with st.expander("Agent 2 Output - Rankings"):
        st.text(st.session_state.full_state.get("summary", ""))

# ============================================
# AGENT 1: PROPERTY SEARCH
# ============================================

def agent1_search(state: CompleteState) -> CompleteState:
    """Search for properties"""
    
    query = state["query"][:800]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a real estate assistant. Format query for optimized property search in 400 words or less."),
        ("user", f"Query: {query}")
    ])
    
    chain = prompt | ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=800)
    refined_query = chain.invoke({"query": query}).content
    
    if len(refined_query) > 800:
        refined_query = refined_query[:397].rsplit(" ", 1)[0] + "..."
    
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    search_results = tavily_client.search(
        query=refined_query,
        include_domains=["zillow.com/homedetails"],
        max_results=3
    )
    
    state["preferences"] = {"refined_query": refined_query}
    state["results"] = search_results
    
    return state


def agent1_format(state: CompleteState) -> CompleteState:
    """Format search results"""
    
    results = state["results"].get("results", [])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Format houses in readable manner. Include: crime rate, price, property type, commute, school district, features. No preamble."""),
        ("user", "Search results: {results}")
    ])
    
    chain = prompt | ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
    response = chain.invoke({"results": results})
    
    state["agent1_final"] = response.content
    
    return state


# AGENT 2
def agent2_analyze(state: CompleteState) -> CompleteState:
    """Analyze and rate properties"""
    
    user_prefs = {
        "budget_priority": "High - want best value under $450k",
        "safety_priority": "Very High - low crime is essential",
        "commute_priority": "Medium - prefer under 30 min",
        "schools_priority": "High - good schools important",
        "other_notes": "Willing to pay more for safety and schools"
    }
    
    state["user_preferences"] = user_prefs
    
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "Real estate analyst. Analyze each property and rate 1-10 based on user preferences. Consider: price, crime, commute, schools, features."),
        ("user", "Properties:\n{properties}\n\nUser Preferences:\n{preferences}\n\nAnalyze each with rating, strengths, weaknesses.")
    ])
    
    pref_text = "\n".join([f"- {k}: {v}" for k, v in user_prefs.items()])
    
    llm = ChatGroq(model=model_name, temperature=0.3)
    chain = analysis_prompt | llm
    
    response = chain.invoke({
        "properties": state["agent1_final"],
        "preferences": pref_text
    })
    
    state["analysis"] = response.content
    
    return state


def agent2_summarize(state: CompleteState) -> CompleteState:
    """Create summary and rankings"""
    
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Real estate advisor. Rank properties from best to worst and provide clear recommendation."),
        ("user", "Analysis:\n{analysis}\n\nRank properties 1-3 and recommend best.")
    ])
    
    llm = ChatGroq(model=model_name, temperature=0.3)
    chain = summary_prompt | llm
    
    response = chain.invoke({"analysis": state["analysis"]})
    
    state["summary"] = response.content
    
    return state


# AGENT 3
def agent3_generate_report(state: CompleteState) -> CompleteState:
    """Generate final comprehensive report"""
    
    report_prompt = ChatPromptTemplate.from_messages([
        ("system", "Professional real estate advisor. Create clear, well-formatted final recommendation report."),
        ("user", """Create report:

ANALYSIS: {analysis}

SUMMARY: {summary}

PRIORITIES: {preferences}

---

Format with:

HOME BUYING RECOMMENDATION REPORT
Date: {date}
Location: North Carolina

EXECUTIVE SUMMARY
[3-4 sentences recommending #1 property]

TOP 3 PROPERTIES

Property #1: [Name] - Rating: X/10 
- Price: $XXX,XXX
- STRENGTHS ✓ [list]
- WEAKNESSES ⚠ [list]

[Repeat for #2 and #3]

FINAL RECOMMENDATION
RECOMMENDED: Property #1 - [Name]
[2-3 paragraphs why]
CONFIDENCE: High/Medium/Low

FINANCIAL ESTIMATE
- Price: $XXX,XXX
- Down Payment (20%): $XX,XXX
- Monthly Cost: $X,XXX

NEXT STEPS
1. Schedule showing
2. Get pre-approved
3. Conduct inspection
4. Review HOA docs
5. Make offer""")
    ])
    
    prefs_text = "\n".join([f"• {k}: {v}" for k, v in state["user_preferences"].items()])
    current_date = datetime.now().strftime("%B %d, %Y")
    
    llm = ChatGroq(model=model_name, temperature=0.2)
    chain = report_prompt | llm
    
    response = chain.invoke({
        "analysis": state["analysis"],
        "summary": state["summary"],
        "preferences": prefs_text,
        "date": current_date
    })
    
    state["final_recommendation"] = response.content
    state["report_metadata"] = {
        "status": "success",
        "generated_date": current_date
    }
    
    return state


# ============================================
# FOOTER
# ============================================

st.divider()
st.caption("🏠 Smart Home Buying Assistant | Powered by AI | Built with LangGraph, Groq, and Tavily")