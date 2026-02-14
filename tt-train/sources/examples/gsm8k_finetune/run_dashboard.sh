#!/bin/bash
# Simple script to run the LLM Fine-tuning Dashboard

echo "=========================================="
echo "LLM Fine-tuning Dashboard"
echo "=========================================="
echo ""

# Check if requirements are installed
echo "Checking dependencies..."
python3 -c "import streamlit; import plotly; import pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install -r requirements_streamlit.txt
fi

echo ""
echo "Starting dashboard..."
echo "The dashboard will open in your browser at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "=========================================="
echo ""

# Run the Streamlit app
streamlit run streamlit_finetune_app.py
