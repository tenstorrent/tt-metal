#!/bin/bash
# Run from repo root inside python_env.
# Usage: bash models/experimental/rfdetr_medium/demo/run_streamlit.sh
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
source python_env/bin/activate
python3 -m streamlit run models/experimental/rfdetr_medium/demo/streamlit_demo.py --server.port 8501 --server.headless true
