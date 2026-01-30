#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Run YuNet Streamlit client

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  YuNet Face Detection - Streamlit Client"
echo "============================================"
echo ""
echo "Make sure the FastAPI server is running first!"
echo "Client will start on: http://0.0.0.0:8501"
echo ""

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "ERROR: streamlit not found. Install with:"
    echo "  pip install streamlit streamlit-webrtc opencv-python requests av"
    exit 1
fi

# Start the client
python -m streamlit run yunet_streamlit.py \
    --server.port 8501 \
    --server.address 0.0.0.0
