#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Run Face Recognition Streamlit client
# Note: This client runs separately from the TT device server
# It only needs streamlit and basic packages (no ttnn required)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Install dependencies (to user site-packages)
echo "Installing dependencies..."
/usr/bin/python3 -m pip install -q -r requirements.txt

# Run client with system python (which has streamlit in user site-packages)
echo "Starting Face Recognition client on http://localhost:8501"
/usr/bin/python3 -m streamlit run face_recognition_streamlit.py --server.port 8501
