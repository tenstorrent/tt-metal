#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Run the Streamlit client for YOLOv11 pose estimation web demo
cd /home/ubuntu/pose/tt-metal/models/demos/yolov11/pose_web_demo/client

# Default values
SERVER_URL="${SERVER_URL:-http://localhost:8000}"
DEVICE="${DEVICE:-0}"

# Run streamlit with server URL and device parameters
streamlit run yolov11_pose.py --server.port 8501 --server.address 0.0.0.0 -- --server-url "$SERVER_URL" --device "$DEVICE"
