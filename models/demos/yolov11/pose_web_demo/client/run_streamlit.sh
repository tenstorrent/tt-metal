#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Run the Streamlit client for YOLOv11 pose estimation web demo
cd /home/ubuntu/pose/tt-metal/models/demos/yolov11/pose_web_demo/client

# Default values
SERVER_URL="${SERVER_URL:-http://localhost:8000}"
DEVICE="${DEVICE:-0}"
ENABLE_HTTPS="${ENABLE_HTTPS:-false}"

echo "Starting YOLOv11 Pose Estimation Client..."
echo "Server URL: $SERVER_URL"
echo "Camera Device: $DEVICE"
echo ""
echo "⚠️  IMPORTANT: Camera access requires HTTPS!"
echo "If you see 'navigator.mediaDevices is undefined', use one of:"
echo "1. HTTPS: ./run_streamlit_https.sh"
echo "2. Ngrok: ngrok http 8501 (then visit https://xxxxx.ngrok.io)"
echo "3. Local development: Use Chrome with --allow-http-screen-capture"
echo ""
echo "Make sure the pose estimation server is running and accessible!"
echo "If using remote server, ensure SSH port forwarding is active."
echo ""

# Run streamlit with server URL and device parameters
if [ "$ENABLE_HTTPS" = "true" ]; then
    echo "Running with HTTPS support..."
    streamlit run yolov11_pose.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false -- --server-url "$SERVER_URL" --device "$DEVICE"
else
    streamlit run yolov11_pose.py --server.port 8501 --server.address 0.0.0.0 -- --server-url "$SERVER_URL" --device "$DEVICE"
fi
