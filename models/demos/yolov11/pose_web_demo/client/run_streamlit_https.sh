#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Run the Streamlit client with HTTPS for camera access
cd /home/ubuntu/pose/tt-metal/models/demos/yolov11/pose_web_demo/client

# Default values
SERVER_URL="${SERVER_URL:-http://localhost:8000}"
DEVICE="${DEVICE:-0}"

echo "Starting YOLOv11 Pose Estimation Client with HTTPS..."
echo "Server URL: $SERVER_URL"
echo "Camera Device: $DEVICE"
echo ""

# Create self-signed certificates if they don't exist
if [ ! -f "cert.pem" ] || [ ! -f "key.pem" ]; then
    echo "Generating self-signed SSL certificates..."
    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    echo "SSL certificates created: cert.pem, key.pem"
    echo "⚠️  Note: You may see a security warning in your browser - click 'Advanced' and 'Proceed to localhost'"
    echo ""
fi

echo "Make sure the pose estimation server is running and accessible!"
echo "If using remote server, ensure SSH port forwarding is active."
echo ""

# Run streamlit with HTTPS
ENABLE_HTTPS=true streamlit run yolov11_pose.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.sslCertFile cert.pem \
    --server.sslKeyFile key.pem \
    -- --server-url "$SERVER_URL" --device "$DEVICE"
