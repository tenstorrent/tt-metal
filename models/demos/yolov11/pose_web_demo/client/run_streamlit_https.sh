#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Run the Streamlit client with HTTPS for camera access
cd /home/ubuntu/pose/tt-metal/models/demos/yolov11/pose_web_demo/client

# Default values
SERVER_URL="${SERVER_URL:-http://YOUR_UBUNTU_IP:8000}"  # Replace YOUR_UBUNTU_IP with actual IP
DEVICE="${DEVICE:-0}"

echo "Starting YOLOv11 Pose Estimation Client with HTTPS..."
echo "Server URL: $SERVER_URL"
echo "Camera Device: $DEVICE"
echo ""

# Create self-signed certificates if they don't exist
if [ ! -f "cert.pem" ] || [ ! -f "key.pem" ]; then
    echo "Generating self-signed SSL certificates..."

    # Create a temporary config file for SAN
    cat > ssl.conf << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = State
L = City
O = Organization
CN = localhost

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = 127.0.0.1
IP.1 = 127.0.0.1
IP.2 = 0.0.0.0
EOF

    # Generate certificate with SAN
    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -config ssl.conf

    # Clean up
    rm ssl.conf

    echo "SSL certificates created: cert.pem, key.pem"
    echo "⚠️  Note: You may see a security warning in your browser - click 'Advanced' and 'Proceed to localhost (unsafe)'"
    echo ""
fi

echo "Make sure the pose estimation server is running and accessible!"
echo "If using remote server, ensure SSH port forwarding is active."
echo ""
echo "🔒 HTTPS Server starting..."
echo "📱 Access the client at: https://localhost:8501"
echo "   (NOT https://0.0.0.0:8501 - use localhost or 127.0.0.1)"
echo ""
echo "🔐 When you see 'Your connection is not private':"
echo "   1. Click 'Advanced'"
echo "   2. Click 'Proceed to localhost (unsafe)'"
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
