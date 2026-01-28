#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Entrypoint script for Face Matching API container

set -e

# Check if tt-metal is mounted
if [ ! -d "/tt-metal" ]; then
    echo "ERROR: tt-metal not mounted at /tt-metal"
    echo "Please mount tt-metal: -v /path/to/tt-metal:/tt-metal"
    exit 1
fi

# Set up environment
export TT_METAL_HOME=/tt-metal
export PYTHONPATH=/tt-metal:/app:$PYTHONPATH
export LD_LIBRARY_PATH=/tt-metal/build/lib:$LD_LIBRARY_PATH

# Check for Tenstorrent device
if [ ! -e "/dev/tenstorrent" ]; then
    echo "WARNING: /dev/tenstorrent not found. Running in CPU-only mode may not work."
fi

# Change to tt-metal directory for proper imports
cd /tt-metal

case "$1" in
    serve)
        echo "Starting Face Matching API server..."
        echo "  - TT_METAL_HOME: $TT_METAL_HOME"
        echo "  - API endpoint: http://0.0.0.0:8000"
        echo "  - Verify endpoint: POST /api/v1/verify"
        echo ""
        exec python3 -m uvicorn models.experimental.sface.web_demo.server.fast_api_face_recognition:app \
            --host 0.0.0.0 \
            --port 8000
        ;;
    shell)
        echo "Starting interactive shell..."
        exec /bin/bash
        ;;
    *)
        exec "$@"
        ;;
esac
