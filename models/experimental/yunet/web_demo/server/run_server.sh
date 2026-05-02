#!/bin/bash
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Run YuNet FastAPI server on Tenstorrent hardware

set -e

# Navigate to tt-metal root (5 levels up from server/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_METAL_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

cd "$TT_METAL_ROOT"
export TT_METAL_HOME="$TT_METAL_ROOT"
export PYTHONPATH="$TT_METAL_ROOT"

echo "============================================"
echo "  YuNet Face Detection - FastAPI Server"
echo "============================================"
echo ""
echo "TT_METAL_HOME: $TT_METAL_HOME"
echo "Server will start on: http://0.0.0.0:8000"
echo ""

# Install dependencies if needed
if ! python -c "import uvicorn" 2>/dev/null; then
    echo "Installing dependencies..."
    uv pip install -q fastapi uvicorn python-multipart Pillow
fi

# Start the server
python -m uvicorn models.experimental.yunet.web_demo.server.fast_api_yunet:app \
    --host 0.0.0.0 \
    --port 8000
