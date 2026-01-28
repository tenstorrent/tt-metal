#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Run Face Recognition FastAPI server
# Requires: source python_env/bin/activate (for ttnn)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# server -> web_demo -> sface -> experimental -> models -> tt-metal
TT_METAL_ROOT="$SCRIPT_DIR/../../../../.."
cd "$TT_METAL_ROOT" || exit 1

echo "Running from: $(pwd)"

# Check if python_env is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Error: Please activate python_env first:"
    echo "  source python_env/bin/activate"
    exit 1
fi

# Install dependencies to user site-packages
echo "Installing dependencies..."
/usr/bin/python3 -m pip install -q fastapi uvicorn python-multipart Pillow

# Run server with access to both virtualenv (ttnn) and user site-packages (uvicorn)
echo "Starting Face Recognition server on http://0.0.0.0:8000"
PYTHONPATH="$VIRTUAL_ENV/lib/python3.10/site-packages:$PYTHONPATH" \
/usr/bin/python3 -m uvicorn models.experimental.sface.web_demo.server.fast_api_face_recognition:app --host 0.0.0.0 --port 8000 --reload
