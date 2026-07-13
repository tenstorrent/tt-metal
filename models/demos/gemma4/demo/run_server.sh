#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_METAL_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

cd "$TT_METAL_ROOT"
export TT_METAL_HOME="$TT_METAL_ROOT"
export PYTHONPATH="$TT_METAL_ROOT"

if ! python -c "import uvicorn" 2>/dev/null; then
    echo "Installing server dependencies..."
    uv pip install -r models/demos/gemma4/demo/requirements-server.txt
fi

HOST="${GEMMA4_SERVER_HOST:-0.0.0.0}"
PORT="${GEMMA4_SERVER_PORT:-8000}"

echo "Gemma4 FastAPI server on http://${HOST}:${PORT}"
echo "HF_MODEL=${HF_MODEL:-<unset>}"
echo "TT_CACHE_PATH=${TT_CACHE_PATH:-<unset>}"

exec python -m uvicorn models.demos.gemma4.demo.serve_fastapi:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level info
