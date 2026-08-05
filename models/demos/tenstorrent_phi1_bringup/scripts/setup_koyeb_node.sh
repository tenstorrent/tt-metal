#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Quick setup script for stateless Tenstorrent N300 Koyeb instances (`microsoft/phi-1` bring-up)

set -e

echo "==================================================================="
echo "  Tenstorrent N300 Koyeb Node Initialization (#18287 Phi-1 Stage 1)"
echo "==================================================================="

# 1. Check system and TT-Metal environment
if command -v tt-smi &> /dev/null; then
    echo "[1/3] Checking Tenstorrent Wormhole (N300) hardware status..."
    tt-smi -s
else
    echo "[1/3] tt-smi not found in PATH. Ensure Tenstorrent drivers are loaded on the Koyeb base image."
fi

# 2. Install required Python libraries into /opt/venv
echo "[2/3] Checking/installing Python dependencies ('transformers', 'torch', 'ttnn')..."
if command -v uv &> /dev/null; then
    uv pip install --python /opt/venv/bin/python3 transformers torch accelerate
else
    /usr/bin/pip install --target /opt/venv/lib/python3.10/site-packages transformers torch accelerate
fi

# 3. Verify TTNN API accessibility
/opt/venv/bin/python3 -c "
try:
    import ttnn
    print('[OK] TTNN API successfully imported version:', getattr(ttnn, '__version__', 'unknown'))
except Exception as e:
    print('[WARNING] TTNN import check:', e)
"

echo "==================================================================="
echo "  Node Setup Complete! Run demo with:"
echo "  PYTHONPATH=. /opt/venv/bin/python3 demo/demo.py"
echo "==================================================================="
