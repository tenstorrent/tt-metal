#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# Exit on error
set -e

echo "=== SDXL Standalone Server Launcher ==="

# Setup logging
LOG_FILE="sdxl_server_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1
echo "Logging to: $LOG_FILE"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_METAL_HOME="$SCRIPT_DIR"

# Activate Python environment
echo "Activating Python environment..."
if [ ! -f "${TT_METAL_HOME}/python_env/bin/activate" ]; then
    echo "Error: Python environment not found at ${TT_METAL_HOME}/python_env"
    echo "Please run: ./create_venv.sh"
    exit 1
fi

source "${TT_METAL_HOME}/python_env/bin/activate"

# Set Python path
export PYTHONPATH="${TT_METAL_HOME}"

# Set library path
export LD_LIBRARY_PATH="${TT_METAL_HOME}/build/lib:${LD_LIBRARY_PATH}"

# Set device environment variables (all 4 devices for T3K)
export TT_VISIBLE_DEVICES="0,1,2,3"
export TT_METAL_VISIBLE_DEVICES="0,1,2,3"

# Set mesh configuration
export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="7,7"

# Set HuggingFace cache to local directory (auto-download enabled)
# Using local cache instead of /mnt/MLPerf which requires special permissions
export HF_HOME="${HOME}/.cache/huggingface"

# Enable persistent caching for faster subsequent startups
# Use default cache (~/.cache/tt-metal-cache) to share with tt-media-server
# This allows reusing pre-compiled programs that work with l1_small_size=30000
# export TT_METAL_CACHE="${HOME}/.cache/tt_metal_sdxl"
export TTNN_CONFIG_OVERRIDES='{"enable_model_cache": true, "model_cache_path": "'${HOME}'/.cache/ttnn/models"}'

# Performance settings
export TT_MM_THROTTLE_PERF=5

# Logging
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo ""
echo "Environment configured:"
echo "  TT_METAL_HOME: $TT_METAL_HOME"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  TT_VISIBLE_DEVICES: $TT_VISIBLE_DEVICES"
echo "  HF_HOME: $HF_HOME"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Check devices
echo ""
echo "Checking TT devices..."
TT_SMI_VENV="/home/tt-admin/tt-smi/venv"

if [ ! -f "${TT_SMI_VENV}/bin/activate" ]; then
    echo "Warning: tt-smi venv not found at ${TT_SMI_VENV}"
else
    # Use tt-smi snapshot for non-interactive device listing
    source "${TT_SMI_VENV}/bin/activate"

    echo "Available devices:"
    tt-smi -s --snapshot_no_tty 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    devices = data.get('device_info', [])
    if not devices:
        print('  No devices found')
        sys.exit(1)
    for idx, dev in enumerate(devices):
        board_info = dev.get('board_info', {})
        bus_id = board_info.get('bus_id', 'N/A')
        board_type = board_info.get('board_type', 'Unknown')
        # Only show local devices (those with valid bus IDs)
        if bus_id != 'N/A':
            print(f'  Device {idx}: {bus_id} ({board_type})')
except Exception as e:
    print(f'  Error checking devices: {e}')
    sys.exit(1)
" || echo "Warning: Could not verify TT devices"

    deactivate
fi

# Parse --clear-cache flag (must happen before server starts)
for arg in "$@"; do
    if [ "$arg" = "--clear-cache" ]; then
        echo ""
        echo "Clearing caches..."
        rm -rf "${HOME}/.cache/tt-metal-cache"
        rm -rf "${HOME}/.cache/ttnn/models"
        echo "✓ Caches cleared (Note: This will trigger recompilation)"
        break
    fi
done

# Parse --dev flag
DEV_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--dev" ]; then
        DEV_MODE=true
        export SDXL_DEV_MODE=true
        export TT_VISIBLE_DEVICES="0"
        export TT_METAL_VISIBLE_DEVICES="0"
        echo ""
        echo "*** DEV MODE: Single worker, fast startup ***"
        break
    fi
done

# Filter out --clear-cache from arguments (handled by this script)
PYTHON_ARGS=()
for arg in "$@"; do
    if [ "$arg" != "--clear-cache" ]; then
        PYTHON_ARGS+=("$arg")
    fi
done

# Start server
echo ""
if [ "$DEV_MODE" = "true" ]; then
    echo "Starting SDXL server in DEV MODE..."
    echo "This will take 5-8 minutes for initial warmup."
else
    echo "Starting SDXL server..."
    echo "This will take 5-10 minutes for initial warmup."
fi
echo "Server will be ready when you see: 'All workers ready. Server is accepting requests.'"
echo "Press Ctrl+C to stop the server."
echo ""

"${TT_METAL_HOME}/python_env/bin/python" "${TT_METAL_HOME}/sdxl_server.py" "${PYTHON_ARGS[@]}"
