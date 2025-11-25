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

# Start server
echo ""
echo "Starting SDXL server..."
echo "This will take 5-10 minutes for initial warmup."
echo "Server will be ready when you see: 'All workers ready. Server is accepting requests.'"
echo "Press Ctrl+C to stop the server."
echo ""

python "${TT_METAL_HOME}/sdxl_server.py"
