#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# Exit on error
set -e

echo "=== Unified Image Generation Server Launcher ==="

# ---------------------------------------------------------------------------
# Parse --model flag (must be done before any environment setup)
# ---------------------------------------------------------------------------
MODEL="sdxl"
for arg in "$@"; do
    if [ "$arg" = "--model" ]; then
        # Next argument will be the model name — handled by the loop below
        NEXT_IS_MODEL=true
    elif [ "${NEXT_IS_MODEL:-false}" = "true" ]; then
        MODEL="$arg"
        NEXT_IS_MODEL=false
    fi
done

if [ "$MODEL" != "sdxl" ] && [ "$MODEL" != "sd35" ]; then
    echo "Error: --model must be 'sdxl' or 'sd35' (got: '$MODEL')"
    exit 1
fi

echo "Model: $MODEL"

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
LOG_FILE="${MODEL}_server_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1
echo "Logging to: $LOG_FILE"

# ---------------------------------------------------------------------------
# Resolve script directory
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_METAL_HOME="$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Activate Python environment
# ---------------------------------------------------------------------------
echo "Activating Python environment..."
if [ ! -f "${TT_METAL_HOME}/python_env/bin/activate" ]; then
    echo "Error: Python environment not found at ${TT_METAL_HOME}/python_env"
    echo "Please run: ./create_venv.sh"
    exit 1
fi

source "${TT_METAL_HOME}/python_env/bin/activate"

# ---------------------------------------------------------------------------
# Common environment variables (both models)
# ---------------------------------------------------------------------------
export PYTHONPATH="${TT_METAL_HOME}"
export LD_LIBRARY_PATH="${TT_METAL_HOME}/build/lib:${LD_LIBRARY_PATH}"
export HF_HOME="${HOME}/.cache/huggingface"
export TTNN_CONFIG_OVERRIDES='{"enable_model_cache": true, "model_cache_path": "'${HOME}'/.cache/ttnn/models"}'
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

# ---------------------------------------------------------------------------
# Model-specific environment variables
# ---------------------------------------------------------------------------
if [ "$MODEL" = "sdxl" ]; then
    # T3K: 4 devices, 7x7 grid, MM throttle
    export TT_VISIBLE_DEVICES="0,1,2,3"
    export TT_METAL_VISIBLE_DEVICES="0,1,2,3"
    export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="7,7"
    export TT_MM_THROTTLE_PERF=5
elif [ "$MODEL" = "sd35" ]; then
    # LoudBox: 8 devices, no grid override, no throttle
    export TT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    export TT_METAL_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    # No TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE for SD3.5
    # No TT_MM_THROTTLE_PERF for SD3.5
fi

# ---------------------------------------------------------------------------
# Check devices via tt-smi (optional, continues if not available)
# ---------------------------------------------------------------------------
echo ""
echo "Checking TT devices..."
TT_SMI_VENV="/home/tt-admin/tt-smi/venv"

if [ ! -f "${TT_SMI_VENV}/bin/activate" ]; then
    echo "Warning: tt-smi venv not found at ${TT_SMI_VENV}"
else
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
        if bus_id != 'N/A':
            print(f'  Device {idx}: {bus_id} ({board_type})')
except Exception as e:
    print(f'  Error checking devices: {e}')
    sys.exit(1)
" || echo "Warning: Could not verify TT devices"

    deactivate
fi

# ---------------------------------------------------------------------------
# Handle --clear-cache flag
# ---------------------------------------------------------------------------
for arg in "$@"; do
    if [ "$arg" = "--clear-cache" ]; then
        echo ""
        echo "Clearing caches..."
        rm -rf "${HOME}/.cache/tt-metal-cache"
        rm -rf "${HOME}/.cache/ttnn/models"
        echo "Caches cleared (this will trigger recompilation on next startup)"
        break
    fi
done

# ---------------------------------------------------------------------------
# Handle --dev flag
# ---------------------------------------------------------------------------
DEV_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--dev" ]; then
        DEV_MODE=true
        if [ "$MODEL" = "sdxl" ]; then
            # SDXL dev mode: single device only
            export SDXL_DEV_MODE=true
            export TT_VISIBLE_DEVICES="0"
            export TT_METAL_VISIBLE_DEVICES="0"
            echo ""
            echo "*** SDXL DEV MODE: Single worker, fast startup ***"
        elif [ "$MODEL" = "sd35" ]; then
            # SD3.5 dev mode: keep all 8 devices but use fewer steps and no trace
            export SD35_DEV_MODE=true
            echo ""
            echo "*** SD3.5 DEV MODE: Reduced steps, no trace capture ***"
        fi
        break
    fi
done

# ---------------------------------------------------------------------------
# Print effective environment
# ---------------------------------------------------------------------------
echo ""
echo "Environment configured:"
echo "  MODEL:           $MODEL"
echo "  TT_METAL_HOME:   $TT_METAL_HOME"
echo "  PYTHONPATH:      $PYTHONPATH"
echo "  TT_VISIBLE_DEVICES: $TT_VISIBLE_DEVICES"
echo "  HF_HOME:         $HF_HOME"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# ---------------------------------------------------------------------------
# Build Python argument list (strip script-only flags)
# ---------------------------------------------------------------------------
PYTHON_ARGS=()
SKIP_NEXT=false
for arg in "$@"; do
    if [ "${SKIP_NEXT}" = "true" ]; then
        SKIP_NEXT=false
        continue
    fi
    if [ "$arg" = "--clear-cache" ]; then
        # Handled above, not passed to Python
        continue
    fi
    if [ "$arg" = "--model" ]; then
        # --model is consumed here; pass it along to server.py too
        PYTHON_ARGS+=("$arg")
        SKIP_NEXT=false  # Next value will be appended naturally
        continue
    fi
    PYTHON_ARGS+=("$arg")
done

# ---------------------------------------------------------------------------
# Start server
# ---------------------------------------------------------------------------
echo ""
if [ "$DEV_MODE" = "true" ]; then
    echo "Starting ${MODEL} server in DEV MODE..."
    if [ "$MODEL" = "sd35" ]; then
        echo "This will take 2-5 minutes (no trace capture in dev mode)."
    else
        echo "This will take 5-8 minutes for initial warmup."
    fi
else
    echo "Starting ${MODEL} server..."
    if [ "$MODEL" = "sd35" ]; then
        echo "This will take 15-25 minutes for first-run trace capture."
    else
        echo "This will take 5-10 minutes for initial warmup."
    fi
fi
echo "Server will be ready when you see: 'All workers ready. ${MODEL} server is accepting requests.'"
echo "Press Ctrl+C to stop the server."
echo ""

"${TT_METAL_HOME}/python_env/bin/python" "${TT_METAL_HOME}/server.py" "${PYTHON_ARGS[@]}"
