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
# Detect available TT devices via tt-smi, set TT_VISIBLE_DEVICES, and validate
# ---------------------------------------------------------------------------
echo ""
echo "Detecting TT devices..."
TT_SMI_VENV="/home/tt-admin/tt-smi/venv"
DETECTED_DEVICE_COUNT=0
DETECTED_DEVICE_IDS=""

if [ ! -f "${TT_SMI_VENV}/bin/activate" ]; then
    echo "Warning: tt-smi venv not found at ${TT_SMI_VENV}, skipping device detection"
else
    source "${TT_SMI_VENV}/bin/activate"

    DEVICE_INFO=$(tt-smi -s --snapshot_no_tty 2>/dev/null | python3 -c '
import sys, json
try:
    data = json.load(sys.stdin)
    # Count ALL devices: PCIe-attached (n300 L) and remote (n300 R).
    # On a LoudBox, 4 n300 boards each have L (PCIe) + R (ethernet) chip = 8 total.
    # The R chips have bus_id="N/A" but are still valid TT devices.
    devices = data.get("device_info", [])
    pcie_ids = [str(i) for i, dev in enumerate(devices)
                if dev.get("board_info", {}).get("bus_id", "N/A") != "N/A"]
    ids = ",".join(pcie_ids)
    print(f"{len(devices)}:{ids}")
    for i, dev in enumerate(devices):
        board_info = dev.get("board_info", {})
        bus = board_info.get("bus_id", "N/A")
        btype = board_info.get("board_type", "Unknown")
        loc = "(PCIe)" if bus != "N/A" else "(remote)"
        print(f"  Device {i}: {btype} {loc} bus={bus}", file=sys.stderr)
except Exception as e:
    print(f"0:", file=sys.stdout)
    print(f"  Error: {e}", file=sys.stderr)
' 2>/tmp/tt_device_info_$$.txt)

    cat /tmp/tt_device_info_$$.txt  # Print device list to stdout
    rm -f /tmp/tt_device_info_$$.txt

    DETECTED_DEVICE_COUNT=$(echo "$DEVICE_INFO" | cut -d: -f1)
    DETECTED_DEVICE_IDS=$(echo "$DEVICE_INFO" | cut -d: -f2)

    echo "Detected ${DETECTED_DEVICE_COUNT} TT device(s) (PCIe + remote)"

    deactivate
fi

# ---------------------------------------------------------------------------
# Set TT_VISIBLE_DEVICES based on detected devices (not hardcoded)
# SD35 requires exactly 8 devices (2x4 mesh). SDXL requires at least 1.
# ---------------------------------------------------------------------------
if [ "$MODEL" = "sdxl" ]; then
    if [ "${DETECTED_DEVICE_COUNT}" -gt 0 ] 2>/dev/null && [ -n "$DETECTED_DEVICE_IDS" ]; then
        # Use detected IDs, capped at 4 for SDXL (T3K: 4 devices)
        SDXL_IDS=$(echo "$DETECTED_DEVICE_IDS" | python3 -c "
import sys
ids = sys.stdin.read().strip().split(',')
print(','.join(ids[:4]))
")
        export TT_VISIBLE_DEVICES="$SDXL_IDS"
        export TT_METAL_VISIBLE_DEVICES="$SDXL_IDS"
    else
        # Fall back to assuming 4 devices (T3K default)
        export TT_VISIBLE_DEVICES="0,1,2,3"
        export TT_METAL_VISIBLE_DEVICES="0,1,2,3"
    fi
    export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="7,7"
    export TT_MM_THROTTLE_PERF=5

elif [ "$MODEL" = "sd35" ]; then
    # SD3.5 requires 8 devices (2x4 mesh). Fail fast with a clear error if unavailable.
    if [ "${DETECTED_DEVICE_COUNT}" -gt 0 ] 2>/dev/null; then
        if [ "${DETECTED_DEVICE_COUNT}" -lt 8 ]; then
            echo ""
            echo "ERROR: SD3.5 requires 8 TT devices (LoudBox, 2x4 mesh)."
            echo "       Only ${DETECTED_DEVICE_COUNT} device(s) detected."
            echo ""
            echo "SD3.5 uses a 2x4 mesh with CFG parallelism — the pipeline"
            echo "requires at least 8 chips and cannot run on T3K (4 devices)."
            echo ""
            echo "To run SDXL on this machine: ./launch_server.sh --model sdxl"
            exit 1
        fi
        # Use exactly the first 8 detected device IDs
        SD35_IDS=$(echo "$DETECTED_DEVICE_IDS" | python3 -c "
import sys
ids = sys.stdin.read().strip().split(',')
print(','.join(ids[:8]))
")
        export TT_VISIBLE_DEVICES="$SD35_IDS"
        export TT_METAL_VISIBLE_DEVICES="$SD35_IDS"
    else
        echo "Warning: Could not detect device count. Assuming 8 devices for SD3.5."
        export TT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
        export TT_METAL_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    fi
    # No TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE for SD3.5
    # No TT_MM_THROTTLE_PERF for SD3.5
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
