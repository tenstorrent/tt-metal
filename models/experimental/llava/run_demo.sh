#!/bin/bash
# LLaVA Demo Runner — handles environment setup automatically
#
# Usage:
#   bash models/demos/llava/run_demo.sh                              # Full demo (image → text)
#   bash models/demos/llava/run_demo.sh --vision-only                 # Vision pipeline only
#   bash models/demos/llava/run_demo.sh --image img.jpg --question "?" # Custom image + question
#   bash models/demos/llava/run_demo.sh --mesh N150                   # Single chip

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESTORE_CPP=false
RESTORE_TT_METAL=false

# Cleanup function
cleanup() {
    if [ "$RESTORE_CPP" = true ] && [ -d "$TT_METAL_ROOT/ttnn/cpp_backup" ]; then
        mv "$TT_METAL_ROOT/ttnn/cpp_backup" "$TT_METAL_ROOT/ttnn/cpp"
        echo "✅ Restored ttnn/cpp"
    fi
    if [ "$RESTORE_TT_METAL" = true ] && [ -d "$TT_METAL_ROOT/tt_metal_backup" ]; then
        mv "$TT_METAL_ROOT/tt_metal_backup" "$TT_METAL_ROOT/tt_metal"
        echo "✅ Restored tt_metal"
    fi
}
trap cleanup EXIT

# Parse args
MESH_DEVICE=${MESH_DEVICE:-N300}
DEMO_TYPE="full"
PASSTHROUGH_ARGS=()

# Python interpreter: PYTHON_BIN overrides the auto-detection order below
# (repo build venv -> /opt/venv -> whatever `python3` resolves to)
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -x "$TT_METAL_ROOT/python_env/bin/python" ] && "$TT_METAL_ROOT/python_env/bin/python" -c "import ttnn.device" >/dev/null 2>&1; then
        PYTHON_BIN="$TT_METAL_ROOT/python_env/bin/python"
    elif [ -x /opt/venv/bin/python ] && /opt/venv/bin/python -c "import ttnn.device" >/dev/null 2>&1; then
        PYTHON_BIN=/opt/venv/bin/python
    else
        PYTHON_BIN=python3
    fi
fi
# Sanity check: the interpreter must have a working ttnn with ttnn.device
if ! "$PYTHON_BIN" -c "import ttnn.device" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN has no working ttnn.device module (stale/broken ttnn install?)" >&2
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --mesh)
            MESH_DEVICE="$2"
            shift 2
            ;;
        --vision-only)
            DEMO_TYPE="vision"
            shift
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# Auto-detect ttnn installation
if [ -f "$TT_METAL_ROOT/ttnn/ttnn/_ttnn.cpython-310-x86_64-linux-gnu.so" ]; then
    export TT_METAL_HOME="$TT_METAL_ROOT"
    export PYTHONPATH="$TT_METAL_ROOT/ttnn:$TT_METAL_ROOT:$PYTHONPATH"
    echo "ttnn source: Built from source tree"
else
    export TT_METAL_HOME="/opt/venv/lib/python3.10/site-packages/ttnn"
    export PYTHONPATH="$TT_METAL_ROOT:$PYTHONPATH"
    echo "ttnn source: Pre-installed wheel (/opt/venv)"
    if [ -d "$TT_METAL_ROOT/ttnn/cpp" ]; then
        mv "$TT_METAL_ROOT/ttnn/cpp" "$TT_METAL_ROOT/ttnn/cpp_backup"
        RESTORE_CPP=true
        echo "⚠️  Temporarily hid ttnn/cpp"
    fi
    # Also hide tt_metal/ — the source tree's fabric/impl/kernels/edm_fabric/
    # contains a newer fabric_erisc_router.cpp that uses symbols not declared
    # in the installed wheel's headers (e.g. sender_ch_wait_static_connection),
    # causing an erisc build failure during fabric initialization.
    if [ -d "$TT_METAL_ROOT/tt_metal" ]; then
        mv "$TT_METAL_ROOT/tt_metal" "$TT_METAL_ROOT/tt_metal_backup"
        RESTORE_TT_METAL=true
        echo "⚠️  Temporarily hid tt_metal"
    fi
fi

export MESH_DEVICE

echo "Device: $MESH_DEVICE"
echo ""

# Reset device
echo "Resetting device..."
tt-smi -r --no_reinit 2>/dev/null || true
sleep 2

# Run demo
if [ "$DEMO_TYPE" = "vision" ]; then
    echo "Running: Vision Pipeline Demo"
    "$PYTHON_BIN" "$SCRIPT_DIR/demo/demo_vision_pipeline.py" "${PASSTHROUGH_ARGS[@]}"
else
    echo "Running: Full LLaVA Demo (image + prompt → text)"
    "$PYTHON_BIN" "$SCRIPT_DIR/demo/demo_llava_full.py" "${PASSTHROUGH_ARGS[@]}"
fi
