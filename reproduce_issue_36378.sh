#!/bin/bash
# reproduce_issue_36378.sh
#
# Prerequisites:
#   - T3K system with 8 Wormhole chips and working fabric links (run tt-topology -l mesh
#     if QSFP links are down)
#   - tt-metal built (./build_metal.sh)
#   - Python venv created (./create_venv.sh)
#   - Llama-3.2-90B-Vision-Instruct weights downloaded into $HF_HOME
#     (huggingface-cli download meta-llama/Llama-3.2-90B-Vision-Instruct)
#
# Usage:
#   bash reproduce_issue_36378.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_FILE="models/tt_transformers/tests/test_attention.py"
ORIGINAL_PCC_LINE='    pcc = 0.986  # pcc reduced from .99 while investigating issue #36378'
PATCHED_PCC_LINE='    pcc = 0.99'

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Reproducing GitHub Issue #36378"
echo " t3k_llama3.2-90b attention PCC regression"
echo "============================================================"
echo ""

source python_env/bin/activate
export TT_METAL_HOME="$REPO_ROOT"
export PYTHONPATH="$TT_METAL_HOME:${PYTHONPATH:-}"
export HF_HOME="/localdev/$USER/hf"
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
export TT_CACHE_HOME="/localdev/$USER/hf/tt_cache"
export MESH_DEVICE=T3K

HF_MODEL="meta-llama/Llama-3.2-90B-Vision-Instruct"
TT_CACHE_PATH="$TT_CACHE_HOME/$HF_MODEL"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "[1/5] Running pre-flight checks..."

if [ ! -f build/lib/libtt_metal.so ]; then
    echo "ERROR: tt-metal is not built. Run ./build_metal.sh first." >&2
    exit 1
fi

if [ ! -d "$HF_HOME/hub/models--meta-llama--Llama-3.2-90B-Vision-Instruct" ]; then
    echo "ERROR: Llama-3.2-90B-Vision-Instruct weights not found in $HF_HOME." >&2
    echo "       Download with: huggingface-cli download meta-llama/Llama-3.2-90B-Vision-Instruct" >&2
    exit 1
fi

NUM_DEVICES=$(python3 -c "import ttnn; print(ttnn.get_num_devices())" 2>/dev/null)
if [ "$NUM_DEVICES" -lt 8 ]; then
    echo "ERROR: Need 8 TT devices for T3K tests, found $NUM_DEVICES." >&2
    echo "       If fabric links are down, run: tt-topology -l mesh" >&2
    exit 1
fi

echo "  Build:   OK"
echo "  Weights: OK"
echo "  Devices: $NUM_DEVICES"
echo ""

# ---------------------------------------------------------------------------
# Patch: restore the original 0.99 PCC threshold
# ---------------------------------------------------------------------------
echo "[2/5] Patching PCC threshold from 0.986 -> 0.99 in $TEST_FILE..."

if ! grep -qF "$ORIGINAL_PCC_LINE" "$TEST_FILE"; then
    echo "ERROR: Could not find the expected PCC line to patch in $TEST_FILE." >&2
    echo "       The source may have changed. Look for the pcc assignment near line 71." >&2
    exit 1
fi

sed -i "s|$ORIGINAL_PCC_LINE|$PATCHED_PCC_LINE|" "$TEST_FILE"
echo "  Done."
echo ""

# ---------------------------------------------------------------------------
# Ensure we always revert the patch, even on failure
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "[5/5] Reverting PCC threshold patch..."
    cd "$REPO_ROOT"
    sed -i "s|$PATCHED_PCC_LINE|$ORIGINAL_PCC_LINE|" "$TEST_FILE"
    echo "  Reverted $TEST_FILE to original state."
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Run the failing test
# ---------------------------------------------------------------------------
echo "[3/5] Running test_attention_inference[batch_size=1, default_attention]..."
echo "  Model: $HF_MODEL"
echo "  This will take ~30-60 seconds."
echo ""

set +e
HF_MODEL="$HF_MODEL" TT_CACHE_PATH="$TT_CACHE_PATH" \
    pytest --timeout 900 -v -s "$TEST_FILE" \
    -k "256-1-page_params0-default_attention" 2>&1 | tee /tmp/issue_36378_output.log
TEST_EXIT=$?
set -e

echo ""

# ---------------------------------------------------------------------------
# Report results
# ---------------------------------------------------------------------------
echo "[4/5] Results"
echo "============================================================"

PCC_VALUES=$(grep "PCC:" /tmp/issue_36378_output.log 2>/dev/null || true)
FAILURES=$(grep "Attention Failed" /tmp/issue_36378_output.log 2>/dev/null || true)
MIN_PCC=$(grep "PCC:" /tmp/issue_36378_output.log 2>/dev/null \
    | grep -oP '[\d.]+$' \
    | sort -n \
    | head -1 || echo "N/A")

if [ $TEST_EXIT -ne 0 ]; then
    echo ""
    echo "  ISSUE REPRODUCED: test exited with code $TEST_EXIT"
    echo ""
    echo "  Lowest PCC observed: $MIN_PCC"
    echo ""
    if [ -n "$FAILURES" ]; then
        echo "  Failing positions:"
        echo "$FAILURES" | sed 's/^/    /'
    fi
    echo ""
    echo "  The PCC drops below 0.99 (to ~0.986) at certain decode positions"
    echo "  because PR #36177 changed the intermediate CB data format from"
    echo "  Float16_b to Float32 in the DRAM-sharded matmul, which paradoxically"
    echo "  worsens the Pearson correlation for this model."
else
    echo ""
    echo "  Tests passed (PCC stayed above 0.99 on this run)."
    echo "  Lowest PCC observed: $MIN_PCC"
    echo ""
    echo "  NOTE: The failure can be non-deterministic. Re-run the script to retry."
fi

echo "============================================================"
echo ""
echo "Full log saved to /tmp/issue_36378_output.log"
