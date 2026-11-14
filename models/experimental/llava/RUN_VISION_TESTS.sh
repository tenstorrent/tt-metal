#!/bin/bash
# LLaVA CLIP Encoder Test Runner
# Run this after tt-metal build completes successfully
#
# IMPORTANT: This script auto-detects whether to use the source-tree ttnn build
# or the pre-installed ttnn wheel from /opt/venv. If the source tree is not built
# (no _ttnn.so), it falls back to the installed package which has compiled kernels.
# When using the installed wheel, the source tree's newer kernel sources
# (ttnn/cpp/) are temporarily hidden to prevent JIT compilation failures caused
# by version mismatch between source-tree kernels and installed-package headers.

echo "============================================"
echo "LLaVA CLIP Encoder Test Suite"
echo "============================================"
echo ""

TT_METAL_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"
RESTORE_CPP=false
RESTORE_TT_METAL=false

# Cleanup function: restore hidden directories if they were moved
cleanup() {
    if [ "$RESTORE_CPP" = true ] && [ -d "$TT_METAL_ROOT/ttnn/cpp_backup" ]; then
        mv "$TT_METAL_ROOT/ttnn/cpp_backup" "$TT_METAL_ROOT/ttnn/cpp"
        echo "✅ Restored ttnn/cpp (source tree kernels)"
    fi
    if [ "$RESTORE_TT_METAL" = true ] && [ -d "$TT_METAL_ROOT/tt_metal_backup" ]; then
        mv "$TT_METAL_ROOT/tt_metal_backup" "$TT_METAL_ROOT/tt_metal"
        echo "✅ Restored tt_metal (source tree kernels)"
    fi
}
trap cleanup EXIT

# Auto-detect ttnn installation: source tree build vs installed wheel
# The source tree's ttnn/ttnn/__init__.py will shadow the installed package
# if PYTHONPATH includes $TT_METAL_ROOT/ttnn, but it needs _ttnn.so to work.
if [ -f "$TT_METAL_ROOT/ttnn/ttnn/_ttnn.cpython-310-x86_64-linux-gnu.so" ]; then
    # Source tree is built — use it directly
    export TT_METAL_HOME="$TT_METAL_ROOT"
    export PYTHONPATH="$TT_METAL_ROOT/ttnn:$TT_METAL_ROOT:$PYTHONPATH"
    echo "ttnn source: Built from source tree"
else
    # Source tree not built — use pre-installed wheel from /opt/venv
    # CRITICAL: Do NOT add $TT_METAL_ROOT/ttnn to PYTHONPATH, as the source
    # tree's ttnn/ttnn/__init__.py would shadow the installed package and fail
    # with "ModuleNotFoundError: No module named 'ttnn._ttnn'" (no compiled .so).
    export TT_METAL_HOME="/opt/venv/lib/python3.10/site-packages/ttnn"
    export PYTHONPATH="$TT_METAL_ROOT:$PYTHONPATH"
    echo "ttnn source: Pre-installed wheel (/opt/venv)"

    # Hide source tree kernels to prevent JIT from finding incompatible versions.
    # The source tree's ttnn/cpp/ has newer kernel code that uses named compile-time
    # args not supported by the installed package's compile_time_args.h headers.
    # The JIT compiler adds source-tree kernel paths as include directories, causing
    # "'__builtin_unreachable()' is not a constant expression" compilation errors.
    if [ -d "$TT_METAL_ROOT/ttnn/cpp" ]; then
        mv "$TT_METAL_ROOT/ttnn/cpp" "$TT_METAL_ROOT/ttnn/cpp_backup"
        RESTORE_CPP=true
        echo "⚠️  Temporarily hid ttnn/cpp (source tree kernels incompatible with installed wheel)"
    fi
    # Also hide tt_metal/ — the source tree's fabric/impl/kernels/edm_fabric/
    # contains a newer fabric_erisc_router.cpp that uses symbols not declared
    # in the installed wheel's headers (e.g. sender_ch_wait_static_connection),
    # causing an erisc build failure during fabric initialization.
    if [ -d "$TT_METAL_ROOT/tt_metal" ]; then
        mv "$TT_METAL_ROOT/tt_metal" "$TT_METAL_ROOT/tt_metal_backup"
        RESTORE_TT_METAL=true
        echo "⚠️  Temporarily hid tt_metal (source tree fabric kernels incompatible with installed wheel)"
    fi
fi
export MESH_DEVICE=N300

echo "Environment Setup:"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  TT_METAL_HOME: $TT_METAL_HOME"
echo "  MESH_DEVICE: $MESH_DEVICE"
echo ""

set -e  # Exit on error (disabled inside run_test for exit code capture)

# Navigate to tt-metal directory
cd "$TT_METAL_ROOT"

# ============================================
# Device Reset and Cache Clearing
# ============================================
echo "Preparing device for testing..."
echo ""

echo "⚙️  Resetting n300 device..."
if tt-smi -r --no_reinit; then
    echo "✅ Device reset complete"
else
    echo "⚠️  Device reset command failed, continuing anyway..."
fi
sleep 2

echo ""
echo "⚙️  Clearing kernel cache for fresh JIT compilation..."
if [ -d ~/.cache/tt-metal-cache/ ]; then
    rm -rf ~/.cache/tt-metal-cache/
    echo "✅ Cache cleared"
else
    echo "ℹ️  Cache directory doesn't exist (fresh state)"
fi
echo ""

echo "============================================"
echo ""

# Function to run a test
run_test() {
    local test_name=$1
    local test_path=$2

    echo "============================================"
    echo "Running: $test_name"
    echo "============================================"
    # Disable set -e so we can capture the exit code
    set +e
    scripts/run_safe_pytest.sh "$test_path" -v -s
    local exit_code=$?
    set -e

    if [ $exit_code -eq 0 ]; then
        echo "✅ $test_name PASSED"
    else
        echo "❌ $test_name FAILED (exit code: $exit_code)"
    fi
    echo ""

    return $exit_code
}

# Test 1: Output Shape Validation (simplest test)
echo "Test 1/4: Output Shape Validation"
echo "This test verifies the encoder produces correct output dimensions"
echo ""
run_test "Shape Test" \
  "models/demos/llava/tests/test_vision_encoders.py::test_clip_encoder_output_shape" || {
    echo "⚠️  Shape test failed. This is critical - debug before continuing."
    exit 1
}

# Test 2: PCC Validation (main test)
echo "Test 2/4: PCC Validation (Primary Metric)"
echo "This test compares TTNN output vs PyTorch reference"
echo "Target: PCC >= 0.90"
echo ""
run_test "PCC Test" \
  "models/demos/llava/tests/test_vision_encoders.py::test_clip_encoder_pcc" || {
    echo "⚠️  PCC test failed. This may require iteration."
    echo "   - Check PCC value in output"
    echo "   - If PCC < 0.90, debug QuickGELU or attention"
    echo "   - Check /tmp/test_output.log for details"
}

# Test 3: Determinism Check
echo "Test 3/4: Determinism Validation"
echo "This test verifies reproducibility (same input → same output)"
echo ""
run_test "Determinism Test" \
  "models/demos/llava/tests/test_vision_encoders.py::test_clip_encoder_deterministic" || {
    echo "⚠️  Determinism test failed. Check for random operations."
}

# Test 4: Single Chip Test (n150)
echo "Test 4/4: Single Chip Validation (n150)"
echo "This test verifies encoder works on single-chip n150"
echo ""
export MESH_DEVICE=N150
run_test "Single Chip Test" \
  "models/demos/llava/tests/test_vision_encoders.py::test_clip_encoder_single_chip" || {
    echo "⚠️  Single chip test failed. Check tensor replication logic."
}

echo "============================================"
echo "Test Suite Complete"
echo "============================================"
echo ""
echo "Summary:"
echo "  - Shape Test: Check output above"
echo "  - PCC Test: Check output above (MOST IMPORTANT)"
echo "  - Determinism Test: Check output above"
echo "  - Single Chip Test: Check output above"
echo ""
echo "Next Steps:"
echo "  1. If all tests pass → Week 1 COMPLETE! 🎉"
echo "  2. If PCC < 0.90 → Debug and iterate"
echo "  3. Review logs in /tmp/ for details"
echo ""
