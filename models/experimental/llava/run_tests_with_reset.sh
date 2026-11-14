#!/bin/bash
# LLaVA Test Runner with Automatic Device Reset and Cache Clearing
#
# Usage:
#   ./run_tests_with_reset.sh                    # Run all vision encoder tests
#   ./run_tests_with_reset.sh test_clip_pcc      # Run specific test
#   ./run_tests_with_reset.sh --help             # Show help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
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

# Parse arguments
TEST_PATTERN="${1:-}"
if [ "$TEST_PATTERN" = "--help" ] || [ "$TEST_PATTERN" = "-h" ]; then
    echo "LLaVA Test Runner with Device Reset"
    echo ""
    echo "Usage:"
    echo "  $0                    # Run all vision encoder tests"
    echo "  $0 test_clip_pcc      # Run specific test matching pattern"
    echo "  $0 --help             # Show this help"
    echo ""
    echo "This script:"
    echo "  1. Resets n300 device (tt-smi -r)"
    echo "  2. Clears kernel cache (~/.cache/tt-metal-cache/)"
    echo "  3. Sets environment variables"
    echo "  4. Runs tests via run_safe_pytest.sh"
    exit 0
fi

echo "============================================"
echo "LLaVA Test Runner (with device reset)"
echo "============================================"
echo ""

# Step 1: Reset device
echo "Step 1/4: Resetting n300 device..."
if tt-smi -r --no_reinit; then
    echo "✅ Device reset complete"
else
    echo "⚠️  Device reset command failed, continuing anyway..."
fi
sleep 2
echo ""

# Step 2: Clear cache for fresh JIT compilation
echo "Step 2/4: Clearing kernel cache..."
if [ -d ~/.cache/tt-metal-cache/ ]; then
    rm -rf ~/.cache/tt-metal-cache/
    echo "✅ Cache cleared"
else
    echo "ℹ️  Cache directory doesn't exist (fresh state)"
fi
echo ""

# Step 3: Set environment
echo "Step 3/4: Setting environment variables..."
# Auto-detect ttnn installation: source tree build vs installed wheel
# The source tree's ttnn/ttnn/__init__.py will shadow the installed package
# if PYTHONPATH includes $TT_METAL_ROOT/ttnn, but it needs _ttnn.so to work.
if [ -f "$TT_METAL_ROOT/ttnn/ttnn/_ttnn.cpython-310-x86_64-linux-gnu.so" ]; then
    # Source tree is built — use it directly
    export TT_METAL_HOME="$TT_METAL_ROOT"
    export PYTHONPATH="$TT_METAL_ROOT/ttnn:$TT_METAL_ROOT:$PYTHONPATH"
    echo "  ttnn source: Built from source tree"
else
    # Source tree not built — use pre-installed wheel from /opt/venv
    # CRITICAL: Do NOT add $TT_METAL_ROOT/ttnn to PYTHONPATH, as the source
    # tree's ttnn/ttnn/__init__.py would shadow the installed package and fail
    # with "ModuleNotFoundError: No module named 'ttnn._ttnn'" (no compiled .so).
    export TT_METAL_HOME="/opt/venv/lib/python3.10/site-packages/ttnn"
    export PYTHONPATH="$TT_METAL_ROOT:$PYTHONPATH"
    echo "  ttnn source: Pre-installed wheel (/opt/venv)"

    # Hide source tree kernels to prevent JIT from finding incompatible versions
    if [ -d "$TT_METAL_ROOT/ttnn/cpp" ]; then
        mv "$TT_METAL_ROOT/ttnn/cpp" "$TT_METAL_ROOT/ttnn/cpp_backup"
        RESTORE_CPP=true
        echo "  ⚠️  Temporarily hid ttnn/cpp (incompatible with installed wheel)"
    fi
    if [ -d "$TT_METAL_ROOT/tt_metal" ]; then
        mv "$TT_METAL_ROOT/tt_metal" "$TT_METAL_ROOT/tt_metal_backup"
        RESTORE_TT_METAL=true
        echo "  ⚠️  Temporarily hid tt_metal (incompatible with installed wheel)"
    fi
fi
export MESH_DEVICE=N300

echo "  PYTHONPATH: $PYTHONPATH"
echo "  TT_METAL_HOME: $TT_METAL_HOME"
echo "  MESH_DEVICE: $MESH_DEVICE"
echo "✅ Environment ready"
echo ""

# Step 4: Run tests
echo "Step 4/4: Running tests..."
cd "$TT_METAL_ROOT"

if [ -z "$TEST_PATTERN" ]; then
    # Run all vision encoder tests
    scripts/run_safe_pytest.sh models/demos/llava/tests/test_vision_encoders.py -v
else
    # Run specific test pattern
    scripts/run_safe_pytest.sh models/demos/llava/tests/test_vision_encoders.py -k "$TEST_PATTERN" -v
fi

EXIT_CODE=$?

echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test Run Complete - All tests passed!"
else
    echo "❌ Test Run Complete - Some tests failed (exit code: $EXIT_CODE)"
fi
echo "============================================"

exit $EXIT_CODE
