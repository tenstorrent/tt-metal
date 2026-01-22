#!/bin/bash
# Run script for Fused CCL Tests on Galaxy and T3K
#
# Usage:
#   ./run_fused_ccl_tests.sh [galaxy|t3k] [check|perf|all] [test_filter]
#
# Examples:
#   ./run_fused_ccl_tests.sh galaxy check                    # Run functional tests on Galaxy
#   ./run_fused_ccl_tests.sh galaxy perf                     # Run performance tests on Galaxy
#   ./run_fused_ccl_tests.sh galaxy all                      # Run all tests on Galaxy
#   ./run_fused_ccl_tests.sh galaxy perf llama70b_decode     # Run specific test
#   ./run_fused_ccl_tests.sh t3k check                       # Run functional tests on T3K
#   ./run_fused_ccl_tests.sh t3k perf                        # Run performance tests on T3K

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"

# Parse arguments
DEVICE_TYPE="${1:-galaxy}"
TEST_TYPE="${2:-check}"
TEST_FILTER="${3:-}"

# Validate device type
if [[ "$DEVICE_TYPE" != "galaxy" && "$DEVICE_TYPE" != "t3k" ]]; then
    echo "Error: Invalid device type '$DEVICE_TYPE'. Use 'galaxy' or 't3k'."
    exit 1
fi

# Set up environment
echo "========================================"
echo "Fused CCL Tests Runner"
echo "========================================"
echo "Device: $DEVICE_TYPE"
echo "Test type: $TEST_TYPE"
echo "TT_METAL_HOME: $TT_METAL_HOME"
echo "========================================"

cd "$TT_METAL_HOME"
export TT_METAL_HOME
export PYTHONPATH="$TT_METAL_HOME"

# Activate virtual environment if it exists
if [[ -f "python_env/bin/activate" ]]; then
    source python_env/bin/activate
    echo "Activated python_env"
fi

# Build pytest command based on device type
if [[ "$DEVICE_TYPE" == "galaxy" ]]; then
    TEST_FILE="tests/ttnn/unit_tests/operations/ccl/test_fused_ccl_galaxy.py"
    TEST_FUNC="test_matmul_reduce_scatter_galaxy"
else
    TEST_FILE="tests/ttnn/unit_tests/operations/ccl/test_fused_ccl_t3k.py"
    TEST_FUNC="test_matmul_reduce_scatter_t3k"
fi

# Build filter string
FILTER=""
case "$TEST_TYPE" in
    "check")
        FILTER="check"
        ;;
    "perf")
        FILTER="perf"
        ;;
    "all")
        FILTER=""
        ;;
    *)
        echo "Error: Invalid test type '$TEST_TYPE'. Use 'check', 'perf', or 'all'."
        exit 1
        ;;
esac

# Add custom test filter if provided
if [[ -n "$TEST_FILTER" ]]; then
    if [[ -n "$FILTER" ]]; then
        FILTER="$FILTER and $TEST_FILTER"
    else
        FILTER="$TEST_FILTER"
    fi
fi

# Build pytest command
PYTEST_CMD="pytest ${TEST_FILE}::${TEST_FUNC} -v --tb=short"
if [[ -n "$FILTER" ]]; then
    PYTEST_CMD="$PYTEST_CMD -k \"$FILTER\""
fi

echo ""
echo "Running: $PYTEST_CMD"
echo "========================================"
echo ""

# Run tests
eval $PYTEST_CMD

echo ""
echo "========================================"
echo "Tests completed!"
echo "========================================"
