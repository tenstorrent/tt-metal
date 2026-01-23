#!/bin/bash
# Run script for Fused CCL Tests on Galaxy and T3K
#
# Usage:
#   ./run_fused_ccl_tests.sh [galaxy|t3k] [check|perf|all] [test_filter]
#
# Examples:
#   ./run_fused_ccl_tests.sh galaxy check                    # Run functional tests on Galaxy
#   ./run_fused_ccl_tests.sh galaxy perf                     # Run performance comparison tests on Galaxy
#   ./run_fused_ccl_tests.sh galaxy all                      # Run all tests on Galaxy
#   ./run_fused_ccl_tests.sh galaxy perf llama70b_decode     # Run specific perf test
#   ./run_fused_ccl_tests.sh t3k check                       # Run functional tests on T3K
#   ./run_fused_ccl_tests.sh t3k perf                        # Run performance comparison tests on T3K
#
# Performance tests run both fused and non-fused operations and print a comparison report.

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

# Build pytest command based on device type and test type
if [[ "$DEVICE_TYPE" == "galaxy" ]]; then
    TEST_FILE="tests/ttnn/unit_tests/operations/ccl/test_fused_ccl_galaxy.py"
    case "$TEST_TYPE" in
        "check")
            TEST_FUNC="test_all_gather_matmul_galaxy_check"
            ;;
        "perf")
            TEST_FUNC="test_all_gather_matmul_galaxy_perf_comparison"
            ;;
        "all")
            TEST_FUNC=""  # Run all tests in file
            ;;
        *)
            echo "Error: Invalid test type '$TEST_TYPE'. Use 'check', 'perf', or 'all'."
            exit 1
            ;;
    esac
else
    TEST_FILE="tests/ttnn/unit_tests/operations/ccl/test_fused_ccl_t3k.py"
    case "$TEST_TYPE" in
        "check")
            TEST_FUNC="test_matmul_reduce_scatter_t3k_check"
            ;;
        "perf")
            TEST_FUNC="test_matmul_reduce_scatter_t3k_perf_comparison"
            ;;
        "all")
            TEST_FUNC=""  # Run all tests in file
            ;;
        *)
            echo "Error: Invalid test type '$TEST_TYPE'. Use 'check', 'perf', or 'all'."
            exit 1
            ;;
    esac
fi

# Build pytest command
if [[ -n "$TEST_FUNC" ]]; then
    PYTEST_CMD="pytest ${TEST_FILE}::${TEST_FUNC} -v --tb=short"
else
    PYTEST_CMD="pytest ${TEST_FILE} -v --tb=short"
fi

# Add custom test filter if provided
if [[ -n "$TEST_FILTER" ]]; then
    PYTEST_CMD="$PYTEST_CMD -k \"$TEST_FILTER\""
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
