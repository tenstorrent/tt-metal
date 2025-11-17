#!/bin/bash
#
# Simplified code coverage runner
# Automatically handles C++ binaries and Python tests, then generates HTML report
#
# Usage:
#   .github/actions/code-coverage/run_coverage.sh <test>
#
# Examples:
#   # C++ binary
#   .github/actions/code-coverage/run_coverage.sh ./build_ASanCoverage/test/tt_metal/test_add_two_ints
#
#   # Python pytest
#   .github/actions/code-coverage/run_coverage.sh tests/ttnn/unit_tests/operations/matmul/test_matmul.py
#
#   # Python pytest with arguments
#   .github/actions/code-coverage/run_coverage.sh "tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul"
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"

# Default values - keep coverage files in .github/coverage to avoid clutter
COVERAGE_DIR="${COVERAGE_DIR:-.github/coverage}"
HTML_OUTPUT_DIR="${COVERAGE_HTML_DIR:-.github/coverage/html}"

# Parse arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <test>"
    echo ""
    echo "Examples:"
    echo "  $0 ./build_ASanCoverage/test/tt_metal/test_add_two_ints"
    echo "  $0 tests/ttnn/unit_tests/operations/matmul/test_matmul.py"
    echo "  $0 \"tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul\""
    exit 1
fi

TEST="$1"
shift || true  # Get remaining arguments if any

echo "=========================================="
echo "Code Coverage Runner"
echo "=========================================="
echo "Test: $TEST"
echo "Coverage directory: $COVERAGE_DIR"
echo "HTML output: $HTML_OUTPUT_DIR"
echo ""

# Determine test type
IS_CPP_TEST=false
IS_PYTHON_TEST=false

if [ -f "$TEST" ] && [ -x "$TEST" ]; then
    # It's an executable file (C++ binary)
    IS_CPP_TEST=true
    echo "Detected: C++ binary"
elif [[ "$TEST" == *.py ]] || [[ "$TEST" == *test*.py ]] || [[ "$TEST" == *pytest* ]] || command -v pytest &> /dev/null && python3 -m pytest --collect-only "$TEST" &>/dev/null; then
    # It's a Python test file
    IS_PYTHON_TEST=true
    echo "Detected: Python test"
else
    echo "Error: Could not determine test type for: $TEST"
    echo "  - For C++ binaries, provide full path to executable"
    echo "  - For Python tests, provide path to .py file or pytest path"
    exit 1
fi

# Clean up old coverage files for a fresh start
echo ""
echo "Cleaning up old coverage files..."
if [ -d "$COVERAGE_DIR" ]; then
    # Remove old profraw files (C++ coverage)
    find "$COVERAGE_DIR" -name "*.profraw" -type f -delete 2>/dev/null || true
    # Remove old profdata files (merged C++ coverage)
    find "$COVERAGE_DIR" -name "*.profdata" -type f -delete 2>/dev/null || true
    # Remove old .coverage files (Python coverage)
    find "$COVERAGE_DIR" -name ".coverage" -type f -delete 2>/dev/null || true
    find "$REPO_ROOT" -maxdepth 1 -name ".coverage" -type f -delete 2>/dev/null || true
    # Remove old coverage info files
    find "$COVERAGE_DIR" -name "*.info" -type f -delete 2>/dev/null || true
    echo "  ✓ Cleaned old coverage data files"
else
    echo "  ℹ No existing coverage directory to clean"
fi

# Clean up old HTML output
if [ -d "$HTML_OUTPUT_DIR" ]; then
    rm -rf "$HTML_OUTPUT_DIR" 2>/dev/null || true
    echo "  ✓ Cleaned old HTML report"
fi

# Create coverage directory (fresh) - ensure it exists before tests run
mkdir -p "$COVERAGE_DIR"

# Setup coverage environment
echo ""
echo "Setting up coverage environment..."
OLD_LD_PRELOAD="${LD_PRELOAD:-}"
unset LD_PRELOAD  # Temporarily unset to avoid instrumenting bash
source "$SCRIPT_DIR/setup_coverage_env.sh" || {
    echo "Error: Failed to setup coverage environment"
    exit 1
}

# Restore LD_PRELOAD if it was set (for Python tests)
if [ -n "$OLD_LD_PRELOAD" ]; then
    export LD_PRELOAD="$OLD_LD_PRELOAD"
fi

echo ""
echo "Running test with coverage instrumentation..."
echo ""

# Run the test with coverage
# Always continue to generate coverage report, even if test fails
if [ "$IS_CPP_TEST" = true ]; then
    # C++ binary - use run_cpp_test.sh helper
    echo "Running C++ binary: $TEST"
    # Disable exit on error - we always want to generate coverage report
    set +e
    "$SCRIPT_DIR/run_cpp_test.sh" "$TEST" "$@"
    TEST_EXIT_CODE=$?
    set -e
    if [ $TEST_EXIT_CODE -ne 0 ]; then
        echo "⚠ Note: Test exited with code $TEST_EXIT_CODE (test may have failed or LeakSanitizer warnings)"
        echo "  Continuing with coverage collection anyway - this is useful for debugging..."
    fi

    # Find the binary and related libraries for coverage export
    TEST_BINARY="$TEST"
    if [[ "$TEST_BINARY" != /* ]]; then
        TEST_BINARY="$REPO_ROOT/$TEST_BINARY"
    fi

    # Find build directory
    BUILD_DIR=""
    if [ -L "$REPO_ROOT/build" ]; then
        BUILD_DIR="$(readlink -f "$REPO_ROOT/build")"
    elif [ -d "$REPO_ROOT/build_ASanCoverage" ]; then
        BUILD_DIR="$REPO_ROOT/build_ASanCoverage"
    else
        # Try to find build directory from binary path
        if [[ "$TEST_BINARY" == *"/build_"* ]]; then
            BUILD_DIR="$(echo "$TEST_BINARY" | sed 's|/test/.*||')"
        fi
    fi

    # Find libraries in build directory
    CPP_OBJECTS="$TEST_BINARY"
    if [ -n "$BUILD_DIR" ]; then
        if [ -f "$BUILD_DIR/lib/libtt_metal.so" ]; then
            CPP_OBJECTS="$CPP_OBJECTS $BUILD_DIR/lib/libtt_metal.so"
        fi
        if [ -f "$BUILD_DIR/lib/libtt_stl.so" ]; then
            CPP_OBJECTS="$CPP_OBJECTS $BUILD_DIR/lib/libtt_stl.so"
        fi
    fi

    ENABLE_PYTHON_COVERAGE="false"
    ENABLE_KERNEL_COVERAGE="true"

elif [ "$IS_PYTHON_TEST" = true ]; then
    # Python test - use coverage run
    echo "Running Python test: $TEST"

    # Ensure coverage is installed
    if ! python3 -c "import coverage" 2>/dev/null; then
        echo "Installing coverage module..."
        pip install coverage || {
            echo "Error: Failed to install coverage module"
            exit 1
        }
    fi

    # Run pytest with coverage
    # Always continue to generate coverage report, even if test fails
    set +e
    coverage run --source="$REPO_ROOT" -m pytest "$TEST" "$@"
    TEST_EXIT_CODE=$?
    set -e
    if [ $TEST_EXIT_CODE -ne 0 ]; then
        echo "⚠ Note: Test exited with code $TEST_EXIT_CODE (test may have failed)"
        echo "  Continuing with coverage collection anyway - this is useful for debugging..."
    fi

    # Move .coverage file to coverage directory if it exists
    if [ -f ".coverage" ]; then
        mv .coverage "$COVERAGE_DIR/.coverage" || true
    fi

    CPP_OBJECTS=""
    ENABLE_PYTHON_COVERAGE="true"
    ENABLE_KERNEL_COVERAGE="true"
fi

echo ""
echo "=========================================="
echo "Generating coverage report..."
echo "=========================================="
echo ""

# Check if any coverage data was generated
HAS_COVERAGE_DATA=false
if [ "$IS_CPP_TEST" = true ]; then
    if find "$COVERAGE_DIR" -name "*.profraw" -type f 2>/dev/null | grep -q .; then
        HAS_COVERAGE_DATA=true
    fi
elif [ "$IS_PYTHON_TEST" = true ]; then
    if [ -f "$COVERAGE_DIR/.coverage" ] || [ -f ".coverage" ]; then
        HAS_COVERAGE_DATA=true
    fi
fi

if [ "$HAS_COVERAGE_DATA" = false ] && [ "$ENABLE_KERNEL_COVERAGE" != "true" ]; then
    echo "⚠ Warning: No coverage data found after test run."
    echo "  This might mean:"
    echo "    - The test didn't execute successfully"
    echo "    - Coverage instrumentation isn't working"
    echo "    - Files are being written to a different location"
    echo ""
    echo "  Checking coverage directory contents..."
    ls -la "$COVERAGE_DIR" 2>/dev/null || echo "  Coverage directory doesn't exist or is empty"
    echo ""
fi

# Generate the coverage report
"$SCRIPT_DIR/entrypoint.sh" \
    --coverage-dir "$COVERAGE_DIR" \
    --source-dir "$REPO_ROOT" \
    --cpp-objects "$CPP_OBJECTS" \
    --enable-cpp-coverage "$([ -n "$CPP_OBJECTS" ] && echo "true" || echo "false")" \
    --enable-python-coverage "$ENABLE_PYTHON_COVERAGE" \
    --enable-kernel-coverage "$ENABLE_KERNEL_COVERAGE" \
    --html-output-dir "$HTML_OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Coverage report complete!"
echo "=========================================="
echo ""
echo "HTML Report: $HTML_OUTPUT_DIR/index.html"
echo ""
echo "To view the report:"
echo "  1. Copy it out of the container: docker cp <container>:$HTML_OUTPUT_DIR ./coverage_report"
echo "  2. Or open it directly if you have access to the container filesystem"
echo ""
