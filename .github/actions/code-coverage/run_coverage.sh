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
#   .github/actions/code-coverage/run_coverage.sh ./build_LLVMCoverage/test/tt_metal/test_add_two_ints
#
#   # Python pytest
#   .github/actions/code-coverage/run_coverage.sh tests/ttnn/unit_tests/operations/matmul/test_matmul.py
#
#   # Python pytest with arguments
#   .github/actions/code-coverage/run_coverage.sh "tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul"
#

set -e

START_EPOCH_SECONDS="$(date +%s)"
INVOCATION_CWD="$(pwd)"
COVERAGE_DURATION_FILE="${INVOCATION_CWD}/coverage_duration.txt"

write_duration_report() {
    local exit_code="$1"
    local end_epoch elapsed hours minutes

    end_epoch="$(date +%s)"
    elapsed=$((end_epoch - START_EPOCH_SECONDS))
    if [ "$elapsed" -lt 0 ]; then
        elapsed=0
    fi
    hours=$((elapsed / 3600))
    minutes=$(((elapsed % 3600) / 60))

    if {
        printf "run_coverage.sh duration: %02dh %02dm\n" "$hours" "$minutes"
        printf "Exit code: %s\n" "$exit_code"
    } > "$COVERAGE_DURATION_FILE" 2>/dev/null; then
        echo "✓ Duration written to $COVERAGE_DURATION_FILE"
    else
        echo "⚠ Warning: Unable to write duration file at $COVERAGE_DURATION_FILE"
    fi
}

trap 'write_duration_report $?' EXIT

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
    echo "  $0 ./build_LLVMCoverage/test/tt_metal/test_add_two_ints"
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

# Check for Python test first (before checking if file is executable, since .py files can be executable)
if [[ "$TEST" == *.py ]] || [[ "$TEST" == *test*.py ]] || [[ "$TEST" == *pytest* ]]; then
    # It's a Python test file
    IS_PYTHON_TEST=true
    echo "Detected: Python test"
elif [ -f "$TEST" ] && [ -x "$TEST" ]; then
    # It's an executable file (C++ binary)
    IS_CPP_TEST=true
    echo "Detected: C++ binary"
elif command -v pytest &> /dev/null && python3 -m pytest --collect-only "$TEST" &>/dev/null 2>&1; then
    # Try pytest collection as fallback (might be a pytest path like "tests/path::test_function")
    IS_PYTHON_TEST=true
    echo "Detected: Python test (via pytest)"
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

# Clean up generated folder (contains kernel_names.txt which is regenerated during test execution)
if [ -d "$REPO_ROOT/generated" ]; then
    echo "Cleaning up generated folder..."
    rm -rf "$REPO_ROOT/generated" 2>/dev/null || true
    echo "  ✓ Cleaned generated folder (kernel_names.txt will be regenerated)"
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

# For Python tests with ASanCoverage builds, we MUST have LD_PRELOAD set (Python loads shared libraries dynamically)
# But for Coverage builds (no ASan), we don't need LD_PRELOAD
# Detect build type to determine if ASan is needed
    BUILD_TYPE=""
    if [ -L "$REPO_ROOT/build" ]; then
        BUILD_LINK="$(readlink -f "$REPO_ROOT/build")"
        if [[ "$BUILD_LINK" == *"build_LLVMCoverage"* ]]; then
            BUILD_TYPE="LLVMCoverage"
        elif [[ "$BUILD_LINK" == *"build_ASanCoverage"* ]]; then
            BUILD_TYPE="ASanCoverage"
        fi
    elif [ -d "$REPO_ROOT/build_LLVMCoverage" ]; then
        BUILD_TYPE="LLVMCoverage"
    elif [ -d "$REPO_ROOT/build_ASanCoverage" ]; then
        BUILD_TYPE="ASanCoverage"
    fi

if [ "$IS_PYTHON_TEST" = true ] && [ "$BUILD_TYPE" = "ASanCoverage" ]; then
    # Find Clang ASan runtime if not already set
    CLANG_ASAN_RUNTIME=""
    if [ -z "$LD_PRELOAD" ]; then
        if [ -f "/usr/lib/llvm-17/lib/clang/17/lib/linux/libclang_rt.asan-x86_64.so" ]; then
            CLANG_ASAN_RUNTIME="/usr/lib/llvm-17/lib/clang/17/lib/linux/libclang_rt.asan-x86_64.so"
        elif [ -f "/usr/lib/llvm-18/lib/clang/18/lib/linux/libclang_rt.asan-x86_64.so" ]; then
            CLANG_ASAN_RUNTIME="/usr/lib/llvm-18/lib/clang/18/lib/linux/libclang_rt.asan-x86_64.so"
        else
            CLANG_ASAN_RUNTIME=$(find /usr/lib/llvm-* -name "libclang_rt.asan-x86_64.so" 2>/dev/null | head -1)
        fi

        if [ -n "$CLANG_ASAN_RUNTIME" ] && [ -f "$CLANG_ASAN_RUNTIME" ]; then
            export LD_PRELOAD="$CLANG_ASAN_RUNTIME"
            echo "✓ Set LD_PRELOAD for Python test (ASanCoverage build): $CLANG_ASAN_RUNTIME"
        else
            echo "⚠ Warning: Could not find Clang ASan runtime for LD_PRELOAD"
            echo "  Python tests may fail with ASan symbol errors"
        fi
    else
        # LD_PRELOAD is already set, extract the runtime directory from it
        CLANG_ASAN_RUNTIME="$LD_PRELOAD"
    fi

    # Also ensure LD_LIBRARY_PATH includes the Clang runtime directory
    if [ -n "$CLANG_ASAN_RUNTIME" ] && [ -f "$CLANG_ASAN_RUNTIME" ]; then
        CLANG_RUNTIME_DIR=$(dirname "$CLANG_ASAN_RUNTIME")
        if [[ ":$LD_LIBRARY_PATH:" != *":$CLANG_RUNTIME_DIR:"* ]]; then
            export LD_LIBRARY_PATH="$CLANG_RUNTIME_DIR:${LD_LIBRARY_PATH}"
            echo "✓ Added Clang runtime to LD_LIBRARY_PATH: $CLANG_RUNTIME_DIR"
        fi
    fi
elif [ "$IS_PYTHON_TEST" = true ] && [ "$BUILD_TYPE" = "LLVMCoverage" ]; then
    echo "✓ LLVMCoverage build detected - no ASan runtime needed for Python tests"
fi

# Restore LD_PRELOAD if it was set before (but Python tests take precedence)
if [ "$IS_PYTHON_TEST" != true ] && [ -n "$OLD_LD_PRELOAD" ]; then
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
    elif [ -d "$REPO_ROOT/build_LLVMCoverage" ]; then
        BUILD_DIR="$REPO_ROOT/build_LLVMCoverage"
    elif [ -d "$REPO_ROOT/build_ASanCoverage" ]; then
        # Fallback to ASanCoverage for backwards compatibility
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
    # Check without LD_PRELOAD to avoid ASan interfering with the check
    OLD_LD_PRELOAD_CHECK="${LD_PRELOAD:-}"
    unset LD_PRELOAD
    COVERAGE_AVAILABLE=false
    if python3 -c "import coverage" 2>/dev/null; then
        COVERAGE_AVAILABLE=true
    fi
    if [ -n "$OLD_LD_PRELOAD_CHECK" ]; then
        export LD_PRELOAD="$OLD_LD_PRELOAD_CHECK"
    fi

    if [ "$COVERAGE_AVAILABLE" = false ]; then
        echo "Installing coverage module..."
        # Temporarily unset LD_PRELOAD for pip install to avoid ASan errors
        TEMP_LD_PRELOAD="${LD_PRELOAD:-}"
        unset LD_PRELOAD
        # Don't fail on LeakSanitizer warnings - they're from Python itself, not the installation
        set +e
        pip install coverage 2>&1 | grep -v "LeakSanitizer\|SUMMARY: AddressSanitizer" || true
        PIP_EXIT_CODE=$?
        set -e
        # Restore LD_PRELOAD
        if [ -n "$TEMP_LD_PRELOAD" ]; then
            export LD_PRELOAD="$TEMP_LD_PRELOAD"
        fi
        # Check if coverage is actually installed (without LD_PRELOAD to avoid ASan)
        unset LD_PRELOAD
        if ! python3 -c "import coverage" 2>/dev/null; then
            echo "Error: Failed to install coverage module"
            exit 1
        fi
        if [ -n "$TEMP_LD_PRELOAD" ]; then
            export LD_PRELOAD="$TEMP_LD_PRELOAD"
        fi
        if [ $PIP_EXIT_CODE -ne 0 ]; then
            echo "⚠ Note: pip exited with code $PIP_EXIT_CODE (likely LeakSanitizer warnings), but coverage is installed"
        fi
    else
        echo "✓ Coverage module already installed"
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
# Temporarily unset LD_PRELOAD for entrypoint.sh to avoid ASan interfering with Python imports
# entrypoint.sh will handle its own environment setup
OLD_LD_PRELOAD_FOR_REPORT="${LD_PRELOAD:-}"
unset LD_PRELOAD
"$SCRIPT_DIR/entrypoint.sh" \
    --coverage-dir "$COVERAGE_DIR" \
    --source-dir "$REPO_ROOT" \
    --cpp-objects "$CPP_OBJECTS" \
    --enable-cpp-coverage "$([ -n "$CPP_OBJECTS" ] && echo "true" || echo "false")" \
    --enable-python-coverage "$ENABLE_PYTHON_COVERAGE" \
    --enable-kernel-coverage "$ENABLE_KERNEL_COVERAGE" \
    --html-output-dir "$HTML_OUTPUT_DIR"
ENTRYPOINT_EXIT_CODE=$?
# Restore LD_PRELOAD if it was set
if [ -n "$OLD_LD_PRELOAD_FOR_REPORT" ]; then
    export LD_PRELOAD="$OLD_LD_PRELOAD_FOR_REPORT"
fi

if [ $ENTRYPOINT_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Failed to generate coverage report (exit code: $ENTRYPOINT_EXIT_CODE)"
    exit $ENTRYPOINT_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Coverage report complete!"
echo "=========================================="
echo ""
echo "HTML Report: $HTML_OUTPUT_DIR/index.html"
echo ""
echo "To view the report:"
echo "  0. turn the report into a zip file: .github/actions/code-coverage/zip_coverage.sh"
echo "  1. Copy it out of the container: docker cp <container>:.github/coverage/coverage_report.tar.gz ./coverage_report.tar.gz"
echo "  2. unzip the file on your local machine: tar -xzf coverage_report.tar.gz"
echo "  3. open the index.html file: open coverage_report/html/index.html"
echo ""
