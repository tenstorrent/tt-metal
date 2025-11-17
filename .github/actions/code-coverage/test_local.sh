#!/bin/bash
#
# Local test script for code coverage action
# This script helps validate the coverage tool works correctly on your local VM
#

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COVERAGE_DIR="$REPO_ROOT/coverage"
HTML_OUTPUT_DIR="$COVERAGE_DIR/html"

echo "=========================================="
echo "Local Coverage Tool Test Script"
echo "=========================================="
echo "Repository root: $REPO_ROOT"
echo "Coverage directory: $COVERAGE_DIR"
echo ""

# Check if we're in the right directory
if [ ! -f "$REPO_ROOT/build_metal.sh" ]; then
    echo "ERROR: This script must be run from the tt-metal repository root"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Setup coverage environment (installs missing dependencies and sets env vars)
echo "Setting up coverage environment..."
if [ -f "$SCRIPT_DIR/setup_coverage_env.sh" ]; then
    # Source the setup script to get environment variables
    # Temporarily unset LD_PRELOAD to avoid instrumenting bash itself
    OLD_LD_PRELOAD="${LD_PRELOAD:-}"
    unset LD_PRELOAD
    # Source the setup script (this will install dependencies and set env vars)
    source "$SCRIPT_DIR/setup_coverage_env.sh" || {
        echo "⚠ Warning: setup_coverage_env.sh had errors, but continuing..."
    }
    # Don't restore LD_PRELOAD here - we'll set it only when running actual tests
    # The setup script sets it, but we'll manage it more carefully
else
    echo "⚠ Warning: setup_coverage_env.sh not found, skipping auto-setup"
fi

# Check prerequisites (after setup script may have installed them)
echo ""
echo "Checking prerequisites..."

MISSING_TOOLS=()

if ! command -v llvm-profdata &> /dev/null; then
    MISSING_TOOLS+=("llvm-profdata")
fi

if ! command -v llvm-cov &> /dev/null; then
    MISSING_TOOLS+=("llvm-cov")
fi

if ! python3 -c "import coverage" 2>/dev/null; then
    MISSING_TOOLS+=("python3-coverage (pip install coverage)")
fi

if ! command -v genhtml &> /dev/null; then
    MISSING_TOOLS+=("genhtml (install lcov package)")
fi

if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    echo "ERROR: Missing required tools:"
    for tool in "${MISSING_TOOLS[@]}"; do
        echo "  - $tool"
    done
    echo ""
    echo "Please run: source .github/actions/code-coverage/setup_coverage_env.sh"
    echo "Or install manually and try again."
    exit 1
fi

echo "✓ All prerequisites met"
echo ""

# Check if build exists
BUILD_DIR="$REPO_ROOT/build_ASanCoverage"
if [ ! -d "$BUILD_DIR" ]; then
    echo "WARNING: ASanCoverage build directory not found: $BUILD_DIR"
    echo "You may need to build first:"
    echo "  ./build_metal.sh --build-type ASanCoverage --build-tests"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create coverage directory
mkdir -p "$COVERAGE_DIR"

# Check for existing coverage data
HAS_PROFRAW=false
HAS_PYTHON_COV=false
HAS_KERNEL_NAMES=false

if ls "$COVERAGE_DIR"/*.profraw 1> /dev/null 2>&1; then
    HAS_PROFRAW=true
    echo "✓ Found .profraw files"
fi

if [ -f "$COVERAGE_DIR/.coverage" ] || [ -f "$REPO_ROOT/.coverage" ]; then
    HAS_PYTHON_COV=true
    echo "✓ Found Python coverage file"
fi

if [ -f "$REPO_ROOT/generated/watcher/kernel_names.txt" ]; then
    HAS_KERNEL_NAMES=true
    echo "✓ Found kernel_names.txt"
fi

if [ "$HAS_PROFRAW" = false ] && [ "$HAS_PYTHON_COV" = false ] && [ "$HAS_KERNEL_NAMES" = false ]; then
    echo ""
    echo "WARNING: No coverage data found!"
    echo ""
    echo "To generate coverage data, run:"
    echo ""
    echo "  # Build with coverage"
    echo "  ./build_metal.sh --build-type ASanCoverage --build-tests"
    echo ""
    echo "  # Run tests with coverage"
    echo "  export LLVM_PROFILE_FILE=\"coverage/%p.profraw\""
    echo "  export TT_METAL_WATCHER_APPEND=1"
    echo "  coverage run -m pytest tests/tt_eager/python_api_testing/unit_testing/ -xvvv"
    echo "  ./build_ASanCoverage/test/tt_metal/unit_tests --gtest_filter=\"*\""
    echo ""
    read -p "Continue anyway to test the tool? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# Find C++ objects
CPP_OBJECTS=()

# Look for common library files
for lib in "$BUILD_DIR/lib/libtt_metal.so" "$BUILD_DIR/lib/libtt_stl.so"; do
    if [ -f "$lib" ]; then
        CPP_OBJECTS+=("$lib")
    fi
done

# Look for test binaries
for test_bin in "$BUILD_DIR/test/tt_metal/unit_tests" "$BUILD_DIR/test/tt_metal"/*; do
    if [ -f "$test_bin" ] && [ -x "$test_bin" ]; then
        CPP_OBJECTS+=("$test_bin")
        break  # Just use first executable found
    fi
done

if [ ${#CPP_OBJECTS[@]} -eq 0 ]; then
    echo "WARNING: No C++ objects found for coverage"
    echo "You may need to specify them manually"
    CPP_OBJECTS_STR=""
else
    echo "Found C++ objects:"
    for obj in "${CPP_OBJECTS[@]}"; do
        echo "  - $obj"
    done
    CPP_OBJECTS_STR="${CPP_OBJECTS[*]}"
fi

echo ""

# Run the coverage tool
echo "Running coverage report generator..."
echo ""

cd "$REPO_ROOT"

# Save LD_PRELOAD if it was set, but unset it for running the script
# The entrypoint.sh doesn't need LD_PRELOAD - it's only needed when running actual tests
SAVED_LD_PRELOAD="${LD_PRELOAD:-}"
unset LD_PRELOAD

# Run entrypoint.sh (it will handle coverage collection)
"$SCRIPT_DIR/entrypoint.sh" \
    --coverage-dir "$COVERAGE_DIR" \
    --kernel-names-file "generated/watcher/kernel_names.txt" \
    --source-dir "$REPO_ROOT" \
    --cpp-objects "$CPP_OBJECTS_STR" \
    --html-output-dir "$HTML_OUTPUT_DIR" || {
    echo ""
    echo "⚠ Warning: Coverage report generation had errors"
    echo "Check the output above for details"
}

# Restore LD_PRELOAD if it was set (for future test runs)
if [ -n "$SAVED_LD_PRELOAD" ]; then
    export LD_PRELOAD="$SAVED_LD_PRELOAD"
fi

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Coverage report: $HTML_OUTPUT_DIR/index.html"
echo ""
echo "To view the report, open:"
echo "  file://$HTML_OUTPUT_DIR/index.html"
echo ""
echo "Or use:"
echo "  xdg-open $HTML_OUTPUT_DIR/index.html  # Linux"
echo "  open $HTML_OUTPUT_DIR/index.html      # macOS"
