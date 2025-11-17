#!/bin/bash
#
# Helper script to run C++ test binaries with coverage
# This handles the LD_PRELOAD issue for statically-linked ASan binaries
#
# Usage:
#   .github/actions/code-coverage/run_cpp_test.sh <path_to_binary> [args...]
#
# Example:
#   .github/actions/code-coverage/run_cpp_test.sh ./build_ASanCoverage/test/tt_metal/test_add_two_ints
#
# Note: This script will exit with the test's exit code, but the caller
# should continue to generate coverage reports even if the test fails.

# Don't use set -e here - we want to return the actual exit code

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"

# Setup coverage environment (but don't set LD_PRELOAD for C++ binaries)
OLD_LD_PRELOAD="${LD_PRELOAD:-}"
unset LD_PRELOAD

# Source setup script to get other environment variables
source "$SCRIPT_DIR/setup_coverage_env.sh" || {
    echo "Error: Failed to setup coverage environment"
    exit 1
}

# For C++ binaries with statically-linked ASan, we should NOT use LD_PRELOAD
# It causes "incompatible ASan runtimes" errors
unset LD_PRELOAD

# Set LD_LIBRARY_PATH to find the libraries
export LD_LIBRARY_PATH="$REPO_ROOT/build/lib:${LD_LIBRARY_PATH}"

# Run the test binary
echo "Running: $@"
echo "Environment:"
echo "  LLVM_PROFILE_FILE: ${LLVM_PROFILE_FILE:-not set}"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  LD_PRELOAD: ${LD_PRELOAD:-not set (correct for static ASan binaries)}"
echo ""

# Run the test - always return the exit code so caller can decide what to do
# Coverage data will be available regardless of exit code
"$@"
EXIT_CODE=$?
exit $EXIT_CODE
