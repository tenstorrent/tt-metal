#!/bin/bash
#
# Helper script to run tests with coverage collection
# This sets up the environment and runs tests, ensuring LD_PRELOAD is only
# applied to the actual test executables, not bash itself
#
# Usage:
#   .github/actions/code-coverage/run_tests_with_coverage.sh <test_command>
#
# Example:
#   .github/actions/code-coverage/run_tests_with_coverage.sh "coverage run -m pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py"
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Setup coverage environment (but don't set LD_PRELOAD yet)
cd "$REPO_ROOT"
OLD_LD_PRELOAD="${LD_PRELOAD:-}"
unset LD_PRELOAD

# Source setup script to get other environment variables
source "$SCRIPT_DIR/setup_coverage_env.sh" || {
    echo "Error: Failed to setup coverage environment"
    exit 1
}

# Now set LD_PRELOAD for running tests (but not for bash itself)
# We'll use env to run the command with LD_PRELOAD set
if [ -n "$LD_PRELOAD" ]; then
    echo "Running tests with ASan runtime preloaded..."
    # Use env to set LD_PRELOAD only for the test command
    env LD_PRELOAD="$LD_PRELOAD" "$@"
else
    echo "Running tests (no ASan runtime to preload)..."
    "$@"
fi
