#!/bin/bash
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Run nd-tests with proper environment setup
# Usage:
#   ./nd-tests/run_tests.sh                    # Run all tests
#   ./nd-tests/run_tests.sh <test_name>        # Run specific test
#
# Available tests:
#   test_allocation_during_active_trace_triggers_warning
#   test_allocation_during_trace_execution_2cq
#   test_stress_allocation_during_trace
#   test_verify_environment_info

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set up environment
USER_SITE_PACKAGES="$HOME/.local/lib/python3.10/site-packages"
export PYTHONPATH="$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$TT_METAL_HOME/build/lib:$TT_METAL_HOME:$USER_SITE_PACKAGES:$PYTHONPATH"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:$LD_LIBRARY_PATH"

# Activate venv if available
if [ -f /opt/venv/bin/activate ]; then
    source /opt/venv/bin/activate
elif [ -f "$TT_METAL_HOME/venv/bin/activate" ]; then
    source "$TT_METAL_HOME/venv/bin/activate"
fi

TEST_FILE="$SCRIPT_DIR/test_trace_allocation_warning.py"

# Run from nd-tests directory to use our local conftest.py
# Use --ignore to skip root conftest dependencies
cd "$SCRIPT_DIR"

if [ -n "$1" ]; then
    # Run specific test
    pytest "test_trace_allocation_warning.py::$1" -v -s -p no:cacheprovider --confcutdir="$SCRIPT_DIR"
else
    # Run all tests
    pytest "test_trace_allocation_warning.py" -v -s -p no:cacheprovider --confcutdir="$SCRIPT_DIR"
fi
