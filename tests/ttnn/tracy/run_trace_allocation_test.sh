#!/bin/bash
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Run trace allocation warning tests with proper environment setup
# Usage:
#   ./run_trace_allocation_test.sh                    # Run all tests
#   ./run_trace_allocation_test.sh <test_name>        # Run specific test
#
# Available tests:
#   test_allocation_during_active_trace_triggers_warning
#   test_allocation_during_trace_execution_2cq
#   test_stress_allocation_during_trace
#   test_verify_environment_info

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Set up environment
export PYTHONPATH="$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$TT_METAL_HOME/build/lib:$TT_METAL_HOME:$PYTHONPATH"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:$LD_LIBRARY_PATH"

# Activate venv if available
if [ -f /opt/venv/bin/activate ]; then
    source /opt/venv/bin/activate
elif [ -f "$TT_METAL_HOME/venv/bin/activate" ]; then
    source "$TT_METAL_HOME/venv/bin/activate"
fi

TEST_FILE="$SCRIPT_DIR/test_trace_allocation_warning.py"

if [ -n "$1" ]; then
    # Run specific test
    pytest "$TEST_FILE::$1" -v -s
else
    # Run all tests
    pytest "$TEST_FILE" -v -s
fi
