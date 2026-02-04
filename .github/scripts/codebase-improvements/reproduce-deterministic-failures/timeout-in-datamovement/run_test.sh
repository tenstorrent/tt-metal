#!/bin/bash

# Runner script for timeout-in-datamovement reproduction test
# This script sets up the required environment and runs the test

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Gather Timeout Reproduction Test"
echo "=========================================="
echo ""

# Required environment variables
echo "Setting up environment..."

# CRITICAL: Device operation timeout (must be 5 seconds to reproduce)
export TT_METAL_OPERATION_TIMEOUT_SECONDS=5

# Architecture
export ARCH_NAME=${ARCH_NAME:-wormhole_b0}

# Metal paths
export TT_METAL_HOME=${TT_METAL_HOME:-/tt-metal}
export PYTHONPATH=${PYTHONPATH:-/tt-metal}

# Logging
export LOGURU_LEVEL=${LOGURU_LEVEL:-INFO}

echo "  TT_METAL_OPERATION_TIMEOUT_SECONDS=$TT_METAL_OPERATION_TIMEOUT_SECONDS"
echo "  ARCH_NAME=$ARCH_NAME"
echo "  TT_METAL_HOME=$TT_METAL_HOME"
echo "  PYTHONPATH=$PYTHONPATH"
echo ""

# Activate virtual environment if it exists
if [ -f "/opt/venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source /opt/venv/bin/activate
elif [ -f "$TT_METAL_HOME/python_env/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$TT_METAL_HOME/python_env/bin/activate"
else
    echo "Warning: No virtual environment found"
fi

echo ""
echo "Running test..."
echo "=========================================="
echo ""

# Change to test directory
cd "$SCRIPT_DIR/tests"

# Run the test with pytest
# Use -x to stop on first failure
# Use -v for verbose output
# Use --timeout=120 to prevent hanging
pytest test_gather_timeout_stress.py -x -v --timeout=120 "$@"

TEST_RESULT=$?

echo ""
echo "=========================================="
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ Test PASSED"
else
    echo "❌ Test FAILED (exit code: $TEST_RESULT)"
fi
echo "=========================================="

exit $TEST_RESULT
