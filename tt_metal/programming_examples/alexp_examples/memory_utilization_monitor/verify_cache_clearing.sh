#!/bin/bash
# Test if disable_and_clear_program_cache() reports deallocations

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║    Testing Program Cache Clearing Deallocation Tracking       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if server is running
if pgrep -x allocation_server_poc > /dev/null; then
    echo "✓ Allocation server is running"
else
    echo "✗ Allocation server is NOT running"
    echo ""
    echo "Starting allocation server..."
    ./allocation_server_poc > /tmp/alloc_server_cache_test.log 2>&1 &
    SERVER_PID=$!
    sleep 2

    if ! pgrep -x allocation_server_poc > /dev/null; then
        echo "Failed to start server!"
        exit 1
    fi
    echo "✓ Server started (PID: $SERVER_PID)"
    KILL_SERVER=true
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "INSTRUCTIONS:"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "1. Open a SECOND terminal and run the monitor:"
echo "   cd $SCRIPT_DIR"
echo "   ./allocation_monitor_client -a -r 500"
echo ""
echo "2. Watch the monitor during the test for these events:"
echo "   • Tensor allocations (~2MB per device)"
echo "   • Tensor deallocations"
echo "   • 36KB cached buffers remaining"
echo "   • [STEP 4] Cache clearing - WATCH CLOSELY!"
echo "   • Did the 36KB disappear? ← KEY QUESTION"
echo ""
echo "Press Enter when monitor is running to start the test..."
read

# Source environment
if [ -f "/home/tt-metal-apv/build_Release_tracy/env_vars_setup.sh" ]; then
    source /home/tt-metal-apv/build_Release_tracy/env_vars_setup.sh
elif [ -f "/home/tt-metal-apv/build/env_vars_setup.sh" ]; then
    source /home/tt-metal-apv/build/env_vars_setup.sh
fi

export TT_ALLOC_TRACKING_ENABLED=1

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Running test..."
echo "════════════════════════════════════════════════════════════════"
echo ""

python3 test_cache_clear_simple.py

EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Test finished!"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test ran successfully"
    echo ""
    echo "QUESTION: Did the monitor show the 36KB being deallocated in step 4?"
    echo ""
    echo "  [YES] → Program cache clearing is tracked correctly! ✓"
    echo "          The 36KB buffers were deallocated when cache was cleared."
    echo ""
    echo "  [NO]  → Need to investigate MeshBuffer::deallocate() tracking. ✗"
    echo "          The cached kernel buffers were freed but not reported."
    echo ""
else
    echo "✗ Test failed with exit code $EXIT_CODE"
fi

# Cleanup
if [ "$KILL_SERVER" = true ]; then
    echo "Stopping allocation server..."
    pkill -x allocation_server_poc || true
    echo "✓ Server stopped"
fi

exit $EXIT_CODE
