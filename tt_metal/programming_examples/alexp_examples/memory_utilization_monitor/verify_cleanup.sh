#!/bin/bash
set -e

echo "========================================"
echo "Verifying Complete Cleanup"
echo "========================================"
echo ""

cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor

# Kill any existing server
pkill -f allocation_server_poc || true
sleep 1

# Start server
echo "Starting server..."
./allocation_server_poc > verify_cleanup.log 2>&1 &
SERVER_PID=$!
sleep 2

export TT_ALLOC_TRACKING_ENABLED=1

echo "Running test..."
python test_mesh_allocation.py > /dev/null 2>&1

echo ""
echo "Waiting 5 seconds for all cleanup to complete..."
sleep 5

echo ""
echo "Checking server log for final state..."
echo "========================================"
tail -30 verify_cleanup.log | grep -A 20 "Current Statistics:" | tail -25

echo ""
echo "========================================"
echo "Key Question: Does it say 'Active allocations: 0'?"
echo "========================================"

kill $SERVER_PID || true
