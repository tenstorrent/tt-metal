#!/bin/bash
set -e

echo "Testing successive runs to check for accumulation"
echo "=================================================="

# Kill any existing server
pkill -f allocation_server_poc || true
sleep 1

# Start fresh server
./allocation_server_poc > successive_test.log 2>&1 &
SERVER_PID=$!
sleep 2

export TT_ALLOC_TRACKING_ENABLED=1

echo ""
echo "RUN 1"
echo "-----"
python test_mesh_allocation.py 2>&1 | grep -E "(Step|Active allocations|freed)" | tail -10
sleep 3

echo ""
echo "Checking allocations after RUN 1:"
tail -20 successive_test.log | grep "Active allocations:"

echo ""
echo "RUN 2"
echo "-----"
python test_mesh_allocation.py 2>&1 | grep -E "(Step|Active allocations|freed)" | tail -10
sleep 3

echo ""
echo "Checking allocations after RUN 2:"
tail -20 successive_test.log | grep "Active allocations:"

echo ""
echo "Killing server and checking final log..."
kill $SERVER_PID || true
sleep 1

echo ""
echo "Last 50 lines of server log:"
tail -50 successive_test.log
