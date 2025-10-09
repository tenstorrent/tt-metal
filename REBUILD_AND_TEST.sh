#!/bin/bash
set -e

echo "========================================="
echo "Rebuilding TT-Metal with deallocation fix"
echo "========================================="

cd /home/tt-metal-apv
./build_metal.sh

echo ""
echo "========================================="
echo "Reinstalling Python bindings"
echo "========================================="
pip install -e . --quiet

echo ""
echo "========================================="
echo "Starting allocation server"
echo "========================================="
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
pkill -f allocation_server_poc || true
sleep 1
./allocation_server_poc > server_test.log 2>&1 &
SERVER_PID=$!
echo "Server started (PID: $SERVER_PID)"

echo ""
echo "========================================="
echo "Running mesh allocation test"
echo "========================================="
sleep 2
export TT_ALLOC_TRACKING_ENABLED=1
python test_mesh_allocation.py

echo ""
echo "========================================="
echo "Checking server log for final state"
echo "========================================="
sleep 2
kill $SERVER_PID || true
sleep 1

echo ""
echo "Last 50 lines of server log:"
tail -50 server_test.log

echo ""
echo "========================================="
echo "Checking for 'Active allocations: 0'"
echo "========================================="
if tail -20 server_test.log | grep -q "Active allocations: 0"; then
    echo "✅ SUCCESS: All allocations properly freed!"
else
    echo "❌ FAILED: Still have active allocations"
    tail -20 server_test.log | grep "Active allocations:"
fi
