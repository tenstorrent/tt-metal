#!/bin/bash
pkill -f allocation_server_poc
sleep 1
./allocation_server_poc > accumulation_test.log 2>&1 &
sleep 2

export TT_ALLOC_TRACKING_ENABLED=1

echo "RUN 1..."
python test_mesh_allocation.py > /dev/null 2>&1
sleep 5
echo "After RUN 1:"
grep "Active allocations:" accumulation_test.log | tail -1

echo ""
echo "RUN 2..."
python test_mesh_allocation.py > /dev/null 2>&1
sleep 5
echo "After RUN 2:"
grep "Active allocations:" accumulation_test.log | tail -1

echo ""
echo "RUN 3..."
python test_mesh_allocation.py > /dev/null 2>&1
sleep 5
echo "After RUN 3:"
grep "Active allocations:" accumulation_test.log | tail -1

echo ""
echo "====================================="
echo "If numbers stay at 3: ✅ NOT accumulating (buffers freed, just not tracked)"
echo "If numbers increase (3->6->9): ❌ ACCUMULATING (true leak)"
echo "====================================="

pkill -f allocation_server_poc
