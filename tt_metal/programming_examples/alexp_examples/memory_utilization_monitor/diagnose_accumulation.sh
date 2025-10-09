#!/bin/bash

echo "========================================="
echo "Diagnosing Memory Accumulation"
echo "========================================="
echo ""

cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor

# Kill any existing server
pkill -f allocation_server_poc || true
sleep 1

# Start fresh server with detailed logging
echo "Starting allocation server..."
./allocation_server_poc > accumulation_diagnosis.log 2>&1 &
SERVER_PID=$!
sleep 2

export TT_ALLOC_TRACKING_ENABLED=1

echo ""
echo "Running test 3 times to check for accumulation..."
echo ""

for i in 1 2 3; do
    echo "----------------------------------------"
    echo "RUN $i"
    echo "----------------------------------------"
    python test_accumulation.py 2>&1 | tail -5
    sleep 3

    echo ""
    echo "After RUN $i - Server Statistics:"
    tail -30 accumulation_diagnosis.log | grep -A 20 "Current Statistics:" | tail -20
    echo ""
done

echo ""
echo "Stopping server..."
kill $SERVER_PID || true
sleep 1

echo ""
echo "========================================="
echo "Analysis"
echo "========================================="
echo ""

echo "1. Allocation counts per run:"
grep "Active allocations:" accumulation_diagnosis.log | tail -10

echo ""
echo "2. Checking for buffers that were allocated but never freed:"
echo "   (Looking for buffer IDs in allocations that don't appear in deallocations)"

# Extract all allocated buffer IDs
grep "Allocated.*buffer_id=" accumulation_diagnosis.log | \
    sed 's/.*buffer_id=\([0-9]*\).*/\1/' | sort -u > /tmp/allocated_buffers.txt

# Extract all freed buffer IDs
grep "Freed buffer" accumulation_diagnosis.log | \
    sed 's/.*buffer \([0-9]*\).*/\1/' | sort -u > /tmp/freed_buffers.txt

# Find buffers that were allocated but never freed
echo ""
echo "Buffer IDs allocated but NOT freed:"
comm -23 /tmp/allocated_buffers.txt /tmp/freed_buffers.txt | head -20

echo ""
echo "3. Final server state (last 50 lines):"
tail -50 accumulation_diagnosis.log

echo ""
echo "========================================="
echo "Full log saved to: accumulation_diagnosis.log"
echo "========================================="
