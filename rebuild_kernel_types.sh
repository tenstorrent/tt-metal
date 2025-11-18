#!/bin/bash
# Rebuild with kernel type tracking

cd /home/ttuser/aperezvicente/tt-metal-apv

echo "====================================================================="
echo "  Rebuilding with Kernel Type Tracking"
echo "====================================================================="
echo ""

echo "Step 1: Rebuild allocation server"
cmake --build build --target allocation_server_poc -j$(nproc) 2>&1 | tail -5
echo "  ✓ Server rebuilt"

echo ""
echo "Step 2: Rebuild tt_metal library (for GraphTracker + AllocationClient)"
cmake --build build --target tt_metal -j$(nproc) 2>&1 | tail -5
echo "  ✓ tt_metal rebuilt"

echo ""
echo "Step 3: Restart server"
pkill -f allocation_server_poc
sleep 2
./build/programming_examples/allocation_server_poc > out_kernel_types.log 2>&1 &
echo "  ✓ Server started (PID: $!)"

echo ""
echo "Step 4: Test kernel type tracking"
source python_env/bin/activate
export TT_ALLOC_TRACKING_ENABLED=1
python3 -c "import ttnn; dev = ttnn.open_device(0); import time; time.sleep(3); ttnn.close_device(dev)"

echo ""
echo "Step 5: Check kernel types in log"
echo "  Looking for 'Fabric kernel' and 'Dispatch kernel' messages..."
grep -E "Fabric kernel|Dispatch kernel|Application kernel" out_kernel_types.log | head -10

echo ""
echo "====================================================================="
echo "  Done! Check out_kernel_types.log for full details"
echo "====================================================================="
