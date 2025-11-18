# Testing Guide - CB and Kernel Tracking with MeshDevice Support

## Quick Start

### 1. Enable Tracking
```bash
export TT_ALLOC_TRACKING_ENABLED=1
```

### 2. Start Allocation Server
```bash
cd /home/ttuser/aperezvicente/tt-metal-apv
./build/programming_examples/allocation_server_poc > out.log 2>&1 &
```

### 3. Run Test Application
```bash
# Single device test
./build/programming_examples/matmul/matmul_single_core/matmul_single_core

# OR multi-device test (4 devices)
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and DP-4-b1"
```

### 4. Monitor with tt_smi_umd
```bash
./build/programming_examples/tt_smi_umd
# Press '1' for detailed view
# Press '2' for chart view
# Press 'q' to quit
```

## What to Look For

### In tt_smi_umd (View 1)
```
Device 0:
  Memory Breakdown:
    L1 Buffers:  [████░░░░░░] XX.X MB / 306.0 MB (XX.X%)
    CBs:         [██░░░░░░░░]  X.X MB / 306.0 MB ( X.X%)  ← Should see this!
    Kernels:     [█░░░░░░░░░]  X.X MB / 306.0 MB ( X.X%)  ← Should see this!
    Total L1:    [█████░░░░░] XX.X MB / 306.0 MB (XX.X%)
```

### In out.log
```bash
# Check CB allocations
grep "CB_ALLOC" out.log | head -20

# Expected output (multi-device):
✓ [CB_ALLOC] Device 0: +0.25 MB (Total: 0.25 MB)
✓ [CB_ALLOC] Device 1: +0.25 MB (Total: 0.25 MB)
✓ [CB_ALLOC] Device 2: +0.25 MB (Total: 0.25 MB)
✓ [CB_ALLOC] Device 3: +0.25 MB (Total: 0.25 MB)

# Check kernel loads
grep "KERNEL_LOAD" out.log | head -20

# Expected output (multi-device):
✓ [KERNEL_LOAD] Device 0: +2.1 MB (Total: 2.1 MB)
✓ [KERNEL_LOAD] Device 1: +2.1 MB (Total: 2.1 MB)
✓ [KERNEL_LOAD] Device 2: +2.1 MB (Total: 2.1 MB)
✓ [KERNEL_LOAD] Device 3: +2.1 MB (Total: 2.1 MB)
```

## Verification Commands

### Check All Devices Have CB Activity
```bash
grep "CB_ALLOC" out.log | awk '{print $3}' | sort | uniq -c
```
Expected: All devices (0, 1, 2, 3) should appear

### Check All Devices Have Kernel Activity
```bash
grep "KERNEL_LOAD" out.log | awk '{print $3}' | sort | uniq -c
```
Expected: All devices (0, 1, 2, 3) should appear

### Count Messages Per Device
```bash
echo "=== CB Allocations per Device ==="
grep "CB_ALLOC" out.log | awk '{print $3}' | sort | uniq -c

echo "=== Kernel Loads per Device ==="
grep "KERNEL_LOAD" out.log | awk '{print $3}' | sort | uniq -c

echo "=== Total L1 Allocations per Device ==="
grep "Allocated.*L1 on device" out.log | awk '{print $NF}' | tr -d '()' | sort | uniq -c
```

### Check for Deallocations
```bash
grep -E "CB_FREE|KERNEL_UNLOAD" out.log | tail -20
```
Expected: See deallocations when programs are destroyed

## Test Scenarios

### Test 1: Single Device - Simple Program
```bash
# Run
./build/programming_examples/hello_world_compute_kernel/hello_world_compute_kernel

# Verify
grep -E "CB_ALLOC|KERNEL_LOAD" out.log | grep "Device 0"
# Should see both CB and kernel allocations on Device 0
```

### Test 2: Multi-Device - Data Parallel
```bash
# Run (uses 4 devices)
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and DP-4-b1"

# Verify all devices
for dev in 0 1 2 3; do
    echo "=== Device $dev ==="
    grep "CB_ALLOC" out.log | grep "Device $dev:" | wc -l
    grep "KERNEL_LOAD" out.log | grep "Device $dev:" | wc -l
done
# All devices should show non-zero counts
```

### Test 3: Check Deallocation on Program Exit
```bash
# Get current allocation count
BEFORE=$(grep "CB_ALLOC" out.log | wc -l)

# Run and exit a program
./build/programming_examples/matmul/matmul_single_core/matmul_single_core

# Check deallocations
grep "CB_FREE" out.log | tail -10
grep "KERNEL_UNLOAD" out.log | tail -10
# Should see deallocations matching allocations
```

## Troubleshooting

### Problem: No CB_ALLOC messages
**Solution**:
- Check program actually uses circular buffers
- Verify `TT_ALLOC_TRACKING_ENABLED=1` is set
- Rebuild: `cmake --build build --target tt_metal -j8`

### Problem: No KERNEL_LOAD messages
**Solution**:
- Check program has kernels (compute/datamovement)
- Verify tracking is called at dispatch time
- Rebuild: `cmake --build build --target tt_metal -j8`

### Problem: Only Device 0 shows activity (multi-device test)
**Solution**:
- Verify test uses `data_parallel > 1`
- Check MeshDevice detection is working
- Look for "Mesh device: track all sub-devices" in debug logs

### Problem: Server not receiving messages
**Solution**:
```bash
# Check server is running
ps aux | grep allocation_server_poc

# Check socket exists
ls -la /tmp/tt_alloc_server.sock

# Restart server
pkill allocation_server_poc
./build/programming_examples/allocation_server_poc > out.log 2>&1 &
```

### Problem: tt_smi_umd shows 0 for CBs/Kernels
**Solution**:
- Wait a few seconds for initial query
- Check refresh rate (default 500ms)
- Verify server is receiving messages: `tail -f out.log`

## Expected Behavior

### Single Device Program
- ✅ CB_ALLOC messages on Device 0
- ✅ KERNEL_LOAD messages on Device 0
- ✅ tt_smi_umd shows CBs and Kernels for Device 0
- ✅ Other devices show 0 (correct)

### Multi-Device Program (DP-4)
- ✅ CB_ALLOC messages on Devices 0, 1, 2, 3
- ✅ KERNEL_LOAD messages on Devices 0, 1, 2, 3
- ✅ tt_smi_umd shows CBs and Kernels for all 4 devices
- ✅ Counts roughly equal across devices

### On Program Exit
- ✅ CB_FREE messages for all allocated CBs
- ✅ KERNEL_UNLOAD messages for all loaded kernels
- ✅ Memory freed on all devices
- ✅ tt_smi_umd shows decreased usage

## Success Criteria

✅ **CB Tracking Works**: All devices show CB allocations in multi-device test
✅ **Kernel Tracking Works**: All devices show kernel loads at dispatch time
✅ **Deallocation Works**: Memory freed on program exit
✅ **tt_smi_umd Display**: Real-time updates with CBs and Kernels shown
✅ **No Memory Leaks**: Allocations match deallocations over time

## Quick Reference

### Environment Variables
```bash
export TT_ALLOC_TRACKING_ENABLED=1  # Enable tracking
export TT_METAL_SLOW_DISPATCH_MODE=1  # Optional: slow dispatch for debugging
```

### Key Files
- **Server Log**: `out.log`
- **Server Socket**: `/tmp/tt_alloc_server.sock`
- **Monitor**: `./build/programming_examples/tt_smi_umd`

### Key Commands
```bash
# Start server
./build/programming_examples/allocation_server_poc > out.log 2>&1 &

# Monitor
./build/programming_examples/tt_smi_umd

# Check logs
tail -f out.log
grep -E "CB_ALLOC|KERNEL_LOAD" out.log

# Stop server
pkill allocation_server_poc
```

## Need Help?

If tracking isn't working:
1. Check environment variable is set
2. Verify server is running (`ps aux | grep allocation_server`)
3. Check out.log for error messages
4. Verify library was rebuilt with latest changes
5. Test with a simple single-device program first
