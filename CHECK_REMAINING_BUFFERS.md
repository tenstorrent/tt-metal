# Investigating Remaining Buffers (36KB + 22KB)

## Steps to Diagnose

### 1. Check if buffers are actually accumulating

Run:
```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor

# Start server
pkill -f allocation_server_poc
./allocation_server_poc > debug_server.log 2>&1 &

# Start monitor in another terminal
./allocation_monitor_client -a -r 500

# Run accumulation test
export TT_ALLOC_TRACKING_ENABLED=1
python test_accumulation.py
```

**Watch the monitor output:**
- After RUN 1: Note DRAM values
- After RUN 2: Check if DRAM increased
- After RUN 3: Check if DRAM increased again

### 2. Check server log for what's NOT being freed

```bash
# After test completes, check server log
tail -200 debug_server.log | grep -E "(Allocated|Freed|Active allocations)"

# Look for patterns like:
# - Allocations that don't have corresponding deallocations
# - Buffer IDs that appear in allocations but not in frees
```

### 3. Identify buffer types

Look at the server statistics output to see:
- Which devices have remaining buffers
- What types of buffers (DRAM vs L1)
- How many buffers remain

### 4. Common Causes

**If accumulation occurs:**

1. **System Buffers Not Freed**: Command queue and dispatch buffers
   - Location: Created during `MeshDevice::initialize()`
   - Expected: Should be freed in `MeshDevice::close()` or device destructor

2. **Trace Buffers**: If tracing is enabled
   - Check if `TT_METAL_SLOW_DISPATCH_MODE` is set

3. **L1 Circular Buffers**: Program-related buffers
   - Note: `track_deallocate_cb` cannot report deallocations (no address available)

4. **Device-Local Buffers Not Marked**: Some buffers might not be going through `mark_as_deallocated()`

## Expected Behavior

**Correct behavior:**
- After each run, DRAM should return to ~0 (or small baseline <100KB)
- Active allocations should be 0
- No "Deallocation for unknown buffer" warnings

**If accumulating:**
- DRAM increases by ~36KB per run
- Active allocations increases
- Need to find which buffers are not being freed

## Next Steps Based on Results

### If NO accumulation (returns to baseline):
✅ Fix is working! The 36KB you saw was transient system buffers.

### If accumulation continues:
Need to add tracking to:
1. Device initialization/cleanup
2. System buffer allocation/deallocation
3. Check if `CloseDevice` is calling proper cleanup
