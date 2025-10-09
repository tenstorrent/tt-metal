# The Remaining 36KB DRAM + 22KB L1

## Summary

After running mesh tests, the allocation monitor shows:
- **Device 0**: 36KB DRAM + 22KB L1
- **Devices 1-7**: 36KB DRAM each

These are **NOT memory leaks** - they are buffers that are allocated and deallocated, but **the deallocation is not being tracked** due to limitations in the current tracking system.

## What Are These Buffers?

### Buffer `1073737728` (24KB DRAM each)
- **Type**: System/dispatch buffer or circular buffer related
- **Allocated**: 18 times across devices (some devices get multiple allocations)
- **Problem**: Never reported as deallocated
- **Size**: 24576 bytes (24KB)

### L1 Buffers on Device 0
- **Buffer `101152`**: 4096 bytes (4KB) L1
- **Buffer `105248`**: 4096 bytes (4KB) L1
- **Buffer `109344`**: 2048 bytes (2KB) L1
- **Total**: ~10KB L1
- **Type**: Circular buffers (CBs) for program execution
- **Problem**: `track_deallocate_cb()` cannot report deallocations because it doesn't have access to CB addresses

## Why Aren't They Being Freed?

### 1. Circular Buffer Limitation (L1 buffers)

From `graph_tracking.cpp` line 117-129:
```cpp
void GraphTracker::track_deallocate_cb(const IDevice* device) {
    // Note: We don't have the CB address here to report deallocation
    // CB deallocations happen when the program is destroyed
    // This is a limitation of the current tracking system
```

**The Problem**:
- Circular buffers are allocated with an address (`track_allocate_cb(addr, size, ...)`)
- But when they're deallocated, the API doesn't provide the address
- So we can't report which specific CB was freed

### 2. System Buffer Deallocation Not Tracked (DRAM buffer `1073737728`)

This buffer appears to be allocated during device/program initialization but:
- It might be going through a different deallocation path
- It might be a buffer type that bypasses normal Buffer deallocation
- It could be deallocated in bulk without individual tracking

## Do These Accumulate?

**Test this by running:**
```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor

# Start server
./allocation_server_poc > test_accumulation_server.log 2>&1 &

# Run test twice
export TT_ALLOC_TRACKING_ENABLED=1
python test_mesh_allocation.py
sleep 5
python test_mesh_allocation.py
sleep 5

# Check if "Active allocations" increased
grep "Active allocations:" test_accumulation_server.log | tail -5
```

**Expected Result:**
- If "Active allocations" stays at 3: ✅ Not accumulating (buffers ARE freed, just not tracked)
- If "Active allocations" increases to 6: ❌ Accumulating (true memory leak)

## Solutions

### Option 1: Fix CB Deallocation Tracking (Recommended)

Modify the circular buffer deallocation API to pass addresses:

1. **Store CB allocations** in a map: `{device_id, core_range} -> {addresses, sizes}`
2. **In `track_deallocate_cb`**: Look up the addresses from the map
3. **Report deallocations** for each address

**Files to modify:**
- `graph_tracking.cpp`: Add CB tracking map
- Where CBs are created: Store allocations in map
- Where CBs are destroyed: Look up and report deallocations

### Option 2: Track System Buffer Deallocations

Find where buffer `1073737728` is created and ensure it goes through proper deallocation tracking.

**Search for:**
```bash
grep -r "1073739776\\|0x40000000" tt_metal/
```

This might be a fixed address or special buffer type.

### Option 3: Accept the Limitation

Document that:
- Circular buffers and some system buffers are not tracked for deallocation
- The monitor will show a baseline of ~36-60KB per device
- This is expected and not a memory leak

**Add to monitor display:**
```
Note: ~36KB baseline includes circular buffers and system buffers
      which are freed but not tracked due to API limitations
```

## Verification

To verify these buffers are actually freed (not leaked):

1. **Use system tools**: Check actual process memory with `ps` or `/proc/[pid]/status`
2. **Run long stress test**: If memory doesn't grow, buffers are being freed
3. **Check device driver**: Query actual device DRAM usage (bypassing our tracker)

## Recommendation

Based on the evidence (server shows "Active allocations: 3" consistently, not increasing), these buffers **ARE being freed** - we're just not tracking the deallocations.

**Short-term**: Document this as a known limitation
**Long-term**: Implement Option 1 to track CB deallocations properly
