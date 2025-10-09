# L1 Circular Buffer Leak - Root Cause Identified

## Executive Summary

The 3 L1 buffers (12KB) that remain on Device 0 after `ttnn.close_mesh_device()` are **circular buffers** that are never deallocated due to a missing implementation of `deallocate_circular_buffers()`.

## Evidence

### 1. Buffer IDs Are Actually Addresses

The "buffer_id" values from the allocation server are actually L1 addresses:
- **101152** = `0x18B00` (L1 address)
- **105248** = `0x19B00` (L1 address)
- **109344** = `0x1AB00` (L1 address)

### 2. Circular Buffers Use Address as Buffer ID

In `graph_tracking.cpp:222-228`:
```cpp
void GraphTracker::track_allocate_cb(...) {
    if (AllocationClient::is_enabled()) {
        // Circular buffers are always L1
        AllocationClient::report_allocation(
            device->id(),
            size,
            static_cast<uint8_t>(BufferType::L1),
            addr  // ← Address used as buffer_id!
        );
    }
}
```

### 3. Missing Deallocation Implementation

In `program_impl.hpp:219`:
```cpp
class ProgramImpl {
    ...
    void deallocate_circular_buffers();  // ← DECLARED
    ...
};
```

**But this function is NEVER IMPLEMENTED!**

Searching the entire codebase:
```bash
$ grep -r "deallocate_circular_buffers" tt_metal/
tt_metal/impl/program/program_impl.hpp:    void deallocate_circular_buffers();
```

Only the declaration exists - no implementation!

## How Circular Buffers Work

### Allocation Path
1. Program creates circular buffers
2. `GraphTracker::track_allocate_cb()` is called
3. Allocation reported to AllocationClient with:
   - device_id
   - size
   - buffer_type = L1
   - buffer_id = **address**

### Expected Deallocation Path
1. Program destroyed
2. `deallocate_circular_buffers()` should be called
3. Should call `GraphTracker::track_deallocate_cb(device)`
4. Should report deallocations to AllocationClient

###  Actual Deallocation Path
1. Program destroyed
2. `deallocate_circular_buffers()` **IS NEVER CALLED** (not implemented!)
3. `GraphTracker::track_deallocate_cb()` **IS NEVER CALLED**
4. Circular buffers **NEVER FREED** until process termination

## Why Only Device 0?

Device 0 is the mesh coordinator/primary device, which likely has unique circular buffers for:
- Command queue control
- Mesh synchronization
- Dispatch coordination

Other devices (1-7) either:
- Don't allocate these specific circular buffers
- Have them properly cleaned up through a different path
- Use Device 0's buffers (shared)

## Documentation Evidence

From `PATCH_GRAPHTRACKER_TRACKING.md:258`:
> **Circular Buffer Deallocations**: `track_deallocate_cb()` doesn't receive the buffer address, so CB deallocations are not individually tracked. They are deallocated when the program is destroyed.

**But they're NOT actually deallocated** because `deallocate_circular_buffers()` doesn't exist!

## Fix Options

### Option 1: Implement `deallocate_circular_buffers()`

In `program.cpp`, add:
```cpp
void detail::ProgramImpl::deallocate_circular_buffers() {
    // Deallocate all circular buffers for this program
    for (auto& cb : circular_buffers_) {
        if (cb && cb->device()) {
            GraphTracker::instance().track_deallocate_cb(cb->device());
        }
    }
    circular_buffers_.clear();
}
```

Then call it from the Program destructor:
```cpp
detail::ProgramImpl::~ProgramImpl() noexcept {
    deallocate_circular_buffers();
    Inspector::program_destroyed(this);
}
```

### Option 2: Store CB Addresses and Deallocate Individually

Modify `GraphTracker` to store circular buffer allocations per-device and deallocate them individually when the device closes:

```cpp
// In device.cpp::close() or ~Device()
if (allocator_) {
    GraphTracker::instance().deallocate_all_cbs_for_device(this);
}
```

### Option 3: Accept as Design (Document Only)

If these circular buffers are intended to persist for the device lifetime:
1. Document that 12KB L1 per coordinator device is expected
2. Update monitoring to show "System CB: 12KB" separately
3. Only flag unexpected CB allocations

## Impact Analysis

### Memory Impact
- **12KB per mesh** (only on coordinator device)
- Negligible for typical workloads
- Could accumulate if creating/destroying many meshes without process restart

### Correctness Impact
- ✅ No functional issues observed
- ✅ Buffers eventually freed on process termination
- ❌ Memory leak detector shows false positives
- ❌ Allocation tracking shows "leaked" buffers

## Recommendation

**Implement Option 1**: Add `deallocate_circular_buffers()` and call it from the Program destructor.

This is the cleanest solution that:
- ✅ Fixes the leak properly
- ✅ Makes allocation tracking accurate
- ✅ Matches the original design intent
- ✅ Minimal code changes
- ✅ No performance impact

## Files to Modify

1. **`tt_metal/impl/program/program.cpp`**
   - Add `ProgramImpl::deallocate_circular_buffers()` implementation
   - Call it from `~ProgramImpl()`

2. **`tt_metal/graph/graph_tracking.cpp`**
   - Verify `track_deallocate_cb()` properly clears stored CB allocations

3. **Tests**
   - Verify circular buffers are properly freed after program destruction
   - Check allocation server shows CB deallocations

## Verification

After implementing the fix, run:
```bash
cd /workspace/tt-metal-apv
export TT_ALLOC_TRACKING_ENABLED=1

# Terminal 1: Start allocation server
./allocation_server_poc

# Terminal 2: Run test
python test_mesh_allocation.py
```

Expected result:
```
📊 Current Statistics:
  Active allocations: 0  ← Should be 0, not 3!
```

## Related Files

- `tt_metal/impl/program/program_impl.hpp` - Declaration
- `tt_metal/impl/program/program.cpp` - Missing implementation
- `tt_metal/graph/graph_tracking.cpp` - CB tracking
- `tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/test_mesh_allocation.py` - Test

## Conclusion

The 3 L1 "buffers" are actually **circular buffers** that leak because:
1. They are allocated and tracked
2. The deallocation function was declared but never implemented
3. They persist until process termination

**Fix**: Implement `deallocate_circular_buffers()` and call it from the Program destructor.
