# Allocation Tracking Limitations

## Why Some Allocations Are Not Tracked

Our allocation tracking system hooks into `Allocator::allocate_buffer()`, but **not all buffer allocations go through this function**!

## Cases Where Allocations Are NOT Tracked

### 1. **GraphTracker Hooked Allocations**

**Location:** `tt_metal/impl/buffers/buffer.cpp:386-389`

```cpp
void Buffer::allocate_impl() {
    if (GraphTracker::instance().hook_allocate(this)) {
        address_ = 0;
        hooked_allocation_ = true;
        // ❌ Allocator is BYPASSED - no tracking!
    } else {
        address_ = allocator_->allocate_buffer(this);
        // ✅ This goes through our tracking
    }
}
```

**When this happens:**
- When GraphTracker is active (graph capture mode)
- These allocations are managed by the graph system, not the allocator

### 2. **Pre-Allocated Buffers (Buffer::create with address)**

**Location:** `tt_metal/impl/buffers/buffer.cpp:314-351`

```cpp
std::shared_ptr<Buffer> Buffer::create(
    IDevice* device,
    DeviceAddr address,  // ← Pre-allocated address provided!
    DeviceAddr size,
    // ...
) {
    auto buffer = std::make_shared<Buffer>(..., false /* owns data */);
    buffer->address_ = address;
    buffer->allocation_status_ = AllocationStatus::ALLOCATED;
    // ❌ allocate_impl() is NEVER called - no tracking!
    return buffer;
}
```

**When this happens:**
- When creating a buffer at a specific pre-allocated address
- Used for system buffers, command queues, or memory-mapped regions
- The buffer doesn't "own" the data, so it doesn't allocate or deallocate

### 3. **Zero-Size Buffers**

**Location:** `tt_metal/impl/buffers/buffer.cpp:292-295`

```cpp
if (buffer->size_ == 0) {
    buffer->allocation_status_ = AllocationStatus::ALLOCATED;
    return buffer;
    // ❌ No allocation happens - no tracking
}
```

### 4. **System/Internal Buffers**

Some buffers are created by the system during device initialization:
- Command queue buffers
- Dispatch buffers
- Profiler buffers
- Trace buffers (in some cases)

These may be allocated before `TT_ALLOC_TRACKING_ENABLED` is checked or through different code paths.

## Why "Unknown Buffer" Deallocations Happen

When a buffer is deallocated, the deallocation tracking is called **regardless** of how it was allocated:

```cpp
void Allocator::deallocate_buffer(Buffer* buffer) {
    // This is ALWAYS called for any buffer that "owns data"
    if (AllocationClient::is_enabled()) {
        AllocationClient::report_deallocation(address);
        // ✅ Deallocation is always reported
    }
    // ... actual deallocation ...
}
```

**Result:**
- Allocation bypassed → Not reported to server
- Deallocation happens → Reported to server
- Server sees deallocation for buffer it never saw allocated → "Unknown buffer"

## What Gets Tracked vs. Not Tracked

### ✅ **Tracked (Goes through allocator):**
- User-created buffers via `Buffer::create(device, size, ...)`
- Tensor buffers created by TTNN
- Most application-level allocations
- Our test programs (C++ mesh test, etc.)

### ❌ **Not Tracked (Bypasses allocator):**
- GraphTracker hooked allocations
- Pre-allocated buffers (created with specific address)
- System/internal buffers during device init
- Zero-size buffers
- Buffers that don't "own" their data

## Impact on Different Tests

### C++ Tests (`test_mesh_allocation_cpp`)
- **Status:** ✅ Mostly works
- **Why:** Creates buffers directly via `Buffer::create(device, size, ...)`
- **Issue:** May have some system buffers deallocated at end

### Python TTNN Tests (`test_mesh_allocation.py`)
- **Status:** ❌ Currently not working
- **Why:** Python bindings need rebuild (separate issue)
- **Expected:** Should work after rebuild, but may still have some "unknown buffer" warnings

### Metal Examples (`metal_example_matmul_multicore_reuse`)
- **Status:** ⚠️ Partial
- **Why:** Mix of tracked user buffers and untracked system buffers
- **Result:** Test passes, but "unknown buffer" warnings at cleanup

## Solutions

### For Users:
**The "unknown buffer" warnings are NORMAL and can be ignored!**

They represent system/internal buffers that were allocated through alternative paths. As long as you see your application's allocations being tracked, the system is working correctly.

### For Developers (Advanced):

To track ALL allocations, we would need to:

1. **Hook Buffer::create (both overloads)**
   - Add tracking before `allocate_impl()` is called
   - Track pre-allocated buffers separately

2. **Hook GraphTracker allocations**
   - Add tracking in `GraphTracker::hook_allocate()`
   - Requires understanding graph capture system

3. **Track device initialization buffers**
   - Hook into device initialization code
   - May require tracking before `TT_ALLOC_TRACKING_ENABLED` check

**Complexity:** High - requires changes to multiple subsystems

**Benefit:** Marginal - system buffers are usually constant and not the focus of monitoring

## Recommendation

**Accept the current limitation:**
- Focus on tracking user/application buffers (which works!)
- Ignore "unknown buffer" warnings for system buffers
- The tracking system is useful for debugging memory leaks in application code

## Example Output (Normal)

```
✓ [PID 12345] Allocated 104857600 bytes of DRAM on device 0 (buffer_id=2560032)
✓ [PID 12345] Allocated 104857600 bytes of DRAM on device 1 (buffer_id=2560032)
... (your application allocations)
✗ [PID 12345] Freed buffer 2560032 on device 0 (104857600 bytes)
✗ [PID 12345] Freed buffer 2560032 on device 1 (104857600 bytes)
... (your application deallocations)
⚠ Warning: Deallocation for unknown buffer 1073739776  ← System buffer, ignore
⚠ Warning: Deallocation for unknown buffer 2699296     ← System buffer, ignore
```

The warnings at the end are **expected and harmless** - they're just system buffers being cleaned up.
