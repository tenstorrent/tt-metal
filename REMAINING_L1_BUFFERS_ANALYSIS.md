# Remaining L1 Buffers Analysis

## Issue Summary

After running `test_mesh_allocation.py` with allocation tracking enabled, **3 L1 buffers (12KB total) remain allocated on Device 0** even after:
- All tensors explicitly deallocated
- Program cache cleared with `disable_and_clear_program_cache()`
- Mesh device closed with `ttnn.close_mesh_device()`

These buffers are only freed when the process terminates, triggering the dead process cleanup mechanism.

## Evidence from Allocation Server

From the server output:
```
✓ [PID 108740] Allocated 4096 bytes of L1 on device 0 (buffer_id=101152)
✓ [PID 108740] Allocated 4096 bytes of L1 on device 0 (buffer_id=105248)
✓ [PID 108740] Allocated 4096 bytes of L1 on device 0 (buffer_id=109344)

... (test runs, all user tensors deallocated) ...

📊 Current Statistics:
  Device 0:
    Buffers: 3
    DRAM: 0 bytes (0 KB)
    L1: 12288 bytes (12 KB)
    Total: 12288 bytes
  Active allocations: 3

⚠️  Detected dead processes, cleaning up orphaned buffers...
   PID 108740 is dead, removing its buffers...
   ✓ Removed 3 buffers (0.0117188 MB) from PID 108740
```

## What Are These Buffers?

These 3 × 4KB L1 buffers are likely:

### Hypothesis 1: Reference Counted Buffers
Looking at your `graph_tracking.cpp` modifications, I see buffer tracking with reference counting:
```cpp
✓ [PID 108740] Allocated 4096 bytes of L1 on device 0 (buffer_id=101152, ref_count=2)
✓ [PID 108740] Allocated 4096 bytes of L1 on device 0 (buffer_id=105248, ref_count=2)
✓ [PID 108740] Allocated 2048 bytes of L1 on device 0 (buffer_id=109344, ref_count=2)
```

These buffers had `ref_count=2`, suggesting they're shared/system buffers that aren't fully released during normal cleanup.

### Hypothesis 2: Device Control Structures
These could be:
- **Command queue control buffers** - Used for device communication
- **Dispatch core control structures** - Required for fabric/ethernet dispatch
- **Mesh coordination buffers** - Used for multi-device synchronization

### Hypothesis 3: Pre-allocated System Buffers
From `buffer.cpp` modifications:
```cpp
if (!owns_data_) {
    // Pre-allocated buffers (e.g., DRAM, MeshBuffer device-local buffers)
    // don't own memory, so we can't deallocate it
```

These might be pre-allocated control buffers that:
- Don't have explicit deallocation paths
- Are assumed to persist for the device lifetime
- Should be freed during device close but aren't

## Why Aren't They Freed?

### Normal Cleanup Flow
1. ✅ User tensors deallocated → **Works correctly**
2. ✅ Program cache cleared → **Works correctly** (~36KB freed)
3. ✅ Mesh device closed → **Works correctly** (~14-15MB DRAM freed)
4. ❌ **These 3 L1 buffers remain** → **Missing deallocation path**

### Possible Root Causes

#### 1. Missing Deallocation in Device Close
The `ttnn.close_mesh_device()` might not be calling the proper cleanup for these control buffers.

**Investigation needed:**
```bash
# Check device.cpp for device close implementation
git diff tt_metal/impl/device/device.cpp | grep -A 20 "~Device\|close"
```

#### 2. Reference Counting Issue
The buffers show `ref_count=2`, suggesting they're shared. If both references aren't properly released:
- First reference released during normal cleanup
- Second reference remains → leak

**Check:** Are there hidden references in:
- Allocator structures?
- Sub-device manager?
- Mesh coordination layer?

#### 3. Allocator Cleanup Order
From `allocator.cpp`:
```cpp
void Allocator::deallocate_buffers() {
    std::lock_guard<std::mutex> lock(mutex_);
    // By this point, allocated_buffers_ is typically empty
    dram_manager_->deallocate_all();
    l1_manager_->deallocate_all();
    l1_small_manager_->deallocate_all();
}
```

If `allocated_buffers_` doesn't include these control buffers, they won't be tracked and won't be deallocated.

## Investigation Steps

### Step 1: Identify Buffer Origin
Add stack trace tracking to see where these buffers are allocated:

```cpp
// In graph_tracking.cpp::track_allocate()
if (buffer->buffer_type() == BufferType::L1 && buffer->size() == 4096) {
    // Print stack trace for 4KB L1 allocations
    std::cout << "Allocating 4KB L1 buffer " << buffer->address()
              << " from: [add stack trace]" << std::endl;
}
```

### Step 2: Check Reference Counting
Verify reference counting behavior:

```cpp
// Track when ref_count changes
void track_ref_count_change(buffer_id, old_count, new_count, reason) {
    if (buffer_id == 101152 || buffer_id == 105248 || buffer_id == 109344) {
        log("Ref count change: " + buffer_id + " " + old_count + " -> " + new_count);
    }
}
```

### Step 3: Audit Device Close Path
Check if device close properly deallocates all L1:

```cpp
// In device.cpp::~Device() or close()
// Before closing, dump all remaining buffers
allocator_->dump_allocated_buffers();
```

### Step 4: Check Sub-Device Manager
From your device.cpp modification:
```cpp
sub_device_manager_tracker_ = std::make_unique<SubDeviceManagerTracker>(
    this,
    std::move(allocator),  // <-- Allocator ownership transferred
    sub_devices);
```

Does the SubDeviceManager properly clean up when destroyed?

## Workaround Options

### Option 1: Force Cleanup in Device Destructor
```cpp
// In device.cpp::~Device()
if (allocator_) {
    // Force deallocation of all remaining buffers
    allocator_->force_cleanup_all_buffers();
}
```

### Option 2: Track These Buffers Separately
```cpp
// Mark system buffers as "expected to persist"
enum class BufferLifetime {
    USER,      // Should be freed explicitly
    SYSTEM,    // Freed on device close
    PERSISTENT // Expected to persist (these 3 buffers?)
};
```

### Option 3: Accept as Design
If these are truly control buffers required for device operation:
- Document that 12KB L1 per device is expected
- Update the monitoring to show "System L1: 12KB" separately
- Only flag unexpected allocations

## Expected vs Actual Memory After Cleanup

### Expected (per device)
- DRAM: 0 bytes (all freed)
- L1: 0 bytes (all freed)
- Total: 0 bytes

### Actual (Device 0)
- DRAM: 0 bytes ✅
- L1: 12,288 bytes (3 × 4KB buffers) ❌
- Total: 12KB leak

### Other Devices (1-7)
- No remaining allocations ✅

**Key observation:** Only Device 0 has remaining L1. This suggests it's related to:
- Primary device coordination
- Mesh controller role
- First device in topology

## Next Steps

1. **Immediate:** Add logging to identify where these buffers are allocated
2. **Short-term:** Verify device close properly cleans up L1 allocator
3. **Long-term:** Fix the deallocation path or document if intentional

## Related Files

- `tt_metal/impl/buffers/buffer.cpp` - Buffer lifecycle
- `tt_metal/impl/device/device.cpp` - Device initialization/cleanup
- `tt_metal/impl/allocator/allocator.cpp` - Memory management
- `tt_metal/graph/graph_tracking.cpp` - Allocation tracking
- `tt_metal/distributed/sd_mesh_command_queue.cpp` - Mesh coordination

## Questions for Investigation

1. Are these buffers allocated during `open_mesh_device()`?
2. Do they have a corresponding deallocation in `close_mesh_device()`?
3. Why only Device 0 and not devices 1-7?
4. Are they tracked in `allocator->allocated_buffers_`?
5. Should they be freed in `Allocator::deallocate_buffers()`?
