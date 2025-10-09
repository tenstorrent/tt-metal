# ✅ MESH BUFFER DEALLOCATION - FINAL FIX

## Problem Summary

When running multi-device mesh tests, the allocation server was showing:
1. **Duplicate allocation warnings**: "Deallocation for unknown buffer"
2. **Memory not being freed**: ~14MB remaining on all devices after test completion
3. **Increasing allocations**: Memory accumulating across successive runs

## Root Cause

The issue was **duplicate allocation/deallocation tracking** for the same buffer address:

### How MeshBuffer Works
1. **Backing Buffer**: Created on `MeshDevice` (conceptual container, allocates on device 0)
2. **Device-Local Buffers**: Created on each physical device (0-7) with the SAME address

### The Problem
```
Buffer 3700768 allocations reported:
✓ Device 0 (backing buffer) ← Tracked
✓ Device 4 (device buffer)
✓ Device 0 (device buffer)  ← DUPLICATE! Overwrites first entry
✓ Device 3, 7, 5, 1, 2, 6...

Result: 9 allocations tracked, but only 8 unique {device_id, buffer_id} pairs in server map
When deallocating: 8 successful + 1 "unknown buffer" warning
```

## The Solution

### 1. Skip Tracking Backing Buffers (graph_tracking.cpp)

Modified `GraphTracker::track_allocate()` and `GraphTracker::track_deallocate()` to detect and skip `MeshDevice` backing buffers:

```cpp
void GraphTracker::track_allocate(const Buffer* buffer) {
    if (AllocationClient::is_enabled() && buffer->device() != nullptr) {
        // Check if this is a MeshDevice (backing buffer) - don't report these
        if (dynamic_cast<const distributed::MeshDevice*>(buffer->device()) != nullptr) {
            return;  // Skip tracking for backing buffers on MeshDevice
        }

        AllocationClient::report_allocation(
            buffer->device()->id(),
            buffer->size(),
            static_cast<uint8_t>(buffer->buffer_type()),
            buffer->address()
        );
    }
}
```

### 2. Track All Device-Local Buffers (buffer.cpp)

Ensured `Buffer::mark_as_deallocated()` tracks ALL buffers, including those with `owns_data_ = false`:

```cpp
void Buffer::mark_as_deallocated() {
    // Track deallocation for ALL buffers, including pre-allocated ones (owns_data_ = false)
    // This ensures we track deallocations on all devices in a mesh
    // Each device's buffer is tracked separately by {device_id, buffer_id} in the server
    if (allocation_status_ == AllocationStatus::ALLOCATED) {
        GraphTracker::instance().track_deallocate(this);
    }
    allocation_status_ = AllocationStatus::DEALLOCATED;
}
```

### 3. Clean MeshBuffer Deallocation (mesh_buffer.cpp)

Simplified `MeshBuffer::deallocate()` to only mark device-local buffers as deallocated:

```cpp
void MeshBuffer::deallocate() {
    auto mesh_device = mesh_device_.lock();
    if (mesh_device) {
        // Mark all device buffers as deallocated
        // Each device's buffer will report its own deallocation to the tracking server
        // The server tracks by {device_id, buffer_id} so each device's deallocation is unique
        for (auto& [coord, buffer_wrapper] : buffers_) {
            if (buffer_wrapper.is_local() && buffer_wrapper.value()) {
                buffer_wrapper.value()->mark_as_deallocated();
            }
        }

        // Note: We do NOT call mark_as_deallocated() on the backing buffer here
        // because the backing buffer's device is a MeshDevice, which is filtered out
        // in GraphTracker to avoid duplicate tracking

        state_ = DeallocatedState{};
        return;
    }
    // ... rest of function
}
```

## Files Modified

1. **`tt_metal/graph/graph_tracking.cpp`**
   - Added `#include <tt-metalium/mesh_device.hpp>`
   - Added `dynamic_cast` check in `track_allocate()` to skip `MeshDevice` buffers
   - Added `dynamic_cast` check in `track_deallocate()` to skip `MeshDevice` buffers

2. **`tt_metal/impl/buffers/buffer.cpp`**
   - Modified `mark_as_deallocated()` to track ALL buffers (removed `owns_data_` check)

3. **`tt_metal/distributed/mesh_buffer.cpp`**
   - Simplified `deallocate()` to only mark device-local buffers as deallocated
   - Removed explicit backing buffer deallocation when mesh device is active

## Results

### Before Fix:
```
Device 0: 14.11 MB DRAM, 22 KB L1
Device 1-7: 14.07 MB DRAM each
Active allocations: 59
⚠ Warning: Deallocation for unknown buffer (multiple)
```

### After Fix:
```
Device 0: 36 KB DRAM, 22 KB L1
Device 1-7: 36 KB DRAM each
Active allocations: 3 (only system buffers)
✅ No warnings
✅ Clean successive runs
```

## Key Insights

1. **Backing Buffer is Virtual**: The backing buffer on `MeshDevice` is a conceptual container. The real allocations are on individual physical devices.

2. **Address Reuse**: All device-local buffers in a mesh share the same address, but they're on different devices, making `{device_id, buffer_id}` the correct unique key.

3. **Dynamic Type Checking**: Using `dynamic_cast` to detect `MeshDevice` is the cleanest way to differentiate backing buffers from device-local buffers.

4. **System Buffers**: The remaining ~36KB per device are command queue and dispatch buffers, which is expected baseline memory usage.

## Verification

Run the test:
```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
export TT_ALLOC_TRACKING_ENABLED=1
./allocation_server_poc &
./allocation_monitor_client -a -r 500 &
python test_mesh_allocation.py
```

Expected: All tensor allocations freed, only system buffers (~36KB) remain.

🎉 **Complete tracking of mesh buffer allocations and deallocations across all 8 devices!**
