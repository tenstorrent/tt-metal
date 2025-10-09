# Mesh Device Allocation Tracking Issues

## Overview

After implementing the GraphTracker patch, allocation tracking works for all 8 devices, but there are three issues:

1. **L1 allocations only show on device 0**
2. **Device 0 has extra buffers compared to other devices**
3. **Deallocations are not tracked properly**

## Issue 1: L1 Only on Device 0

### Why This Happens

**Circular buffers (L1) are program-specific, not mesh-wide!**

Looking at the code flow:

```cpp
// tt_metal/distributed/mesh_workload.cpp:102
void MeshWorkloadImpl::compile_program(..., MeshDevice* mesh_device) {
    auto& program = programs_.at(device_range);
    program.impl().compile(mesh_device);
    program.impl().allocate_circular_buffers(mesh_device);  // ← Called on MeshDevice
    program.impl().validate_circular_buffer_region(mesh_device);
}
```

When `allocate_circular_buffers(mesh_device)` is called with a `MeshDevice*`:
- It uses `mesh_device->allocator()` which returns the **reference device's allocator** (device 0)
- Circular buffers are allocated on **device 0 only**

```cpp
// tt_metal/impl/program/program.cpp:847
void detail::ProgramImpl::allocate_circular_buffers(const IDevice* device) {
    uint64_t base_cb_address = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    // ... allocates CBs using device 0's allocator ...

    GraphTracker::instance().track_allocate_cb(
        circular_buffer->core_ranges(),
        computed_addr,
        circular_buffer->size(),
        circular_buffer->globally_allocated(),
        device);  // ← This is MeshDevice, which reports as device 0!
}
```

### Why This is Correct Behavior

In a mesh workload:
- **DRAM buffers** are replicated across all devices (data buffers)
- **L1 circular buffers** are program state, allocated per-program, not per-device
- The program runs on the "reference device" (device 0) which manages the mesh

### Solution

This is **not a bug** - it's the correct behavior! L1 circular buffers are program-local and only need to be allocated once for the mesh program.

If you want to see L1 allocations on all devices, you would need to run separate programs on each device (not a mesh workload).

---

## Issue 2: Device 0 Has Extra Buffers

### Why This Happens

Device 0 has more buffers because it serves multiple roles:

1. **Backing Buffer** - Line 96-103 in `mesh_buffer.cpp`:
   ```cpp
   std::shared_ptr<Buffer> backing_buffer = Buffer::create(
       mesh_device,  // ← Creates on MeshDevice (device 0)
       device_local_size,
       ...
   );
   ```
   This creates the "master" buffer that allocates the address used by all devices.

2. **Per-Device Buffer** - Line 119-128:
   ```cpp
   std::shared_ptr<Buffer> buffer = Buffer::create(
       device()->get_device(coord),  // ← Creates on each device
       address_,  // ← Uses pre-allocated address
       ...
   );
   ```
   This creates a buffer on each device (including device 0 again) using the same address.

3. **Circular Buffers** - As explained above, L1 CBs are only on device 0.

4. **System Buffers** - Command queues, dispatch buffers, etc. may be on device 0 as the reference device.

### Example from Your Output

```
Device 0: Buffers: 20, DRAM: 14827520 bytes, L1: 22528 bytes
Device 1: Buffers: 10, DRAM: 14753792 bytes
Device 2: Buffers: 10, DRAM: 14753792 bytes
...
```

Device 0 has:
- 10 buffers like other devices (per-device buffers)
- ~10 extra buffers (backing buffers + L1 circular buffers + system buffers)

### Solution

This is **expected behavior**! Device 0 is the "coordinator" device in a mesh and has additional responsibilities.

---

## Issue 3: Deallocations Not Tracked Properly

### Why This Happens

**`MeshBuffer::deallocate()` doesn't actually deallocate the per-device buffers!**

Looking at the code (lines 152-165 in `mesh_buffer.cpp`):

```cpp
void MeshBuffer::deallocate() {
    auto mesh_device = mesh_device_.lock();
    if (mesh_device) {
        state_ = DeallocatedState{};  // ← Just marks as deallocated!
        return;  // ← EXITS WITHOUT DEALLOCATING BUFFERS!
    }

    // This path only runs if mesh_device is already destroyed
    if (std::holds_alternative<OwnedBufferState>(state_)) {
        auto& owned_state = std::get<OwnedBufferState>(state_);
        owned_state.backing_buffer->mark_as_deallocated();  // ← Only marks, doesn't deallocate!
    }
    state_ = DeallocatedState{};
}
```

**The per-device buffers (`buffers_` map) are never explicitly deallocated!**

They are deallocated implicitly when:
1. The `MeshBuffer` destructor runs
2. The `buffers_` map is destroyed
3. The `shared_ptr<Buffer>` destructors run
4. Each `Buffer` destructor calls `deallocate_impl()`

### Why Only Device 0 Shows Deallocations

From your output:
```
✗ [PID 1691149] Freed buffer 3700768 on device 0 (1048576 bytes)
✗ [PID 1691149] Freed buffer 3610656 on device 0 (524288 bytes)
✗ [PID 1691149] Freed buffer 3655712 on device 0 (524288 bytes)
```

Only device 0 buffers are being deallocated because:
1. The **backing buffer** (owned by device 0) is deallocated
2. The **per-device buffers** on devices 1-7 are **pre-allocated** (`owns_data_ = false`)
3. Pre-allocated buffers don't call `deallocate_impl()` in their destructor!

Looking at `Buffer::deallocate()` (line 412-417):

```cpp
void Buffer::deallocate() {
    if (!owns_data_) {  // ← Pre-allocated buffers (devices 1-7) have owns_data_ = false
        return;  // ← EXITS WITHOUT DEALLOCATING!
    }
    this->deallocate_impl();
}
```

### Solution

We need to track deallocations for pre-allocated buffers too! Add tracking to `Buffer::mark_as_deallocated()`:

```cpp
void Buffer::mark_as_deallocated() {
    // Track deallocation even for pre-allocated buffers
    if (allocation_status_ == AllocationStatus::ALLOCATED) {
        GraphTracker::instance().track_deallocate(this);
    }
    allocation_status_ = AllocationStatus::DEALLOCATED;
}
```

This will catch deallocations for:
- Pre-allocated buffers (devices 1-7)
- Buffers deallocated when mesh device is destroyed
- Any buffer that's marked as deallocated without going through `deallocate_impl()`

---

## Summary

| Issue | Root Cause | Status | Fix Needed? |
|-------|-----------|--------|-------------|
| **L1 only on device 0** | Circular buffers are program-local, not mesh-wide | ✅ Expected | ❌ No - this is correct |
| **Device 0 extra buffers** | Device 0 is coordinator with backing buffers + system buffers | ✅ Expected | ❌ No - this is correct |
| **Deallocations not tracked** | Pre-allocated buffers don't call `deallocate_impl()` | ❌ Bug | ✅ Yes - add tracking to `mark_as_deallocated()` |

## Testing the Fix

After applying the deallocation fix, you should see:

```
✗ [PID xxx] Freed buffer 3700768 on device 0 (1048576 bytes)
✗ [PID xxx] Freed buffer 3700768 on device 1 (1048576 bytes)
✗ [PID xxx] Freed buffer 3700768 on device 2 (1048576 bytes)
✗ [PID xxx] Freed buffer 3700768 on device 3 (1048576 bytes)
✗ [PID xxx] Freed buffer 3700768 on device 4 (1048576 bytes)
✗ [PID xxx] Freed buffer 3700768 on device 5 (1048576 bytes)
✗ [PID xxx] Freed buffer 3700768 on device 6 (1048576 bytes)
✗ [PID xxx] Freed buffer 3700768 on device 7 (1048576 bytes)
```

All devices should show deallocations with the same buffer address (because they share the same pre-allocated address).
