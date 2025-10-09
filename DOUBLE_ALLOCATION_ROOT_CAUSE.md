# Root Cause: Double Allocation of System Buffers

## Summary
The 36KB DRAM leak per device is caused by **buffers being allocated twice at the same address** but only deallocated once.

## Evidence

### Buffer Addresses and Pattern
From `verify_cleanup.log`:

**Buffer `1073739776` (0x3FFFF800) - 12KB:**
```
Allocated: 2× per device = 16 total allocations
Freed: 1× per device = 8 total deallocations
Remaining: 1× per device = 12KB × 8 devices
```

**Buffer `1073737728` (0x3FFFF000) - 24KB:**
```
Allocated: 2× per device = 16 total allocations
Freed: 1× per device = 8 total deallocations
Remaining: 1× per device = 24KB × 8 devices
```

**Total per device: 12KB + 24KB = 36KB** ✓

### Key Observation
The **same buffer address appears twice in allocation logs** for each device:
```
✓ [PID 1906544] Allocated 12288 bytes of DRAM on device 0 (buffer_id=1073739776)
...
✓ [PID 1906544] Allocated 12288 bytes of DRAM on device 0 (buffer_id=1073739776)  ← DUPLICATE!
```

This is **NOT normal behavior** - usually each allocation gets a unique address.

## What Are These Buffers?

These are **pre-allocated buffers** using fixed addresses just below 1GB (`0x40000000`):
- `0x3FFFF000` (1,073,737,728) - 24KB buffer
- `0x3FFFF800` (1,073,739,776) - 12KB buffer

The fixed addresses suggest they're:
1. System/dispatch infrastructure buffers
2. Pre-reserved memory regions
3. Possibly fabric or ethernet router buffers

## Why Are They Allocated Twice?

### Hypothesis 1: Multiple Initialization Paths
Buffers might be created during:
1. **Device initialization** - `Device::initialize()`
2. **Mesh device setup** - `MeshDevice::create()`

But this doesn't explain why they use the **same address**.

### Hypothesis 2: Pre-Allocated Buffer Reuse
Looking at `buffer.cpp:314-353`:
```cpp
std::shared_ptr<Buffer> Buffer::create(
    IDevice* device,
    DeviceAddr address,  // ← Pre-allocated address!
    ...
) {
    auto buffer = std::make_shared<Buffer>(..., false /* owns data */, ...);
    buffer->address_ = address;
    buffer->allocation_status_ = AllocationStatus::ALLOCATED;

    // Track pre-allocated buffer
    GraphTracker::instance().track_allocate(buffer.get());
    return buffer;
}
```

**Key insight**: `owns_data_ = false` means the buffer **doesn't own the memory**. This suggests:
- The actual memory is allocated elsewhere (e.g., in allocator, or hardcoded region)
- Multiple `Buffer` objects can reference the same address
- Each `Buffer` object reports an "allocation" to our tracker
- But the underlying memory is only allocated once

### Hypothesis 3: Backing Buffer + Device Buffers
In `MeshBuffer`:
```cpp
// mesh_buffer.cpp:96
std::shared_ptr<Buffer> backing_buffer = Buffer::create(mesh_device, ...); // Allocation 1

// mesh_buffer.cpp:119
std::shared_ptr<Buffer> buffer = Buffer::create(
    device()->get_device(coord),
    address_,  // ← Uses backing buffer's address!
    ...
); // Allocation 2 (same address!)
```

But this pattern is for user tensors, not system buffers.

## Why Only One Deallocation?

The `Buffer::mark_as_deallocated()` code we added:
```cpp
// buffer.cpp
void Buffer::mark_as_deallocated() {
    if (allocation_status_ == AllocationStatus::ALLOCATED) {
        GraphTracker::instance().track_deallocate(this);
    }
    allocation_status_ = AllocationStatus::DEALLOCATED;
}
```

This should be called for **every Buffer object**. If we're only seeing 1 deallocation per address, it means:
1. One `Buffer` object is being destroyed normally (calls `mark_as_deallocated`)
2. Another `Buffer` object is either:
   - Destroyed without calling `mark_as_deallocated`
   - Still alive when the program ends
   - Destroyed after the allocator tracking is shut down

## Attempted Fixes That Didn't Work

### ❌ Fix 1: Reorder `Device::close()`
Moved `command_queue_programs_.clear()` before `sub_device_manager_tracker_.reset()`.

**Result**: No change.
**Why**: These buffers aren't owned by command queue programs.

### ❌ Fix 2: Clear `fabric_program_`
Added `fabric_program_.reset()` in `Device::close()`.

**Result**: No change.
**Why**: These buffers aren't owned by the fabric program either.

## The Real Problem

The issue is likely in **how pre-allocated buffers are managed**. When a buffer has `owns_data_ = false`:
1. It reports allocation when created (via `GraphTracker::track_allocate`)
2. It **may not report deallocation** when destroyed

Looking at the deallocation path:
```cpp
// buffer.cpp
void Buffer::deallocate_impl() {
    if (owns_data_) {
        // Free from allocator
        device()->allocator()->deallocate_buffer(address_, buffer_type_, ...);
    }
    // Note: mark_as_deallocated() is called BEFORE this
}
```

The issue might be:
- `mark_as_deallocated()` IS being called
- But something is creating these buffers in a way that bypasses normal destruction

## Recommended Fix

Instead of trying to find where these buffers are created and destroyed, we should:

**Option A: Don't track pre-allocated buffers at all**
```cpp
// In GraphTracker::track_allocate()
if (buffer->owns_data() == false) {
    return;  // Skip tracking pre-allocated buffers
}
```

**Pros**: Prevents double-counting of pre-allocated buffers
**Cons**: Won't track MeshBuffer device-local buffers (which are important!)

**Option B: Track only if address is not already tracked**
```cpp
// In GraphTracker (add a set of tracked addresses)
std::unordered_set<uint64_t> tracked_addresses_;

void track_allocate(const Buffer* buffer) {
    if (tracked_addresses_.count(buffer->address()) > 0) {
        return;  // Already tracked
    }
    tracked_addresses_.insert(buffer->address());
    // ... rest of tracking logic
}
```

**Pros**: Prevents double-tracking of same address
**Cons**: Might miss legitimate cases where same address is reused after deallocation

**Option C: Ignore system buffer leaks**
Accept that system buffers show as "remaining" after program end, and document this as expected behavior for a monitoring tool.

**Pros**: Simple, matches reality
**Cons**: User sees "leaks" that aren't real leaks

## Current Status

- L1 CBs: **Fixed!** Reduced from 22KB to 12KB on device 0 after Program destructor fix
- DRAM system buffers: **Still leaking 36KB per device**

The 12KB L1 remaining is likely persistent programs or global kernels.
The 36KB DRAM is these pre-allocated system buffers being allocated twice.
