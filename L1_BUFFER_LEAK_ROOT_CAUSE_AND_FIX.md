# L1 Buffer Leak - Root Cause Analysis and Fix

## Executive Summary

**3 L1 buffers (12KB) remain on Device 0 after `ttnn.close_mesh_device()`**

These are **pre-allocated control buffers** created during mesh initialization that:
- ✅ ARE tracked by GraphTracker (visible in allocation server)
- ❌ Are NOT in `allocator->allocated_buffers_` (skipped during cleanup)
- Only freed when the Python process terminates

---

## Evidence from Allocation Server

### Initial Allocation (During Mesh Init)
```
✓ [PID 141211] Allocated 4096 bytes of L1 on device 0 (buffer_id=101152)
✓ [PID 141211] Allocated 4096 bytes of L1 on device 0 (buffer_id=105248)
✓ [PID 141211] Allocated 4096 bytes of L1 on device 0 (buffer_id=109344)
```

### Reference Counted (Used During Operations)
```
✓ [PID 141211] Allocated 4096 bytes of L1 on device 0 (buffer_id=101152, ref_count=2)
✓ [PID 141211] Allocated 4096 bytes of L1 on device 0 (buffer_id=105248, ref_count=2)
✓ [PID 141211] Allocated 2048 bytes of L1 on device 0 (buffer_id=109344, ref_count=2)
```
Note: Buffer 109344 size changes from 4096 → 2048 bytes!

### Device Close - Allocator Shows Empty
```
🗑️  SubDeviceManager destructor: cleaning up 1 allocators
   Allocator 0: 0 buffers in allocated_buffers_ set  ← EMPTY!
   Calling deallocate_buffers() for allocator 0...
```

### Final State - Buffers Still Present
```
📊 Current Statistics:
  Device 0:
    Buffers: 3
    L1: 12288 bytes (12 KB)  ← STILL HERE!
  Active allocations: 3
```

### Only Freed When Process Dies
```
⚠️  Detected dead processes, cleaning up orphaned buffers...
   PID 141211 is dead, removing its buffers...
   ✓ Removed 3 buffers (0.0117188 MB) from PID 141211
```

---

## Root Cause Analysis

### Why These Buffers Exist

These are **system control buffers** used for device operations, likely:
- Command queue metadata
- Dispatch coordination structures
- Mesh synchronization primitives

### Why They're Not Tracked by Allocator

They're created via `Buffer::create()` with `owns_data_ = false`:

**File: `buffer.cpp:314-354`**
```cpp
std::shared_ptr<Buffer> Buffer::create(
    IDevice* device,
    DeviceAddr address,  // ← Pre-allocated address
    DeviceAddr size,
    DeviceAddr page_size,
    const BufferType buffer_type,
    const BufferShardingArgs& sharding_args,
    const std::optional<bool> bottom_up,
    const std::optional<SubDeviceId> sub_device_id) {

    auto buffer = std::make_shared<Buffer>(
        device,
        size,
        page_size,
        buffer_type,
        sharding_args,
        bottom_up,
        sub_device_id,
        false /* owns data */,  ← NOT OWNED BY BUFFER!
        Private());

    buffer->address_ = address;  // Use pre-existing address
    buffer->allocation_status_ = AllocationStatus::ALLOCATED;

    // Track for monitoring
    GraphTracker::instance().track_allocate(buffer.get());  ← TRACKED HERE

    return buffer;
}
```

**These buffers SKIP `Allocator::allocate_buffer()`** which is where buffers normally get added to `allocated_buffers_`:

**File: `allocator.cpp:141-144`**
```cpp
DeviceAddr Allocator::allocate_buffer(Buffer* buffer) {
    // ... allocation logic ...
    allocated_buffers_.insert(buffer);  ← ONLY HAPPENS HERE
    return address;
}
```

Pre-allocated buffers never call this function!

### Why Cleanup Fails

**File: `allocator.cpp:173-206`**
```cpp
void Allocator::deallocate_buffers() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!allocated_buffers_.empty()) {
        // ⚠️ Would clean up buffers here...
        // BUT allocated_buffers_ is EMPTY for these 3 L1 buffers!
    }

    // These only clear bank managers, not individual buffers
    dram_manager_->deallocate_all();
    l1_manager_->deallocate_all();  ← Clears banks, but buffers still "exist"
}
```

The bank managers clear their internal state, but the `Buffer` objects with `owns_data_=false` are still alive somewhere, and GraphTracker still tracks them.

### Why Only Device 0?

Device 0 is likely the **primary/coordinator device** in the mesh, responsible for:
- Mesh-wide synchronization
- Coordinating distributed operations
- Managing command queue dispatch across devices

---

## The Fix

### Option 1: Track Pre-Allocated Buffers (RECOMMENDED)

Maintain a separate list of pre-allocated buffers and explicitly deallocate them:

**File: `allocator.hpp`**
```cpp
class Allocator {
private:
    std::unordered_set<Buffer*> allocated_buffers_;      // Normal buffers
    std::unordered_set<Buffer*> preallocated_buffers_;   // NEW: Pre-allocated buffers

public:
    void register_preallocated_buffer(Buffer* buffer);
    void deallocate_preallocated_buffers();  // NEW
};
```

**File: `allocator.cpp`**
```cpp
void Allocator::register_preallocated_buffer(Buffer* buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    preallocated_buffers_.insert(buffer);
}

void Allocator::deallocate_buffers() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Deallocate pre-allocated buffers first
    for (auto* buffer : preallocated_buffers_) {
        GraphTracker::instance().track_deallocate(buffer);
    }
    preallocated_buffers_.clear();

    // Rest of existing cleanup...
    for (auto* buffer : allocated_buffers_) {
        GraphTracker::instance().track_deallocate(buffer);
    }
    allocated_buffers_.clear();

    dram_manager_->deallocate_all();
    l1_manager_->deallocate_all();
    l1_small_manager_->deallocate_all();
    trace_buffer_manager_->deallocate_all();
}
```

**File: `buffer.cpp:340`** (where pre-allocated buffers are created)
```cpp
// Track allocation for pre-allocated buffers (owns_data_ = false)
GraphTracker::instance().track_allocate(buffer.get());

// NEW: Register with allocator for cleanup
if (auto* allocator = device->allocator()) {
    allocator->register_preallocated_buffer(buffer.get());
}
```

### Option 2: Store References to Pre-Allocated Buffers

Keep shared_ptrs to these buffers so they're destroyed during device close:

**File: `device_impl.hpp`**
```cpp
class Device {
private:
    std::vector<std::shared_ptr<Buffer>> control_buffers_;  // NEW

public:
    void register_control_buffer(std::shared_ptr<Buffer> buffer);
};
```

**File: `device.cpp`**
```cpp
void Device::register_control_buffer(std::shared_ptr<Buffer> buffer) {
    control_buffers_.push_back(buffer);
}

bool Device::close() {
    // ... existing cleanup ...

    // Explicitly deallocate control buffers
    for (auto& buffer : control_buffers_) {
        if (buffer && buffer->is_allocated()) {
            buffer->deallocate();
        }
    }
    control_buffers_.clear();

    sub_device_manager_tracker_.reset(nullptr);
    // ... rest of cleanup ...
}
```

### Option 3: Fix Buffer Lifecycle (MOST CORRECT)

Ensure ALL buffers, including pre-allocated ones, go through proper cleanup:

**File: `buffer.cpp:416-428`**
```cpp
void Buffer::deallocate() {
    if (!owns_data_) {
        // Pre-allocated buffers still need to report deallocation
        if (allocation_status_ == AllocationStatus::ALLOCATED && device_->is_initialized()) {
            GraphTracker::instance().track_deallocate(this);
            allocation_status_ = AllocationStatus::DEALLOCATED;

            // NEW: Notify allocator to remove from pre-allocated set
            if (auto* allocator = device_->allocator()) {
                allocator->remove_preallocated_buffer(this);
            }
        }
        return;
    }
    this->deallocate_impl();
}
```

---

## Recommended Action Plan

1. **Implement Option 1** (track pre-allocated buffers separately)
2. **Find where these 3 buffers are created** and register them
3. **Test that cleanup works** (server shows 0 allocations after close)
4. **Verify no regressions** on all device types

---

## Finding Where Buffers Are Created

The buffers are allocated during **mesh device initialization**. Search for:

```bash
# Look for Buffer::create calls with address parameter
grep -r "Buffer::create.*address" tt_metal/

# Look for L1 buffer creation during device init
grep -r "BufferType::L1" tt_metal/impl/device/
grep -r "BufferType::L1" tt_metal/impl/dispatch/

# Look for 4KB allocations
grep -r "4096" tt_metal/impl/dispatch/
grep -r "4096" tt_metal/impl/device/
```

Likely locations:
- `tt_metal/impl/dispatch/hardware_command_queue.cpp`
- `tt_metal/impl/dispatch/system_memory_manager.cpp`
- `tt_metal/impl/device/device.cpp` (command queue initialization)

---

## Success Criteria

After the fix:
```
📊 Current Statistics:
  Active allocations: 0  ✅

✓ No dead process cleanup needed
✓ All devices show 0 allocations after close
✓ Clean exit without orphaned buffers
```

---

## Impact Assessment

- **Memory leak**: 12KB per session (minor but should be fixed)
- **Correctness**: Violates clean shutdown expectations
- **Monitoring**: Causes false positives in tracking systems
- **Production**: Could accumulate in long-running services

**Priority**: Medium (not critical, but should be fixed for correctness)
