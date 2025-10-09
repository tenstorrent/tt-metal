# How to Fix the Double-Free Bug

## Problem Summary
Buffers are being freed multiple times, causing "unknown buffer" warnings. The tracking system correctly detects this - now we need to fix the root cause.

---

## Step 1: Identify Where Double-Frees Happen

### A. Check mark_as_deallocated() vs deallocate() Calls

**File to investigate:** `tt_metal/impl/buffers/buffer.cpp`

Both functions send deallocation messages:

```cpp
void Buffer::mark_as_deallocated() {
    if (allocation_status_ == AllocationStatus::ALLOCATED) {
        GraphTracker::instance().track_deallocate(this);  // ← FREE message #1
    }
    allocation_status_ = AllocationStatus::DEALLOCATED;
}

void Buffer::deallocate_impl() {
    if (device_->is_initialized() && size_ != 0) {
        GraphTracker::instance().track_deallocate(this);  // ← FREE message #2
        // ...
    }
    allocation_status_ = AllocationStatus::DEALLOCATED;
}
```

**Search for the issue:**
```bash
cd /home/tt-metal-apv
grep -rn "mark_as_deallocated" tt_metal/ ttnn/ | grep -v "\.hpp:" | head -20
```

**Look for patterns like:**
```cpp
// BAD: Both functions called on same buffer
buffer->mark_as_deallocated();
buffer->deallocate();  // ← Sends FREE again!
```

**Fix:** Remove duplicate calls. Use ONLY `deallocate()` or ONLY `mark_as_deallocated()`.

---

## Step 2: Fix MeshDevice Buffer Cleanup

### B. Check MeshDevice Destructor

**Files to investigate:**
- `tt_metal/impl/tt_metal/mesh_device.cpp`
- `ttnn/cpp/ttnn/operations/core/core.cpp` (close_mesh_device)

**Search for the issue:**
```bash
cd /home/tt-metal-apv
grep -rn "close_mesh_device\|~MeshDevice" tt_metal/ ttnn/ --include="*.cpp" -A 10 | head -50
```

**Look for patterns like:**
```cpp
// BAD: Multiple cleanup levels
void MeshDevice::cleanup() {
    // Cleanup all device buffers
    for (auto& device : devices_) {
        device->cleanup_buffers();  // ← Frees buffers
    }

    // Also cleanup mesh-level buffers
    mesh_buffers_.clear();  // ← Frees same buffers again!
}
```

**Fix:** Ensure each buffer is freed exactly once, at only one level of ownership.

---

## Step 3: Add Protection Against Double-Frees

### C. Add Guard in Buffer::deallocate_impl()

**File:** `tt_metal/impl/buffers/buffer.cpp`

**Current code (~line 432):**
```cpp
void Buffer::deallocate_impl() {
    if (allocation_status_ == AllocationStatus::ALLOCATION_REQUESTED) {
        return;
    }

    if (device_->is_initialized() && size_ != 0) {
        GraphTracker::instance().track_deallocate(this);
        // ...
    }

    allocation_status_ = AllocationStatus::DEALLOCATED;
}
```

**Fixed code:**
```cpp
void Buffer::deallocate_impl() {
    // GUARD: Already deallocated
    if (allocation_status_ == AllocationStatus::DEALLOCATED) {
        return;  // ← Prevents double-free
    }

    if (allocation_status_ == AllocationStatus::ALLOCATION_REQUESTED) {
        return;
    }

    if (device_->is_initialized() && size_ != 0) {
        GraphTracker::instance().track_deallocate(this);
        // ...
    }

    allocation_status_ = AllocationStatus::DEALLOCATED;
}
```

**Apply the fix:**
```bash
cd /home/tt-metal-apv
```

---

## Step 4: Add Debug Assertions

### D. Detect Double-Frees in Debug Builds

**File:** `tt_metal/impl/buffers/buffer.cpp`

Add assertion in `deallocate_impl()`:

```cpp
void Buffer::deallocate_impl() {
    // DEBUG: Catch double-frees
    TT_FATAL(
        allocation_status_ != AllocationStatus::DEALLOCATED,
        "Double-free detected! Buffer {} on device {} was already deallocated",
        address_, device_->id()
    );

    // ... rest of function ...
}
```

This will make tests FAIL if double-free occurs, making the bug obvious.

---

## Step 5: Fix Specific Known Issues

### E. Check Buffer Sharing in MeshDevice

**Problem:** Same buffer ID used on multiple devices, unclear ownership.

**File:** `ttnn/cpp/ttnn/tensor/tensor.cpp` or `tt_metal/impl/buffers/buffer.cpp`

**Search:**
```bash
cd /home/tt-metal-apv
grep -rn "replicate\|distribute.*buffer\|mesh.*buffer" ttnn/ --include="*.cpp" -B 5 -A 5 | head -100
```

**Look for:**
```cpp
// BAD: Buffer shared but cleanup unclear
std::shared_ptr<Buffer> create_mesh_buffer(...) {
    auto buffer = Buffer::create(...);

    // Replicate to all devices
    for (auto& device : mesh_devices) {
        device->add_buffer(buffer);  // ← Who owns it?
    }

    return buffer;  // ← Multiple references!
}
```

**Fix:** Use `std::shared_ptr` properly or implement clear ownership:
```cpp
// GOOD: Clear ownership
std::shared_ptr<Buffer> create_mesh_buffer(...) {
    auto buffer = Buffer::create(...);

    // Replicate to all devices with shared ownership
    for (auto& device : mesh_devices) {
        device->add_buffer(buffer);  // shared_ptr ref count ++
    }

    return buffer;  // Destructor will only free when ALL refs are gone
}
```

---

## Step 6: Implement the Actual Fix

### Recommended Approach: Guard Against Double-Frees

**File to modify:** `tt_metal/impl/buffers/buffer.cpp`
