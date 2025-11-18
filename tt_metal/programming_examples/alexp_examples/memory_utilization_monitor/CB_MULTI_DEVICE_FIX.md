# CB Tracking Multi-Device Fix

## Problems Fixed

### Problem 1: CB Deallocation Showed "-0 MB"
**Root Cause**: `AllocationClient::report_cb_deallocation()` wasn't passing the `size` parameter to the server.

**Symptoms**:
```
✗ [CB_FREE] Device 0: -0 MB (Total: 21.7494 MB)
```

**Fix**:
1. Updated `report_cb_deallocation()` to accept `size` parameter
2. Updated `GraphTracker::track_deallocate_cb()` to pass `cb.size` from stored `CBAllocation`
3. Updated `send_cb_deallocation_message()` to include `msg.size` in the protocol

**Files Modified**:
- `tt_metal/impl/allocator/allocation_client.hpp` - Added size parameter to API
- `tt_metal/impl/allocator/allocation_client.cpp` - Updated implementation
- `tt_metal/graph/graph_tracking.cpp` - Pass size when reporting deallocation

---

### Problem 2: CBs Only Tracked on Device 0
**Root Cause**: `ProgramImpl` only tracked ONE device (`cb_device_`) for CB deallocation, even when programs ran on multiple devices.

**How Multi-Device Programs Work**:
1. Same `Program` object is executed on multiple devices (Device 0, 1, 2, 3)
2. `allocate_circular_buffers(device)` is called for EACH device
3. BUT it early-returns after first call due to `local_circular_buffer_allocation_needed_ = false`
4. Only the FIRST device's CBs were tracked
5. Deallocation only happened for the stored `cb_device_` (first device only)

**Symptoms**:
```
✓ [CB_ALLOC] Device 0: +0.25 MB  ← Device 0 tracked
✓ [CB_ALLOC] Device 0: +0.125 MB ← Device 0 tracked
... (no Device 1, 2, 3 allocations shown)
```

**Fix**:
Changed `ProgramImpl` to track **all devices** that allocate CBs:

```cpp
// BEFORE: Single device
const IDevice* cb_device_ = nullptr;

// AFTER: Multiple devices
std::unordered_set<const IDevice*> cb_devices_;
```

In `allocate_circular_buffers()`:
```cpp
// BEFORE:
this->cb_device_ = device;

// AFTER:
this->cb_devices_.insert(device);  // Tracks ALL devices
```

In `deallocate_circular_buffers()`:
```cpp
// BEFORE:
tt::tt_metal::GraphTracker::instance().track_deallocate_cb(this->cb_device_);

// AFTER:
for (const IDevice* device : this->cb_devices_) {
    tt::tt_metal::GraphTracker::instance().track_deallocate_cb(device);
}
```

**Files Modified**:
- `tt_metal/impl/program/program_impl.hpp` - Changed to `cb_devices_` set
- `tt_metal/impl/program/program.cpp` - Update allocation/deallocation logic

---

## How It Works Now

### Multi-Device CB Allocation Flow

1. **Program Created Once**:
   ```cpp
   Program program;
   CreateCircularBuffer(program, cores, config);  // CBs defined
   ```

2. **Program Executed on Device 0**:
   ```cpp
   allocate_circular_buffers(device0);
   → cb_devices_.insert(device0)  // Track device 0
   → GraphTracker::track_allocate_cb(..., device0)  // Report to server
   → Server: Device 0 cb_allocated += size
   ```

3. **Same Program Executed on Device 1**:
   ```cpp
   allocate_circular_buffers(device1);
   → EARLY RETURN (CBs already allocated)
   → BUT: cb_devices_.insert(device1) happened BEFORE early return!
   ```

**WAIT - We still have the early return problem!**

Let me check the code flow again...

Actually looking at line 843-845:
```cpp
if (not this->local_circular_buffer_allocation_needed_) {
    return;  // Happens BEFORE we insert device!
}
```

The `insert(device)` at line 848 happens AFTER the early return check!

**This means the fix doesn't fully solve the problem yet!**

---

## Additional Fix Needed

We need to track the device EVEN when CBs are already allocated. Let me update:

```cpp
void ProgramImpl::allocate_circular_buffers(const IDevice* device) {
    // Store device for later deallocation tracking (ALWAYS, even if already allocated)
    this->cb_devices_.insert(device);

    // If CBs already allocated, we're done
    if (not this->local_circular_buffer_allocation_needed_) {
        return;
    }

    // ... rest of allocation logic ...
}
```

But wait - this would cause duplicate tracking! Each device would report the SAME CB allocation.

**The real issue**: When a program runs on multiple devices, are CBs allocated ONCE (shared) or PER-DEVICE (separate)?

Let me check if CBs are in L1 (per-core, per-device) or globally shared...

---

## Investigation: Are CBs Per-Device or Shared?

Looking at the code:
- `allocate_circular_buffers()` allocates CBs in **L1 memory** (line 850: `get_base_allocator_addr(HalMemType::L1)`)
- L1 is **per-core, per-device** - NOT shared across devices
- Each device has its OWN L1 memory

**Conclusion**: When a program runs on multiple devices, CBs are allocated SEPARATELY on EACH device's L1!

But the current code:
- Only allocates CBs ONCE (due to `local_circular_buffer_allocation_needed_` flag)
- Assumes the CB addresses are the SAME across all devices
- This works because CB allocation is deterministic (same addresses on all devices)

**So the program assumes CBs have the same layout on all devices, but only does the address calculation once!**

---

## The Correct Fix

Since CB allocation is skipped for devices 1+, but CBs ARE physically present on those devices, we need to:

1. Track allocation for ALL devices (even when layout calculation is skipped)
2. Report the SAME size to the server for each device

Updated fix:

```cpp
void ProgramImpl::allocate_circular_buffers(const IDevice* device) {
    // Calculate CB addresses (only needed once)
    if (this->local_circular_buffer_allocation_needed_) {
        // ... address calculation logic ...
        this->local_circular_buffer_allocation_needed_ = false;
    }

    // Track CB allocation for THIS device (even if layout was cached)
    if (!this->circular_buffers_.empty()) {
        this->cb_devices_.insert(device);
        for (const auto& circular_buffer : this->circular_buffers_) {
            GraphTracker::instance().track_allocate_cb(
                circular_buffer->core_ranges(),
                circular_buffer->address(),  // Same address on all devices
                circular_buffer->size(),
                circular_buffer->globally_allocated(),
                device);  // Different device!
        }
    }
}
```

This way:
- Device 0: Calculates addresses + reports allocation ✅
- Device 1, 2, 3: Skips calculation, but STILL reports allocation ✅
- Deallocation: Reports for ALL devices ✅

---

## Status

✅ **CB deallocation size fix**: Complete
⚠️  **Multi-device tracking**: Partially fixed (need to move tracking outside early-return)

The multi-device fix requires moving the `track_allocate_cb()` call to happen for EVERY device, not just the first one.
