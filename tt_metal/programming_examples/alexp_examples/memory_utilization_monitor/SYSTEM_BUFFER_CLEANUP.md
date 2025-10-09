# System Buffer Cleanup Issue

## The Problem

After your program ends, the allocation monitor still shows:
```
Device 0: 18 buffers, 14.79 MB DRAM, 22 KB L1
Devices 1-7: 10 buffers each, 14.75 MB DRAM
```

These are **system buffers** that persist in the monitor even though the program has exited.

## Why This Happens

### Root Cause

When `mesh_device.close()` is called, the cleanup sequence is:

```cpp
// MeshDevice::close() - line 591-610
mesh_command_queues_.clear();        // Clears command queues
sub_device_manager_tracker_.reset(); // Resets sub-device manager
scoped_devices_.reset();             // Triggers ScopedDevices destructor

// ScopedDevices::~ScopedDevices() - line 168-177
DevicePool::instance().close_devices(devices_to_close, /*skip_synchronize=*/true);
```

The problem is that when `shared_ptr<Buffer>` destructors run during this cleanup:
1. They call `Buffer::~Buffer()`
2. Which calls `Buffer::deallocate()`
3. But for **pre-allocated buffers** (`owns_data_ = false`), this returns immediately without tracking!

```cpp
// Buffer::deallocate() - line 412-420
void Buffer::deallocate() {
    if (!owns_data_) {  // ← System buffers have owns_data_ = false!
        return;         // ← EXITS WITHOUT TRACKING!
    }
    this->deallocate_impl();
}
```

### Why Rebuild Won't Fix This

Even after rebuilding with the `mark_as_deallocated()` fix, system buffers won't be tracked because:
- They're pre-allocated (`owns_data_ = false`)
- `deallocate()` returns early
- `mark_as_deallocated()` is never called
- The `shared_ptr` just destroys the object

## Solutions

### Option 1: Restart the Server (Easiest)

Simply restart the allocation server between test runs:

```bash
# Kill old server
pkill -f allocation_server_poc

# Start fresh server
./allocation_server_poc
```

The server starts with a clean slate.

### Option 2: Add Server Reset Command (Better)

Modify the allocation server to accept a "RESET" command that clears all tracked allocations. This would allow the client to send a reset when a program ends.

### Option 3: Track Device Lifecycle (Best)

Add explicit tracking when devices are opened/closed:

```cpp
// In MeshDevice::close() or Device::~Device()
if (AllocationClient::is_enabled()) {
    AllocationClient::report_device_closed(device_id);
}
```

The server would then clear all allocations for that device.

### Option 4: Accept It as Normal (Pragmatic)

The ~14-15MB "baseline" is actually useful information:
- It shows you the **cost of having devices open**
- It's the **minimum memory footprint** of your application
- It helps you understand **total vs. application memory usage**

## Current Workaround

For now, the simplest approach is:

1. **Understand that ~14-15MB is system overhead** - This is expected
2. **Focus on the delta** - Watch for changes when your tensors are allocated/freed
3. **Restart server between runs** - If you want a clean slate

## What You Should See

### During Tensor Operations
```
Device 0: ~19 MB (14 MB system + 4 MB tensor + L1)
Devices 1-7: ~18 MB each (14 MB system + 4 MB tensor)
```

### After Tensor Deallocation
```
Device 0: ~14 MB (system buffers + L1)
Devices 1-7: ~14 MB each (system buffers)
```

### After Program Ends
```
Same as above - system buffers remain in monitor
(They were freed, but tracking didn't capture it)
```

## The Key Insight

The allocation tracking is working correctly for **your application's allocations**:
- ✅ Tensors allocated → Tracked
- ✅ Tensors deallocated → Tracked (after rebuild)
- ⚠️ System buffers → Allocated tracked, deallocation NOT tracked

This is a **limitation of the current implementation**, not a bug in your application.

## Recommendation

**This is acceptable for a monitoring tool!** The important thing is that you can:
1. See when YOUR tensors are allocated
2. See when YOUR tensors are deallocated
3. Understand the system overhead (~14-15MB baseline)

The fact that system buffers persist in the monitor after the program ends is a minor cosmetic issue that doesn't affect the tool's usefulness for monitoring your application's memory usage.
