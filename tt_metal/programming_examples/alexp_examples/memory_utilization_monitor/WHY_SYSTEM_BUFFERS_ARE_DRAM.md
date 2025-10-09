# Why System Buffers Appear as DRAM

## Question
"Why are system buffers appearing as DRAM?"

## Answer

**They ARE DRAM buffers!** This is correct and expected behavior.

## What Are These System Buffers?

The ~14-15MB of DRAM you see on each device are **command queue and dispatch system buffers**. These are essential infrastructure buffers that TT-Metal creates during device initialization to enable communication between the host and the device.

### Buffer Types

According to `tt_metal/impl/flatbuffer/buffer_types.fbs`:

```cpp
enum BufferType: ushort {
  DRAM = 0,           // ← System buffers use this
  L1 = 1,
  SystemMemory = 2,   // ← This is for HOST memory, not device memory
  L1Small = 3,
  Trace = 4,
}
```

**Important:** `BufferType::SYSTEM_MEMORY` refers to **host system memory** (hugepages), NOT device-side system buffers!

## Where Are These Buffers Created?

### 1. During Device Initialization
```cpp
// tt_metal/impl/device/device.cpp
void Device::init_command_queue_host() {
    sysmem_manager_ = std::make_unique<SystemMemoryManager>(this->id_, this->num_hw_cqs());
    // Creates command queues which allocate DRAM buffers on device
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        command_queues_.push_back(std::make_unique<HWCommandQueue>(...));
    }
}
```

### 2. HWCommandQueue Constructor
```cpp
// tt_metal/impl/dispatch/hardware_command_queue.cpp
HWCommandQueue::HWCommandQueue(...) {
    // Allocates DRAM-aligned blocks for command queue operations
    prefetcher_dram_aligned_block_size_ =
        MetalContext::instance().hal().get_alignment(HalMemType::DRAM);
    // ... creates ringbuffer cache managers and other infrastructure
}
```

### 3. Mesh Device Initialization
```cpp
// tt_metal/distributed/fd_mesh_command_queue.cpp
FDMeshCommandQueue::FDMeshCommandQueue(...) {
    // Also uses DRAM alignment for command queue buffers
    prefetcher_dram_aligned_block_size_ =
        MetalContext::instance().hal().get_alignment(HalMemType::DRAM);
}
```

## Why DRAM and Not L1?

System buffers are allocated in **DRAM** (not L1) because:

1. **Size Requirements**: Command queues need ~14-15MB per device
   - L1 is much smaller (~1MB) and reserved for compute operations
   - DRAM is larger (~8-12GB) and suitable for infrastructure

2. **Persistence**: These buffers must persist for the entire device lifetime
   - They handle all host-device communication
   - They manage program dispatch and completion queues

3. **Access Patterns**: Command queue operations involve:
   - Large data transfers between host and device
   - DRAM is optimized for these access patterns

4. **L1 is Reserved**: L1 memory is precious and reserved for:
   - Kernel execution
   - Circular buffers
   - Intermediate computation results

## What's in These 14-15MB?

The system buffers include:

- **Issue Queue**: Commands from host to device
- **Completion Queue**: Responses from device to host
- **Dispatch Buffers**: Program dispatch infrastructure
- **Prefetch Buffers**: Command prefetching and caching
- **Config Buffers**: Worker core configuration
- **Ringbuffer Cache**: For efficient command streaming

## Tracking Implications

### What's Tracked ✅
- **Allocation**: System buffers are tracked when created during `Device::init_command_queue_host()`
- **Type**: Correctly reported as `BufferType::DRAM` (type 0)
- **Size**: ~14-15MB per device

### What's NOT Tracked ❌
- **Deallocation**: System buffers don't call `GraphTracker::track_deallocate()` when freed
  - They use `owns_data_ = false` (pre-allocated addresses)
  - They skip deallocation tracking in `Buffer::deallocate()`

## Is This a Problem?

**No!** This is expected and acceptable for a monitoring tool:

1. **System buffers are constant**: They're allocated once and freed once
2. **You can see your application's memory**: Tensor allocations/deallocations are tracked correctly
3. **The baseline is known**: ~14-15MB DRAM + ~22KB L1 per device
4. **Monitor the delta**: What matters is the *change* in memory usage

## Workaround

If you want a "clean" view between test runs:

```bash
# Restart the server to reset baseline
pkill -f allocation_server_poc
./allocation_server_poc &
```

## Summary

✅ **System buffers ARE DRAM buffers** - this is correct!
✅ **They're essential infrastructure** - required for device operation
✅ **Your tracking works perfectly** - application allocations/deallocations are tracked
✅ **The limitation is cosmetic** - doesn't affect monitoring usefulness

The ~14-15MB DRAM you see is the **cost of doing business** with TT-Metal's command queue system. It's not a bug, it's a feature! 🎯
