# Ring Buffer API - Usage Guide

## Overview

We've added public APIs to query the kernel ring buffer state in real-time. This provides visibility into how much L1 memory is currently occupied by cached kernel binaries.

## API Hierarchy

```
RingbufferCacheManager::get_statistics()
    ↓
HWCommandQueue::get_ringbuffer_stats()
    ↓
Device::get_ringbuffer_usage()
```

## Usage Examples

### 1. Query from Device (Easiest)

```cpp
#include <tt-metalium/host_api.hpp>

IDevice* device = tt::tt_metal::CreateDevice(0);

// Get ring buffer usage for default command queue
auto stats = device->get_ringbuffer_usage();

std::cout << "Ring Buffer Usage:" << std::endl;
std::cout << "  Total Size: " << stats.total_size_bytes << " bytes" << std::endl;
std::cout << "  Used: " << stats.used_bytes << " bytes" << std::endl;
std::cout << "  Cached Programs: " << stats.num_cached_programs << std::endl;
std::cout << "  Utilization: " << (100.0 * stats.used_bytes / stats.total_size_bytes) << "%" << std::endl;

tt::tt_metal::CloseDevice(device);
```

### 2. Query specific command queue

```cpp
// Query ring buffer for command queue 1
auto stats = device->get_ringbuffer_usage(1);
```

### 3. Query from CommandQueue directly

```cpp
#include <tt-metalium/host_api.hpp>
#include "tt_metal/impl/dispatch/hardware_command_queue.hpp"

IDevice* device = tt::tt_metal::CreateDevice(0);
auto& cq = device->command_queue();
auto& hw_cq = dynamic_cast<const tt::tt_metal::HWCommandQueue&>(cq);

auto stats = hw_cq.get_ringbuffer_stats();
```

## Return Values

```cpp
struct {
    uint32_t total_size_bytes;      // Total ring buffer capacity (~67-69KB)
    uint32_t used_bytes;             // Currently occupied by cached kernels
    uint32_t num_cached_programs;   // Number of programs in cache
};
```

## Typical Values

- **Total Size**: ~67-69 KB (depends on chip architecture)
- **Used Bytes**: 0 KB to ~67 KB
- **Cached Programs**: 0 to several dozen (depends on workload)

## Integration Points

### For tt_smi_umd

Add real-time ring buffer monitoring:

```cpp
// In TTSmiUMD::run()
for (const auto& dev : devices) {
    auto stats = dev.get_ringbuffer_usage();
    std::cout << "  Kernel Ring Buffer: "
              << format_bytes(stats.used_bytes) << " / "
              << format_bytes(stats.total_size_bytes)
              << " (" << stats.num_cached_programs << " programs cached)"
              << std::endl;
}
```

### For allocation_server_poc

Track ring buffer as part of device stats:

```cpp
// Query each device's ring buffer usage
auto device = tt::tt_metal::CreateDeviceMinimal(device_id);
auto rb_stats = device->get_ringbuffer_usage();

// Store in DeviceInfo response
response.kernel_ringbuffer_total = rb_stats.total_size_bytes;
response.kernel_ringbuffer_used = rb_stats.used_bytes;
response.kernel_ringbuffer_programs = rb_stats.num_cached_programs;
```

## Notes

1. **Fast Dispatch Only**: Ring buffer is only used with Fast Dispatch mode
2. **Per Command Queue**: Each command queue has its own ring buffer
3. **Dynamic**: Values change as programs are dispatched and cached
4. **Eviction**: When ring buffer fills, old kernels are automatically evicted

## What This Gives Us

✅ **Real-time** kernel L1 usage (not just total binary sizes)
✅ **Accurate** measurements for actual cached kernels
✅ **Per-device** granularity
✅ **No guesswork** - direct query from hardware command queue state

This solves the problem where we were tracking total program binary sizes (~70 MB) but actual L1 usage was limited by the ring buffer (~67 KB)!
