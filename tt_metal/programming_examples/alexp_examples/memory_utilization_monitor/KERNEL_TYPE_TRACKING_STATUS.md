# Kernel Type Tracking - Implementation Status

## Overview

Added kernel type tracking to distinguish between:
- **Application kernels** (user programs)
- **Fabric kernels** (inter-device routing, 56 KB each)
- **Dispatch kernels** (command queue infrastructure, 46 KB each)

## Server-Side Changes ✅ DONE

### 1. Message Protocol
- Reused `buffer_type` field in `AllocMessage` for kernel types:
  - `0` = Application kernel
  - `1` = Fabric kernel
  - `2` = Dispatch kernel

### 2. KernelInfo Structure
```cpp
struct KernelInfo {
    uint64_t kernel_id;
    int device_id;
    uint64_t size;
    pid_t owner_pid;
    uint8_t kernel_type;  // NEW: 0=App, 1=Fabric, 2=Dispatch
    std::chrono::steady_clock::time_point alloc_time;
};
```

### 3. Per-Type Statistics
```cpp
struct DeviceStats {
    std::atomic<uint64_t> kernel_allocated;      // Total
    std::atomic<uint64_t> kernel_application;    // App kernels
    std::atomic<uint64_t> kernel_fabric;         // Fabric kernels (56 KB each)
    std::atomic<uint64_t> kernel_dispatch;       // Dispatch kernels (46 KB each)
};
```

### 4. Tracking Handlers
- `KERNEL_LOAD`: Stores kernel type, updates per-type counters
- `KERNEL_UNLOAD`: Reads kernel type, decrements per-type counters
- `cleanup_dead_processes()`: Handles per-type counters during cleanup

### 5. Logging
Now shows kernel type in messages:
```
✓ [KERNEL_LOAD] Fabric kernel on Device 0: +0.0546875 MB
✓ [KERNEL_LOAD] Dispatch kernel on Device 0: +0.0449219 MB
✓ [KERNEL_LOAD] Application kernel on Device 0: +0.1250000 MB
```

## Client-Side Changes ⚠️ TODO

Need to modify the client to send kernel type when reporting:

### Option 1: Detect by Call Stack (Complex)
Parse backtrace to detect `configure_fabric()` or `configure_command_queue_programs()`:

```cpp
// In GraphTracker::track_kernel_load()
uint8_t kernel_type = 0;  // Default: Application

#ifdef TT_METAL_KERNEL_BACKTRACE
// Get call stack
auto bt = tt::assert::backtrace(5, 0);
for (const auto& frame : bt) {
    if (frame.find("configure_fabric") != std::string::npos) {
        kernel_type = 1;  // Fabric
        break;
    } else if (frame.find("configure_command_queue") != std::string::npos) {
        kernel_type = 2;  // Dispatch
        break;
    }
}
#endif

AllocationClient::report_kernel_load(device->id(), kernel_size, kernel_id, kernel_type);
```

### Option 2: Detect by Size (Simple but Fragile)
Use kernel size as heuristic:

```cpp
// In GraphTracker::track_kernel_load()
uint8_t kernel_type = 0;  // Default: Application

// Heuristic based on known system kernel sizes
if (kernel_size == 57344) {       // 56 KB
    kernel_type = 1;  // Fabric
} else if (kernel_size == 47104) { // 46 KB
    kernel_type = 2;  // Dispatch
}

AllocationClient::report_kernel_load(device->id(), kernel_size, kernel_id, kernel_type);
```

### Option 3: Add Explicit Parameter (Clean)
Modify the kernel load functions to accept a kernel type parameter:

```cpp
// In Device::configure_fabric()
detail::ProgramImpl::finalize_offsets(device, KERNEL_TYPE_FABRIC);

// In Device::configure_command_queue_programs()
detail::ProgramImpl::finalize_offsets(device, KERNEL_TYPE_DISPATCH);

// In GraphTracker::track_kernel_load()
void track_kernel_load(uint64_t kernel_size, uint64_t kernel_id,
                       const IDevice* device, uint8_t kernel_type = 0);
```

## Recommendation

**Use Option 2 (Size Heuristic)** for now:
- ✅ Simple to implement
- ✅ No API changes needed
- ✅ Works with current code
- ⚠️ Fragile if kernel sizes change
- ⚠️ May misclassify if user kernel happens to be same size

Later, can upgrade to Option 3 for robustness.

## Implementation Steps

1. ✅ Update server to store and track kernel types
2. ⚠️ Update AllocationClient to accept kernel_type parameter
3. ⚠️ Update GraphTracker to detect/pass kernel type
4. ⚠️ Rebuild both server and tt_metal library
5. ⚠️ Update tt_smi_umd to display per-type kernel breakdown

## Expected Output After Full Implementation

### Server Log:
```
✓ [KERNEL_LOAD] Fabric kernel on Device 0: +0.0546875 MB (Total: 0.0546875 MB)
✓ [KERNEL_LOAD] Fabric kernel on Device 0: +0.0546875 MB (Total: 0.109375 MB)
✓ [KERNEL_LOAD] Dispatch kernel on Device 0: +0.0449219 MB (Total: 0.154297 MB)
✓ [KERNEL_LOAD] Dispatch kernel on Device 0: +0.0449219 MB (Total: 0.199219 MB)
✓ [KERNEL_LOAD] Application kernel on Device 0: +0.1250000 MB (Total: 0.324219 MB)
```

### tt_smi_umd:
```
Device 0 (Blackhole):
  L1 Memory:
    Buffers:        0.0B
    CBs:            2.1 MB
    Kernels:        324.0KB
      - Fabric:     112 KB  (2 kernels)
      - Dispatch:   92 KB   (2 kernels)
      - Application: 120 KB (5 kernels)
    Total:          2.4 MB / 306 MB
```

## Testing

```bash
# Rebuild server
cmake --build build --target allocation_server_poc -j$(nproc)

# Start server
pkill -f allocation_server_poc
./build/programming_examples/allocation_server_poc > test_types.log 2>&1 &

# Run test
export TT_ALLOC_TRACKING_ENABLED=1
python3 -c "import ttnn; ttnn.open_device(0); import time; time.sleep(5)"

# Check log for kernel types
grep "KERNEL_LOAD" test_types.log | head -10
# Should see "Fabric kernel", "Dispatch kernel", "Application kernel"
```

## Status

- ✅ Server-side tracking: IMPLEMENTED
- ⚠️ Client-side reporting: TODO (needs Option 2 or 3)
- ⚠️ tt_smi_umd display: TODO (after client-side done)

Currently, all kernels will be reported as type 0 (Application) until client-side is updated.
