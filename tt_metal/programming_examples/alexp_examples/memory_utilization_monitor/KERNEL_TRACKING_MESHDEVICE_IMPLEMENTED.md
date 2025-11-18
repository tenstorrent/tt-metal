# Kernel Tracking with MeshDevice Support - Implementation Complete

## Overview

Implemented full kernel memory tracking with MeshDevice support, following the same pattern used for Circular Buffer (CB) tracking.

## Problem Solved

**Issue**: When programs ran on a `MeshDevice`, kernel allocations were only tracked for Device 0, not for all devices in the mesh.

**Root Cause**: The `allocate_kernel_bin_buf_on_device()` function received a `MeshDevice*` (which contains multiple sub-devices) but treated it as a single device, only reporting kernel allocations for the first device.

**Solution**: Detect `MeshDevice` instances using `dynamic_cast`, extract all sub-devices, and track kernel allocations/deallocations for each device in the mesh.

## Changes Made

### 1. GraphTracker API Extensions (`graph_tracking.hpp`)

Added kernel tracking methods to `GraphTracker`:

```cpp
void track_kernel_load(
    uint64_t kernel_size,
    uint64_t kernel_id,
    const IDevice* device);

void track_kernel_unload(
    uint64_t kernel_id,
    const IDevice* device);
```

Added data structure for tracking kernel allocations:

```cpp
struct KernelAllocation {
    uint64_t kernel_id;
    uint64_t size;
};
std::mutex kernel_mutex;
std::unordered_map<const IDevice*, std::vector<KernelAllocation>> device_kernel_allocations;
```

### 2. GraphTracker Implementation (`graph_tracking.cpp`)

Implemented kernel tracking methods:

- **`track_kernel_load()`**: Stores kernel allocation info and reports to `AllocationClient`
- **`track_kernel_unload()`**: Finds and removes kernel from tracking, reports deallocation

Both methods:
- Use mutex protection for thread safety
- Serialize tracking calls to prevent race conditions
- Report to both `AllocationClient` and `TracyMemoryMonitor`

### 3. Program Kernel Allocation Tracking (`program.cpp`)

#### Kernel Load Tracking

Modified `allocate_kernel_bin_buf_on_device()`:

```cpp
// Detect MeshDevice and extract sub-devices
std::vector<const IDevice*> devices_to_track;
const tt::tt_metal::distributed::MeshDevice* mesh_device =
    dynamic_cast<const tt::tt_metal::distributed::MeshDevice*>(device);

if (mesh_device != nullptr) {
    // Track all sub-devices in the mesh
    for (IDevice* sub_device : mesh_device->get_devices()) {
        devices_to_track.push_back(sub_device);
    }
} else {
    // Single device
    devices_to_track.push_back(device);
}

// Report kernel allocation for ALL devices
uint64_t kernel_id = static_cast<uint64_t>(this->get_id());
for (const IDevice* dev : devices_to_track) {
    GraphTracker::instance().track_kernel_load(
        binary_data_size_bytes,
        kernel_id,
        dev);
}
```

#### Kernel Unload Tracking

Added new `deallocate_kernel_buffers()` method:

```cpp
void detail::ProgramImpl::deallocate_kernel_buffers() {
    if (!this->kernels_buffer_.empty()) {
        uint64_t kernel_id = static_cast<uint64_t>(this->get_id());

        for (const auto& [device_id, kernel_buffer] : this->kernels_buffer_) {
            if (kernel_buffer) {
                const IDevice* device = kernel_buffer->device();

                // Apply same MeshDevice detection logic
                std::vector<const IDevice*> devices_to_track;
                // ... (same as above)

                // Report kernel unload for all tracked devices
                for (const IDevice* dev : devices_to_track) {
                    GraphTracker::instance().track_kernel_unload(kernel_id, dev);
                }
            }
        }

        this->kernels_buffer_.clear();
    }
}
```

Called from destructor:

```cpp
detail::ProgramImpl::~ProgramImpl() noexcept {
    deallocate_circular_buffers();
    deallocate_kernel_buffers();  // NEW
    Inspector::program_destroyed(this);
}
```

### 4. Header Declarations (`program_impl.hpp`)

Added method declaration:

```cpp
void deallocate_kernel_buffers();
```

## How It Works

### Kernel Load Flow

```
Program Compilation
    ↓
allocate_kernel_bin_buf_on_device(device)
    ↓
Detect if device is MeshDevice
    ↓
Extract all sub-devices (or use single device)
    ↓
For each device:
    GraphTracker::track_kernel_load(size, kernel_id, device)
        ↓
    AllocationClient::report_kernel_load(device_id, size, kernel_id)
        ↓
    Unix Socket → allocation_server_poc
        ↓
    Updates device_stats.kernel_allocated
```

### Kernel Unload Flow

```
Program Destruction
    ↓
~ProgramImpl()
    ↓
deallocate_kernel_buffers()
    ↓
For each kernel buffer:
    Detect if device is MeshDevice
        ↓
    Extract all sub-devices
        ↓
    For each device:
        GraphTracker::track_kernel_unload(kernel_id, device)
            ↓
        AllocationClient::report_kernel_unload(device_id, kernel_id)
            ↓
        Unix Socket → allocation_server_poc
            ↓
        Updates device_stats.kernel_allocated
```

## Benefits

1. **Accurate Multi-Device Tracking**: Kernel allocations now correctly reported for ALL devices in a mesh
2. **Consistent with CB Tracking**: Uses the exact same MeshDevice detection pattern
3. **Thread-Safe**: Mutex protection prevents race conditions
4. **Automatic Cleanup**: Kernel deallocations tracked automatically in destructor

## Testing

To verify kernel tracking on all devices:

```bash
# Run multi-device test
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and DP-4-b1"

# Monitor with tt_smi_umd (View 1)
./build/programming_examples/tt_smi_umd

# Expected: All 4 devices should show kernel allocations/deallocations
```

## Files Modified

1. `tt_metal/api/tt-metalium/graph_tracking.hpp` - Added kernel tracking API
2. `tt_metal/graph/graph_tracking.cpp` - Implemented kernel tracking methods
3. `tt_metal/impl/program/program.cpp` - Added kernel tracking calls with MeshDevice support
4. `tt_metal/impl/program/program_impl.hpp` - Added `deallocate_kernel_buffers()` declaration

## Related

- **CB Tracking MeshDevice Fix**: Same pattern applied for circular buffers
- **Allocation Server Protocol**: Uses `KERNEL_LOAD` (type 10) and `KERNEL_UNLOAD` (type 11) messages
- **AllocationClient**: Client-side API already existed, now properly hooked up

## Notes

- **Kernel ID**: Uses program ID as kernel identifier (unique per program)
- **Size Tracking**: Tracks DRAM buffer size (where kernel binaries are stored)
- **L1 Memory**: Kernel code memory in L1 is implicitly tracked via this mechanism
- **Ring Buffer Reuse**: Kernel buffers can be reused; tracking happens at program level
