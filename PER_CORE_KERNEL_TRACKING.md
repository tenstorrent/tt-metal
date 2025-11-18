# Per-Core Kernel Tracking Implementation

## Problem
The previous kernel tracking only reported the **binary size** (e.g., 56 KB for Fabric, 46 KB for Dispatch), which is the DRAM buffer size. However, **kernels are distributed across multiple cores**, so the actual L1 memory usage is:

```
Actual L1 Usage = Binary Size × Number of Cores
```

For example:
- **Fabric kernel**: 56 KB binary, runs on **16 ethernet cores** = **896 KB actual L1 usage**
- **Dispatch kernel**: 46 KB binary, runs on **2 dispatch cores** = **92 KB actual L1 usage**

## Solution Implemented

### 1. Updated `track_kernel_load` Signature
**File**: `tt_metal/api/tt-metalium/graph_tracking.hpp`

```cpp
void track_kernel_load(
    uint64_t kernel_size,       // Binary size (DRAM buffer size)
    uint64_t kernel_id,
    const IDevice* device,
    uint8_t kernel_type = 0,    // 0=Application, 1=Fabric, 2=Dispatch
    uint32_t num_cores = 1);    // NEW: Number of cores this kernel runs on
```

### 2. Updated `track_kernel_load` Implementation
**File**: `tt_metal/graph/graph_tracking.cpp`

**Key Changes**:
- Calculate `total_l1_size = kernel_size * num_cores`
- Store `total_l1_size` for accurate deallocation tracking
- Report `total_l1_size` to the allocation server
- Log all details in backtrace:
  - Binary Size (e.g., 56 KB)
  - Number of Cores (e.g., 16)
  - Total L1 Usage (e.g., 896 KB)

```cpp
void GraphTracker::track_kernel_load(..., uint32_t num_cores) {
    uint64_t total_l1_size = kernel_size * num_cores;

    // Store total L1 size for accurate deallocation
    device_kernel_allocations[device].push_back({kernel_id, total_l1_size});

    // Log details
    std::cout << "Binary Size: " << (kernel_size / 1024.0) << " KB"
              << ", Cores: " << num_cores
              << ", Total L1: " << (total_l1_size / 1024.0) << " KB" << std::endl;

    // Report total L1 size to allocation server
    AllocationClient::report_kernel_load(device->id(), total_l1_size, kernel_id, kernel_type);
}
```

### 3. Updated Kernel Tracking Call Sites
**File**: `tt_metal/impl/program/program.cpp`

In `ProgramImpl::finalize_offsets()` (Slow Dispatch path):

```cpp
// Calculate number of cores this program runs on
uint32_t total_cores = 0;
std::vector<std::vector<CoreCoord>> logical_cores_list = this->logical_cores();
for (const auto& cores_for_type : logical_cores_list) {
    total_cores += cores_for_type.size();
}

// Track kernel load with core count
for (const IDevice* dev : devices_to_track) {
    tt::tt_metal::GraphTracker::instance().track_kernel_load(
        kernel_size,
        kernel_id,
        dev,
        kernel_type,
        total_cores);  // Pass number of cores
}
```

### 4. Removed Duplicate Tracking
**File**: `tt_metal/tt_metal.cpp`

**Removed**: The `detail::TrackKernelDispatch()` call in `LaunchProgram()` to prevent double-counting.

**Why**: Kernel tracking is now done exclusively in `finalize_offsets()`, which is called for both Slow Dispatch (directly) and Fast Dispatch (internally).

## How to Use

### Rebuild
```bash
cd /home/ttuser/aperezvicente/tt-metal-apv
cmake --build build --target tt_metal -j$(nproc)
```

### Test
```bash
# Start allocation server
./build/programming_examples/allocation_server_poc > out.log 2>&1 &

# Run a test
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and DP-4-b1" 2>&1 | tee trace_kernels.log

# Check the logs
grep "KERNEL_LOAD" trace_kernels.log
```

### Expected Output (Example)

**Old Output** (binary size only):
```
🔍 KERNEL_LOAD Backtrace (Device 0, Type: Fabric, Size: 56 KB, ID: 0x...)
```

**New Output** (with core count and total L1):
```
🔍 KERNEL_LOAD Backtrace (Device 0, Type: Fabric, Binary Size: 56 KB, Cores: 16, Total L1: 896 KB, ID: 0x...)
```

**Allocation Server Log**:
```
✓ [KERNEL_LOAD] Fabric kernel on Device 0: +0.875 MB (Total: 0.875 MB)
✓ [KERNEL_LOAD] Dispatch kernel on Device 0: +0.090 MB (Total: 0.965 MB)
```

## Benefits

1. **Accurate L1 Tracking**: Now tracks actual L1 memory footprint, not just binary size
2. **Visibility**: Logs show both binary size and per-core distribution
3. **Correct Deallocation**: Stores total L1 size, so deallocation subtracts the correct amount
4. **Consistent Tracking**: Works for all kernel types (Application, Fabric, Dispatch)

## Technical Details

### Core Count Calculation
Uses `ProgramImpl::logical_cores()` which returns a vector of vectors:
- Outer vector: One entry per programmable core type (TENSIX, ACTIVE_ETH, etc.)
- Inner vector: List of logical core coordinates for that type

Total cores = sum of all core lists across all types.

### Example: Fabric Kernel
- **Binary Size**: 57344 bytes (56 KB)
- **Runs on**: 16 ethernet cores
- **Total L1 Usage**: 57344 × 16 = 917504 bytes (896 KB)
- **Reported to server**: 917504 bytes

### Example: Application Kernel
- **Binary Size**: 100 KB
- **Runs on**: 80 Tensix cores
- **Total L1 Usage**: 100 KB × 80 = 8000 KB (7.8 MB)
- **Reported to server**: 8000 KB

## Files Modified

1. `tt_metal/api/tt-metalium/graph_tracking.hpp` - Added `num_cores` parameter
2. `tt_metal/graph/graph_tracking.cpp` - Updated implementation to calculate and report total L1
3. `tt_metal/impl/program/program.cpp` - Calculate core count and pass to tracker
4. `tt_metal/tt_metal.cpp` - Removed duplicate `TrackKernelDispatch()` call

## Next Steps

After rebuild, you should see:
- More accurate L1 kernel usage in `tt_smi_umd`
- Fabric kernels showing ~896 KB per device (not 56 KB)
- Dispatch kernels showing ~92 KB per device (not 46 KB)
- Total system kernels: ~988 KB per device (not ~102 KB)
