# Kernel Deallocation Analysis: tt-metal vs tt-metal-apv

## Problem Statement
In `tt-metal-apv`, kernels are properly deallocated after a workload finishes (showing ~127.4MB after cleanup).
In `tt-metal`, kernels remain allocated even after workloads finish (showing 382.2MB persistent usage).

## Root Cause: Missing Kernel Size Tracking Infrastructure

### Key Difference: Tracking Data Structures

#### tt-metal-apv (Working Version)
The `GraphTracker` class in `tt_metal/api/tt-metalium/graph_tracking.hpp` contains:

```cpp
// Track kernel allocations for proper deallocation
struct KernelAllocation {
    uint64_t kernel_id;
    uint64_t size;        // ← CRITICAL: Stores the kernel size!
};
std::mutex kernel_mutex;
std::unordered_map<const IDevice*, std::vector<KernelAllocation>> device_kernel_allocations;
```

**What this enables:**
1. When `track_kernel_load()` is called, it stores `{kernel_id, total_l1_size}` in the map
2. When `track_kernel_unload()` is called, it looks up the kernel by ID and retrieves the **actual size**
3. The allocation server receives the correct size and can properly deallocate

#### tt-metal (Broken Version)
The `GraphTracker` class **DOES NOT** have these structures. The header ends with:

```cpp
private:
    GraphTracker() = default;
    ~GraphTracker() = default;

    std::vector<std::shared_ptr<IGraphProcessor>> processors;
    std::shared_ptr<IGraphHooks> hook;
    std::mutex hooked_buffers_mutex;
    std::unordered_set<const Buffer*> hooked_buffers;
    // ← Missing: CBAllocation, KernelAllocation, device_kernel_allocations, etc.
};
```

**What this breaks:**
1. `track_kernel_load()` has nowhere to store the kernel size
2. `track_kernel_unload()` cannot look up the size
3. The allocation server receives `size=0` and cannot properly track deallocation

### Code Comparison

#### tt-metal-apv: track_kernel_unload() (Working)
```cpp
void GraphTracker::track_kernel_unload(uint64_t kernel_id, const IDevice* device) {
    // Find and remove this specific kernel from tracking
    std::vector<KernelAllocation> kernels_to_unload;
    {
        std::lock_guard<std::mutex> lock(kernel_mutex);
        auto it = device_kernel_allocations.find(device);
        if (it != device_kernel_allocations.end()) {
            // Find and remove this specific kernel
            auto& kernels = it->second;
            for (auto kernel_it = kernels.begin(); kernel_it != kernels.end(); ++kernel_it) {
                if (kernel_it->kernel_id == kernel_id) {
                    kernels_to_unload.push_back(*kernel_it);  // ← Gets size!
                    kernels.erase(kernel_it);
                    break;
                }
            }
        }
    }

    // Report each kernel unload to the tracking server
    if (device != nullptr) {
        std::lock_guard<std::mutex> tracking_lock(g_allocation_tracking_mutex);
        for (const auto& kernel : kernels_to_unload) {
            if (AllocationClient::is_enabled()) {
                AllocationClient::report_kernel_unload(device->id(), kernel.size, kernel.kernel_id);
                // ↑ Reports ACTUAL size
            }
        }
    }
}
```

#### tt-metal: track_kernel_unload() (Broken)
```cpp
void GraphTracker::track_kernel_unload(uint64_t kernel_id, const IDevice* device) {
    // Report kernel unload to tracking server
    if (device != nullptr) {
        // Note: We don't know the exact size here, server will look it up from kernel_id
        if (AllocationClient::is_enabled()) {
            AllocationClient::report_kernel_unload(device->id(), 0, kernel_id);
            // ↑ Reports size=0 (incorrect!)
        }
    }
}
```

### Why This Causes the 382.2MB Persistent Usage

1. **Kernels are loaded**: The allocation server tracks ~382.2MB of kernel allocations
2. **Workload finishes**: `Program` destructor calls `deallocate_kernel_buffers()`
3. **Unload is called**: But `tt-metal` version sends `size=0` to server
4. **Server doesn't deallocate**: Without the size, the server cannot match and remove the allocation
5. **Memory appears "leaked"**: The 382.2MB remains in the server's accounting

## Solution

The fix requires copying the tracking infrastructure from `tt-metal-apv` to `tt-metal`:

### Files That Need to Be Updated

1. **`tt_metal/api/tt-metalium/graph_tracking.hpp`**
   - Add `CBAllocation` struct
   - Add `KernelAllocation` struct
   - Add `cb_mutex` and `device_cb_allocations`
   - Add `kernel_mutex` and `device_kernel_allocations`

2. **`tt_metal/graph/graph_tracking.cpp`**
   - Update `track_kernel_load()` to store kernel sizes
   - Update `track_kernel_unload()` to retrieve and report kernel sizes
   - Add `g_allocation_tracking_mutex` for thread safety
   - Update `track_allocate_cb()` and `track_deallocate_cb()` similarly for CBs

## Verification

After applying the fix, you should see:
- Kernels deallocate properly when programs are destroyed
- Memory usage in `tt_smi_umd` drops from 382.2MB to ~0MB after workload completes
- Application kernels (not persistent system kernels) are tracked correctly

## Note on Persistent System Kernels

Some kernels (dispatch, fabric) are **designed** to stay loaded for the entire device session. These are:
- Loaded once at device initialization
- Kept in L1 for performance
- Only deallocated when the device is closed

The 382.2MB you saw is a mix of:
- Application kernels (should deallocate) ← **This was broken**
- System kernels (persistent by design) ← **This is correct behavior**

With the fix, only the persistent system kernels remain after workloads complete.
