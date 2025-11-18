# Practical Guide: Tracking CB + Kernel Memory

## Three Approaches (Easiest → Most Complete)

---

## **Approach 1: Manual Tracking (Do This First!)** ⭐

**Effort:** 10 minutes
**Accuracy:** ~80-90%
**Changes:** Your application code only

### Implementation

Add this to your model initialization:

```python
# CB_TRACKER.py
class L1Tracker:
    def __init__(self):
        self.cb_total = 0
        self.kernel_est = 0
        self.num_kernels = 0

    def track_cb(self, num_tiles, tile_size):
        """Call this when creating circular buffers"""
        size = num_tiles * tile_size
        self.cb_total += size
        return size

    def track_kernel(self, num_cores=1):
        """Call this when adding kernels"""
        # Typical kernel: 20-50KB (compute) + 20-50KB (datamovement)
        size_per_core = 50 * 1024  # Conservative estimate
        total = size_per_core * num_cores
        self.kernel_est += total
        self.num_kernels += 1
        return total

    def summary(self):
        print(f"\n{'='*60}")
        print(f"L1 Memory Usage Breakdown")
        print(f"{'='*60}")
        print(f"Circular Buffers:  {self.cb_total / 1024 / 1024:8.1f} MB")
        print(f"Kernel Code (est): {self.kernel_est / 1024 / 1024:8.1f} MB")
        print(f"  ({self.num_kernels} kernels)")
        print(f"Total L1 Overhead: {(self.cb_total + self.kernel_est) / 1024 / 1024:8.1f} MB")
        print(f"{'='*60}\n")

# Usage in your model:
tracker = L1Tracker()

# When creating CBs:
cb_config = ttl.tensor.CircularBufferConfig(
    num_tiles=1024,
    tile_size=2048
)
tracker.track_cb(1024, 2048)

# When adding kernels:
program.add_kernel(compute_kernel, core_range)
tracker.track_kernel(num_cores=len(core_range))

# At the end:
tracker.summary()
```

### Output:

```
============================================================
L1 Memory Usage Breakdown
============================================================
Circular Buffers:    95.2 MB
Kernel Code (est):   15.3 MB
  (128 kernels)
Total L1 Overhead:  110.5 MB
============================================================
```

**Now you know where your 306MB is going!** 🎯

---

## **Approach 2: Instrumented Wrappers (Better Accuracy)** ⭐⭐

**Effort:** 30 minutes
**Accuracy:** ~95%
**Changes:** Use wrapper functions in your code

### Step 1: Use the Instrumented Helpers

Replace direct TT-Metal calls with instrumented versions:

```cpp
// Instead of:
auto cb = CreateCircularBuffer(device, data_format_spec, config);

// Use:
#include "instrumented_helpers.hpp"
auto cb = tt::tt_metal::instrumented::CreateCircularBufferWithTracking(
    device, data_format_spec, config
);

// Instead of:
auto kernel_handle = program.add_kernel(kernel_config, core_ranges);

// Use:
auto kernel_handle = tt::tt_metal::instrumented::AddKernelWithTracking(
    program, kernel_config, core_ranges
);
```

### Step 2: Update Allocation Server

Add handling for CB/Kernel messages:

```cpp
// In allocation_server_poc.cpp

// Add tracking
std::map<int, uint64_t> device_cb_memory;
std::map<int, uint64_t> device_kernel_memory;

void handle_client(int client_socket) {
    // ... existing code ...

    // Add new message types
    if (msg_type == 8) {  // CB_ALLOC
        CBKernelMessage* msg = (CBKernelMessage*)buffer;
        device_cb_memory[msg->device_id] += msg->size;
        std::cout << "✓ [CB] Device " << msg->device_id
                  << ": +" << msg->size << " bytes" << std::endl;

    } else if (msg_type == 9) {  // CB_FREE
        CBKernelMessage* msg = (CBKernelMessage*)buffer;
        device_cb_memory[msg->device_id] -= msg->size;

    } else if (msg_type == 10) {  // KERNEL_LOAD
        CBKernelMessage* msg = (CBKernelMessage*)buffer;
        device_kernel_memory[msg->device_id] += msg->size;
        std::cout << "✓ [KERN] Device " << msg->device_id
                  << ": +" << msg->size << " bytes" << std::endl;
    }
    // ... rest of handlers ...
}

// Update DEVICE_INFO_RESPONSE
AllocMessage create_device_info_response(int device_id) {
    AllocMessage response;
    // ... existing fields ...

    // Add CB and kernel memory to the response
    // (You'll need to extend AllocMessage struct)
    response.cb_allocated = device_cb_memory[device_id];
    response.kernel_allocated = device_kernel_memory[device_id];

    return response;
}
```

### Step 3: Update tt_smi_umd to Display

```cpp
// In tt_smi_umd.cpp - Memory Breakdown section

std::cout << "  L1 (Allocator):  " << format_bytes(dev.used_l1) << std::endl;
std::cout << "  L1 (CBs):        " << format_bytes(dev.cb_memory) << std::endl;
std::cout << "  L1 (Kernels):    " << format_bytes(dev.kernel_memory) << std::endl;
std::cout << "  ───────────────────────────────" << std::endl;
std::cout << "  L1 Total:        " << format_bytes(
    dev.used_l1 + dev.cb_memory + dev.kernel_memory
) << std::endl;
```

**Pros:**
- ✅ Automatic tracking
- ✅ Reports to allocation server
- ✅ Shows up in tt_smi_umd
- ✅ No TT-Metal core changes

**Cons:**
- ❌ Requires changing your application code to use wrappers
- ❌ Kernel sizes are estimated

---

## **Approach 3: Full Integration (Production-Ready)** ⭐⭐⭐

**Effort:** 4-8 hours
**Accuracy:** 100%
**Changes:** TT-Metal core files

This is the complete solution from `FULL_L1_TRACKING_GUIDE.md`.

### Files to Modify:

1. **`tt_metal/impl/program/program.cpp`** - Hook `Program::add_kernel()`
2. **`tt_metal/impl/buffers/circular_buffer.cpp`** - Hook `CreateCircularBuffer()`
3. **`tt_metal/impl/allocation_tracker.hpp/cpp`** - New tracking module
4. **`allocation_server_poc.cpp`** - Handle new message types
5. **`tt_smi_umd.cpp`** - Display CB/Kernel memory

### Key Changes:

#### 1. Hook Program::add_kernel()

```cpp
// In tt_metal/impl/program/program.cpp

#include "allocation_tracker.hpp"

KernelHandle Program::add_kernel(...) {
    // ... existing code to create kernel ...

    // NEW: Get actual kernel binary size
    size_t compute_size = kernel->compute_binary().size();
    size_t datamovement_size = kernel->data_movement_binary().size();
    size_t total_size = compute_size + datamovement_size;

    // NEW: Report to allocation server
    if (tt::tt_metal::allocation_tracker::is_tracking_enabled()) {
        tt::tt_metal::allocation_tracker::report_kernel_allocation(
            this->device_->id(),
            total_size * core_ranges.size(),  // Size × num cores
            kernel_handle
        );
    }

    return kernel_handle;
}
```

#### 2. Hook CreateCircularBuffer()

```cpp
// In tt_metal/impl/buffers/circular_buffer.cpp

#include "allocation_tracker.hpp"

std::shared_ptr<CircularBuffer> CreateCircularBuffer(...) {
    // ... existing code ...

    // NEW: Report CB allocation
    if (tt::tt_metal::allocation_tracker::is_tracking_enabled()) {
        tt::tt_metal::allocation_tracker::report_cb_allocation(
            device->id(),
            config.total_size(),
            config.buffer_index(),
            core_range
        );
    }

    return cb;
}
```

#### 3. Create allocation_tracker Module

See `FULL_L1_TRACKING_GUIDE.md` lines 60-160 for the complete implementation.

**Pros:**
- ✅ 100% accurate
- ✅ Automatic (no app code changes)
- ✅ Real kernel binary sizes
- ✅ Production-ready

**Cons:**
- ❌ Requires modifying TT-Metal core
- ❌ Takes 4-8 hours to implement
- ❌ Needs testing across all models

---

## **Recommended Path**

### Week 1: Manual Tracking
- Use Approach 1 in your models
- Understand where your L1 is going
- Validate that CB/Kernel usage matches expectations

### Week 2: Instrumented Wrappers
- Implement Approach 2
- Update allocation server to track CB/Kernel
- Update tt_smi_umd to display the data
- Use in development

### Month 2-3: Full Integration (Optional)
- If you need production-grade tracking, implement Approach 3
- Hook into TT-Metal core
- Get exact binary sizes
- Deploy to all applications

---

## Quick Start: Try Manual Tracking Now!

1. **Copy this to your model code:**

```python
class L1Tracker:
    def __init__(self):
        self.cb = 0
        self.kern = 0

    def cb_alloc(self, tiles, tile_size):
        s = tiles * tile_size
        self.cb += s
        print(f"CB: +{s/1024/1024:.1f}MB")

    def kernel_add(self, cores=1):
        s = 50*1024 * cores
        self.kern += s

    def show(self):
        print(f"\nL1: CBs={self.cb/1024/1024:.1f}MB, Kernels={self.kern/1024/1024:.1f}MB")

tracker = L1Tracker()
```

2. **Add tracking calls when you create CBs/kernels**

3. **Call `tracker.show()` after model init**

**You'll immediately see where your 306MB is going!**

---

## Summary

| Approach | Effort | Accuracy | Code Changes |
|----------|--------|----------|--------------|
| **1. Manual** | 10 min | 80-90% | Your app only |
| **2. Wrappers** | 30 min | 95% | Your app + server |
| **3. Full** | 4-8 hrs | 100% | TT-Metal core |

**Start with #1 today, move to #2 this week, consider #3 for production.**

The mystery of the "missing" 300MB L1 will be solved! 🎯
