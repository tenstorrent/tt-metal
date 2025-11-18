# Why MemoryReporter Integration Was Reverted

## The Problem: Device Conflicts 🚫

We tried to integrate `MemoryReporter` into `tt_smi_umd` to show full L1 memory usage (including circular buffers and kernel code), but it causes **device conflicts** with running applications.

## What Went Wrong

### The Conflict

```
┌─────────────────────────────────────────────────────────┐
│  Your Application (e.g., Llama-3 model)                │
│  ↓                                                       │
│  Creates Device(0)                                       │
│  ↓                                                       │
│  Device is now "owned" by this process                  │
│  ↓                                                       │
│  Allocates buffers, runs kernels, etc.                  │
└─────────────────────────────────────────────────────────┘

                    CONFLICT! ⚠️

┌─────────────────────────────────────────────────────────┐
│  tt_smi_umd (monitoring tool)                           │
│  ↓                                                       │
│  Tries CreateDeviceMinimal(0) for MemoryReporter        │
│  ↓                                                       │
│  ❌ FAILS: Device already in use!                       │
│  ❌ OR: Causes instability in the running app           │
└─────────────────────────────────────────────────────────┘
```

### Why It Happens

**TT-Metal's device ownership model:**
- Only **one process** can "own" a device at a time
- Creating a `Device` instance (even `CreateDeviceMinimal`) requires exclusive access
- Trying to create a second instance conflicts with the first

**MemoryReporter requires a Device instance:**
```cpp
// This requires creating a Device:
auto device = CreateDeviceMinimal(device_id);  // ❌ Conflicts!
auto l1_view = GetMemoryView(device, BufferType::L1);
```

## Why Not nvidia-smi?

**nvidia-smi doesn't have this problem because:**

```
┌────────────────────────────────────────────────┐
│  NVIDIA Architecture                           │
│                                                 │
│  Application                                    │
│  ↓ CUDA Runtime                                │
│  ↓ CUDA Driver (kernel space)                  │
│  ↓ GPU Hardware                                │
│                                                 │
│  nvidia-smi                                     │
│  ↓ NVML Library                                │
│  ↓ CUDA Driver (kernel space) ← Same driver!  │
│  ↓ GPU Hardware                                │
│                                                 │
│  Both use the same kernel driver               │
│  Driver tracks ALL allocations from ALL apps   │
└────────────────────────────────────────────────┘
```

**Tenstorrent Architecture:**

```
┌────────────────────────────────────────────────┐
│  Tenstorrent Architecture                      │
│                                                 │
│  Application                                    │
│  ↓ TT-Metal                                    │
│  ↓ TT-UMD                                      │
│  ↓ TT-KMD (kernel)                             │
│  ↓ Device Hardware                             │
│                                                 │
│  tt_smi_umd                                    │
│  ↓ TT-Metal (CreateDevice)                    │
│  ↓ TT-UMD                                      │
│  ↓ TT-KMD (kernel)                             │
│  ↓ Device Hardware                             │
│                                                 │
│  Problem: Both try to "own" the device!        │
└────────────────────────────────────────────────┘
```

**Key difference:**
- **NVIDIA**: All allocations tracked in **kernel driver** (shared state)
- **Tenstorrent**: Allocations tracked in **user-space allocator** (per-process state)

## The Solution: Allocation Server

This is **exactly why** we built the allocation server!

```
┌─────────────────────────────────────────────────────────┐
│  Application Process                                     │
│  ↓                                                        │
│  CreateDevice(0)                                         │
│  ↓                                                        │
│  Allocate Buffer                                         │
│  ↓                                                        │
│  Send IPC message → Allocation Server                    │
│                     (Unix domain socket)                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Allocation Server (separate process)                    │
│  ↓                                                        │
│  Receives allocation events from ALL processes           │
│  ↓                                                        │
│  Aggregates memory usage                                 │
│  ↓                                                        │
│  Responds to queries from tt_smi_umd                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  tt_smi_umd (monitoring tool)                           │
│  ↓                                                        │
│  Query allocation server via IPC                         │
│  ↓                                                        │
│  ✅ NO device creation needed!                          │
│  ✅ NO conflicts!                                        │
└─────────────────────────────────────────────────────────┘
```

**Benefits:**
- ✅ No device conflicts
- ✅ Cross-process tracking
- ✅ Real-time updates
- ✅ Works with multiple processes using devices

## What About CB + Kernel Memory?

**The missing piece:** Circular buffers and kernel code aren't tracked by the allocation server (yet).

### Why They're Not Tracked

1. **CBs are allocated at kernel setup time**, not through the global allocator
2. **Kernel code is loaded at compile time**, outside the allocator
3. **Allocation server only sees `Buffer::Buffer()` and `Buffer::~Buffer()` calls**

### How to Track Them

See `FULL_L1_TRACKING_GUIDE.md` for the full solution, but here's the summary:

**Option 1: Hook into Program/CB Creation (Best, but requires TT-Metal changes)**
```cpp
// In Program::add_kernel()
KernelHandle Program::add_kernel(const std::shared_ptr<Kernel> &kernel, ...) {
    // ... existing code ...

    size_t kernel_code_size = kernel->compute_binary().size()
                            + kernel->data_movement_binary().size();

    // NEW: Report to allocation server
    report_kernel_allocation(device_id, kernel_code_size);
}

// In CreateCircularBuffer()
CircularBuffer CreateCircularBuffer(Device *device, uint32_t size, ...) {
    // ... existing code ...

    // NEW: Report to allocation server
    report_cb_allocation(device_id, size, buffer_index, core_range);
}
```

**Then the allocation server would track:**
- ✅ Explicit buffers (DRAM, L1, L1_SMALL, TRACE)
- ✅ Circular buffers
- ✅ Kernel code
- ✅ Total = Complete picture!

**Option 2: Use DumpDeviceMemoryState (Quick & dirty)**
```cpp
// In your application (not tt_smi_umd!)
#include <tt-metalium/memory_reporter.hpp>

// After model is loaded:
DumpDeviceMemoryState(device);  // Writes to .reports/tt_metal/*.csv

// Parse the CSV files to see detailed L1 breakdown
```

**Option 3: Manual Accounting**
Track CB sizes in your application code:
```python
# In your model code
total_cb_size = 0
for kernel_config in model.kernel_configs:
    for cb in kernel_config.circular_buffers:
        total_cb_size += cb.size

print(f"Total CB memory: {total_cb_size / 1024 / 1024:.1f} MB")
```

## Current Status

**What tt_smi_umd shows:**
- ✅ Allocator-tracked memory (DRAM, L1, L1_SMALL, TRACE)
- ✅ Cross-process aggregation
- ✅ Real-time updates
- ❌ CB + Kernel memory (not visible)

**Example output:**
```
Memory Breakdown:

Device 0 (Wormhole_B0):
----------------------------------------------------------------------
  DRAM:     2.5GB    / 24.0GB    [███░░░░░░░░░░░░░░░░░░░░░]
  L1:       1.5MB    / 306.0MB   [░░░░░░░░░░░░░░░░░░░░░]
  L1_SMALL: 512KB
  TRACE:    2.3MB
```

**What's missing:**
- ~90-100 MB of circular buffers
- ~10-30 MB of kernel code
- ~5 MB of firmware overhead

**Total "real" L1 usage: ~95-135 MB** (but only 1.5MB visible to allocator!)

## Recommendations

### For Development/Debugging

Use `DumpDeviceMemoryState` in your application:
```python
import tt_lib as ttl

# After model initialization
device = ttl.device.GetDefaultDevice()
ttl.device.DumpDeviceMemoryState(device)

# Check .reports/tt_metal/ directory for CSV files
```

### For Production Monitoring

Keep using `tt_smi_umd` with the allocation server:
- Shows allocator-tracked memory across all processes
- No device conflicts
- Real-time monitoring
- Sufficient for most use cases

### For Complete L1 Tracking

Implement CB/Kernel tracking hooks (see `FULL_L1_TRACKING_GUIDE.md`):
- Requires modifying TT-Metal core
- ~500-1000 lines of code
- Worth it if you need per-core CB visibility

## Lessons Learned

1. **Device ownership matters**: Can't have two processes owning the same device
2. **Kernel-level tracking is powerful**: NVIDIA does it, we should too (future work)
3. **IPC is the right solution**: Allocation server avoids conflicts
4. **Different layers track different things**:
   - TT-KMD: Physical memory pages
   - TT-UMD: Device-level allocator
   - TT-Metal: Application-level buffers
5. **MemoryReporter is for in-process use**: Not for monitoring tools

## Summary

**Why we reverted MemoryReporter integration:**
- ❌ Creates device conflicts with running applications
- ❌ Causes instability
- ❌ Doesn't work when devices are in use

**What we use instead:**
- ✅ Allocation Server (IPC-based, no conflicts)
- ✅ Cross-process tracking
- ✅ Real-time monitoring
- ✅ Works with any number of processes

**Future: Track CB + Kernel memory:**
- Hook into Program/CB creation
- Report to allocation server
- Complete L1 visibility
- See `FULL_L1_TRACKING_GUIDE.md`

---

**The allocation server IS the right solution!** 🎯

It's the equivalent of what NVIDIA's kernel driver does - aggregate allocations from all processes without conflicting with them.
