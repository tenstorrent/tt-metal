# Memory Monitor Limitations - Cross-Process Memory Tracking

## TL;DR: The Monitor CANNOT See Other Processes' Memory

**The memory monitor can only see allocations made within the same process.** This is a fundamental limitation of how TT-Metal's allocator tracking works.

## Why Cross-Process Tracking Doesn't Work

### The Problem

When you run:
- **Terminal 1**: `./memory_monitor`
- **Terminal 2**: `python test_script.py`

The monitor shows **0% utilization** even though Python allocated 200MB because:

### Each Process Has Its Own Allocator

```
┌─────────────────────────────────────┐
│  Process 1: memory_monitor          │
│  ┌───────────────────────────────┐  │
│  │ Device Instance A             │  │
│  │  ├─ Allocator A               │  │
│  │  │   └─ Tracks: 0 bytes       │  │  ← Sees nothing!
│  │  └─ Points to: /dev/tenstorrent/0│
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  Process 2: python script           │
│  ┌───────────────────────────────┐  │
│  │ Device Instance B             │  │
│  │  ├─ Allocator B               │  │
│  │  │   └─ Tracks: 200MB         │  │  ← Has the data!
│  │  └─ Points to: /dev/tenstorrent/0│
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘

        Both point to same hardware
                    ↓
        ┌────────────────────────┐
        │  Physical Device       │
        │  /dev/tenstorrent/0    │
        │  (200MB actually used) │
        └────────────────────────┘
```

### Why This Happens

1. **Allocator Lives in User Space**: The allocator tracking is in your process's memory, not in the kernel or on the device

2. **No Shared Allocator State**: Each `CreateDevice()` call creates a NEW allocator instance with its own tracking

3. **No OS-Level Memory View**: Unlike CPU memory (where you can use `/proc`), there's no system-wide view of device memory

4. **Host-Side Tracking Only**: The allocator doesn't query the device - it tracks allocations made through its API

## What IS Being Tracked

The allocator tracks:
- ✅ Allocations made through **the same device instance**
- ✅ Buffers created in **the same process**
- ✅ Memory allocated via **the same allocator object**

The allocator does NOT track:
- ❌ Other processes' allocations
- ❌ Kernel driver allocations
- ❌ Firmware/hardware internal memory
- ❌ Direct memory writes bypassing the allocator

## Solutions

### Solution 1: Integrated Test Mode (WORKS!)

Use `memory_monitor_test` which allocates memory in the same process:

```bash
# Build the test version
cmake --build build-cmake --target memory_monitor_test -j

# Run with automatic test allocations
./build/programming_examples/memory_monitor_test -t -r 500
```

This works because allocations happen **in the same process** as the monitoring!

### Solution 2: Python Library Integration (Future Work)

To monitor Python scripts, you'd need to:
1. Export the allocator state to shared memory
2. Have the monitor read from shared memory
3. Or: Build the monitor as a Python extension that shares the device

This would require significant engineering.

### Solution 3: Kernel-Level Tracking (Not Available)

Ideal solution: TT-KMD exposes memory utilization via sysfs:
```bash
cat /sys/class/tenstorrent/tenstorrent!0/memory_stats/dram_used
cat /sys/class/tenstorrent/tenstorrent!0/memory_stats/l1_used
```

**Status**: Not currently available in TT-KMD

## Comparison with Other Systems

### GPU Memory Monitoring

**NVIDIA**: `nvidia-smi` CAN see all processes because:
- Driver tracks allocations at kernel level
- Exposed via `/proc` or sysfs
- Works across all processes

**Tenstorrent**: No equivalent system-wide tracking (yet)

### Why GPUs Have It

```
GPU Memory Tracking:
  Application → libcuda.so → Kernel Driver → /proc/driver/nvidia
                                  ↓
                          Tracks ALL allocations
                                  ↓
                          nvidia-smi reads /proc
```

```
TT Memory Tracking:
  Application → TT-Metal → Allocator (in-process)
                              ↓
                    Only tracks THIS process
                              ↓
                    No system-wide view
```

## Workarounds for Current Limitations

### 1. Single-Process Architecture

If you need monitoring, structure your code to run everything in one process:

```python
# Instead of:
# Terminal 1: ./monitor
# Terminal 2: python script.py

# Do this:
# Combined script that monitors its own allocations
```

### 2. Instrumentation

Add logging to your application:

```python
import ttnn

def allocate_with_logging(size, name):
    buffer = ttnn.allocate(size, ...)
    print(f"Allocated {name}: {size} bytes")
    return buffer
```

### 3. Memory Reports

Use TT-Metal's built-in memory reporting:

```python
ttnn.device.EnableMemoryReports()
# Run your workload
ttnn.device.dump_device_memory_state(device, "snapshot")
# Check the CSV files generated
```

## What About Physical Hardware?

**Question**: "But the memory IS physically used on the device, can't we read that?"

**Answer**: No direct way currently. Here's why:

1. **No Hardware Memory Controller Query**: The device doesn't expose current memory usage through registers

2. **Firmware Doesn't Track**: The on-device firmware doesn't maintain allocation tables accessible to host

3. **DMA Regions**: The kernel driver manages DMA regions but doesn't track application-level allocations

4. **Security/Isolation**: By design, processes can't see each other's memory state

## Future Enhancements

To enable cross-process monitoring, TT-Metal would need:

### Option A: Kernel-Level Tracking
```c
// In TT-KMD
struct tt_memory_stats {
    uint64_t dram_allocated;
    uint64_t l1_allocated;
    // ...
};

// Exposed via:
// /sys/class/tenstorrent/tenstorrent!0/memory/dram_used
// /sys/class/tenstorrent/tenstorrent!0/memory/l1_used
```

### Option B: Shared Memory Allocator
```cpp
// Allocator lives in shared memory
// All processes accessing device 0 use same allocator instance
auto allocator = SharedAllocator::get_instance(device_id);
```

### Option C: Allocation Server
```
Process 1 ──┐
Process 2 ──┼──→ Allocation Server ──→ Tracks all allocations
Process 3 ──┘        (daemon)              Exposes via IPC
                                              ↓
                                       Memory Monitor reads IPC
```

## Conclusion

The memory monitor is **working correctly** - it accurately tracks allocations made through its own device instance. The "limitation" is actually how the architecture is designed:

- ✅ **Per-process tracking** - Fast, no overhead, no synchronization needed
- ❌ **No cross-process visibility** - Can't see other processes' allocations

For **development and debugging**:
- Use `memory_monitor_test -t` to see real-time changes
- Or instrument your application to log allocations
- Or use CSV reports with `dump_device_memory_state()`

For **production monitoring**:
- Would require kernel-level tracking (future enhancement)
- Or shared allocator architecture (significant redesign)

**The tool works perfectly for its intended use case: monitoring allocations made within the same process!**
