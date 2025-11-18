# MemoryReporter Integration - Reverted ⚠️

## What Happened

We attempted to integrate `MemoryReporter` into `tt_smi_umd` to show full L1 memory usage (including circular buffers and kernel code), but **had to revert it** due to device conflicts.

## The Problem

### Device Ownership Conflict

```cpp
// Your application:
Device* app_device = CreateDevice(0);  // Owns the device

// tt_smi_umd tries to query MemoryReporter:
Device* monitor_device = CreateDeviceMinimal(0);  // ❌ CONFLICT!
// Either fails or causes instability in the running app
```

**TT-Metal only allows one process to own a device at a time.**

## Why This Happened

### Different from nvidia-smi

**NVIDIA's approach:**
- All allocations tracked in **kernel driver** (shared across processes)
- `nvidia-smi` queries the kernel driver (no device "ownership" needed)
- No conflicts!

**Tenstorrent's current architecture:**
- Allocations tracked in **user-space allocator** (per-process)
- To query allocator, you need to create a `Device` instance
- Creating `Device` requires exclusive access → conflict!

## What Was Changed (and Reverted)

### Files Modified Then Reverted

1. **tt_smi_umd.cpp**
   - Added includes for `memory_reporter.hpp` and `buffer_types.hpp` → **REVERTED**
   - Added fields to `DeviceInfo` struct for MemoryReporter data → **REVERTED**
   - Added `query_memory_reporter_l1()` method → **REVERTED**
   - Modified memory breakdown display to show inferred CB+Kernel usage → **REVERTED**

All changes have been removed, `tt_smi_umd` is back to its original state.

## The Right Solution: Allocation Server

The allocation server **IS** the correct approach:

```
Application Process → Allocate Buffer → Send IPC message → Allocation Server
                                                               ↓
tt_smi_umd → Query via IPC ← Allocation Server aggregates all processes
```

**Benefits:**
- ✅ No device ownership conflicts
- ✅ Cross-process tracking
- ✅ Real-time updates
- ✅ Works with multiple concurrent processes

## What About CB + Kernel Memory?

**Current state:**
- Allocation server tracks: DRAM, L1, L1_SMALL, TRACE (explicit allocations)
- NOT tracked: Circular buffers, kernel code

**To track CBs and kernels:**

See `FULL_L1_TRACKING_GUIDE.md` for the complete solution. Summary:

1. **Hook into Program::add_kernel()** - track kernel code size
2. **Hook into CreateCircularBuffer()** - track CB allocations
3. **Report to allocation server** via IPC
4. **Display in tt_smi_umd** - now you see everything!

**Effort:** ~500-1000 lines of code, requires modifying TT-Metal core files.

## Current Capabilities

**What tt_smi_umd shows (with allocation server running):**

```
Memory Breakdown:

Device 0 (Wormhole_B0):
----------------------------------------------------------------------
  DRAM:     2.5GB    / 24.0GB    [███░░░░░░░░░░░░░░░░░░░░░]
  L1:       1.5MB    / 306.0MB   [░░░░░░░░░░░░░░░░░░░░░]
  L1_SMALL: 512KB
  TRACE:    2.3MB
```

**What's NOT shown (but exists in hardware):**
- ~90-100 MB of circular buffers (pre-allocated per-core)
- ~10-30 MB of kernel code
- ~5 MB of firmware overhead

**Reality:** Your 306MB L1 is probably 95% used, you just can't see the CB/kernel part yet!

## Recommended Workflow

### For everyday monitoring:
```bash
# Terminal 1: Start allocation server
./build/programming_examples/allocation_server_poc

# Terminal 2: Monitor with tt_smi_umd
./build/programming_examples/tt_smi_umd -w
```

### For detailed L1 debugging:
```python
# In your application code
import tt_lib as ttl

device = ttl.device.GetDefaultDevice()

# Dumps detailed memory breakdown to CSV
ttl.device.DumpDeviceMemoryState(device)

# Check .reports/tt_metal/ directory
```

### For production (future):
Implement CB/Kernel tracking hooks so allocation server tracks everything!

## Documentation

- **`WHY_NO_MEMORY_REPORTER.md`** - Detailed explanation of why MemoryReporter doesn't work for monitoring
- **`FULL_L1_TRACKING_GUIDE.md`** - How to implement complete L1 tracking (including CBs)
- **`MEMORY_REPORTER_VS_ALLOCATION_SERVER.md`** - Comparison of both approaches
- **`L1_MEMORY_NOT_TRACKED.md`** - What L1 memory the allocator doesn't track

## Summary

**Attempted:** Integrate MemoryReporter to show full L1 usage
**Result:** Device conflicts with running applications
**Solution:** Reverted changes, stick with allocation server
**Future:** Implement CB/Kernel tracking hooks in TT-Metal core

**Current status:**
- ✅ `tt_smi_umd` works correctly (no conflicts)
- ✅ Cross-process memory tracking (via allocation server)
- ❌ CB + Kernel memory not visible (known limitation)

**The allocation server is the right architecture!** It's exactly what we need for `nvidia-smi` style monitoring. The only missing piece is hooking CB/Kernel creation to report to the server.
