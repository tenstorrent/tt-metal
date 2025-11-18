# Firmware (FW) Tracking Status

## What We're Currently Tracking ✅

### 1. **Dispatch/Fabric Kernels (L1 Memory)**
- **Size**: ~204 KB per device (4 kernels)
- **Type**: System kernels loaded as "programs"
- **Where**: Loaded via `Device::configure_fabric()` and `Device::configure_command_queue_programs()`
- **Tracking**: ✅ **FULLY TRACKED**
  - Allocation: `KERNEL_LOAD` messages
  - Deallocation: `KERNEL_UNLOAD` messages
  - PID tracking: ✅ **FIX APPLIED** (collects PIDs from `kernel_allocations_`)
  - Cleanup: ✅ Automatic when process dies

**Components:**
- 2× Fabric routing kernels (56 KB each)
- 2× Command queue/dispatch kernels (46 KB each)

### 2. **Application Kernels (L1 Memory)**
- **Type**: User program kernels
- **Tracking**: ✅ **FULLY TRACKED**
  - Same mechanism as system kernels
  - Properly deallocated when program ends

### 3. **Circular Buffers (L1 Memory)**
- **Type**: Data streaming buffers
- **Tracking**: ✅ **FULLY TRACKED**
  - `CB_ALLOC` / `CB_FREE` messages
  - PID tracking with automatic cleanup

### 4. **Regular Buffers (DRAM & L1)**
- **Type**: Data tensors, activations
- **Tracking**: ✅ **FULLY TRACKED**
  - `ALLOC` / `FREE` messages
  - PID tracking with automatic cleanup

## What We're NOT Tracking ❌

### 1. **Base RISC-V Firmware Binaries**

These are the low-level firmware loaded to RISC-V cores during device initialization:

| Firmware Type | Location | Size | Lifetime |
|--------------|----------|------|----------|
| **BRISC** (Base RISC) | L1 memory | ~10-20 KB | Permanent (until device close) |
| **NCRISC** (NoC RISC) | L1 memory | ~10-20 KB | Permanent |
| **TRISC** (Tensix RISC) | L1 memory | ~10-20 KB | Permanent |
| **ERISC** (Ethernet RISC) | L1 memory | ~10-20 KB | Permanent |
| **Lite Fabric FW** | Eth cores | Variable | Permanent |

**Characteristics:**
- Loaded directly via `write_to_device()` / `cluster.write_core()`
- Happens during `Device::initialize()` / firmware initialization
- **NOT** tracked by allocator (direct memory writes)
- **NOT** tracked by our kernel tracking (not "programs")
- Permanent until device reset

**Why Not Tracked:**
1. Loaded before any tracking starts
2. Use direct memory writes, not allocator
3. Fixed size, never deallocated
4. Part of device initialization, not application memory

**Estimated Total:** ~100-200 KB per device (across all RISC cores)

### 2. **Device Mailboxes & Configuration**
- Mailbox regions for core communication
- Configuration data structures
- Control registers

**Estimated:** ~10-50 KB per device

## Total L1 Baseline Memory Usage

For a typical device:

| Component | Tracked? | Size per Device | Notes |
|-----------|----------|-----------------|-------|
| **Dispatch/Fabric Kernels** | ✅ Yes | ~204 KB | System infrastructure |
| **Base RISC FW** | ❌ No | ~100-200 KB | Fixed firmware binaries |
| **Mailboxes/Config** | ❌ No | ~10-50 KB | Device control structures |
| **Total Baseline** | Partial | **~314-454 KB** | Fixed until device close |

**Application Memory** (varies with workload):
- User buffers: 0 - several MB
- User CBs: 0 - several MB
- User kernels: 0 - several MB
- **All tracked** ✅

## Do We Need to Track Base Firmware?

### Arguments AGAINST tracking base FW:

1. **Fixed and Permanent**
   - Loaded once at device init
   - Never deallocated until device close
   - Size doesn't change

2. **Not Application Memory**
   - Part of device infrastructure
   - Users can't control or affect it
   - Always present, like device "ROM"

3. **No Cleanup Needed**
   - Automatically freed on device close
   - No risk of leaks (fixed allocation)
   - No per-process tracking needed

4. **Complexity**
   - Would require hooking low-level `write_core()` calls
   - Hard to distinguish from other memory writes
   - Minimal benefit for the effort

### Arguments FOR tracking base FW:

1. **Complete Memory Picture**
   - Shows true L1 baseline
   - Helps users understand "why is 500 KB already used?"
   - More accurate memory reporting

2. **Debugging**
   - Verify FW loaded correctly
   - Track FW version changes
   - Identify FW-related issues

### Recommendation

**DON'T track base FW individually**, but:

1. ✅ **Document the baseline** - users should know ~300-400 KB is used by system
2. ✅ **Track what matters** - application memory (buffers, CBs, kernels) ← We do this!
3. ✅ **Report baseline** - show "System: ~350 KB, Application: X MB" in `tt_smi_umd`

## Implementation: Show System Baseline in tt_smi_umd

Instead of tracking individual FW components, we can report:

```
Device 0 (Blackhole):
----------------------------------------------------------------------
  L1 Memory:
    System Baseline:  ~350 KB   [Fixed infrastructure]
      - Dispatch/Fabric: 204 KB  [Tracked]
      - RISC FW:        ~150 KB  [Estimated]

    Application Memory:
      Buffers:          5.2 MB    [██░░░░░░] 1.7%
      CBs:              2.1 MB    [█░░░░░░░] 0.7%
      Kernels:          1.8 MB    [█░░░░░░░] 0.6%

    Total L1:           9.45 MB  / 306 MB [███░░░░░] 3.1%
```

This gives users the complete picture without complex low-level tracking!

## Current Status Summary

**What We Track:** ✅
- ✅ Dispatch/Fabric kernels (204 KB system baseline)
- ✅ Application kernels
- ✅ Circular buffers
- ✅ Regular buffers (DRAM, L1, L1_SMALL, TRACE)
- ✅ PID tracking with automatic cleanup

**What We Don't Track:** ❌
- ❌ Base RISC-V firmware (~150 KB, fixed)
- ❌ Mailboxes/config structures (~50 KB, fixed)

**Recommendation:**
- Keep current tracking (it's comprehensive for application memory)
- Add "System Baseline" display in `tt_smi_umd` to show estimated fixed overhead
- Document that ~300-400 KB is expected baseline per device

**This gives 95%+ visibility with 20% of the complexity!** 🎯
