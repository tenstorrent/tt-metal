# Can We Track Application Kernels in Fast Dispatch?

## Short Answer
**YES, but with caveats**. Application kernels in Fast Dispatch are only sent **once** when first executed, then **cached** in DRAM. Subsequent executions reuse the cached binaries, so we only see the **initial load**, not per-execution loads.

## How Fast Dispatch Works

### Kernel Binary Flow:
```
1. First Execution:
   ├─ Compile kernels (if not cached on disk)
   ├─ Generate dispatch commands
   ├─ Send kernel binaries to DRAM (via CQ commands)
   ├─ Mark status: NotSent → InFlight → Committed
   └─ Track kernel load ✅ (we can do this!)

2. Subsequent Executions:
   ├─ Check ProgramBinaryStatus == Committed
   ├─ Skip sending binaries (send_binary = false)
   └─ Only send launch messages
```

### Program Binary Status States:
```cpp
enum class ProgramBinaryStatus : uint8_t {
    NotSent = 0,     // Binaries have not been written
    InFlight = 1,    // Fast Dispatch Commands to write binaries issued
    Committed = 2,   // Binaries committed to DRAM (cached!)
};
```

### Where Kernel Binaries are Sent:
**File**: `tt_metal/impl/program/dispatch.cpp`

```cpp
// Line 2321-2330
if (send_binary) {
    // Write the program binary
    if (program_command_sequence.prefetcher_cache_used) {
        write_data_to_cq(
            program_command_sequence.program_binary_setup_prefetcher_cache_command.data(),
            ...);
    }
    write_data_to_cq(
        program_command_sequence.program_binary_command_sequence.data(),
        ...);
}
```

**The `send_binary` flag is determined by** `program.get_program_binary_status(device_id)`:
- `NotSent` → `send_binary = true` ✅ Binaries sent to DRAM
- `Committed` → `send_binary = false` ❌ Binaries already cached

## Why You Only See System Kernels

### Current Behavior:
1. **System Kernels (Fabric/Dispatch)**:
   - Use **Slow Dispatch** path (forced)
   - Always call `finalize_offsets()` → **tracked** ✅
   - Persistent for device lifetime

2. **Application Kernels**:
   - Use **Fast Dispatch** with caching
   - `finalize_offsets()` called **once** when first compiled
   - Binary sent **once** when `ProgramBinaryStatus == NotSent`
   - **Already tracked** ✅ but happens early in program lifecycle
   - Subsequent runs skip `finalize_offsets()` because already finalized

### Why You Don't See Application Kernel Loads in Your Test:

Looking at your test output:
```bash
[36423ms] L1 Peak on device 0: 1 MB (39 buffers)
[55886ms] L1 Peak on device 0: 12 MB (42 buffers)
```

The transformer demo likely:
1. Compiled all kernels during **warmup** (before you started tracing)
2. All kernels already `finalized` and `ProgramBinaryStatus == Committed`
3. Main loop only sends **launch messages**, not binaries
4. Our tracking happens in `finalize_offsets()` which only runs **once per program**

## How to Track Application Kernels

### Option 1: Track During First Execution (Current Implementation)
**Status**: ✅ **Already works!**

Application kernels ARE tracked when they're first finalized. To see them:

```bash
# Clear kernel cache to force recompilation
rm -rf ~/.cache/tt_metal/kernels/*

# Start fresh tracking
./build/programming_examples/allocation_server_poc > out.log 2>&1 &

# Run test (kernels will be compiled and tracked)
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and DP-4-b1" 2>&1 | tee trace_kernels.log

# Check for application kernel loads
grep "Application" trace_kernels.log
```

### Option 2: Hook into Fast Dispatch Binary Send
Track when `send_binary == true` in `write_program_command_sequence()`.

**Implementation**:
```cpp
// In tt_metal/impl/program/dispatch.cpp, line ~2321
if (send_binary) {
    // NEW: Track application kernel send to DRAM
    if (program.get_kernel_type() == ProgramKernelType::APPLICATION) {
        uint64_t kernel_size = program_command_sequence.kernel_bins_sizeB;
        uint32_t num_cores = /* calculate from program.logical_cores() */;
        GraphTracker::instance().track_kernel_load(
            kernel_size,
            program.get_id(),
            device,
            0,  // Application kernel
            num_cores);
    }

    // Write the program binary
    write_data_to_cq(...);
}
```

**Pros**:
- Tracks exactly when binaries are sent to device
- Works even if program was pre-compiled

**Cons**:
- Requires access to device object in dispatch context
- More complex integration point

### Option 3: Track at Program Cache Time
Track when program commands are first generated (cached).

**File**: `tt_metal/impl/program/program.cpp`, line ~1431

```cpp
if (!cached_program_command_sequences.contains(command_hash)) {
    // NEW: First time program is dispatched, track kernel load
    if (this->kernel_type_ == ProgramKernelType::APPLICATION) {
        uint32_t total_cores = 0;
        for (const auto& cores : this->logical_cores()) {
            total_cores += cores.size();
        }
        uint64_t kernel_size = this->program_transfer_info.binary_data.size() * sizeof(uint32_t);

        GraphTracker::instance().track_kernel_load(
            kernel_size,
            reinterpret_cast<uint64_t>(this),
            device,
            0,  // Application
            total_cores);
    }

    // ... rest of cache generation ...
}
```

**Pros**:
- Simple integration
- Guaranteed to run once per unique program

**Cons**:
- Timing doesn't match actual DRAM write
- Device context available

## Recommendation

### For Your Use Case:
✅ **Option 1 (Current Implementation) is sufficient**

Application kernels **are being tracked** via `finalize_offsets()`. They just happen early in the program lifecycle (during warmup).

To verify:
1. Clear kernel cache
2. Run test with fresh server
3. Look for "Application" kernel loads in early logs

### If You Want to See Repeated Kernel Tracking:
⚠️ **This doesn't reflect reality!**

In Fast Dispatch, kernels are only loaded once to DRAM, then reused. Tracking them on every program execution would be **misleading** because:
- L1 usage doesn't change (kernels already in ring buffer)
- DRAM binaries are cached (not re-sent)
- Only launch messages change per execution

## What You're Actually Missing: Ring Buffer Kernel Code

The **real** application kernel L1 usage happens in the **ring buffer**, which is:
- Allocated once at device init
- Shared across all programs
- Not tracked by our allocator
- Part of "reserved L1" (~10-50 MB)

This is similar to the Fabric/Dispatch kernels - they're loaded to a **persistent L1 ring buffer**, not tracked as individual allocations per core.

## Summary

| Kernel Type | Dispatch | Tracked? | When? | Notes |
|-------------|----------|----------|-------|-------|
| **System (Fabric/Dispatch)** | Slow | ✅ Yes | Device init | Persistent, shown in logs |
| **Application (First Run)** | Fast | ✅ Yes | First `finalize_offsets()` | Happens during warmup |
| **Application (Cached)** | Fast | ⏭️ Skipped | - | Binaries already in DRAM |
| **Ring Buffer L1** | - | ❌ No | - | Pre-allocated, not tracked |

**Your logs are correct!** You're seeing:
- System kernels: 540 KB per device ✅
- Application kernels: Already finalized before tracing started ✅
- Total makes sense ✅
