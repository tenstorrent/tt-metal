# Kernel Size Tracking Fix - The Root Cause

## Problem Discovered

After implementing kernel tracking, we saw:
- ✅ `CB_ALLOC` messages working correctly (~22.47 MB per device)
- ✅ `KERNEL_LOAD` messages appearing
- ❌ **But all kernel sizes were 0 MB!**

## Root Cause: Wrong Size Variable

We were tracking `kernel_bins_sizeB` which represents kernel text stored in the **config buffer**. However:

### Modern Dispatch (Blackhole, Fast Dispatch):
- **Kernels are NOT stored in the config buffer**
- Instead, they're loaded to a **ring buffer in L1** dynamically
- `get_core_kernel_stored_in_config_buffer()` returns `false` (line 332 in `dispatch.cpp`)
- This causes `kernel_text_size = max_offset - base_offset = 0` (line 354)

### The Actual Kernel Size:
The REAL kernel binary size is stored in:
```cpp
program_transfer_info.binary_data.size() * sizeof(uint32_t)
```

This is the size of the **DRAM buffer** that holds kernel binaries before they're loaded to L1 (see line 1397 in `program.cpp`).

## Solution Implemented

Changed BOTH tracking locations to use the correct size:

### 1. Fast Dispatch Path (`finalize_program_offsets` - line 1897)
```cpp
// OLD (WRONG):
if (program->kernel_bins_sizeB > 0) {
    tt::tt_metal::GraphTracker::instance().track_kernel_load(
        program->kernel_bins_sizeB,  // ❌ This is 0 for modern dispatch!
        kernel_id,
        dev);
}

// NEW (CORRECT):
uint64_t kernel_size = program->program_transfer_info.binary_data.size() * sizeof(uint32_t);
if (kernel_size > 0) {
    tt::tt_metal::GraphTracker::instance().track_kernel_load(
        kernel_size,  // ✅ Actual binary size!
        kernel_id,
        dev);
}
```

### 2. Slow Dispatch Path (`finalize_offsets` - line 1775)
Same fix applied to the instance method path.

## Why This Matters

### Config Buffer Size (`kernel_bins_sizeB`):
- Represents kernel text embedded in dispatch commands
- **0 for modern chips** (Blackhole) with ring buffer dispatch
- Only non-zero for older chips or specific dispatch modes

### Binary Data Size (`program_transfer_info.binary_data.size()`):
- Represents actual compiled kernel code
- Stored in DRAM, loaded to L1 ring buffer as needed
- **This is what actually consumes L1 memory!**

## Expected Results After Fix

After rebuilding with this fix:
```
✓ [KERNEL_LOAD] Device 0: +0.125 MB (Total: 0.125 MB)
✓ [KERNEL_LOAD] Device 1: +0.125 MB (Total: 0.125 MB)
```

Instead of:
```
✓ [KERNEL_LOAD] Device 0: +0 MB (Total: 0 MB)  ❌
✓ [KERNEL_LOAD] Device 1: +0 MB (Total: 0 MB)  ❌
```

## Files Modified

1. **`tt_metal/impl/program/program.cpp`**
   - Line 1775: Fixed Slow Dispatch kernel size (instance `finalize_offsets`)
   - Line 1897: Fixed Fast Dispatch kernel size (static `finalize_program_offsets`)

## Build & Test

```bash
cmake --build build --target tt_metal -j$(nproc)
export TT_ALLOC_TRACKING_ENABLED=1
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"
grep "KERNEL_LOAD" out.log | head -10
```

You should now see non-zero kernel sizes!

## Technical Notes

### Kernel Memory Model
- Kernels are compiled to binaries and stored in DRAM buffers
- At dispatch time, kernels are loaded to L1 **ring buffer** (circular queue)
- Ring buffer size is typically 128KB-256KB depending on chip
- Multiple programs share the ring buffer (kernels evicted/reloaded as needed)
- `program_transfer_info.binary_data` size represents the DRAM storage, which equals the L1 space used when loaded

### Why We Track Binary Size
Even though kernels use a ring buffer (not permanently in L1), tracking the binary size tells us:
1. **Maximum L1 footprint** when this program's kernels are loaded
2. **DRAM usage** for storing the binaries
3. **Relative complexity** of programs (larger binaries = more complex kernels)

This is useful for understanding memory pressure and program characteristics!
