# Circular Buffer (CB) Tracking - Final Implementation Summary

## What Was Fixed

### Issue
CB tracking showed incorrect values in mesh device setups:
- All devices showed the same aggregated value
- Values could spike to 100-250 MiB (exceeding physical L1 limits)
- Duplicate program registration causing overcounting

### Solution Implemented

**1. Physical CB Tracking with Address Overlap Merging**
- Track L1 regions per core for each program
- Merge overlapping addresses (handles cached/traced programs sharing addresses)
- Calculate actual physical L1 usage per device

**2. Per-Device Program Registration**
- Programs register only once per device
- Track `new_devices` vs already-registered devices
- Prevents duplicate registration

**3. Accurate Deallocation**
- Programs unregister when destroyed
- CB memory freed when programs are deallocated
- SHM stats updated automatically

## Files Modified

### Core Implementation
1. **`tt_metal/impl/device/device.cpp`**
   - `Device::get_total_cb_allocated()` - Physical CB tracking with overlap merging
   - `Device::register_program()` / `unregister_program()` - Program lifecycle tracking

2. **`tt_metal/impl/program/program.cpp`**
   - `ProgramImpl::allocate_circular_buffers()` - Register only with NEW devices
   - `ProgramImpl::deallocate_circular_buffers()` - Unregister on cleanup
   - `ProgramImpl::get_cb_l1_regions_per_core()` - Return per-core L1 regions

3. **`tt_metal/impl/program/program_impl.hpp`**
   - Added `get_cb_l1_regions_per_core()` method
   - Added `get_num_cb_devices()` method

4. **`tt_metal/api/tt-metalium/program.hpp`**
   - Public API for CB region queries

5. **`tt_metal/impl/device/update_allocator_stats.cpp`**
   - Query-based SHM updates (accurate with program caching)

## Expected Behavior

### During Inference
- **CB usage**: ~29 MiB per device (stable)
- **No duplicate registration warnings**
- **CB drops to 0 when programs destroyed**

### Mesh Device (8 devices)
```
Device 0: CB = 29.0 MiB  ✅
Device 1: CB = 29.0 MiB  ✅
...
Device 7: CB = 29.0 MiB  ✅
```

Each device reports its own CB usage correctly.

## About "Total L1" Spikes

**If you see temporary Total L1 spikes to 100-130 MiB:**
- This is from `l1_allocated` (regular L1 buffers), NOT CBs
- Occurs during program compilation/loading
- Temporary accounting artifact (allocation tracked before deallocation)
- **Normal behavior** as long as:
  - Spikes are temporary (< 1 second)
  - Returns to < 90 MiB baseline
  - No OOM errors occur
  - CB component remains stable ~29 MiB

## Key Insights

1. **CB memory stays allocated for program lifetime**
   - With program caching/tracing, CBs persist even after inference
   - This is expected and efficient (no reallocation overhead)

2. **Physical vs Logical CB tracking**
   - Multiple cached programs can share the same L1 addresses
   - Physical tracking merges overlaps → accurate per-device values
   - Logical sum would exceed physical L1 limits

3. **Kernel binaries not tracked in "Total L1"**
   - Kernels live in reserved L1 region
   - Not included in tracked L1 allocations

## Documentation

See `CB_LIFECYCLE_RESEARCH.md` for complete details on CB allocation/deallocation lifecycle.
