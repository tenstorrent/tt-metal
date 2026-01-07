# Memory Tracking Fix - Final Summary

## Problems Fixed

### 1. ✅ Total L1 Capacity (Harvesting Bug)
**Problem**: tt-smi showed 171.6 MiB total (120 cores) but device has only 64 active cores
**Fix**: Use `soc_desc.get_grid_size(CoreType::TENSIX)` instead of `soc_desc.grid_size`
**Result**: Now shows 91.5 MiB total (64 cores × 1.43 MiB)

### 2. ✅ Kernel Tracking (Removed)
**Problem**: Kernel memory kept growing beyond capacity (cumulative tracking broken)
**Reason**:
- Kernels live in fixed KERNEL_CONFIG region (not dynamic)
- Program caching prevents deallocations from being reported
**Fix**: Disabled kernel tracking in tt-smi
**Result**: Kernel column removed, no more impossible values

### 3. ⚠️  CB Tracking (Approximate)
**Problem**: CBs showed 0B after disabling cumulative tracking
**Discovery**: CBs have separate `CircularBufferAllocator`, NOT part of L1 Buffer allocator
**Current Fix**: Re-enabled cumulative CB tracking (approximate but shows usage)
**Future Fix**: Query `CircularBufferAllocator.l1_regions` from all cached programs for accuracy

## Current tt-smi Output

```
Device 0: 1.3 MiB / 91.5 MiB  ✅
  - Total capacity: CORRECT (accounts for harvesting)
  - L1 + L1_SMALL: 1.3 MiB (from allocator, accurate)
  - CB: 0B → needs rebuild to show cumulative tracking
  - Kernel: removed (was misleading)
```

## Files Modified

1. **tt_metal/programming_examples/.../tt_smi.cpp**
   - Fixed: Use `get_grid_size(TENSIX)` for harvesting
   - Removed: Kernel column from display
   - Updated: L1 calculation (removed kernel tracking)

2. **tt_metal/graph/graph_tracking.cpp**
   - Disabled: Kernel allocation/deallocation tracking
   - Re-enabled: CB allocation/deallocation tracking
   - Added: Comments explaining why

3. **tt_metal/impl/profiler/memory_stats_shm.hpp/cpp**
   - Added: `update_from_allocator()` method declaration
   - Note: Implementation in update_allocator_stats.cpp (circular dependency)

4. **tt_metal/impl/device/update_allocator_stats.cpp** (NEW)
   - Implemented: `update_from_allocator()` - queries L1/DRAM allocators
   - Implemented: `UpdateAllocatorStatsToShm()` - updates all devices

5. **tt_metal/api/tt-metalium/update_allocator_stats.hpp** (NEW)
   - Public API for updating allocator stats to SHM

## What Works Now

✅ **Total L1 capacity accurate** (accounts for harvesting)
✅ **L1/L1_SMALL/DRAM tracked via allocator** (accurate, real-time)
✅ **Kernel tracking removed** (was misleading due to fixed region)
⚠️  **CB tracking approximate** (cumulative, may exceed capacity with many cached programs)

## Next Steps for Accurate CB Tracking

See `ACCURATE_CB_TRACKING_SOLUTION.md` for implementation:

1. Add `ProgramImpl::get_total_cb_allocated_bytes()`
2. Track active programs per-device
3. Query all programs' CB usage
4. Update SHM with actual values

This will give **real-time accurate CB tracking** that works with program caching!

## UMD Tool Discovery

Found `tt_umd` Python API and CLI tools:
- `read_soc_info.py` - reads SOC descriptor, shows harvesting
- `topology` tool - shows chip topology
- `telemetry` tool - reads device telemetry

## Key Learnings

1. **Harvesting matters**: Device 0 has 64 cores, not 120 (16 harvested)
2. **Program caching breaks cumulative tracking**: Destructors don't run
3. **CBs have separate allocator**: Not part of L1 Buffer allocator
4. **Kernel region is fixed**: KERNEL_CONFIG size doesn't grow
5. **Query allocators for ground truth**: Don't rely on cumulative add/subtract
