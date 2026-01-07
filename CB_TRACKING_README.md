# Circular Buffer (CB) Tracking Implementation

## Quick Summary

This implementation provides **accurate per-device CB tracking** for TT-Metal, correctly handling:
- ✅ Mesh device setups (each device reports its own CB usage)
- ✅ Program caching/tracing (handles address reuse)
- ✅ Physical L1 constraints (stays within ~90 MiB limit)

## Files Modified

**Core tracking implementation:**
- `tt_metal/impl/device/device.cpp` - Physical CB calculation with overlap merging
- `tt_metal/impl/program/program.cpp` - Program registration and CB lifecycle
- `tt_metal/impl/program/program_impl.hpp` - CB region query methods
- `tt_metal/api/tt-metalium/program.hpp` - Public API
- `tt_metal/impl/device/update_allocator_stats.cpp` - SHM stats updates

## How It Works

### 1. CB Allocation
When a program allocates CBs:
```cpp
program.allocate_circular_buffers(device);
```
- CBs reserve L1 address ranges on specified cores
- Program registers with each device (only once per device!)
- GraphTracker reports allocation
- SHM stats updated

### 2. CB Usage Query
```cpp
uint64_t cb_usage = device->get_total_cb_allocated();
```
- Collects L1 regions from all active programs
- Groups by core coordinate
- Merges overlapping addresses (handles cached programs sharing addresses)
- Returns physical L1 usage

### 3. CB Deallocation
When program is destroyed:
```cpp
~ProgramImpl()
```
- Unregisters from all devices
- GraphTracker reports deallocation
- SHM stats updated
- CB memory becomes available

## Expected Behavior

### Single Device
```
$ tt-smi
Device 0:
  CB: 29.0 MiB  ✅
  L1: 45.3 MiB
```

### Mesh Device (8 devices)
```
$ tt-smi
Device 0: CB: 29.0 MiB  ✅
Device 1: CB: 29.0 MiB  ✅
...
Device 7: CB: 29.0 MiB  ✅
```
Each device shows its own CB usage, not aggregated values.

## Important Notes

### CB Memory Lifecycle
- **CB memory stays allocated while programs exist**
- With program caching (common in production), CBs persist between inferences
- This is **expected and efficient** (avoids reallocation overhead)
- CB memory freed when programs are destroyed

### "Total L1" vs "CB"
In tt-smi:
```
Total L1 = L1 buffers (regular allocations)
         + L1_SMALL
         + CB (Circular Buffers)
         + (Kernels in reserved region, not tracked)
```

If you see temporary "Total L1" spikes > 90 MiB:
- This is from `L1 buffers`, NOT CBs
- Occurs during program compilation/loading
- **Normal** if temporary (< 1 second) and returns to baseline
- CB component should remain stable

### Physical vs Logical
- **Physical CB**: Actual L1 memory used (what we track)
- **Logical CB**: Sum across all programs (can exceed physical due to address reuse)

With 30 cached programs each using 10 MiB of CBs at the same addresses:
- Logical: 30 × 10 MiB = 300 MiB
- Physical: 10 MiB (addresses shared) ✅

## Documentation

See `CB_LIFECYCLE_RESEARCH.md` for complete lifecycle documentation.

## Build

```bash
cd /home/ttuser/aperezvicente/tt-metal-apv
./build_metal_with_flags.sh
```

## Testing

Run your workload and monitor:
```bash
# Terminal 1: Run workload
pytest your_test.py

# Terminal 2: Monitor CB usage
watch -n 0.5 'tt-smi | grep -E "Device|CB:"'
```

Expected:
- CB stable during inference (~29 MiB per device)
- No "already registered" warnings
- CB drops to 0 when programs destroyed
