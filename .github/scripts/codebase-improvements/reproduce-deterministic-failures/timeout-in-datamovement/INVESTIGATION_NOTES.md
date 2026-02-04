# TT-Metal Project Memory

## Timeout Debugging Pattern
When debugging device timeouts with ttnn operations:
1. **Profile first** - use `time.perf_counter()` to measure each phase (to_device, operation, sync, to_torch)
2. **Isolate** - use `ttnn.synchronize_device()` to separate kernel execution from data transfer
3. **Scale test** - run with different tensor sizes to see scaling behavior

## Gather Operation Performance
- Sequential barriers in writer kernel (one per tile) enable **pipelining** with reader
- DON'T batch all reads before barrier - this breaks pipelining and makes things worse
- The real bottleneck is reader's element-by-element processing (~5M scalar ops for 151936 width)
- **Compute kernels CANNOT help** - see investigation below

## Compute Kernels Cannot Optimize Gather
Investigated compute kernel patterns (2025 investigation). Key findings:

**Why Compute Kernels Don't Work for Gather:**
1. **Tile-based architecture**: All SFPU operations work on entire 32x32 tiles via DST register
2. **No per-element index lookup**: No API for `output[i] = input[indices[i]]` at element level
3. **SFPLOAD/SFPSTORE granularity**: These load/store 4 rows (128 elements) at once, not individual elements
4. **reshuffle_rows is scatter-add, not gather**: Does `output[mask[i]] += input[i]` at row (32-element) level

**Operations Examined:**
- `where_tile`: Conditional selection between two tiles using mask - not index lookup
- `reshuffle_rows`: Row-level scatter-add for embedding backward - inverse of gather
- `topk_local_sort`: Bitonic sort with index tracking - sorts, doesn't do arbitrary lookup
- Embedding ops: Also use dataflow kernels for per-element lookups

**Conclusion**: Gather is fundamentally a dataflow operation. The only path forward:
1. Loop optimizations in reader kernel (unrolling, prefetch)
2. Better multi-core distribution
3. Wait for hardware with random-access support

## Batched Tile Loading (DOESN'T WORK)
Attempted optimization: wait for multiple tiles at once instead of one at a time
- Changed CB size from 2 to 32 tiles
- Reader: `cb_wait_front(cb, 32)` then `cb_pop_front(cb, 32)`
- Writer: still pushes tiles one at a time via `cb_push_back(cb, 1)`
- **Result: DEADLOCK** - device hangs even for small Wt_input (62 tiles)
- Root cause: CB synchronization breaks when reader waits for N tiles but writer pushes one at a time
- The CB API may not be designed for this wait-many/pop-many pattern with push-one writer
- Multi-core kernel (Wt > 60) uses different sync pattern than single-core (Wt <= 60)

## Key ttnn Insights
- `ttnn.to_torch()` waits for kernel completion - what looks like slow data transfer may be kernel execution
- Device timeout (TT_METAL_OPERATION_TIMEOUT_SECONDS) applies to kernel execution, not test duration
- Simple roundtrip (no ops) achieves ~26 MB/s, so raw transfer is fast

## Unexplored: NOC Multicast/Streaming Approach
The multi-core kernel has each of ~56 cores **independently read all input tiles from DRAM** - that's 56× redundant reads. A potential optimization:
- Stream input tiles once via NOC multicast/broadcast
- All cores receive the same tile and process their index tiles against it
- Same O(Wt × n) math but 56× less DRAM bandwidth

**Requires research:**
- Does TT-Metal NOC support efficient multicast?
- How to synchronize cores on "current input tile"?
- Risk: may hit similar coordination issues as batched CB approach

## AI-Assisted Optimization Lessons
1. **Validate architecture before coding** - Check if approach is fundamentally sound for the hardware
2. **Search for prior art** - If no one's solved it in the codebase, ask why
3. **Test CB/API assumptions with minimal examples** - Don't build complex changes on untested foundations
4. **"Hardware limitation" is a valid answer** - Not everything can be optimized in software
5. **Understand the memory hierarchy** - DRAM bandwidth, L1 size, NOC capabilities matter more than loop tricks

## File Locations
- Gather kernels: `ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/`
- System memory manager: `tt_metal/impl/device/system_memory_manager.cpp` (timeout errors)
- Program factory (kernel selection): `ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp` (GATHER_WT_THRESHOLD=60)
