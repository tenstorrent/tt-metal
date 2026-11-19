> **Historical — superseded allocation policy.** This diagnosis and its archived
> evidence apply to the recorded implementation before the PLE private-buffer
> repair. That repair removes the padded reduction clone and private gate
> temporaries used in the memory argument below. The old batch5 failure does
> **not** establish the current runtime's batch limit. Fresh batch5 testing and
> the next observed failure must re-establish capacity; that evidence is pending.
> The original diagnosis is retained below without reinterpretation.

# PLE batch5 trace-capacity diagnosis

Source-only analysis of `ple_batch5_trace_memory.json` and its matching log.
No additional device run was performed for this note. The experiment used the
full resident packed PLE table, BFP8 expert storage, in-place MoE score weighting,
and the default TTNN sum path. This establishes a bound for this implementation
and captured command stream, not a model-architecture or optimized-runtime limit.

## Exact measured allocation

| Quantity | Bytes |
|---|---:|
| Available DRAM across 8 banks | 34,178,731,008 |
| Resident model before request-state allocation | 33,659,191,296 |
| Allocated immediately before capture | 33,707,653,632 |
| Allocated immediately before end-capture | 33,710,930,432 |
| Allocated after end-capture rejected overlap, before release | 34,069,805,568 |
| Exact new trace reservation | 358,875,136 |
| Free immediately before end-capture | 467,800,576 |
| Free after rejected end-capture | 108,925,440 |

The new block is 44,859,392 bytes per bank, 342.25 MiB aggregate. Both the
block-table difference and allocated/free-byte differences give 358,875,136 bytes.
This is the actual allocated trace reservation; unpadded command payload bytes
are not exposed by these snapshots and are not inferred.

`block_table` addresses are allocator-relative. The new block begins at
4,225,098,112; adding the 5,848,704-byte DRAM allocator base gives4,230,946,816,
exactly the fatal trace address. Its end is the 4,278,190,080-byte Blackhole DRAM
view ceiling. Capture's allocation/deletion high-water mark is4,238,878,336,
so the placement overlaps captured addresses by 7,931,520 bytes per bank,
63,452,160 bytes aggregate.

## Why aggregate free bytes at finalization are insufficient

`mesh_trace.cpp` allocates trace storage after recording the forward, when its
temporary tensors have already been released. Replay still needs the addresses
of those temporary allocations. The high-water assertion prevents trace commands
from occupying those addresses. The 108,925,440 free bytes after finalization
therefore do not demonstrate that replay workspace fits.

An independent source-level lower bound establishes a physical coexistence
failure for the current sum implementation. `SparseMoe` keeps the private
weighted `down` tensor alive through `ttnn.sum(down, dim=1)`. Each decode expert
matrix has logical height 1 and physical tiled height 32. Its BF16 storage is:

`512 experts * 32 padded rows * 2560 columns * 2 bytes = 83,886,080bytes (80 MiB)`.

`generic_reductions.cpp` clones tiled input whenever either spatial logical
dimension is nonaligned, then clears implicit padding on that clone before
reducing. The original and clone therefore coexist, requiring at least 160 MiB.
This follows `compute_sparse_matmul_output_shape`, which preserves the input's
logical M dimension, and the reduction's explicit clone-before-padding branch.
No additional transpose or reduction workspace is needed for this lower bound.

Even excluding all request states, inputs, outputs and warmed program resources:

`33,659,191,296 resident model +358,875,136 trace +167,772,160 workspace`
`=34,185,838,592bytes`, exceeding total DRAM by 7,107,584 bytes.

Using the measured prepared pre-capture allocations instead of model-only
storage gives a 55,569,920-byte shortfall. Thus allocation order/fragmentation
explains the exact collision addresses, but eliminating holes alone cannot fit
this captured stream together with the current required sum workspace.
A different reduction or expert representation could change the bound; neither
is asserted impossible or implemented by this diagnosis.

## Supported statement

Batch 4 is the largest tested passing traced PLE decoder batch for the current
fully resident implementation. `ple_final_batch4_controlled.json` passes all
literal whole-output and literal decode gates and all matched-control gates;
its request 2 fresh-prefill contribution discrepancy remains explicitly carried
for stage review. It is not a literal strict-contribution pass.
Batch 5 fails trace/workspace coexistence with the exact measured reservation
and source-derived lower bound above. This statement does not extrapolate to
future fused/compact implementations or claim that the model inherently has a
batch 4 limit. Full resident batch 32 has separate failed allocation evidence.

## Source and evidence identity

- `tt_metal/distributed/mesh_trace.cpp`: allocation precedes overlap assertion.
- `tt_metal/impl/allocator/algorithms/free_list_opt.cpp`: block-table addresses
  omit the allocation offset.
- `ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp`:
  `compute_sparse_matmul_output_shape` preserves logical M.
- `ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp`:
  nonaligned tiled input is cloned before `fill_implicit_tile_padding`.
- `tt/decoder_components.py`: weighted private down tensor remains live through sum.

- SHA256 `e89529fd51f5de716d52d3f1ff24bdddce60a2862edeefe81570d58f6adfe767`: `models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ple_batch5_trace_memory.json`
- SHA256 `e76c83212810aee13fc8c9e35eaae3429b923cbf80a11c5c3c8e5035a04245d6`: `ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp`
- SHA256 `cb63a6e4db11ccbb139884f55b083149da1a7054fe7765cdad6f8ac84ae3d384`: `models/autoports/qwen_qwen3_8_flash_next/tt/decoder_components.py`
