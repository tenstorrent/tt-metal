# ADD (binary_ng) Implementation Analysis

## Overview

The ADD operation in `binary_ng` performs element-wise addition of two tensors (or a tensor and a scalar) on Tenstorrent hardware. It is the modern "next-generation" binary operation framework that replaces the legacy binary path. The operation supports multi-dimensional broadcasting, sharding, pre/post activation fusions, and both FPU and SFPU execution paths.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

## Path Selection: FPU vs SFPU

The path selection is determined by the function `utils::is_binary_sfpu_op()` in `binary_ng_device_operation.cpp`. For **ADD**, the SFPU path is selected when both input tensors have identical data types AND the type is one of: `FLOAT32`, `INT32`, `UINT32`, or `UINT16`. When both inputs are `BFLOAT16` (the most common case), the FPU path is used instead, where ADD maps to the FPU's native `add_tiles` instruction via `EltwiseBinaryType::ELWADD`.

Inside `ProgramFactory::create()`, the `is_sfpu` boolean from `operation_attributes` drives the selection: when true, `OpConfig` is constructed with `std::in_place_type<OpConfig::SfpuBinaryOp>`, which sets `binary_op = SfpuBinaryOp::ADD`. This causes `as_defines()` to emit `BINARY_SFPU_INIT` and `BINARY_SFPU_OP` macros (resolving to `add_binary_tile_init()` / `add_binary_tile` for float types, or `add_int_tile_init()` / `add_int_tile<DataFormat::...>` for integer types). The kernel file path routing in `get_kernel_file_path()` then selects SFPU-specific compute kernel files (e.g., `eltwise_binary_sfpu_no_bcast.cpp` instead of `eltwise_binary_no_bcast.cpp`). The FPU path uses entirely different compute kernel files and maps ADD to the matrix unit's native add instruction; it is not analyzed further here.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32 elements) |
| **Unit size** | 1 tile per cycle (`num_tiles_per_cycle = 1`) |
| **Total units** | `c.physical_volume() / (tile_height * tile_width)` -- total output tiles |
| **Loop structure** | Outer loop over complete broadcast iterations, inner loop per tile within a broadcast group |

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|----------------|----------------|
| **Logical shape** | Up to rank-6+ (collapsed to 5D internally: nD, D, N, C, H, W) | Same rank as A (with broadcast-compatible dims) |
| **Dimension convention** | [..., D, N, C, H, W] | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED (height, width, block) | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 (or absent for scalar path) |
| **Data type** | FLOAT32, INT32, UINT32, or UINT16 (SFPU path) | Same as A (SFPU path requires matching types) |

### Output Tensor

| Property | Output Tensor C |
|----------|-----------------|
| **Logical shape** | Broadcast-expanded shape of A and B |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Configurable; defaults to input dtype |

### Layout Transformations

No explicit tilize/untilize is performed within the operation. All inputs and outputs must already be in TILE_LAYOUT. When input and output data types differ and the operation is not a quantization op, a TYPECAST post-activation is appended to the compute pipeline.

## Data Flow Pattern

The data flow varies by whether `input_tensor_b` is a tensor or a scalar.

### Two-Tensor Path (SFPU, no broadcast -- `SubtileBroadcastType::NONE`)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader (`reader_interleaved_no_bcast.cpp` -- ng variant) | DRAM/L1 (A and B buffers) | CB c_0 (A), CB c_1 (B) | reserve_back, noc_async_read_page, push_back (per tile) |
| 2 | Compute (`eltwise_binary_sfpu_no_bcast.cpp`) | CB c_0 (A), CB c_1 (B) | CB c_2 (C) | wait_front, copy_tile to DEST, BINARY_SFPU_OP, pack_tile, push_back, pop_front |
| 3 | Writer (`writer_interleaved_no_bcast.cpp` -- ng variant) | CB c_2 (C) | DRAM/L1 (C buffer) | wait_front, noc_async_write_page, pop_front |

### Scalar Path (SFPU)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1a | Writer (`writer_interleaved_scalar.cpp`) fills B scalar | Packed scalar arg | CB c_1 (B) | reserve_back, fill_with_val, push_back (once) |
| 1b | Reader (`reader_interleaved_no_bcast.cpp` -- old variant) | DRAM/L1 (A buffer) | CB c_0 (A) | reserve_back, noc_async_read_page, push_back (per tile) |
| 2 | Compute (`eltwise_binary_sfpu_scalar.cpp`) | CB c_0 (A), CB c_1 (B) | CB c_2 (C) | wait_front (B once), then loop: wait_front (A), copy_tile, BINARY_SFPU_OP, pack_tile, push_back, pop_front (A) |
| 3 | Writer (`writer_interleaved_scalar.cpp`) writes C | CB c_2 (C) | DRAM/L1 (C buffer) | wait_front, noc_async_write_page, pop_front |

### Broadcast Path (SFPU, e.g., `SubtileBroadcastType::SCALAR_A/B`, `COL_A/B`)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader (broadcast-specific ng variant) | DRAM/L1 | CB c_0, CB c_1 | Reads tiles with stride-based broadcasting logic |
| 2 | Compute (`eltwise_binary_sfpu.cpp`) | CB c_0, CB c_1 | CB c_2 | Outer loop: wait broadcast input once, inner loop: process non-broadcast input, SFPU op, pack, pop non-broadcast, then pop broadcast |
| 3 | Writer (`writer_interleaved_no_bcast.cpp` -- ng) | CB c_2 | DRAM/L1 | wait_front, write, pop_front |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src_a | Input A staging | 2 tiles (interleaved) or shard_volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_src_b | Input B staging | 2 tiles (interleaved, tensor) or 1 tile (scalar) or shard_volume (sharded) | 1 tile | Double/Single | Reader or Writer (scalar) | Compute | Program |
| c_2 | cb_out | Output staging | 2 tiles (interleaved) or shard_volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Compute | Writer | Program |
| c_3 | cb_post_lhs | LHS activation intermediate | 1 tile | 1 tile | Single | Compute (PREPROCESS) | Compute (main) | Block |
| c_4 | cb_post_rhs | RHS activation intermediate | 1 tile | 1 tile | Single | Compute (PREPROCESS) | Compute (main) | Block |
| c_5 | cb_row_bcast_a | Row broadcast buffer for A | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_6 | cb_row_bcast_b | Row broadcast buffer for B | 2 tiles | 1 tile | Double | Reader | Compute | Program |

**Notes**: CBs c_3 and c_4 are only created when LHS/RHS activations are non-empty. CBs c_5 and c_6 are only created for ROW_A/ROW_A_COL_B and ROW_B/ROW_B_COL_A broadcast types respectively. For the SFPU path, intermediate format for c_3/c_4 matches the input data format (unlike the FPU path which may convert to Float16_b for exp-containing ops).

## Pipeline Pattern Summary

- **Interleaved (no sharding)**: CB c_0 and c_1 have capacity=2, block=1 -- **Double-buffered**, allowing read-compute overlap. CB c_2 also capacity=2, block=1 -- **Double-buffered** for compute-write overlap.
- **Sharded**: CB capacities match the shard volume. Entire shard is pushed at once -- effectively **Single-buffered** (all data available in L1 from the start).
- **Scalar path**: CB c_1 has capacity=1 (single scalar tile, loaded once) -- **Single-buffered**.

## Index Calculations

The program factory decomposes tensor shapes into 5 dimensions: `[nD, D, N, C, Ht, Wt]` where `nD` collapses all dimensions beyond rank 5, and `Ht`/`Wt` are tile counts along height and width.

**Stride computation for broadcasting**: For each input, per-dimension strides are computed as products of inner dimensions, but multiplied by a boolean `(dim > 1)`. When a dimension has size 1, its stride becomes 0, effectively implementing broadcasting by not advancing the read pointer along that dimension.

For example, `c_stride = aHt * aWt * (aC > 1)` means: if A's C dimension is 1, stride is 0 (broadcast); otherwise stride equals `Ht * Wt`.

**Reader tile offset calculation**: Inside reader kernels, the start tile ID (from the output space) is decomposed into `[start_nd, start_d, start_n, start_c, start_th, start_tw]` coordinates, then mapped to the input tile space using the stride values. The nested 6-deep loop iterates through all output tiles, advancing input offsets by dimension-specific strides.

**TensorAccessor**: Both reader and writer kernels use `TensorAccessor` for DRAM page addressing. Compile-time args encode the accessor configuration; common runtime args carry the shape information needed for bank mapping.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads within a row of tiles (innermost loop over `tw`), then stepping through tile rows (`th`), channels (`c`), batches (`n`), etc. Each tile read uses `noc_async_read_page` followed by `noc_async_read_barrier` (synchronous, one tile at a time). For the two-tensor no-broadcast reader, A and B tiles are read in lockstep.
- **Sharded**: No DRAM reads. Entire shard is in L1. Reader just calls `cb_reserve_back` + `cb_push_back` to make tiles visible to compute.

### Write Pattern
- **Interleaved**: Sequential tile writes matching the output tile order. Each tile: `cb_wait_front` -> `noc_async_write_page` -> `noc_async_write_barrier` -> `cb_pop_front`.
- **Sharded**: No DRAM writes. Output CB is backed by the output tensor's L1 buffer. Writer kernel body is skipped (`#if !DST_SHARDED`).

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (uses full worker grid) |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` or shard grid |
| **Total cores** | `compute_with_storage_grid.x * compute_with_storage_grid.y` (zero-start) or `all_device_cores.num_cores()` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` (interleaved); shard tiles (sharded) |
| **Load balancing** | Two-group split: group 1 gets `ceil(total_tiles / num_cores)` tiles, group 2 gets one fewer. Cores outside both groups receive zero-arg arrays and exit immediately. |

For sharded operation, each core processes exactly its shard's tiles. The `ShardShapeGenerator` handles edge cases where the last core in each dimension may have a smaller shard.

The `zero_start_grid` optimization is used when the worker grid is a single rectangle starting at (0,0) and shards also start at (0,0), enabling faster `grid_to_cores` work distribution.

## Arguments

### Compile-Time Arguments

**Compute Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Always 1; tiles produced per read-compute-write cycle |

**Reader Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..M | TensorAccessor args (A) | uint32_t[] | Accessor configuration for input A |
| M+1..N | TensorAccessor args (B) | uint32_t[] | Accessor configuration for input B |
| N+1 | has_sharding | uint32_t | 1 if any tensor is natively sharded, 0 otherwise |

**Writer Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..M | TensorAccessor args (C) | uint32_t[] | Accessor configuration for output C |
| M+1 | has_sharding | uint32_t | 1 if any tensor is natively sharded, 0 otherwise |

### Runtime Arguments

**Reader Kernel** (21 args per core):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Base address of input A buffer |
| 1 | c_start_id | uint32_t | Starting output tile ID for this core |
| 2 | a_num_tiles | uint32_t | Number of tiles in A's shard (sharded only, else 0) |
| 3 | c_num_tiles | uint32_t | Number of output tiles for this core |
| 4 | c_current_shard_width | uint32_t | Shard width in tiles (sharded only, else 0) |
| 5 | nD_stride | uint32_t | A's stride for collapsed dims >5 (0 if dim=1) |
| 6 | d_stride | uint32_t | A's stride for D dimension |
| 7 | n_stride | uint32_t | A's stride for N dimension |
| 8 | c_stride | uint32_t | A's stride for C dimension |
| 9 | cD | uint32_t | Output D dimension |
| 10 | cN | uint32_t | Output N dimension |
| 11 | cC | uint32_t | Output C dimension |
| 12 | cHt | uint32_t | Output height in tiles |
| 13 | cWt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Output collapsed nD dimension |
| 15 | src_addr_b | uint32_t | Base address of input B buffer (0 if scalar) |
| 16 | nD_stride_b | uint32_t | B's stride for collapsed dims |
| 17 | d_stride_b | uint32_t | B's stride for D dimension |
| 18 | n_stride_b | uint32_t | B's stride for N dimension |
| 19 | c_stride_b | uint32_t | B's stride for C dimension |
| 20 | b_num_tiles | uint32_t | Number of tiles in B's shard (sharded only, else 0) |

**Writer Kernel** (two-tensor path, 11 args):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Base address of output C buffer |
| 1 | c_start_id | uint32_t | Starting output tile ID |
| 2 | c_num_tiles | uint32_t | Number of output tiles for this core |
| 3 | c_current_shard_width | uint32_t | Shard width in tiles |
| 4 | cD | uint32_t | Output D dimension |
| 5 | cN | uint32_t | Output N dimension |
| 6 | cC | uint32_t | Output C dimension |
| 7 | cHt | uint32_t | Output height in tiles |
| 8 | cWt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Output collapsed nD dimension |
| 10 | (unused) | uint32_t | Set to 0 |

**Writer Kernel** (scalar path, 11 args):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar | uint32_t | Bit-packed scalar value for B |
| 1 | dst_addr | uint32_t | Base address of output C buffer |
| 2 | c_start_id | uint32_t | Starting output tile ID |
| 3 | c_num_tiles | uint32_t | Number of output tiles |
| 4 | c_current_shard_width | uint32_t | Shard width in tiles |
| 5 | cD | uint32_t | Output D dimension |
| 6 | cN | uint32_t | Output N dimension |
| 7 | cC | uint32_t | Output C dimension |
| 8 | cHt | uint32_t | Output height in tiles |
| 9 | cWt | uint32_t | Output width in tiles |
| 10 | cND | uint32_t | Output collapsed nD dimension |

**Compute Kernel** (4 args):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total tiles to process on this core |
| 1 | tile_freq | uint32_t | Broadcast frequency (1 for NONE/ROW, Ht*Wt for SCALAR, Wt for COL) |
| 2 | tile_start | uint32_t | Starting offset within the broadcast group |
| 3 | compute_scalar_value | uint32_t | Quantization zero-point (0 for ADD) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 (A, B) | CB c_0, CB c_1 | Read input tiles via NoC |
| Compute | TRISC (unpack+math+pack) | N/A | CB c_0, CB c_1 | CB c_2 | Unpack to DEST, SFPU add, pack |
| Writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 (C) | Write output tiles via NoC |

### Reader Kernel (Two-Tensor No-Broadcast Path)

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` |
| Assigned cores | All active cores in worker grid |

**Key Logic**:
- Reads both A and B tiles in lockstep using TensorAccessor-based page addressing.
- **Sharded fast path**: When `SRC_SHARDED` is defined, skips DRAM reads entirely; calls `cb_reserve_back(cb_id_src, src_num_tiles)` + `cb_push_back` to expose the pre-loaded L1 data to compute. Same pattern for `SRC_SHARDED_B`.
- **Interleaved path**: Decomposes `start_tile_id` into 6D coordinates, computes initial tile offsets for A and B using their respective stride arrays, then iterates through a 6-deep nested loop (nD, D, N, C, Ht, Wt). For each tile, issues `noc_async_read_page` for both A and B, calls a single `noc_async_read_barrier`, then pushes both tiles.
- **Synchronization**: Produces to CB c_0 and CB c_1 via `cb_reserve_back` / `cb_push_back`. Each iteration: reserve 1 tile on both CBs, read both tiles, barrier, push both.

### Compute Kernel (SFPU No-Broadcast)

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` |
| Assigned cores | All active cores in worker grid |

**Key Logic**:
- Initializes with `unary_op_init_common(cb_post_lhs, cb_out)`.
- Calls `BINARY_SFPU_INIT` (expands to `add_binary_tile_init()` for float ADD, or `add_int_tile_init()` for integer ADD) once at the top if no pre/post activations are active.
- **Main loop**: For each tile (0 to `num_tiles`):
  1. `PREPROCESS(LHS, ...)` -- if LHS activations exist, copies tile from c_0 to c_3 applying activation. Otherwise no-op.
  2. `cb_wait_front(cb_post_lhs, 1)` -- waits for LHS tile.
  3. `PREPROCESS(RHS, ...)` -- same for RHS (c_1 to c_4).
  4. `cb_wait_front(cb_post_rhs, 1)` -- waits for RHS tile.
  5. `cb_reserve_back(cb_out, 1)` -- reserves output space.
  6. `tile_regs_acquire()` -- acquires DEST registers.
  7. `copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs)` then `copy_tile(cb_post_lhs, 0, 0)` -- unpacks LHS to DEST[0].
  8. `copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs)` then `copy_tile(cb_post_rhs, 0, 1)` -- unpacks RHS to DEST[1].
  9. `BINARY_SFPU_OP(0, 1, 0)` -- executes SFPU binary add: `add_binary_tile(0, 1, 0)`. Result in DEST[0].
  10. `PROCESS_POST_ACTIVATIONS(0)` -- applies any post-activation (e.g., TYPECAST).
  11. `tile_regs_commit()` then `tile_regs_wait()`.
  12. `pack_tile(0, cb_out)` -- packs DEST[0] to output CB.
  13. `tile_regs_release()`.
  14. `cb_push_back(cb_out, 1)` -- publishes output tile.
  15. `cb_pop_front(cb_post_lhs, 1)` and `cb_pop_front(cb_post_rhs, 1)` -- releases input tiles.
- **Synchronization**: Waits on CB c_0/c_3 (LHS) and CB c_1/c_4 (RHS) via `cb_wait_front`; pops both after processing. Produces to CB c_2 via `cb_reserve_back` / `cb_push_back`.

### Compute Kernel (SFPU Scalar)

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp` |
| Assigned cores | All active cores (when B is a scalar) |

**Key Logic**:
- RHS scalar tile is loaded once: `PREPROCESS(RHS, ...)` + `cb_wait_front(cb_post_rhs, 1)` before the main loop.
- **Main loop**: For each tile, only LHS is waited on per iteration. RHS tile stays resident in its CB. After all tiles are processed, RHS is popped once.
- Same DEST register pattern: LHS -> DEST[0], RHS -> DEST[1], `BINARY_SFPU_OP(0, 1, 0)`, pack DEST[0].
- **Synchronization**: Waits on CB c_0/c_3 per tile, waits on CB c_1/c_4 once. Pops LHS per tile, pops RHS once at the end.

### Compute Kernel (SFPU Broadcast)

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu.cpp` |
| Assigned cores | All active cores (when broadcast type requires repeated application of one input) |

**Key Logic**:
- Uses `BCAST_INPUT` compile-time define to determine which input is the "broadcast" (loaded once per group) and which is the "other" (loaded each iteration).
- `process_tile()` function: waits for broadcast input once, then loops `freq` times over the other input, performing the SFPU binary op for each tile.
- `kernel_main()` computes `complete_iterations` and `remaining_iterations` from `num_tiles`, `tile_freq`, and `tile_start` to handle partial broadcast groups at the start/end of a core's work.

### Writer Kernel (Two-Tensor Path)

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` |
| Assigned cores | All active cores (two-tensor path) |

**Key Logic**:
- **Sharded fast path**: When `DST_SHARDED` is defined, the entire kernel body is compiled out. Output is already in L1 via the CB's backing buffer.
- **Interleaved path**: Decomposes `start_tile_id` into 6D coordinates, iterates through the same nested loop structure as the reader, writing one tile at a time via `noc_async_write_page`.
- **Synchronization**: Consumes from CB c_2 via `cb_wait_front(cb_id_dst, 1)` / `cb_pop_front(cb_id_dst, 1)`.

### Writer Kernel (Scalar Path)

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp` |
| Assigned cores | All active cores (scalar path) |

**Key Logic**:
- **Dual responsibility**: First fills a single tile in CB c_1 with the packed scalar value (using `fill_with_val` or `fill_with_val<1024, float>`), then writes output tiles from CB c_2 to DRAM.
- The scalar fill happens once at kernel start; the output write loop is identical in structure to the no-bcast writer.
- **Synchronization**: Produces to CB c_1 (one tile, once). Consumes from CB c_2 per tile.

## Implementation Notes

- **Program factory variants**: There is a single `ProgramFactory` that handles all cases (scalar, two-tensor, all broadcast types, sharded/interleaved). The factory selects kernels dynamically based on `subtile_broadcast_type`, `is_sfpu`, and whether `input_tensor_b` is present.

- **Type-based operation variants**: The SFPU ADD path supports FLOAT32 (using `add_binary_tile`), INT32 (`add_int_tile<DataFormat::Int32>`), UINT32 (`add_int_tile<DataFormat::UInt32>`), and UINT16 (`add_int_tile<DataFormat::UInt16>`). BFLOAT16 ADD uses the FPU path exclusively.

- **UnpackToDestFP32 mode**: For the SFPU path (when `op_type != BinaryOpType::POWER`), all four source CBs (c_0, c_1, c_3, c_4) are set to `UnpackToDestMode::UnpackToDestFp32`. This means data is unpacked to FP32 precision in the DEST registers regardless of the source data format, ensuring maximum precision during the SFPU computation.

- **Broadcast type selection**: Nine subtile broadcast modes are supported (NONE, SCALAR_A/B, ROW_A/B, COL_A/B, ROW_A_COL_B, ROW_B_COL_A). The broadcast type is determined by comparing the last two dimensions of A and B. For ADD specifically, stride-based broadcasting is used: strides are multiplied by `(dim > 1)` to zero-out movement along broadcast dimensions.

- **Sharding support and constraints**: Height, width, and block sharding are all supported. Native L1 sharding requires: non-scalar path, both tensors have identical shapes and memory configs, no DRAM buffers, no uneven shards on the output, and matching grids for A/B/C. When these conditions are not met, the operation falls back to the interleaved (TensorAccessor) path even for sharded tensors.

- **FP32 dest accumulation**: Enabled when output format is UInt32/Int32/Float32, or when both inputs are Float32 or both are Int32/UInt32. This is set via `fp32_dest_acc_en` in the `ComputeConfig`.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/common/inc/sfpu/ckernel_sfpu_binary.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `add_binary_tile(0, 1, 0)` (via the `BINARY_SFPU_OP` macro), which is defined in `eltwise_binary_sfpu.h`.
2. `add_binary_tile` wraps `MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>(0, 1, 0)))`, routing the call to the math RISC-V thread.
3. `llk_math_eltwise_binary_sfpu_binop` (in `llk_math_eltwise_binary_sfpu_binop.h`) calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>()` with a function pointer to `calculate_sfpu_binary<APPROXIMATE, BinaryOp::ADD, 8, is_fp32_dest_acc_en>`.
4. `_llk_math_eltwise_binary_sfpu_params_` (in `llk_math_eltwise_binary_sfpu_params.h`) sets up the DEST write address, stalls until SFPU is ready, then loops over tile faces (4 faces for `VectorMode::RC`), calling the provided function pointer once per face and advancing the DEST pointer by 16 rows between faces via `TTI_SETRWC`.
5. `calculate_sfpu_binary` (in `ckernel_sfpu_binary.h`, the arch-specific overlay) delegates directly to `_calculate_sfpu_binary_<APPROXIMATE, BinaryOp::ADD, 8>`.
6. `_calculate_sfpu_binary_` (in `tt_llk/.../ckernel_sfpu_binary.h`) executes the inner loop: 8 iterations per face, loading two vectors from DEST, adding them, and storing back.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (the default). All 4 faces of the 32x32 tile are processed (Face 0 = top-left 16x16, Face 1 = top-right, Face 2 = bottom-left, Face 3 = bottom-right).
- **Operation invocation**: The core SFPU function `calculate_sfpu_binary` is called once per face, 4 times total per tile. Each invocation processes 8 iterations (rows) within the face, for a total of 32 row-operations per tile (4 faces x 8 iterations).
- **DEST address progression**: Before the face loop, `_llk_math_eltwise_binary_sfpu_start_` sets the DEST write address to 0 (always passes `dst_index=0` since the compute kernel uses tile indices 0, 1, and 0 for in0/in1/out). After each face, two `TTI_SETRWC` instructions each advance the DEST counter by 8 rows (total +16 rows per face), moving to the next face. Within each face, `dst_reg++` (INCRWC) advances by 1 row per iteration. After all 4 faces, `_llk_math_eltwise_binary_sfpu_done_` clears the DEST address.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`), so Style A (inline-commented source code) is used.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h
// (Wormhole B0 variant is identical)

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
// For ADD: APPROXIMATION_MODE=true, BINOP=BinaryOp::ADD, ITERATIONS=8
{
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) // 8 iterations per face (one per row of 32-wide vector elements)
    {
        // dst_tile_size_sfpi=32: each tile occupies 32 SFPI-addressable rows in DEST (64 actual / stride of 2)
        constexpr std::uint32_t dst_tile_size_sfpi = 32;
        sfpi::vFloat in0                           = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD from DEST[0*32 + current_row]
        sfpi::vFloat in1                           = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD from DEST[1*32 + current_row]
        sfpi::vFloat result                        = 0.0f; // initialized but overwritten immediately for ADD

        if constexpr (BINOP == BinaryOp::ADD)
        {
            result = in0 + in1; // SFPADD: element-wise FP32 addition across 32-wide vector lane
        }
        // SUB, MUL, DIV, RSUB, POW, XLOGY branches omitted (compile-time dead code for ADD)

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // SFPSTORE result to DEST[0*32 + current_row]
        sfpi::dst_reg++; // INCRWC: advance DEST row pointer by 1 for next iteration
    }
}
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Description |
|-------------|------------------|-------------|
| **SFPLOAD** | `sfpi::dst_reg[index]` (read) | Loads a 32-wide vector from a DEST register row into an SFPU local register (LREG). Two loads per iteration: one for `in0` (from tile 0) and one for `in1` (from tile 1). |
| **SFPADD** | `in0 + in1` (vFloat operator+) | Performs element-wise FP32 addition on two 32-wide vectors. Maps to `__builtin_rvtt_sfpadd` (Wormhole) / `__builtin_rvtt_bh_sfpadd` (Blackhole). This is the core computational instruction for binary ADD. |
| **SFPSTORE** | `sfpi::dst_reg[index] = result` (write) | Stores a 32-wide vector result back to a DEST register row. One store per iteration, writing to tile 0's position (since `odst=0`). |
| **INCRWC** | `sfpi::dst_reg++` | Increments the DEST register read/write pointer by 1 row. Maps to `__builtin_rvtt_ttincrwc`. Called once per iteration to advance to the next row within the face. |
| **SETRWC** | `TTI_SETRWC(...)` (in params dispatch) | Resets/advances the DEST counter by 8 rows between faces. Two SETRWC calls per face = +16 rows, moving from one 16x16 face to the next within the 32x32 tile. |
| **STALLWAIT** | `TTI_STALLWAIT(...)` (in params dispatch) | Stalls the math thread until the SFPU pipeline is idle. Called at the start (`STALL_SFPU, MATH`) and on Wormhole also at the end (`STALL_CFG, WAIT_SFPU`). |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST[0..31]** (tile 0) | Holds input A tile data (loaded by `copy_tile(cb_post_lhs, 0, 0)`) and also receives the output result. The SFPU reads from `DEST[0*32 + row]` for `in0` and writes the result back to `DEST[0*32 + row]`. |
| **DEST[32..63]** (tile 1) | Holds input B tile data (loaded by `copy_tile(cb_post_rhs, 0, 1)`). The SFPU reads from `DEST[1*32 + row]` for `in1`. This tile is consumed but not overwritten. |
| **LREG[0..3]** (SFPU local regs) | Used implicitly by the SFPI compiler. `in0`, `in1`, and `result` are mapped to SFPU local registers. The ADD operation is simple enough that only 2-3 LREGs are needed simultaneously (two for inputs, one for output). |

### Address Mode Configuration

The address mode is configured during initialization by `eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>()` in `llk_math_eltwise_binary_sfpu.h`.

For the binary ADD operation (dispatched with `SfpuType::unused`), only `ADDR_MOD_7` is configured:

| Field | Value | Description |
|-------|-------|-------------|
| `srca.incr` | 0 | No auto-increment for source A addressing |
| `srcb.incr` | 0 | No auto-increment for source B addressing |
| `dest.incr` | 0 | No auto-increment for DEST addressing |

This means ADDR_MOD_7 performs no automatic address advancement -- all DEST pointer movement is handled explicitly by `dst_reg++` (INCRWC) within the inner loop and `TTI_SETRWC` between faces. The choice of ADDR_MOD_7 avoids conflicts with ADDR_MOD_0 and ADDR_MOD_2, which are used by the A2D (unpack-to-DEST) path.

The `ADDR_MOD_6` variant (with `dest.incr=2`) is only configured for integer multiply, min/max operations and is not used for binary ADD.

The address mode configuration is identical across Wormhole B0 and Blackhole hardware generations.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary_ng operation work in TTNN? What is its program factory structure, and how does it handle SFPU vs FPU paths? What are the different subtile broadcast modes?"
   **Reason**: Needed architectural overview of the binary_ng framework before diving into source code.
   **Key Findings**: Confirmed single ProgramFactory design, SFPU/FPU path selection based on `is_binary_sfpu_op()`, and the nine subtile broadcast types with their reader/compute kernel mappings.

2. [SFPU] **Query**: "How does the binary SFPU add operation work? Trace the call chain from add_binary_tile through the LLK layers to the core SFPU implementation."
   **Reason**: Needed to identify all file paths in the SFPU call chain for binary ADD before reading source code.
   **Key Findings**: Confirmed the 5-layer abstraction: API header -> LLK binop dispatch -> LLK params dispatch -> calculate_sfpu_binary wrapper -> _calculate_sfpu_binary_ core. Identified file paths for both Blackhole and Wormhole variants.

3. [SFPU] **Query**: "How does add_binary_tile work in the LLK? Trace from the compute API through llk_math to the ckernel SFPU implementation." (tenstorrent/tt-llk)
   **Reason**: Cross-referencing the tt-llk repo for the core SFPU implementation details and iteration structure.
   **Key Findings**: Confirmed _calculate_sfpu_binary_ uses a loop of 8 iterations with sfpi::dst_reg load/store and operator+ for addition. Confirmed dst_reg++ increments the DEST pointer per iteration.

4. [SFPU] **Query**: "What SFPU instruction does the vFloat operator+ map to? What instruction does dst_reg[index] read/write map to?" (tenstorrent/sfpi)
   **Reason**: Needed to map SFPI C++ abstractions to concrete SFPU hardware instructions.
   **Key Findings**: operator+ maps to SFPADD (__builtin_rvtt_sfpadd), dst_reg reads map to SFPLOAD, writes map to SFPSTORE.

5. [SFPU] **Query**: "What SFPU instruction does dst_reg++ map to? When you write vFloat result = 0.0f, does that emit SFPLOADI?" (tenstorrent/sfpi)
   **Reason**: Needed to understand the remaining SFPU instructions in the kernel.
   **Key Findings**: dst_reg++ maps to TTINCRWC (__builtin_rvtt_ttincrwc). Constant 0.0f is typically optimized to use vConst0 rather than SFPLOADI.

### Documentation References
1. **Source**: `binary_ng_utils.cpp` (lines 131-367)
   **Reason**: Understanding OpConfig construction and how ADD maps to SFPU init/op functions.
   **Key Information**: ADD maps to `add_binary_tile_init()` / `add_binary_tile` for float, `add_int_tile_init()` / `add_int_tile<DataFormat::...>` for integer types.

2. **Source**: `binary_ng_device_operation.cpp` (lines 16-66)
   **Reason**: Understanding SFPU path selection criteria for ADD.
   **Key Information**: ADD uses SFPU when `a == b && (a == FLOAT32 || a == INT32 || a == UINT32 || a == UINT16)`.

3. **Source**: `binary_ng_utils.cpp` (lines 81-129, `get_kernel_file_path`)
   **Reason**: Mapping KernelName enums to actual file paths for SFPU variants.
   **Key Information**: SFPU kernels use `eltwise_binary_sfpu_*.cpp` variants; two-tensor path uses `kernels_ng/` directory for reader/writer.

### Confluence References
[No Confluence references were needed for this analysis. The binary ADD SFPU kernel is straightforward and fully documented via DeepWiki and source code.]

### Glean References
[No Glean references were needed for this analysis.]
