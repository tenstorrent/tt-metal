# MUL (binary_ng) Implementation Analysis

## Overview

The MUL operation performs element-wise multiplication of two tensors (`c = a * b`) using the **binary_ng** (next-generation binary) program factory. This is a unified framework that supports dozens of binary operations through compile-time macro specialization. The MUL operation can execute on either the FPU (matrix unit) or SFPU (vector unit) depending on data types and configuration flags. This analysis focuses exclusively on the SFPU execution path.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

## Path Selection: FPU vs SFPU

The binary_ng framework selects between FPU and SFPU paths via the `is_binary_sfpu_op()` function in `binary_ng_device_operation.cpp` (line 15). For MUL specifically (line 28-29):

```cpp
case MUL:
    return !fast_and_approximate_mode || (a == b && (a == FLOAT32 || a == INT32 || a == UINT32 || a == UINT16));
```

This means **MUL uses the SFPU path** when:
1. `fast_and_approximate_mode` is **false** (the default) -- in this case SFPU is always selected regardless of data types, OR
2. `fast_and_approximate_mode` is **true** AND both inputs share the same dtype that is one of: FLOAT32, INT32, UINT32, or UINT16.

When `fast_and_approximate_mode` is true and the types do not match the conditions above, the FPU path is used instead (FPU MUL is `mul_tiles` via the matrix unit). In the program factory (`binary_ng_program_factory.cpp`, line 578), the path selection materializes as:

```cpp
const auto op_config = is_sfpu_op ? OpConfig(op_type, std::in_place_type<OpConfig::SfpuBinaryOp>, a_dtype)
                                  : OpConfig(op_type, std::in_place_type<OpConfig::FpuBinaryOp>, a_dtype);
```

This `OpConfig` generates the `BINARY_SFPU_INIT` and `BINARY_SFPU_OP` macro defines that are injected into the compute kernel at compile time. For SFPU MUL, the init/op functions are resolved by `get_sfpu_init_fn()` in `binary_ng_utils.cpp` (line 387-393):
- For integer types (INT32/UINT32/UINT16): `mul_int_tile_init<DataFormat::X>()` / `mul_int_tile<DataFormat::X>`
- For floating-point types: `mul_binary_tile_init()` / `mul_binary_tile`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `c.physical_volume() / tile_hw` (total tiles in the output tensor) |
| **Loop structure** | Each compute iteration processes `num_tiles_per_cycle` tiles (hardcoded to 1) |

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|----------------|----------------|
| **Logical shape** | Up to 6D, collapsed for rank > 5 | Up to 6D (or scalar) |
| **Dimension convention** | [..., D, N, C, H, W] (last 5 dims extracted) | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED (height/width/block) | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 (or absent if scalar) |
| **Data type** | FLOAT32, INT32, UINT32, UINT16, BFLOAT16 | Same as A on SFPU path (or scalar) |

### Output Tensor

| Property | Output Tensor C |
|----------|-----------------|
| **Logical shape** | Broadcast-expanded shape of A and B |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Determined by operation output dtype |

### Layout Transformations

No explicit tilize/untilize is performed within the program factory. All inputs and outputs are expected in TILE_LAYOUT. The SFPU path uses `copy_tile` (unpack to DEST) rather than FPU matmul, so no format conversion occurs during compute. When `a_dtype != c_dtype` (and not a quant op), a `TYPECAST` post-activation is automatically appended.

## Data Flow Pattern

The data flow depends on whether B is a tensor or a scalar, and whether sharding is used. Below describes the two-tensor interleaved (no-broadcast) SFPU case:

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (A buffer) | CB c_0 | reserve_back(c_0, 1), noc_async_read_page, push_back(c_0, 1) |
| 1 | Reader | DRAM (B buffer) | CB c_1 | reserve_back(c_1, 1), noc_async_read_page, push_back(c_1, 1) |
| 2 | Compute | CB c_0, CB c_1 | CB c_2 | wait_front(c_0, 1), wait_front(c_1, 1), reserve_back(c_2, 1), copy_tile to DEST, BINARY_SFPU_OP, pack_tile, push_back(c_2, 1), pop_front(c_0, 1), pop_front(c_1, 1) |
| 3 | Writer | CB c_2 | DRAM (C buffer) | wait_front(c_2, 1), noc_async_write_page, pop_front(c_2, 1) |

For the **scalar case** (B absent), the writer kernel fills a single tile in CB c_1 with the scalar value at startup, and the compute kernel reads that tile once (never pops it), reusing it for all output tiles.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src_a | Input A tiles | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_src_b | Input B tiles or scalar | 2 tiles (tensor) / 1 tile (scalar) / shard volume (sharded) | 1 tile | Double (tensor) / Single (scalar/sharded) | Reader or Writer (scalar) | Compute | Program |
| c_2 | cb_out | Output tiles | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Compute | Writer | Program |
| c_3 | cb_lhs_intermediate | LHS activation intermediate | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_4 | cb_rhs_intermediate | RHS activation intermediate | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_5 | cb_row_bcast_a | Row broadcast buffer for A | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_6 | cb_row_bcast_b | Row broadcast buffer for B | 2 tiles | 1 tile | Double | Reader | Compute | Program |

Notes:
- CBs c_3 and c_4 are only created when LHS/RHS pre-activations are present. For plain MUL, these are not used.
- CBs c_5 and c_6 are only created for row-broadcast subtile broadcast types (ROW_A, ROW_B, ROW_A_COL_B, ROW_B_COL_A).
- On the SFPU path, intermediate data formats for c_3/c_4 match the input data format (unlike FPU which may use Float16_b for some ops).

## Pipeline Pattern Summary

For the interleaved (non-sharded) path:
- **CB c_0, c_1**: Capacity = 2 tiles, block size = 1 tile => **Double-buffered**. Reader can write one tile while compute consumes the other.
- **CB c_2**: Capacity = 2 tiles, block size = 1 tile => **Double-buffered**. Compute can produce one tile while writer drains the other.

For the sharded path:
- **CB c_0, c_1, c_2**: Capacity = shard volume, block size = shard volume => **Single-buffered**. All data is present in L1 from the start.

## Index Calculations

The reader and writer kernels use a 6-level nested loop to traverse tensor dimensions in the order: `nD -> D -> N -> C -> Ht -> Wt`. The starting position is computed from `start_tile_id` by decomposing it into per-dimension offsets:

```
tiles_per_n = C * Ht * Wt
tiles_per_d = N * tiles_per_n
tiles_per_nd = D * tiles_per_d

start_nd  = start_tile_id / tiles_per_nd
start_d   = (start_tile_id % tiles_per_nd) / tiles_per_d
...
start_tw  = offset_c % Wt
```

For broadcasting, each input tensor has its own set of stride values (`nD_stride`, `d_stride`, `n_stride`, `c_stride`) computed by the host. A stride of 0 for a dimension means that dimension is broadcast (the tensor has size 1 along that dimension). The stride calculation in the program factory (lines 395-399) uses the pattern `aHt * aWt * aC * aN * (aD > 1)` -- if a dimension has size 1, the stride is zeroed, effectively repeating the same data.

The `TensorAccessor` API is used for DRAM address translation, converting logical tile indices to physical page addresses across interleaved memory banks.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Tile-by-tile sequential reads through the nested dimension loop. Each tile is read via `noc_async_read_page` with an immediate `noc_async_read_barrier` -- no batching of reads.
- **Sharded**: No reads at runtime. The reader kernel simply does `cb_reserve_back` + `cb_push_back` for the entire shard volume, making the pre-existing L1 data visible to compute.

### Write Pattern
- **Interleaved**: Tile-by-tile sequential writes. Each tile is written via `noc_async_write_page` with an immediate `noc_async_write_barrier`.
- **Sharded**: No writes at runtime. The output CB is backed by the sharded output buffer directly.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (uses full worker grid) |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `grid.x * grid.y` (zero-start) or `all_device_cores.num_cores()` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` (interleaved); shard tiles (sharded) |
| **Load balancing** | Two-group split via `split_work_to_cores`: core_group_1 gets `ceil(total_tiles / num_cores)` tiles, core_group_2 gets one fewer tile. Cores outside both groups are no-ops (all-zero runtime args). |

For sharded tensors, the core grid is determined by the shard specification, and each core processes exactly its shard's tiles.

## Arguments

### Compile-Time Arguments

**Compute kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Tiles per compute cycle (always 1) |

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessor args (A) | uint32_t[] | Compiled tensor accessor parameters for input A |
| N+1..M | TensorAccessor args (B) | uint32_t[] | Compiled tensor accessor parameters for input B |
| M+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

**Writer kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessor args (C) | uint32_t[] | Compiled tensor accessor parameters for output C |
| N+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

### Runtime Arguments

**Reader kernel (two-tensor path, 21 args):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input A buffer address |
| 1 | start_tile_id / c_start_id | uint32_t | Starting output tile index for this core |
| 2 | a_num_tiles | uint32_t | Number of A shard tiles (sharded) or 0 |
| 3 | c_num_tiles | uint32_t | Number of output tiles for this core |
| 4 | c_current_shard_width | uint32_t | Width of current shard in tiles (sharded) or 0 |
| 5 | nD_stride | uint32_t | A's stride for collapsed dims > 5 |
| 6 | d_stride | uint32_t | A's stride for D dimension |
| 7 | n_stride | uint32_t | A's stride for N dimension |
| 8 | c_stride | uint32_t | A's stride for C dimension |
| 9 | cD | uint32_t | Output D dimension |
| 10 | cN | uint32_t | Output N dimension |
| 11 | cC | uint32_t | Output C dimension |
| 12 | cHt | uint32_t | Output height in tiles |
| 13 | cWt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Output collapsed dims > 5 |
| 15 | src_addr_b | uint32_t | Input B buffer address |
| 16 | nD_stride_b | uint32_t | B's stride for collapsed dims > 5 |
| 17 | d_stride_b | uint32_t | B's stride for D dimension |
| 18 | n_stride_b | uint32_t | B's stride for N dimension |
| 19 | c_stride_b | uint32_t | B's stride for C dimension |
| 20 | b_num_tiles | uint32_t | Number of B shard tiles (sharded) or 0 |

**Writer kernel (two-tensor path, 11 args):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output C buffer address |
| 1 | start_tile_id | uint32_t | Starting output tile index |
| 2 | c_num_tiles | uint32_t | Number of output tiles |
| 3 | c_current_shard_width | uint32_t | Shard width in tiles |
| 4 | cD | uint32_t | Output D dimension |
| 5 | cN | uint32_t | Output N dimension |
| 6 | cC | uint32_t | Output C dimension |
| 7 | cHt | uint32_t | Output height in tiles |
| 8 | cWt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Output collapsed dims > 5 |
| 10 | (unused) | uint32_t | Reserved (set to 0) |

**Compute kernel (4 args):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total tiles to process on this core |
| 1 | freq | uint32_t | Broadcast frequency (1 for no-bcast, Wt for col-bcast, Ht*Wt for scalar-bcast) |
| 2 | counter | uint32_t | Starting offset within the broadcast cycle |
| 3 | compute_scalar_value | uint32_t | Quantization zero-point or 0 |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | RISCV_0 | NOC0 | DRAM/L1 (A, B buffers) | CB c_0, CB c_1 | Read A and B tiles via TensorAccessor |
| Compute | RISCV_2 (unpack + math + pack) | N/A | CB c_0, CB c_1 | CB c_2 | copy_tile to DEST, mul_binary_tile SFPU op, pack_tile |
| Writer | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 (C buffer) | Write output tiles via TensorAccessor |

### Reader Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` (two-tensor) or `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp` (scalar) |
| **Assigned cores** | All worker cores in the grid |

**Key Logic:**
- For **interleaved** tensors: reads one tile at a time from both A and B using `noc_async_read_page` with `TensorAccessor` for address translation. Issues a `noc_async_read_barrier` after each tile pair.
- For **sharded** tensors: simply calls `cb_reserve_back(cb, shard_tiles)` + `cb_push_back(cb, shard_tiles)` to expose pre-existing L1 data to the compute kernel. No actual data movement occurs.
- Uses a 6-level nested loop (`nD -> D -> N -> C -> Ht -> Wt`) iterating through the output tile space, with per-dimension stride offsets that handle broadcasting (stride = 0 means broadcast).
- For sharded mode with non-full-width shards, `end_tw` is limited to `start_tw + dst_shard_width` to only read the relevant width slice.
- **Synchronization**: Produces to CB c_0 (A) and CB c_1 (B). Each tile: `cb_reserve_back -> write -> cb_push_back`. Compute consumes via `cb_wait_front/cb_pop_front`.

### Compute Kernel (SFPU, no-broadcast)

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` |
| **Assigned cores** | All worker cores in the grid |

**Key Logic:**
- Initializes with `unary_op_init_common(cb_post_lhs, cb_out)` which sets up the unpack/pack pipeline.
- Calls `BINARY_SFPU_INIT` once at startup (expands to `mul_binary_tile_init()` for float MUL, or `mul_int_tile_init<DataFormat::X>()` for integer MUL).
- Main loop iterates `num_tiles` times. Each iteration:
  1. Waits for LHS tile: `cb_wait_front(cb_post_lhs, 1)`
  2. Waits for RHS tile: `cb_wait_front(cb_post_rhs, 1)`
  3. Reserves output: `cb_reserve_back(cb_out, 1)`
  4. `tile_regs_acquire()` -- acquires DEST register file
  5. `copy_tile(cb_post_lhs, 0, 0)` -- unpacks LHS to DEST slot 0
  6. `copy_tile(cb_post_rhs, 0, 1)` -- unpacks RHS to DEST slot 1
  7. `BINARY_SFPU_OP(0, 1, 0)` -- executes `mul_binary_tile(0, 1, 0)` which multiplies DEST[0] by DEST[1], storing result in DEST[0]
  8. `PROCESS_POST_ACTIVATIONS(0)` -- applies any post-activation (e.g., TYPECAST). Empty for plain MUL with matching types.
  9. `tile_regs_commit()` -- hands DEST to packer
  10. `tile_regs_wait()` -- waits for packer availability
  11. `pack_tile(0, cb_out)` -- packs DEST[0] to output CB
  12. `tile_regs_release()` -- releases DEST
  13. `cb_push_back(cb_out, 1)`, `cb_pop_front(cb_post_lhs, 1)`, `cb_pop_front(cb_post_rhs, 1)`
- For MUL without pre/post activations, `cb_post_lhs = c_0` and `cb_post_rhs = c_1` (no intermediate CBs used).
- **Synchronization**: Consumes from CB c_0 and c_1 (wait_front/pop_front), produces to CB c_2 (reserve_back/push_back).

### Compute Kernel (SFPU, with broadcast)

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu.cpp` |
| **Assigned cores** | All worker cores in the grid |

**Key Logic:**
- Used when one input has a broadcast dimension (col-broadcast or scalar-broadcast via SubtileBroadcastType).
- The `BCAST_INPUT` define (0 or 1) determines which input is the "broadcast" operand (loaded once per cycle) and which is the "other" operand (loaded every iteration).
- Uses `process_tile()` function with a `freq` parameter controlling how many "other" tiles are processed per broadcast tile. The broadcast tile is loaded once, then `freq` iterations process different "other" tiles against it.
- The `tile_start` parameter handles partial first cycles when a core's work doesn't align with broadcast boundaries.
- Same SFPU operation sequence as no-broadcast: `copy_tile -> copy_tile -> BINARY_SFPU_OP -> pack_tile`.

### Compute Kernel (SFPU, scalar)

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp` |
| **Assigned cores** | All worker cores in the grid |

**Key Logic:**
- Used when B is a scalar (not a tensor). The scalar is pre-filled into CB c_1 by the writer kernel.
- RHS tile is waited on once before the loop (`cb_wait_front(cb_post_rhs, 1)`) and popped after the loop (`cb_pop_front(cb_post_rhs, 1)`).
- Inside the loop, only LHS tiles change; RHS remains the same scalar tile throughout.

### Writer Kernel (two-tensor path)

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` |
| **Assigned cores** | All worker cores in the grid |

**Key Logic:**
- For **interleaved**: reads one tile at a time from CB c_2 and writes to DRAM using `noc_async_write_page` via `TensorAccessor`. Uses the same 6-level nested loop structure as the reader.
- For **sharded** (`DST_SHARDED=1`): the entire kernel body is compiled out. The output CB directly backs the sharded output buffer, so no data movement is needed.
- **Synchronization**: Consumes from CB c_2 via `cb_wait_front/cb_pop_front`.

### Writer Kernel (scalar path)

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp` |
| **Assigned cores** | All worker cores in the grid |

**Key Logic:**
- Fills a single tile in CB c_1 with the packed scalar value using `FILL_WITH_VALUE_FLOAT` or `FILL_WITH_VALUE` macros.
- Then writes output tiles from CB c_2 to DRAM using the same nested loop pattern.
- The scalar fill happens once at kernel startup, before the write loop begins.

## Implementation Notes

- **Program factory variants**: There is a single `ProgramFactory::create()` method. The FPU vs SFPU distinction is handled via compile-time defines (`BINARY_SFPU_INIT`, `BINARY_SFPU_OP` for SFPU; `BINARY_OP`, `BINARY_OP_TYPE` for FPU). The kernel file path is selected by `get_kernel_file_path()` based on the `is_sfpu` flag.

- **Type-based operation variants**: SFPU MUL supports FLOAT32, BFLOAT16, INT32, UINT32, and UINT16. For integer types, the init/op functions are `mul_int_tile_init<DataFormat::X>()` / `mul_int_tile<DataFormat::X>`. For floating-point types, they are `mul_binary_tile_init()` / `mul_binary_tile`.

- **UnpackToDestFP32 mode**: For SFPU operations (except POWER), `UnpackToDestMode::UnpackToDestFp32` is enabled on all source CBs (c_0, c_1, c_3, c_4). This forces unpacking to FP32 precision in the DEST register file, ensuring maximum accuracy for SFPU vector operations. For POWER, FP32 unpacking is only enabled when the corresponding input dtype is FLOAT32.

- **Broadcast type selection**: The `SubtileBroadcastType` enum determines how broadcasting is handled at the tile level: `NONE` (no broadcast), `ROW_A/ROW_B` (row broadcast), `COL_A/COL_B` (column broadcast), `SCALAR_A/SCALAR_B` (full tile broadcast), and `ROW_B_COL_A/ROW_A_COL_B` (mixed). This selects which reader kernel variant, compute kernel variant, and broadcast CBs are used. Stride-based broadcasting is used in the reader kernel through per-dimension stride values (stride=0 for broadcast dimensions).

- **Sharding support and constraints**: All three sharding modes are supported (HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED). Native L1 sharding is used when: (1) the output is sharded, (2) `is_native_L1_sharding()` returns true (requires evenly sharded output, matching shapes and grids for two-tensor operations, and L1 buffer type), and (3) the output is not unevenly sharded. When these conditions are not met, the operation falls back to interleaved mode using TensorAccessor for address translation, even if the tensors are technically sharded.

- **FP32 dest accumulation**: Enabled when output format is UInt32, Int32, or Float32, or when both inputs are Float32, Int32, or UInt32. This is set via `fp32_dest_acc_en` in the `ComputeConfig` (line 727-731).

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the floating-point MUL path (`mul_binary_tile`).

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `mul_binary_tile(0, 1, 0)` (expanded from the `BINARY_SFPU_OP` macro).
2. `mul_binary_tile` in the API header calls `llk_math_eltwise_binary_sfpu_binop_mul<APPROX, BinaryOp::MUL, DST_ACCUM_MODE>(0, 1, 0)`, guarded by the `MATH()` macro so it only runs on the math RISC-V.
3. `llk_math_eltwise_binary_sfpu_binop_mul` calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>()`, passing `ckernel::sfpu::calculate_sfpu_binary_mul<APPROX, BinaryOp::MUL, 8, is_fp32_dest_acc_en>` as the callable, with `dst_index_in0=0`, `dst_index_in1=1`, `dst_index_out=0`, and `vector_mode=VectorMode::RC`.
4. `_llk_math_eltwise_binary_sfpu_params_` calls `_llk_math_eltwise_binary_sfpu_start_` (sets DEST write address, stalls until math ready), then loops 4 times (one per face in RC mode) calling `calculate_sfpu_binary_mul`, advancing the DEST read/write counter by 16 rows (2x `TTI_SETRWC` with increment 8) between faces.
5. `calculate_sfpu_binary_mul` is the core SFPU function that performs the actual element-wise multiplication on 8 iterations (rows) per face.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the 32x32 tile are processed. Each face is a 16x16 sub-tile. The RC mode iterates through Face 0, Face 1, Face 2, Face 3 sequentially.
- **Operation invocation**: The core SFPU function `calculate_sfpu_binary_mul` is called once per face (4 times total). Each call processes `ITERATIONS=8` rows, covering all 16 rows of a face (8 SFPU iterations x 2 rows per SFPU-visible row = 16 rows). The SFPU processes 32 elements per row (one full row of the face) via its SIMD vector pipeline.
- **DEST address progression**: Between faces, the DEST read/write counter is advanced by 16 via two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` calls (each increments by 8). Within `calculate_sfpu_binary_mul`, the `dst_reg++` auto-increments the SFPU row pointer by `SFP_DESTREG_STRIDE` (2) after each iteration, so 8 iterations advance by 16 rows.

### Annotated SFPU Kernel Source

This kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) is used.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h

// Helper: Convert float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE)
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);
    // Extract bit 16 (LSB of future bf16 mantissa) for tie-breaking
    sfpi::vUInt lsb = (bits >> 16) & 1;
    // Add 0x7fff + lsb: implements RNE rounding by biasing toward even
    bits = bits + 0x7fffU + lsb;
    // Truncate lower 16 bits to produce bf16-in-fp32 representation
    bits = bits & 0xFFFF0000U;
    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_mul(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // APPROXIMATION_MODE=true/false (unused for MUL), BINOP=BinaryOp::MUL, ITERATIONS=8, is_fp32_dest_acc_en=DST_ACCUM_MODE
    constexpr uint dst_tile_size_sfpi = 32; // Each tile occupies 32 SFPU-visible rows in DEST
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // Load row from DEST tile 0
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // Load row from DEST tile 1

        sfpi::vFloat result = in0 * in1; // SFPMUL: element-wise FP32 multiply

        if constexpr (!is_fp32_dest_acc_en) {
            // When DEST is not in FP32 mode, round result to bfloat16 for accuracy matching FPU behavior
            result = float32_to_bf16_rne(result);
            // FPU convention: 0 * x = 0 and x * 0 = 0 (handles NaN * 0 and Inf * 0 edge cases)
            v_if(in0 == 0 || in1 == 0) { result = 0.0f; }
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // Store result to DEST tile 0
        sfpi::dst_reg++; // Advance SFPU row pointer by SFP_DESTREG_STRIDE (2 rows)
    }
}
```

### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|--------------------------|-------------|
| `SFPLOAD` (via `dst_reg[idx]` read) | Loads a vector of 32 elements from the specified DEST register row into an SFPU LREG. Used twice per iteration to load `in0` and `in1` from their respective tile locations. |
| `SFPMUL` (via `in0 * in1`) | Performs element-wise floating-point multiplication of two SFPU vector registers, producing a 32-element result vector. This is the core operation -- a 2-cycle FP multiply. |
| `SFPSTORE` (via `dst_reg[idx] = result`) | Stores a vector of 32 elements from an SFPU LREG back to the specified DEST register row. |
| `SFPIADD` (via integer `bits + 0x7fffU + lsb`) | Integer addition used in the `float32_to_bf16_rne` helper for the RNE rounding bias. Only executed when `!is_fp32_dest_acc_en`. |
| `SFPSHFT` / `SFPSHFT2` (via `bits >> 16`) | Bit shift right by 16 to extract the bf16 mantissa LSB. Part of the RNE rounding helper. |
| `SFPAND` (via `bits & 0xFFFF0000U`, `(bits >> 16) & 1`) | Bitwise AND used for masking in the RNE rounding helper. |
| `SFPSETCC` / `SFPENCC` (via `v_if` / `v_endif`) | Condition code manipulation for the `v_if(in0 == 0 \|\| in1 == 0)` guard in the bf16 path. Sets CC based on comparison, then re-enables all lanes. |
| `SFPLOADI` (via immediate constants `0x7fffU`, `0xFFFF0000U`) | Loads immediate constant values into SFPU LREGs. Used for the RNE rounding constants. |
| `SFPMOV` (via `result = 0.0f` inside `v_if`) | Moves/loads a constant zero into the result register for lanes where either input is zero. |
| `TTI_SETRWC` | Set Read/Write Counter -- advances the DEST row pointer between faces. Not an SFPU instruction per se, but a math unit control instruction used in the parameters dispatch loop. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST[tile_0]** (rows 0..31) | Input tile A (`in0`), also used as the output location. Loaded via `copy_tile(cb_post_lhs, 0, 0)` before the SFPU kernel runs. After the kernel, contains the multiplication result. |
| **DEST[tile_1]** (rows 32..63) | Input tile B (`in1`). Loaded via `copy_tile(cb_post_rhs, 0, 1)` before the SFPU kernel runs. Read-only during the SFPU kernel. |
| **LREG0** | Primary working register. Holds `in0` after load, then `result` after multiplication. Also used for intermediate values in the bf16 RNE path (`bits`, `lsb`). |
| **LREG1** | Holds `in1` after load from DEST tile 1. |
| **LREG2-3** | Scratch registers used by the SFPI compiler for intermediate values in the bf16 RNE rounding path (e.g., holding the shifted/masked integer bits). Not used when `is_fp32_dest_acc_en=true`. |
| **DEST write pointer** | Managed by `TTI_SETRWC` in the parameters dispatch. Starts at row 0 of the output tile, advances by 16 between faces. Within the SFPU kernel, `dst_reg++` advances by `SFP_DESTREG_STRIDE=2` per iteration. |

### Address Mode Configuration

The ADDR_MOD configuration is set during initialization in `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()`, which calls `eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>()`.

For floating-point MUL (where `sfpu_op = SfpuType::unused`), only **ADDR_MOD_7** is configured:

| Field | Value | Description |
|-------|-------|-------------|
| `srca.incr` | 0 | No auto-increment on source A address |
| `srcb.incr` | 0 | No auto-increment on source B address |
| `dest.incr` | 0 | No auto-increment on DEST address |

ADDR_MOD_7 is chosen to avoid conflicts with ADDR_MOD_0 and ADDR_MOD_2 which are used by the A2D (copy_tile/unpack-to-DEST) pipeline. The zero-increment configuration means the SFPU kernel manages its own DEST addressing via `dst_reg++` (which increments the SFPU's internal row pointer) and the parameters dispatch manages face transitions via explicit `TTI_SETRWC` calls.

The `ADDR_MOD_6` variant (with `dest.incr = 2`) is only configured for integer multiply (`SfpuType::mul_int32`, `SfpuType::mul_uint16`) and min/max operations -- it is **not** used for floating-point MUL.

This configuration is identical between **Wormhole B0** and **Blackhole** -- both use the same `eltwise_binary_sfpu_configure_addrmod` template with identical field values. The only minor difference in the start/done functions is that Blackhole's `_llk_math_eltwise_binary_sfpu_start_` omits `math::set_addr_mod_base()` and `_llk_math_eltwise_binary_sfpu_done_` omits `math::clear_addr_mod_base()` and the `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU)` stall.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng operation work in TTNN? What are the different implementation paths (FPU vs SFPU), how is the path selected, and what kernels does it use?"
   **Reason**: Needed architectural context for the binary_ng framework before diving into source code.
   **Key Findings**: Confirmed the FPU/SFPU path selection mechanism, identified kernel file naming conventions (sfpu vs non-sfpu variants), and learned about the `OpConfig` class that generates compile-time defines.

2. [SFPU] **Query**: "Where is mul_binary_tile implemented? Trace the call chain from the compute kernel API through llk_math to the ckernel SFPU implementation for the binary multiplication SFPU operation."
   **Reason**: Needed to identify the exact file paths for each abstraction layer of the SFPU mul kernel.
   **Key Findings**: Confirmed the call chain: `mul_binary_tile` (API) -> `llk_math_eltwise_binary_sfpu_binop_mul` (LLK) -> `calculate_sfpu_binary_mul` (ckernel). Identified that the metal-specific `ckernel_sfpu_binary.h` in `tt_metal/hw/ckernels/` wraps and extends the tt_llk version with a dedicated `calculate_sfpu_binary_mul` function (separate from the generic `_calculate_sfpu_binary_` template).

3. [SFPU] **Query**: "How is mul_binary_tile implemented in the LLK layer? What SFPU instructions does the binary multiply kernel use?"
   **Reason**: Needed to understand the LLK dispatch mechanism and which SFPU instructions are emitted.
   **Key Findings**: Confirmed that `*` on `sfpi::vFloat` types corresponds to the `SFPMUL` instruction (2-cycle FP multiply). The `_llk_math_eltwise_binary_sfpu_params_` function handles face iteration with `VectorMode::RC` processing all 4 faces.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp` (lines 15-65)
   **Reason**: Needed to understand the exact conditions under which MUL selects the SFPU path.
   **Key Information**: MUL uses SFPU by default (when `fast_and_approximate_mode` is false), and conditionally for specific integer/float type combinations when approximate mode is enabled.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp` (lines 369-477)
   **Reason**: Needed to understand how SFPU binary operations are mapped to init/op functions.
   **Key Information**: `get_sfpu_init_fn()` maps `SfpuBinaryOp::MUL` to `mul_binary_tile_init()`/`mul_binary_tile` (float) or `mul_int_tile_init<>`/`mul_int_tile<>` (integer).

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp` (lines 81-129)
   **Reason**: Needed to understand kernel file path resolution for SFPU variants.
   **Key Information**: `get_kernel_file_path()` selects between `eltwise_binary_sfpu_no_bcast.cpp`, `eltwise_binary_sfpu.cpp`, and `eltwise_binary_sfpu_scalar.cpp` depending on broadcast type and `is_sfpu` flag.
