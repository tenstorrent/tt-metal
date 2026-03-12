# ADD (binary_ng) Implementation Analysis

## Overview

The ADD operation within the `binary_ng` framework performs element-wise addition of two tensors (or a tensor and a scalar). It is part of the unified `binary_ng` program factory, which handles all binary element-wise operations through a single configurable code path. The SFPU path is selected when input data types are FLOAT32, INT32, UINT32, or UINT16 (with both inputs matching), or when specific operation types mandate SFPU execution.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

## Path Selection: FPU vs SFPU

The `binary_ng` framework supports both FPU and SFPU execution paths, selected via the `is_binary_sfpu_op()` function in `binary_ng_device_operation.cpp` (line 15). For the ADD operation specifically, the SFPU path is selected when:

```
case ADD: return a == b && (a == FLOAT32 || a == INT32 || a == UINT32 || a == UINT16);
```

That is, ADD uses the SFPU path when **both inputs have the same data type** and that type is one of FLOAT32, INT32, UINT32, or UINT16. When the condition is not met (e.g., both inputs are BFLOAT16, or input types differ), the FPU path is used instead, which maps ADD to the FPU's native `ADD_tiles` operation via `OpConfig(op_type, std::in_place_type<OpConfig::FpuBinaryOp>, ...)`. The FPU path uses different compute kernels (`eltwise_binary_no_bcast.cpp`, `eltwise_binary.cpp`, `eltwise_binary_scalar.cpp`) that call `ADD_tiles` through the FPU matrix engine rather than the SFPU vector unit. The path flag `is_sfpu` is stored in the operation attributes and passed through to kernel file selection via `get_kernel_file_path()`.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `c.physical_volume() / (tile_height * tile_width)` -- total tiles in the output tensor |
| **Loop structure** | The compute kernel processes 1 tile per iteration (`num_tiles_per_cycle = 1`). The main loop runs `num_tiles` iterations, where `num_tiles` is the per-core tile count assigned at runtime. |

## Tensor Format and Layout

### Input Tensors

| Property | Input A | Input B (tensor) | Input B (scalar) |
|----------|---------|-------------------|-------------------|
| **Logical shape** | Up to 6+ dims, collapsed as [nD, D, N, C, Ht, Wt] | Same convention as A | Scalar value packed into uint32_t |
| **Dimension convention** | Last 5 dims: [..., D, N, C, H, W]; dims beyond 5 collapsed into nD | Same as A | N/A |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | N/A (single tile fill) |
| **Memory layout** | INTERLEAVED or SHARDED (height/width/block) | INTERLEAVED or SHARDED | N/A |
| **Buffer type** | DRAM or L1 | DRAM or L1 | N/A |
| **Data type** | FLOAT32, INT32, UINT32, UINT16 (SFPU path) | Same as A (SFPU path requires matching types for ADD) | Same as A dtype |

### Output Tensor

| Property | Output C |
|----------|----------|
| **Logical shape** | Broadcast-expanded shape of A and B |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Configurable (may differ from input; typecast post-activation inserted if needed) |

### Layout Transformations

- No explicit tilize/untilize within the operation -- inputs and outputs are expected in TILE_LAYOUT.
- If input and output data types differ and the operation is not a quantization op, a TYPECAST post-activation is automatically appended (line 604-609 in the program factory).
- Broadcasting is handled through stride-based index calculations: when a dimension has size 1 in one input, its stride is set to 0, causing the reader to re-read the same tile for that dimension.

## Data Flow Pattern

### Tensor-Tensor Path (b is a tensor)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader (RISC-V 0) | DRAM/L1 (A buffer) | CB c_0 | `cb_reserve_back(c_0, 1)` -> `noc_async_read_page` -> `cb_push_back(c_0, 1)` |
| 1b | Reader (RISC-V 0) | DRAM/L1 (B buffer) | CB c_1 | `cb_reserve_back(c_1, 1)` -> `noc_async_read_page` -> `cb_push_back(c_1, 1)` |
| 2 | Compute (Unpack + SFPU) | CB c_0, CB c_1 | CB c_2 | `cb_wait_front(c_0, 1)` + `cb_wait_front(c_1, 1)` -> `copy_tile` to DEST -> `BINARY_SFPU_OP` -> `pack_tile` -> `cb_push_back(c_2, 1)` -> `cb_pop_front(c_0/c_1, 1)` |
| 3 | Writer (RISC-V 1) | CB c_2 | DRAM/L1 (C buffer) | `cb_wait_front(c_2, 1)` -> `noc_async_write_page` -> `cb_pop_front(c_2, 1)` |

### Tensor-Scalar Path (b is a scalar)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 0 | Writer (RISC-V 1) | Scalar arg | CB c_1 | Fills one tile with scalar value once: `cb_reserve_back(c_1, 1)` -> `FILL_WITH_VALUE` -> `cb_push_back(c_1, 1)` |
| 1 | Reader (RISC-V 0) | DRAM/L1 (A buffer) | CB c_0 | `cb_reserve_back(c_0, 1)` -> `noc_async_read_page` -> `cb_push_back(c_0, 1)` per tile |
| 2 | Compute (Unpack + SFPU) | CB c_0, CB c_1 | CB c_2 | RHS tile read once; LHS tile read per iteration. `copy_tile` to DEST -> `BINARY_SFPU_OP` -> `pack_tile` -> `cb_push_back(c_2, 1)` -> `cb_pop_front(c_0, 1)` per tile |
| 3 | Writer (RISC-V 1) | CB c_2 | DRAM/L1 (C buffer) | `cb_wait_front(c_2, 1)` -> `noc_async_write_page` -> `cb_pop_front(c_2, 1)` |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|---------------------|-----------|----------|----------|----------|
| c_0 | cb_src_a | Input A staging | 2 (interleaved) or shard_volume (sharded) | 1 | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_src_b | Input B staging | 2 (tensor, interleaved), 1 (scalar), or shard_volume (sharded) | 1 | Double (tensor) / Single (scalar/sharded) | Reader or Writer (scalar) | Compute | Program |
| c_2 | cb_out | Output staging | 2 (interleaved) or shard_volume (sharded) | 1 | Double (interleaved) / Single (sharded) | Compute | Writer | Program |
| c_3 | cb_lhs_intermediate | LHS pre-processed activations | 1 | 1 | Single | Compute (preprocess) | Compute (main) | Block (conditional) |
| c_4 | cb_rhs_intermediate | RHS pre-processed activations | 1 | 1 | Single | Compute (preprocess) | Compute (main) | Block (conditional) |
| c_5 | cb_row_bcast_a | Row broadcast staging for A | 2 | 1 | Double | Reader | Compute | Program (conditional) |
| c_6 | cb_row_bcast_b | Row broadcast staging for B | 2 | 1 | Double | Reader | Compute | Program (conditional) |

**Notes**:
- CBs c_3 and c_4 are only created when LHS/RHS pre-activations are non-empty. For plain ADD, these are not created since ADD has no `process_lhs` or `process_rhs`.
- CBs c_5 and c_6 are only created for row broadcast scenarios (SubtileBroadcastType::ROW_A, ROW_B, ROW_A_COL_B, ROW_B_COL_A).
- For the SFPU path, intermediate data format is the same as input data format (unlike FPU which may convert to Float16_b for exp-based ops).

## Pipeline Pattern Summary

- **Interleaved mode**: CBs c_0, c_1, c_2 each have capacity=2 with block_size=1, enabling **double-buffered** operation. The reader can write the next tile while the compute kernel processes the current one, and the compute kernel can write the next result while the writer drains the previous one.
- **Sharded mode**: CBs are sized to hold the entire shard (capacity=shard_volume, block_size=shard_volume), making them effectively **single-buffered** -- the entire shard is loaded at once.
- **Scalar path**: CB c_1 holds exactly 1 tile (the filled scalar), which is read repeatedly without popping until all tiles are processed.

## Index Calculations

The operation supports tensors up to 6+ dimensions through a collapsed dimension scheme:
- Dimensions beyond 5 are collapsed into a single `nD` dimension.
- The logical shape is decomposed into `[nD, D, N, C, Ht, Wt]` where Ht and Wt are tile counts.
- Broadcasting is implemented through **stride-based indexing**: each dimension's stride is computed as `dim_tiles * (dim > 1)`. When a dimension has size 1, its stride becomes 0, causing the same tiles to be re-read across that dimension.

The reader kernel computes tile offsets using:
```
tile_offset = start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride + start_th * Wt
```

TensorAccessor is used for physical page-to-bank mapping. The `TensorAccessorArgs` are split into compile-time args (bank/page configuration) and common runtime args (tensor shape info), appended to kernel arguments via `append_to()`.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Tiles are read one at a time via `noc_async_read_page()` with an immediate barrier (`noc_async_read_barrier()`). The 6-nested loop iterates in row-major order: nD -> D -> N -> C -> Ht -> Wt. Each tile read is followed by a barrier, yielding sequential single-tile reads.
- **Sharded**: The entire shard is made available at once via `cb_reserve_back(cb_id_src, src_num_tiles); cb_push_back(...)` -- no NoC reads needed since data is already in L1.

### Write Pattern
- **Interleaved**: Tiles are written one at a time via `noc_async_write_page()` with an immediate barrier. Same 6-nested loop order as the reader.
- **Sharded**: Output CB is backed by the sharded output buffer directly; no explicit writes needed.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (when zero_start_grid) or arbitrary CoreRangeSet |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` (typically device compute grid) |
| **Total cores** | `compute_with_storage_grid.x * compute_with_storage_grid.y` (zero-start) or `all_device_cores.num_cores()` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles (interleaved); shard tile count (sharded) |
| **Load balancing** | Two-group split: `core_group_1` gets `ceil(total_tiles / num_cores)` tiles, `core_group_2` gets one fewer. Cores outside both groups are assigned 0 tiles (noop). |

The `split_work_to_cores()` utility divides total output tiles across available cores. For sharded mode, each core processes exactly its shard. The `zero_start_grid` optimization applies when the core grid is a single rectangle starting at (0,0) -- this enables faster `grid_to_cores` algorithms.

## Arguments

### Compile-Time Arguments

**Compute Kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Tiles produced per compute cycle (always 1) |

**Reader Kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs (A) | uint32_t[] | Compile-time tensor accessor args for input A |
| N+1..M | TensorAccessorArgs (B) | uint32_t[] | Compile-time tensor accessor args for input B |
| M+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

**Writer Kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs (C) | uint32_t[] | Compile-time tensor accessor args for output C |
| N+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

### Runtime Arguments

**Reader Kernel (tensor-tensor path, 21 args):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input A buffer address |
| 1 | start_tile_id | uint32_t | Starting output tile ID for this core |
| 2 | src_num_tiles (A) | uint32_t | Number of A shard tiles (sharded only, else 0) |
| 3 | dst_num_tiles | uint32_t | Number of output tiles to process on this core |
| 4 | dst_shard_width | uint32_t | Shard width in tiles (sharded only, else 0) |
| 5 | nD_stride | uint32_t | A's stride for collapsed nD dimension |
| 6 | d_stride | uint32_t | A's stride for D dimension |
| 7 | n_stride | uint32_t | A's stride for N dimension |
| 8 | c_stride | uint32_t | A's stride for C dimension |
| 9 | D | uint32_t | Output D dimension |
| 10 | N | uint32_t | Output N dimension |
| 11 | C | uint32_t | Output C dimension |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Output collapsed nD dimension |
| 15 | src_addr_b | uint32_t | Input B buffer address |
| 16 | nD_stride_b | uint32_t | B's stride for collapsed nD dimension |
| 17 | d_stride_b | uint32_t | B's stride for D dimension |
| 18 | n_stride_b | uint32_t | B's stride for N dimension |
| 19 | c_stride_b | uint32_t | B's stride for C dimension |
| 20 | src_num_tiles_b | uint32_t | Number of B shard tiles (sharded only, else 0) |

**Writer Kernel (tensor-tensor path, 11 args):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output C buffer address |
| 1 | start_tile_id | uint32_t | Starting output tile ID |
| 2 | dst_num_tiles | uint32_t | Number of output tiles for this core |
| 3 | dst_shard_width | uint32_t | Shard width in tiles (sharded only) |
| 4 | D | uint32_t | Output D dimension |
| 5 | N | uint32_t | Output N dimension |
| 6 | C | uint32_t | Output C dimension |
| 7 | Ht | uint32_t | Output height in tiles |
| 8 | Wt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Output collapsed nD dimension |
| 10 | (reserved) | uint32_t | Set to 0 |

**Writer Kernel (scalar path, 11 args):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar | uint32_t | Scalar value packed into uint32 (bf16 pair or float/int bits) |
| 1 | dst_addr | uint32_t | Output C buffer address |
| 2 | start_tile_id | uint32_t | Starting output tile ID |
| 3 | dst_num_tiles | uint32_t | Number of output tiles |
| 4 | dst_shard_width | uint32_t | Shard width in tiles |
| 5-10 | D, N, C, Ht, Wt, cND | uint32_t | Output shape dimensions |

**Compute Kernel (4 args):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Number of tiles to process on this core |
| 1 | freq | uint32_t | Broadcast frequency (1 for NONE/ROW; Ht*Wt for SCALAR; Wt for COL) |
| 2 | counter | uint32_t | Start offset within broadcast cycle |
| 3 | compute_scalar_value | uint32_t | Quantization zero-point or WHERE scalar (0 for plain ADD) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | RISC-V 0 | NOC0 | DRAM/L1 (A, B) | CB c_0, CB c_1 | Read A and B tiles via TensorAccessor |
| Compute | RISC-V 2 (Unpack + SFPU + Pack) | N/A | CB c_0 (LHS), CB c_1 (RHS) | CB c_2 | `copy_tile` to DEST, `add_binary_tile` (SFPU), `pack_tile` |
| Writer | RISC-V 1 | NOC1 | CB c_2 (or scalar fill to CB c_1) | DRAM/L1 (C) | Write output tiles via TensorAccessor |

### Reader Kernel

| Property | Value |
|----------|-------|
| **File (tensor-tensor)** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` |
| **File (scalar)** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp` |
| **Assigned cores** | All worker cores |

**Key Logic:**
- For **interleaved** tensors, iterates a 6-deep nested loop (nD, D, N, C, Ht, Wt) reading one tile at a time with `noc_async_read_page()` followed by `noc_async_read_barrier()`.
- For **sharded** tensors, simply calls `cb_reserve_back(cb_id, shard_tiles); cb_push_back(cb_id, shard_tiles)` to make the L1-resident shard available -- no NoC reads.
- In the tensor-tensor variant, reads from both A (CB c_0) and B (CB c_1) are interleaved within the same tile loop iteration, sharing a single `noc_async_read_barrier()`.
- Broadcasting is achieved through stride arguments: when a dimension has size 1 in input A or B, its stride is 0, so `tile_offset` does not advance along that dimension, effectively re-reading the same data.
- **Synchronization**: Pushes tiles into CB c_0 and CB c_1 via `cb_push_back`. The compute kernel consumes them with `cb_wait_front`.

### Compute Kernel (SFPU -- No Broadcast)

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` |
| **Assigned cores** | All worker cores |

**Key Logic:**
- The main loop iterates `num_tiles` times (one output tile per iteration).
- For each tile: waits on both LHS (CB c_0) and RHS (CB c_1) with `cb_wait_front`, reserves output space in CB c_2 with `cb_reserve_back`.
- Acquires tile registers via `tile_regs_acquire()`.
- Unpacks LHS tile to DEST register slot `i*2` using `copy_tile(cb_post_lhs, i, i*2)`, then unpacks RHS tile to slot `i*2+1` using `copy_tile(cb_post_rhs, i, i*2+1)`.
- The `copy_tile_to_dst_init_short_with_dt()` calls configure the unpacker for the correct data format when switching between LHS and RHS CBs.
- Executes `BINARY_SFPU_OP(i*2, i*2+1, i*2)` which expands to `add_binary_tile(i*2, i*2+1, i*2)` for ADD. This runs the SFPU vector add on the two DEST tiles, storing the result back to DEST slot `i*2`.
- Applies `PROCESS_POST_ACTIVATIONS(i*2)` if any post-activations are defined (empty for plain ADD).
- Commits tile registers, waits for them, packs result from DEST to CB c_2 via `pack_tile(i*2, cb_out)`, then releases registers.
- **Synchronization**: `cb_wait_front(c_0, 1)` and `cb_wait_front(c_1, 1)` block until reader has pushed tiles. After processing, `cb_pop_front(c_0, 1)` and `cb_pop_front(c_1, 1)` free input slots. `cb_push_back(c_2, 1)` signals the writer.
- For ADD, `BINARY_SFPU_INIT` expands to `add_binary_tile_init();` which is called once before the loop (when no pre/post activations are present).

### Compute Kernel (SFPU -- Scalar)

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp` |
| **Assigned cores** | All worker cores |

**Key Logic:**
- RHS tile (the scalar-filled tile in CB c_1) is waited on **once before the loop** with `cb_wait_front(cb_post_rhs, 1)`.
- The main loop processes only LHS tiles -- waits on CB c_0 each iteration but reuses the single RHS tile.
- Same DEST register layout: LHS at slot `i*2`, RHS at slot `i*2+1`, result at slot `i*2`.
- After all tiles are processed, the scalar tile is popped from RHS CB with `cb_pop_front(cb_post_rhs, 1)`.

### Compute Kernel (SFPU -- Broadcast)

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu.cpp` |
| **Assigned cores** | All worker cores |

**Key Logic:**
- Used for column broadcast, scalar broadcast (tile-level), and mixed broadcast scenarios.
- Takes `freq` (broadcast frequency) and `tile_start` (start offset) as runtime args.
- The broadcast operand is loaded once per `freq` iterations; the non-broadcast operand is loaded every iteration.
- `process_tile()` handles one broadcast cycle: loads the broadcast operand, then iterates `freq` times over the other operand.
- `BCAST_INPUT` compile-time define (0 or 1) selects which operand is broadcast.

### Writer Kernel (Tensor-Tensor)

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` |
| **Assigned cores** | All worker cores |

**Key Logic:**
- For interleaved output, iterates the same 6-deep nested loop as the reader, writing one tile at a time via `noc_async_write_page()` with an immediate barrier.
- For sharded output, the writer body is entirely compiled out (`#if !DST_SHARDED` guards), as output data is already in the correct L1 location via the CB-backed shard buffer.
- **Synchronization**: `cb_wait_front(c_2, 1)` blocks until compute has pushed a result. `cb_pop_front(c_2, 1)` frees the slot after writing.

### Writer Kernel (Scalar)

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp` |
| **Assigned cores** | All worker cores |

**Key Logic:**
- Performs dual duty: first fills CB c_1 with one tile containing the packed scalar value, then writes output tiles from CB c_2.
- The scalar fill uses `FILL_WITH_VALUE_FLOAT` or `FILL_WITH_VALUE` depending on data type, writing 1024 elements (32x32 tile) of the scalar value.
- Output writing follows the same 6-deep loop as the tensor-tensor writer.

## Implementation Notes

- **Program factory variants**: There is a single `BinaryNgDeviceOperation::ProgramFactory` that handles all binary_ng operations. The kernel variant is selected based on `SubtileBroadcastType` (NONE, SCALAR_A/B, ROW_A/B, COL_A/B, ROW_B_COL_A, ROW_A_COL_B) and whether `b` is a tensor or scalar. The factory also has an `override_runtime_arguments()` method for program caching.

- **Type-based operation variants**: SFPU ADD supports FLOAT32 (via `add_binary_tile`), INT32 (via `add_int_tile<DataFormat::Int32>`), UINT32 (via `add_int_tile<DataFormat::UInt32>`), and UINT16 (via `add_int_tile<DataFormat::UInt16>`). The init/op function pair is selected in `get_sfpu_init_fn()` (line 376-381 of `binary_ng_utils.cpp`).

- **UnpackToDestFP32 mode**: For SFPU operations (except POWER), all input CBs (c_0, c_1, c_3, c_4) are configured with `UnpackToDestMode::UnpackToDestFp32`. This ensures data is unpacked to FP32 precision in the DEST registers before SFPU execution (lines 740-755 of the program factory).

- **Broadcast type selection**: The `SubtileBroadcastType` determines which kernel variants are used. For standard ADD of same-shaped tensors, NONE is used. Broadcasting is stride-based for dimension broadcasts (ROW, COL, SCALAR). The `is_llk_bcast` function checks if LLK-level broadcast can be used (only for BFLOAT16 row/row-col broadcasts, which is never on the SFPU path since SFPU requires FLOAT32/INT32/UINT32/UINT16).

- **Sharding support and constraints**: Height, width, and block sharding are all supported. Native L1 sharding is used when: (1) the shard grid and output grid align, (2) both inputs have identical shapes and memory configs, (3) tensors are not uneven (dimensions evenly divisible by shard shape), and (4) no DRAM buffers are involved. When native sharding cannot be used, the operation falls back to treating tensors as interleaved with TensorAccessor-based indexing.

- **FP32 dest accumulation**: Enabled when output format is UInt32, Int32, or Float32, or when both inputs are Float32, Int32, or UInt32 (line 727-731). This ensures full precision is maintained through the DEST accumulation register.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_binary.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `add_binary_tile(i*2, i*2+1, i*2)` (defined in `eltwise_binary_sfpu.h`), which wraps `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>(idst0, idst1, odst)` inside the `MATH` macro.
2. `llk_math_eltwise_binary_sfpu_binop` (in `llk_math_eltwise_binary_sfpu_binop.h`) calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>()` with a function pointer to `ckernel::sfpu::calculate_sfpu_binary<APPROXIMATE, BinaryOp::ADD, 8, is_fp32_dest_acc_en>`.
3. `_llk_math_eltwise_binary_sfpu_params_` (in `llk_math_eltwise_binary_sfpu_params.h`) calls `_llk_math_eltwise_binary_sfpu_start_` to set the DEST write address and stall until SFPU is ready, then iterates over tile faces in the selected `VectorMode`, calling the SFPU function once per face, with `TTI_SETRWC` instructions to advance the DEST pointer between faces, and finally calls `_llk_math_eltwise_binary_sfpu_done_` to clear DEST address state.
4. `calculate_sfpu_binary` (in `ckernel_sfpu_binary.h`, metal layer) delegates directly to `_calculate_sfpu_binary_<APPROXIMATION_MODE, BinaryOp::ADD, 8>()`.
5. `_calculate_sfpu_binary_` (in `ckernel_sfpu_binary.h`, tt_llk layer) is the core SFPU microcode: it loops 8 iterations, loading one row from each input tile in DEST, computing `in0 + in1`, writing the result, and advancing `dst_reg++`.

For initialization: `add_binary_tile_init()` calls `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::ADD>()`, which calls `llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu_binary_init<APPROXIMATE, BinaryOp::ADD>)`. This runs `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` (configures SFPU config register, sets ADDR_MOD_7, resets counters) followed by `_sfpu_binary_init_<APPROXIMATE, BinaryOp::ADD>()` which is a no-op for ADD (no special init needed since ADD does not use reciprocal or log).

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (the default). All 4 faces of the 32x32 tile are processed. Each face contains 8 rows of the SFPU's 32-wide vector. The 4 faces cover the full tile: face 0 = top-left 16x16, face 1 = top-right 16x16, face 2 = bottom-left 16x16, face 3 = bottom-right 16x16.
- **Operation invocation**: The core SFPU function `calculate_sfpu_binary` is called once per face (4 times total per tile). Each call internally loops 8 iterations (`ITERATIONS=8`), processing one row per iteration. So 4 faces x 8 rows = 32 rows total, covering the full 32x32 tile (each row is 32 elements wide across the SFPU's vector datapath).
- **DEST address progression**: Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice (incrementing by 8+8=16 DEST rows per face). Within each face, `dst_reg++` in the SFPU microcode advances the DEST read/write pointer by 1 row (stride of `SFP_DESTREG_STRIDE=2` in hardware units) per iteration.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h
// (Blackhole version is identical)

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{   // For ADD: APPROXIMATION_MODE=true (via APPROX), BINOP=BinaryOp::ADD, ITERATIONS=8
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) // 8 iterations = 8 rows per face
    {
        // Each tile occupies 32 sfpi rows in DEST (64 HW rows / SFP_DESTREG_STRIDE=2)
        constexpr std::uint32_t dst_tile_size_sfpi = 32;
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // Load row from input tile 0 in DEST
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // Load row from input tile 1 in DEST
        sfpi::vFloat result = 0.0f; // Initialize result (overwritten immediately for ADD)

        if constexpr (BINOP == BinaryOp::ADD)
        {
            result = in0 + in1; // SFPU vector addition: compiles to SFPADD instruction
        }
        // Other BINOP branches (SUB, MUL, DIV, RSUB, POW, XLOGY) elided for ADD

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // Store result row to output tile in DEST
        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (SFP_DESTREG_STRIDE HW rows)
    }
}
```

### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|--------------------------|-------------|
| `SFPLOAD` (via `dst_reg[offset]` read) | Loads a 32-wide vector from a DEST register row into an SFPU register (vFloat). The offset `dst_index * 32` selects which tile's data to read. |
| `SFPADD` (via `in0 + in1`) | Performs element-wise floating-point addition of two 32-wide SFPU vector registers. This is the core operation for the ADD kernel. |
| `SFPLOADI` (via `result = 0.0f`) | Loads an immediate constant (0.0f) into an SFPU register. In practice this initialization is dead code for the ADD path since `result` is immediately overwritten, but the compiler may or may not optimize it away. |
| `SFPSTORE` (via `dst_reg[offset] = result`) | Stores a 32-wide vector from an SFPU register back to a DEST register row. |
| `INCRWC` (via `dst_reg++`) | Increments the DEST register read/write counter by `SFP_DESTREG_STRIDE` (2 hardware rows = 1 sfpi row), advancing to the next row within the current face. |
| `TTI_SETRWC` (in params dispatch) | Sets the DEST read/write counter. Used between faces to advance by 16 DEST rows (2 calls incrementing by 8 each). |
| `TTI_STALLWAIT` (in start/done) | Stalls the MATH pipeline until the SFPU is ready (start) or until SFPU completes (done, Wormhole only). |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST registers** | Three tiles are resident in DEST simultaneously: input tile 0 at slot `idst0` (e.g., slot 0), input tile 1 at slot `idst1` (e.g., slot 1), and the output tile at slot `odst` (e.g., slot 0, same as input 0 -- result overwrites in0). Each tile occupies 32 sfpi rows (64 HW rows). |
| **SFPU LREGs (L0-L7)** | The compiler allocates `in0`, `in1`, and `result` to SFPU local registers. Since the ADD kernel body is simple (3 variables), it fits within the 8 available LREGs without spilling. |
| **SFPU CC (Condition Code)** | Not used by the ADD path. No `v_if`/`v_endif` or `SFPSETCC`/`SFPENCC` instructions are involved. |
| **Programmable constants** | Not used for ADD. The `_sfpu_binary_init_` function is a no-op for `BinaryOp::ADD` -- no LREG constants are loaded. |

### Address Mode Configuration

The SFPU binary operation configures **ADDR_MOD_7** during initialization (`eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>()`):

| Field | Value | Description |
|-------|-------|-------------|
| `srca.incr` | 0 | No auto-increment for SRC_A address |
| `srcb.incr` | 0 | No auto-increment for SRC_B address |
| `dest.incr` | 0 | No auto-increment for DEST address |

This configuration is identical across **Wormhole B0** and **Blackhole**. The ADDR_MOD_7 slot is chosen specifically to avoid conflicts with A2D (unpack-to-dest) operations, which use ADDR_MOD_0 and ADDR_MOD_2. Since `SfpuType::unused` does not match any of the special types (mul_int32, mul_uint16, max, min, etc.), ADDR_MOD_6 is not configured.

DEST addressing within the SFPU kernel itself is managed entirely through explicit `dst_reg[offset]` indexing and `dst_reg++` increments, not through hardware auto-increment. Between faces, the params dispatch uses `TTI_SETRWC` to manually advance the DEST counter.

**Minor difference between Wormhole B0 and Blackhole**: In the `_llk_math_eltwise_binary_sfpu_start_` function, Wormhole calls `math::set_addr_mod_base()` (and `clear_addr_mod_base()` in `_done_`), while Blackhole omits these calls. Additionally, Wormhole's `_done_` function includes a `TTI_STALLWAIT(STALL_CFG, WAIT_SFPU)` to wait for SFPU completion before clearing state, while Blackhole omits this stall.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng operation work in TTNN? What are the different implementation paths (FPU vs SFPU), how is the path selected, and what kernels does it use?"
   **Reason**: Needed to understand the overall architecture and path selection mechanism before reading source code.
   **Key Findings**: Confirmed that path selection happens via `is_binary_sfpu_op()`, identified the kernel naming scheme (sfpu vs non-sfpu variants), and learned that SFPU operations use `UnpackToDestFp32` mode. The program factory is unified for all binary_ng operations.

2. [SFPU] **Query**: "How does the add_binary_tile SFPU function work in the binary_ng operation? Trace the call chain from the compute kernel API through the LLK layer down to the core ckernel SFPU implementation."
   **Reason**: Needed to identify the full call chain from the compute API to the underlying SFPU microcode, including all intermediate file paths.
   **Key Findings**: Confirmed the 5-layer call chain: `add_binary_tile` -> `llk_math_eltwise_binary_sfpu_binop` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_sfpu_binary` -> `_calculate_sfpu_binary_`. Identified all file paths for Wormhole and Blackhole architectures.

3. [SFPU] **Query**: "How is add_binary_tile implemented in the LLK layer? What is the call chain from the compute API through llk_math to the ckernel SFPU add function? What SFPU instructions does it use?"
   **Reason**: Needed LLK-specific details on the SFPU instruction mapping and the params dispatch mechanism.
   **Key Findings**: Confirmed that `SFPADD` is the core hardware instruction, learned about the `VectorMode::RC` face iteration pattern, and identified the `_sfpu_binary_init_` function as a no-op for ADD.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp` (lines 15-64)
   **Reason**: Contains the `is_binary_sfpu_op()` function that determines FPU vs SFPU path selection.
   **Key Information**: ADD uses SFPU when `a == b && (a == FLOAT32 || a == INT32 || a == UINT32 || a == UINT16)`.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp` (lines 376-381)
   **Reason**: Contains `get_sfpu_init_fn()` which maps SfpuBinaryOp::ADD to the actual LLK function calls.
   **Key Information**: ADD maps to `add_binary_tile_init()` / `add_binary_tile` for float types, and `add_int_tile_init()` / `add_int_tile<DataFormat::T>` for integer types.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp` (lines 81-129)
   **Reason**: Contains `get_kernel_file_path()` which maps kernel enum names to actual file paths.
   **Key Information**: The kernel file selection depends on both `is_sfpu` and `is_where_op` flags, branching to `eltwise_binary_sfpu_*.cpp` files for the SFPU path.

### Confluence References

No Confluence references were needed for this analysis. The ADD kernel is straightforward (single `SFPADD` instruction) and DeepWiki provided sufficient detail on the instruction semantics.

### Glean References

No Glean references were needed for this analysis.
