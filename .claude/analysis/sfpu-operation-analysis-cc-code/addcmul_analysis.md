# ADDCMUL Implementation Analysis

## Overview

ADDCMUL computes `output = input_a + scalar * input_b * input_c` element-wise across three input tensors and a scalar multiplier. This is a ternary operation that combines addition and conditional multiplication. The operation is implemented through the shared ternary program factory at:

`ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_program_factory.cpp`

ADDCMUL exclusively uses the **TTT (Tensor-Tensor-Tensor)** variant, meaning all three operands are tensors (the scalar is passed as a runtime argument to the compute kernel, not as a separate tensor input).

## Path Selection: FPU vs SFPU

The program factory determines whether to use the FPU or SFPU path at lines 1059-1067 of `ternary_program_factory.cpp`. The `is_fpu` flag is set based on the following conditions:

```
is_fpu = (predicate_tensor.dtype() == value_true_tensor.dtype()) &&
         (predicate_tensor.dtype() == value_false_tensor.dtype()) &&
         (predicate_tensor.dtype() != DataType::FLOAT32 &&
          predicate_tensor.dtype() != DataType::INT32 &&
          predicate_tensor.dtype() != DataType::UINT32);
```

The **FPU path** is selected only when ALL three input tensors have the same data type AND that data type is NOT FLOAT32, INT32, or UINT32 (i.e., it is BF16). The FPU path uses `ternary_addc_ops_fpu.cpp` which leverages the FPU's `mul_tiles` and `binary_dest_reuse_tiles` instructions for hardware-accelerated multiply-add.

The **SFPU path** is selected when:
- Any input tensor has a different data type from the others, OR
- The common data type is FLOAT32, INT32, or UINT32

Additionally, when `dtype == DataType::INT32`, a specialized INT32 SFPU kernel (`ternary_addcmul_int_sfpu.cpp` or `ternary_addcmul_int_sfpu_bcast.cpp`) overrides the standard SFPU kernel via `override_addcmul_compute_kernel()`.

The `is_fpu` flag affects kernel file path resolution via `get_kernel_file_path(kernel_name, is_fpu)`, which selects between FPU and SFPU kernel source files for reader, compute, and writer.

## Work Unit Definition

One work unit is a single **tile** (32x32 elements). The compute kernel processes exactly `num_tiles_per_cycle = 1` output tile per iteration: it reads one tile from each of the three input CBs, performs the addcmul computation in DST registers, and packs one tile to the output CB.

## Tensor Format and Layout

### Input Tensors

| Property | Input A (Predicate/input_a) | Input B (value_true/input_b) | Input C (value_false/input_c) |
|---|---|---|---|
| Dimension Convention | ND, D, N, C, Ht, Wt (up to rank 6+) | Same as Input A | Same as Input A |
| Tensor Layout | TILE (32x32) | TILE (32x32) | TILE (32x32) |
| Memory Layout | Interleaved or Sharded | Interleaved or Sharded | Interleaved or Sharded |
| Buffer Type | DRAM or L1 | DRAM or L1 | DRAM or L1 |
| Data Type | BF16, FP32, INT32, UINT32 | BF16, FP32, INT32, UINT32 | BF16, FP32, INT32, UINT32 |

### Output Tensor

| Property | Output |
|---|---|
| Dimension Convention | Same as broadcast-resolved output shape |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Interleaved or Sharded |
| Buffer Type | DRAM or L1 |
| Data Type | Matches operation dtype attribute |

### Layout Transformations

No explicit tilize/untilize operations are performed. All tensors are expected in tiled format. When broadcasting is required, the reader kernel handles sub-tile replication (column fill, row fill, scalar fill) in L1 before pushing to the compute kernel's input CBs.

## Data Flow Pattern

**Interleaved (no-broadcast) path:**

1. **Reader kernel** iterates through tiles assigned to its core using a 6-level nested loop (ND, D, N, C, Ht, Wt). For each tile:
   - Issues `noc_async_read_page` for the corresponding tile from each of the three input tensors (using TensorAccessor for address resolution)
   - Waits for NoC read completion (`noc_async_read_barrier`)
   - Pushes one tile to each of `cb_in0`, `cb_in1`, `cb_in2`
2. **Compute kernel** processes one tile per iteration:
   - Waits for one tile on each of `cb_in0`, `cb_in1`, `cb_in2`
   - Copies all three tiles into DST registers (DST[0], DST[1], DST[2])
   - Executes `addcmul_tile(DST[0], DST[1], DST[2], DST[0], scalar)` via SFPU
   - Packs DST[0] into `cb_out`
   - Pops one tile from each input CB
3. **Writer kernel** iterates through output tiles:
   - Waits for one tile on `cb_out`
   - Issues `noc_async_write_page` to write the tile to DRAM/L1
   - Pops the tile from `cb_out`

**Sharded path:** When tensors are L1-sharded, the reader kernel does not perform NoC reads. Instead, it issues `cb_reserve_back` / `cb_push_back` on the sharded input CBs (which are already mapped to L1 shard buffers). The compute and writer paths remain unchanged. For sharded output, the writer kernel becomes a no-op (the output CB is already the shard buffer).

## Circular Buffer Configuration

### No-Broadcast / Outer-Broadcast Case

| CB ID | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|---|---|---|---|---|---|---|---|
| c_0 | Input A (predicate/input_a) | 2 (or shard volume) | 1 | Double-buffered | Reader | Compute | Full kernel |
| c_1 | Input B (value_true/input_b) | 2 (or shard volume) | 1 | Double-buffered | Reader | Compute | Full kernel |
| c_2 | Input C (value_false/input_c) | 2 (or shard volume) | 1 | Double-buffered | Reader | Compute | Full kernel |
| c_3 | Output | 2 (or shard volume) | 1 | Double-buffered | Compute | Writer | Full kernel |

### Row-Broadcast Case (additional CBs)

When `broadcast_type == ROW_BCAST` and variant is TTT, additional CBs c_4, c_5, c_6 are created:

| CB ID | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|---|---|---|---|---|---|---|---|
| c_4 | Row-broadcast scratch for input A | 2 | 1 | Double-buffered | Reader | Compute | Full kernel |
| c_5 | Row-broadcast scratch for input B | 2 | 1 | Double-buffered | Reader | Compute | Full kernel |
| c_6 | Row-broadcast scratch for input C | 2 | 1 | Double-buffered | Reader | Compute | Full kernel |

## Pipeline Pattern Summary

In the interleaved (non-sharded) configuration, all CBs have capacity of 2 tiles with a block size of 1 tile, providing **double-buffering**. This allows the reader to fill one tile slot while the compute kernel processes the other, enabling overlap between data movement and computation.

In sharded configurations, CB capacity equals the full shard volume, making them effectively **single-buffered** -- the entire shard is loaded once and consumed from L1 without streaming.

## Index Calculations

The reader kernel maps a linear `start_tile_id` to multi-dimensional coordinates using the output tensor dimensions:

```
tiles_per_nd = D * N * C * Ht * Wt
start_nd = start_tile_id / tiles_per_nd
start_d  = (start_tile_id % tiles_per_nd) / tiles_per_d
start_n  = ... / tiles_per_n
start_c  = ... / HtWt
start_th = ... / Wt
start_tw = ... % Wt
```

For each input tensor, a separate `tile_offset` is computed using per-tensor strides. The stride for dimension X is nonzero only when that dimension is greater than 1 in the input tensor (broadcasting is implicit: stride=0 means the same data is reused). The stride values are:
- `nD_stride = Ht * Wt * C * N * D * (ND > 1)`
- `d_stride = Ht * Wt * C * N * (D > 1)`
- `n_stride = Ht * Wt * C * (N > 1)`
- `c_stride = Ht * Wt * (C > 1)`

TensorAccessor is used for physical address resolution, converting a logical tile index to the actual DRAM/L1 bank address.

## Memory Access Patterns

### Read Pattern

The reader kernel iterates tiles in row-major order within each Ht x Wt tile block (innermost loops are `th` then `tw`), with outer loops over C, N, D, ND dimensions. For each tile, three independent NoC reads are issued (one per input tensor) followed by a barrier. This is a **sequential tile-by-tile** pattern with three concurrent reads per tile.

For width-sharded tensors, the `end_tw` is limited to `start_tw + dst_shard_width`, meaning only a subset of columns are read per row.

### Write Pattern

The writer kernel uses the same nested loop structure as the reader, writing one tile at a time via `noc_async_write_page` with an immediate barrier. This is also a **sequential tile-by-tile** pattern. Width-sharded writes adjust `dst_tile_offset` to account for skipped columns.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | 2D rectangular grid from `operation_attributes.worker_grid` |
| Work Splitting | `split_work_to_cores(grid, num_output_tiles, row_major)` |
| Load Balancing | Two-group split: core_group_1 gets `ceil(tiles/cores)` tiles, core_group_2 gets `floor(tiles/cores)` |
| Remainder Handling | Remainder tiles assigned to core_group_1 (first N cores get one extra tile) |
| Idle Core Handling | Cores outside core_group_1 and core_group_2 receive zero-initialized runtime args and skip processing |
| Sharded Mode | Core grid comes from shard spec; each core processes its local shard |

For sharded tensors, the ShardShapeGenerator computes per-core shard shapes, accounting for edge cores that may have fewer tiles. The `c_start_id` for each core in sharded mode is calculated as:
```
c_start_id = (core_index / shards_per_width) * (shard_height * output_Wt) + (core_index % shards_per_width) * shard_width
```

## Arguments

### Compile-Time Arguments

**Reader Kernel (TTT variant):**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_src0 | uint32_t | CB index for input A (predicate tensor) |
| 1 | cb_id_src1 | uint32_t | CB index for input B (true tensor) |
| 2 | cb_id_src2 | uint32_t | CB index for input C (false tensor) |
| 3+ | TensorAccessorArgs (src0) | varies | Compile-time args for input A tensor accessor |
| N+ | TensorAccessorArgs (src1) | varies | Compile-time args for input B tensor accessor |
| M+ | TensorAccessorArgs (src2) | varies | Compile-time args for input C tensor accessor |
| last | has_sharding | uint32_t | Whether sharding is active (0 or 1) |

**Compute Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 -- tiles processed per iteration |
| 1 | scalar_is_true_value | uint32_t | Always 0 for ADDCMUL TTT (used by TST variant) |

**Writer Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_out | uint32_t | CB index for output tensor |
| 1+ | TensorAccessorArgs (dst) | varies | Compile-time args for output tensor accessor |
| last | has_sharding | uint32_t | Whether sharding is active (0 or 1) |

### Runtime Arguments

**Reader Kernel (27 args):**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src0_addr | uint32_t | DRAM/L1 address of input A buffer |
| 1 | src1_addr | uint32_t | DRAM/L1 address of input B buffer |
| 2 | src2_addr | uint32_t | DRAM/L1 address of input C buffer |
| 3 | num_tiles | uint32_t | Number of tiles assigned to this core |
| 4 | start_tile_id | uint32_t | Starting tile index for this core |
| 5 | nD_stride | uint32_t | Input A stride for ND dimension |
| 6 | d_stride | uint32_t | Input A stride for D dimension |
| 7 | n_stride | uint32_t | Input A stride for N dimension |
| 8 | c_stride | uint32_t | Input A stride for C dimension |
| 9 | D | uint32_t | Output D dimension |
| 10 | N | uint32_t | Output N dimension |
| 11 | C | uint32_t | Output C dimension |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Output ND dimension (collapsed higher dims) |
| 15 | nD_stride_b | uint32_t | Input B stride for ND dimension |
| 16 | d_stride_b | uint32_t | Input B stride for D dimension |
| 17 | n_stride_b | uint32_t | Input B stride for N dimension |
| 18 | c_stride_b | uint32_t | Input B stride for C dimension |
| 19 | src1_num_tiles | uint32_t | Input B total tiles (for sharded mode) |
| 20 | nD_stride_c | uint32_t | Input C stride for ND dimension |
| 21 | d_stride_c | uint32_t | Input C stride for D dimension |
| 22 | n_stride_c | uint32_t | Input C stride for N dimension |
| 23 | c_stride_c | uint32_t | Input C stride for C dimension |
| 24 | src2_num_tiles | uint32_t | Input C total tiles (for sharded mode) |
| 25 | dst_shard_width | uint32_t | Output shard width in tiles (0 if not sharded) |
| 26 | src0_num_tiles | uint32_t | Input A total tiles (for sharded mode) |

**Compute Kernel (4 args):**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_tiles | uint32_t | Number of tiles to process on this core |
| 1 | freq | uint32_t | Broadcast frequency (0 for NONE/OUTER/ROW) |
| 2 | counter | uint32_t | Broadcast start counter (0 for NONE/OUTER/ROW) |
| 3 | scalar_arg | uint32_t | Packed scalar value for the `value` multiplier |

**Writer Kernel (11 args):**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | DRAM/L1 address of output buffer |
| 1 | num_tiles | uint32_t | Number of tiles to write on this core |
| 2 | start_id | uint32_t | Starting output tile index |
| 3 | dst_shard_width | uint32_t | Output shard width in tiles |
| 4 | D | uint32_t | Output D dimension |
| 5 | N | uint32_t | Output N dimension |
| 6 | C | uint32_t | Output C dimension |
| 7 | Ht | uint32_t | Output height in tiles |
| 8 | Wt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Output ND dimension |
| 10 | padding | uint32_t | Reserved (always 0) |

## Kernel Implementations

| Kernel | File | Type | Assigned Cores |
|---|---|---|---|
| Reader | `ternary_reader_nosubtilebcast_ttt.cpp` | Reader (DataMovement) | All worker cores |
| Compute (SFPU, non-INT32) | `ternary_addc_ops_sfpu.cpp` | Compute | All worker cores |
| Compute (SFPU, broadcast) | `ternary_addc_ops_sfpu_bcast.cpp` | Compute | All worker cores |
| Compute (INT32, no-bcast) | `ternary_addcmul_int_sfpu.cpp` | Compute | All worker cores |
| Compute (INT32, bcast) | `ternary_addcmul_int_sfpu_bcast.cpp` | Compute | All worker cores |
| Writer | `ternary_writer_nobcast.cpp` | Writer (DataMovement) | All worker cores |

### Reader Kernel

| Property | Value |
|---|---|
| File | `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_reader_nosubtilebcast_ttt.cpp` |
| Assigned Cores | All worker cores in the operation grid |

**Key Logic:**

- For sharded inputs (`SRC_SHARDED_A/B/C` defines): issues `cb_reserve_back` + `cb_push_back` to make L1 shard data available to compute. No NoC reads are needed.
- For interleaved inputs: creates a `TensorAccessor` per input tensor for address translation.
- Iterates through a 6-level nested loop (ND, D, N, C, Ht, Wt) starting from `start_tile_id`.
- For each tile, computes three separate tile offsets (one per input) using per-tensor stride arrays. This enables outer-dimension broadcasting where a tensor with dimension size 1 has stride 0.
- Issues three parallel `noc_async_read_page` calls (one per input), waits with `noc_async_read_barrier`, then pushes one tile to each input CB.
- Width-sharding support: limits the innermost `tw` loop to `[start_tw, start_tw + dst_shard_width)`.
- **Synchronization**: Produces tiles to c_0, c_1, c_2 via `cb_reserve_back` / `cb_push_back` (one tile at a time for interleaved; full shard for sharded).

### Compute Kernel (SFPU, No-Broadcast)

| Property | Value |
|---|---|
| File | `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addc_ops_sfpu.cpp` |
| Assigned Cores | All worker cores in the operation grid |

**Key Logic:**

- Initializes with `unary_op_init_common(cb_in0, cb_out)`.
- Main loop processes `num_tiles` tiles, one per iteration.
- For each tile:
  1. `cb_wait_front(cb_in0/1/2, 1)` -- waits for one tile on each input CB.
  2. `cb_reserve_back(cb_out, 1)` -- reserves output space.
  3. `tile_regs_acquire()` -- acquires DST register file.
  4. `copy_tile(cb_in0, 0, 0)` -- unpacks input A tile to DST[0].
  5. `copy_tile(cb_in1, 0, 1)` -- unpacks input B tile to DST[1].
  6. `copy_tile(cb_in2, 0, 2)` -- unpacks input C tile to DST[2].
  7. `TERNARY_SFPU_OP_INIT()` -> `addcmul_tile_init()` -- initializes SFPU.
  8. `TERNARY_SFPU_OP_FUNC(0, 1, 2, 0, scalar_arg)` -> `addcmul_tile<DataFormat>(0, 1, 2, 0, scalar_arg)` -- computes `DST[0] + scalar * DST[1] * DST[2] -> DST[0]`.
  9. `tile_regs_commit()` / `tile_regs_wait()` -- commits and waits for SFPU completion.
  10. `pack_tile(0, cb_out)` -- packs DST[0] to output CB.
  11. `tile_regs_release()` -- releases DST.
  12. `cb_push_back(cb_out, 1)` -- publishes output tile.
  13. `cb_pop_front(cb_in0/1/2, 1)` -- frees input tiles.
- **Synchronization**: Consumes from c_0, c_1, c_2 via `cb_wait_front` / `cb_pop_front`. Produces to c_3 via `cb_reserve_back` / `cb_push_back`.

### Compute Kernel (SFPU, Broadcast)

| Property | Value |
|---|---|
| File | `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addc_ops_sfpu_bcast.cpp` |
| Assigned Cores | All worker cores in the operation grid |

**Key Logic:**

- Uses compile-time `BCAST_A`, `BCAST_B`, `BCAST_C` defines to determine which inputs are broadcast.
- Broadcast inputs are waited on (`cb_wait_front`) **before** the inner tile loop and popped (`cb_pop_front`) **after** the loop -- the single tile is reused across multiple output tiles.
- Non-broadcast inputs are waited/popped inside the inner loop as usual.
- Uses a two-level iteration scheme: `complete_iterations = (num_tiles + tile_start) / tile_freq` complete broadcast periods, plus `remaining_iterations` for the partial final period.
- The per-tile computation is identical to the no-broadcast case: copy 3 tiles to DST, call `addcmul_tile`, pack result.
- **Synchronization**: Same CB pattern as no-broadcast, but broadcast CBs are held across multiple tiles within each `freq` period.

### Compute Kernel (INT32 SFPU, No-Broadcast)

| Property | Value |
|---|---|
| File | `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addcmul_int_sfpu.cpp` |
| Assigned Cores | All worker cores (only when dtype == INT32) |

**Key Logic:**

- Unlike the generic SFPU path, this kernel decomposes addcmul into discrete integer operations:
  1. Copies three input tiles to DST[0], DST[1], DST[2].
  2. `fill_tile_int<DataFormat::Int32>(3, scalar_arg)` -- fills DST[3] with the scalar value.
  3. `mul_int_tile<DataFormat::Int32>(3, 1, 3)` -- DST[3] = scalar * DST[1].
  4. `mul_int_tile<DataFormat::Int32>(3, 2, 2)` -- DST[2] = DST[3] * DST[2] (= scalar * B * C).
  5. `add_int_tile<DataFormat::Int32>(0, 2, 0)` -- DST[0] = DST[0] + DST[2] (= A + scalar*B*C).
- Uses 4 DST register slots instead of 3.
- **Synchronization**: Identical pattern to the generic SFPU no-broadcast kernel.

### Writer Kernel

| Property | Value |
|---|---|
| File | `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_writer_nobcast.cpp` |
| Assigned Cores | All worker cores in the operation grid |

**Key Logic:**

- For sharded output (`DST_SHARDED` define): the kernel body is empty -- the output CB is already the shard buffer in L1.
- For interleaved output: creates a `TensorAccessor` for output address resolution.
- Iterates through the same 6-level nested loop structure as the reader (ND, D, N, C, Ht, Wt).
- For each tile: `cb_wait_front` -> `noc_async_write_page` -> `noc_async_write_barrier` -> `cb_pop_front`.
- Width-sharding support: adjusts `dst_tile_offset` by `(Wt - dst_shard_width)` after each row to account for skipped columns.
- **Synchronization**: Consumes from c_3 via `cb_wait_front` / `cb_pop_front`.

## Implementation Notes

- **Program factory variants**: There is only one program factory (`TernaryProgramFactory`). The factory is always selected for ADDCMUL. The operation resolves which specific kernels to use via `TernaryKernelConfig` and the `kernel_config_map` lookup table in `ternary_op_utils.cpp`.

- **Type-based operation variants**: ADDCMUL supports BF16, FP32, INT32, and UINT32 data types. For INT32, a completely different compute kernel is used (`ternary_addcmul_int_sfpu.cpp`) that decomposes the operation into `fill_tile_int` + `mul_int_tile` + `add_int_tile` because the standard `addcmul_tile` SFPU function does not support integer arithmetic. For FP32, the same SFPU kernel is used but templated with `DataFormat::Float32`. BF16 uses `DataFormat::Float16_b`.

- **UnpackToDestFP32 mode**: Enabled per-CB when the corresponding input tensor is FLOAT32. Controlled via `unpack_to_dest_mode[cb_index] = UnpackToDestMode::UnpackToDestFp32`. Also enabled for the output CB when output dtype is FLOAT32.

- **Broadcast type selection**: ADDCMUL supports NONE, OUTER_BCAST, ROW_BCAST, SCALAR_BCAST, and COL_BCAST for the TTT variant. Broadcast type is auto-detected from input tensor shapes via `get_broadcast_type()`. Stride-based broadcasting is used for outer dimensions; sub-tile broadcasting (col fill, row fill, scalar fill) is handled by specialized reader kernels for col/scalar/row cases.

- **Sharding support and constraints**: L1 sharding is supported for HEIGHT_SHARDED, WIDTH_SHARDED, and BLOCK_SHARDED memory layouts. Native L1 sharding requires: (1) all sharded inputs have identical shape and memory config, (2) no uneven shards, (3) all sharded buffers in L1 (not DRAM), and (4) shard grids match the output grid. When these conditions are not met, the operation falls back to the interleaved (TensorAccessor) path even if inputs are sharded.

- **FP32 dest accumulation**: Enabled when output format is UInt32, Int32, or Float32 (`fp32_dest_acc_en` flag in ComputeConfig). This ensures the DST register file operates in 32-bit precision for these data types.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/addcmul.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/llk_math_eltwise_ternary_sfpu_addcmul.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_ternary_sfpu_params.h` (with init in `llk_math_eltwise_ternary_sfpu.h`) |

### Call Chain

1. The compute kernel calls the macro `TERNARY_SFPU_OP_FUNC(0, 1, 2, 0, scalar_arg)` which resolves to `addcmul_tile<DataFormat>(0, 1, 2, 0, scalar_arg)` via preprocessor define set in `ternary_op_utils.cpp`.
2. `addcmul_tile<data_format>()` in `addcmul.h` wraps the call inside `MATH(...)`, invoking `llk_math_eltwise_ternary_sfpu_addcmul<APPROX, DST_ACCUM_MODE, data_format>(idst0, idst1, idst2, odst, value)`.
3. `llk_math_eltwise_ternary_sfpu_addcmul()` in `llk_math_eltwise_ternary_sfpu_addcmul.h` delegates to `_llk_math_eltwise_ternary_sfpu_params_<APPROXIMATE>(sfpu::calculate_addcmul<...>, dst_index0, dst_index1, dst_index2, odst, vector_mode, value)`.
4. `_llk_math_eltwise_ternary_sfpu_params_()` in `llk_math_eltwise_ternary_sfpu_params.h` handles the start/done lifecycle and face iteration, calling `sfpu::calculate_addcmul()` once per face (4 times for VectorMode::RC).
5. `sfpu::calculate_addcmul()` in `ckernel_sfpu_addcmul.h` executes the raw SFPU instructions that perform the addcmul computation on 8 sub-face rows per iteration.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) is active, processing all 4 faces of a 32x32 tile. Each face covers a 16x16 sub-tile.
- **Operation invocation**: The params dispatch calls `calculate_addcmul()` once per face in a loop of 4 iterations. Each call to `calculate_addcmul()` internally loops `ITERATIONS=8` times (one per 2-row sub-face slice), processing all 16 rows of the face.
- **DEST address progression**: Between faces, the params dispatch issues two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` instructions per face, advancing the DEST read/write counter by 16 rows total (2 x 8 rows). Within each face, `sfpi::dst_reg++` advances the DEST pointer by 2 rows after each of the 8 SFPU iterations, covering the 16-row face.

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions but has **no condition code manipulation** -- all instructions used (`SFPLOADI`, `SFPLOAD`, `SFPMUL`, `SFPMAD`, `SFPNOP`, `SFP_STOCH_RND`, `SFPSTORE`) are documented as having no CC effect. This is Style A.

The Blackhole and Wormhole implementations are identical except for the ADDR_MOD index used in `SFPLOAD`/`SFPSTORE` (Blackhole uses `ADDR_MOD_7` directly; Wormhole uses `ADDR_MOD_3` with the address modifier base shifted to bank 4-7, so `ADDR_MOD_3` maps to hardware slot 7). Both resolve to the same physical configuration: `dest.incr = 0`.

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h
// (Wormhole variant is identical except ADDR_MOD_7 -> ADDR_MOD_3)

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_addcmul(
    const uint dst_index_in0,  // input_a
    const uint dst_index_in1,  // input_b
    const uint dst_index_in2,  // input_c
    const uint dst_index_out,  // output
    const uint value) {        // scalar value to multiply with input_b
    // APPROXIMATION_MODE not used, is_fp32_dest_acc_en controls rounding, ITERATIONS=8
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b || data_format == DataFormat::Bfp8_b,
        "Unsupported data format for calculate_addcmul(). Only Float32, Float16_b (BFloat16), and Bfp8_b (BFloat8B) "
        "are allowed.");

    constexpr InstrModLoadStore mod0 =
        (data_format == DataFormat::Float32) ? InstrModLoadStore::FP32 : InstrModLoadStore::DEFAULT;
    // mod0 selects FP32 or BF16 load/store format conversion
    constexpr uint dst_tile_size = 64; // each tile occupies 64 rows in DEST
    // Load scalar value into LREG3 as two 16-bit halves
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, value & 0xFFFF); // lower 16 bits
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, value >> 16);    // upper 16 bits
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // Load input_b row from DEST into LREG1
        TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, dst_index_in1 * dst_tile_size);
        // LREG4 = LREG1 * LREG3 + 0 = input_b * scalar
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
        // Load input_a row from DEST into LREG0
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_7, dst_index_in0 * dst_tile_size);
        // Load input_c row from DEST into LREG2
        TT_SFPLOAD(p_sfpu::LREG2, mod0, ADDR_MOD_7, dst_index_in2 * dst_tile_size);
        // LREG5 = LREG2 * LREG4 + LREG0 = input_c * (scalar * input_b) + input_a
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG5, 0);
        TTI_SFPNOP; // pipeline bubble required before STOCH_RND or STORE
        if constexpr (!is_fp32_dest_acc_en) {
            // Round FP32 result to FP16A (BF16) before storing back to BF16-format DEST
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,          // deterministic round-to-nearest-even
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16A, // convert FP32 -> FP16A (BF16)
                0,
                p_sfpu::LREG5,
                p_sfpu::LREG5,
                InstrModLoadStore::FP16A);
        }
        // Store result back to DEST at output tile position
        TT_SFPSTORE(p_sfpu::LREG5, mod0, ADDR_MOD_7, dst_index_out * dst_tile_size);
        sfpi::dst_reg++; // advance DEST row pointer by 2 rows for next iteration
    }
}
```

### SFPU Instructions Used

| Instruction | Description |
|---|---|
| `SFPLOADI` | Loads a 16-bit immediate value into the lower or upper half of an SFPU local register. Used here to construct the full 32-bit scalar value in LREG3 from two halves. |
| `SFPLOAD` | Loads a vector of values from the DEST register file into an SFPU local register, with optional format conversion (FP32 or BF16). The address is computed from the tile index times the tile size, offset by the current `dst_reg` position. |
| `SFPMUL` | Performs `dst = srcA * srcB + srcC` in the MAD unit. Here used as a pure multiply by setting srcC to `LCONST_0` (zero). Computes `scalar * input_b`. |
| `SFPMAD` | Fused multiply-add: `dst = srcA * srcB + srcC`. Computes `input_c * (scalar * input_b) + input_a` in a single instruction. |
| `SFPNOP` | Pipeline no-operation. Required to allow the MAD result to settle before it can be consumed by the stochastic rounding unit or store unit. |
| `SFP_STOCH_RND` | Stochastic/deterministic rounding instruction. Here configured for round-to-nearest-even with FP32-to-FP16A (BF16) conversion. Only emitted when `is_fp32_dest_acc_en` is false (i.e., DEST is in BF16 mode and the SFPU's FP32 intermediate result must be narrowed). |
| `SFPSTORE` | Stores a vector from an SFPU local register back to the DEST register file, with optional format conversion. Writes the computed addcmul result to the output tile position. |

### SFPU Register Usage

| Register | Usage |
|---|---|
| **LREG0** | Holds input_a values loaded from DEST (overwritten each iteration) |
| **LREG1** | Holds input_b values loaded from DEST (overwritten each iteration) |
| **LREG2** | Holds input_c values loaded from DEST (overwritten each iteration) |
| **LREG3** | Holds the scalar `value` constant, loaded once before the loop via two SFPLOADI instructions. Persists across all iterations. |
| **LREG4** | Temporary: holds the intermediate `scalar * input_b` product |
| **LREG5** | Temporary: holds the final `input_c * (scalar * input_b) + input_a` result, which is stored back to DEST |
| **LCONST_0** | Hard-coded constant register containing 0.0, used as the addend in SFPMUL to perform a pure multiply |
| **dst_reg** | SFPU internal DEST row pointer, auto-incremented by `dst_reg++` (advances by 2 rows per iteration, covering 16 rows over 8 iterations per face) |

### Address Mode Configuration

The ADDR_MOD used by this SFPU operation is configured in `_llk_math_eltwise_ternary_sfpu_init_()` via `eltwise_ternary_sfpu_configure_addrmod<SfpuType::addcmul>()`. The configuration is identical across Blackhole and Wormhole:

**ADDR_MOD_7** (hardware slot 7):
```
addr_mod_t {
    .srca = { .incr = 0 },
    .srcb = { .incr = 0 },
    .dest = { .incr = 0 },
}.set(ADDR_MOD_7);
```

All three fields (srca, srcb, dest) have zero increment, meaning the ADDR_MOD does **not** auto-advance the DEST pointer. Instead, DEST row progression is handled explicitly by `sfpi::dst_reg++` at the end of each loop iteration in the kernel, and by `TTI_SETRWC` instructions in the params dispatch between faces.

**Wormhole address modifier base**: On Wormhole, `_llk_math_eltwise_ternary_sfpu_start_()` calls `math::set_addr_mod_base()` which sets the base to 1, shifting the ADDR_MOD index space from 0-3 to 4-7. Thus the Wormhole ckernel uses `ADDR_MOD_3` (3 + 4 = hardware slot 7), which is the same physical configuration as Blackhole's `ADDR_MOD_7`.

**Blackhole address modifier base**: On Blackhole, no base shift is applied. The ckernel uses `ADDR_MOD_7` directly (hardware slot 7). The done function clears the base via `TTI_SETC16(2, 0)`.

The `where` operation (also a ternary SFPU op) additionally configures `ADDR_MOD_6` with `dest.incr = 2`, but ADDCMUL does not use this -- it only uses ADDR_MOD_7 with zero increments.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How are ternary eltwise SFPU operations implemented? What is the ternary_program_factory and how does it set up kernels for operations like addcmul?"
   **Reason**: Needed to understand the overall architecture of the ternary operation framework before reading source code.
   **Key Findings**: Confirmed TTT variant usage for ADDCMUL, identified kernel naming conventions (ComputeNoBcastAddcOp, ComputeBcastAddcOp), and learned about the kernel_config_map lookup mechanism.

2. **Query**: "What is addcmul_tile and addcmul_tile_init? How does the addcmul SFPU operation work?"
   **Reason**: Needed to understand the low-level SFPU function that implements the addcmul computation.
   **Key Findings**: SFPU operations follow a start-calculate-done pattern. The addcmul tile function operates on DST registers in-place. Binary SFPU operations take dst_index_in0, dst_index_in1, dst_index_out parameters. For addcmul, the signature includes an additional scalar parameter.

3. **Query**: "How does split_work_to_cores work for distributing tiles across cores?"
   **Reason**: Needed to understand the core distribution strategy for interleaved tensors.
   **Key Findings**: Returns a 6-tuple with num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2. Remainder tiles are handled by giving core_group_1 one extra tile each.

4. [SFPU] **Query**: "How is the addcmul_tile SFPU operation implemented? What is the call chain from addcmul_tile through LLK to the core SFPU ckernel function? What file contains the _calculate_addcmul function?"
   **Reason**: Needed to trace the full SFPU call chain from compute API to core implementation.
   **Key Findings**: Confirmed `calculate_addcmul` (not `_calculate_addcmul`) is the core function in `ckernel_sfpu_addcmul.h`. Call chain: `addcmul_tile` -> `llk_math_eltwise_ternary_sfpu_addcmul` -> `_llk_math_eltwise_ternary_sfpu_params_` -> `sfpu::calculate_addcmul`.

5. [SFPU] **Query**: "How is addcmul_tile implemented in the LLK layer? What is llk_math_eltwise_unary_sfpu_addcmul and where is the core SFPU calculate function for addcmul?"
   **Reason**: Cross-referenced tt-llk repo for LLK-level details on ternary SFPU dispatch.
   **Key Findings**: Confirmed ternary SFPU operations use `_llk_math_eltwise_ternary_sfpu_params_` which handles face iteration (VectorMode::RC = 4 faces), start/done lifecycle, and SETRWC-based DEST address progression between faces.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_op_utils.cpp`
   **Reason**: Contains the kernel_config_map that maps (op_type, variant, broadcast_type) tuples to specific kernel files.
   **Key Information**: ADDCMUL uses ComputeNoBcastAddcOp for NONE/OUTER/ROW broadcasts and ComputeBcastAddcOp for SCALAR/COL broadcasts. All use WriterNoBcastTernary and variant-specific readers.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_device_operation.hpp`
   **Reason**: Defines the operation_attributes_t struct to understand which parameters drive kernel selection.
   **Key Information**: Key attributes are ternary_op_type, ternary_variant, broadcast_type, scalar_input_a (used as the multiplier for ADDCMUL), and dtype.

### Confluence References

1. [SFPU] **Page**: Tensix SFPU Instruction Set Architecture (Page ID: 1170505767)
   **Reason**: Consulted to verify CC behavior of SFPU instructions used in the addcmul kernel.
   **Key Findings**: Confirmed that SFPMUL, SFPMAD, SFPLOAD, SFPSTORE, SFPLOADI, and SFPNOP do not modify condition codes. This validates the Style A classification (no CC state machine diagram needed).
