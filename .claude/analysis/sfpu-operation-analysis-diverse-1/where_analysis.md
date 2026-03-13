# WHERE Implementation Analysis

## Overview

The WHERE operation implements conditional element-wise selection: `output = predicate ? value_true : value_false`. It is a ternary operation that selects between two values based on a boolean-like predicate tensor. The operation supports three input variants (TTT, TTS, TST), multiple broadcast modes (NONE, OUTER, COL, ROW, SCALAR), and both interleaved and sharded memory layouts.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_program_factory.cpp`

## Path Selection: FPU vs SFPU

The WHERE operation uses **exclusively the SFPU path**. The FPU path selection logic (lines 1059-1067 of the program factory) only applies to `ADDCMUL` and `ADDCDIV` operations:

```cpp
bool is_fpu = false;
if (operation_attributes.ternary_op_type == TernaryOpType::ADDCMUL ||
    operation_attributes.ternary_op_type == TernaryOpType::ADDCDIV) {
    is_fpu = (predicate_tensor.dtype() == value_true_tensor.value().dtype()) &&
             (predicate_tensor.dtype() == value_false_tensor.value().dtype()) &&
             (predicate_tensor.dtype() != DataType::FLOAT32 && ...);
}
```

Since `TernaryOpType::WHERE` never enters this branch, `is_fpu` remains `false` for all WHERE invocations. All compute kernel paths for WHERE resolve to SFPU kernel files (e.g., `ternary_sfpu_no_bcast_ttt.cpp`, `ternary_sfpu_col_scalar_bcast_ttt.cpp`, etc.).

## Work Unit Definition

One work unit is **one 32x32 tile**. The compute kernel processes exactly one tile per iteration: it copies three input tiles (predicate, true, false) to destination registers, executes the SFPU `where_tile` operation, and packs one output tile. The compile-time constant `num_tiles_per_cycle` is always set to 1.

## Tensor Format and Layout

### Input Tensors

| Property | Predicate Tensor (A) | True Value Tensor (B) | False Value Tensor (C) |
|---|---|---|---|
| Semantic Role | Condition mask | Value when predicate is true | Value when predicate is false |
| Dimension Convention | Up to rank 6+ (ND, D, N, C, Ht, Wt) | Same | Same |
| Tensor Layout | TILE (32x32) | TILE (32x32) | TILE (32x32) |
| Memory Layout | INTERLEAVED or SHARDED (L1) | INTERLEAVED or SHARDED (L1) | INTERLEAVED or SHARDED (L1) |
| Buffer Type | DRAM or L1 | DRAM or L1 | DRAM or L1 |
| Data Type | BFLOAT16, FLOAT32, INT32, UINT32 | Same range | Same range |

**Notes on variants**: In TTS mode, the false value is a scalar (no tensor C). In TST mode, the true value is a scalar (no tensor B). In TTT mode, all three inputs are tensors. The TSS variant is not yet supported in the device operation.

### Output Tensor

| Property | Output |
|---|---|
| Dimension Convention | Broadcasted shape of all inputs |
| Tensor Layout | TILE (32x32) |
| Memory Layout | INTERLEAVED or SHARDED (L1) |
| Buffer Type | DRAM or L1 |
| Data Type | Matches predicate dtype or explicit override |

### Layout Transformations

No tilize/untilize operations are performed. All inputs and outputs are expected in tiled layout. For broadcast modes (COL_BCAST, ROW_BCAST, SCALAR_BCAST), the reader kernel performs tile-level fill operations (e.g., `fill_tile_with_first_column`, `fill_tile_with_first_row`, `fill_tile_with_first_element`) to replicate data within tiles. For ROW_BCAST with BF16 on TTT, the compute kernel uses `unary_bcast<BroadcastType::ROW>` LLK to perform row broadcast in the compute pipeline.

## Data Flow Pattern

### TTT No-Broadcast Path (primary path)

1. **Reader** reads three tiles from DRAM/L1 (predicate, true, false) using `TensorAccessor` for address calculation, issuing `noc_async_read_page` for each. For sharded inputs, it does `cb_reserve_back` / `cb_push_back` to expose the pre-placed L1 data.
2. Each tile is pushed into its respective CB (c_0, c_1, c_2).
3. **Compute** waits on all three CBs (`cb_wait_front`), copies tiles to DST registers 0/1/2 via `copy_tile`, executes `where_tile_init()` then `where_tile<Format>(0, 1, 2, 0)`, packs result from DST[0] to output CB c_3.
4. **Writer** waits on c_3 (`cb_wait_front`), writes tile to DRAM/L1 via `noc_async_write_page`, pops the CB.

### TTS/TST Path

Same as TTT except only two tensor CBs are used (c_0, c_1). The compute kernel fills the missing scalar value directly into a DST register using `fill_tile` before calling the SFPU operation.

### Broadcast Paths (COL_BCAST, SCALAR_BCAST)

The compute kernel uses a `process_tile` function with frequency-based iteration. Broadcast CBs are waited on once outside the inner loop and popped after all tiles in the frequency cycle are processed. Non-broadcast CBs are waited/popped per tile inside the loop.

### ROW_BCAST Path (TTT only, BF16)

Uses additional intermediate CBs (c_4, c_5, c_6). The compute kernel first performs `unary_bcast<BroadcastType::ROW>` on each broadcast input, writing results to the intermediate CBs, then reads from the effective CBs for the SFPU operation.

## Circular Buffer Configuration

### TTT No-Broadcast / Outer-Broadcast (Interleaved)

| CB ID | Name/Purpose | Data Format | Num Pages | Block Size (tiles) | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|
| c_0 | Predicate input | predicate dtype | 2 | 1 | Double | Reader | Compute |
| c_1 | True value input | true dtype | 2 | 1 | Double | Reader | Compute |
| c_2 | False value input | false dtype | 2 | 1 | Double | Reader | Compute |
| c_3 | Output | output dtype | 2 | 1 | Double | Compute | Writer |

### TTT Row-Broadcast (BF16)

Same as above plus:

| CB ID | Name/Purpose | Data Format | Num Pages | Block Size (tiles) | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|
| c_4 | Broadcast A intermediate | predicate dtype | 2 | 1 | Double | Compute | Compute |
| c_5 | Broadcast B intermediate | true dtype | 2 | 1 | Double | Compute | Compute |
| c_6 | Broadcast C intermediate | false dtype | 2 | 1 | Double | Compute | Compute |

### TTS / TST (Interleaved)

| CB ID | Name/Purpose | Data Format | Num Pages | Block Size (tiles) | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|
| c_0 | Predicate input | predicate dtype | 2 | 1 | Double | Reader | Compute |
| c_1 | Tensor operand (true for TTS, false for TST) | tensor dtype | 2 | 1 | Double | Reader | Compute |
| c_3 | Output | output dtype | 2 | 1 | Double | Compute | Writer |

### Sharded Mode

When sharding is active, CB page counts are set to the shard volume (number of tiles in the shard) instead of the default 2. The CB is backed by the tensor's L1 buffer directly via `UpdateDynamicCircularBufferAddress`.

## Pipeline Pattern Summary

- **Interleaved mode**: All CBs have capacity = 2 pages, block size = 1 tile. This enables **double-buffering** -- the reader can fill the next tile while compute processes the current one.
- **Sharded mode**: CBs have capacity = shard volume. All shard tiles are exposed at once (single bulk transfer), so there is no streaming overlap -- it is effectively **single-buffered bulk**.

## Index Calculations

The reader and writer kernels use a nested loop structure over dimensions `(ND, D, N, C, Ht, Wt)` where:
- `ND` = collapsed dimensions beyond rank 5
- `D` = padded_shape[-5] (or 1)
- `N` = padded_shape[-4]
- `C` = padded_shape[-3]
- `Ht` = padded_shape[-2] / tile_height
- `Wt` = padded_shape[-1] / tile_width

**Stride calculation** for broadcast support: Each tensor has per-dimension strides computed as:
- `nD_stride = Ht * Wt * C * N * D * (ND > 1)`
- `d_stride = Ht * Wt * C * N * (D > 1)`
- `n_stride = Ht * Wt * C * (N > 1)`
- `c_stride = Ht * Wt * (C > 1)`

When a dimension has size 1 (broadcast), its stride is 0, causing repeated access to the same tile offset.

**TensorAccessor** is used for physical memory address resolution. Compile-time args encode the accessor configuration, and common runtime args provide the tensor's shape for bank mapping.

## Memory Access Patterns

### Read Pattern

- **Interleaved**: Sequential tile-by-tile reads via `noc_async_read_page`. Tiles are read in row-major order within each (Ht, Wt) block, then iterated over C, N, D, ND dimensions. Each read is followed by `noc_async_read_barrier` before pushing to CB.
- **Sharded**: No reads needed -- data is already in L1. The reader simply does `cb_reserve_back` / `cb_push_back` to expose the shard.
- **Width-sharded**: The reader uses `dst_shard_width` to limit the Wt range per core, reading only the assigned column slice.

### Write Pattern

- **Interleaved**: Sequential tile-by-tile writes via `noc_async_write_page`, with barrier after each tile. Same dimension ordering as reader. For width-sharded inputs with interleaved output, the writer adjusts `dst_tile_offset` to account for skipped columns.
- **Sharded**: No writes needed -- output CB is backed by the output L1 buffer directly.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Rectangular grid from `operation_attributes.worker_grid` |
| Work Splitting | `split_work_to_cores` divides total output tiles across cores |
| Core Groups | Group 1: floor(tiles/cores) + 1 tiles; Group 2: floor(tiles/cores) tiles |
| Remainder Handling | Extra tiles distributed to first N cores (group 1) |
| Zero-start optimization | When grid starts at (0,0), uses `compute_with_storage_grid` for more efficient enumeration |
| Sharded mode | Core grid comes from the shard spec; each core processes its own shard |
| Noop cores | Cores outside active range receive zero-initialized runtime args |

## Arguments

### Compile-Time Arguments

#### Reader Kernel (TTT variant)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_src0 | uint32_t | CB index for predicate tensor (c_0) |
| 1 | cb_id_src1 | uint32_t | CB index for true tensor (c_1) |
| 2 | cb_id_src2 | uint32_t | CB index for false tensor (c_2) |
| 3+ | TensorAccessorArgs (src0) | multiple | Bank mapping params for predicate |
| N+ | TensorAccessorArgs (src1) | multiple | Bank mapping params for true tensor |
| M+ | TensorAccessorArgs (src2) | multiple | Bank mapping params for false tensor |
| last | has_sharding | uint32_t | 1 if sharding is active, 0 otherwise |

#### Reader Kernel (TTS/TST variant)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_src0 | uint32_t | CB index for predicate tensor (c_0) |
| 1 | cb_id_src1 | uint32_t | CB index for tensor operand (c_1) |
| 2+ | TensorAccessorArgs (src0) | multiple | Bank mapping params for predicate |
| N+ | TensorAccessorArgs (src1) | multiple | Bank mapping params for tensor operand |

#### Compute Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 -- tiles processed per cycle |
| 1 | scalar_is_true_value | uint32_t | 1 for TST (scalar is true), 0 for TTS (scalar is false); not used for TTT |

#### Writer Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_out | uint32_t | CB index for output (c_3) |
| 1+ | TensorAccessorArgs (dst) | multiple | Bank mapping params for output |
| last | has_sharding | uint32_t | 1 if sharding is active |

### Runtime Arguments

#### Reader (27 args, TTT variant)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src0_addr | uint32_t | Predicate tensor DRAM/L1 address |
| 1 | src1_addr | uint32_t | True tensor DRAM/L1 address |
| 2 | src2_addr | uint32_t | False tensor DRAM/L1 address |
| 3 | num_tiles | uint32_t | Number of output tiles for this core |
| 4 | start_id | uint32_t | Starting tile ID for this core |
| 5-8 | pred strides (nD, d, n, c) | uint32_t | Predicate tensor strides per dimension |
| 9-14 | D, N, C, Ht, Wt, ND | uint32_t | Output tensor dimensions |
| 15-18 | true strides (nD, d, n, c) | uint32_t | True tensor strides per dimension |
| 19 | b_num_tiles | uint32_t | True tensor shard tile count (sharding) |
| 20-23 | false strides (nD, d, n, c) | uint32_t | False tensor strides per dimension |
| 24 | f_num_tiles | uint32_t | False tensor shard tile count (sharding) |
| 25 | dst_shard_width | uint32_t | Output shard width in tiles (sharding) |
| 26 | src0_num_tiles | uint32_t | Predicate shard tile count (sharding) |

#### Writer (11 args)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Output tensor DRAM/L1 address |
| 1 | num_tiles | uint32_t | Tiles to write for this core |
| 2 | start_id | uint32_t | Starting output tile ID |
| 3 | dst_shard_width | uint32_t | Output shard width in tiles |
| 4-9 | D, N, C, Ht, Wt, ND | uint32_t | Output tensor dimensions |
| 10 | padding | uint32_t | Reserved (0) |

#### Compute (4 args)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_tiles | uint32_t | Total tiles to process on this core |
| 1 | freq | uint32_t | Broadcast frequency (Wt for COL, HtWt for SCALAR, 0 for NONE/OUTER/ROW) |
| 2 | counter | uint32_t | Starting offset within broadcast cycle |
| 3 | scalar_arg | uint32_t | Packed scalar value (TTS/TST only; 0 for TTT) |

## Kernel Implementations

| Kernel Role | File | Assigned Cores | Brief Description |
|---|---|---|---|
| Reader (TTT no-bcast/outer) | `kernels/dataflow/ternary_reader_nosubtilebcast_ttt.cpp` | All worker cores | Reads 3 tensors tile-by-tile with stride-based indexing |
| Reader (TTT col bcast) | `kernels/dataflow/ternary_reader_colbcast_ttt.cpp` | All worker cores | Reads 3 tensors with column fill for broadcast dims |
| Reader (TTT row bcast) | `kernels/dataflow/ternary_reader_rowbcast_ttt.cpp` | All worker cores | Reads 3 tensors with row fill for broadcast dims |
| Reader (TTT scalar bcast) | `kernels/dataflow/ternary_reader_scalar_ttt.cpp` | All worker cores | Reads 3 tensors with scalar fill for broadcast dims |
| Reader (TTS/TST no-bcast) | `kernels/dataflow/ternary_reader_nobcast_tst_tts.cpp` | All worker cores | Reads 2 tensors for TTS/TST variant |
| Reader (TTS/TST outer bcast) | `kernels/dataflow/tst_tts_reader_outer_bcast.cpp` | All worker cores | Reads 2 tensors with outer-dim broadcasting |
| Reader (TTS/TST col bcast) | `kernels/dataflow/tts_tst_reader_col_bcast.cpp` | All worker cores | Reads 2 tensors with column broadcasting |
| Reader (TTS/TST row bcast) | `kernels/dataflow/tts_tst_reader_row_bcast.cpp` | All worker cores | Reads 2 tensors with row broadcasting |
| Reader (TTS/TST scalar bcast) | `kernels/dataflow/tst_tts_reader_scalar_bcast.cpp` | All worker cores | Reads 2 tensors with scalar broadcasting |
| Compute (TTT no-bcast) | `kernels/compute/ternary_sfpu_no_bcast_ttt.cpp` | All worker cores | 3-input SFPU where: copy 3 tiles to DST, call where_tile |
| Compute (TTT col/scalar bcast) | `kernels/compute/ternary_sfpu_col_scalar_bcast_ttt.cpp` | All worker cores | Frequency-based broadcast loop for 3 inputs |
| Compute (TTT row bcast) | `kernels/compute/ternary_sfpu_row_bcast_ttt.cpp` | All worker cores | LLK unary_bcast<ROW> + SFPU where |
| Compute (TTS/TST no-bcast) | `kernels/compute/ternary_sfpu_no_bcast_tts_tst.cpp` | All worker cores | 2-input + scalar fill SFPU where |
| Compute (TTS/TST col/scalar bcast) | `kernels/compute/ternary_sfpu_col_scalar_bcast_tts_tst.cpp` | All worker cores | Frequency-based broadcast loop for 2 inputs + scalar |
| Writer | `kernels/dataflow/ternary_writer_nobcast.cpp` | All worker cores | Writes output tiles to DRAM/L1 |

### Reader Kernel (TTT No-Broadcast)

| Property | Value |
|---|---|
| File | `kernels/dataflow/ternary_reader_nosubtilebcast_ttt.cpp` |
| Assigned Cores | All worker cores |

**Key Logic**:
- For each sharded input: `cb_reserve_back(cb, num_tiles)` + `cb_push_back(cb, num_tiles)` exposes the L1 shard to the compute kernel without any data movement.
- For each interleaved input: creates a `TensorAccessor` from compile-time and common runtime args, then iterates through the 6D loop (ND, D, N, C, Ht, Wt).
- Per tile: `cb_reserve_back(cb, 1)` -> `noc_async_read_page(offset, accessor, l1_addr)` -> `noc_async_read_barrier()` -> `cb_push_back(cb, 1)`.
- Stride-based indexing supports broadcast: when a dimension has size 1, its stride is 0, so the same tile is re-read.
- Width-sharding support: `end_tw = start_tw + dst_shard_width` limits tile reads to the assigned column slice.
- Conditional compilation via `SRC_SHARDED_A/B/C` defines controls per-tensor sharding behavior.

### Compute Kernel (TTT No-Broadcast)

| Property | Value |
|---|---|
| File | `kernels/compute/ternary_sfpu_no_bcast_ttt.cpp` |
| Assigned Cores | All worker cores |

**Key Logic**:
- Simple tile-at-a-time loop over `num_tiles`.
- Per tile: waits on c_0, c_1, c_2 (1 tile each), reserves c_3.
- `tile_regs_acquire()` -- acquires exclusive access to DST registers.
- `copy_tile_to_dst_init_short(cb)` + `copy_tile(cb, 0, dst_reg)` for each input: predicate -> DST[0], true -> DST[1], false -> DST[2].
- `TERNARY_SFPU_OP_INIT()` expands to `where_tile_init()`.
- `TERNARY_SFPU_OP_FUNC(0, 1, 2, 0)` expands to `where_tile<Format>(0, 1, 2, 0)` which performs the conditional selection on the SFPU.
- `tile_regs_commit()` -> `tile_regs_wait()` -> `pack_tile(0, cb_out)` -> `tile_regs_release()`.
- Pops all three input CBs and pushes output CB after each tile.
- **Synchronization**: waits on c_0, c_1, c_2 before processing; pushes to c_3 after packing; pops c_0, c_1, c_2 after processing.

### Compute Kernel (TTT Col/Scalar Broadcast)

| Property | Value |
|---|---|
| File | `kernels/compute/ternary_sfpu_col_scalar_bcast_ttt.cpp` |
| Assigned Cores | All worker cores |

**Key Logic**:
- Uses a `process_tile` function with `freq` (broadcast frequency) and `tile_start` parameters.
- Broadcast CBs (controlled by `BCAST_A/B/C` defines) are waited on once outside the inner `freq` loop and popped once after.
- Non-broadcast CBs are waited/popped per tile inside the inner loop.
- The outer loop iterates `complete_iterations = (num_tiles + tile_start) / tile_freq` times, plus a partial iteration for remaining tiles.
- Same copy_tile -> SFPU op -> pack_tile pattern as no-broadcast kernel.

### Compute Kernel (TTS/TST No-Broadcast)

| Property | Value |
|---|---|
| File | `kernels/compute/ternary_sfpu_no_bcast_tts_tst.cpp` |
| Assigned Cores | All worker cores |

**Key Logic**:
- Only two input CBs (c_0 = predicate, c_1 = tensor operand).
- Copies predicate to DST[0].
- Based on `scalar_is_true` compile-time arg: TST copies tensor to DST[2] and fills scalar to DST[1]; TTS copies tensor to DST[1] and fills scalar to DST[2].
- Scalar fill uses `fill_tile_init()` + `FILL_LLK(reg, value)` which is type-dispatched (float vs int).
- Same SFPU where_tile call with all three DST registers.

### Writer Kernel

| Property | Value |
|---|---|
| File | `kernels/dataflow/ternary_writer_nobcast.cpp` |
| Assigned Cores | All worker cores |

**Key Logic**:
- When `DST_SHARDED` is defined, the writer is a no-op (output is already in L1).
- When interleaved: creates `TensorAccessor` for output, iterates the same 6D loop as the reader.
- Per tile: `cb_wait_front(cb_out, 1)` -> `noc_async_write_page(offset, accessor, l1_addr)` -> `noc_async_write_barrier()` -> `cb_pop_front(cb_out, 1)`.
- Width-sharding adjustment: after each Ht row, `dst_tile_offset += (Wt - dst_shard_width)` to skip non-owned columns.

## Implementation Notes

- **Program factory variants**: There is a single `TernaryProgramFactory` that handles all ternary operations (WHERE, LERP, ADDCMUL, ADDCDIV). The `TernaryKernelConfig` hash map selects the appropriate reader/compute/writer kernel combination based on `(op_type, variant, broadcast_type)`. For WHERE specifically, there are 17 configurations covering all variant x broadcast_type combinations.
- **Type-based operation variants**: WHERE supports BFLOAT16, FLOAT32, INT32, and UINT32 data types. The compute define `TERNARY_SFPU_OP_FUNC` is templated on `DataFormat` -- `Float32` for FLOAT32, `Int32` for INT32, and `Float16_b` for all others (including BFLOAT16 and UINT32). Scalar fill uses different LLK macros for float vs int types (`FILL_WITH_VALUE_FLOAT` vs `FILL_WITH_VALUE_INT`).
- **UnpackToDestFP32 mode**: Enabled per-CB when the corresponding input tensor has `DataType::FLOAT32`. The `unpack_to_dest_mode` vector is populated for c_0, c_1, c_2, and c_3 based on each tensor's dtype. This ensures FP32 data is unpacked to DEST in full precision.
- **Broadcast type selection**: The broadcast type is determined by `get_broadcast_type()` which compares the shapes of all input tensors. It supports NONE, OUTER_BCAST, COL_BCAST, ROW_BCAST, SCALAR_BCAST (TTT), SCALAR_A_BCAST, and SCALAR_B_BCAST (TTS/TST). The broadcast type drives kernel selection and reader fill behavior.
- **Sharding support and constraints**: Native L1 sharding is supported for TTT when all tensor shapes and memory configs match, tensors are in L1 (not DRAM), shard grids are identical, and shards are evenly distributed. Uneven shards and mismatched grids fall back to interleaved mode via TensorAccessor. Height, width, and block sharding are all handled by the `ShardShapeGenerator` class which computes per-core shard shapes including edge cases.
- **FP32 dest accumulation**: Enabled when output format is UInt32, Int32, or Float32. Set via `ComputeConfig::fp32_dest_acc_en`.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/where.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/llk_math_eltwise_ternary_sfpu_where.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/common/inc/sfpu/ckernel_sfpu_where.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_ternary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `where_tile<DataFormat>(0, 1, 2, 0)` (API Header), which wraps the call inside the `MATH(...)` macro to ensure it runs only on the math RISC-V.
2. Inside, it calls `llk_math_eltwise_ternary_sfpu_where<APPROX, data_format>(idst0, idst1, idst2, odst)` (LLK Dispatch), which passes `_calculate_where_<APPROX, data_format, 8>` as a callable to the params dispatch.
3. The params dispatch function `_llk_math_eltwise_ternary_sfpu_params_` (Parameters Dispatch) handles synchronization, face iteration (4 faces in RC mode), and calls the SFPU function once per face.
4. The core SFPU function `_calculate_where_<APPROX, data_format, 8>` (Core SFPU Implementation) records the instruction sequence into a replay buffer and replays it 8 times per face (8 rows of 32 elements = 256 elements per 16x16 face).

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the 32x32 tile are processed -- the params dispatch loops `face = 0..3`, calling the SFPU function once per face.
- **Operation invocation**: For each of the 4 faces, the core SFPU function is called with the same `dst_index_in0/in1/in2/out` arguments. After each face call, `TTI_SETRWC` advances the DEST read/write counter by 16 rows (two increments of 8 rows each) to move to the next 16x16 face.
- **DEST address progression**: The DEST base address is set to 0 at the start via `_llk_math_eltwise_ternary_sfpu_start_<DST_SYNC_MODE>(0)`. Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice, incrementing the DEST counter by 16 rows total. The SFPU function's own SFPLOAD/SFPSTORE offsets are computed from the tile-level `dst_index` arguments and are independent of this counter -- but when `ADDR_MOD_6` is used for SFPSTORE, `dest.incr=2` advances the row pointer by 2 after each store, enabling sequential row writes within a face.

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions with SFPSETCC/SFPENCC condition code manipulation. However, the CC flow is straightforward (single SFPSETCC, single SFPENCC, no nesting), so **Style A** inline annotation is used.

The kernel has two code paths controlled by `DISABLE_SFPLOADMACRO`:
- **DISABLE_SFPLOADMACRO defined** (fallback path): Uses basic `TT_SFPLOAD`/`TT_SFPSTORE` with `lltt::record`/`lltt::replay`.
- **DISABLE_SFPLOADMACRO not defined** (optimized path, default on Blackhole): Uses `TT_SFPLOADMACRO` and `load_replay_buf` for multi-issue scheduling, with two sub-paths depending on whether output overwrites the predicate input (`dst_index_out == dst_index_in0`).

The Blackhole and Wormhole implementations are structurally identical; they differ only in ADDR_MOD numbering. On Wormhole, `set_addr_mod_base()` shifts the base by +4, so ADDR_MOD_3 maps to physical slot 7 and ADDR_MOD_2 maps to physical slot 6 -- the same physical slots that Blackhole references directly as ADDR_MOD_7 and ADDR_MOD_6. Both files are shown below.

#### Blackhole Implementation

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_where.h

template <bool APPROXIMATION_MODE, DataFormat data_format, int ITERATIONS>
inline void _calculate_where_( // APPROXIMATION_MODE: not used, ITERATIONS=8
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_in2, const std::uint32_t dst_index_out)
{
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b || data_format == DataFormat::Int32 || data_format == DataFormat::UInt32,
        "Unsupported data format for _calculate_where_(). Only Float32, Int32, UInt32, and Float16_b are allowed.");

    int offset0 = (dst_index_in0 * 32) << 1; // Byte offset to predicate tile in DEST (tile * 32 rows * 2 bytes)
    int offset1 = (dst_index_in1 * 32) << 1; // Byte offset to true-value tile in DEST
    int offset2 = (dst_index_in2 * 32) << 1; // Byte offset to false-value tile in DEST

    // LO16 for bfloat16 (read lower 16 bits), INT32 for 32-bit formats (read full 32 bits)
    constexpr std::uint32_t mod0 = data_format == DataFormat::Float16_b ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;

#ifdef DISABLE_SFPLOADMACRO
    int offset3 = (dst_index_out * 32) << 1;

    lltt::record(0, 6); // Record 6 instructions into replay buffer slot 0
    TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_7, offset0); // Load predicate row into LREG0; ADDR_MOD_7 has dest.incr=0
    TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, offset1); // Load true-value row into LREG1
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0); // InstrMod=6: CC.Res=1 where LREG0==0 (predicate is zero)
    TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, offset2); // CC-guarded: overwrite LREG1 with false-value only where CC.Res=1
    TTI_SFPENCC(0, 0, 0, sfpi::SFPENCC_MOD1_EU_R1); // InstrMod=0: CC.Res=1 for all lanes, CC.En unchanged (reset predication)
    TT_SFPSTORE(p_sfpu::LREG1, mod0, ADDR_MOD_6, offset3); // Store LREG1 to output; ADDR_MOD_6 has dest.incr=2

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) // Replay 8 times (8 rows per 16x16 face)
    {
        lltt::replay(0, 6);
    }
#else
    if (dst_index_out == dst_index_in0)
    {
        // Implementation notes, see the original file for more details

        load_replay_buf(
            0,
            3,
            [offset0, offset1, offset2]
            {
                TT_SFPLOADMACRO((0 << 2), mod0, ADDR_MOD_7, offset0); // Load predicate + trigger Macro 0 (SFPSETCC + SFPSTORE)
                TT_SFPLOADMACRO((2 << 2), mod0, ADDR_MOD_7, offset1); // Load true-value + trigger Macro 2 (SFPENCC)
                TT_SFPLOAD(0, mod0, ADDR_MOD_6, offset2);             // Load false-value; ADDR_MOD_6 advances dest by 2
            });

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            lltt::replay(0, 3); // 3 cycles per row -- optimal pipeline utilization
        }
    }
    else
    {
        // Implementation notes, see the original file for more details

        int offset3 = (dst_index_out * 32) << 1;

        load_replay_buf(
            0,
            4,
            [offset0, offset1, offset2, offset3]
            {
                TT_SFPLOADMACRO((1 << 2), mod0, ADDR_MOD_7, offset0); // Load predicate + trigger Macro 1 (SFPSETCC)
                TT_SFPLOADMACRO((2 << 2), mod0, ADDR_MOD_7, offset1); // Load true-value + trigger Macro 2 (SFPENCC)
                TT_SFPLOAD(0, mod0, ADDR_MOD_7, offset2);             // Load false-value
                TT_SFPSTORE(0, mod0, ADDR_MOD_6, offset3);            // Store result; ADDR_MOD_6 advances dest by 2
            });

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            lltt::replay(0, 4); // 4 cycles per row
        }
    }
#endif
}

template <bool APPROXIMATION_MODE>
inline void _init_where_()
{
#ifndef DISABLE_SFPLOADMACRO
    // InstructionTemplate[0]: SFPSETCC with LREG_EQ0 mode
    TTI_SFPSETCC(0, 0, 12, 6); // imm12=12 selects template slot, InstrMod=6 is SFPSETCC_MOD1_LREG_EQ0

    // InstructionTemplate[1]: SFPENCC -- resets CC.Res to 1 for all lanes
    TTI_SFPENCC(0, 0, 13, 0);  // imm12=13 selects template slot, InstrMod=0

    // Macro 0: in-place where(a, b, c, a) -- 3 cycles per row
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 4; // template[0]=SFPSETCC at delay 4
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x00 | (2 << 3) | 3; // store using ADDR_MOD from load at delay 3

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0); // Write Macro 0 config to LoadMacroConfig[0]
    }

    // Macro 1: non-in-place where(a, b, c, d) -- 4 cycles per row
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 4; // template[0]=SFPSETCC at delay 4
        constexpr std::uint32_t mad_bits    = 0;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 1, 1); // Write Macro 1 config (lower half only)
    }

    // Macro 2: triggers SFPENCC (template[1])
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 5; // template[1]=SFPENCC at delay 5
        constexpr std::uint32_t mad_bits    = 0;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 2, 1); // Write Macro 2 config (lower half only)
    }

    // Misc config: UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1 for all 3 macros
    TTI_SFPCONFIG(0x770, 8, 1);
#endif
}
```

#### Wormhole Implementation

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_where.h
// Structurally identical to Blackhole. Only ADDR_MOD numbers differ due to addr_mod_base offset:
//   Blackhole ADDR_MOD_7 (loads) == Wormhole ADDR_MOD_3 + base offset 4
//   Blackhole ADDR_MOD_6 (store) == Wormhole ADDR_MOD_2 + base offset 4

template <bool APPROXIMATION_MODE, DataFormat data_format, int ITERATIONS>
inline void _calculate_where_(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_in2, const std::uint32_t dst_index_out)
{
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b || data_format == DataFormat::Int32 || data_format == DataFormat::UInt32,
        "Unsupported data format for _calculate_where_(). Only Float32, Int32, UInt32, and Float16_b are allowed.");

    int offset0 = (dst_index_in0 * 32) << 1;
    int offset1 = (dst_index_in1 * 32) << 1;
    int offset2 = (dst_index_in2 * 32) << 1;

    constexpr std::uint32_t mod0 = data_format == DataFormat::Float16_b ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;

#ifdef DISABLE_SFPLOADMACRO
    int offset3 = (dst_index_out * 32) << 1;

    lltt::record(0, 6);
    TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_3, offset0); // ADDR_MOD_3 + base 4 = physical ADDR_MOD_7
    TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_3, offset1);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_3, offset2);
    TTI_SFPENCC(0, 0, 0, sfpi::SFPENCC_MOD1_EU_R1);
    TT_SFPSTORE(p_sfpu::LREG1, mod0, ADDR_MOD_2, offset3); // ADDR_MOD_2 + base 4 = physical ADDR_MOD_6

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        lltt::replay(0, 6);
    }
#else
    if (dst_index_out == dst_index_in0)
    {
        // Implementation notes, see the original file for more details

        lltt::record(0, 3);
        TT_SFPLOADMACRO((0 << 2), mod0, ADDR_MOD_3, offset0);
        TT_SFPLOADMACRO((2 << 2), mod0, ADDR_MOD_3, offset1);
        TT_SFPLOAD(0, mod0, ADDR_MOD_2, offset2);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            lltt::replay(0, 3);
        }
    }
    else
    {
        // Implementation notes, see the original file for more details

        int offset3 = (dst_index_out * 32) << 1;

        lltt::record(0, 4);
        TT_SFPLOADMACRO((1 << 2), mod0, ADDR_MOD_3, offset0);
        TT_SFPLOADMACRO((2 << 2), mod0, ADDR_MOD_3, offset1);
        TT_SFPLOAD(0, mod0, ADDR_MOD_3, offset2);
        TT_SFPSTORE(0, mod0, ADDR_MOD_2, offset3);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            lltt::replay(0, 4);
        }
    }
#endif
}

template <bool APPROXIMATION_MODE>
inline void _init_where_()
{
#ifndef DISABLE_SFPLOADMACRO
    TTI_SFPSETCC(0, 0, 12, 6);
    TTI_SFPENCC(0, 0, 13, 0);

    // Macro 0, 1, 2 and misc config identical to Blackhole
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 4;
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x00 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 4;
        constexpr std::uint32_t mad_bits    = 0;
        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 1, 1);
    }
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 5;
        constexpr std::uint32_t mad_bits    = 0;
        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 2, 1);
    }
    TTI_SFPCONFIG(0x770, 8, 1);
#endif
}
```

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `SFPLOAD` (via `TT_SFPLOAD`) | Loads a row of 32 elements from a DEST register tile face into an SFPU local register (LREG). Uses `InstrModLoadStore::LO16` for bfloat16 (reads lower 16 bits as bfloat16) or `InstrModLoadStore::INT32` for 32-bit formats (Float32, Int32, UInt32). |
| `SFPLOADMACRO` (via `TT_SFPLOADMACRO`) | Performs an SFPLOAD and simultaneously schedules a pre-configured instruction on a secondary SFPU sub-unit (simple unit, MAD unit, or store unit) according to a macro template. Enables multi-issue execution: the load, simple (SFPSETCC/SFPENCC), and store operations overlap across pipeline stages, achieving 3-4 cycles per row instead of 6. |
| `SFPSETCC` (via `TTI_SFPSETCC`) | Sets per-lane condition code result (`CC.Res`). With `InstrMod=6` (`SFPSETCC_MOD1_LREG_EQ0`): sets `CC.Res=1` for each lane where the specified LREG equals zero. This identifies lanes where the predicate is false/zero, enabling the subsequent conditional load to overwrite only those lanes. |
| `SFPENCC` (via `TTI_SFPENCC`) | Controls condition code state. With `InstrMod=0` (`SFPENCC_MOD1_EU_R1`): sets `CC.Res=1` for all lanes while keeping `CC.En` unchanged. This effectively ends the conditional block by making all lanes active again. |
| `SFPSTORE` (via `TT_SFPSTORE`) | Stores an SFPU local register value back to a DEST register tile face row. Uses the same `LO16`/`INT32` data mode as loads. |
| `SFPLOADI` (via `TTI_SFPLOADI`) | Loads an immediate 16-bit value into LREG0's lower or upper half. Used during `_init_where_` to stage macro configuration data (simple_bits, mad_bits, round_bits, store_bits) before writing to `LoadMacroConfig` via `SFPCONFIG`. |
| `SFPCONFIG` (via `TTI_SFPCONFIG`) | Writes configuration data to SFPU control registers. Used to program `LoadMacroConfig` entries (macros 0, 1, 2) and `InstructionTemplate` slots (slots 0, 1) during initialization, as well as miscellaneous flags (`UsesLoadMod0ForStore`, `WaitForElapsedInstructions`). |
| `STALLWAIT` (via `TTI_STALLWAIT`) | Pipeline synchronization barrier. Used at SFPU start (`STALL_SFPU, MATH`) to wait for prior math operations to complete before SFPU begins, and at SFPU done (`STALL_CFG, WAIT_SFPU`) to wait for SFPU to finish before clearing state. |
| `SETRWC` (via `TTI_SETRWC`) | Sets/increments the read-write counter for DEST register addressing. Used between faces to advance the DEST pointer by 16 rows (two increments of 8), moving to the next 16x16 face within the 32x32 tile. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Holds the predicate row (32 elements). Loaded from `DEST[offset0]`. Used by `SFPSETCC` to test which lanes have zero predicate values. In the `DISABLE_SFPLOADMACRO` path, LREG0 is loaded explicitly; in the `SFPLOADMACRO` path, LREG0 is the implicit target register. |
| **LREG1** | Holds the selected output value. First loaded with true-value from `DEST[offset1]`. Then, for lanes where the predicate is zero (CC-guarded), overwritten with false-value from `DEST[offset2]`. After the SFPENCC reset, LREG1 contains the final per-lane result: true-value where predicate was non-zero, false-value where predicate was zero. Stored back to `DEST[offset3]`. In the `SFPLOADMACRO` path, LREG0 is reused as the implicit load target for all three loads (the macro schedules parallel operations). |
| **DEST[offset0]** | Source tile for the predicate (tile index `dst_index_in0`, typically DST[0]). |
| **DEST[offset1]** | Source tile for the true value (tile index `dst_index_in1`, typically DST[1]). |
| **DEST[offset2]** | Source tile for the false value (tile index `dst_index_in2`, typically DST[2]). |
| **DEST[offset3]** | Output tile (tile index `dst_index_out`, typically DST[0] -- same as predicate, enabling the in-place optimization). |
| **CC.Res** | Per-lane condition code result bit. Set to 1 by SFPSETCC for lanes where predicate == 0; reset to 1 for all lanes by SFPENCC. Controls which lanes are active for the conditional false-value load. |
| **CC.En** | Per-lane condition code enable bit. Not explicitly modified by this kernel (SFPENCC with InstrMod=0 keeps CC.En unchanged). |

### Address Mode Configuration

Two ADDR_MOD slots are configured for the where operation, set during `_llk_math_eltwise_ternary_sfpu_init_` via `eltwise_ternary_sfpu_configure_addrmod<SfpuType::where>()`:

| ADDR_MOD Slot | srca.incr | srcb.incr | dest.incr | Purpose |
|---------------|-----------|-----------|-----------|---------|
| **ADDR_MOD_7** | 0 | 0 | 0 | Used for SFPLOAD instructions. No auto-increment -- the row address stays fixed, allowing the replay buffer to re-read from the same base offset each iteration. Row advancement is handled by `lltt::replay` incrementing the replay counter. |
| **ADDR_MOD_6** | 0 | 0 | 2 | Used for SFPSTORE instructions. `dest.incr=2` advances the DEST write pointer by 2 rows after each store, enabling sequential row output within a face. |

**Hardware generation differences**: The `addr_mod_t` struct values are identical between Wormhole and Blackhole (both configure the same physical slots with the same increment values). The apparent ADDR_MOD number difference in the `_calculate_where_` code (Wormhole uses ADDR_MOD_3/ADDR_MOD_2; Blackhole uses ADDR_MOD_7/ADDR_MOD_6) is resolved by the addr_mod_base mechanism: Wormhole calls `math::set_addr_mod_base()` at SFPU start, which sets an offset of +4, so ADDR_MOD_3 maps to physical slot 7 and ADDR_MOD_2 maps to physical slot 6. Blackhole does not use addr_mod_base, so it references physical slots 7 and 6 directly.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the ternary 'where' operation work in TTNN? What are the SFPU vs FPU paths for ternary operations?"
   **Reason**: Initial reconnaissance to understand the operation's architecture, variant types, and kernel organization.
   **Key Findings**: WHERE uses TTT/TTS/TST variants, the SFPU path is the default (FPU only for ADDCMUL/ADDCDIV with matching BF16 types), and the `where_tile` SFPU function performs the conditional selection. Multiple broadcast types are supported.

2. [SFPU] **Query**: "How does the where_tile SFPU function work? What is the call chain from where_tile through LLK layers to the ckernel SFPU implementation? What files implement the where_tile SFPU kernel?" (tenstorrent/tt-metal)
   **Reason**: Needed to trace the full abstraction layer path from the compute API down to the core SFPU implementation.
   **Key Findings**: Confirmed the 4-layer abstraction (API -> LLK dispatch -> params dispatch -> ckernel SFPU). Identified `_calculate_where_` as the core function with ITERATIONS=8 template parameter. Located architecture-specific files for both Wormhole and Blackhole.

3. [SFPU] **Query**: "How does the where_tile SFPU function work in the LLK layer? What is the call chain from where_tile_init and where_tile through llk_math_eltwise_ternary_sfpu to ckernel_sfpu_where?" (tenstorrent/tt-llk)
   **Reason**: Needed detailed understanding of the ternary SFPU dispatch mechanism and address modifier configuration.
   **Key Findings**: Confirmed that `eltwise_ternary_sfpu_configure_addrmod` sets ADDR_MOD_7 (dest.incr=0) and ADDR_MOD_6 (dest.incr=2). Identified the Wormhole addr_mod_base offset mechanism (ADDR_MOD_3+4=7, ADDR_MOD_2+4=6). Confirmed the params dispatch iterates over 4 faces with SETRWC advancement.

### Confluence References

1. [SFPU] **Page**: Tensix SFPU Instruction Set Architecture (Page ID: 1170505767)
   **Sections consulted**: SFPSETCC, SFPENCC
   **Reason**: Needed authoritative ISA-level details on condition code manipulation instructions used in the where kernel.
   **Key Findings**: SFPSETCC with InstrMod=6 sets CC.Res where RG[VC] is 0 (CmpRes check). SFPENCC with InstrMod=0 sets CC.Res=1 (unconditionally) and keeps CC.En unchanged. Confirmed that SFPSETCC execution is predicated by current LaneEnabled state, while SFPENCC executes on all lanes regardless.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_op_utils.hpp` and `.cpp`
   **Reason**: Kernel name mapping, broadcast type detection, compute defines, and file path resolution.
   **Key Information**: Complete kernel configuration map with 17 WHERE entries; `get_kernel_file_path` maps enum names to actual file paths; `get_compute_defines` sets `TERNARY_SFPU_OP_INIT` and `TERNARY_SFPU_OP_FUNC` macros.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/ternary/common/ternary_op_types.hpp`
   **Reason**: Understanding enum definitions for TernaryOpType, TernaryVariant, and TernaryBroadcastType.
   **Key Information**: WHERE is one of four ternary op types; 4 variants (TTT, TTS, TST, TSS); 7 broadcast types plus INVALID_BCAST.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_device_operation.hpp`
   **Reason**: Understanding the operation attributes and tensor args structures.
   **Key Information**: Single `TernaryProgramFactory` variant; operation attributes include scalar inputs for TTS/TST variants; worker grid is stored in attributes.
