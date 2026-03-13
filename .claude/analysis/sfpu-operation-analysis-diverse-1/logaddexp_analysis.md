# LOGADDEXP Implementation Analysis

## Overview

LOGADDEXP computes `log(exp(a) + exp(b))` element-wise for two input tensors. It is a composite operation implemented in the `binary_ng` framework by decomposing into three stages: (1) apply `exp` to both inputs as pre-activations, (2) perform `ADD` as the core binary SFPU operation, (3) apply `log` as a post-activation. This decomposition is configured entirely through the `OpConfig` mechanism in the program factory -- there is no dedicated kernel for LOGADDEXP.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

## Path Selection: FPU vs SFPU

The `binary_ng` framework supports both FPU and SFPU execution paths, selected at dispatch time in `ttnn::prim::binary_ng()` (in `binary_ng_device_operation.cpp`). The function `is_binary_sfpu_op()` determines the path based on the operation type and input data types.

For LOGADDEXP specifically (line 32-36 of `binary_ng_device_operation.cpp`), the SFPU path is selected **only when both inputs are FLOAT32** (`a == FLOAT32 && b == FLOAT32`). When both inputs are BFLOAT16, the FPU path is used instead, where `OpConfig` configures `FpuBinaryOp::ADD` with EXP pre-activations and LOG post-activation using FPU-compatible unary operations. The `is_sfpu` flag is stored in `operation_attributes` and checked throughout the program factory to select kernel file paths, circular buffer formats, and compute configuration. In the SFPU path, the `OpConfig` constructor (line 226-231 of `binary_ng_utils.cpp`) sets `process_lhs = EXP`, `process_rhs = EXP`, `binary_op = SfpuBinaryOp::ADD`, and `postprocess = LOG`.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32 elements) |
| **Unit size** | 1 tile (`num_tiles_per_cycle = 1`) |
| **Total units** | `output_tensor.physical_volume() / (tile_height * tile_width)` |
| **Loop structure** | Single loop over output tiles; each iteration reads 1 tile from each input, preprocesses (EXP), computes (ADD), postprocesses (LOG), writes 1 output tile |

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|----------------|----------------|
| **Logical shape** | Arbitrary (up to rank 10, broadcast-compatible) | Arbitrary (up to rank 10, broadcast-compatible) |
| **Dimension convention** | [..., D, N, C, H, W] (last 5 dims significant) | [..., D, N, C, H, W] |
| **Tensor layout** | TILE | TILE |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | FLOAT32 (required for SFPU path) | FLOAT32 (required for SFPU path) |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Broadcast-compatible output of A and B shapes |
| **Dimension convention** | [..., D, N, C, H, W] |
| **Tensor layout** | TILE |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | FLOAT32 (default, matching input) |

### Layout Transformations

No explicit tilize/untilize occurs within the operation. Both inputs and outputs must already be in TILE layout. The pre-activation EXP and post-activation LOG are applied in-place in the DEST register file within the compute kernel using the PREPROCESS macro, which copies tiles through intermediate CBs (c_3, c_4) when activations are present.

## Data Flow Pattern

For the SFPU path with two tensor inputs and no broadcast (SubtileBroadcastType::NONE), the "no_bcast" variant is used:

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (tensor A) | CB c_0 | reserve_back, noc_async_read_page, push_back |
| 1 | Reader | DRAM/L1 (tensor B) | CB c_1 | reserve_back, noc_async_read_page, push_back |
| 2 | Compute (PREPROCESS LHS) | CB c_0 | CB c_3 | wait_front(c_0), reserve_back(c_3), copy+EXP, push_back(c_3), pop_front(c_0) |
| 3 | Compute (PREPROCESS RHS) | CB c_1 | CB c_4 | wait_front(c_1), reserve_back(c_4), copy+EXP, push_back(c_4), pop_front(c_1) |
| 4 | Compute (BINARY+POST) | CB c_3, CB c_4 | CB c_2 | wait_front(c_3, c_4), reserve_back(c_2), copy_tile to DEST, ADD, LOG, pack_tile, push_back(c_2), pop_front(c_3, c_4) |
| 5 | Writer | CB c_2 | DRAM/L1 (output) | wait_front, noc_async_write_page, pop_front |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|---------------------|-----------|----------|----------|----------|
| c_0 | cb_pre_lhs | Input A staging | 2 (interleaved) or shard volume (sharded) | 1 | Double / Shard | Reader | Compute (PREPROCESS LHS) | Block |
| c_1 | cb_pre_rhs | Input B staging | 2 (interleaved) or shard volume (sharded) | 1 | Double / Shard | Reader | Compute (PREPROCESS RHS) | Block |
| c_2 | cb_out | Output staging | 2 (interleaved) or shard volume (sharded) | 1 | Double / Shard | Compute | Writer | Block |
| c_3 | cb_post_lhs | LHS after EXP | 1 | 1 | Single | Compute (PREPROCESS LHS) | Compute (BINARY) | Block |
| c_4 | cb_post_rhs | RHS after EXP | 1 | 1 | Single | Compute (PREPROCESS RHS) | Compute (BINARY) | Block |

Notes:
- CB c_3 is created because LOGADDEXP defines `process_lhs = EXP`, making `PROCESS_LHS_ACTIVATIONS(i)` non-empty. Its data format is `a_data_format` (FLOAT32 on the SFPU path) since `is_sfpu_op` is true (line 645-646 of the program factory -- on the FPU path it would be Float16_b because `op_has_exp` is true for LOGADDEXP).
- CB c_4 is created because `process_rhs = EXP` is set. Same format logic applies.
- CBs c_5 and c_6 are not created in the NONE broadcast case.

## Pipeline Pattern Summary

- **c_0, c_1**: Double-buffered (capacity=2, block=1) in interleaved mode, enabling overlap between reader NoC transfers and compute consumption.
- **c_2**: Double-buffered (capacity=2, block=1) in interleaved mode, enabling overlap between compute production and writer NoC transfers.
- **c_3, c_4**: Single-buffered (capacity=1, block=1). These are intermediate scratch buffers used entirely within the compute kernel, so no cross-kernel overlap is needed.

## Index Calculations

The reader kernel uses a multi-dimensional index decomposition to map a linear start_tile_id to a position within the (nD, D, N, C, Ht, Wt) logical space:

1. Compute `HtWt = Ht * Wt`, `tiles_per_n = C * HtWt`, `tiles_per_d = N * tiles_per_n`, `tiles_per_nd = D * tiles_per_d`.
2. Decompose `start_tile_id` modularly to get `(start_nd, start_d, start_n, start_c, start_th, start_tw)`.
3. For tensor A, a `tile_offset` is computed from these start positions using stride values passed as runtime args. Strides encode broadcasting: if a dimension size is 1 in the input but larger in the output, the stride for that dimension is 0 (computed as `dim_product * (dim > 1)` in the host).
4. The `TensorAccessor` utility maps tile indices to physical DRAM/L1 page addresses through `noc_async_read_page`.

The writer uses an identical nesting structure but with a simpler linear `dst_tile_offset` since the output tensor has the full broadcasted shape.

## Memory Access Patterns

### Read Pattern

- **Interleaved**: Tiles are read one at a time in row-major order within each (nD, D, N, C, H) slice, iterating across width tiles. The 6-level nested loop handles broadcasting by using stride multipliers that zero out when a dimension is broadcast.
- **Sharded**: When inputs are sharded in L1, the reader issues `cb_reserve_back` + `cb_push_back` for the full shard volume at once (no NoC reads needed -- data is already local).

### Write Pattern

- **Interleaved**: Tiles written one at a time via `noc_async_write_page` with a barrier after each tile. Output tile offset is a simple linear counter.
- **Sharded**: No explicit writes; the output CB is backed by the sharded buffer directly.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (uses device compute grid) |
| **Grid dimensions** | device-dependent (e.g., 8x8 on Wormhole) |
| **Total cores** | `compute_with_storage_grid.x * compute_with_storage_grid.y` or `worker_grid.num_cores()` |
| **Work per core** | `ceil(total_output_tiles / num_cores)` for group 1; remainder tiles for group 2 |
| **Load balancing** | Two-group split: `core_group_1` gets `num_tiles_per_core_group_1` tiles, `core_group_2` gets `num_tiles_per_core_group_2` tiles (one fewer). Cores outside both groups get zero-args and exit immediately. |

Work distribution is done by `tt::tt_metal::split_work_to_cores()`, which divides the total output tile count evenly across available cores, with any remainder distributed to a second core group.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs(A) | uint32_t[] | Tensor accessor compile-time args for input A |
| N+1..M | TensorAccessorArgs(B) | uint32_t[] | Tensor accessor compile-time args for input B |
| M+1 | has_sharding | uint32_t | 1 if any tensor uses native L1 sharding, 0 otherwise |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs(C) | uint32_t[] | Tensor accessor compile-time args for output C |
| N+1 | has_sharding | uint32_t | 1 if any tensor uses native L1 sharding, 0 otherwise |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 -- tiles produced per read-compute-write cycle |

### Runtime Arguments

#### Reader Kernel (21 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input A buffer base address |
| 1 | start_tile_id | uint32_t | Starting output tile index for this core (c_start_id) |
| 2 | src_num_tiles | uint32_t | Number of A tiles in shard (0 if not sharded) |
| 3 | dst_num_tiles | uint32_t | Number of output tiles assigned to this core |
| 4 | dst_shard_width | uint32_t | Width of output shard in tiles (0 if not sharded) |
| 5 | nD_stride | uint32_t | Stride for collapsed dims >5 (A) |
| 6 | d_stride | uint32_t | Stride for D dimension (A) |
| 7 | n_stride | uint32_t | Stride for N dimension (A) |
| 8 | c_stride | uint32_t | Stride for C dimension (A) |
| 9 | D | uint32_t | Output D dimension |
| 10 | N | uint32_t | Output N dimension |
| 11 | C | uint32_t | Output C dimension |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Output collapsed nD dimension |
| 15 | src_addr_b | uint32_t | Input B buffer base address |
| 16 | nD_stride_b | uint32_t | Stride for collapsed dims >5 (B) |
| 17 | d_stride_b | uint32_t | Stride for D dimension (B) |
| 18 | n_stride_b | uint32_t | Stride for N dimension (B) |
| 19 | c_stride_b | uint32_t | Stride for C dimension (B) |
| 20 | src_num_tiles_b | uint32_t | Number of B tiles in shard (0 if not sharded) |

#### Writer Kernel (11 args, tensor-tensor variant)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address |
| 1 | start_tile_id | uint32_t | Starting output tile index for this core |
| 2 | dst_num_tiles | uint32_t | Number of output tiles for this core |
| 3 | dst_shard_width | uint32_t | Width of output shard in tiles |
| 4 | D | uint32_t | Output D dimension |
| 5 | N | uint32_t | Output N dimension |
| 6 | C | uint32_t | Output C dimension |
| 7 | Ht | uint32_t | Output height in tiles |
| 8 | Wt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Output collapsed nD dimension |
| 10 | (unused) | uint32_t | Reserved (set to 0) |

#### Compute Kernel (4 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total output tiles for this core |
| 1 | freq | uint32_t | Broadcast frequency (1 for NONE broadcast) |
| 2 | counter | uint32_t | Starting counter within broadcast cycle (0 for NONE) |
| 3 | compute_scalar_value | uint32_t | Unused for LOGADDEXP (set to 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | RISC-V 0 (BRISC) | NOC0 | DRAM/L1 (A, B) | CB c_0, CB c_1 | Read tiles via TensorAccessor |
| Compute | RISC-V 2 (TRISC0/1/2) | N/A | CB c_0, c_1 (via c_3, c_4) | CB c_2 | EXP (pre-LHS), EXP (pre-RHS), add_binary_tile (SFPU ADD), LOG (post) |
| Writer | RISC-V 1 (NCRISC) | NOC1 | CB c_2 | DRAM/L1 (output) | Write tiles via TensorAccessor |

### Reader Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` |
| **Assigned cores** | All cores in `all_device_cores` |

**Key Logic**:
- For interleaved inputs: creates `TensorAccessor` for A and B from compile-time args and buffer addresses. Iterates through a 6-level nested loop (nD, D, N, C, Ht, Wt) reading one tile at a time from each input.
- For sharded inputs: issues `cb_reserve_back` + `cb_push_back` for the entire shard volume immediately (no NoC reads), since sharded data is already in L1 backing the CB.
- Stride values encode broadcasting: when an input dimension is 1 but the output dimension is larger, the corresponding stride is 0, causing the reader to re-read the same tile.
- **Synchronization**: Produces to CB c_0 (input A) and CB c_1 (input B) one tile at a time via `cb_reserve_back(cb, 1)` then `cb_push_back(cb, 1)` after each NoC read completes (barrier).

### Compute Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` |
| **Assigned cores** | All cores in `all_device_cores` |

**Key Logic**:
- The `no_bcast` SFPU variant is selected for `SubtileBroadcastType::NONE` (equal-shaped tensors).
- Per-tile loop (`num_tiles` iterations, each producing 1 output tile):
  1. **PREPROCESS LHS**: The `PREPROCESS` macro (defined in `eltwise_utils_sfpu.hpp`) detects that `HAS_ACTIVATIONS(LHS)` is true (because `PROCESS_LHS_ACTIVATIONS` is defined as `EXP`). It: waits on c_0, reserves c_3, copies tile to DEST, applies EXP via SFPU, packs to c_3, pops c_0, pushes c_3. Includes `pack_reconfig_data_format` calls to switch between output and intermediate formats.
  2. **PREPROCESS RHS**: Same pattern for c_1 -> c_4 with EXP.
  3. **BINARY + POST**: Waits on c_3 and c_4, reserves c_2. Acquires tile regs. Copies LHS tile from c_3 to DEST slot 0, copies RHS tile from c_4 to DEST slot 1. Calls `BINARY_SFPU_INIT` (which expands to `add_binary_tile_init();`) then `BINARY_SFPU_OP(0, 1, 0)` (which expands to `add_binary_tile(0, 1, 0)`) performing SFPU addition. Then `PROCESS_POST_ACTIVATIONS(0)` applies LOG. Commits tile regs, waits, packs to c_2.
  4. Pushes c_2, pops c_3 and c_4.
- Because LOGADDEXP has both pre-activations and a post-activation, `BINARY_SFPU_INIT` is placed inside the post-activation conditional block (re-initialized each tile iteration).
- **UnpackToDestFP32 mode** is enabled for all source CBs (c_0, c_1, c_3, c_4) because LOGADDEXP is not POWER and the SFPU path always uses FP32 unpacking.
- **Synchronization**: Consumes from c_3 and c_4 via `cb_wait_front`, produces to c_2 via `cb_reserve_back` + `cb_push_back`. Internally consumes/produces c_0->c_3 and c_1->c_4 through the PREPROCESS macro.

### Writer Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` |
| **Assigned cores** | All cores in `all_device_cores` |

**Key Logic**:
- For interleaved output: creates `TensorAccessor` from compile-time args and output buffer address. Iterates through the same 6-level nested loop structure as the reader but for the output shape. Writes one tile at a time with `noc_async_write_page` followed by a write barrier.
- For sharded output: the `!DST_SHARDED` guard skips all write logic; the output CB is directly backed by the sharded L1 buffer.
- Output tile offset is computed as `start_tile_id + num_tiles_written`, incrementing linearly since the output has the full broadcasted shape (no stride tricks needed).
- **Synchronization**: Consumes from CB c_2 via `cb_wait_front(c_2, 1)` then `cb_pop_front(c_2, 1)` after each tile is written.

## Implementation Notes

- **Program factory variants**: There is a single `BinaryNgDeviceOperation::ProgramFactory` that handles all binary_ng operations. The factory is selected unconditionally -- there is only one program factory for binary_ng.
- **Type-based operation variants**: On the SFPU path, LOGADDEXP requires both inputs to be FLOAT32. On the FPU path (BFLOAT16 inputs), the same decomposition (EXP + ADD + LOG) is used but with FPU `ADD_tiles` as the core operation and intermediate CBs (c_3, c_4) use Float16_b format rather than the input format. INT32 inputs are not supported for LOGADDEXP.
- **UnpackToDestFP32 mode**: Always enabled on the SFPU path for LOGADDEXP (not POWER exception). All four source CBs (c_0, c_1, c_3, c_4) have `UnpackToDestMode::UnpackToDestFp32`, ensuring SFPU operations work in full FP32 precision in the DEST register.
- **Broadcast type selection**: Supports all `SubtileBroadcastType` variants (NONE, SCALAR_A/B, ROW_A/B, COL_A/B, ROW_A_COL_B, ROW_B_COL_A). The broadcast type determines which reader kernel variant, compute kernel variant, and whether CBs c_5/c_6 are created. For NONE, the `no_bcast` kernels are used. Stride-based broadcasting is encoded in reader runtime args.
- **Sharding support and constraints**: Height, width, and block sharding are all supported. Native L1 sharding requires: same shapes for A and B, no unevenness, all L1 buffers, and matching shard grids. If these conditions are not met, the operation falls back to interleaved mode via TensorAccessor.
- **FP32 dest accumulation**: Enabled when output is FLOAT32 (which it always is on the SFPU path), or when both inputs are FLOAT32 (which they must be for SFPU LOGADDEXP). The `fp32_dest_acc_en` flag is set in `ComputeConfig`.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to. LOGADDEXP is a composite operation that invokes **three** independent SFPU kernels per tile: EXP (pre-activation on LHS), EXP (pre-activation on RHS), binary ADD (core operation), and LOG (post-activation). This analysis focuses on the **binary ADD** SFPU kernel, which is the core operation unique to LOGADDEXP's SFPU dispatch path. The EXP and LOG pre/post-activations are standard unary SFPU operations invoked via the `PREPROCESS` macro and `PROCESS_POST_ACTIVATIONS` macro respectively, and each has its own independent SFPU call chain (documented in their respective unary operation analyses).

### SFPU Abstraction Layers

The binary ADD SFPU kernel follows a four-layer abstraction:

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h` (shared) / `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` (arch-specific wrapper) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `add_binary_tile(0, 1, 0)` (via the `BINARY_SFPU_OP` macro), which is defined in `eltwise_binary_sfpu.h`.
2. `add_binary_tile` wraps `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>(idst0, idst1, odst)` in the `MATH` macro, routing the call to the math RISC-V processor.
3. `llk_math_eltwise_binary_sfpu_binop` (in `llk_math_eltwise_binary_sfpu_binop.h`) calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>()` with `ckernel::sfpu::calculate_sfpu_binary<APPROXIMATE, BinaryOp::ADD, 8, is_fp32_dest_acc_en>` as the callable.
4. `_llk_math_eltwise_binary_sfpu_params_` (in `llk_math_eltwise_binary_sfpu_params.h`) sets up the DEST write address, stalls until the SFPU is ready, then iterates over tile faces (4 faces for `VectorMode::RC`), calling the SFPU function once per face and advancing the DEST pointer by 16 rows (2x `SETRWC` of 8) between faces.
5. `calculate_sfpu_binary` (in the arch-specific `ckernel_sfpu_binary.h`) is a thin wrapper that delegates to `_calculate_sfpu_binary_<APPROXIMATION_MODE, BinaryOp::ADD, 8>()`.
6. `_calculate_sfpu_binary_` (in the shared `ckernel_sfpu_binary.h` under `tt_llk`) performs the actual element-wise addition across 8 iterations (rows) per face, reading from two DEST tile slots and writing the result back.

The init path follows a parallel chain: `add_binary_tile_init()` calls `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::ADD>()`, which calls `llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu_binary_init<APPROXIMATE, BinaryOp::ADD>)`. This initializes the SFPU config register, configures address modifiers, resets counters, and then calls `_sfpu_binary_init_<APPROXIMATE, BinaryOp::ADD>()` -- which is a no-op for ADD (no special initialization required).

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the 32x32 tile are processed. Each face is a 16x16 sub-tile.
- **Operation invocation**: The params dispatch loops `for (int face = 0; face < 4; face++)`, calling `calculate_sfpu_binary(dst_index_in0, dst_index_in1, dst_index_out)` once per face. Inside `_calculate_sfpu_binary_`, there is an inner loop of 8 iterations (ITERATIONS=8), processing 8 rows of vector width per call, which covers the 16x16 face via 8 SFPU vector operations (each processing 2 rows via the SFPU's 32-wide vector lane, or equivalently, 8 groups of 32 elements that tile into the 16x16 face).
- **DEST address progression**: Before any face processing, `_llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(0)` sets the DEST write address to tile index 0. Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice (incrementing the DEST read/write counter by 8+8=16 rows), which advances to the next 16-row face. Within each face, `sfpi::dst_reg++` in the inner loop auto-increments the SFPU's DEST register pointer by 1 row after each iteration. After all 4 faces, `_llk_math_eltwise_binary_sfpu_done_()` clears the DEST address.

### Annotated SFPU Kernel Source

The binary ADD kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`), so Style A applies.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{ // For LOGADDEXP ADD: APPROXIMATION_MODE=true, BINOP=BinaryOp::ADD, ITERATIONS=8
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        // dst_tile_size_sfpi=32: each tile occupies 32 SFPU-addressable rows in DEST
        constexpr std::uint32_t dst_tile_size_sfpi = 32;
        sfpi::vFloat in0                           = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // Load from DEST slot 0 (LHS tile)
        sfpi::vFloat in1                           = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // Load from DEST slot 1 (RHS tile)
        sfpi::vFloat result                        = 0.0f;

        if constexpr (BINOP == BinaryOp::ADD) // Active branch for LOGADDEXP
        {
            result = in0 + in1; // SFPU ADD: element-wise FP32 addition via SFPADD instruction
        }
        // Other branches (SUB, MUL, DIV, RSUB, POW, XLOGY) are compile-time eliminated

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // Store result to DEST slot 0 (output overwrites LHS)
        sfpi::dst_reg++; // Advance DEST pointer by 1 row for next iteration
    }
}
```

The wrapper in the arch-specific file simply delegates:

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    _calculate_sfpu_binary_<APPROXIMATION_MODE, BINOP, ITERATIONS>(dst_index_in0, dst_index_in1, dst_index_out);
    // is_fp32_dest_acc_en is not used for ADD; it is only relevant for MUL (bf16 rounding) and DIV
}
```

The init function is a no-op for ADD:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{
    // For ADD: none of the constexpr branches are taken (DIV/POW need reciprocal init, XLOGY needs log init)
    // No special initialization required for ADD
}
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| **SFPLOADI** | `sfpi::vFloat result = 0.0f` | Load immediate floating-point constant (0.0f) into an SFPU local register |
| **SFPLOAD** | `sfpi::dst_reg[idx]` (read) | Load a vector from the DEST register file at the specified row offset into an SFPU local register |
| **SFPADD** | `in0 + in1` | Element-wise floating-point addition of two SFPU local registers (2-cycle instruction) |
| **SFPSTORE** | `sfpi::dst_reg[idx] = result` (write) | Store an SFPU local register vector back to the DEST register file at the specified row offset |
| **INCRWC** | `sfpi::dst_reg++` | Increment the DEST register write counter by 1, advancing to the next row |
| **SETRWC** | `TTI_SETRWC(...)` (in params dispatch) | Set/increment the read/write counter for DEST addressing; used between faces to advance by 8 rows |
| **STALLWAIT** | `TTI_STALLWAIT(STALL_SFPU, MATH)` | Stall until the SFPU is ready before beginning SFPU operations (in `_llk_math_eltwise_binary_sfpu_start_`) |

Note: The `SFPADD` instruction is the only math instruction used in the ADD path. The `result = in0 + in1` SFPI expression compiles down to a single `SFPADD` that takes two local register sources and produces a local register result. The preceding `result = 0.0f` initialization is redundant for the ADD branch (the value is immediately overwritten) but is present because the function supports multiple `BINOP` specializations and the compiler optimizes it away.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST[dst_index_in0 * 32 + row]** | Source: LHS tile data (exp(a)), read once per iteration. `dst_index_in0` = 0 for LOGADDEXP. |
| **DEST[dst_index_in1 * 32 + row]** | Source: RHS tile data (exp(b)), read once per iteration. `dst_index_in1` = 1 for LOGADDEXP. |
| **DEST[dst_index_out * 32 + row]** | Destination: output tile data (exp(a)+exp(b)), written once per iteration. `dst_index_out` = 0 for LOGADDEXP, so output overwrites the LHS tile in DEST. |
| **LREG0-LREG3** | SFPU local registers used transiently by the compiler to hold `in0`, `in1`, `result`, and the immediate 0.0f. The exact allocation depends on the compiler's register assignment, but typically: LREG0 for `in0` (loaded via SFPLOAD), LREG1 for `in1` (loaded via SFPLOAD), LREG2 for `result` (SFPADD output, then stored via SFPSTORE). |
| **SFPU DEST write counter** | Auto-incremented via `dst_reg++` (INCRWC) each iteration. Reset/advanced between faces via `SETRWC`. |

### Address Mode Configuration

The address mode for binary SFPU operations is configured in `eltwise_binary_sfpu_configure_addrmod()` during initialization. For `BinaryOp::ADD`, the `SfpuType` passed is `SfpuType::unused`, so only the default ADDR_MOD_7 is configured:

**ADDR_MOD_7** (used for all binary SFPU operations including ADD):
- `srca.incr = 0` -- no auto-increment for source A
- `srcb.incr = 0` -- no auto-increment for source B
- `dest.incr = 0` -- no auto-increment for DEST

This configuration is the same for both Wormhole B0 and Blackhole. The zero-increment ADDR_MOD_7 is chosen because the SFPU kernel manages DEST addressing explicitly through `dst_reg++` (INCRWC) and the params dispatch uses `SETRWC` to advance between faces. ADDR_MOD_7 is used to avoid conflicts with ADDR_MOD_0 and ADDR_MOD_2, which are reserved for the A2D (Accumulate-to-DEST) copy tile path.

The only difference between Wormhole and Blackhole in the LLK binary SFPU layer is in `_llk_math_eltwise_binary_sfpu_start_` and `_llk_math_eltwise_binary_sfpu_done_`: Wormhole additionally calls `math::set_addr_mod_base()` at start and `math::clear_addr_mod_base()` at done, while Blackhole omits these calls. The ADDR_MOD configuration itself (ADDR_MOD_7 with all-zero increments) is identical.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng operation work in TTNN? What is its program factory structure, and how does it handle SFPU vs FPU paths? What kernels does it use?"
   **Reason**: To understand the overall architecture of binary_ng before reading source code.
   **Key Findings**: Confirmed single ProgramFactory with SFPU/FPU path selection via `is_binary_sfpu_op()`. Identified kernel naming convention (sfpu vs non-sfpu variants) and the OpConfig mechanism that maps BinaryOpType to kernel defines. Confirmed that LOGADDEXP uses a decomposition pattern (pre-activations + core op + post-activation).

2. [SFPU] **Query**: "How does add_binary_tile work in the SFPU binary operations? What is the call chain from add_binary_tile through LLK to the ckernel SFPU implementation?"
   **Reason**: To trace the full SFPU dispatch path from the compute API entry point through LLK layers to the core SFPU implementation.
   **Key Findings**: Confirmed the call chain: `add_binary_tile` -> `llk_math_eltwise_binary_sfpu_binop` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_sfpu_binary` -> `_calculate_sfpu_binary_`. Identified that the core implementation lives in `ckernel_sfpu_binary.h` under `tt_llk`, with arch-specific wrappers in the `hw/ckernels` directories.

3. [SFPU] **Query**: "How does add_binary_tile work? Trace the call chain from the compute API through LLK dispatch to the ckernel SFPU implementation. What SFPU instructions does the binary add use?" (asked to `tenstorrent/tt-llk`)
   **Reason**: To get LLK-specific details on the binary SFPU dispatch, including SFPU instruction usage and the params dispatch mechanism.
   **Key Findings**: Confirmed that `BinaryOp::ADD` uses the `SFPADD` instruction. The `_calculate_sfpu_binary_` function uses `sfpi::vFloat` addition which compiles to SFPADD. The params dispatch (`_llk_math_eltwise_binary_sfpu_params_`) handles face iteration with SETRWC between faces.

### Documentation References

1. **Source**: `binary_ng_device_operation.cpp` (lines 16-66)
   **Reason**: To understand the exact conditions under which LOGADDEXP takes the SFPU path.
   **Key Information**: LOGADDEXP requires `a == FLOAT32 && b == FLOAT32` for the SFPU path.

2. **Source**: `binary_ng_utils.cpp` (lines 226-231)
   **Reason**: To understand how LOGADDEXP is decomposed into pre/post activations.
   **Key Information**: `process_lhs = EXP`, `process_rhs = EXP`, `binary_op = ADD`, `postprocess = LOG`.

3. **Source**: `binary_ng_utils.cpp` (lines 377-381, `get_sfpu_init_fn`)
   **Reason**: To identify the exact SFPU function used for the ADD operation.
   **Key Information**: For `SfpuBinaryOp::ADD` with non-integer types, the init/op functions are `add_binary_tile_init()` / `add_binary_tile`.

4. **Source**: `eltwise_utils_sfpu.hpp`
   **Reason**: To understand the PREPROCESS macro that handles pre-activations.
   **Key Information**: PREPROCESS copies tile to DEST, applies activation (EXP), packs to intermediate CB, with `pack_reconfig_data_format` calls around it.

### Confluence References

No Confluence references were needed for this analysis. The binary ADD SFPU kernel uses only the `SFPADD` instruction, whose behavior (element-wise floating-point addition) is straightforward and sufficiently documented by DeepWiki and source code inspection.

### Glean References

No Glean references were needed for this analysis.
