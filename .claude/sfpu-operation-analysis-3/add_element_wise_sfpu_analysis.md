# ADD (element_wise_sfpu) Implementation Analysis

## Overview

The ADD operation via the SFPU path performs element-wise addition of two input tensors using the Special Function Processing Unit (SFPU) instead of the FPU matrix engine. This path is selected when both input tensors share the same data type and that type is one of FLOAT32, INT32, UINT32, or UINT16. The SFPU path copies both input tiles into the DEST register and executes `add_binary_tile` (for float types) or `add_int_tile` (for integer types) as an SFPU function, rather than using the FPU's `add_tiles` instruction.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Path Selection: FPU vs SFPU

The factory selection is performed in `BinaryDeviceOperation::select_program_factory()` (in `binary_device_operation.cpp`). When both tensors have the same shape (no broadcasting needed), the function calls `utils::is_binary_sfpu_op(op, dtype1, dtype2)`. For `BinaryOpType::ADD`, this function returns `true` when `dtype1 == dtype2` and the common type is one of `FLOAT32`, `INT32`, `UINT32`, or `UINT16`. When the check returns true, `ElementWiseMultiCoreSfpu{}` is selected; otherwise `ElementWiseMultiCore{}` (the FPU path) is selected. The FPU path is therefore used for BF16 and other non-32-bit / non-UINT16 types, or when broadcasting is required.

## Work Unit Definition

One work unit is one **tile** (32x32 elements). The compute kernel processes tiles in blocks of `per_core_block_size` tiles, iterating over `per_core_block_cnt` blocks. For non-sharded tensors, `block_size` is 1 and `block_cnt` equals the number of tiles assigned to the core. For sharded tensors, `block_size` is the largest power-of-2 factor of `num_tiles_per_shard`, and `block_cnt = num_tiles_per_shard / block_size`.

## Tensor Format and Layout

### Input Tensors

| Property | Input A (src0) | Input B (src1) |
|---|---|---|
| **Dim Convention** | NHWC (logical) | NHWC (logical) |
| **Tensor Layout** | TILE (32x32) | TILE (32x32) |
| **Memory Layout** | Interleaved or Sharded | Interleaved or Sharded |
| **Buffer Type** | DRAM or L1 | DRAM or L1 |
| **Data Type** | FLOAT32, INT32, UINT32, or UINT16 (must match B) | FLOAT32, INT32, UINT32, or UINT16 (must match A) |

### Output Tensor

| Property | Output |
|---|---|
| **Dim Convention** | NHWC (logical) |
| **Tensor Layout** | TILE (32x32) |
| **Memory Layout** | Interleaved or Sharded |
| **Buffer Type** | DRAM or L1 |
| **Data Type** | Same as input (FLOAT32/INT32/UINT32/UINT16) |

### Layout Transformations

No tilize/untilize operations are performed. Data remains in tile layout throughout the pipeline. No format conversions occur for the basic ADD case without fused activations.

## Data Flow Pattern

1. **Reader kernel** reads tiles for both input A and input B from DRAM/L1 (or signals sharded data availability) into CB c_0 and CB c_1 respectively, one tile at a time (interleaved) or all tiles at once (sharded).
2. **Compute kernel** waits for tiles in CB c_0 and CB c_1 (or c_3/c_4 if pre-scaling is active).
3. Both input tiles are copied into DEST registers: input A goes to DEST[i*2], input B goes to DEST[i*2+1].
4. The SFPU binary operation (`add_binary_tile` or `add_int_tile`) is invoked, reading from DEST[i*2] and DEST[i*2+1], writing the result back to DEST[i*2].
5. The result tile at DEST[i*2] is packed into CB c_2.
6. **Writer kernel** reads tiles from CB c_2 and writes them to the output buffer in DRAM/L1 (or leaves them in place for sharded output).

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|---|---|---|---|---|---|---|---|---|
| c_0 | cb_src0 | Input A tiles | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | 1 (interleaved read) or `per_core_block_size` (compute) | Sharded: Single; Interleaved: Double | Reader | Compute | Full pipeline |
| c_1 | cb_src1 | Input B tiles | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | 1 (interleaved read) or `per_core_block_size` (compute) | Sharded: Single; Interleaved: Double | Reader | Compute | Full pipeline |
| c_3 | cb_interim0 | Pre-scaled input A (conditional) | `max_block_size` | `per_core_block_size` | Single | Compute (pre-scale) | Compute (main) | Conditional: only when `SFPU_OP_INIT_PRE_IN0_0` defined |
| c_4 | cb_interim1 | Pre-scaled input B (conditional) | `max_block_size` | `per_core_block_size` | Single | Compute (pre-scale) | Compute (main) | Conditional: only when `SFPU_OP_INIT_PRE_IN1_0` defined |
| c_2 | cb_output | Output tiles | Sharded/block-width: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | `per_core_block_size` (compute push) / 1 (writer pop, interleaved) | Sharded: Single; Interleaved: Double | Compute | Writer | Full pipeline |

**Note for ADD specifically**: For basic ADD (FLOAT32 or integer types) without `input_tensor_a_activation`, the pre-scale defines `SFPU_OP_INIT_PRE_IN0_0` and `SFPU_OP_INIT_PRE_IN1_0` are NOT defined. Therefore CB c_3 and CB c_4 are NOT created. The compute kernel reads directly from c_0 and c_1.

## Pipeline Pattern Summary

- **Interleaved path**: CB c_0, c_1, and c_2 each have capacity `2 * max_block_size` tiles. The reader pushes 1 tile at a time, compute consumes `per_core_block_size` (which equals `max_block_size` = 1 for interleaved), and writer pops 1 tile at a time. With capacity of 2 tiles, this provides **double-buffering**, allowing reader and compute to overlap.
- **Sharded path**: CB c_0 and c_1 are globally allocated (backed by the input tensor's L1 buffer). All shard tiles are signaled at once. CB c_2 is similarly backed by the output buffer. This is effectively **single-buffered** -- all tiles are produced then consumed in one shot.

## Index Calculations

The reader kernel uses `TensorAccessor` for DRAM address resolution. For interleaved tensors, `noc_async_read_tile(tile_id, accessor, l1_addr)` is called with a linear tile ID. The `TensorAccessor` maps this to the correct bank and offset.

For the block/width sharded path, tile IDs are computed as:
```
start_id = (core_index / num_shards_per_width) * (block_height * block_width * num_shards_per_width)
         + (core_index % num_shards_per_width) * block_width
```
Within each shard, tiles are read row-by-row: `row_start_tile_id += num_cores_y * block_width` per row.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile-by-tile reads. Each tile is read via `noc_async_read_tile` with an `noc_async_read_barrier` after each pair of reads (one from each input). This is a linear scan through the tile ID space.
- **Block/Width sharded reader**: Two-level nested loop -- outer over block height, inner over block width. Row stride = `num_cores_y * block_width` tiles.
- **Sharded (both inputs)**: No reads; the reader just signals `cb_reserve_back` + `cb_push_back` on the full shard.

### Write Pattern
- **Interleaved**: Sequential single-tile writes via `noc_async_write_page` with `noc_async_writes_flushed` after each tile, followed by a final `noc_async_write_barrier`.
- **Sharded output**: No writes; the writer just calls `cb_wait_front` on all output tiles (data is already in place).
- **Block/Width sharded to interleaved**: The special writer kernel reads the full block from CB, then writes unpadded tiles row-by-row to DRAM, skipping padding tiles.

## Core Distribution Strategy

| Property | Value |
|---|---|
| **Grid Topology** | Rectangular grid from `operation_attributes.worker_grid` |
| **Work Splitting** | `split_work_to_cores()` for interleaved; shard grid for sharded |
| **Row Major** | Yes (interleaved default); shard orientation for sharded |
| **Core Group 1** | Cores with `ceil(num_tiles / num_cores)` tiles |
| **Core Group 2** | Remaining cores with `floor(num_tiles / num_cores)` tiles (interleaved only) |
| **Remainder Handling** | Two core groups with different tile counts; excess cores receive 0 tiles |
| **Zero-start optimization** | When the grid is a single rectangle starting at (0,0) and any shard grid also starts at (0,0), a faster `grid_to_cores` path is used instead of the generic `corerange_to_cores` |

## Arguments

### Compile-Time Arguments

**Reader Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `block_or_width_sharded` | uint32_t | 1 if block or width sharded, 0 otherwise |
| 1+ | `TensorAccessorArgs(src0)` | variable | Accessor args for input A (only if not IN0_SHARDED) |
| N+ | `TensorAccessorArgs(src1)` | variable | Accessor args for input B (only if not IN1_SHARDED) |

**Writer Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `output_cb_index` | uint32_t | CB index for output (c_2) |
| 1+ | `TensorAccessorArgs(dst)` | variable | Accessor args for output buffer |

**Compute Kernel:**

No indexed compile-time args. Configuration is via `ComputeConfig`:
- `fp32_dest_acc_en`: true if output is Float32/Int32/UInt32
- `unpack_to_dest_mode`: `UnpackToDestFp32` for all CBs (since op is not POWER)
- `defines`: The operation-specific defines from `get_defines_fp32()`

### Runtime Arguments

**Reader Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `src0_addr` | uint32_t | Base address of input A buffer |
| 1 | `src1_addr` | uint32_t | Base address of input B buffer |
| 2 | `num_tiles` | uint32_t | Total tiles this core processes |
| 3 | `start_id` | uint32_t | Starting tile ID for this core |
| 4 | `block_height` | uint32_t | Shard block height in tiles (0 if not sharded) |
| 5 | `block_width` | uint32_t | Shard block width in tiles (0 if not sharded) |
| 6 | `num_cores_y` | uint32_t | Number of shards per width dimension |

**Compute Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks to process |
| 1 | `per_core_block_size` | uint32_t | Tiles per block |

**Writer Kernel (interleaved, non-block-sharded):**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `dst_addr` | uint32_t | Base address of output buffer |
| 1 | `num_pages` | uint32_t | Number of tiles to write |
| 2 | `start_id` | uint32_t | Starting tile ID for writes |

**Writer Kernel (block/width sharded to interleaved):**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `dst_addr` | uint32_t | Base address of output buffer |
| 1 | `block_height_tiles` | uint32_t | Block height in tiles |
| 2 | `block_width_tiles` | uint32_t | Block width in tiles |
| 3 | `unpadded_block_height_tiles` | uint32_t | Unpadded block height |
| 4 | `unpadded_block_width_tiles` | uint32_t | Unpadded block width |
| 5 | `output_width_tiles` | uint32_t | Full output width in tiles |
| 6 | `block_num_tiles` | uint32_t | Total tiles in block |
| 7 | `start_id_offset` | uint32_t | Starting tile offset |
| 8 | `start_id_base` | uint32_t | Base tile ID (0) |

## Kernel Implementations

| Kernel | File | Type | Assigned Cores |
|---|---|---|---|
| Reader | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` | DataMovement (Reader) | All worker cores |
| Compute | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` | Compute | All worker cores |
| Writer (interleaved) | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | DataMovement (Writer) | All worker cores |
| Writer (block-sharded to interleaved) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` | DataMovement (Writer) | All worker cores |

### Reader Kernel

| Property | Value |
|---|---|
| **File** | `reader_binary_interleaved_start_id.cpp` |
| **Assigned Cores** | All worker cores in `all_device_cores` |

**Key Logic:**
- If `IN0_SHARDED` is defined, immediately calls `cb_reserve_back(cb_id_in0, num_tiles)` and `cb_push_back(cb_id_in0, num_tiles)` to signal that data is already in place. Same for `IN1_SHARDED`.
- For non-sharded inputs with `block_or_width_sharded` mode: uses a nested loop (height x width) to read tiles. Row stride is `num_cores_y * block_width`, which maps the 2D shard layout to the linear tile space.
- For non-sharded inputs without block/width sharding: simple sequential loop from `start_id` to `start_id + num_tiles`, reading one tile at a time.
- Each non-sharded tile read uses `noc_async_read_tile` via `TensorAccessor`, followed by `noc_async_read_barrier` before pushing to the CB. This serializes reads with a barrier after each tile pair.
- **Synchronization**: Pushes tiles to CB c_0 and CB c_1. The compute kernel consumes from these CBs via `cb_wait_front`.

### Compute Kernel

| Property | Value |
|---|---|
| **File** | `eltwise_binary_sfpu_kernel.cpp` |
| **Assigned Cores** | All worker cores in `all_device_cores` |

**Key Logic:**
- Initializes with `unary_op_init_common(cb_in0, cb_out0)`.
- For ADD with FLOAT32: defines `BINOP_INIT` = `add_binary_tile_init();` and `BINARY_SFPU_OP` = `add_binary_tile(i*2, i*2+1, i*2);`. No pre-scaling defines are set.
- For ADD with INT32: defines `ADD_INT_INIT` = `add_int_tile_init();` and `BINARY_SFPU_OP` = `add_int_tile<DataFormat::Int32>(i*2, i*2+1, i*2);`. Similar patterns for UINT32 and UINT16.
- Main loop iterates `per_core_block_cnt` times. Each iteration:
  1. Waits for `per_core_block_size` tiles in `cb_inp0` (c_0) and `cb_inp1` (c_1).
  2. Reserves `per_core_block_size` tiles in `cb_out0` (c_2).
  3. Acquires tile registers (`tile_regs_acquire` + `tile_regs_wait`).
  4. Copies all input A tiles to DEST[i*2] via `copy_tile(cb_inp0, i, i*2)`.
  5. Switches data type context with `copy_tile_to_dst_init_short_with_dt`.
  6. For each input B tile: copies to DEST[i*2+1], runs the init macro, then runs the SFPU op macro, then packs result from DEST[i*2] to `cb_out0`.
  7. Commits and releases tile registers.
  8. Pops tiles from input CBs, pushes to output CB.
- **Synchronization**: Waits on CB c_0 and c_1 via `cb_wait_front`, pushes to CB c_2 via `cb_push_back`. Uses `tile_regs_acquire/commit/wait/release` for DEST register synchronization between unpack and pack pipelines.

### Writer Kernel (Interleaved)

| Property | Value |
|---|---|
| **File** | `writer_unary_interleaved_start_id.cpp` |
| **Assigned Cores** | All worker cores in `all_device_cores` |

**Key Logic:**
- If `OUT_SHARDED` is defined, just calls `cb_wait_front(cb_id_out, num_pages)` -- data is already in the output buffer via the globally-allocated CB.
- For non-sharded output: sequential loop writing one tile at a time via `noc_async_write_page` using `TensorAccessor`. Each write is followed by `noc_async_writes_flushed`, then `cb_pop_front(cb_id_out, 1)`. Final `noc_async_write_barrier` ensures all writes complete.
- **Synchronization**: Waits on CB c_2 via `cb_wait_front`, pops via `cb_pop_front`.

### Writer Kernel (Block-Sharded to Interleaved)

| Property | Value |
|---|---|
| **File** | `writer_unary_sharded_blocks_interleaved_start_id.cpp` |
| **Assigned Cores** | All worker cores (only used when `block_or_width_sharded && !out_sharded`) |

**Key Logic:**
- Waits for the entire block (`cb_wait_front(cb_id_out, block_num_tiles)`), then writes unpadded tiles to DRAM using a nested height x width loop. Skips padding tiles by advancing the L1 read pointer past padded width.
- Row stride in output is `output_width_tiles` tiles.
- **Synchronization**: Single `cb_wait_front` for all block tiles, single `cb_pop_front` after all writes complete.

## Implementation Notes

- **Program factory variants**: Two program factories exist for non-broadcast binary operations: `ElementWiseMultiCore` (FPU path) and `ElementWiseMultiCoreSfpu` (SFPU path). Selection is based on `is_binary_sfpu_op()` which checks the operation type and data types. Additional broadcast factories exist (`BroadcastWidthMultiCore`, `BroadcastHeightMultiCore`, `BroadcastHeightAndWidthMultiCore`) but these do not have SFPU variants.
- **Type-based operation variants**: For ADD, the SFPU path supports FLOAT32 (using `add_binary_tile`), INT32 (using `add_int_tile<DataFormat::Int32>`), UINT32 (using `add_int_tile<DataFormat::UInt32>`), and UINT16 (using `add_int_tile<DataFormat::UInt16>`). BF16 ADD uses the FPU path instead.
- **UnpackToDestFP32 mode**: For all ops except POWER, `UnpackToDestMode::UnpackToDestFp32` is set on all input and interim CBs (c_0, c_1, c_3, c_4). This ensures data is unpacked to full FP32 precision in the DEST register regardless of the source format.
- **Broadcast type selection**: N/A for this SFPU path. Broadcasting requires equal tensor shapes; when shapes differ, the FPU-based broadcast factories are selected instead.
- **Sharding support and constraints**: All three sharding modes (height, width, block) are supported for inputs and output independently. The program factory detects sharding from any of the three tensors (src0, src1, output) and configures globally-allocated circular buffers accordingly. A special writer kernel (`writer_unary_sharded_blocks_interleaved_start_id`) handles the case where inputs are block/width-sharded but output is interleaved.
- **FP32 dest accumulation**: Enabled when the output data format is Float32, Int32, or UInt32 (via `fp32_dest_acc_en` in `ComputeConfig`). This keeps the DEST accumulator in full 32-bit precision.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel functions that the compute kernel dispatches to. The ADD operation has **two distinct SFPU paths**: a floating-point path (`add_binary_tile` using SFPI abstractions) and an integer path (`add_int_tile` using raw `TT_`/`TTI_` instructions).

### SFPU Abstraction Layers

**Floating-point path** (`add_binary_tile`):

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_binary.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

**Integer path** (`add_int_tile`):

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/add_int_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_add_int.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_add_int.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` (shared with float path) |

### Call Chain

**Floating-point path:**

1. The compute kernel invokes `add_binary_tile_init()` and `add_binary_tile(i*2, i*2+1, i*2)` (via the `BINOP_INIT` and `BINARY_SFPU_OP` macros).
2. `add_binary_tile_init()` calls `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::ADD>()`, which calls `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` (configures SFPU config reg, ADDR_MOD_7, resets counters) followed by `sfpu_binary_init<APPROX, BinaryOp::ADD>()` which calls `_sfpu_binary_init_<APPROX, BinaryOp::ADD>()`. For ADD, the init function is a no-op (no reciprocal or log initialization needed).
3. `add_binary_tile(idst0, idst1, odst)` calls `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>(idst0, idst1, odst)`, which calls `_llk_math_eltwise_binary_sfpu_params_<APPROX>(calculate_sfpu_binary<APPROX, BinaryOp::ADD, 8, false>, idst0, idst1, odst, VectorMode::RC)`.
4. The params dispatch function calls `_llk_math_eltwise_binary_sfpu_start_` (sets DEST write address, stalls until SFPU ready), then iterates over 4 faces in RC mode, calling `calculate_sfpu_binary(...)` per face with `TTI_SETRWC` to advance the DEST pointer between faces.
5. `calculate_sfpu_binary` is a thin wrapper that calls `_calculate_sfpu_binary_<APPROX, BinaryOp::ADD, 8>(idst0, idst1, odst)`, the core SFPU implementation.

**Integer path:**

1. The compute kernel invokes `add_int_tile_init()` and `add_int_tile<DataFormat::Int32>(i*2, i*2+1, i*2)` (via the `ADD_INT_INIT` and `BINARY_SFPU_OP` macros).
2. `add_int_tile_init()` calls `llk_math_eltwise_binary_sfpu_add_int_init<APPROX>()`, which calls `llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROX>()` (same init as the float path, no additional SFPU-specific init callback).
3. `add_int_tile<DataFormat::Int32>(idst0, idst1, odst)` calls `llk_math_eltwise_binary_sfpu_add_int<APPROX, 8, DataFormat::Int32, false>(idst0, idst1, odst)`, which resolves `INSTRUCTION_MODE = InstrModLoadStore::INT32` and calls `_llk_math_eltwise_binary_sfpu_params_<APPROX>(_add_int_<APPROX, 8, INT32, false>, idst0, idst1, odst, VectorMode::RC)`.
4. The params dispatch iterates over 4 faces identically to the float path, calling `_add_int_` per face.

### Parameters Dispatch Summary

The parameters dispatch function `_llk_math_eltwise_binary_sfpu_params_` is shared between the float and integer paths. It is identical on Wormhole and Blackhole.

- **Vector mode**: `VectorMode::RC` (the default for both `add_binary_tile` and `add_int_tile`). In RC mode, all 4 faces of the 32x32 tile are processed. Each face is 16x16 = 256 elements, processed as 8 iterations of 32 SFPU lanes (4 rows x 8 columns per SFPU vector).
- **Operation invocation**: The core SFPU function is called once per face, for a total of 4 calls per tile. Each call internally loops 8 iterations (`ITERATIONS=8`), processing 32 elements per iteration, totaling 256 elements per face and 1024 elements per tile.
- **DEST address progression**: After each face, two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` instructions advance the DEST read/write pointer by 8 rows each (total 16 rows = one face width in DEST). The SFPU function itself uses `dst_reg++` (float path) or explicit `ADDR_MOD_3` (integer path) to advance within the face between iterations. After all 4 faces, `_llk_math_eltwise_binary_sfpu_done_()` clears the DEST address.

### Annotated SFPU Kernel Source

This operation has two distinct SFPU kernels. Both use Style A annotation.

#### Floating-Point Kernel (`_calculate_sfpu_binary_` with `BinaryOp::ADD`)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{ // APPROXIMATION_MODE=true, BINOP=BinaryOp::ADD, ITERATIONS=8
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr std::uint32_t dst_tile_size_sfpi = 32;
        sfpi::vFloat in0                           = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // load 32 lanes from DEST tile 0
        sfpi::vFloat in1                           = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // load 32 lanes from DEST tile 1
        sfpi::vFloat result                        = 0.0f;

        if constexpr (BINOP == BinaryOp::ADD)
        {
            result = in0 + in1; // SFPADD instruction: element-wise FP32 addition
        }
        // SUB, MUL, DIV, RSUB, POW, XLOGY branches omitted (not active for ADD)

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // store 32 lanes back to DEST output tile
        sfpi::dst_reg++; // advance DEST pointer by 1 row (SFP_DESTREG_STRIDE=2 DEST rows)
    }
}
```

The `_sfpu_binary_init_` function for `BinaryOp::ADD` is a no-op -- the `constexpr if` only enters for `DIV`, `POW`, or `XLOGY`:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{ // BINOP=BinaryOp::ADD -- no initialization needed
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW)
    {
        _init_sfpu_reciprocal_<false>(); // not taken for ADD
    }
    else if constexpr (BINOP == BinaryOp::XLOGY)
    {
        _init_log_<APPROXIMATION_MODE>(); // not taken for ADD
    }
}
```

#### Integer Kernel (`_add_int_`)

The integer kernel uses raw `TT_`/`TTI_` instructions but has no condition code manipulation (no `SFPSETCC`, `SFPENCC`, or CC-modifying modifiers on `SFPIADD`). The `SFPIADD` instruction here uses `instr_mod=4` which maps to `CC_NONE`, explicitly disabling CC updates. This is Style A.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_add_int.h

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _add_int_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{ // ITERATIONS=8, INSTRUCTION_MODE=INT32 (for Int32/UInt32) or LO16 (for UInt16), SIGN_MAGNITUDE_FORMAT=false
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // sfpload_instr_mod: for non-sign-magnitude, equals INSTRUCTION_MODE directly (INT32=4 or LO16=8)
    constexpr int sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? INT32_2S_COMP : to_underlying(INSTRUCTION_MODE);

    constexpr std::uint32_t dst_tile_size = 64; // each tile occupies 64 DEST rows

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Load operand A from DEST into LREG0 as integer
        TT_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        // Load operand B from DEST into LREG1 as integer
        TT_SFPLOAD(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        // Integer add: LREG0 = LREG0 + LREG1; instr_mod=4 means CC_NONE (no CC update)
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        // Store result from LREG0 back to DEST
        TT_SFPSTORE(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, dst_index_out * dst_tile_size);
        sfpi::dst_reg++; // advance DEST pointer by 1 row
    }
}
```

### SFPU Instructions Used

**Floating-point path (`_calculate_sfpu_binary_` with ADD):**

| Instruction | SFPI Abstraction | Description |
|---|---|---|
| `SFPLOAD` | `dst_reg[offset]` (read) | Loads 32 FP32 lanes from DEST register into an SFPU LREG. Used twice per iteration to load `in0` and `in1` from their respective DEST tiles. |
| `SFPADD` | `in0 + in1` | Performs element-wise FP32 addition across 32 SIMD lanes. This is the core compute instruction. Two-cycle latency. |
| `SFPSTORE` | `dst_reg[offset] = result` (write) | Stores 32 FP32 lanes from an SFPU LREG back to the DEST register. Used once per iteration to write the result. |
| `SETRWC` | `dst_reg++` (implicit via params dispatch) | Advances the DEST read/write counter. Used within the iteration loop (via `dst_reg++`) and between faces (via explicit `TTI_SETRWC` in the params dispatch). |

**Integer path (`_add_int_`):**

| Instruction | Raw Form | Description |
|---|---|---|
| `SFPLOAD` | `TT_SFPLOAD(LREG, instr_mod, ADDR_MOD_3, offset)` | Loads 32 integer lanes from DEST into an SFPU LREG. The `instr_mod` parameter selects the data format: `INT32` (value 4) for 32-bit integers, `LO16` (value 8) for 16-bit unsigned integers. |
| `SFPIADD` | `TTI_SFPIADD(0, LREG1, LREG0, 4)` | Integer addition: `LREG0 = LREG0 + LREG1`. The `instr_mod=4` (`CC_NONE`) disables condition code side effects. The immediate operand (first arg) is 0, so this is purely register-to-register. |
| `SFPSTORE` | `TT_SFPSTORE(LREG, instr_mod, ADDR_MOD_3, offset)` | Stores 32 integer lanes from an SFPU LREG back to DEST. Same `instr_mod` as SFPLOAD for format consistency. |
| `SETRWC` | `dst_reg++` / `TTI_SETRWC` | Advances the DEST pointer between iterations and between faces. |

### SFPU Register Usage

**Floating-point path:**

| Register | Usage |
|---|---|
| **DEST[idst0 * 32]** | Input tile A. Read by `dst_reg[dst_index_in0 * dst_tile_size_sfpi]` to load `in0`. Also serves as the output location when `odst == idst0` (which is always the case for ADD: `odst = i*2 = idst0`). |
| **DEST[idst1 * 32]** | Input tile B. Read by `dst_reg[dst_index_in1 * dst_tile_size_sfpi]` to load `in1`. Not written to. |
| **LREG0-LREG3** | Used implicitly by the SFPI compiler for `in0`, `in1`, `result`, and intermediate values. The SFPI abstraction manages LREG allocation transparently. |

**Integer path:**

| Register | Usage |
|---|---|
| **DEST[idst0 * 64]** | Input tile A. Loaded into LREG0 via `SFPLOAD`. Note the DEST tile size is 64 (not 32) because raw SFPLOAD uses physical DEST rows, while SFPI `dst_reg` uses logical rows with stride 2. |
| **DEST[idst1 * 64]** | Input tile B. Loaded into LREG1 via `SFPLOAD`. |
| **DEST[odst * 64]** | Output tile. Written from LREG0 via `SFPSTORE`. When `odst == idst0`, this overwrites the input A tile in-place. |
| **LREG0** | Holds operand A, then accumulates the result of `LREG0 + LREG1`. Stored back to DEST. |
| **LREG1** | Holds operand B. Read-only in the `SFPIADD` instruction (source operand). |

### Address Mode Configuration

**ADDR_MOD_7** (configured in `eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>`):

This is configured during the init phase (`_llk_math_eltwise_binary_sfpu_init_`). For `SfpuType::unused` (which is what ADD uses), only ADDR_MOD_7 is set:

| Field | Value | Description |
|---|---|---|
| `srca.incr` | 0 | No auto-increment for source A |
| `srcb.incr` | 0 | No auto-increment for source B |
| `dest.incr` | 0 | No auto-increment for DEST |

This is the same on both Wormhole and Blackhole. ADDR_MOD_7 is used by the params dispatch infrastructure (not directly by the core SFPU kernel). The zero-increment configuration means DEST addressing is managed explicitly through `TTI_SETRWC` calls between faces rather than auto-incrementing.

The conditional ADDR_MOD_6 (with `dest.incr = 2`) is only configured for `mul_int32`, `mul_uint16`, `max`, `min`, and their int32/uint32 variants -- not for ADD.

**ADDR_MOD_3** (used in `_add_int_`):

ADDR_MOD_3 is referenced by the integer kernel's `TT_SFPLOAD` and `TT_SFPSTORE` instructions. This address mode is not explicitly configured by the binary SFPU init function. It relies on the default initialization state or prior configuration from `math::reset_counters(p_setrwc::SET_ABD_F)` called during init. In practice, ADDR_MOD_3 functions as a no-auto-increment mode for the integer SFPLOAD/SFPSTORE operations, with the DEST pointer advancement handled by `dst_reg++` between iterations and `TTI_SETRWC` between faces.

**Wormhole vs Blackhole differences:**

The ADDR_MOD configuration is identical between Wormhole and Blackhole for this operation. The only architectural difference is in `_llk_math_eltwise_binary_sfpu_done_()`:
- **Wormhole**: Calls `math::clear_dst_reg_addr()`, then `TTI_STALLWAIT(STALL_CFG, WAIT_SFPU)`, then `math::clear_addr_mod_base()`.
- **Blackhole**: Only calls `math::clear_dst_reg_addr()` (no stall wait, no addr_mod_base clear).

Additionally, `_llk_math_eltwise_binary_sfpu_start_`:
- **Wormhole**: Calls `set_dst_write_addr`, `set_addr_mod_base()`, then `TTI_STALLWAIT`.
- **Blackhole**: Calls `set_dst_write_addr`, then `TTI_STALLWAIT` (no `set_addr_mod_base`).

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary eltwise SFPU program factory work? What kernels does it use for reader, compute, and writer? How does it differ from the FPU binary path?"
   **Reason**: Initial reconnaissance to understand the overall architecture of the SFPU binary path before reading source code.
   **Key Findings**: Confirmed the three kernels used (reader_binary_interleaved_start_id, eltwise_binary_sfpu_kernel, writer_unary_interleaved_start_id), the factory selection mechanism via `is_binary_sfpu_op`, and the fundamental difference: SFPU copies tiles to DEST then invokes SFPU functions, whereas FPU operates directly on tiles via matrix engine instructions.

2. [SFPU] **Query**: "How does the add_binary_tile SFPU function work? What is the call chain from add_binary_tile through LLK layers down to the core ckernel SFPU implementation?"
   **Reason**: Needed to trace the full call chain from the compute API `add_binary_tile` through the LLK dispatch layers to the core SFPU implementation.
   **Key Findings**: Confirmed the call chain: `add_binary_tile` -> `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_sfpu_binary` -> `_calculate_sfpu_binary_`. Identified the init chain similarly. Located the API header at `eltwise_binary_sfpu.h`.

3. [SFPU] **Query**: "How does add_binary_tile work in the LLK layer? What is the call chain from the compute API through llk_math_eltwise_binary_sfpu to the core ckernel_sfpu_add function? What SFPU instructions does the add binary operation use? Also explain add_int_tile."
   **Reason**: Needed deeper detail on the LLK layer mechanics, especially for the integer path and SFPU instruction usage.
   **Key Findings**: Confirmed that the float ADD path uses SFPADD instruction, that `_sfpu_binary_init_` is a no-op for ADD, and that the params dispatch uses `ADDR_MOD_7` with zero increments. Learned that the integer path follows a parallel call chain through `llk_math_eltwise_binary_sfpu_add_int` using `SFPLOAD`/`SFPIADD`/`SFPSTORE` raw instructions.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp` (lines 22-66)
   **Reason**: Needed to understand the exact conditions under which the SFPU path is selected for ADD.
   **Key Information**: ADD uses SFPU path when `a == b && (a == FLOAT32 || a == INT32 || a == UINT32 || a == UINT16)`.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 184-552)
   **Reason**: Needed to understand what preprocessor defines are generated for ADD in the SFPU path.
   **Key Information**: For FLOAT32 ADD, defines `BINOP_INIT` = `add_binary_tile_init()` and `BINARY_SFPU_OP` = `add_binary_tile(i*2, i*2+1, i*2)`. For INT32 ADD, defines `ADD_INT_INIT` = `add_int_tile_init()` and uses `add_int_tile<DataFormat::Int32>`.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp`
   **Reason**: Needed to understand the runtime argument setup and core distribution logic shared between FPU and SFPU paths.
   **Key Information**: The `set_eltwise_binary_runtime_args` template function handles both creation and override cases, with optimized zero-start-grid paths and two-group core splitting for load balancing.
