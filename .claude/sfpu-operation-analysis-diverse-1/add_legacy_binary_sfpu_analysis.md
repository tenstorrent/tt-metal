# ADD (Legacy Binary SFPU) Implementation Analysis

## Overview

The ADD operation performs element-wise addition of two input tensors using the SFPU (Special Function Processing Unit) path. This is a "legacy" binary SFPU operation in that it uses the older `BinaryDeviceOperation` framework (as opposed to the newer `binary_ng` infrastructure). The SFPU path is selected when both input tensors share the same data type and that type is one of FLOAT32, INT32, UINT32, or UINT16.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Path Selection: FPU vs SFPU

The `BinaryDeviceOperation::select_program_factory` function (in `binary_device_operation.cpp`) determines which program factory to use. The decision tree is:

1. If a scalar operand is present, the `BroadcastHeightAndWidthMultiCore` factory is selected (neither FPU nor SFPU element-wise path).
2. If both tensors have the same height and width (no broadcasting needed), the function calls `utils::is_binary_sfpu_op(op, dtype1, dtype2)`.
3. For `BinaryOpType::ADD`, `is_binary_sfpu_op` returns `true` when `a == b` and the shared dtype is one of `{FLOAT32, INT32, UINT32, UINT16}`. When true, `ElementWiseMultiCoreSfpu` is returned; otherwise `ElementWiseMultiCore` (FPU path) is used.
4. If the tensors have different shapes, various broadcast factories are selected instead.

**The SFPU path is selected for ADD when**: both input tensors have identical shapes AND identical data types AND the data type is FLOAT32, INT32, UINT32, or UINT16.

## Work Unit Definition

One work unit is one **tile** (32x32 elements). Tiles are grouped into blocks of size `max_block_size` (for non-sharded) or determined by shard dimensions (for sharded). The compute kernel processes `per_core_block_cnt` blocks, each containing `per_core_block_size` tiles.

## Tensor Format and Layout

### Input Tensor(s)

| Property | Input A (src0) | Input B (src1) |
|---|---|---|
| Dimension Convention | NHWC (last dim is width) | NHWC (last dim is width) |
| Tensor Layout | TILE (32x32) | TILE (32x32) |
| Memory Layout | Interleaved or Sharded | Interleaved or Sharded |
| Buffer Type | DRAM or L1 | DRAM or L1 |
| Data Type | FLOAT32, INT32, UINT32, or UINT16 | Same as Input A |

### Output Tensor(s)

| Property | Output |
|---|---|
| Dimension Convention | NHWC (last dim is width) |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Interleaved or Sharded |
| Buffer Type | DRAM or L1 |
| Data Type | Same as inputs |

### Layout Transformations

No tilize/untilize conversions are performed. Both inputs and outputs must already be in tiled layout. No resharding is done within the operation itself.

## Data Flow Pattern

1. **Reader kernel** reads tiles from input A (DRAM/L1) into CB c_0, and input B into CB c_1. For sharded inputs, the CB is backed by the shard buffer directly (globally allocated address), so the reader just does `cb_reserve_back`/`cb_push_back` to make the data available.
2. **Compute kernel** waits for tiles in CB c_0 (or c_3 if pre-scaling) and CB c_1 (or c_4 if pre-scaling). For each block:
   - If pre-input0 activation is defined: copies tiles from c_0 to DST, applies SFPU activation, packs to c_3.
   - If pre-input1 activation is defined: copies tiles from c_1 to DST, applies SFPU activation, packs to c_4.
   - Copies input A tiles to even DST slots (i*2) and input B tiles to odd DST slots (i*2+1).
   - Executes the binary SFPU operation (`add_binary_tile` for FP32, or `add_int_tile` for integer types).
   - Packs result from DST slot i*2 into CB c_2.
3. **Writer kernel** reads tiles from CB c_2 and writes them to the output buffer in DRAM/L1. For sharded output, it simply waits for all tiles in c_2.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|---|---|---|---|---|---|---|---|---|
| c_0 | cb_src0 | Input A tiles | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | 1 (reader pushes 1 at a time for interleaved) | Double-buffered (interleaved) or full-shard (sharded) | Reader | Compute | Full program |
| c_1 | cb_src1 | Input B tiles | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | 1 (reader pushes 1 at a time for interleaved) | Double-buffered (interleaved) or full-shard (sharded) | Reader | Compute | Full program |
| c_2 | cb_out0 | Output tiles | Sharded/block-width: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | `per_core_block_size` (compute pushes a block) | Double-buffered (interleaved) or full-shard (sharded) | Compute | Writer | Full program |
| c_3 | cb_inp0 | Interim for pre-scaled input A | `max_block_size` | `per_core_block_size` | Single-buffered | Compute (pre-scale phase) | Compute (main phase) | Conditional (only if `SFPU_OP_INIT_PRE_IN0_0` defined) |
| c_4 | cb_inp1 | Interim for pre-scaled input B | `max_block_size` | `per_core_block_size` | Single-buffered | Compute (pre-scale phase) | Compute (main phase) | Conditional (only if `SFPU_OP_INIT_PRE_IN1_0` defined) |

**Note on ADD specifically**: For a plain ADD (no `input_tensor_a_activation`), neither c_3 nor c_4 are created. The pre-scaling CBs are only allocated when fused pre-input activations are requested (e.g., for HYPOT which squares both inputs before adding).

## Pipeline Pattern Summary

- **Interleaved path**: CB c_0 and c_1 have capacity `2 * max_block_size` with the reader pushing 1 tile at a time, enabling double-buffering overlap between reader and compute. CB c_2 similarly has `2 * max_block_size` capacity with compute pushing `block_size` tiles, enabling double-buffering between compute and writer.
- **Sharded path**: CBs are backed by the full shard in L1, so there is no streaming overlap -- all data is already resident. The reader is a no-op (just marks the CB as available), and the writer waits for all output tiles at once.

## Index Calculations

- **Interleaved (non-sharded)**: Tiles are linearly indexed. Each core gets a contiguous range `[start_id, start_id + num_tiles_per_core)`. The `TensorAccessor` handles mapping from tile index to physical DRAM bank address.
- **Block/width sharded**: Tile IDs are computed as `start_id = (core_idx / num_shards_per_width) * (block_height * block_width * num_shards_per_width) + (core_idx % num_shards_per_width) * block_width`. The reader iterates in a 2D pattern (block_height x block_width) with row stride of `num_cores_y * block_width`.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile-by-tile reads via `noc_async_read_tile`. Each tile read is followed by `noc_async_read_barrier()` before the next tile, so reads are serialized per tile.
- **Sharded**: No NoC reads needed -- data is already in L1 shard buffers.
- **Block/width sharded (non-sharded input)**: 2D strided reads -- iterates block_width tiles per row, then jumps by `num_cores_y * block_width` tiles for the next row.

### Write Pattern
- **Interleaved**: Sequential tile-by-tile writes via `noc_async_write_page`, with `noc_async_writes_flushed()` after each tile.
- **Sharded output**: No NoC writes -- output CB is backed by the output shard buffer.
- **Block/width sharded to interleaved**: 2D strided writes -- iterates unpadded_block_width tiles per row, strides by output_width tiles between rows.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Rectangular (single CoreRange starting at (0,0) preferred for fast path) |
| Work Splitting | `split_work_to_cores` for interleaved; shard grid for sharded |
| Core Group 1 | Cores with `num_tiles_per_core_group_1` tiles each |
| Core Group 2 | Cores with `num_tiles_per_core_group_2` tiles each (remainder handling) |
| Row Major | Yes (for interleaved); matches shard orientation for sharded |
| Load Balancing | Group 1 gets `ceil(num_tiles / num_cores)` tiles, Group 2 gets one fewer |
| Remainder Handling | Excess cores beyond `num_cores` get zero-tile arguments (noop) |

## Arguments

### Compile-Time Arguments

**Reader Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | block_or_width_sharded | uint32_t | 1 if input is block or width sharded, 0 otherwise |
| 1+ | TensorAccessorArgs (src0) | varies | Tensor accessor params for input A (omitted if IN0_SHARDED) |
| N+ | TensorAccessorArgs (src1) | varies | Tensor accessor params for input B (omitted if IN1_SHARDED) |

**Writer Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs (dst) | varies | Tensor accessor params for output buffer |

**Compute Kernel:**

No explicit compile-time args. Configuration is entirely through `#define` macros (see below).

### Runtime Arguments

**Reader Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src0_addr | uint32_t | Base address of input A buffer |
| 1 | src1_addr | uint32_t | Base address of input B buffer |
| 2 | num_tiles | uint32_t | Total tiles this core processes |
| 3 | start_id | uint32_t | Starting tile index for this core |
| 4 | block_height | uint32_t | Shard block height in tiles (0 if not sharded) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 if not sharded) |
| 6 | num_cores_y | uint32_t | Number of shards per width dimension |

**Writer Kernel (interleaved output):**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Base address of output buffer |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile index |

**Writer Kernel (block/width sharded to interleaved):**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Base address of output buffer |
| 1 | block_height_tiles | uint32_t | Block height in tiles |
| 2 | block_width_tiles | uint32_t | Block width in tiles |
| 3 | unpadded_block_height_tiles | uint32_t | Actual (unpadded) block height |
| 4 | unpadded_block_width_tiles | uint32_t | Actual (unpadded) block width |
| 5 | output_width_tiles | uint32_t | Full output width in tiles |
| 6 | block_num_tiles | uint32_t | Total tiles in one block |
| 7 | start_id_offset | uint32_t | Tile offset for this core's shard |
| 8 | start_id_base | uint32_t | Always 0 |

**Compute Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_size | uint32_t | Number of tiles per block |

## Kernel Implementations

| Kernel | File | Type | Assigned Cores |
|---|---|---|---|
| Reader | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` | ReaderDataMovement | all_device_cores |
| Compute | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` | Compute | all_device_cores |
| Writer (interleaved) | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | WriterDataMovement | all_device_cores |
| Writer (block/width to interleaved) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` | WriterDataMovement | all_device_cores |

### Reader Kernel

| Property | Value |
|---|---|
| File | `reader_binary_interleaved_start_id.cpp` |
| Assigned Cores | all_device_cores |

**Key Logic:**
- For sharded inputs (`IN0_SHARDED` / `IN1_SHARDED` defined): immediately calls `cb_reserve_back(cb, num_tiles)` then `cb_push_back(cb, num_tiles)` to expose the pre-loaded shard to the compute kernel. No NoC reads occur.
- For interleaved inputs: creates a `TensorAccessor` from compile-time args and the runtime base address. Iterates tile-by-tile in a simple loop from `start_id` to `start_id + num_tiles`.
- For block/width sharded (non-sharded input): uses a 2D iteration pattern -- outer loop over `block_height` rows, inner loop over `block_width` columns, with row stride of `num_cores_y * block_width`.
- Each tile read: `cb_reserve_back(cb, 1)` -> `noc_async_read_tile(tile_id, accessor, l1_addr)` -> `noc_async_read_barrier()` -> `cb_push_back(cb, 1)`.
- **Synchronization**: Pushes to CB c_0 and CB c_1. Compute waits on these via `cb_wait_front`.

### Compute Kernel

| Property | Value |
|---|---|
| File | `eltwise_binary_sfpu_kernel.cpp` |
| Assigned Cores | all_device_cores |

**Key Logic:**
- Outer loop: `per_core_block_cnt` blocks.
- For plain ADD (FP32): the `BINOP_INIT` macro expands to `add_binary_tile_init()` and `BINARY_SFPU_OP` expands to `add_binary_tile(i*2, i*2+1, i*2)`. For INT32/UINT32/UINT16: `ADD_INT_INIT` expands to `add_int_tile_init()` and `BINARY_SFPU_OP` expands to `add_int_tile<DataFormat::...>(i*2, i*2+1, i*2)`.
- Input tile placement in DST: Input A tiles go to even DST indices (i*2), Input B tiles go to odd DST indices (i*2+1). This is done via `copy_tile` with `copy_tile_to_dst_init_short_with_dt` to handle potential data format differences between the two input CBs.
- The SFPU operation reads from DST[i*2] and DST[i*2+1], writes result to DST[i*2].
- Result is packed from DST[i*2] to CB c_2 via `pack_tile(i*2, cb_out0)`.
- Pre-scaling (conditional, not used for plain ADD): tiles are copied from input CB to DST, SFPU unary op applied, then packed to interim CB (c_3 or c_4).
- **Synchronization**: Waits on CB c_0/c_3 and CB c_1/c_4 via `cb_wait_front`. Pops from input CBs and pushes to CB c_2 via `cb_pop_front` / `cb_push_back`. Uses `tile_regs_acquire` / `tile_regs_commit` / `tile_regs_wait` / `tile_regs_release` for DST register synchronization between unpack and pack stages.

### Writer Kernel (Interleaved)

| Property | Value |
|---|---|
| File | `writer_unary_interleaved_start_id.cpp` |
| Assigned Cores | all_device_cores |

**Key Logic:**
- For sharded output (`OUT_SHARDED` defined): simply calls `cb_wait_front(cb_out, num_pages)` and returns. The output CB is backed by the output shard buffer, so no write is needed.
- For interleaved output: iterates tile-by-tile from `start_id` to `start_id + num_pages`. Each iteration: `cb_wait_front(cb_out, 1)` -> `noc_async_write_page(tile_id, accessor, l1_addr)` -> `noc_async_writes_flushed()` -> `cb_pop_front(cb_out, 1)`.
- Final `noc_async_write_barrier()` ensures all writes complete.
- **Synchronization**: Waits on CB c_2 via `cb_wait_front`, pops via `cb_pop_front`.

### Writer Kernel (Block/Width Sharded to Interleaved)

| Property | Value |
|---|---|
| File | `writer_unary_sharded_blocks_interleaved_start_id.cpp` |
| Assigned Cores | all_device_cores |

**Key Logic:**
- Waits for the full block of tiles in CB c_2: `cb_wait_front(cb_out, block_num_tiles)`.
- Iterates in 2D: rows up to `unpadded_block_height_tiles`, columns up to `unpadded_block_width_tiles`. Skips padding tiles at the end of each row (advances L1 read pointer by `padded_width_diff`).
- Row stride in output: `output_width_tiles` tiles per row.
- Single `noc_async_write_barrier()` at the end, then `cb_pop_front` for the full block.
- **Synchronization**: Waits on CB c_2 via `cb_wait_front`, pops full block via `cb_pop_front`.

## Implementation Notes

- **Program factory variants**: The binary operation has multiple factories: `ElementWiseMultiCore` (FPU), `ElementWiseMultiCoreSfpu` (SFPU, analyzed here), `BroadcastHeightMultiCore`, `BroadcastWidthMultiCore`, `BroadcastHeightAndWidthMultiCore`. Factory selection is based on tensor shapes and data types as described in the Path Selection section.
- **Type-based operation variants**: ADD supports four SFPU paths based on data type: FP32 (`add_binary_tile`), INT32 (`add_int_tile<DataFormat::Int32>`), UINT32 (`add_int_tile<DataFormat::UInt32>`), UINT16 (`add_int_tile<DataFormat::UInt16>`). Each uses a different init and op macro.
- **UnpackToDestFP32 mode**: For ADD (non-POWER ops), `UnpackToDestMode::UnpackToDestFp32` is set on all input CBs (c_0, c_1, c_3, c_4) unconditionally, regardless of actual data type. This ensures maximum precision during SFPU computation.
- **Broadcast type selection**: This factory handles only the same-shape (no broadcast) case. Broadcasting is handled by separate program factories.
- **Sharding support and constraints**: Supports height sharded, width sharded, and block sharded inputs/outputs. Mixed sharding (some sharded, some interleaved) is supported. When block/width sharded input produces interleaved output, a specialized writer kernel handles the 2D write pattern.
- **FP32 dest accumulation**: Enabled when output data type is Float32, Int32, or UInt32 (via `fp32_dest_acc_en` flag in ComputeConfig).

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel functions that the compute kernel dispatches to. The ADD operation has two distinct SFPU paths: **floating-point** (`add_binary_tile` using SFPI abstractions) and **integer** (`add_int_tile` using raw `TT_`/`TTI_` instructions).

### SFPU Abstraction Layers

**Path 1: Floating-point ADD (`add_binary_tile`)**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_binary.h` (shared `_calculate_sfpu_binary_`) and `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` (arch-specific `calculate_sfpu_binary` wrapper) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

**Path 2: Integer ADD (`add_int_tile`)**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/add_int_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_add_int.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_add_int.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` (same as FP path) |

### Call Chain

**Floating-point path:**
1. The compute kernel calls `add_binary_tile(i*2, i*2+1, i*2)` (defined in `eltwise_binary_sfpu.h`), which wraps `MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>(idst0, idst1, odst)))`.
2. `llk_math_eltwise_binary_sfpu_binop` (in `llk_math_eltwise_binary_sfpu_binop.h`) delegates to `_llk_math_eltwise_binary_sfpu_params_` passing `calculate_sfpu_binary<APPROX, BinaryOp::ADD, 8, is_fp32_dest_acc_en>` as the callable.
3. `_llk_math_eltwise_binary_sfpu_params_` (in `llk_math_eltwise_binary_sfpu_params.h`) handles SFPU start/stall, iterates over tile faces based on `VectorMode::RC`, and calls the callable once per face.
4. `calculate_sfpu_binary` (arch-specific wrapper in `ckernel_sfpu_binary.h`) directly calls `_calculate_sfpu_binary_<APPROX, BinaryOp::ADD, 8>`, the core SFPU function in the tt_llk submodule.

**Integer path:**
1. The compute kernel calls `add_int_tile<DataFormat::Int32>(i*2, i*2+1, i*2)` (defined in `add_int_sfpu.h`), which wraps `MATH((llk_math_eltwise_binary_sfpu_add_int<APPROX, 8, DataFormat::Int32, false>(idst0, idst1, odst)))`.
2. `llk_math_eltwise_binary_sfpu_add_int` (in `llk_math_eltwise_binary_sfpu_add_int.h`) selects `InstrModLoadStore::INT32` for Int32/UInt32 or `InstrModLoadStore::LO16` for UInt16, then delegates to the same `_llk_math_eltwise_binary_sfpu_params_` with `_add_int_<APPROX, 8, INSTRUCTION_MODE, false>` as the callable.
3. The params dispatch is identical to the FP path (same `_llk_math_eltwise_binary_sfpu_params_` function).
4. `_add_int_` (in `ckernel_sfpu_add_int.h`) is the core SFPU function that uses raw `TT_SFPLOAD`/`TTI_SFPIADD`/`TT_SFPSTORE` instructions.

**Init paths:**
- FP: `add_binary_tile_init()` -> `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::ADD>()` -> `llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROX>(sfpu_binary_init<APPROX, BinaryOp::ADD>)` -> `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` (configures SFPU config reg, ADDR_MOD_7, resets counters) then calls `sfpu_binary_init` -> `_sfpu_binary_init_<APPROX, BinaryOp::ADD>()` which is a no-op for ADD (only DIV/POW/XLOGY need init).
- INT: `add_int_tile_init()` -> `llk_math_eltwise_binary_sfpu_add_int_init<APPROX>()` -> `llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROX>()` -> same `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` with no additional init callback.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (the default, passed from the LLK dispatch layer). This processes all 4 faces of a 32x32 tile. Each face represents a 16x16 sub-block (256 elements), and the SFPU processes 8 rows (32 elements per SFPU vector) per invocation with 8 iterations.
- **Operation invocation**: In RC mode, the params dispatch loops `for (face = 0; face < 4; face++)`, calling the core SFPU function once per face. Each invocation of the core function runs its internal loop of `ITERATIONS=8`, processing 8 rows of 32 elements each (= 256 elements = one 16x16 face). Total: 4 faces x 8 iterations = 32 SFPU invocations per tile.
- **DEST address progression**: After each face, the params dispatch executes two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` instructions, each incrementing the DEST read/write counter by 8 rows. Combined, this advances by 16 rows (= one 16x16 face = `DEST_FACE_WIDTH`). Within each invocation, the core function uses `dst_reg++` (SFPI auto-increment) to advance one row after each iteration. The `_llk_math_eltwise_binary_sfpu_start_` function initializes the DEST write address to 0 at the start.

### Annotated SFPU Kernel Source

This operation has two distinct SFPU kernels. Both are presented below.

#### Kernel 1: Floating-Point Binary ADD (SFPI-based -- Style A)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{ // For ADD: APPROXIMATION_MODE=true/false (unused for ADD), BINOP=BinaryOp::ADD, ITERATIONS=8
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr std::uint32_t dst_tile_size_sfpi = 32;
        sfpi::vFloat in0                           = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // Load 32 elements from DST tile for input A
        sfpi::vFloat in1                           = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // Load 32 elements from DST tile for input B
        sfpi::vFloat result                        = 0.0f;

        if constexpr (BINOP == BinaryOp::ADD)
        {
            result = in0 + in1; // SFPU vector float addition; compiles to SFPADD instruction
        }
        // Other BINOP branches (SUB, MUL, DIV, RSUB, POW, XLOGY) omitted -- not active for ADD

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // Store result back to DST output tile
        sfpi::dst_reg++; // Advance DEST read/write pointer by 1 row (32 elements)
    }
}
```

Note: The Blackhole arch-specific wrapper `calculate_sfpu_binary` in `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` is a trivial pass-through that calls `_calculate_sfpu_binary_` directly with identical template parameters. No bf16 rounding or special-casing is applied for ADD (unlike MUL and DIV which have bf16 RNE rounding when `is_fp32_dest_acc_en=false`).

#### Kernel 2: Integer ADD (TTI-based -- Style B, Blackhole)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_add_int.h

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _add_int_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    // Operand A is input1 (int32/uint16/uint32)
    // Operand B is input2 (int32/uint16/uint32)
    // Output is int32/uint16/uint32
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // INSTRUCTION_MODE = InstrModLoadStore::INT32_2S_COMP enables LOAD/STORE operations to convert INT32 sign-magnitude to 2's complement.
    // However, in Blackhole, this mode has no effect and the data format remains unchanged.

    // If LOAD/STORE have the value in INT sign-magnitude format and SFPU needs it as 2's complement.
    constexpr auto INSTR_MOD_CAST = InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP;

    // size of each tile in Dest is 64 rows
    constexpr std::uint32_t dst_tile_size = 64;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // operand A
        TT_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, dst_index_in0 * dst_tile_size);
        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            apply_sign_magnitude_conversion(p_sfpu::LREG0, p_sfpu::LREG2, INSTR_MOD_CAST);
        }

        // operand B
        TT_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, dst_index_in1 * dst_tile_size);
        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            apply_sign_magnitude_conversion(p_sfpu::LREG1, p_sfpu::LREG2, INSTR_MOD_CAST);
        }

        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);

        // LREG_0 -> dest
        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            apply_sign_magnitude_conversion(p_sfpu::LREG0, p_sfpu::LREG1, INSTR_MOD_CAST);
        }
        TT_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, dst_index_out * dst_tile_size);
        sfpi::dst_reg++;
    }
}
```

For the standard ADD path called from the compute kernel, `SIGN_MAGNITUDE_FORMAT=false`, so all `apply_sign_magnitude_conversion` branches are compiled out. The effective instruction sequence per iteration is:

1. `SFPLOAD LREG0` from DEST at input A offset
2. `SFPLOAD LREG1` from DEST at input B offset
3. `SFPIADD 0, LREG1, LREG0, imod=4` -- LREG0 = LREG0 + LREG1 (imod=4 means CC_NONE, no condition code update)
4. `SFPSTORE LREG0` to DEST at output offset
5. `dst_reg++` (advance DEST pointer)

**Wormhole B0 variant**: The Wormhole `_add_int_` is structurally identical but uses `ADDR_MOD_3` instead of `ADDR_MOD_7`, and computes `sfpload_instr_mod` differently when `SIGN_MAGNITUDE_FORMAT=true` (using `INT32_2S_COMP` constant instead of Blackhole's `apply_sign_magnitude_conversion` approach). For the standard `SIGN_MAGNITUDE_FORMAT=false` case, the instruction sequence is the same: SFPLOAD, SFPLOAD, SFPIADD(imod=4), SFPSTORE, dst_reg++.

### SFPU Instructions Used

**Floating-point ADD path (`_calculate_sfpu_binary_` with `BinaryOp::ADD`):**

| Instruction / Intrinsic | Description |
|---|---|
| `sfpi::dst_reg[offset]` (read) | **SFPLOAD** -- Loads a 32-element vector from the DEST register file at the given row offset into an SFPU local register (LREG). Interpreted as float32. |
| `in0 + in1` (sfpi::vFloat addition) | **SFPADD** -- Performs element-wise IEEE 754 float32 addition on two 32-element vectors in SFPU local registers. |
| `sfpi::dst_reg[offset] = result` (write) | **SFPSTORE** -- Stores a 32-element vector from an SFPU local register back to the DEST register file at the given row offset. |
| `sfpi::dst_reg++` | **INCRWC** (increment read/write counter) -- Advances the DEST base address by one row (32 elements) for subsequent load/store operations. |

**Integer ADD path (`_add_int_`):**

| Instruction | Description |
|---|---|
| `TT_SFPLOAD(LREG, mode, addr_mod, offset)` | Loads a 32-element vector from DEST into the specified LREG. The `mode` parameter (`INT32` or `LO16`) controls integer data interpretation. Uses `ADDR_MOD_7` (Blackhole) or `ADDR_MOD_3` (Wormhole). |
| `TTI_SFPIADD(imm=0, LREG1, LREG0, imod=4)` | Integer add: `LREG0 = LREG0 + LREG1 + 0`. The `imod=4` sets `CC_NONE`, meaning no condition code update occurs. This is a 32-bit integer addition for all 32 lanes simultaneously. |
| `TT_SFPSTORE(LREG, mode, addr_mod, offset)` | Stores a 32-element vector from the specified LREG back to DEST. Same mode and addr_mod as SFPLOAD. |
| `sfpi::dst_reg++` | **INCRWC** -- same as float path, advances DEST pointer by one row. |

### SFPU Register Usage

**Floating-point ADD:**

| Register | Usage |
|---|---|
| LREG0 (implicit via sfpi::vFloat) | Holds `in0` loaded from `dst_reg[dst_index_in0 * 32]` |
| LREG1 (implicit via sfpi::vFloat) | Holds `in1` loaded from `dst_reg[dst_index_in1 * 32]` |
| LREG (implicit via sfpi::vFloat) | Holds `result` after addition, stored to `dst_reg[dst_index_out * 32]` |
| DEST register file | Source for both input tiles and destination for the output tile. Input A is at DST[i*2], input B at DST[i*2+1], output written to DST[i*2]. The SFPU accesses DEST via sfpi::dst_reg with a tile size of 32 rows (due to SFP_DESTREG_STRIDE=2, mapping 64 physical DEST rows to 32 SFPI-visible rows). |

**Integer ADD:**

| Register | Usage |
|---|---|
| LREG0 (`p_sfpu::LREG0`) | Loaded with operand A from DEST; also receives the result of SFPIADD (accumulator). Stored back to DEST as the output. |
| LREG1 (`p_sfpu::LREG1`) | Loaded with operand B from DEST; used as the addend in SFPIADD. |
| LREG2 (`p_sfpu::LREG2`) | Only used when `SIGN_MAGNITUDE_FORMAT=true` as a scratch register for `apply_sign_magnitude_conversion`. Not used in the standard ADD path. |
| DEST register file | Same layout as float path: input A at DST[i*2], input B at DST[i*2+1], output at DST[i*2]. Tile size is 64 physical rows for raw TT_ instructions (vs 32 for SFPI due to stride). |

### Address Mode Configuration

The address mode is configured during initialization by `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()`, which calls `eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>()`.

**ADDR_MOD_7** (used by both Blackhole and Wormhole for binary SFPU ops):

```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}.set(ADDR_MOD_7);
```

All fields are zero-increment. This means the address mode itself does not auto-increment any source or destination register addresses between SFPU instructions. Instead, address progression is handled explicitly:
- Within the core SFPU function: `sfpi::dst_reg++` (INCRWC) advances the DEST pointer by 1 row after each iteration.
- Between faces: `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` called twice advances the DEST counter by 16 rows total (one face width).

The `SfpuType::unused` template parameter means the `ADDR_MOD_6` variant (with `.dest = {.incr = 2}`) is NOT configured -- that variant is only for mul_int32, mul_uint16, and max/min operations.

**Wormhole vs Blackhole difference**: The `_llk_math_eltwise_binary_sfpu_start_` function on Wormhole additionally calls `math::set_addr_mod_base()`, and `_llk_math_eltwise_binary_sfpu_done_` calls `math::clear_addr_mod_base()` with an extra `TTI_STALLWAIT(STALL_CFG, WAIT_SFPU)`. On Blackhole, these addr_mod_base operations are not present. The ADDR_MOD_7 configuration itself is identical across both architectures.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary eltwise SFPU program factory work in ttnn? What kernels does it use? How does it differ from the FPU path?"
   **Reason**: Initial architectural understanding of the SFPU binary operation infrastructure.
   **Key Findings**: Confirmed the three-kernel structure (reader, SFPU compute, writer), that the SFPU path is selected via `is_binary_sfpu_op`, and that the key difference is the compute kernel using SFPU instructions rather than FPU matrix operations.

2. [SFPU] **Query**: "How does the add_binary_tile SFPU operation work? What is the call chain from add_binary_tile through LLK to the core SFPU ckernel implementation? What files contain the LLK dispatch and core SFPU implementation for binary add?"
   **Reason**: Needed to trace the full abstraction layer hierarchy from API to core SFPU function for floating-point binary addition.
   **Key Findings**: Confirmed the call chain: `add_binary_tile` -> `llk_math_eltwise_binary_sfpu_binop` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_sfpu_binary` -> `_calculate_sfpu_binary_`. Identified file paths for each layer across Blackhole and Wormhole architectures.

3. [SFPU] **Query**: "How does add_binary_tile work in the LLK layer? What is the call chain from add_binary_tile through llk_math_eltwise_binary_sfpu down to the core ckernel_sfpu_add implementation? Show the file paths for each layer. Also explain how add_int_tile works for integer addition." (asked to tenstorrent/tt-llk)
   **Reason**: Needed tt_llk-specific details on the binary SFPU dispatch and integer addition implementation.
   **Key Findings**: Confirmed the params dispatch pattern with VectorMode::RC processing all 4 faces, the SFPIADD instruction for integer add, and the `_add_int_` function location in `ckernel_sfpu_add_int.h`. Clarified the distinction between float path (SFPI abstractions) and integer path (raw TT_ instructions).

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp`
   **Reason**: Understanding path selection logic between FPU and SFPU.
   **Key Information**: `is_binary_sfpu_op` function returns true for ADD when both dtypes match and are FLOAT32/INT32/UINT32/UINT16; `select_program_factory` dispatches to `ElementWiseMultiCoreSfpu` when this check passes and shapes are equal.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp`
   **Reason**: Understanding the preprocessor defines generated for the SFPU compute kernel.
   **Key Information**: For ADD with FP32, defines `BINOP_INIT` = `add_binary_tile_init()` and `BINARY_SFPU_OP` = `add_binary_tile(i*2, i*2+1, i*2)`. For INT32, uses `ADD_INT_INIT` and `add_int_tile<DataFormat::Int32>`. The `BINARY_SFPU_OP` define is always inserted at line 535.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp`
   **Reason**: Understanding runtime argument assignment and core distribution strategy.
   **Key Information**: `set_eltwise_binary_runtime_args` is a shared template function used by both FPU and SFPU factories. It handles work splitting via `split_work_to_cores`, supports zero-start-grid optimization, and manages sharded/interleaved/mixed configurations.

### Confluence References
No Confluence references were needed for this analysis. The SFPU instructions used (SFPLOAD, SFPADD, SFPIADD, SFPSTORE) are well-documented in DeepWiki and the source code comments.

### Glean References
No Glean references were needed for this analysis.
