# ADD (SFPU) Implementation Analysis

## Overview

The ADD (SFPU) operation performs element-wise addition of two input tensors using the SFPU (Special Function Processing Unit) vector engine on Tensix cores. Unlike FPU-based binary add which uses the matrix engine, the SFPU variant operates on tiles loaded into DST registers and executes addition via SFPU instructions. This path is selected when `utils::is_binary_sfpu_op` returns true for the given operation type and tensor configuration.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

The SFPU add path supports multiple data type variants:
- **Float types** (BFloat16, Float32): Uses `add_binary_tile` via the `BINOP_INIT` / `BINARY_SFPU_OP` define path
- **INT32**: Uses `add_int_tile<DataFormat::Int32>` via the `ADD_INT_INIT` define path
- **UINT32**: Uses `add_int_tile<DataFormat::UInt32>` via the `ADD_INT_INIT` define path
- **UINT16**: Uses `add_int_tile<DataFormat::UInt16>` via the `ADD_INT_INIT` define path

## Work Unit Definition

One work unit is a **block of tiles**. The block size is determined by `find_max_block_size(num_tiles_per_shard)` for sharded inputs, or defaults to 1 for interleaved inputs. The compute kernel processes `per_core_block_cnt` blocks, each containing `per_core_block_size` tiles. Each tile is a 32x32 element matrix in the hardware's native tile format.

## Tensor Format and Layout

### Input Tensor A (src0)

| Property | Value |
|---|---|
| Dimension Convention | NCHW (arbitrary shape, flattened to total tile count) |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Interleaved or Sharded (height, width, or block) |
| Buffer Type | DRAM or L1 |
| Data Type | BFloat16, Float32, Int32, UInt32, or UInt16 |

### Input Tensor B (src1)

| Property | Value |
|---|---|
| Dimension Convention | NCHW (must match tensor A shape) |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Interleaved or Sharded (height, width, or block) |
| Buffer Type | DRAM or L1 |
| Data Type | BFloat16, Float32, Int32, UInt32, or UInt16 (must match A) |

### Output Tensor

| Property | Value |
|---|---|
| Dimension Convention | Same as input |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Interleaved or Sharded |
| Buffer Type | DRAM or L1 |
| Data Type | Same as input or specified output dtype |

### Layout Transformations

No tilize/untilize operations are performed within this program factory. Both inputs and output must already be in TILE layout. The `unpack_to_dest_mode` is set to `UnpackToDestFp32` for all non-POWER operations, meaning tiles are unpacked to FP32 precision in DST registers before the SFPU operation executes.

## Data Flow Pattern

1. **Reader kernel** reads tiles from DRAM/L1 for both input tensors (or marks sharded CBs as ready)
2. Tiles land in CB c_0 (input A) and CB c_1 (input B)
3. **Compute kernel** waits for `per_core_block_size` tiles in both input CBs
4. For each tile in the block:
   - `copy_tile(cb_inp0, i, i*2)`: Copies input A tile `i` from CB to DST register slot `i*2`
   - `copy_tile(cb_inp1, i, i*2+1)`: Copies input B tile `i` from CB to DST register slot `i*2+1`
   - Executes `BINARY_SFPU_OP` which expands to `add_binary_tile(i*2, i*2+1, i*2)`: Adds DST[i*2+1] to DST[i*2], result in DST[i*2]
   - `pack_tile(i*2, cb_out0)`: Packs result from DST[i*2] to output CB
5. Compute kernel pops input CBs and pushes output CB
6. **Writer kernel** reads from output CB and writes tiles to DRAM/L1 (or does nothing for sharded output)

The key design choice is that both operands are placed into adjacent DST register slots (even/odd indexing), which allows the SFPU to operate on them in-place. The result overwrites the first operand's slot.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|
| c_0 | cb_src0 | Input tensor A | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | 1 (reader pushes one at a time) or `num_tiles_per_shard` (sharded) | Double-buffered (interleaved) or Single (sharded) | Reader | Compute |
| c_1 | cb_src1 | Input tensor B | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | 1 (reader pushes one at a time) or `num_tiles_per_shard` (sharded) | Double-buffered (interleaved) or Single (sharded) | Reader | Compute |
| c_2 | cb_out0 | Output | Sharded/block-width: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | `per_core_block_size` (compute pushes block) | Double-buffered (interleaved) or Single (sharded) | Compute | Writer |
| c_3 | cb_interm0 | Interim for pre-scaled input A | `max_block_size` | `per_core_block_size` | Single-buffered | Compute (pre-scale phase) | Compute (main phase) |
| c_4 | cb_interm1 | Interim for pre-scaled input B | `max_block_size` | `per_core_block_size` | Single-buffered | Compute (pre-scale phase) | Compute (main phase) |

**Note on c_3 and c_4**: These interim CBs are only created when the operation requires pre-scaling of inputs (indicated by the `SFPU_OP_INIT_PRE_IN0_0` and `SFPU_OP_INIT_PRE_IN1_0` defines). For plain ADD, neither pre-scaling define is set, so **c_3 and c_4 are not allocated**. They are used by composite operations like LOGADDEXP (which applies EXP before ADD) or HYPOT (which applies SQUARE before ADD).

## Pipeline Pattern Summary

For the standard interleaved case with ADD:
- **c_0, c_1**: Capacity = `2 * max_block_size`, reader pushes 1 tile at a time. This provides double-buffering -- the reader can fill the next tile slot while compute processes the current block.
- **c_2**: Capacity = `2 * max_block_size`, compute pushes `block_size` tiles at a time. Double-buffered -- writer can drain one block while compute fills the next.

For sharded inputs, CBs are backed by the tensor's L1 buffer directly (`set_globally_allocated_address`), so all shard tiles are available immediately with no DMA transfers needed.

## Index Calculations

### Interleaved Path
Tile indexing uses a simple linear scheme. Each core is assigned a contiguous range of tiles starting at `start_id = num_tiles_read` (cumulative offset). The reader iterates `tile_id` from `start_id` to `start_id + num_tiles`, using `TensorAccessor` to map tile IDs to physical DRAM bank addresses.

### Block/Width Sharded Path
For block or width sharded tensors, tile indexing uses a 2D scheme:
```
start_id = (core_index / num_shards_per_width) * (block_height * block_width * num_shards_per_width)
         + (core_index % num_shards_per_width) * block_width
```
The reader then iterates in row-major order within the shard: for each of `block_height` rows, it reads `block_width` tiles, advancing by `num_cores_y * block_width` between rows to account for the sharding stride.

### DST Register Indexing
Within the compute kernel, tile `i` of input A maps to DST slot `i*2` (even indices), and tile `i` of input B maps to DST slot `i*2+1` (odd indices). The SFPU operation reads both and writes the result to the even slot: `add_binary_tile(i*2, i*2+1, i*2)`.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads via `noc_async_read_tile`. One tile at a time with a barrier after each tile pair. This is a simple sequential access pattern following the linear tile ID space.
- **Sharded**: No reads needed -- the CB is backed by the L1 shard buffer. The reader simply does `cb_reserve_back` + `cb_push_back` to signal data availability.
- **Block/Width Sharded (non-sharded input)**: Strided reads -- within each shard row, tiles are read sequentially, but between rows there is a stride of `num_cores_y * block_width` tiles.

### Write Pattern
- **Interleaved**: Sequential tile writes via `noc_async_write_page`, one tile at a time, with flushes between tiles and a final barrier.
- **Sharded**: No writes needed -- the CB is backed by the output L1 buffer. The writer simply does `cb_wait_front` to acknowledge completion.
- **Block/Width Sharded to Interleaved**: Uses a specialized writer (`writer_unary_sharded_blocks_interleaved_start_id.cpp`) that handles the 2D block-to-interleaved mapping with padding awareness.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Determined by `operation_attributes.worker_grid` (typically rectangular) |
| Work Splitting | `split_work_to_cores` for interleaved; shard grid for sharded |
| Load Balancing | Two core groups: group 1 gets `ceil(num_tiles / num_cores)` tiles, group 2 gets `floor(num_tiles / num_cores)` tiles |
| Remainder Handling | Excess tiles distributed to group 1 cores; unused cores get zero-length work |
| Iteration Order | Row-major (default for interleaved); shard orientation-dependent for sharded |

For the zero-start-grid optimization (single rectangular grid starting at (0,0) with shards also starting at (0,0)), a faster grid-to-cores mapping is used. Otherwise, the generic `CoreRangeSet`-based distribution is applied.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | block_or_width_sharded | uint32_t | 1 if tensor is block or width sharded, 0 otherwise |
| 1+ | TensorAccessorArgs(src0) | multiple | Tensor accessor params for input A (only if not IN0_SHARDED) |
| varies | TensorAccessorArgs(src1) | multiple | Tensor accessor params for input B (only if not IN1_SHARDED) |

#### Writer Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs(dst) | multiple | Tensor accessor params for output buffer |

#### Compute Kernel

Compile-time arguments are passed via `ComputeConfig` and `eltwise_defines`:

| Define | Value (for float ADD) | Description |
|---|---|---|
| BINOP_INIT | `add_binary_tile_init();` | SFPU init function call |
| BINARY_SFPU_OP | `add_binary_tile(i*2, i*2+1, i*2);` | Per-tile SFPU operation |
| fp32_dest_acc_en | true if output is Float32/Int32/UInt32 | Enables FP32 accumulation in DST |
| unpack_to_dest_mode | UnpackToDestFp32 for all CBs (non-POWER ops) | Tiles unpacked to FP32 in DST |

For integer ADD, `ADD_INT_INIT` replaces `BINOP_INIT`, and the op name becomes `add_int_tile<DataFormat::X>`.

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src0_addr | uint32_t | Base address of input tensor A buffer |
| 1 | src1_addr | uint32_t | Base address of input tensor B buffer |
| 2 | num_tiles | uint32_t | Total tiles to process on this core |
| 3 | start_id | uint32_t | Starting tile index for this core |
| 4 | block_height | uint32_t | Shard height in tiles (0 for interleaved) |
| 5 | block_width | uint32_t | Shard width in tiles (0 for interleaved) |
| 6 | num_cores_y | uint32_t | Number of shards per width dimension (0 for interleaved) |

#### Compute Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_size | uint32_t | Number of tiles per block |

#### Writer Kernel (Interleaved output)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Base address of output buffer |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile index for output |

#### Writer Kernel (Block/Width Sharded to Interleaved)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Base address of output buffer |
| 1 | block_height | uint32_t | Shard block height in tiles |
| 2 | block_width | uint32_t | Shard block width in tiles |
| 3 | unpadded_block_height | uint32_t | Actual height without padding |
| 4 | unpadded_block_width | uint32_t | Actual width without padding |
| 5 | output_width | uint32_t | Full output width in tiles |
| 6 | block_size | uint32_t | Total tiles in shard block |
| 7 | start_id | uint32_t | Starting tile index for output |
| 8 | padding_flag | uint32_t | Always 0 |

## Kernel Implementations

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
- **Type**: ReaderDataMovementConfig (runs on BRISC, uses NoC0)
- **Key Logic**:
  - For sharded inputs: Simply calls `cb_reserve_back` / `cb_push_back` to signal that data is already present in the CB (backed by L1 shard buffer).
  - For interleaved inputs: Uses `TensorAccessor` + `noc_async_read_tile` to fetch tiles one at a time from DRAM into the CB. Issues a `noc_async_read_barrier` after each tile pair.
  - For block/width sharded mode: Uses a 2D loop (block_height x block_width) with strided row access (`row_start_tile_id += num_cores_y * block_width`) to handle the sharding layout.
- **Synchronization**: Produces into c_0 and c_1.

### Compute Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`
- **Type**: ComputeConfig (runs on TRISC cores: unpack, math, pack)
- **Key Logic**:
  - Outer loop iterates `per_core_block_cnt` blocks.
  - For each block: waits for data in cb_inp0 and cb_inp1, reserves output space in cb_out0.
  - Uses `tile_regs_acquire` / `tile_regs_wait` for DST register lifecycle.
  - Copies input A tiles to even DST slots (i*2), input B tiles to odd DST slots (i*2+1).
  - Executes the SFPU operation (for ADD: `add_binary_tile(i*2, i*2+1, i*2)`).
  - Packs result from DST[i*2] to output CB.
  - The `copy_tile_to_dst_init_short_with_dt` calls handle data format switching between the two inputs.
- **Synchronization**: Consumes from c_0/c_1 (or c_3/c_4 if pre-scaling), produces into c_2.

### Writer Kernel

- **File** (interleaved): `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **File** (block/width sharded to interleaved): `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp`
- **Type**: WriterDataMovementConfig (runs on NCRISC, uses NoC1)
- **Key Logic**:
  - For sharded output: Simply calls `cb_wait_front` to acknowledge compute completion. Data is already in the correct L1 location.
  - For interleaved output: Uses `TensorAccessor` + `noc_async_write_page` to write tiles one at a time, with `noc_async_writes_flushed` between tiles and a final `noc_async_write_barrier`.
- **Synchronization**: Consumes from c_2.

## Implementation Notes

1. **Fused Activation Support**: The ADD operation can fuse a RELU activation. When fused activations consist of a single RELU, the `PACK_RELU` define is set, which configures the packer hardware to apply ReLU during the pack step (zero cost). For other fused activations, post-processing SFPU chains are generated.

2. **Program Caching**: The `override_runtime_arguments` method enables efficient program reuse. When tensor shapes remain the same but addresses change (e.g., different input data), only runtime args are updated without recompiling kernels.

3. **Zero-Start Grid Optimization**: When the worker grid is a single rectangle starting at (0,0) and any shard grids also start at (0,0), faster grid-to-core mapping functions are used, avoiding the overhead of generic `CoreRangeSet` iteration.

4. **UnpackToDestFp32 Mode**: For all non-POWER binary SFPU operations, tiles are unpacked directly to FP32 format in DST registers regardless of input data format. This ensures maximum precision during the SFPU computation. The POWER operation is special-cased to only use FP32 unpack when inputs are already Float32.

5. **No Pre-scaling for Plain ADD**: The interim CBs (c_3, c_4) are conditionally created based on the presence of `SFPU_OP_INIT_PRE_IN0_0` / `SFPU_OP_INIT_PRE_IN1_0` defines. For plain ADD, these defines are not set, so cb_inp0 aliases to cb_in0 and cb_inp1 aliases to cb_in1, avoiding unnecessary data movement.

6. **Reader Issues Barrier Per Tile**: In the interleaved path, the reader issues `noc_async_read_barrier` after every tile (or tile pair for block/width sharded). This serializes reads but ensures data is available before pushing to the CB. The double-buffered CB capacity (2 * max_block_size) provides overlap potential between reader and compute.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the floating-point ADD path.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/common/inc/sfpu/ckernel_sfpu_binary.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `add_binary_tile(idst0, idst1, odst)` (defined in `eltwise_binary_sfpu.h`), which wraps the call in the `MATH()` macro to ensure it only runs on the Math RISC-V core.
2. Inside, it calls `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>(idst0, idst1, odst)` (in `llk_math_eltwise_binary_sfpu_binop.h`), which forwards to `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>()` with `calculate_sfpu_binary<APPROX, BinaryOp::ADD, 8, is_fp32_dest_acc_en>` as the callable.
3. `_llk_math_eltwise_binary_sfpu_params_` (in `llk_math_eltwise_binary_sfpu_params.h`) sets the DST write address, stalls until the SFPU is ready, then iterates over 4 tile faces (in RC vector mode), calling the SFPU function once per face and advancing the DST read/write counter by 16 rows between faces via `TTI_SETRWC`.
4. Each face invocation calls `calculate_sfpu_binary(...)` (in the metal-layer `ckernel_sfpu_binary.h`), which delegates to `_calculate_sfpu_binary_<APPROX, BinaryOp::ADD, 8>(...)` in the tt_llk submodule's `ckernel_sfpu_binary.h`.
5. `_calculate_sfpu_binary_` loops 8 iterations per face (processing 8 rows of the 16x16 face at a time, since each SFPI row contains 2 datums worth of a 64-element vector), loading `in0` and `in1` from destination registers, computing `result = in0 + in1`, writing back, and advancing `dst_reg++`.

For initialization, `add_binary_tile_init()` calls `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::ADD>()`, which calls `llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROX>(sfpu_binary_init<APPROX, BinaryOp::ADD>)`. This initializes the SFPU config register, configures address modes, resets counters, and then calls `_sfpu_binary_init_<APPROX, BinaryOp::ADD>()`. For ADD, `_sfpu_binary_init_` is a no-op (no special initialization is needed since ADD does not use reciprocal or log).

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h
// (Wormhole variant is identical)

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) // APPROXIMATION_MODE=true, BINOP=BinaryOp::ADD, ITERATIONS=8
{
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr std::uint32_t dst_tile_size_sfpi = 32;
        sfpi::vFloat in0                           = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD from DEST at tile offset for input A
        sfpi::vFloat in1                           = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD from DEST at tile offset for input B
        sfpi::vFloat result                        = 0.0f;

        if constexpr (BINOP == BinaryOp::ADD)
        {
            result = in0 + in1; // SFPADD (alias of SFPMAD with one operand = 1.0): computes FP32 addition
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // SFPSTORE result back to DEST at output tile offset
        sfpi::dst_reg++; // Advance DEST row pointer by SFP_DESTREG_STRIDE (2) for next vector
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void _sfpu_binary_init_() // APPROXIMATION_MODE=true, BINOP=BinaryOp::ADD
{
    // For ADD, no special initialization is needed (no reciprocal, no log).
    // The if-constexpr branches for DIV/POW (reciprocal init) and XLOGY (log init) are not taken.
}
```

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| **SFPLOAD** (opcode 0x70) | Loads a value from the Destination register file into an SFPU local register (LREG). Used twice per iteration to load `in0` and `in1` from their respective DST tile offsets. The instruction supports multiple input formats (FP16_A, FP16_B, FP32, etc.) and converts to FP32 in the LREG. |
| **SFPADD / SFPMAD** (opcode 0x85/0x84) | Performs the floating-point addition. SFPADD is an alias of SFPMAD (Fused Multiply-Add) where one multiplicand is 1.0, so `A * 1.0 + B = A + B`. The SFPI compiler emits this when it sees `in0 + in1`. Operates on FP32 inputs, produces FP32 output. IPC=1, latency=2 cycles. Flushes subnormals and sets exception flags. |
| **SFPSTORE** (opcode 0x72) | Stores a value from an SFPU local register back to the Destination register file. Used once per iteration to write the addition result. Supports format conversion on store (e.g., FP32 to FP16_B). IPC=1, latency=2 (SrcS) or 3 (Dest) cycles. |
| **SFPLOADI** | Loads an immediate value (0.0f) into an LREG for the `result` initialization. The compiler may optimize this away since `result` is unconditionally overwritten by the ADD branch. |
| **TTI_SETRWC** | Not an SFPU instruction per se, but a Tensix instruction issued by the params dispatch layer between face iterations. Advances the DEST read/write counter by 8 rows (called twice per face = 16 rows) to move to the next 16x16 face within the 32x32 tile. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG[0-5]** (local registers) | The SFPI compiler allocates up to 6 local registers (L0-L5) for intermediate values. For the ADD kernel, typically 3 LREGs are used: one for `in0`, one for `in1`, and one for `result`. All values are in FP32 format within LREGs. |
| **LREG[7]** (special) | Can be used for indirect addressing via `RG[7][3:0]`. Not explicitly used by the ADD kernel, but the SFPLOAD/SFPSTORE instruction encoding can optionally reference it for indirect register selection. |
| **DEST registers** | The Destination register file is the primary data store. Each tile occupies 32 SFPI-addressable rows (64 physical rows / SFP_DESTREG_STRIDE of 2). Input A is at `dst_index_in0 * 32`, input B at `dst_index_in1 * 32`, and the output is written to `dst_index_out * 32`. The `dst_reg++` call advances the row pointer by SFP_DESTREG_STRIDE (2) after each iteration, processing all 8 iterations x 2-stride = 16 rows per face call. |
| **DEST write address** | Set by `math::set_dst_write_addr<Tile32x32, SrcRegs>(0)` at the start of the params dispatch. This configures the base address for SFPSTORE operations. Cleared by `math::clear_dst_reg_addr()` when done. |

### Address Mode Configuration

The address mode is configured by `eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>()` during initialization. For the ADD operation (which uses `SfpuType::unused` since it is a generic binary op), only `ADDR_MOD_7` is configured:

```
ADDR_MOD_7:
  .srca = {.incr = 0}   // No auto-increment on SrcA register addressing
  .srcb = {.incr = 0}   // No auto-increment on SrcB register addressing
  .dest = {.incr = 0}   // No auto-increment on Dest register addressing
```

This zero-increment configuration is correct because the SFPU kernel manages its own DEST addressing explicitly through `dst_reg++` (which advances by `SFP_DESTREG_STRIDE`) and the `TTI_SETRWC` instructions between faces. The SFPU does not rely on hardware auto-increment for this operation.

The `ADDR_MOD_6` variant (with `.dest = {.incr = 2}`) is only configured for specific operations like `mul_int32`, `max`, `min`, etc. -- not for ADD.

**Wormhole vs Blackhole differences**: The address mode configuration is identical between Wormhole and Blackhole for this operation. The minor difference is that Wormhole's `_llk_math_eltwise_binary_sfpu_start_` also calls `math::set_addr_mod_base()` and the `_done_` function calls `math::clear_addr_mod_base()` with an additional `TTI_STALLWAIT(STALL_CFG, WAIT_SFPU)`. Blackhole omits these calls, suggesting the addr_mod_base mechanism is not needed on the newer architecture.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary SFPU element-wise program factory work? What is the structure of element_wise_multi_core_sfpu_pgm_factory.cpp and how does it set up kernels for binary operations like add?"
   **Reason**: Initial architectural understanding of the program factory structure and how ADD maps to SFPU operations.
   **Key Findings**: Confirmed the three-kernel structure (reader/compute/writer), the role of `get_defines_fp32` in mapping BinaryOpType::ADD to `add_binary_tile` / `add_int_tile`, and the SFPU execution model where tiles are copied to DST registers before SFPU operations execute. Also confirmed that `ElementWiseMultiCoreSfpu` is selected via `utils::is_binary_sfpu_op`.

2. **Query**: "How does add_binary_tile work in the compute kernel API? What is the call chain from add_binary_tile through LLK to the SFPU ckernel implementation?"
   **Reason**: Tracing the full call chain from the compute API to the core SFPU implementation.
   **Key Findings**: Confirmed the call chain: `add_binary_tile` -> `llk_math_eltwise_binary_sfpu_binop` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_sfpu_binary` -> `_calculate_sfpu_binary_`. Identified file locations for each abstraction layer.

3. **Query**: "How does add_binary_tile work? Trace the call chain from add_binary_tile through llk_math_eltwise_binary_sfpu to the ckernel SFPU implementation." (tt-llk repo)
   **Reason**: Understanding the tt_llk submodule's role in the SFPU binary operation implementation.
   **Key Findings**: Confirmed that `_calculate_sfpu_binary_` for ADD performs `result = in0 + in1` using SFPI `vFloat` addition. The function iterates 8 times per face call (ITERATIONS=8), and the params layer iterates over 4 faces in RC mode. Also confirmed the init function is a no-op for ADD.

4. **Query**: "How does SFPI handle binary operations like addition on destination registers?" (sfpi repo)
   **Reason**: Understanding the SFPI programming model for the `+` operator on `vFloat`.
   **Key Findings**: The `vFloat::operator+` maps to `__builtin_rvtt_sfpadd`, which emits the SFPADD instruction (alias of SFPMAD with one operand = 1.0). SFPU local registers (LREGs) hold FP32 intermediate values. The `dst_reg` global provides indexed access to the 64-element vectors in the Destination register file.

### Confluence References

1. **Page**: Tensix SFPU Instruction Set Architecture (page ID 1170505767)
   **Section**: SFPADD
   **Key Information**: SFPADD (opcode 0x85) is an alias of SFPMAD, encoding O4 format, operating on FP32 inputs/outputs, with IPC=1 and latency=2 cycles. It sets exception flags and flushes subnormals. One of `RG[VA]` or `RG[VB]` should be 1.0 to achieve addition semantics.

2. **Page**: Same page, SFPLOAD section
   **Key Information**: SFPLOAD (opcode 0x70) loads from a register file based on AddrMod and Addr into `RG[VD]`, with inline format adjustment. Supports multiple input formats (FP16_A, FP16_B, FP32, SMAG32, etc.) converting to FP32 in LREGs. IPC=1, latency=1 cycle.

3. **Page**: Same page, SFPSTORE section
   **Key Information**: SFPSTORE (opcode 0x72) stores from an LREG to a register file with optional format conversion. IPC=1, latency=2 cycles (SrcS target) or 3 cycles (Dest target).

### Glean References

None required -- the SFPU ADD operation is straightforward and fully documented through DeepWiki and Confluence sources.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 184-543)
   **Reason**: Understanding how BinaryOpType::ADD maps to specific SFPU function calls and defines.
   **Key Information**: ADD produces `BINOP_INIT = add_binary_tile_init()` and `BINARY_SFPU_OP = add_binary_tile(i*2, i*2+1, i*2)` for float types, or `ADD_INT_INIT = add_int_tile_init()` and `add_int_tile<DataFormat::X>` for integer types.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp`
   **Reason**: Understanding runtime argument setup and core distribution logic shared between the FPU and SFPU binary paths.
   **Key Information**: The `set_eltwise_binary_runtime_args` template handles both initial setup and override paths, implements the two-core-group load balancing strategy, and manages sharded vs interleaved argument differences.
