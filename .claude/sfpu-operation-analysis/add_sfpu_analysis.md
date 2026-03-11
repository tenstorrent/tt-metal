# ADD (SFPU) Implementation Analysis

## Overview

The ADD (SFPU) operation performs element-wise addition of two input tensors using the SFPU (Scalar Floating Point Unit) path. Unlike the FPU-based `add_tiles` path which uses the matrix engine's native add instruction, the SFPU path copies both input tiles into DST registers and executes addition via the SFPU vector unit. This enables FP32 accumulation and supports mixed-precision workflows.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

The SFPU path is selected (via `get_defines_fp32`) when the operation needs FP32 destination accumulation or when integer data types are involved. For floating-point ADD, the define `BINOP_INIT` is set to `add_binary_tile_init();` and `BINARY_SFPU_OP` is set to `add_binary_tile(i*2, i*2+1, i*2);` -- meaning input A occupies even DST slots, input B occupies odd DST slots, and the result overwrites the even slot.

For integer types (INT32, UINT32, UINT16), the `ADD_INT_INIT` define is used instead of `BINOP_INIT`, and the operation dispatches to `add_int_tile<DataFormat::...>`.

## Work Unit Definition

One **work unit** is a single 32x32 tile. The compute kernel processes tiles in blocks of `per_core_block_size` tiles per iteration, for `per_core_block_cnt` iterations per core. The block size is determined by `find_max_block_size()` which finds the largest divisor of the total tiles-per-shard (up to 8) for sharded tensors, or defaults to 1 for interleaved tensors.

## Tensor Format and Layout

### Input Tensor A (src0)

| Property | Value |
|---|---|
| Dimension Convention | NHWC (last dim is width) |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Interleaved or Sharded (height/width/block) |
| Buffer Type | DRAM (interleaved) or L1 (sharded) |
| Data Type | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 |

### Input Tensor B (src1)

| Property | Value |
|---|---|
| Dimension Convention | NHWC (last dim is width) |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Interleaved or Sharded (height/width/block) |
| Buffer Type | DRAM (interleaved) or L1 (sharded) |
| Data Type | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 |

### Output Tensor

| Property | Value |
|---|---|
| Dimension Convention | NHWC (last dim is width) |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Interleaved or Sharded (height/width/block) |
| Buffer Type | DRAM (interleaved) or L1 (sharded) |
| Data Type | Matches output tensor configuration |

### Layout Transformations

No tilize/untilize is performed. The operation expects tiled input and produces tiled output. When the output data type differs from the input and constitutes a typecast (checked via `is_typecast()`), an `SFPU_OP_CHAIN_0` define is generated that calls `typecast_tile()` as a post-processing step in the compute kernel.

## Data Flow Pattern

### Interleaved Path (default)

1. **Reader** iterates over tile IDs `[start_id, start_id + num_tiles)` sequentially
2. For each tile: reads src0 tile from DRAM into CB c_0, reads src1 tile from DRAM into CB c_1 (one tile at a time, with `noc_async_read_barrier` between read and push)
3. **Compute** waits for `per_core_block_size` tiles in both c_0 and c_1
4. Copies tiles from c_0 into DST at even indices (0, 2, 4, ...) via `copy_tile(cb_inp0, i, i*2)`
5. Copies tiles from c_1 into DST at odd indices (1, 3, 5, ...) via `copy_tile(cb_inp1, i, i*2+1)`
6. Executes `add_binary_tile(i*2, i*2+1, i*2)` -- adds DST[even] + DST[odd], result in DST[even]
7. Packs DST[even] tiles into CB c_2 via `pack_tile(i*2, cb_out0)`
8. **Writer** waits for one tile in CB c_2, reads L1 address, writes tile to DRAM via `noc_async_write_page`, pops tile

### Sharded Path

When inputs are sharded, the reader simply does `cb_reserve_back` + `cb_push_back` for the full shard (since the CB is globally allocated at the buffer address). When output is sharded, the writer just does `cb_wait_front` and returns (data is already in place). The compute kernel logic is identical.

### Block/Width Sharded with Interleaved Output

A special writer kernel (`writer_unary_sharded_blocks_interleaved_start_id.cpp`) is used. It waits for the full block of tiles, then writes them row-by-row to DRAM, handling padding by skipping padded width/height tiles.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|
| c_0 | cb_src0 | Input A tiles | 2*max_block_size (interleaved) or num_tiles_per_shard (sharded) | 1 (reader pushes 1 at a time) or per_core_block_size (compute consumes) | Double-buffered (interleaved) or Full-shard (sharded) | Reader | Compute |
| c_1 | cb_src1 | Input B tiles | 2*max_block_size (interleaved) or num_tiles_per_shard (sharded) | 1 (reader pushes 1 at a time) or per_core_block_size (compute consumes) | Double-buffered (interleaved) or Full-shard (sharded) | Reader | Compute |
| c_2 | cb_output | Output tiles | 2*max_block_size (interleaved) or num_tiles_per_shard (sharded/block-width) | per_core_block_size (compute pushes) / 1 (writer pops) | Double-buffered (interleaved) or Full-shard (sharded) | Compute | Writer |
| c_3 | cb_interim0 | Pre-processed input A (conditional) | max_block_size | per_core_block_size | Single-buffered | Compute (pre-scale phase) | Compute (main phase) |
| c_4 | cb_interim1 | Pre-processed input B (conditional) | max_block_size | per_core_block_size | Single-buffered | Compute (pre-scale phase) | Compute (main phase) |

**Note on c_3 and c_4**: These are only created when `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defines are present. For plain ADD (floating-point), these defines are NOT generated, so c_3 and c_4 are not allocated. For integer ADD, these are also not generated. They are used by composite operations like LOGADDEXP, HYPOT, etc. that apply pre-scaling to inputs before the binary operation.

## Pipeline Pattern Summary

**Interleaved mode**: CB c_0 and c_1 are sized at `2 * max_block_size` tiles, which enables **double-buffering** -- the reader can write the next block while the compute kernel processes the current block. CB c_2 is similarly double-buffered for compute-writer overlap.

**Sharded mode**: CBs are sized to hold the entire shard. There is no streaming overlap since all data is already in L1. This is effectively a single-shot pattern.

## Index Calculations

### Interleaved Path
Tile IDs are assigned contiguously. Each core receives a `start_id` equal to the cumulative number of tiles assigned to all preceding cores (`num_tiles_read`). The reader iterates `tile_id` from `start_id` to `start_id + num_tiles_per_core`.

The `TensorAccessor` maps logical tile IDs to physical DRAM bank addresses using the buffer's bank mapping (interleaved across DRAM banks).

### Block/Width Sharded Path
For block or width sharded tensors, tile IDs follow a 2D pattern:
```
start_id = (core_index / num_shards_per_width) * (block_height * block_width * num_shards_per_width)
         + (core_index % num_shards_per_width) * block_width
```
The reader then iterates: for each row `h` in `[0, block_height)`, for each column `w` in `[0, block_width)`, reads tile at `row_start_tile_id + w`, then advances `row_start_tile_id += num_cores_y * block_width` (stride across shard rows in the global tile space).

### DST Register Indexing
In the compute kernel, input A tiles are placed at DST indices `i*2` (even slots) and input B tiles at `i*2+1` (odd slots). The SFPU binary operation reads from both and writes the result to the even slot. Pack extracts from `i*2`.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads, one tile at a time per input, with a `noc_async_read_barrier` after each pair of reads. This is a simple sequential pattern with no batching of async reads.
- **Block/Width Sharded (reader side)**: Row-major 2D traversal within each shard block. Each row is read sequentially, then stride to the next row.
- **Sharded (fully)**: No DRAM reads; data is already in L1 via globally-allocated CB.

### Write Pattern
- **Interleaved (standard writer)**: Sequential tile writes, one tile at a time, with `noc_async_writes_flushed` after each tile (not a full barrier, just flush). Final `noc_async_write_barrier` at end.
- **Block/Width Sharded to Interleaved (special writer)**: Waits for entire block, then writes unpadded tiles row-by-row in 2D pattern, skipping padded tiles. Single barrier at end.
- **Sharded output**: No DRAM writes; output CB is globally allocated at the output buffer address.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Determined by `operation_attributes.worker_grid` |
| Work Splitting | `split_work_to_cores()` for interleaved; shard grid for sharded |
| Core Groups | Group 1: cores with `ceil(num_tiles/num_cores)` tiles; Group 2: cores with `floor(num_tiles/num_cores)` tiles |
| Remainder Handling | Extra tiles distributed to Group 1 cores; excess cores get zero work |
| Traversal Order | Row-major (default for interleaved); shard orientation for sharded |
| Zero-Start Grid Optimization | When grid is a single rectangle starting at (0,0), faster `grid_to_cores` path is used |

For sharded tensors, all cores in the shard grid are active, each processing `num_tiles_per_shard` tiles. There is no core group 2 in this case.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | block_or_width_sharded | uint32_t | 1 if block or width sharded, 0 otherwise |
| 1+ | src0_args (TensorAccessorArgs) | varies | Tensor accessor parameters for input A buffer (only if not IN0_SHARDED) |
| N+ | src1_args (TensorAccessorArgs) | varies | Tensor accessor parameters for input B buffer (only if not IN1_SHARDED) |

**Reader Defines**: `IN0_SHARDED` (if src0 is sharded), `IN1_SHARDED` (if src1 is sharded)

#### Writer Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | dst_args (TensorAccessorArgs) | varies | Tensor accessor parameters for output buffer |

**Writer Defines**: `OUT_SHARDED` (if output is sharded)

#### Compute Kernel

Compile-time configuration via `ComputeConfig`:

| Property | Value | Description |
|---|---|---|
| fp32_dest_acc_en | true if output is Float32/Int32/UInt32 | Enable FP32 accumulation in DST |
| unpack_to_dest_mode | UnpackToDestFp32 for all CBs (non-POWER ops) | Unpack directly to DST in FP32 mode |
| defines | from `get_defines_fp32()` | Operation-specific SFPU defines |

**Key Defines for ADD (floating-point)**:
- `BINOP_INIT` = `add_binary_tile_init();`
- `BINARY_SFPU_OP` = `add_binary_tile(i*2, i*2+1, i*2);`

**Key Defines for ADD (integer)**:
- `ADD_INT_INIT` = `add_int_tile_init();`
- `BINARY_SFPU_OP` = `add_int_tile<DataFormat::Int32>(i*2, i*2+1, i*2);` (or UInt32/UInt16 variant)

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src0_addr | uint32_t | Base address of input A buffer |
| 1 | src1_addr | uint32_t | Base address of input B buffer |
| 2 | num_tiles | uint32_t | Number of tiles this core processes |
| 3 | start_id | uint32_t | Starting tile ID for this core |
| 4 | block_height | uint32_t | Shard block height in tiles (0 if not sharded) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 if not sharded) |
| 6 | num_cores_y | uint32_t | Number of shards per width dimension (used for stride) |

#### Compute Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_size | uint32_t | Number of tiles per block |

#### Writer Kernel (Standard Interleaved)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Base address of output buffer |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for writes |

#### Writer Kernel (Block/Width Sharded to Interleaved)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Base address of output buffer |
| 1 | block_height_tiles | uint32_t | Block height in tiles |
| 2 | block_width_tiles | uint32_t | Block width in tiles |
| 3 | unpadded_block_height_tiles | uint32_t | Actual data height (excluding padding) |
| 4 | unpadded_block_width_tiles | uint32_t | Actual data width (excluding padding) |
| 5 | output_width_tiles | uint32_t | Full output width in tiles |
| 6 | block_num_tiles | uint32_t | Total tiles in block (height * width) |
| 7 | start_id_offset | uint32_t | Tile offset for this core's shard |
| 8 | start_id_base | uint32_t | Base tile ID (always 0) |

## Kernel Implementations

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
- **RISC-V Core**: RISC-V 1 (Reader, NoC0)
- **Responsibilities**: Read input A and input B tiles from DRAM (or signal sharded data availability) into CBs c_0 and c_1
- **I/O**: NoC0 reads from DRAM -> writes to L1 CBs c_0, c_1
- **Key Logic**: Uses `TensorAccessor` for DRAM address resolution. For sharded inputs, simply does reserve+push to signal data availability. For interleaved inputs in non-sharded mode, reads one tile at a time with a barrier between each tile-pair. The `block_or_width_sharded` compile-time flag switches between 2D (row/col) traversal and simple sequential traversal.

### Compute Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`
- **RISC-V Core**: Compute (Unpack + Math + Pack)
- **Responsibilities**: Unpack tiles from CBs into DST registers, execute SFPU binary add, pack results to output CB
- **I/O**: Reads from CBs c_0, c_1 (or c_3, c_4 if pre-scaling); Writes to CB c_2
- **Synchronization**: `tile_regs_acquire` / `tile_regs_commit` / `tile_regs_wait` / `tile_regs_release` for DST register management between unpack and pack phases
- **Key Logic**: The kernel uses `copy_tile` (which calls the unpack engine's copy-to-DST) to load tiles, then invokes the SFPU operation via the `BINARY_SFPU_OP` macro. For ADD, this expands to `add_binary_tile(i*2, i*2+1, i*2)`. The `unary_op_init_common(cb_in0, cb_out0)` call initializes unpack and pack engines. The `copy_tile_to_dst_init_short_with_dt` calls handle data type switching between the two input CBs when they have different formats.

### Writer Kernel (Standard)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **RISC-V Core**: RISC-V 0 (Writer, NoC1)
- **Responsibilities**: Write computed tiles from CB c_2 to DRAM output buffer
- **I/O**: Reads from L1 CB c_2 -> NoC1 writes to DRAM
- **Key Logic**: Uses `TensorAccessor` for address resolution. Writes one tile at a time with `noc_async_writes_flushed` for ordering. For sharded output, simply waits for all tiles in the CB (data is already in the output buffer location).

### Writer Kernel (Block/Width Sharded to Interleaved)

- **File**: `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp`
- **RISC-V Core**: RISC-V 0 (Writer, NoC1)
- **Responsibilities**: Write block-sharded computed tiles to interleaved DRAM output, handling padding
- **Key Logic**: Waits for entire block, then iterates over unpadded rows/columns, writing tiles with proper stride (skipping padded tiles). Uses `noc_async_write_tile` for DRAM writes with `TensorAccessor`.

## Implementation Notes

1. **UnpackToDestFp32 Mode**: For all non-POWER binary SFPU ops (including ADD), all input CBs are configured with `UnpackToDestMode::UnpackToDestFp32`. This means tiles are unpacked directly into the DST accumulator in FP32 format, enabling full-precision arithmetic regardless of input format. For POWER, this is conditional on whether the input is already FLOAT32.

2. **DST Register Layout**: The compute kernel uses an interleaved DST layout where input A occupies even indices and input B occupies odd indices. This is a requirement of the binary SFPU API -- `add_binary_tile(dst_A, dst_B, dst_result)` reads from two DST positions and writes to one.

3. **Fused Activation Optimization**: When ADD has a single fused RELU activation and no input_tensor_a_activation, the `PACK_RELU` define is used instead of a separate SFPU activation step. This leverages the hardware packer's built-in ReLU capability, which is more efficient than an additional SFPU pass.

4. **Block Size Selection**: `find_max_block_size(num_tiles_per_shard)` finds the largest factor of the total tiles up to 8. This determines how many tiles are processed together in one compute iteration, balancing DST register pressure against loop overhead.

5. **No Pre-scaling for ADD**: Plain ADD does not generate `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defines, so CBs c_3 and c_4 are never created, and the pre-processing phases in the compute kernel are compiled out. The compute kernel reads directly from c_0 and c_1.

6. **Runtime Args Caching**: The `set_eltwise_binary_runtime_args<bool>` template is instantiated with `true` for initial creation and `false` for `override_runtime_arguments`. The override path directly modifies cached runtime args via `GetRuntimeArgs` instead of creating new vectors, avoiding allocation overhead for repeated invocations.

7. **Reader Barrier Granularity**: The reader issues `noc_async_read_barrier()` after every single tile pair (one from each input). This is conservative -- it serializes DRAM reads but guarantees data is available before pushing to CBs. The double-buffered CB sizing (2 * max_block_size) provides the overlap opportunity between reader and compute.

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

1. The compute kernel macro `BINARY_SFPU_OP` expands to `add_binary_tile(i*2, i*2+1, i*2)`, defined in `eltwise_binary_sfpu.h`.
2. `add_binary_tile` wraps `MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>(idst0, idst1, odst)))`, dispatching onto the math RISC-V.
3. `llk_math_eltwise_binary_sfpu_binop` (in `llk_math_eltwise_binary_sfpu_binop.h`) calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>` with the function pointer `calculate_sfpu_binary<APPROX, ADD, 8, false>`.
4. `_llk_math_eltwise_binary_sfpu_params_` (in `llk_math_eltwise_binary_sfpu_params.h`) calls `_llk_math_eltwise_binary_sfpu_start_` to set the DST write address and stall until SFPU is ready, then iterates over 4 faces (in RC mode), calling the SFPU function once per face with `TTI_SETRWC` to advance the DST row counter between faces.
5. `calculate_sfpu_binary` (in the metal-overlay `ckernel_sfpu_binary.h`) delegates to `_calculate_sfpu_binary_<APPROX, ADD, 8>` in the tt_llk submodule.
6. `_calculate_sfpu_binary_` performs 8 iterations of `result = in0 + in1` using SFPI vector registers, reading from and writing to DST via `sfpi::dst_reg`.

The init path follows a similar chain: `add_binary_tile_init()` -> `llk_math_eltwise_binary_sfpu_binop_init<APPROX, ADD>()` -> `llk_math_eltwise_binary_sfpu_init<unused, APPROX>(sfpu_binary_init<APPROX, ADD>)` -> `_llk_math_eltwise_binary_sfpu_init_<unused>()` (configures ADDR_MOD and resets counters) + `_sfpu_binary_init_<APPROX, ADD>()` (no-op for ADD since it needs no special init like reciprocal or log).

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h
// (Blackhole version is identical)

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{ // For ADD: APPROXIMATION_MODE=true (APPROX), BINOP=BinaryOp::ADD, ITERATIONS=8
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Each tile in DST occupies 64 rows at hardware level; SFPI stride is 2, so 64/2 = 32 SFPI rows per tile
        constexpr std::uint32_t dst_tile_size_sfpi = 32;
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD from DST[in0_tile + current_row]
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD from DST[in1_tile + current_row]
        sfpi::vFloat result = 0.0f; // SFPLOADI with immediate 0.0

        if constexpr (BINOP == BinaryOp::ADD)
        {
            result = in0 + in1; // SFPADD (alias of SFPMAD with one multiplicand = 1.0): result = in0 * 1.0 + in1
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // SFPSTORE to DST[out_tile + current_row]
        sfpi::dst_reg++; // Advance SFPU row pointer by 1 (hardware stride of 2 rows)
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{
    // For ADD: no initialization needed (no reciprocal table or log constants required)
    // This function body is empty when BINOP == BinaryOp::ADD
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW)
    {
        _init_sfpu_reciprocal_<false>();
    }
    else if constexpr (BINOP == BinaryOp::XLOGY)
    {
        _init_log_<APPROXIMATION_MODE>();
    }
}
```

### SFPU Instructions Used

| Instruction | SFPI Construct | Description |
|-------------|---------------|-------------|
| **SFPLOAD** | `sfpi::dst_reg[index]` (read) | Loads a 32-bit FP value from the Destination register file into an SFPU local register (LREG). Opcode 0x70, IPC=1, latency=1. Supports format conversion on load. |
| **SFPADD** / **SFPMAD** | `in0 + in1` (the `+` operator on `vFloat`) | Performs floating-point addition. SFPADD (opcode 0x85) is an alias of SFPMAD (opcode 0x84) with one multiplicand set to 1.0, computing `(A * 1.0) + B`. IPC=1, latency=2. Operates on FP32 inputs, produces FP32 output. Flushes subnormals. |
| **SFPLOADI** | `sfpi::vFloat result = 0.0f` | Loads an immediate FP16_B value into an SFPU local register. Opcode 0x71, IPC=1, latency=1. The `0.0f` initialization is compiled to this instruction. |
| **SFPSTORE** | `sfpi::dst_reg[index] = result` (write) | Stores a 32-bit value from an SFPU local register back to the Destination register file. Opcode 0x72, IPC=1, latency=2 (SrcS) or 3 (Dest). Supports format conversion on store. |
| **TTI_SETRWC** | (in params dispatch) | Sets the read/write counter for SFPU's DST addressing. Used between face iterations to advance the DST row pointer by 16 rows (two increments of 8). Not an SFPU instruction per se, but a Tensix control instruction that adjusts the SFPU's view into DST. |
| **TTI_STALLWAIT** | (in start/done functions) | Stalls the math pipeline until the SFPU is ready (start) or until SFPU completes (done). Ensures synchronization between the Tensix front-end and the SFPU backend. |

### SFPU Register Usage

**SFPU Local Registers (LREGs)**: The SFPU has 8 local registers (RG[0] through RG[7]). For the ADD kernel:
- `in0` is loaded into one LREG (typically LR0 or LR1, compiler-assigned).
- `in1` is loaded into another LREG.
- `result` occupies a third LREG (initially loaded with 0.0f via SFPLOADI, then overwritten by the SFPMAD result).
- The SFPMAD instruction reads two source LREGs and a third (set to 1.0 via `vConst1` or equivalent), writes to a destination LREG.

**Destination (DST) Register File**: The DST register file holds tile data. Each 32x32 tile occupies 32 "SFPI rows" (due to the hardware stride of 2). The SFPU accesses DST rows via an addressing scheme combining a base address (set by `set_dst_write_addr`) and an auto-incrementing row counter (advanced by `dst_reg++` and `TTI_SETRWC`).

- **Input A tile**: Located at DST index `dst_index_in0` (even slot, e.g., 0, 2, 4...). Each SFPI row contains one vector of elements (the SFPU processes one row of 32 elements per cycle across all SFPU slices).
- **Input B tile**: Located at DST index `dst_index_in1` (odd slot, e.g., 1, 3, 5...).
- **Output tile**: Written to DST index `dst_index_out` (same as `dst_index_in0`, i.e., the even slot). The result overwrites input A in-place.

**Per-face iteration**: The kernel processes 8 rows per face call (ITERATIONS=8). With 4 faces per tile (RC vector mode), this covers all 32 SFPI rows of each tile. Between faces, `TTI_SETRWC` advances the DST row counter by 16 hardware rows (2 increments of 8), which equals 8 SFPI rows (accounting for stride-2), aligning to the next face boundary.

### Address Mode Configuration

The address mode is configured in `eltwise_binary_sfpu_configure_addrmod()` during initialization. For ADD (which uses `SfpuType::unused`), only `ADDR_MOD_7` is configured:

```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

All three increments are zero because the SFPU kernel manages its own DST addressing via the `dst_reg++` auto-increment and explicit `TTI_SETRWC` calls. The ADDR_MOD simply ensures no unintended auto-increment occurs from the Tensix address mode machinery.

**Wormhole vs Blackhole**: The `eltwise_binary_sfpu_configure_addrmod` function and the ADDR_MOD_7 configuration are identical between Wormhole B0 and Blackhole. One minor difference exists in the `_llk_math_eltwise_binary_sfpu_start_` function: Wormhole calls `math::set_addr_mod_base()` before the stall and `math::clear_addr_mod_base()` in the done function, while Blackhole omits these calls. This reflects a difference in how the two architectures manage the address mode base register, but does not affect the ADDR_MOD_7 configuration itself.

**ADDR_MOD_6** is additionally configured (with `dest.incr = 2`) for certain other binary SFPU operations (mul_int32, max, min variants), but this is not used for ADD.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the SFPU binary element-wise operation program factory work? What kernels does it use for reader, compute, and writer? How are circular buffers configured and how is work distributed across cores?"
   **Reason**: Initial reconnaissance to understand the overall architecture and identify key files before reading source code.
   **Key Findings**: Confirmed three-kernel architecture (reader/compute/writer), identified all kernel file paths, understood CB configuration patterns for sharded vs interleaved, and learned about the interim CB usage for pre-scaling operations.

2. **Query**: "How does add_binary_tile work in the SFPU binary operations? Trace the call chain from add_binary_tile through the LLK layers to the ckernel SFPU implementation."
   **Reason**: Needed to map the complete call chain from the compute API down to the SFPU kernel and identify all file paths involved.
   **Key Findings**: Confirmed the chain: `add_binary_tile` -> `llk_math_eltwise_binary_sfpu_binop<ADD>` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_sfpu_binary<ADD>` -> `_calculate_sfpu_binary_<ADD>`. Identified that SFPADD is an alias of SFPMAD.

3. **Query**: "How does add_binary_tile work? Trace from the compute API through llk_math_eltwise_binary_sfpu to the ckernel implementation. What SFPU instructions does it use?"
   **Reason**: Cross-referenced with tt-llk repo for deeper LLK-level understanding and SFPU instruction mapping.
   **Key Findings**: Confirmed that the `+` operator on `sfpi::vFloat` maps to the SFPADD hardware instruction (opcode 0x85), which is a 2-cycle FMA operation. The params dispatch iterates over 4 faces with SETRWC between each.

### Confluence References

1. **Source**: Tensix SFPU Instruction Set Architecture (Page ID: 1170505767)
   **Sections consulted**: SFPADD, SFPLOADI, SFPLOAD, SFPSTORE, SFPMAD instruction definitions.
   **Key Findings**: SFPADD (opcode 0x85) is confirmed as an alias of SFPMAD that declares intent for one operand to be 1.0. It performs the full FMA operation `(A * B) + C` in FP32 with IPC=1 and latency=2. SFPLOAD (opcode 0x70) loads from register files with format conversion, IPC=1, latency=1. SFPSTORE (opcode 0x72) stores back with format conversion, IPC=1, latency=2-3. All instructions flush subnormals to zero (SFPADD/SFPMAD) or do not (SFPLOAD/SFPSTORE).

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp`
   **Reason**: Needed to understand what defines are generated for ADD operation specifically.
   **Key Information**: For floating-point ADD: `BINOP_INIT` = `add_binary_tile_init()`, `BINARY_SFPU_OP` = `add_binary_tile(i*2, i*2+1, i*2)`. For integer ADD: `ADD_INT_INIT` = `add_int_tile_init()` with type-specific `add_int_tile<>`. No pre-scaling defines generated.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp`
   **Reason**: Needed to understand runtime argument assignment and core distribution logic.
   **Key Information**: Uses `split_work_to_cores` for interleaved, shard grid for sharded. Supports zero-start-grid optimization. Core group 1/2 splitting handles remainder tiles. Block/width sharded uses specialized 2D tile ID calculation.

3. **Source**: `tt_metal/api/tt-metalium/work_split.hpp`
   **Reason**: Needed to understand `find_max_block_size` behavior.
   **Key Information**: Finds largest divisor of a value up to max_block_size (default 8).
