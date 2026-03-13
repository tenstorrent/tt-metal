# LCM (Least Common Multiple) Implementation Analysis

## Overview
The LCM operation computes the element-wise least common multiple of two integer tensors: `y = lcm(a, b)`. It is implemented as a binary SFPU operation using the formula `lcm(a, b) = |a| / gcd(a, b) * |b|`, where GCD is computed via the binary GCD algorithm on the SFPU. Both inputs must be INT32 or UINT32, and individual element magnitudes are constrained to |value| <= 2^15 - 1 (32,767) due to the internal multiplication strategy that splits 16-bit halves.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Path Selection: FPU vs SFPU

The binary device operation uses `BinaryDeviceOperation::select_program_factory` to choose between `ElementWiseMultiCore` (FPU path) and `ElementWiseMultiCoreSfpu` (SFPU path). The selection occurs when both input tensors have the same height and width (no broadcasting required). The helper function `utils::is_binary_sfpu_op(op, dtype1, dtype2)` is then consulted. For `BinaryOpType::LCM`, this function returns `true` when both inputs are `DataType::INT32` or both are `DataType::UINT32`. If LCM is requested with non-matching or unsupported types, the operation would not pass validation. Since LCM is exclusively an SFPU operation (there is no FPU path for it), the `ElementWiseMultiCoreSfpu` factory is always selected for valid LCM calls.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `physical_volume / TILE_HW` |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_size` tiles per block |

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|---------------|---------------|
| **Logical shape** | [N, ...] (arbitrary rank) | [N, ...] (same H, W as A) |
| **Dimension convention** | NHWC | NHWC |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | INT32 or UINT32 | INT32 or UINT32 (must match A) |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input A |
| **Dimension convention** | NHWC |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | INT32 or UINT32 (matches inputs) |

### Layout Transformations
No tilize/untilize or format conversions are performed. All tensors must already be in TILE_LAYOUT. The compute kernel operates on tiles directly in DST registers.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src0, src1) | CB c_0, CB c_1 | reserve_back, push_back (per tile) |
| 2 | Compute | CB c_0, CB c_1 | CB c_2 | wait_front, copy_tile to DST, SFPU lcm_tile, pack_tile, pop_front, push_back |
| 3 | Writer | CB c_2 | DRAM/L1 (output) | wait_front, pop_front (per tile) |

The reader fetches one tile at a time from each input into CB c_0 and CB c_1. The compute kernel waits for a full block of tiles in both input CBs, copies them into DST registers (input A at even indices, input B at odd indices), executes the LCM SFPU operation, packs results to CB c_2, and releases the input CBs. The writer drains CB c_2 tile-by-tile to the output buffer.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (interleaved) | Capacity (sharded) | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------------------|-------------------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input A staging | 2 tiles | num_tiles_per_shard | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_src1 | Input B staging | 2 tiles | num_tiles_per_shard | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_2 | cb_out0 | Output staging | 2 tiles | num_tiles_per_shard | 1 tile | Double (interleaved) / Single (sharded) | Compute | Writer | Program |

**Notes**:
- CB c_3 and c_4 (intermediate buffers for input prescaling) are NOT allocated for LCM because the `get_defines_fp32` function does not define `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` for `BinaryOpType::LCM`. LCM has no input prescaling stage.
- For interleaved mode, capacity is `2 * max_block_size` tiles where `max_block_size` defaults to 1 for non-sharded, giving 2 tiles (double-buffered).
- For sharded inputs/outputs, the CB is backed by the tensor's L1 buffer directly via `set_globally_allocated_address`.

## Pipeline Pattern Summary

In interleaved mode, CB c_0, c_1, and c_2 each have capacity of 2 tiles with a block size of 1 tile, enabling **double-buffered** overlap between reader-compute and compute-writer stages. In sharded mode, all shard tiles are loaded at once (single-buffered bulk transfer).

## Index Calculations

The reader kernel uses `TensorAccessor` for interleaved inputs to translate a linear tile ID (`start_id + offset`) to a physical DRAM bank address and local offset. For sharded inputs, all tiles are already in L1, so the reader simply does `reserve_back / push_back` on the entire shard to make tiles visible to compute.

For the block-or-width-sharded path, tile IDs are computed as:
- `row_start_tile_id = start_id`, incremented by `num_cores_y * block_width` per row
- Within each row, tiles are accessed sequentially from `tile_id` to `tile_id + block_width`

The start_id per core in the interleaved path is the cumulative sum of tiles assigned to previous cores (`num_tiles_read`).

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads from DRAM, one tile at a time per input, with `noc.async_read_barrier()` after each tile pair.
- **Sharded**: No reads needed; data is already in L1. The reader marks the shard as available via `reserve_back / push_back`.
- **Block/width sharded (non-sharded input)**: Strided reads with `num_cores_y * block_width` stride between rows.

### Write Pattern
- **Interleaved**: Sequential tile writes to DRAM, one tile at a time, with flush and barrier.
- **Sharded output**: Writer does `wait_front(num_pages)` and the output CB is backed by the output tensor's L1 buffer directly. No explicit writes needed.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (row-major) or 2D (sharded) |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `all_device_cores.num_cores()` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_tiles / num_cores)` tiles, group 2 gets `floor(num_tiles / num_cores)` tiles |

Work distribution uses `tt::tt_metal::split_work_to_cores` for interleaved mode, which creates two core groups to handle remainder tiles. For sharded mode, each core processes exactly its shard's tiles (`shard_shape[0] * shard_shape[1] / TILE_HW`). Cores beyond the active set receive zero-tile assignments.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_or_width_sharded | uint32_t | 1 if block/width sharded, 0 otherwise |
| 1+ | src0_args (TensorAccessorArgs) | varies | Tensor accessor parameters for input A (omitted if IN0_SHARDED) |
| N+ | src1_args (TensorAccessorArgs) | varies | Tensor accessor parameters for input B (omitted if IN1_SHARDED) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | dst_args (TensorAccessorArgs) | varies | Tensor accessor parameters for output buffer |

#### Compute Kernel
No compile-time arguments. Behavior is controlled by preprocessor defines: `BINOP_INIT` -> `lcm_tile_init();` and `BINARY_SFPU_OP` -> `lcm_tile(i*2, i*2+1, i*2);`.

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | Address of input tensor A buffer |
| 1 | src1_addr | uint32_t | Address of input tensor B buffer |
| 2 | num_tiles | uint32_t | Total tiles this core must process |
| 3 | start_id | uint32_t | Starting tile index for this core |
| 4 | block_height | uint32_t | Shard block height in tiles (0 if interleaved) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 if interleaved) |
| 6 | num_cores_y | uint32_t | Number of shards per width dimension (0 if interleaved) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_size | uint32_t | Number of tiles per block |

#### Writer Kernel (interleaved, non-block-sharded)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Address of output tensor buffer |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile index for output |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 | CB c_0, CB c_1 | Read input A and B tiles |
| Compute | TRISC (RISCV_2) | N/A | CB c_0, CB c_1 | CB c_2 | copy_tile to DST, lcm_tile SFPU op, pack_tile |
| Writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 | Write output tiles |

### Reader Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` |
| Assigned cores | All worker cores (`all_device_cores`) |

**Key Logic**:
- For sharded inputs (`IN0_SHARDED` / `IN1_SHARDED`), the reader simply calls `cb.reserve_back(num_tiles)` and `cb.push_back(num_tiles)` to make the entire shard visible to compute. No NoC reads are performed.
- For interleaved inputs, iterates `tile_id` from `start_id` to `start_id + num_tiles`, performing one `noc.async_read` per tile per input, followed by `noc.async_read_barrier()` before pushing.
- For block/width-sharded layout with interleaved inputs, uses a 2D loop (block_height x block_width) with row stride of `num_cores_y * block_width`.
- **Synchronization**: Produces into CB c_0 and CB c_1. Each tile: `reserve_back(1)` -> NoC read -> `async_read_barrier()` -> `push_back(1)`.

### Compute Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` |
| Assigned cores | All worker cores (`all_device_cores`) |

**Key Logic**:
- Outer loop iterates `per_core_block_cnt` times. For LCM, no prescaling stage is active (no `SFPU_OP_INIT_PRE_IN0_0` / `SFPU_OP_INIT_PRE_IN1_0`), so `cb_inp0 = cb_in0 = c_0` and `cb_inp1 = cb_in1 = c_1`.
- Waits for `per_core_block_size` tiles in both CB c_0 and CB c_1, then reserves the same count in CB c_2.
- Acquires tile registers (`tile_regs_acquire` + `tile_regs_wait`).
- Copies input A tiles to DST at even indices (`i*2`) and input B tiles to DST at odd indices (`i*2+1`) using `copy_tile` with `copy_tile_to_dst_init_short_with_dt` for proper data type handling between the two inputs.
- For each tile pair, executes `BINOP_INIT` -> `lcm_tile_init()` and `BINARY_SFPU_OP` -> `lcm_tile(i*2, i*2+1, i*2)`. The result overwrites DST at index `i*2`.
- Packs result from DST[i*2] to CB c_2 via `pack_tile`.
- After all tiles in the block: `tile_regs_commit`, `tile_regs_release`, then `cb_pop_front` on both inputs and `cb_push_back` on output.
- **Synchronization**: Consumes CB c_0 and c_1 via `cb_wait_front` / `cb_pop_front`. Produces CB c_2 via `cb_reserve_back` / `cb_push_back`.

**SFPU Algorithm Details** (from `ckernel_sfpu_lcm.h`):
1. Load input A and B from DST into SFPU LREGs.
2. Compute `gcd(a, b)` using the binary GCD algorithm (`calculate_sfpu_gcd_body<15>`), which operates on absolute values with up to 15 iterations (inputs constrained to 15-bit magnitude).
3. Compute `1/gcd(a, b)` using Newton's method (2 iterations) for reciprocal approximation in FP32 space.
4. Reload `|a|` from DST, cast to FP32, multiply by `1/gcd` to get `|a|/gcd(a,b)`, then round back to integer.
5. Reload `|b|` from DST, then compute `(|a|/gcd) * |b|` using `calculate_sfpu_mul_u16_to_u32_body`, which splits each operand into high and low 8-bit halves, multiplies all four partial products in FP32, and reconstructs the 32-bit integer result.
6. Store the result back to DST at the output index.

### Writer Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` |
| Assigned cores | All worker cores (`all_device_cores`) |

**Key Logic**:
- For sharded output (`OUT_SHARDED`), simply calls `cb.wait_front(num_pages)` -- the output CB is backed by the output tensor's L1 buffer, so no explicit write is needed.
- For interleaved output, iterates tile IDs from `start_id` to `start_id + num_pages`, performing `cb.wait_front(1)` -> `noc.async_write` -> `noc.async_writes_flushed()` -> `cb.pop_front(1)` per tile, with a final `noc.async_write_barrier()`.
- Uses `TensorAccessor` for address translation from tile ID to physical bank address.
- **Synchronization**: Consumes CB c_2 via `wait_front(1)` / `pop_front(1)` per tile.

## Implementation Notes

- **Program factory variants**: LCM uses exclusively the `ElementWiseMultiCoreSfpu` program factory. There is no FPU path. The factory is selected when `is_binary_sfpu_op` returns true for LCM with matching INT32/UINT32 types. An alternative writer kernel (`writer_unary_sharded_blocks_interleaved_start_id.cpp`) is used when block/width sharded input produces interleaved output.
- **Type-based operation variants**: Only INT32 and UINT32 are supported. Both inputs must have the same data type. No FP32/BF16 paths exist for LCM.
- **UnpackToDestFP32 mode**: Enabled for all CBs (c_0, c_1, c_3, c_4) since LCM is not `BinaryOpType::POWER`. This allows the SFPU to work with FP32 precision in DST registers during the reciprocal and multiplication stages.
- **Broadcast type selection**: N/A. LCM requires both inputs to have the same height and width. No broadcasting is supported.
- **Sharding support and constraints**: Height, width, and block sharding are all supported for inputs and output. When any tensor is sharded, its shard spec determines the core grid and work distribution. The CB is allocated as globally-addressed (backed by tensor buffer) for sharded tensors.
- **FP32 dest accumulation**: Enabled when output data type is Float32, Int32, or UInt32. Since LCM operates on INT32/UINT32, `fp32_dest_acc_en` is always true for this operation.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary SFPU element-wise program factory work? What kernels does it use for reader, compute, and writer? How does it handle interleaved vs sharded tensors?"
   **Reason**: Needed architectural understanding of the SFPU binary program factory pattern before reading source code.
   **Key Findings**: Confirmed the three kernel files (reader_binary_interleaved_start_id, eltwise_binary_sfpu_kernel, writer_unary_interleaved_start_id), the sharded vs interleaved CB configuration strategy, and the role of defines like IN0_SHARDED/IN1_SHARDED/OUT_SHARDED.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 362-365, 535)
   **Reason**: Needed to determine which preprocessor defines are generated for LCM.
   **Key Information**: LCM defines `BINOP_INIT` as `lcm_tile_init()` and `BINARY_SFPU_OP` as `lcm_tile(i*2, i*2+1, i*2)`. No `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` are set (no input prescaling).

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp` (lines 22-66, 69-94)
   **Reason**: Needed to understand the SFPU path selection logic.
   **Key Information**: `is_binary_sfpu_op` returns true for LCM when both inputs are INT32 or UINT32. `select_program_factory` returns `ElementWiseMultiCoreSfpu{}` when the SFPU check passes and input dimensions match.

3. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_lcm.h`
   **Reason**: Needed to understand the SFPU-level LCM algorithm.
   **Key Information**: LCM is computed as `|a|/gcd(a,b) * |b|` using: binary GCD (15-bit), Newton's reciprocal (2 iterations), FP32 multiplication with integer rounding, and a 16-bit-split multiplication for the final product.

4. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h`
   **Reason**: Needed to understand the GCD subroutine used by LCM.
   **Key Information**: Binary GCD uses SFPU instruction replay for loop unrolling, operating on absolute values with swap-and-subtract pattern. For LCM, it runs with `max_input_bits=15` (14 iterations instead of 30 for full 31-bit GCD).
