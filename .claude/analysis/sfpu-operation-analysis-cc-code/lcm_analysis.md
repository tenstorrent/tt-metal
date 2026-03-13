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

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/lcm.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_lcm.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_lcm.h` (also depends on `ckernel_sfpu_gcd.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `lcm_tile(i*2, i*2+1, i*2)` (defined in `tt_metal/hw/inc/api/compute/lcm.h`), which wraps the call inside the `MATH()` macro to execute on the math RISC-V.
2. This invokes `llk_math_eltwise_binary_sfpu_lcm<APPROX>(idst0, idst1, odst)` (in `llk_math_eltwise_binary_sfpu_lcm.h`), which delegates to the parameters dispatch template.
3. `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(sfpu::calculate_sfpu_lcm, dst_index0, dst_index1, odst, VectorMode::RC)` (in `llk_math_eltwise_binary_sfpu_params.h`) sets the DST write address, stalls until SFPU is ready, then loops over all 4 faces calling `calculate_sfpu_lcm` per face, advancing the DST read/write counter by 16 rows between faces.
4. `calculate_sfpu_lcm<8>(dst_index_in0, dst_index_in1, dst_index_out)` (in `ckernel_sfpu_lcm.h`) executes 8 iterations (processing 8 rows of the 16-row face), each iteration: loads inputs from DST, calls `calculate_sfpu_gcd_body<15>()`, computes reciprocal via Newton's method, computes `|a|/gcd * |b|` via `calculate_sfpu_mul_u16_to_u32_body()`, and stores the result.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of a 32x32 tile are processed. Each face consists of 16 rows of 16 elements.
- **Operation invocation**: The params dispatch loops `for (int face = 0; face < 4; face++)`, calling `calculate_sfpu_lcm<8>(dst_index_in0, dst_index_in1, dst_index_out)` once per face. Each call internally iterates 8 times (8 rows per face, with 2 rows per SFPU vector, totaling 16 rows = one face). ITERATIONS=8 is the default template parameter.
- **DEST address progression**: Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice per face (advancing by 8+8=16 rows). Within `calculate_sfpu_lcm`, `dst_reg++` at the end of each iteration advances the SFPU dest register pointer by 1 row (each SFPU operation works on 1 row of 16 elements at a time, but `dst_reg++` increments the implicit row counter used by `SFPLOAD`/`SFPSTORE` for the next iteration's row offset).

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions with complex CC manipulation inside the GCD subroutine (`calculate_sfpu_gcd_body`). The main `calculate_sfpu_lcm` function and `calculate_sfpu_mul_u16_to_u32_body` helper use `SFPIADD_MOD1_CC_NONE` (no CC updates), so they follow Style A inline comments. The GCD body uses `SFPSETCC`, `SFPENCC`, `SFPLZ` with CC modifiers, and `SFPSWAP` under CC guard, qualifying for a Style B CC State Machine diagram.

#### GCD Initialization (Replay Buffer Setup)

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h

inline void calculate_sfpu_gcd_init() {
    TTI_REPLAY(0, 7 * 4, 0, 1); // Begin recording 28 instructions into replay buffer
    #pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        // We store {-a, a} in {LREG0, LREG2}, which is convenient for isolating the LSB of a.
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG2, 0); // LREG2 = +a
        TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG0, 0); // LREG0 &= a (isolate LSB and overwrite -a)
        TTI_SFPLZ(0, p_sfpu::LREG0, p_sfpu::LREG0, SFPLZ_MOD1_CC_NE0); // LREG0 = clz(LREG0), disable lanes where a == 0
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE); // LREG0 += d
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG0, SFPSHFT2_MOD1_SHFT_LREG); // LREG0 = a >> -LREG0, making a definitely odd (now both a and b are odd)
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, SFPSWAP_MOD1_VEC_MIN_MAX); // ensure b < a
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // a = b - a (now a is even)
    }
}
```

#### GCD Body (Binary GCD Algorithm)

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h

template <int max_input_bits = 31>
inline void calculate_sfpu_gcd_body() {
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
    TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG2, 0);

    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);
    TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);
    TTI_SFPLZ(0, p_sfpu::LREG3, p_sfpu::LREG3, 0);

    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG2, SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, 6);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, 0);

    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);

    int iterations = max_input_bits - 1;

    #pragma GCC unroll 7
    while (iterations / 4 > 0) {
        TTI_REPLAY(0, 7 * 4, 0, 0);
        iterations -= 4;
    }

    TTI_REPLAY(0, 7 * iterations - 1, 0, 0);

    TTI_SFPENCC(0, 0, 0, 0);
}
```

**CC State Machine -- `calculate_sfpu_gcd_body` (pre-loop setup)**

```
calculate_sfpu_gcd_body — CC State Transitions (pre-loop setup)
════════════════════════════════════════════════════════════════

  CC State: ALL_ENABLED                   <-- initial state
       |
       |  SFPMOV  L2 = L0                 (no CC effect)
       |  SFPOR   L2 |= L1               (no CC effect) -- c = a | b
       |  SFPMOV  L3 = L2                 (no CC effect)
       |  SFPIADD L3 = -L3  CC_NONE      (no CC effect) -- d = -c
       |  SFPAND  L3 &= L2               (no CC effect) -- d = c & (-c), isolates LSB
       |  SFPLZ   L3 = clz(L3)           (no CC effect) -- d = leading zeros count
       |  SFPSHFT2 L2 = L1 << L3         (no CC effect) -- c = b << d
       |
       v
  +----------------------------------+
  | SFPSETCC  LREG2, mod=6           |
  |                                  |
  | CC <- (LREG2 == 0)              |
  | i.e. enabled where b is even    |
  | (b shifted by d produced zero)  |
  +---------------+------------------+
                  |
                  v
  CC State: ENABLED where b is even (b << clz(LSB(a|b)) == 0)
       |
       |  SFPSWAP L0, L1                  (CC-guarded: swap a,b only in even-b lanes)
       |
       v
  +----------------------------------+
  | SFPENCC                          |
  |                                  |
  | CC <- ALL_ENABLED                |
  +---------------+------------------+
                  |
                  v
  CC State: ALL_ENABLED
       |
       |  SFPABS  L0 = |L0|              (all lanes)
       |  SFPABS  L1 = |L1|              (all lanes)
       |  SFPIADD L0 = -L0  CC_NONE      (no CC effect) -- a = -|a|
       |  SFPIADD L3 = -L3  CC_NONE      (no CC effect) -- d = -d
       |
       v
  [Enter replay loop: 14 iterations of 7-instruction GCD step]
```

**CC State Machine -- GCD Replay Loop (each 7-instruction iteration)**

Each replayed iteration executes the following 7 instructions recorded in `calculate_sfpu_gcd_init`. The CC is manipulated by `SFPLZ` with `SFPLZ_MOD1_CC_NE0`:

```
GCD Replay Iteration — CC State Transitions
════════════════════════════════════════════════════════════════

  CC State: ALL_ENABLED                   <-- start of iteration
       |
       |  SFPABS  L2 = |L0|              (no CC effect) -- L2 = +a (since L0 = -a)
       |  SFPAND  L0 &= L2              (no CC effect) -- isolate LSB of a
       |
       v
  +----------------------------------+
  | SFPLZ  L0 = clz(L0)             |
  | SFPLZ_MOD1_CC_NE0               |
  |                                  |
  | CC <- (L0_input != 0)           |
  | Disables lanes where a == 0     |
  | (GCD already found for those)   |
  +---------------+------------------+
                  |
                  v
  CC State: ENABLED where a != 0
       |
       |  SFPIADD L0 += L3  CC_NONE      (CC-guarded: only a!=0 lanes, no CC update)
       |  SFPSHFT2 L0 = L2 >> (-L0)      (CC-guarded: strip trailing zeros, make a odd)
       |  SFPSWAP L0, L1  VEC_MIN_MAX    (CC-guarded: ensure b <= a by swapping)
       |  SFPIADD L0 = L1 - L0  CC_NONE  (CC-guarded: a = b - a, making a even)
       |
       v
  [CC persists into next iteration or cleared by final SFPENCC]
```

Note: For LCM with `max_input_bits=15`, `iterations = 14`. The main loop replays `14/4 = 3` batches of 4 iterations (12 iterations), then the remaining `14 - 12 = 2` iterations are replayed with `TTI_REPLAY(0, 7*2-1, 0, 0)` = 13 instructions (the last instruction of the final iteration is skipped as it only affects `a`, which is no longer needed). After all iterations, `TTI_SFPENCC` resets CC to ALL_ENABLED.

#### LCM Main Function

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_lcm.h

template <int ITERATIONS = 8>
inline void calculate_sfpu_lcm(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64; // 64 rows per tile in DEST

        TT_SFPLOAD(p_sfpu::LREG0, 4, 3, dst_index_in0 * dst_tile_size);  // Load a from DST (int32 raw mode)
        TT_SFPLOAD(p_sfpu::LREG1, 4, 3, dst_index_in1 * dst_tile_size);  // Load b from DST (int32 raw mode)

        // Binary GCD: assumes |a| < 2^15 and |b| < 2^15. Result gcd in LREG1.
        calculate_sfpu_gcd_body<15>();

        // --- Newton's method reciprocal of gcd(a,b) ---
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG2, 0);    // LREG2 = float(gcd), cast sign-magnitude int to FP32
        TTI_SFPSETSGN(1, p_sfpu::LREG2, p_sfpu::LREG1, 1); // LREG1 = |LREG2|, force positive sign (imm=1 sets sign to +)
        TTI_SFPSETEXP(126, p_sfpu::LREG1, p_sfpu::LREG1, 1); // Normalize: set exponent to 126 (range [0.5, 1.0))
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG12, p_sfpu::LREG0, 0); // LREG0 = LREG1 * LREG13 + LREG12 = x * (-32/17) + (48/17), initial Newton estimate
        TTI_SFPEXEXP(0, p_sfpu::LREG2, p_sfpu::LREG3, 0); // LREG3 = unbiased exponent of LREG2 (the original float gcd)

        // 1st Newton iteration: r = r * (2 - x*r)
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, 0); // LREG2 = r*x + 1.0 (should be 2 - x*r but uses MAD trick)
        TTI_SFPNOP;                                         // Pipeline stall for MAD result
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0, 0);    // LREG0 = LREG2*r + r = refined reciprocal
        TTI_SFPNOP;                                         // Pipeline stall
        // 2nd Newton iteration
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, 0); // LREG2 = r*x + 1.0
        TTI_SFPNOP;
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0, 0);    // LREG0 = refined 1/normalized_gcd
        TTI_SFPIADD((-126) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_IMM); // LREG3 = exp - 126, adjust exponent bias

        // Re-bias the reciprocal to account for original exponent
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG4, 0); // LREG4 = exponent of reciprocal
        TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // LREG3 = LREG4 - LREG3 (combined exponent)
        TTI_SFPSETEXP(0, p_sfpu::LREG0, p_sfpu::LREG3, 0); // Set reciprocal's exponent to LREG3, completing 1/gcd(a,b)

        // Load |a| and multiply by 1/gcd(a,b)
        TT_SFPLOAD(p_sfpu::LREG0, 4, 3, dst_index_in0 * dst_tile_size); // Reload a
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);   // LREG0 = |a|
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);      // Cast |a| to FP32
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // LREG0 = |a| * (1/gcd) in FP32
        TT_SFPLOAD(p_sfpu::LREG1, 4, 3, dst_index_in1 * dst_tile_size); // Reload b
        TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, 0);   // LREG1 = |b|

        // Convert |a|/gcd to integer (FP32 -> unsigned int via round-to-zero)
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, 6); // Mod1=6: FP32_TO_UINT16 (round to zero)

        // Compute lcm(a,b) = (|a|/gcd) * |b| via 8-bit-split multiplication
        calculate_sfpu_mul_u16_to_u32_body(); // Result in LREG4

        TT_SFPSTORE(p_sfpu::LREG4, 4, 3, dst_index_out * dst_tile_size); // Store result to DST
        dst_reg++; // Advance to next row
    }
}
```

#### U16-to-U32 Multiplication Helper

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_lcm.h

// Multiplies two unsigned 16-bit integers in LREG0 and LREG1, producing a 32-bit result in LREG4.
// Splits each operand into high byte (bits[15:8]) and low byte (bits[7:0]),
// computes 4 partial products in FP32, then reassembles via shifts and adds.
inline void calculate_sfpu_mul_u16_to_u32_body() {
    TTI_SFPLOADI(p_sfpu::LREG7, SFPLOADI_MOD0_USHORT, 0xff); // LREG7 = 0x000000FF (byte mask)
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);          // LREG2 = copy of a
    TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);          // LREG3 = copy of b
    TTI_SFPSHFT((-8) & 0xfff, 0, p_sfpu::LREG2, 1);          // LREG2 = a >> 8 (high byte of a)
    TTI_SFPSHFT((-8) & 0xfff, 0, p_sfpu::LREG3, 1);          // LREG3 = b >> 8 (high byte of b)
    TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG0, 0);          // LREG0 = a & 0xFF (low byte of a)
    TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG1, 0);          // LREG1 = b & 0xFF (low byte of b)
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);            // Cast a_lo to FP32
    TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);            // Cast b_lo to FP32
    TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);            // Cast a_hi to FP32
    TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);            // Cast b_hi to FP32
    // Four partial products in FP32
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG4, 0); // LREG4 = a_lo * b_lo
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG5, 0); // LREG5 = a_lo * b_hi
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG6, 0); // LREG6 = a_hi * b_lo
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG7, 0); // LREG7 = a_hi * b_hi
    // Convert FP32 products back to unsigned integers (Mod1=6: FP32_TO_UINT16)
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG4, p_sfpu::LREG4, 6);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG5, p_sfpu::LREG5, 6);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG6, p_sfpu::LREG6, 6);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG7, p_sfpu::LREG7, 6);
    // Shift partial products to their correct bit positions
    TTI_SFPSHFT(8, 0, p_sfpu::LREG5, 1);  // a_lo*b_hi << 8
    TTI_SFPSHFT(8, 0, p_sfpu::LREG6, 1);  // a_hi*b_lo << 8
    TTI_SFPSHFT(16, 0, p_sfpu::LREG7, 1); // a_hi*b_hi << 16
    // Sum all partial products into LREG4
    TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, SFPIADD_MOD1_CC_NONE); // LREG4 += LREG5
    TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG4, SFPIADD_MOD1_CC_NONE); // LREG4 += LREG6
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG4, SFPIADD_MOD1_CC_NONE); // LREG4 += LREG7
}
```

#### LCM Init Function

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_lcm.h

inline void calculate_sfpu_lcm_init()
{
    calculate_sfpu_gcd_init(); // Record GCD loop body into replay buffer

    // Constants for Newton's reciprocal: 1/x ~= (48/17) - x*(32/17)
    sfpi::vConstFloatPrgm0 = 48.0f / 17.0f; // LREG12 = 48/17 ~= 2.8235
    sfpi::vConstFloatPrgm1 = 32.0f / 17.0f; // LREG13 = 32/17 ~= 1.8824
}
```

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `TT_SFPLOAD` | Loads a 32-element vector from a DST register row into an SFPU LREG. Used with mode 4/3 for raw int32 load (no format conversion). |
| `TT_SFPSTORE` | Stores a 32-element vector from an SFPU LREG back to a DST register row. Used with mode 4/3 for raw int32 store. |
| `TTI_SFPMOV` | Copies one LREG to another (lanewise). |
| `TTI_SFPOR` | Bitwise OR between two LREGs. |
| `TTI_SFPAND` | Bitwise AND between two LREGs. Used for masking (isolating LSB, byte extraction). |
| `TTI_SFPABS` | Computes absolute value (two's complement integer mode by default). |
| `TTI_SFPIADD` | Integer add with various modes: `CC_NONE` (no CC update), `ARG_IMM` (immediate operand), `ARG_2SCOMP_LREG_DST` (negate dst before add, i.e. subtraction). |
| `TTI_SFPLZ` | Count leading zeros. With `SFPLZ_MOD1_CC_NE0`, also sets CC to disable lanes where input is zero. |
| `TTI_SFPSHFT` | Arithmetic/logical shift by immediate amount. Negative immediate = right shift. |
| `TTI_SFPSHFT2` | Shift using register value as shift amount (`SFPSHFT2_MOD1_SHFT_LREG`). |
| `TTI_SFPSWAP` | Swaps two LREGs lanewise. With `SFPSWAP_MOD1_VEC_MIN_MAX`, assigns min to VD and max to VC. CC-guarded in setup, unconditional in replay. |
| `TTI_SFPSETCC` | Sets condition code based on LREG value. Mode 6: CC enabled where LREG == 0. |
| `TTI_SFPENCC` | Resets condition code to ALL_ENABLED (re-enables all lanes). |
| `TTI_SFPCAST` | Converts sign-magnitude integer to FP32. |
| `TTI_SFPSETSGN` | Sets the sign bit of an FP32 value. With imm=1/mod=1, forces positive. |
| `TTI_SFPSETEXP` | Sets the exponent field of an FP32 value to an immediate or register value. |
| `TTI_SFPEXEXP` | Extracts the unbiased exponent from an FP32 value into an integer LREG. |
| `TTI_SFPMAD` | Fused multiply-add: `VD = VA * VB + VC`. Used for Newton's method iterations. |
| `TTI_SFPMUL` | Multiply: `VD = VA * VB + LCONST_0` (effectively `VA * VB`). |
| `TTI_SFPNOP` | No-operation. Pipeline stall to ensure previous MAD result is ready. |
| `TTI_SFPLOADI` | Load immediate into LREG. `SFPLOADI_MOD0_USHORT` loads a 16-bit unsigned immediate. |
| `TTI_SFP_STOCH_RND` | Stochastic/deterministic rounding conversion. Mod1=6: FP32 to unsigned 16-bit integer (round to zero). |
| `TTI_REPLAY` | Replays previously recorded instructions from the replay buffer. Used for loop unrolling of GCD iterations without code size expansion. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Input `a` (loaded from DST). Reused as Newton's reciprocal estimate `r`. Later reloaded with `\|a\|` for final multiplication. |
| **LREG1** | Input `b` (loaded from DST). After GCD body, holds `gcd(a,b)`. Later reloaded with `\|b\|`. In mul helper, holds `b_lo`. |
| **LREG2** | Scratch: FP32 version of gcd, copy of `a` in GCD, `a_hi` in mul helper. |
| **LREG3** | GCD: trailing-zero count `d`. Reciprocal: exponent accumulator. Final: holds `1/gcd(a,b)`. In mul helper, `b_hi`. |
| **LREG4** | Reciprocal: scratch for exponent extraction. In mul helper: accumulator for final 32-bit product (output). |
| **LREG5** | Mul helper: partial product `a_lo * b_hi`. |
| **LREG6** | Mul helper: partial product `a_hi * b_lo`. |
| **LREG7** | Mul helper: byte mask `0xFF`, then partial product `a_hi * b_hi`. |
| **LREG12** (`vConstFloatPrgm0`) | Newton's constant `48/17` (addend in initial estimate). |
| **LREG13** (`vConstFloatPrgm1`) | Newton's constant `32/17` (multiplier in initial estimate, negated in the MAD). |
| **LCONST_0** | Hardware constant `0.0`. Used as the addend in SFPMUL (making it a pure multiply). |
| **LCONST_1** | Hardware constant `1.0`. Used as addend in Newton iteration MAD. |
| **DST registers** | Input tiles at `dst_index_in0 * 64` and `dst_index_in1 * 64`; output written to `dst_index_out * 64`. Each tile occupies 64 rows in DST. |

### Address Mode Configuration

The LCM operation uses `SfpuType::lcm`, which does not match any of the special-cased types in `eltwise_binary_sfpu_configure_addrmod` (those are `mul_int32`, `mul_uint16`, `max`, `min`, `max_int32`, `min_int32`, `max_uint32`, `min_uint32`). Therefore, only `ADDR_MOD_7` is configured:

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| **ADDR_MOD_7** | 0 | 0 | 0 | Default SFPU address mode with no auto-increment. DST addressing is managed manually via `dst_reg++` in the SFPU kernel and `TTI_SETRWC` in the params dispatch. |

This configuration is identical between Wormhole B0 and Blackhole. The `ADDR_MOD_6` (with `dest.incr=2`) is NOT configured for LCM since it is only used by mul/max/min integer operations.

Note: The Blackhole `_llk_math_eltwise_binary_sfpu_start_` omits the `math::set_addr_mod_base()` call and `_llk_math_eltwise_binary_sfpu_done_` omits `TTI_STALLWAIT(STALL_CFG, WAIT_SFPU)` and `math::clear_addr_mod_base()`, reflecting minor architectural differences. However, the ADDR_MOD configuration itself is the same.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary SFPU element-wise program factory work? What kernels does it use for reader, compute, and writer? How does it handle interleaved vs sharded tensors?"
   **Reason**: Needed architectural understanding of the SFPU binary program factory pattern before reading source code.
   **Key Findings**: Confirmed the three kernel files (reader_binary_interleaved_start_id, eltwise_binary_sfpu_kernel, writer_unary_interleaved_start_id), the sharded vs interleaved CB configuration strategy, and the role of defines like IN0_SHARDED/IN1_SHARDED/OUT_SHARDED.

2. [SFPU] **Query**: "How does the lcm_tile compute API function work? Trace the call chain from lcm_tile() through the LLK layer down to the ckernel SFPU implementation. What files are involved?"
   **Reason**: Needed to identify the complete file chain from API to SFPU implementation for LCM.
   **Key Findings**: Confirmed the call chain: `lcm_tile()` -> `llk_math_eltwise_binary_sfpu_lcm<APPROX>()` -> `_llk_math_eltwise_binary_sfpu_params_()` -> `calculate_sfpu_lcm()`. Identified both Wormhole B0 and Blackhole implementations are identical. GCD subroutine lives in `ckernel_sfpu_gcd.h`.

3. [SFPU] **Query**: "How does the LCM SFPU kernel work? What is the call chain from llk_math_eltwise_binary_sfpu_lcm through to the _calculate_sfpu_lcm function? What SFPU instructions and registers does it use?"
   **Reason**: Needed understanding of the LLK dispatch pattern for binary SFPU operations.
   **Key Findings**: Confirmed the `_llk_math_eltwise_binary_sfpu_params_` template orchestrates face iteration with `VectorMode::RC`, and `ADDR_MOD_7` is configured with zero increments. The `SfpuType::lcm` enum exists in `llk_sfpu_types.h`.

4. [SFPU] **Query**: "What do the SFPSETSGN, SFPSETEXP, SFPEXEXP, SFPCAST, SFPABS, SFPLZ, SFPSWAP, SFPSHFT2, SFP_STOCH_RND instructions do?"
   **Reason**: Needed detailed understanding of each SFPU instruction used in the LCM kernel to provide accurate annotations.
   **Key Findings**: Confirmed SFPCAST converts sign-magnitude int to FP32, SFPSETSGN sets sign bit, SFPSETEXP sets exponent field, SFPEXEXP extracts unbiased exponent, SFPLZ counts leading zeros with optional CC update, SFPSWAP with VEC_MIN_MAX does lanewise min/max sort, SFP_STOCH_RND with Mod1=6 converts FP32 to unsigned integer.

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

### Confluence References
None consulted for this analysis. The DeepWiki ISA documentation provided sufficient detail for all SFPU instructions used.

### Glean References
None consulted for this analysis.
