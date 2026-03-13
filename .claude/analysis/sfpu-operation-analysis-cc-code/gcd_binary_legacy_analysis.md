# GCD (Binary Legacy) Implementation Analysis

## Overview

The GCD (Greatest Common Divisor) operation computes the element-wise greatest common divisor of two integer tensors: `output = gcd(input_a, input_b)`. It uses the binary GCD algorithm implemented as an SFPU kernel, operating on INT32 or UINT32 tile data in the DST register file.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Path Selection: FPU vs SFPU

The `BinaryDeviceOperation::select_program_factory` method (in `binary_device_operation.cpp`) selects between `ElementWiseMultiCore` (FPU) and `ElementWiseMultiCoreSfpu` (SFPU) factories. When both input tensors have the same height and width (element-wise, no broadcasting), it calls `utils::is_binary_sfpu_op(op, dtype1, dtype2)`. For `BinaryOpType::GCD`, this function returns `true` when both inputs are either `INT32/INT32` or `UINT32/UINT32`. Since GCD is exclusively an integer operation with no FPU path, the SFPU factory is always selected. If broadcasting is needed (height_b==1 or width_b==1), the broadcast program factories are used instead, but those are out of scope for this analysis.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block of tiles |
| **Unit size** | `block_size` tiles (1 for interleaved; `find_max_block_size(num_tiles_per_shard)` for sharded) |
| **Total units** | `per_core_block_cnt` blocks per core |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks; inner loop over `per_core_block_size` tiles within each block |

## Tensor Format and Layout

| Property | Input Tensor A | Input Tensor B | Output Tensor |
|----------|---------------|---------------|---------------|
| **Dimension convention** | NHWC (arbitrary rank, flattened to tiles) | Same shape as A | Same shape as A |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 | DRAM or L1 |
| **Data type** | INT32 or UINT32 | INT32 or UINT32 | INT32 or UINT32 |

### Layout Transformations

No tilize/untilize or format conversions are performed. Both inputs and output must already be in TILE_LAYOUT. The data format for all circular buffers matches the corresponding tensor's data type directly.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src0, src1) | CB c_0 (src0), CB c_1 (src1) | reserve_back, push_back (per tile) |
| 2 | Compute | CB c_0 (input A), CB c_1 (input B) | CB c_2 (output) | wait_front, copy_tile to DST, gcd_tile SFPU op, pack_tile, pop_front, push_back |
| 3 | Writer | CB c_2 (output) | DRAM/L1 (dst) | wait_front, pop_front (per tile) |

For GCD, there is no pre-scaling stage (no `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defines), so data flows directly from CB c_0/c_1 through compute to CB c_2.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (interleaved) | Capacity (sharded) | Block Size | Buffering (interleaved) | Producer | Consumer | Lifetime |
|-------|------|---------|----------------------|-------------------|------------|----------------------|----------|----------|----------|
| c_0 | cb_src0 | Input A staging | 2 tiles | num_tiles_per_shard tiles | 1 tile | Double | Reader | Compute | Program |
| c_1 | cb_src1 | Input B staging | 2 tiles | num_tiles_per_shard tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 tiles | num_tiles_per_shard tiles | 1 tile | Double | Compute | Writer | Program |

**Notes**:
- CB c_3 and c_4 (interim buffers for pre-scaling) are NOT created for GCD since it has no `SFPU_OP_INIT_PRE_IN0_0` / `SFPU_OP_INIT_PRE_IN1_0` defines.
- For interleaved mode, capacity = `2 * max_block_size` tiles. With `max_block_size = 1` (interleaved default), this is 2 tiles, enabling double-buffering.
- For sharded inputs, the CB is backed by the globally-allocated tensor buffer and holds the entire shard.

## Pipeline Pattern Summary

- **Interleaved mode**: All three CBs (c_0, c_1, c_2) are double-buffered (capacity = 2 * block_size), allowing overlap between reader/compute and compute/writer stages.
- **Sharded mode**: Input CBs hold the entire shard at once (single logical pass); output CB similarly holds the full shard. No streaming overlap is needed since data is already in L1.

## Index Calculations

- **Interleaved path**: The reader iterates tiles linearly from `start_id` to `start_id + num_tiles`. Tile IDs are passed to `TensorAccessor` which maps logical tile indices to physical DRAM bank addresses using the interleaved page mapping.
- **Block/width-sharded path**: The reader uses a 2D loop over `block_height` x `block_width` tiles. The start tile ID for each core is computed as: `(core_index / num_shards_per_width) * (block_height * block_width * num_shards_per_width) + (core_index % num_shards_per_width) * block_width`. Row strides use `num_cores_y * block_width`.
- **Compute kernel**: Within each block, tiles are copied to DST at positions `i*2` (input A) and `i*2+1` (input B). The GCD result overwrites position `i*2`, which is then packed to the output CB.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads from DRAM via NoC0, one tile at a time with `noc_async_read_tile`. Each tile read is followed by `noc_async_read_barrier` before pushing to the CB.
- **Sharded**: No DRAM reads; the CB is directly backed by the L1 shard buffer. Reader simply does `cb_reserve_back` / `cb_push_back` to make tiles available.

### Write Pattern
- **Interleaved**: Sequential tile writes to DRAM via NoC1, one tile at a time with `noc_async_write_page`. Uses `noc_async_writes_flushed` between tiles and a final `noc_async_write_barrier`.
- **Sharded**: No DRAM writes; the output CB is backed by L1. Writer simply does `cb_wait_front` on all output tiles.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (row-major linearization of available cores) or shard-grid for sharded |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `num_cores_total` from worker grid |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two-group split: group 1 gets `ceil(total_tiles / num_cores)` tiles, group 2 gets `floor(total_tiles / num_cores)` tiles. Uses `split_work_to_cores`. |

For sharded tensors, each core processes exactly its shard's tiles (`shard_shape[0] * shard_shape[1] / TILE_HW`), and the core grid comes from the shard spec.

## Arguments

### Compile-Time Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_or_width_sharded | uint32_t | 1 if block/width sharded, 0 otherwise |
| 1+ | TensorAccessorArgs(src0) | varies | Accessor params for input A (omitted if IN0_SHARDED) |
| N+ | TensorAccessorArgs(src1) | varies | Accessor params for input B (omitted if IN1_SHARDED) |

**Writer kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs(dst) | varies | Accessor params for output buffer |

**Compute kernel**: No compile-time args. Behavior is controlled entirely through preprocessor defines:
- `BINOP_INIT` = `gcd_tile_init();`
- `BINARY_SFPU_OP` = `gcd_tile(i*2, i*2+1, i*2);`

### Runtime Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | Input A buffer address |
| 1 | src1_addr | uint32_t | Input B buffer address |
| 2 | num_tiles | uint32_t | Total tiles for this core |
| 3 | start_id | uint32_t | Starting tile ID |
| 4 | block_height | uint32_t | Shard block height in tiles (0 if interleaved) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 if interleaved) |
| 6 | num_cores_y | uint32_t | Number of shards per width dimension |

**Compute kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_size | uint32_t | Tiles per block |

**Writer kernel (interleaved/height-sharded):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer address |
| 1 | num_pages | uint32_t | Total tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for output |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 src0, src1 | CB c_0, CB c_1 | Read tiles from both inputs |
| Compute | TRISC (math + pack) | N/A | CB c_0, CB c_1 | CB c_2 | copy_tile to DST, gcd_tile SFPU, pack_tile |
| Writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 dst | Write output tiles |

### Reader Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` |
| **Assigned cores** | All worker cores in `all_device_cores` |

**Key Logic**:
- For sharded inputs (`IN0_SHARDED` / `IN1_SHARDED`): simply calls `cb_reserve_back` + `cb_push_back` to expose the pre-loaded L1 shard data to compute.
- For interleaved inputs with `block_or_width_sharded` mode: uses a 2D loop (`block_height` x `block_width`) reading tiles with strided row access pattern (`row_start_tile_id += num_cores_y * block_width`).
- For fully interleaved inputs: simple linear loop from `start_id` to `start_id + num_tiles`, reading one tile at a time.
- Each tile read uses `noc_async_read_tile` with a `TensorAccessor` for address translation, followed by `noc_async_read_barrier`.
- **Synchronization**: Produces tiles into CB c_0 and CB c_1 via `cb_reserve_back(1)` / `cb_push_back(1)` per tile.

### Compute Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` |
| **Assigned cores** | All worker cores in `all_device_cores` |

**Key Logic**:
- Outer loop iterates `per_core_block_cnt` blocks.
- No pre-scaling stage for GCD (no `SFPU_OP_INIT_PRE_IN0_0` / `SFPU_OP_INIT_PRE_IN1_0`).
- Waits for `per_core_block_size` tiles on both CB c_0 and CB c_1, reserves space on CB c_2.
- Acquires tile registers, then copies input A tiles to DST[i*2] and input B tiles to DST[i*2+1] using `copy_tile` with proper data type initialization via `copy_tile_to_dst_init_short_with_dt`.
- For each tile pair, executes `BINOP_INIT` (`gcd_tile_init()`) followed by `BINARY_SFPU_OP` (`gcd_tile(i*2, i*2+1, i*2)`).
- `gcd_tile_init()` records a 28-instruction replay buffer implementing 4 iterations of the binary GCD inner loop.
- `gcd_tile()` loads tile rows from DST into SFPU LREGs, runs `calculate_sfpu_gcd_body<31>()` which performs 30 iterations of the binary GCD algorithm using TTI_REPLAY to replay the recorded loop body, then stores the result back to DST.
- The binary GCD algorithm: takes absolute values, ensures b is odd by removing trailing zeros (using `SFPLZ` + `SFPSHFT2`), then iteratively computes `a = b - a` keeping a even and swapping to maintain `b < a`, converging to `gcd` in register LREG1.
- Packs result from DST[i*2] to CB c_2 via `pack_tile`.
- **Synchronization**: Consumes from CB c_0 and CB c_1 via `cb_wait_front` / `cb_pop_front`. Produces to CB c_2 via `cb_reserve_back` / `cb_push_back`.

### Writer Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (standard interleaved case) |
| **Assigned cores** | All worker cores in `all_device_cores` |

**Key Logic**:
- For sharded output (`OUT_SHARDED`): simply calls `cb_wait_front(cb_id_out, num_pages)` -- output data is already in L1 via the globally-allocated CB buffer.
- For interleaved output: iterates from `start_id` to `start_id + num_pages`, writing one tile at a time using `noc_async_write_page` with `TensorAccessor` for address mapping.
- Uses `noc_async_writes_flushed` after each tile write for flow control, and `noc_async_write_barrier` at the end.
- **Synchronization**: Consumes from CB c_2 via `cb_wait_front(1)` / `cb_pop_front(1)` per tile.

## Implementation Notes

- **Program factory variants**: Only the `ElementWiseMultiCoreSfpu` factory is used for GCD. The `ElementWiseMultiCore` (FPU) factory is never selected since `is_binary_sfpu_op` always returns `true` for GCD with valid dtypes. A separate writer kernel (`writer_unary_sharded_blocks_interleaved_start_id.cpp`) is selected when the input is block/width-sharded but the output is interleaved.
- **Type-based operation variants**: GCD supports only INT32/INT32 and UINT32/UINT32 input pairs. No floating-point path exists. The defines are identical for both integer types (`gcd_tile_init` / `gcd_tile`).
- **UnpackToDestFP32 mode**: Enabled for all input CBs (c_0, c_1, c_3, c_4) since `op_type != BinaryOpType::POWER`. This ensures 32-bit integer data is unpacked to DEST at full precision.
- **Broadcast type selection**: N/A. GCD via this factory requires both inputs to have identical height and width. Broadcasting would route through different program factories.
- **Sharding support and constraints**: Supports height-sharded, width-sharded, and block-sharded memory layouts. Any combination of sharded/interleaved inputs and output is supported. For block/width-sharded input with interleaved output, a specialized writer kernel handles the reshuffling.
- **FP32 dest accumulation**: Enabled when output dtype is Float32, Int32, or UInt32. For GCD (always integer), this is always enabled, ensuring the 32-bit integer DST accumulator retains full precision.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/gcd.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_gcd.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `gcd_tile_init()` (from `api/compute/gcd.h`), which dispatches to `llk_math_eltwise_binary_sfpu_gcd_init<APPROX>()` on the MATH engine. This calls `_llk_math_eltwise_binary_sfpu_init_<SfpuType::gcd>()` to configure SFPU state and ADDR_MOD_7, then calls `calculate_sfpu_gcd_init()` to record the 7-instruction loop body into a 28-instruction replay buffer (4 unrolled iterations).

2. The compute kernel then calls `gcd_tile(i*2, i*2+1, i*2)` (from `api/compute/gcd.h`), which dispatches to `llk_math_eltwise_binary_sfpu_gcd<APPROX>(idst0, idst1, odst, VectorMode::RC)` on the MATH engine.

3. This calls `_llk_math_eltwise_binary_sfpu_params_<APPROX>(sfpu::calculate_sfpu_gcd, dst_index0, dst_index1, odst, VectorMode::RC)`, which starts the SFPU, loops over 4 tile faces calling `calculate_sfpu_gcd()` per face, advancing the DEST read/write pointer by 16 rows (2x `SETRWC +8`) between faces, then stops the SFPU.

4. `calculate_sfpu_gcd()` iterates 8 times (one per row-group within a face), loading 32 lanes from two DST tiles via `SFPLOAD`, executing the binary GCD algorithm via `calculate_sfpu_gcd_body<31>()`, storing the result via `SFPSTORE`, and incrementing `dst_reg` to advance to the next row.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the 32x32 tile are processed. The params dispatch loops `face = 0..3`, calling the SFPU function once per face.
- **Operation invocation**: For each face, `calculate_sfpu_gcd(dst_index_in0, dst_index_in1, dst_index_out)` is called with its default `ITERATIONS=8`. This processes all 8 row-groups (4 rows each, 32 lanes wide) within the face.
- **DEST address progression**: Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice, advancing the DEST base address by 16 rows total. Within `calculate_sfpu_gcd`, `dst_reg++` advances the SFPU's internal row pointer after each of the 8 iterations. The SFPLOAD/SFPSTORE instructions use a base address of `dst_index * 64` (64 rows per tile in DEST), and the `dst_reg` counter auto-increments to process successive 4-row groups.

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions with CC manipulation via `SFPSETCC`, `SFPENCC`, and `SFPLZ` (with `SFPLZ_MOD1_CC_NE0`). It falls under **Style B** due to the complex CC usage in the init (replay) loop body combined with `SFPSETCC`/`SFPENCC` in the main body.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h
// (Blackhole implementation is identical)

template <int max_input_bits = 31>
inline void calculate_sfpu_gcd_body() {
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0); // c = a
    TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG2, 0); // c |= b

    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG3, 0); // d = c
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // d = -d
    TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG3, 0); // d &= c (isolate LSB)
    TTI_SFPLZ(0, p_sfpu::LREG3, p_sfpu::LREG3, 0); // d = clz(d)

    // Ensure that b is odd: if LSB is zero, then swap with a.
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG2, SFPSHFT2_MOD1_SHFT_LREG); // c = b << d
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, 6); // if c == 0 then b is even
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 0); // swap(a, b)
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0); // a = abs(a)
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, 0); // b = abs(b)

    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // a = -a
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // d = -d

    int iterations = max_input_bits - 1;

    #pragma GCC unroll 7
    while (iterations / 4 > 0) {
        TTI_REPLAY(0, 7 * 4, 0, 0);
        iterations -= 4;
    }

    // Replay 2 more iterations, making a total of 30 iterations.
    // The worst case for 31-bit inputs is 31 iterations, but we can skip the final iteration as it only affects a.
    // In addition, we can skip the final operation of the 30th iteration as it only affects a.
    TTI_REPLAY(0, 7 * iterations - 1, 0, 0);

    TTI_SFPENCC(0, 0, 0, 0);
}

template <int ITERATIONS = 8>
inline void calculate_sfpu_gcd(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Binary GCD algorithm.
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64 rows
        constexpr uint dst_tile_size = 64;

        TT_SFPLOAD(p_sfpu::LREG0, 4, 3, dst_index_in0 * dst_tile_size);  // a
        TT_SFPLOAD(p_sfpu::LREG1, 4, 3, dst_index_in1 * dst_tile_size);  // b

        calculate_sfpu_gcd_body<31>();

        TT_SFPSTORE(p_sfpu::LREG1, 4, 3, dst_index_out * dst_tile_size);
        dst_reg++;
    }
}

inline void calculate_sfpu_gcd_init() {
    TTI_REPLAY(0, 7 * 4, 0, 1);
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

#### CC State Machine: `calculate_sfpu_gcd_body`

The body function has two distinct CC regions: (1) the setup phase with SFPSETCC/SFPENCC around a conditional swap, and (2) the replay loop body (recorded in `calculate_sfpu_gcd_init`) which uses SFPLZ with CC_NE0 to disable lanes where `a == 0` (GCD already found).

```
calculate_sfpu_gcd_body -- CC State Transitions
================================================================

  CC State: ALL_ENABLED                   <-- initial state
       |
       |  SFPMOV  L2 = L0                 (no CC effect)
       |  SFPOR   L2 |= L1               (no CC effect)
       |  SFPMOV  L3 = L2                 (no CC effect)
       |  SFPIADD L3 = -L3  CC_NONE      (no CC effect, CC_NONE opt-out)
       |  SFPAND  L3 &= L2               (no CC effect)
       |  SFPLZ   L3 = clz(L3)  mod1=0   (no CC effect, SFPLZ_MOD1_CC_NONE)
       |  SFPSHFT2 L2 = L1 << L3         (no CC effect)
       |
       v
  +-------------------------------------------+
  | SFPSETCC  L2, mod1=6 (LREG_EQ0)          |
  |                                           |
  | CC <- (LREG2 == 0)                        |
  | i.e. enabled where b shifted left by      |
  | clz(LSB(a|b)) equals zero, meaning b is   |
  | even (its trailing zeros >= clz count)     |
  +-------------------+-----------------------+
                      |
                      v
  CC State: ENABLED where LREG2 == 0 (b is even)
       |
       |  SFPSWAP L0, L1  mod1=0          (CC-guarded: only lanes where b is even get swapped)
       |
       v
  +-------------------------------------------+
  | SFPENCC                                   |
  |                                           |
  | CC <- ALL_ENABLED                         |
  +-------------------+-----------------------+
                      |
                      v
  CC State: ALL_ENABLED
       |
       |  SFPABS  L0 = abs(L0)            (all lanes)
       |  SFPABS  L1 = abs(L1)            (all lanes)
       |  SFPIADD L0 = -L0  CC_NONE       (no CC effect, CC_NONE opt-out)
       |  SFPIADD L3 = -L3  CC_NONE       (no CC effect, CC_NONE opt-out)
       |
       v

  == Replay Loop (30 iterations via TTI_REPLAY) ==
  The replayed instructions come from calculate_sfpu_gcd_init:

  Per iteration (7 instructions):
  CC State: ALL_ENABLED at start of first iteration
  (Note: SFPLZ with SFPLZ_MOD1_CC_NE0 sets CC)
       |
       |  SFPABS  L2 = abs(L0)            (all lanes, since -a -> +a)
       |  SFPAND  L0 &= L2               (all lanes, isolates LSB of a)
       |
       v
  +-------------------------------------------+
  | SFPLZ  L0 = clz(L0), SFPLZ_MOD1_CC_NE0  |
  |                                           |
  | CC <- (L0 != 0 before CLZ)               |
  | Disables lanes where a == 0 (GCD found)  |
  +-------------------+-----------------------+
                      |
                      v
  CC State: ENABLED where a != 0
       |
       |  SFPIADD L0 += L3  CC_NONE       (CC-guarded: only a!=0 lanes; CC_NONE prevents CC update)
       |  SFPSHFT2 L0 = L2 >> -L0         (CC-guarded: strips trailing zeros from a)
       |  SFPSWAP L0, L1  MIN_MAX         (CC-guarded: ensures b <= a as signed integers)
       |  SFPIADD L0 = L1 - L0  CC_NONE   (CC-guarded: a = b - a, making a even again)
       |
       v
  (next iteration starts; CC is NOT reset between iterations --
   lanes where a became 0 stay disabled for all subsequent iterations)

  == End of replay loop ==

  +-------------------------------------------+
  | SFPENCC                                   |
  |                                           |
  | CC <- ALL_ENABLED                         |
  +-------------------+-----------------------+
                      |
                      v
  CC State: ALL_ENABLED
       |
       v
  (function returns; result is in LREG1)
```

**Key CC insight**: The `SFPLZ_MOD1_CC_NE0` in the replay loop progressively disables lanes as they converge to GCD (when `a` becomes 0, that lane's GCD is in `b`/LREG1). Since `SFPENCC` is NOT called between replay iterations, once a lane is disabled it stays disabled for all remaining iterations. This is an optimization -- converged lanes skip unnecessary work. The final `SFPENCC` at the end of `calculate_sfpu_gcd_body` re-enables all lanes before returning.

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `SFPLOAD` (`TT_SFPLOAD`) | Loads 32 datums from DEST register file into an LREG. Mod0=4 (`SFPLOAD_MOD0_FMT_BOB32`, "Bag Of Bits" 32-bit) loads raw 32-bit values without format conversion, ignoring lane-enable. AddrMod=3 selects `ADDR_MOD_3` for DEST address progression. |
| `SFPSTORE` (`TT_SFPSTORE`) | Stores 32 datums from an LREG back to DEST register file. Mod0=4 (`SFPSTORE_MOD0_FMT_BOB32`, "Bag Of Bits" 32-bit) stores raw 32-bit values without format conversion, ignoring lane-enable. AddrMod=3 selects `ADDR_MOD_3`. |
| `SFPMOV` (`TTI_SFPMOV`) | Copies one LREG to another (register-to-register move). No CC effect. |
| `SFPOR` (`TTI_SFPOR`) | Bitwise OR of two LREGs. Used to compute `a \| b` for LSB isolation. No CC effect. |
| `SFPAND` (`TTI_SFPAND`) | Bitwise AND of two LREGs. Used to isolate the least significant set bit via `x & (-x)`. No CC effect. |
| `SFPABS` (`TTI_SFPABS`) | Integer absolute value (mod1=0 defaults to integer mode). Converts negative two's complement to positive. No CC effect. |
| `SFPLZ` (`TTI_SFPLZ`) | Count leading zeros. With mod1=0 (`SFPLZ_MOD1_CC_NONE`): pure CLZ, no CC effect. With mod1=2 (`SFPLZ_MOD1_CC_NE0`): CLZ + sets CC to enabled where input != 0 (before CLZ operation). |
| `SFPIADD` (`TTI_SFPIADD`) | Integer add/subtract. With `SFPIADD_MOD1_CC_NONE` (mod1 bit 2 set): no CC update. With `SFPIADD_MOD1_ARG_2SCOMP_LREG_DST` (mod1 bit 1 set): computes `VC + (-VD)` effectively doing `VC - VD` or negation when VC=0. These modifier bits combine via OR. |
| `SFPSHFT2` (`TTI_SFPSHFT2`) | Bitwise shift by register amount. `SFPSHFT2_MOD1_SHFT_LREG` (mod1=5): shifts VB by VC amount (left if positive, right if negative). Used to strip trailing zeros. No CC effect. |
| `SFPSWAP` (`TTI_SFPSWAP`) | With mod1=0 (`SFPSWAP_MOD1_SWAP`): unconditional register swap (respects CC gating). With mod1=1 (`SFPSWAP_MOD1_VEC_MIN_MAX`): sets VD=min(VC,VD), VC=max(VC,VD) as signed integers. No CC effect. |
| `SFPSETCC` (`TTI_SFPSETCC`) | Sets condition code. Mode 6 (`SFPSETCC_MOD1_LREG_EQ0`): enables lanes where the source LREG equals zero. |
| `SFPENCC` (`TTI_SFPENCC`) | Resets condition code to ALL_ENABLED. Re-enables all lanes for subsequent operations. |
| `REPLAY` (`TTI_REPLAY`) | Replays previously recorded instructions from the replay buffer. With last arg=1: starts recording mode (subsequent instructions are captured into the buffer). With last arg=0: replays `count` instructions from the buffer. Second arg specifies the instruction count. |

### SFPU Register Usage

| Register | Role | Description |
|----------|------|-------------|
| **LREG0** | `a` (negated) | Holds the first operand, stored as `-a` (negated) during the iterative loop. In the init replay body, LREG0 alternates between holding `-a`, then `abs(a)` (via SFPABS), then `clz(LSB)`, then the shift amount, then the odd `a` value, and finally `b - a` (the new even `a`). |
| **LREG1** | `b` (GCD result) | Holds the second operand `b`. After SFPSWAP with MIN_MAX, LREG1 always holds the smaller value. When the algorithm converges (`a == 0`), LREG1 contains the GCD. This register is stored back to DEST as the output. |
| **LREG2** | Temporary `c` | Used as scratch for intermediate values: `a \| b`, shifted `b`, absolute value of `a`. Reused across phases. |
| **LREG3** | Temporary `d` (negate of CLZ) | Holds the count of trailing zeros (as a negated CLZ value) used for right-shifting. In the body setup, computed as `clz(LSB(a\|b))`; in the replay loop, accumulated shift count for stripping trailing zeros from `a`. |
| **LCONST_0** | Zero constant | Hardware constant register holding 0. Used with `SFPIADD` + `2SCOMP_LREG_DST` to negate a register (0 - VD). |
| **dst_reg** | DEST row counter | SFPU internal counter that auto-increments to walk through the 8 row-groups (4 rows each) within a tile face. Incremented via `dst_reg++` after each iteration of the outer loop in `calculate_sfpu_gcd`. |

### Address Mode Configuration

The SFPU init function `_llk_math_eltwise_binary_sfpu_init_<SfpuType::gcd>()` calls `eltwise_binary_sfpu_configure_addrmod<SfpuType::gcd>()`, which configures:

**ADDR_MOD_7** (used for all SFPU operations in this kernel):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
```

All increments are zero -- the DEST address does not auto-increment via ADDR_MOD between SFPU instructions. Instead, DEST progression is handled explicitly:
- **Between row-groups within a face**: `dst_reg++` in the `calculate_sfpu_gcd` loop advances the SFPU's internal row pointer.
- **Between faces**: `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` called twice (advancing by 16 rows total) in the params dispatch loop.

GCD does NOT use ADDR_MOD_6 (the conditional branch for `SfpuType::mul_int32`, `max`, `min`, etc. does not match `SfpuType::gcd`).

The SFPLOAD/SFPSTORE instructions reference `AddrMod=3` (ADDR_MOD_3). This address mode is not explicitly configured by the GCD init function -- it uses whatever default or prior configuration exists for ADDR_MOD_3 from the `A2D` (unpack-to-DEST) pipeline. The key point is that `dst_reg++` is what actually advances the DEST row pointer between iterations, not the ADDR_MOD auto-increment.

This configuration is identical for Wormhole and Blackhole -- the `eltwise_binary_sfpu_configure_addrmod` function and the entire `ckernel_sfpu_gcd.h` implementation are the same across both architectures.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary element-wise SFPU program factory work in ttnn? What kernels does it use and how does it handle broadcasting?"
   **Reason**: Needed architectural context for the legacy binary SFPU path before reading source code.
   **Key Findings**: Confirmed the three kernels used (reader_binary_interleaved_start_id, eltwise_binary_sfpu_kernel, writer_unary_interleaved_start_id), the factory selection logic via `is_binary_sfpu_op`, and that broadcasting routes to different factories.

2. [SFPU] **Query**: "How does the gcd_tile function work in the LLK layer? What is the call chain from gcd_tile through llk_math to the ckernel SFPU implementation?"
   **Reason**: Needed to trace the full call chain from the tile-level API through LLK dispatch to the core SFPU kernel.
   **Key Findings**: DeepWiki did not have GCD-specific documentation but confirmed the general binary SFPU pattern: tile-level API -> llk_math dispatch -> `_llk_math_eltwise_binary_sfpu_params_` -> ckernel SFPU function. The binary SFPU params template handles face iteration and SETRWC advancement.

3. [SFPU] **Query**: "What do the SFPU instructions SFPABS, SFPAND, SFPLZ, SFPSHFT2, SFPSWAP, SFPIADD, SFPLOAD, SFPSTORE, SFPSETCC, SFPENCC do? What are their operand formats and how do they interact with condition codes?"
   **Reason**: Needed detailed semantics of every SFPU instruction used in the GCD kernel to accurately document CC state transitions and register manipulation.
   **Key Findings**: Confirmed SFPABS does integer absolute value; SFPAND is bitwise AND; SFPLZ counts leading zeros with optional CC setting; SFPSHFT2 does register-amount bitwise shifts; SFPSWAP can do min/max or unconditional swap; SFPIADD does integer add/subtract with optional CC update and 2's complement modes; SFPSETCC sets CC based on register comparison; SFPENCC resets CC to all-enabled. SFPLOAD/SFPSTORE move data between DEST and LREGs with format conversion.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 47-52, 358-361, 535)
   **Reason**: To verify how GCD defines are generated and what preprocessor macros control the compute kernel.
   **Key Information**: GCD sets `BINOP_INIT = gcd_tile_init()` and `BINARY_SFPU_OP = gcd_tile(i*2, i*2+1, i*2)`. No pre-scaling defines are set.

2. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h`
   **Reason**: To understand the SFPU-level GCD algorithm implementation.
   **Key Information**: Implements the binary GCD algorithm using SFPU instructions (SFPABS, SFPAND, SFPLZ, SFPSHFT2, SFPSWAP, SFPIADD). Uses TTI_REPLAY for loop unrolling -- init records 4 iterations into replay buffer, then body replays 7x4 + 7x2-1 = 41 instruction groups for 30 total iterations. Handles 31-bit signed integers. Loads/stores tile rows via SFPLOAD/SFPSTORE with BOB32 format (Mod0=4, "Bag Of Bits" raw 32-bit).

3. [SFPU] **Source**: `runtime/sfpi/include/sfpi_constants.h`
   **Reason**: To verify numeric values of SFPU instruction modifier constants and SFPLOAD/SFPSTORE format modes.
   **Key Information**: `SFPSETCC_MOD1_LREG_EQ0 = 6`, `SFPLZ_MOD1_CC_NE0 = 2`, `SFPSWAP_MOD1_VEC_MIN_MAX = 1`, `SFPSHFT2_MOD1_SHFT_LREG = 5`, `SFPLOAD_MOD0_FMT_BOB32 = 4` (Bag Of Bits 32-bit raw load, no format conversion), `SFPSTORE_MOD0_FMT_BOB32 = 4`.

4. [SFPU] **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu.h`
   **Reason**: To understand the SFPU init, start, done functions and ADDR_MOD configuration.
   **Key Information**: `eltwise_binary_sfpu_configure_addrmod` sets ADDR_MOD_7 with all-zero increments for GCD. The start function sets the DEST write address and stalls waiting for SFPU. The done function clears the DEST address and waits for SFPU completion.

5. [SFPU] **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h`
   **Reason**: To understand the face iteration and DEST address progression in the binary SFPU params dispatch.
   **Key Information**: For VectorMode::RC, iterates 4 faces, calling the SFPU function per face with 2x `TTI_SETRWC(CR_D, +8)` between faces (16-row advancement). Blackhole params dispatch is identical.
