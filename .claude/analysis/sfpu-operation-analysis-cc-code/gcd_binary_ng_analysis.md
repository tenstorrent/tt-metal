# GCD (binary_ng) Implementation Analysis

## Overview

The GCD (Greatest Common Divisor) operation computes the elementwise greatest common divisor of two integer tensors: `c = gcd(a, b)`. It is implemented exclusively as an SFPU operation within the `binary_ng` program factory framework. GCD requires both inputs to be `INT32` data type.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

## Path Selection: FPU vs SFPU

The `binary_ng` program factory supports both FPU and SFPU execution paths. The path is selected via the `is_binary_sfpu_op()` function in `binary_ng_device_operation.cpp` (line 16), which examines the `BinaryOpType` and input data types.

For GCD specifically (line 42): `case GCD: return (a == INT32 && b == INT32);`. This means GCD is **always an SFPU operation** and is only valid when both inputs are INT32. If the FPU path is attempted with `BinaryOpType::GCD`, the `OpConfig` constructor will throw `"Unsupported binary op for FPU"` (line 327 of `binary_ng_utils.cpp`). The `is_sfpu` flag in `operation_attributes_t` controls kernel file selection at program creation time via `get_kernel_file_path()`.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `c.physical_volume() / tile_hw` (total output tiles) |
| **Loop structure** | Single tile per iteration; outer loop over all assigned tiles |

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|---------------|---------------|
| **Logical shape** | Arbitrary (up to rank 8+) | Arbitrary (broadcastable to A) |
| **Dimension convention** | [..., D, N, C, H, W] | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | INT32 | INT32 |

### Output Tensor

| Property | Output Tensor C |
|----------|----------------|
| **Logical shape** | Broadcast-compatible output of A and B |
| **Dimension convention** | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | INT32 |

### Layout Transformations

No tilize/untilize or format conversions are performed within this operation. The SFPU GCD operates directly on INT32 tiles in DST registers.

## Data Flow Pattern

The data flow depends on whether B is a tensor or a scalar, and on broadcast type. For the most common case (two tensor inputs, no broadcast -- `SubtileBroadcastType::NONE`):

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (A, B) | CB c_0 (A), CB c_1 (B) | reserve_back, push_back (one tile at a time for each) |
| 2 | Compute | CB c_0, CB c_1 | CB c_2 | wait_front, copy_tile to DST, gcd_tile SFPU op, pack_tile, push_back, pop_front |
| 3 | Writer | CB c_2 | DRAM (C) | wait_front, pop_front |

For scalar B case: the writer kernel fills one tile of CB c_1 with the scalar value, and only the reader reads tensor A into CB c_0.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src_a | Input A staging | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_src_b | Input B staging | 2 tiles (tensor, interleaved), 1 tile (scalar), or shard volume (sharded) | 1 tile | Double (tensor, interleaved) / Single (scalar or sharded) | Reader or Writer (scalar) | Compute | Program |
| c_2 | cb_out | Output staging | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Compute | Writer | Program |

Note: CB c_3 and c_4 are only allocated when LHS or RHS pre-activations are present. GCD has no pre/post activations by default (`process_lhs`, `process_rhs`, `postprocess` are all `std::nullopt`), so c_3 and c_4 are not used. CB c_5 and c_6 are only allocated for ROW_A/ROW_B broadcast types.

## Pipeline Pattern Summary

For the interleaved (non-sharded) case, all three main CBs (c_0, c_1, c_2) have capacity of 2 tiles with a block size of 1 tile, enabling **double-buffered** operation. This allows the reader to fill the next tile while compute processes the current one, and compute to produce the next output while writer drains the current one.

For the sharded case, CBs have capacity equal to the shard volume, and the entire shard is pushed/popped as a single unit, making it effectively **single-buffered** at the shard level.

## Index Calculations

The reader kernel computes a multi-dimensional tile offset using stride-based indexing to support broadcasting:

- `tile_offset = start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride + start_th * Wt`
- Each stride is computed as `dim_stride = (product of inner tile dimensions) * (dim > 1)`. The `(dim > 1)` factor collapses broadcast dimensions to zero stride.
- The start tile ID is decomposed into per-dimension offsets using modular arithmetic against the output shape dimensions.
- `TensorAccessor` handles the mapping from logical tile page IDs to physical DRAM bank addresses.

## Memory Access Patterns

### Read Pattern

For interleaved tensors, tiles are read one at a time in a nested loop order: nD -> D -> N -> C -> Ht -> Wt (innermost). This is effectively row-major within each 2D slice. For broadcast dimensions, the stride is zero, causing repeated reads of the same source tile.

For sharded tensors, the entire shard is made available via `cb_src.reserve_back(src_num_tiles); cb_src.push_back(src_num_tiles)` without any NoC reads -- the data is already in L1.

### Write Pattern

Same nested loop order as read. Tiles are written one at a time to DRAM via `noc.async_write`. For sharded output, no writes occur -- data stays in L1.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (prefers rectangular grid starting at (0,0)) |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `compute_with_storage_grid.x * compute_with_storage_grid.y` (zero-start grid) or `all_device_cores.num_cores()` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two-group split via `split_work_to_cores`: group 1 gets `ceil(total/cores)` tiles, group 2 gets `floor(total/cores)` tiles |

Cores not in either group receive zero-filled runtime args and exit immediately. The `zero_start_grid` optimization is used when the worker grid is a single rectangular CoreRange starting at (0,0).

## Arguments

### Compile-Time Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessor A args | uint32_t[] | Tensor accessor compile-time args for input A |
| N+1..M | TensorAccessor B args | uint32_t[] | Tensor accessor compile-time args for input B |
| M+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

**Writer kernel (tensor B case -- WriterNoBcastNg):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessor C args | uint32_t[] | Tensor accessor compile-time args for output C |
| N+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

**Compute kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 -- tiles produced per read-compute-write cycle |

### Runtime Arguments

**Reader kernel (21 args):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input A buffer address |
| 1 | start_tile_id | uint32_t | Starting output tile ID (= c_start_id) |
| 2 | src_num_tiles | uint32_t | A shard tile count (0 if interleaved) |
| 3 | dst_num_tiles | uint32_t | Number of output tiles for this core |
| 4 | dst_shard_width | uint32_t | Output shard width in tiles (0 if interleaved) |
| 5 | nD_stride | uint32_t | A's stride for collapsed dims > 5 (0 if dim=1) |
| 6 | d_stride | uint32_t | A's stride for D dimension |
| 7 | n_stride | uint32_t | A's stride for N dimension |
| 8 | c_stride | uint32_t | A's stride for C dimension |
| 9 | D | uint32_t | Output D dimension |
| 10 | N | uint32_t | Output N dimension |
| 11 | C | uint32_t | Output C dimension |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Collapsed nD dimension count |
| 15 | src_addr_b | uint32_t | Input B buffer address |
| 16 | nD_stride_b | uint32_t | B's stride for collapsed dims > 5 |
| 17 | d_stride_b | uint32_t | B's stride for D dimension |
| 18 | n_stride_b | uint32_t | B's stride for N dimension |
| 19 | c_stride_b | uint32_t | B's stride for C dimension |
| 20 | src_num_tiles_b | uint32_t | B shard tile count (0 if interleaved) |

**Writer kernel (tensor B case -- 11 args):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output C buffer address |
| 1 | start_tile_id | uint32_t | Starting output tile ID |
| 2 | dst_num_tiles | uint32_t | Number of output tiles for this core |
| 3 | dst_shard_width | uint32_t | Output shard width in tiles |
| 4 | D | uint32_t | Output D dimension |
| 5 | N | uint32_t | Output N dimension |
| 6 | C | uint32_t | Output C dimension |
| 7 | Ht | uint32_t | Output height in tiles |
| 8 | Wt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Collapsed nD dimension count |
| 10 | (unused) | uint32_t | Reserved (set to 0) |

**Compute kernel (4 args):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total tiles to process on this core |
| 1 | freq | uint32_t | Broadcast frequency (1 for NONE broadcast) |
| 2 | counter | uint32_t | Broadcast start counter (0 for NONE broadcast) |
| 3 | compute_scalar_value | uint32_t | Unused for GCD (set to 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | RISCV_0 | NOC0 | DRAM (A, B) | CB c_0, CB c_1 | Read input tiles via TensorAccessor |
| Compute | RISCV_2 | N/A | CB c_0, CB c_1 | CB c_2 | copy_tile to DST, gcd_tile SFPU op, pack_tile |
| Writer | RISCV_1 | NOC1 | CB c_2 | DRAM (C) | Write output tiles via TensorAccessor |

### Reader Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` (tensor B, no broadcast) |
| Assigned cores | All worker cores in `all_device_cores` |

**Key Logic:**
- Reads both tensor A (into CB c_0) and tensor B (into CB c_1) in the same kernel, one tile at a time
- Uses `TensorAccessor` for address resolution from logical tile page IDs to physical DRAM locations
- Computes per-dimension offsets from `start_tile_id` to handle arbitrary starting positions within the tensor
- Separate stride calculations for A and B enable broadcasting: when a dimension has size 1, the corresponding stride is 0, so the same tile is re-read
- Nested 6-deep loop: nD -> D -> N -> C -> Ht -> Wt, iterating through output tile positions
- For sharded inputs, skips NoC reads and directly marks shard tiles as available via `reserve_back`/`push_back`
- **Synchronization**: Produces tiles into CB c_0 and CB c_1 via `cb_reserve_back(1)` then `cb_push_back(1)` after NoC read barrier

### Compute Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` (no broadcast) |
| Assigned cores | All worker cores in `all_device_cores` |

**Key Logic:**
- Iterates `num_tiles` times, processing 1 tile per iteration (`num_tiles_per_cycle = 1`)
- Each iteration: waits for LHS tile in CB c_0 and RHS tile in CB c_1
- Copies LHS tile to DST register slot 0 and RHS tile to DST register slot 1 using `copy_tile()`
- `copy_tile_to_dst_init_short_with_dt()` is called to configure unpack for each source CB's data format
- Calls `BINARY_SFPU_OP(0, 1, 0)` which expands to `gcd_tile(0, 1, 0)` -- computes GCD of DST[0] and DST[1], storing result in DST[0]
- `BINARY_SFPU_INIT` expands to `gcd_tile_init()` which is called once before the loop (no pre/post activations for GCD)
- The underlying LLK call is `llk_math_eltwise_binary_sfpu_gcd<APPROX>(idst0, idst1, odst)` which runs the GCD algorithm on the SFPU vector engine
- Packs result from DST[0] to CB c_2 output via `pack_tile(0, cb_out)`
- GCD has no pre-activations or post-activations, so `PREPROCESS` macros expand to no-ops and `PROCESS_POST_ACTIVATIONS` is empty
- **Synchronization**: `cb_wait_front(cb_post_lhs, 1)` / `cb_wait_front(cb_post_rhs, 1)` to consume inputs; `cb_reserve_back(cb_out, 1)` / `cb_push_back(cb_out, 1)` to produce output; `cb_pop_front` on both inputs after processing

### Writer Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` (tensor B case) |
| Assigned cores | All worker cores in `all_device_cores` |

**Key Logic:**
- Reads computed tiles from CB c_2 and writes them to output DRAM
- Same nested 6-deep loop structure as reader, using output dimensions
- Uses `TensorAccessor` for output address resolution
- For sharded output, the entire kernel body is compiled out (`#if !DST_SHARDED`) -- output stays in L1
- For interleaved output, writes one tile at a time with `noc.async_write()` followed by `noc.async_write_barrier()`
- **Synchronization**: `cb_dst.wait_front(1)` to consume from compute; `cb_dst.pop_front(1)` after write completes

## Implementation Notes

- **Program factory variants**: Single `ProgramFactory` handles all cases. Within it, the tensor-B vs scalar-B path determines which reader/writer kernels are used. GCD is always tensor-B (both inputs must be INT32 tensors); scalar mode would use `WriterScalar` + `ComputeScalar` kernels.
- **Type-based operation variants**: GCD exclusively supports INT32 inputs and INT32 output. The `is_binary_sfpu_op` function returns true only for `(INT32, INT32)`.
- **UnpackToDestFP32 mode**: Enabled for all SFPU ops except POWER. Since GCD is not POWER, `UnpackToDestMode::UnpackToDestFp32` is set for CB c_0, c_1, c_3, c_4. This ensures INT32 data is properly unpacked into the FP32-width DEST registers.
- **Broadcast type selection**: All `SubtileBroadcastType` variants are supported (NONE, SCALAR_A/B, ROW_A/B, COL_A/B, ROW_A_COL_B, ROW_B_COL_A). Stride-based broadcasting is used -- strides of 0 for broadcast dimensions.
- **Sharding support and constraints**: Height, width, and block sharding are supported when both inputs and output share the same shape, same shard spec, and same L1 memory config. The `is_native_L1_sharding()` function enforces these constraints. Uneven shards fall back to interleaved (tensor accessor) mode.
- **FP32 dest accumulation**: `fp32_dest_acc_en` is set to true when both inputs are INT32 (which is always the case for GCD), enabling FP32-width accumulation in DEST registers.

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

1. The compute kernel calls `gcd_tile(0, 1, 0)` (defined in `api/compute/gcd.h`), which wraps the call in `MATH(...)` to ensure it runs on the math RISC-V.
2. Inside `MATH`, `llk_math_eltwise_binary_sfpu_gcd<APPROX>(idst0=0, idst1=1, odst=0)` is invoked (in `llk_math_eltwise_binary_sfpu_gcd.h`).
3. This delegates to `_llk_math_eltwise_binary_sfpu_params_<APPROX>(sfpu::calculate_sfpu_gcd, 0, 1, 0, VectorMode::RC)` (in `llk_math_eltwise_binary_sfpu_params.h`), which handles face iteration and DEST address progression.
4. For each of the 4 faces, `sfpu::calculate_sfpu_gcd(dst_index_in0=0, dst_index_in1=1, dst_index_out=0)` is called (in `ckernel_sfpu_gcd.h`), which loads data from DEST, runs the binary GCD algorithm via `calculate_sfpu_gcd_body<31>()`, and stores the result back.

Similarly, `gcd_tile_init()` calls `llk_math_eltwise_binary_sfpu_gcd_init<APPROX>()`, which calls `llk_math_eltwise_binary_sfpu_init<SfpuType::gcd, APPROX>(sfpu::calculate_sfpu_gcd_init)`. This initializes the SFPU config register, configures ADDR_MOD_7, resets counters, and then executes `calculate_sfpu_gcd_init()` to record the 7-instruction inner loop body into the REPLAY buffer.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the 32x32 tile are processed. Each face is a 16x16 block of elements.
- **Operation invocation**: The params dispatch loops 4 times (once per face). Each iteration calls `calculate_sfpu_gcd(dst_index_in0, dst_index_in1, dst_index_out)`. Inside `calculate_sfpu_gcd`, an inner loop of `ITERATIONS=8` processes 8 sub-rows per face (each sub-row is a vector of 32 elements processed by the SFPU), covering 8 x 2 = 16 rows per face via DEST address auto-increment.
- **DEST address progression**: Between faces, the params dispatch advances the DEST read/write pointer by 16 rows using two `TTI_SETRWC(CR_D, 8, SET_D)` calls (each advances by 8). Within `calculate_sfpu_gcd`, the `dst_reg++` at line 66 auto-increments the SFPU's internal DEST row pointer by 1 after each sub-row iteration (8 increments per face call). The SFPLOAD/SFPSTORE instructions use explicit absolute addressing (`dst_index * dst_tile_size`) to read from the correct input tile slots, while `dst_reg++` advances the write cursor for subsequent rows.

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions with complex CC manipulation (SFPSETCC for conditional swap, SFPLZ with CC_NE0 for conditional lane disable). Style B is used.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h
// (Blackhole version is identical)

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

#### CC State Machine -- `calculate_sfpu_gcd_body`

The `_body` function has one CC block that conditionally swaps a and b to ensure b is odd, plus the replayed inner loop that uses CC from SFPLZ to skip lanes where `a == 0` (GCD already found).

```
calculate_sfpu_gcd_body — CC State Transitions
════════════════════════════════════════════════════════════════

  CC State: ALL_ENABLED                   <-- initial state
       |
       |  SFPMOV L2 = L0                  (no CC effect) -- c = a
       |  SFPOR  L2 |= L1                 (no CC effect) -- c |= b
       |  SFPMOV L3 = L2                  (no CC effect) -- d = c
       |  SFPIADD L3 = -L3, CC_NONE      (no CC effect) -- d = -d
       |  SFPAND L3 &= L2                 (no CC effect) -- d = c & (-c), isolate LSB of (a|b)
       |  SFPLZ  L3 = clz(L3), mod1=0    (no CC effect) -- d = count of trailing zeros
       |  SFPSHFT2 L2 = L1 << L3          (no CC effect) -- c = b << d (test if b has trailing zeros)
       |
       v
  +-------------------------------------+
  | SFPSETCC  mod1=6 (LREG_EQ0)        |
  |   src: LREG2                        |
  |                                     |
  | CC <- (LREG2 == 0)                  |
  |    = (b << d == 0)                  |
  |    = (b is even, all bits shifted   |
  |       out, meaning b's trailing     |
  |       zeros >= clz(LSB(a|b)))       |
  +----------------+--------------------+
                   |
                   v
  CC State: ENABLED where b_shifted == 0 (b is even)
       |
       |  SFPSWAP L0, L1, mod1=0    (CC-guarded: swap a,b only where b is even)
       |
       v
  +-------------------------------------+
  | SFPENCC                             |
  |                                     |
  | CC <- ALL_ENABLED                   |
  +----------------+--------------------+
                   |
                   v
  CC State: ALL_ENABLED
       |
       |  SFPABS L0 = abs(L0)            (all lanes) -- a = abs(a)
       |  SFPABS L1 = abs(L1)            (all lanes) -- b = abs(b)
       |  SFPIADD L0 = -L0, CC_NONE      (no CC effect) -- a = -a (negate for subtraction trick)
       |  SFPIADD L3 = -L3, CC_NONE      (no CC effect) -- d = -d (negate for right-shift via left-shift)
       |
       v
  == Replayed inner loop (30 iterations via TTI_REPLAY) ==
  Each iteration executes the 7 instructions recorded by calculate_sfpu_gcd_init:

       |
       |  SFPABS L2 = abs(L0)            (CC-guarded) -- L2 = +a (since L0 = -a)
       |  SFPAND L0 &= L2                (CC-guarded) -- isolate LSB of a
       |
       v
  +-------------------------------------+
  | SFPLZ  L0 = clz(L0), CC_NE0        |
  |   mod1=2 (SFPLZ_MOD1_CC_NE0)       |
  |                                     |
  | CC <- (L0 != 0)                     |
  |    = (a != 0, GCD not yet found)    |
  +----------------+--------------------+
                   |
                   v
  CC State: ENABLED where a != 0
       |
       |  SFPIADD L0 += L3, CC_NONE      (CC-guarded: only a!=0 lanes) -- L0 = clz(LSB(a)) + d
       |  SFPSHFT2 L0 = L2 >> -L0        (CC-guarded: only a!=0 lanes) -- a >>= trailing zeros, a is now odd
       |  SFPSWAP L0, L1, VEC_MIN_MAX    (CC-guarded: only a!=0 lanes) -- ensure b <= a (signed min/max)
       |
       v
  +-------------------------------------+
  | SFPIADD L0 = L1 - L0, CC_NONE      |
  |   (2SCOMP_LREG_DST negates L0      |
  |    then adds L1, so L0 = b - a)    |
  |   CC_NONE: no CC update             |
  +----------------+--------------------+
                   |
                   v
  CC State: ENABLED where a != 0  (unchanged, CC_NONE)
       |
       |  (loop back for next iteration)
       |
       v
  == End of replayed loop ==

  +-------------------------------------+
  | SFPENCC                             |
  |                                     |
  | CC <- ALL_ENABLED                   |
  +----------------+--------------------+
                   |
                   v
  CC State: ALL_ENABLED
       |
       v  (function returns, result is in LREG1 = b = gcd(a,b))
```

**Key CC observations:**
- The initial SFPSETCC with mod1=6 (`SFPSETCC_MOD1_LREG_EQ0`) tests whether `b << d == 0`, which detects whether b is even after factoring out shared trailing zeros. If so, the SFPSWAP exchanges a and b so the odd value is in the b register.
- In the replayed inner loop, `SFPLZ` with `SFPLZ_MOD1_CC_NE0` (mod1=2) sets CC to disable lanes where `a == 0`. When `a == 0`, the GCD has been found (it is in b), and no further work is needed for that lane. All subsequent instructions in the iteration (SFPIADD, SFPSHFT2, SFPSWAP, SFPIADD) are CC-guarded and skip disabled lanes.
- The CC persists across replay iterations -- once a lane's `a` becomes 0, it stays disabled until `SFPENCC` at the end of `calculate_sfpu_gcd_body`.
- All SFPIADD instructions use `SFPIADD_MOD1_CC_NONE` to avoid overwriting the CC state established by SFPLZ.

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `TT_SFPLOAD` | Loads a vector of 32 elements from a DEST register row into an LREG. Used with mode `(4, 3)` for INT32 rebias-free load. |
| `TT_SFPSTORE` | Stores a vector of 32 elements from an LREG back to a DEST register row. Used with mode `(4, 3)` for INT32 rebias-free store. |
| `TTI_SFPMOV` | Copies one LREG to another. Used to initialize temporary registers (c, d). |
| `TTI_SFPOR` | Bitwise OR of two LREGs. Used to compute `a | b` for trailing zero extraction. |
| `TTI_SFPAND` | Bitwise AND of two LREGs. Used to isolate the least significant bit via `x & (-x)`. |
| `TTI_SFPIADD` | Integer add with optional 2's complement negation of destination. Used for negation (`0 + (-x)`), subtraction (`b - a`), and addition (`clz + d`). The `SFPIADD_MOD1_CC_NONE` modifier suppresses CC updates; `SFPIADD_MOD1_ARG_2SCOMP_LREG_DST` negates the destination operand before adding. |
| `TTI_SFPLZ` | Count leading zeros. Used to find the position of the least significant set bit (via `clz(x & -x)`). With `SFPLZ_MOD1_CC_NE0`, it also sets CC to disable lanes where the input is zero. |
| `TTI_SFPSHFT2` | Barrel shift with shift amount from an LREG (`SFPSHFT2_MOD1_SHFT_LREG`). Used for right-shifting `a` to remove trailing zeros (shift amount is negated for right shift). |
| `TTI_SFPSETCC` | Sets condition code based on LREG value. With mod1=6 (`SFPSETCC_MOD1_LREG_EQ0`), enables CC where the register equals zero. |
| `TTI_SFPSWAP` | Swaps two LREGs. With mod1=0, does a plain conditional swap (CC-guarded). With `SFPSWAP_MOD1_VEC_MIN_MAX` (mod1=1), swaps to put min in VD and max in VC (signed integer comparison). |
| `TTI_SFPABS` | Absolute value of an LREG. Used to ensure both a and b are positive after the initial swap. |
| `TTI_SFPENCC` | Resets condition code to ALL_ENABLED, re-enabling all SIMD lanes. |
| `TTI_REPLAY` | Replays previously recorded instructions from the replay buffer. With `(0, N, 0, 0)` it replays N instructions; with `(0, N, 0, 1)` it records the next N instructions into the buffer. |

### SFPU Register Usage

| Register | Role | Description |
|----------|------|-------------|
| **LREG0** | `a` (negated) | Holds the first operand. After initial setup, stores `-a` to facilitate the `b - a` subtraction via `SFPIADD` with `2SCOMP_LREG_DST`. |
| **LREG1** | `b` | Holds the second operand. Maintained as a positive odd number throughout the algorithm. Contains the final GCD result. |
| **LREG2** | `c` (temporary) | Used for temporary computations: `a | b` for LSB extraction, `abs(a)` for LSB isolation in the inner loop. |
| **LREG3** | `d` (shift accumulator) | Holds the negated count of shared trailing zeros between a and b (the common factor of 2). Used as an accumulated right-shift amount in the inner loop to strip trailing zeros from `a`. |
| **DEST[idst0 * 64 + row]** | Input tile A | Source data for operand `a`, loaded via `TT_SFPLOAD` with absolute addressing. |
| **DEST[idst1 * 64 + row]** | Input tile B | Source data for operand `b`, loaded via `TT_SFPLOAD` with absolute addressing. |
| **DEST[odst * 64 + row]** | Output tile | Destination for GCD result, written via `TT_SFPSTORE` from LREG1. |
| **dst_reg** | SFPU row counter | Auto-incremented by `dst_reg++` after each sub-row iteration; controls which DEST row the next SFPLOAD/SFPSTORE targets (combined with the explicit base offset). |

### Address Mode Configuration

The GCD operation uses `ADDR_MOD_7` configured during `_llk_math_eltwise_binary_sfpu_init_<SfpuType::gcd>()`:

```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

This is a zero-increment address mode -- no automatic DEST address advancement via ADDR_MOD. GCD does **not** configure `ADDR_MOD_6` (which is only set for `mul_int32`, `max`, `min`, and related ops that need `dest.incr = 2`).

DEST addressing within the SFPU kernel is managed explicitly:
- `TT_SFPLOAD`/`TT_SFPSTORE` use absolute addressing: `dst_index * 64` as the base offset, with `dst_reg` providing per-row advancement.
- `dst_reg++` in the `calculate_sfpu_gcd` loop advances the SFPU's internal DEST row pointer after each of the 8 iterations.
- Between faces, the params dispatch uses `TTI_SETRWC(CR_D, 8, SET_D)` twice (advancing by 16 rows) to move to the next face.

The configuration is identical on both Wormhole B0 and Blackhole -- the `eltwise_binary_sfpu_configure_addrmod<SfpuType::gcd>()` function only sets ADDR_MOD_7 with all-zero increments, and the `if constexpr` branch for ADDR_MOD_6 does not match `SfpuType::gcd`.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng operation work? What is the binary_ng program factory structure, how does it select between FPU and SFPU paths, and what kernels does it use?"
   **Reason**: Needed initial architectural understanding of the binary_ng operation framework before reading source code.
   **Key Findings**: The `is_sfpu` flag controls path selection. SFPU kernels are located under `kernels/compute/eltwise_binary_sfpu*.cpp`. The `OpConfig` class maps `BinaryOpType` to SFPU init/op function pairs. Reader reads both inputs when B is a tensor.

2. [SFPU] **Query**: "How is the gcd_tile SFPU operation implemented? What is the call chain from gcd_tile() through the LLK layer down to the ckernel SFPU implementation? What file contains the core SFPU GCD calculation function?"
   **Reason**: Needed to trace the full call chain from the compute API to the core SFPU kernel and identify file locations.
   **Key Findings**: `gcd_tile()` -> `llk_math_eltwise_binary_sfpu_gcd<APPROX>()` -> `_llk_math_eltwise_binary_sfpu_params_()` -> `sfpu::calculate_sfpu_gcd()`. Core implementation is in `ckernel_sfpu_gcd.h`. The `calculate_sfpu_gcd_body` function is reused by LCM.

3. [SFPU] **Query**: "How is the GCD SFPU kernel implemented in the LLK layer? What is llk_math_eltwise_binary_sfpu_gcd, and what ckernel_sfpu function does it call?"
   **Reason**: Needed LLK-level details on binary SFPU dispatch patterns and the params function.
   **Key Findings**: Confirmed the dispatch through `_llk_math_eltwise_binary_sfpu_params_` with face-iteration loop pattern. The params function takes a callable and iterates over faces based on VectorMode.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp` (lines 16-66)
   **Reason**: Needed to understand exact conditions under which GCD selects the SFPU path.
   **Key Information**: GCD returns true for SFPU only when `a == INT32 && b == INT32`.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp` (line 415)
   **Reason**: Needed to confirm the SFPU init/op function mapping for GCD.
   **Key Information**: GCD maps to `gcd_tile_init()` and `gcd_tile` as the SFPU functions.

3. **Source**: `tt_metal/hw/inc/api/compute/gcd.h`
   **Reason**: Needed to understand the LLK-level GCD implementation signature.
   **Key Information**: `gcd_tile(idst0, idst1, odst)` performs elementwise GCD via `llk_math_eltwise_binary_sfpu_gcd<APPROX>`. Both inputs must be int32. Takes three DST register indices (two inputs, one output).

4. [SFPU] **Source**: `runtime/sfpi/include/sfpi_constants.h`
   **Reason**: Needed to resolve numeric modifier values to symbolic names for SFPSETCC, SFPLZ, SFPSWAP, and SFPSHFT2.
   **Key Information**: `SFPSETCC_MOD1_LREG_EQ0 = 6`, `SFPLZ_MOD1_CC_NE0 = 2`, `SFPSWAP_MOD1_VEC_MIN_MAX = 1`, `SFPSHFT2_MOD1_SHFT_LREG = 5`.
