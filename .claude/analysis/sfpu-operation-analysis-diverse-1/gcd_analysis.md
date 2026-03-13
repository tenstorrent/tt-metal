# GCD (Greatest Common Divisor) Implementation Analysis

## Overview

The GCD operation computes the elementwise greatest common divisor of two integer tensors: `c = gcd(a, b)`. It is an SFPU-only binary operation implemented within the `binary_ng` (next-generation binary) framework. The operation requires both inputs to be `INT32` and uses a hardware-optimized binary GCD algorithm executed on the SFPU vector unit.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

## Path Selection: FPU vs SFPU

The `binary_ng` framework uses a single program factory (`BinaryNgDeviceOperation::ProgramFactory::create`) for all binary operations. The FPU vs SFPU path is selected based on the `operation_attributes.is_sfpu` flag, which is determined at operation invocation time by `utils::is_binary_sfpu_op()` in `binary_ng_device_operation.cpp`.

For GCD specifically, `is_binary_sfpu_op` returns `true` if and only if `a == INT32 && b == INT32` (line 42 of the device operation file). GCD has no FPU path -- attempting to use it with the FPU path throws an exception (`"Unsupported binary op for FPU"` at line 327 of `binary_ng_utils.cpp`). Therefore, GCD is unconditionally an SFPU operation, and it is the only path analyzed below.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `c.physical_volume() / (tile_height * tile_width)` -- total output tiles |
| **Loop structure** | Single tile per iteration: reader reads 1 tile of A and 1 tile of B, compute processes 1 tile, writer writes 1 tile of C |

## Tensor Format and Layout

| Property | Input Tensor A | Input Tensor B | Output Tensor C |
|----------|---------------|---------------|-----------------|
| **Logical shape** | Arbitrary (up to rank 8, collapsed to 5D internally) | Arbitrary (broadcast-compatible with A) | Broadcast-compatible output shape |
| **Dimension convention** | [..., D, N, C, H, W] (last 5 dims) | [..., D, N, C, H, W] | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 | DRAM or L1 |
| **Data type** | INT32 (required) | INT32 (required) | INT32 |

### Layout Transformations

No tilize/untilize or format conversions are performed within the operation itself. Both inputs and the output must already be in `TILE_LAYOUT`. Since GCD is INT32-only, no type casting post-activation is applied (the `binary::utils::is_typecast` check would be false for INT32->INT32).

## Data Flow Pattern

For the interleaved (non-sharded) no-broadcast path:

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (tensor A via NoC) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `noc_async_read_barrier`, `cb_push_back(c_0, 1)` |
| 1 | Reader | DRAM (tensor B via NoC) | CB c_1 | `cb_reserve_back(c_1, 1)`, `noc_async_read_page`, `noc_async_read_barrier`, `cb_push_back(c_1, 1)` |
| 2 | Compute | CB c_0, CB c_1 | CB c_2 | `cb_wait_front(c_0, 1)`, `cb_wait_front(c_1, 1)`, `cb_reserve_back(c_2, 1)`, copy tiles to DST, `gcd_tile`, pack, `cb_push_back(c_2, 1)`, `cb_pop_front(c_0, 1)`, `cb_pop_front(c_1, 1)` |
| 3 | Writer | CB c_2 | DRAM (tensor C via NoC) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `noc_async_write_barrier`, `cb_pop_front(c_2, 1)` |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src_a | Input A staging | 2 tiles (interleaved) or shard_volume tiles (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_src_b | Input B staging | 2 tiles (interleaved) or shard_volume tiles (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_2 | cb_out | Output staging | 2 tiles (interleaved) or shard_volume tiles (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Compute | Writer | Program |

Notes:
- CBs c_3 and c_4 (LHS/RHS intermediate activation buffers) are NOT created for GCD because GCD has no `process_lhs` or `process_rhs` activations -- the `PROCESS_LHS_ACTIVATIONS(i)` and `PROCESS_RHS_ACTIVATIONS(i)` defines are empty.
- CBs c_5 and c_6 (row broadcast buffers) are NOT created because GCD with two INT32 tensors uses `SubtileBroadcastType::NONE` in the typical case.
- All CB data formats are `Int32` (since both inputs and output are INT32).

## Pipeline Pattern Summary

For the interleaved path, all three main CBs (c_0, c_1, c_2) have capacity of 2 tiles with a block size of 1 tile, providing **double-buffering**. This allows the reader to fill the next tile while compute processes the current tile, and compute to produce the next output tile while the writer drains the current one.

## Index Calculations

The reader kernel uses a 6-level nested loop structure to map logical tensor coordinates to physical tile indices. The output tensor shape is decomposed into `[cND, D, N, C, Ht, Wt]` dimensions (where `cND` collapses all dimensions beyond rank 5).

For each input tensor, stride values control how logical output tile coordinates map to input tile offsets. Broadcasting is handled through stride multipliers: if a dimension has size 1 in an input, its stride is set to 0 (via the `(dim > 1)` multiplier in runtime args), causing all output tiles along that dimension to read the same input tile.

The writer kernel uses a simpler tile offset that grows linearly (`dst_tile_offset + num_tiles_written`), since the output shape is the full broadcasted shape.

Both kernels use `TensorAccessor` for physical memory address resolution -- this maps logical tile indices to bank-interleaved physical addresses.

## Memory Access Patterns

### Read Pattern
- **Ordering**: Nested loop: ND -> D -> N -> C -> H -> W (row-major within tiles)
- **Access type**: One tile at a time via `noc_async_read_page`
- **Barrier**: `noc_async_read_barrier()` after each pair of A+B tile reads
- **Pattern**: Sequential with potential stride gaps due to broadcasting

### Write Pattern
- **Ordering**: Linear sequential tile output (`dst_tile_offset + num_tiles_written`)
- **Access type**: One tile at a time via `noc_async_write_page`
- **Barrier**: `noc_async_write_barrier()` after each tile write

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (when zero_start_grid) or arbitrary CoreRangeSet |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `compute_with_storage_grid.x * compute_with_storage_grid.y` or `all_device_cores.num_cores()` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two-group split: `core_group_1` gets `ceil(total_tiles / num_cores)` tiles, `core_group_2` gets `floor(total_tiles / num_cores)` tiles. Idle cores receive zeroed-out runtime args. |

Work splitting uses `tt::tt_metal::split_work_to_cores()` which divides total output tiles across available cores. Cores not assigned any tiles receive zero-length runtime arguments and exit immediately.

## Arguments

### Compile-Time Arguments

**Compute Kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 -- tiles produced per read-compute-write cycle |

**Reader Kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessor args (A) | uint32_t | Compile-time portion of TensorAccessorArgs for input A |
| N+1..M | TensorAccessor args (B) | uint32_t | Compile-time portion of TensorAccessorArgs for input B |
| M+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

**Writer Kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessor args (C) | uint32_t | Compile-time portion of TensorAccessorArgs for output C |
| N+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

### Runtime Arguments

**Reader Kernel (21 args per core):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Base address of input tensor A buffer |
| 1 | start_tile_id | uint32_t | Starting output tile index for this core (c_start_id) |
| 2 | src_num_tiles | uint32_t | Number of A tiles in shard (0 if interleaved) |
| 3 | dst_num_tiles | uint32_t | Number of output tiles assigned to this core |
| 4 | dst_shard_width | uint32_t | Shard width in tiles (0 if interleaved) |
| 5 | nD_stride | uint32_t | Stride for collapsed ND dimensions of A |
| 6 | d_stride | uint32_t | Stride for D dimension of A |
| 7 | n_stride | uint32_t | Stride for N dimension of A |
| 8 | c_stride | uint32_t | Stride for C dimension of A |
| 9 | D | uint32_t | Output D dimension size |
| 10 | N | uint32_t | Output N dimension size |
| 11 | C | uint32_t | Output C dimension size |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Collapsed ND dimension count |
| 15 | src_addr_b | uint32_t | Base address of input tensor B buffer |
| 16 | nD_stride_b | uint32_t | Stride for collapsed ND dimensions of B |
| 17 | d_stride_b | uint32_t | Stride for D dimension of B |
| 18 | n_stride_b | uint32_t | Stride for N dimension of B |
| 19 | c_stride_b | uint32_t | Stride for C dimension of B |
| 20 | src_num_tiles_b | uint32_t | Number of B tiles in shard (0 if interleaved) |

**Compute Kernel (4 args per core):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total output tiles for this core |
| 1 | freq | uint32_t | Broadcast frequency (1 for NONE broadcast type) |
| 2 | counter | uint32_t | Broadcast start counter (0 for NONE broadcast type) |
| 3 | compute_scalar_value | uint32_t | Unused for GCD (0) |

**Writer Kernel (11 args per core, two-tensor path):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Base address of output tensor C buffer |
| 1 | start_tile_id | uint32_t | Starting output tile index for this core |
| 2 | dst_num_tiles | uint32_t | Number of output tiles for this core |
| 3 | dst_shard_width | uint32_t | Shard width in tiles (0 if interleaved) |
| 4 | D | uint32_t | Output D dimension |
| 5 | N | uint32_t | Output N dimension |
| 6 | C | uint32_t | Output C dimension |
| 7 | Ht | uint32_t | Output height in tiles |
| 8 | Wt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Collapsed ND dimension count |
| 10 | (unused) | uint32_t | Always 0 |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 (A, B) | CB c_0, CB c_1 | Read A and B tiles via NoC |
| Compute | MATH (RISCV_2) | N/A | CB c_0, CB c_1 | CB c_2 | copy_tile (unpack A, B to DST), gcd_tile (SFPU binary GCD), pack_tile |
| Writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 (C) | Write output tiles via NoC |

### Reader Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` |
| **Assigned cores** | All worker cores in `all_device_cores` grid |

**Key Logic:**
- Reads tiles from both input tensors A (into CB c_0) and B (into CB c_1) in lockstep, one tile at a time.
- Uses `TensorAccessor` to resolve logical tile indices to physical bank-interleaved addresses.
- Iterates through a 6-level nested loop (ND, D, N, C, Ht, Wt) over the output tile space, mapping each output tile coordinate to input tile coordinates via per-dimension strides.
- Broadcasting is implicit: stride values of 0 cause repeated reads of the same input tile.
- For the sharded path (`SRC_SHARDED` / `SRC_SHARDED_B` defines), the reader simply does `cb_reserve_back` + `cb_push_back` on the full shard since data is already in L1.
- **Synchronization**: Produces to CB c_0 and CB c_1. Calls `cb_reserve_back(cb, 1)` before writing, `noc_async_read_barrier()` to ensure NoC transfer completes, then `cb_push_back(cb, 1)` to signal compute.

### Compute Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` |
| **Assigned cores** | All worker cores in `all_device_cores` grid |

**Key Logic:**
- For GCD, the `BINARY_SFPU_INIT` macro expands to `gcd_tile_init()` and `BINARY_SFPU_OP` expands to `gcd_tile(i*2, i*2+1, i*2)`.
- Since GCD has no LHS/RHS/POST activations, `HAS_ACTIVATIONS(LHS)` = 0, `HAS_ACTIVATIONS(RHS)` = 0, `HAS_ACTIVATIONS(POST)` = 0. The `PREPROCESS` macros are no-ops.
- Per-tile loop structure:
  1. `cb_wait_front(cb_post_lhs, 1)` -- wait for A tile (cb_post_lhs = c_0 since no LHS activations)
  2. `cb_wait_front(cb_post_rhs, 1)` -- wait for B tile (cb_post_rhs = c_1)
  3. `cb_reserve_back(cb_out, 1)` -- reserve output space in c_2
  4. `tile_regs_acquire()` -- acquire DST register tile slots
  5. `copy_tile(cb_post_lhs, 0, 0)` -- unpack A tile into DST[0]
  6. `copy_tile(cb_post_rhs, 0, 1)` -- unpack B tile into DST[1]
  7. `gcd_tile(0, 1, 0)` -- execute SFPU binary GCD: DST[0] = gcd(DST[0], DST[1])
  8. `tile_regs_commit()` + `tile_regs_wait()` -- synchronize math and pack pipelines
  9. `pack_tile(0, cb_out)` -- pack result from DST[0] into CB c_2
  10. `tile_regs_release()` -- release DST
  11. `cb_push_back(cb_out, 1)` -- signal writer
  12. `cb_pop_front(cb_post_lhs, 1)` + `cb_pop_front(cb_post_rhs, 1)` -- free input tiles
- **SFPU Algorithm** (in `ckernel_sfpu_gcd.h`): Implements the binary GCD algorithm using SFPU instructions. Per vector lane:
  1. Compute common trailing zeros via `SFPOR` + negation + `SFPAND` + `SFPLZ` (count leading zeros of isolated LSB).
  2. Ensure b is odd by conditionally swapping a and b based on parity check.
  3. Take absolute values of both operands.
  4. Execute 30 iterations of the core binary GCD loop (stored in SFPU replay buffer for efficiency): isolate LSB of a, shift to make a odd, swap to ensure b <= a, compute a = b - a.
  5. Result (b) is stored back to DST.
- The init function `gcd_tile_init()` loads 4 iterations of the loop body into the SFPU replay buffer (`TTI_REPLAY` with record mode), enabling efficient repeated execution via `TTI_REPLAY` in playback mode.
- **Synchronization**: Consumes from CB c_0 and CB c_1, produces to CB c_2.

### Writer Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` |
| **Assigned cores** | All worker cores in `all_device_cores` grid |

**Key Logic:**
- Writes output tiles from CB c_2 to the output tensor C in DRAM/L1.
- Uses the same 6-level nested loop structure as the reader, but iterating over the output shape directly.
- For the interleaved path, each tile is written via `noc_async_write_page` with a linear tile offset, followed by `noc_async_write_barrier`.
- For the sharded path (`DST_SHARDED` define), the writer is a no-op since data is already in the correct L1 location.
- Uses `TensorAccessor` to map logical tile indices to physical addresses.
- **Synchronization**: Consumes from CB c_2. Calls `cb_wait_front(c_2, 1)` to wait for compute output, then `cb_pop_front(c_2, 1)` after writing to free the buffer slot.

## Implementation Notes

- **Program factory variants**: There is a single program factory (`BinaryNgDeviceOperation::ProgramFactory`) for all binary_ng operations. The SFPU vs FPU path is selected within this factory based on the `is_sfpu` attribute. GCD is SFPU-only.

- **Type-based operation variants**: GCD requires both inputs to be `INT32`. The `is_binary_sfpu_op` function returns `true` only for `a == INT32 && b == INT32`. No other data types are supported.

- **UnpackToDestFP32 mode**: For GCD (as with all non-POWER SFPU ops), `UnpackToDestMode::UnpackToDestFp32` is set on CB indices c_0, c_1, c_3, and c_4. This ensures tiles are unpacked to 32-bit representation in the DST registers, which is necessary for INT32 data to maintain precision through the SFPU pipeline.

- **Broadcast type selection**: GCD supports all `SubtileBroadcastType` modes (NONE, SCALAR_A/B, ROW_A/B, COL_A/B, ROW_B_COL_A, ROW_A_COL_B) through the binary_ng framework's kernel selection mechanism. The broadcast type is determined by comparing input shapes to the output shape. However, the typical case for GCD is same-shape inputs (NONE).

- **Sharding support and constraints**: Height, width, and block sharding are supported through the `get_shard_specs` / `is_native_L1_sharding` mechanism. Sharding requires: both inputs sharded with identical shapes and grids, L1 buffer type (not DRAM), even shard sizes (no remainder tiles), and grids starting at (0,0) for the optimized path. If conditions are not met, the operation falls back to interleaved mode via TensorAccessor.

- **FP32 dest accumulation**: Enabled when output is `Int32` or `UInt32` or `Float32`, or when both inputs are of the same 32-bit type. For GCD with INT32 inputs and INT32 output, `fp32_dest_acc_en` is always `true`.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/gcd.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_gcd.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `gcd_tile(0, 1, 0)` (defined in `gcd.h`), which wraps the call inside the `MATH(...)` macro to ensure execution on the math RISC-V core.
2. Inside `MATH`, `llk_math_eltwise_binary_sfpu_gcd<APPROX>(0, 1, 0)` is called (defined in `llk_math_eltwise_binary_sfpu_gcd.h`).
3. This forwards to `_llk_math_eltwise_binary_sfpu_params_<APPROX>(sfpu::calculate_sfpu_gcd, 0, 1, 0, VectorMode::RC)` (defined in `llk_math_eltwise_binary_sfpu_params.h`), which handles the per-face iteration loop and DEST address progression.
4. The params dispatch calls `sfpu::calculate_sfpu_gcd(0, 1, 0)` once per face (4 times for `VectorMode::RC`), which loads data from DEST, calls `calculate_sfpu_gcd_body<31>()` for the actual binary GCD algorithm, and stores the result back.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the 32x32 tile are processed. Each face consists of 16x16 elements.
- **Operation invocation**: In RC mode, the params dispatch loops `for (int face = 0; face < 4; face++)`, calling `calculate_sfpu_gcd(dst_index_in0, dst_index_in1, dst_index_out)` once per face. Inside `calculate_sfpu_gcd`, there is an inner loop of 8 iterations (default `ITERATIONS=8`), where each iteration processes one row-group (4 rows x 16 columns = 64 elements) of the face via SFPU vector operations.
- **DEST address progression**: Between faces, the params dispatch advances the DEST read/write pointer by `2 * SETRWC(CR_D, 8)` = 16 rows. Inside `calculate_sfpu_gcd`, each of the 8 iterations increments `dst_reg` by 1 (advancing 4 rows in DEST). The `SFPLOAD`/`SFPSTORE` instructions use absolute addressing based on `dst_index * 64` (tile size in DEST), so the face-level `SETRWC` increments provide the base offset that the SFPU load/store addresses are relative to.

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions with complex condition code manipulation in `calculate_sfpu_gcd_body`. It is analyzed using **Style B**.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h

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

**CC State Machine -- `calculate_sfpu_gcd_body`:**

```
calculate_sfpu_gcd_body — CC State Transitions
════════════════════════════════════════════════════════════════

  CC State: ALL_ENABLED                   <-- initial state
       |
       |  SFPMOV  L2 = L0                  (no CC effect)
       |  SFPOR   L2 = L1 | L2            (no CC effect)
       |  SFPMOV  L3 = L2                  (no CC effect)
       |  SFPIADD L3 = -L3  CC_NONE       (no CC effect — CC_NONE suppresses update)
       |  SFPAND  L3 = L2 & L3            (no CC effect)
       |  SFPLZ   L3 = clz(L3)            (no CC effect — InstrMod=0, CC not enabled)
       |  SFPSHFT2 L2 = L1 << L3          (no CC effect)
       |
       v
  +---------------------------------------------+
  | SFPSETCC  InstrMod=6                        |
  |                                             |
  | CC <- (L2 == 0)                             |
  | i.e. CC set for lanes where b is even       |
  +-------------------+-------------------------+
                      |
                      v
  CC State: ENABLED where (b << d) == 0 (i.e. b is even)
       |
       |  SFPSWAP L0, L1  mod=0           (CC-guarded: swap a,b only in even-b lanes)
       |
       v
  +---------------------------------------------+
  | SFPENCC                                     |
  |                                             |
  | CC <- ALL_ENABLED                           |
  +-------------------+-------------------------+
                      |
                      v
  CC State: ALL_ENABLED
       |
       |  SFPABS  L0 = abs(L0)            (all lanes)
       |  SFPABS  L1 = abs(L1)            (all lanes)
       |  SFPIADD L0 = -L0  CC_NONE       (no CC effect — CC_NONE suppresses)
       |  SFPIADD L3 = -L3  CC_NONE       (no CC effect — CC_NONE suppresses)
       |
       v
  ── Replay Loop (28 instructions = 4 iterations x 7 instructions) ──
  ── then replay (7*2 - 1 = 13 instructions = ~2 iterations) ──
  ── Total: 30 iterations of the inner GCD loop body ──
  ── (CC manipulation within replay is documented below) ──
       |
       v
  +---------------------------------------------+
  | SFPENCC                                     |
  |                                             |
  | CC <- ALL_ENABLED                           |
  +-------------------+-------------------------+
                      |
                      v
  CC State: ALL_ENABLED                   <-- final state
```

**CC State Machine -- replay loop body (recorded in `calculate_sfpu_gcd_init`):**

Each iteration of the replay buffer contains 7 instructions. The `SFPLZ` instruction with `SFPLZ_MOD1_CC_NE0` (InstrMod=2, i.e. InstrMod[1]=1) sets CC based on whether the input is non-zero, which disables lanes where `a == 0` (GCD has converged for those lanes).

```
Replay loop iteration — CC State Transitions
════════════════════════════════════════════════════════════════

  CC State: (inherited from previous iteration or ALL_ENABLED on first)
       |
       |  SFPABS  L2 = abs(L0)            (CC-guarded: only active lanes)
       |  SFPAND  L0 = L2 & L0            (CC-guarded: isolate LSB of a)
       |
       v
  +---------------------------------------------+
  | SFPLZ  L0 = clz(L0)  SFPLZ_MOD1_CC_NE0    |
  |   InstrMod[1]=1: update CC based on L0      |
  |                                             |
  | CC <- (L0 != 0)                             |
  | Lanes where a==0 are disabled (GCD done)    |
  +-------------------+-------------------------+
                      |
                      v
  CC State: ENABLED where a != 0
       |
       |  SFPIADD L0 += L3  CC_NONE       (CC-guarded: only a!=0 lanes; no CC update)
       |  SFPSHFT2 L0 = L2 >> -L0         (CC-guarded: make a odd)
       |  SFPSWAP L0, L1 VEC_MIN_MAX      (CC-guarded: ensure b <= a as integers)
       |  SFPIADD L0 = L1 - L0  CC_NONE   (CC-guarded: a = b - a, result is even)
       |
       v
  CC State: ENABLED where a != 0  (unchanged, carried to next iteration)
```

### SFPU Instructions Used

| Instruction | Mnemonic | Description |
|-------------|----------|-------------|
| `TTI_SFPMOV` | SFPMOV | Register-to-register move: copies LREG[VC] to LREG[VD] |
| `TTI_SFPOR` | SFPOR | Bitwise OR: LREG[VD] = LREG[VB] \| LREG[VC] |
| `TTI_SFPAND` | SFPAND | Bitwise AND: LREG[VD] = LREG[VB] & LREG[VC] |
| `TTI_SFPLZ` | SFPLZ | Count leading zeros of LREG[VC], store in LREG[VD]. With `SFPLZ_MOD1_CC_NE0` (InstrMod=2), also sets CC.Res = (VC != 0) |
| `TTI_SFPIADD` | SFPIADD | Integer add/subtract. With `SFPIADD_MOD1_ARG_2SCOMP_LREG_DST`, computes VD = VC - VD (two's complement negate-then-add). With `SFPIADD_MOD1_CC_NONE`, CC update is suppressed |
| `TTI_SFPSHFT2` | SFPSHFT2 | Bitwise shift. With `SFPSHFT2_MOD1_SHFT_LREG`, shifts VC by the amount in VB (negative = right shift, positive = left shift) |
| `TTI_SFPSETCC` | SFPSETCC | Set condition code. With InstrMod=6, sets CC.Res = (VC == 0) per lane |
| `TTI_SFPSWAP` | SFPSWAP | Exchange two registers. InstrMod=0: unconditional swap. With `SFPSWAP_MOD1_VEC_MIN_MAX` (InstrMod=1): conditional swap so VD=min(VC,VD), VC=max(VC,VD) using INT32 comparison (Imm12[0]=0) |
| `TTI_SFPENCC` | SFPENCC | Disable predicated execution, resetting CC to ALL_ENABLED |
| `TTI_SFPABS` | SFPABS | Absolute value: LREG[VD] = abs(LREG[VC]) as two's complement integer |
| `TT_SFPLOAD` | SFPLOAD | Load from DEST register file into LREG. InstrMod=4 (SMAG32 format), addr_mode=3 (32-bit DEST addressing) |
| `TT_SFPSTORE` | SFPSTORE | Store from LREG to DEST register file. InstrMod=4 (SMAG32 format), addr_mode=3 (32-bit DEST addressing) |
| `TTI_REPLAY` | REPLAY | Replay buffer control. With load_mode=1: record following instructions into replay buffer. With load_mode=0: play back `len` instructions from the replay buffer starting at `start_idx` |

### SFPU Register Usage

| Register | Role | Description |
|----------|------|-------------|
| **LREG0** | `a` (first operand) | Loaded from DST[dst_index_in0]. Holds the first GCD operand. Negated to `-a` before the main loop. During the loop body, holds the even difference `b - a`. |
| **LREG1** | `b` (second operand / result) | Loaded from DST[dst_index_in1]. Holds the second GCD operand. After convergence, holds the GCD result. Stored back to DST[dst_index_out]. |
| **LREG2** | `c` (temporary) | Used in the preamble to compute `a | b` for isolating the common trailing zeros. In the replay loop body, holds `abs(a)` for LSB isolation. |
| **LREG3** | `d` (trailing zero count) | Holds the negated count of common trailing zeros. Used as the shift amount to test whether `b` is odd, and in the loop to compute the combined shift for removing trailing zeros from `a`. |
| **DEST[in0 * 64 + row_offset]** | Input tile A | Source for `a` values via `TT_SFPLOAD`. Each iteration loads 4 rows x 16 columns. |
| **DEST[in1 * 64 + row_offset]** | Input tile B | Source for `b` values via `TT_SFPLOAD`. |
| **DEST[out * 64 + row_offset]** | Output tile | Destination for GCD result via `TT_SFPSTORE`. Receives LREG1 (the converged `b` value). |
| **CC.Res** | Condition code | Used for two purposes: (1) in the preamble, gates the conditional swap to ensure `b` is odd; (2) in the replay loop, disables lanes where `a == 0` (GCD has converged). |

### Address Mode Configuration

The GCD operation uses `ADDR_MOD_7` configured by `eltwise_binary_sfpu_configure_addrmod<SfpuType::gcd>()`.

Since `SfpuType::gcd` does not match any of the types in the `if constexpr` branch (which checks for `mul_int32`, `mul_uint16`, `max`, `min`, `max_int32`, `min_int32`, `max_uint32`, `min_uint32`), only `ADDR_MOD_7` is configured:

| Field | Value | Description |
|-------|-------|-------------|
| `srca.incr` | 0 | No auto-increment for SRC A |
| `srcb.incr` | 0 | No auto-increment for SRC B |
| `dest.incr` | 0 | No auto-increment for DEST |

`ADDR_MOD_6` is **not** configured for GCD (that branch is only for min/max/mul operations that need `dest.incr = 2`).

The DEST address progression is instead handled explicitly: `calculate_sfpu_gcd` increments `dst_reg` after each iteration (advancing 4 rows in DEST), and the params dispatch uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice between faces to advance the DEST pointer by 16 rows.

This configuration is **identical** for both Wormhole B0 and Blackhole -- the `eltwise_binary_sfpu_configure_addrmod` function and the `llk_math_eltwise_binary_sfpu_params.h` file have the same logic on both architectures. The core SFPU kernel (`ckernel_sfpu_gcd.h`) is also identical between the two architectures.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng program factory work? What are the SFPU vs FPU paths, and how does it select between them? What kernels does it use?"
   **Reason**: Needed to understand the overall architecture of the binary_ng framework before diving into source code.
   **Key Findings**: Confirmed that binary_ng uses a single program factory with `OpConfig` to select SFPU/FPU paths. The `is_binary_sfpu_op` function determines the path. Kernel file selection uses `get_kernel_file_path` with an `is_sfpu` flag.

2. [SFPU] **Query**: "How does the binary_ng compute kernel dispatch SFPU operations? Specifically, how does it invoke the GCD SFPU kernel? What is the call chain from the compute kernel through LLK to the ckernel SFPU implementation?"
   **Reason**: Needed to trace the full call chain from `gcd_tile()` through the LLK abstraction layers down to the core SFPU implementation.
   **Key Findings**: Confirmed the call chain: `gcd_tile()` -> `MATH()` -> `llk_math_eltwise_binary_sfpu_gcd<APPROX>()` -> `_llk_math_eltwise_binary_sfpu_params_()` -> `sfpu::calculate_sfpu_gcd()` -> `calculate_sfpu_gcd_body<31>()`. The init path records 4 iterations of the inner loop into the SFPU replay buffer.

3. [SFPU] **Query**: "What do the following SFPU instructions do: SFPMOV, SFPOR, SFPAND, SFPLZ, SFPSHFT2, SFPSETCC, SFPSWAP, SFPENCC, SFPABS, SFPIADD, SFPLOAD, SFPSTORE, SFPREPLAY?"
   **Reason**: Needed detailed semantics for all SFPU instructions used in the GCD kernel.
   **Key Findings**: Obtained general descriptions of all instructions. SFPSWAP with `SFPSWAP_MOD1_VEC_MIN_MAX` performs conditional min/max exchange. SFPLZ with `SFPLZ_MOD1_CC_NE0` sets CC based on non-zero test. SFPSETCC with InstrMod=6 sets CC when value equals zero. DeepWiki was incomplete on some modifier details, so Confluence was consulted.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp` (lines 323-328)
   **Reason**: Needed to confirm GCD's SFPU-only status and its OpConfig mapping.
   **Key Information**: GCD maps to `SfpuBinaryOp::GCD` and throws for FPU path.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp` (line 42)
   **Reason**: Needed to confirm under what conditions GCD selects the SFPU path.
   **Key Information**: `GCD: return (a == INT32 && b == INT32)` -- only INT32 inputs supported.

3. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h`
   **Reason**: Needed to understand the actual SFPU algorithm for GCD.
   **Key Information**: Implements binary GCD algorithm with 30 iterations using SFPU replay buffer. Uses SFPLZ (count leading zeros), SFPAND, SFPOR, SFPSWAP, SFPIADD, SFPSHFT2, SFPABS instructions. The init function records 4 loop iterations into the replay buffer for efficient execution.

4. **Source**: `tt_metal/hw/inc/api/compute/gcd.h`
   **Reason**: Needed to understand the compute API surface for GCD.
   **Key Information**: `gcd_tile(idst0, idst1, odst)` takes two DST indices as inputs and one as output. Both inputs must be int32. Init function is `gcd_tile_init()`.

### Confluence References

1. [SFPU] **Page**: Tensix SFPU Instruction Set Architecture (page ID: 1170505767)
   **Sections consulted**: SFPSETCC, SFPSWAP, SFPLZ, SFPLOAD
   **Reason**: DeepWiki did not provide exact InstrMod encoding details for SFPSETCC (mod=6), SFPSWAP conditional exchange semantics, or SFPLZ CC update behavior.
   **Key Findings**:
   - `SFPSETCC` InstrMod=6: Sets `CC.Res = (RG[VC] == 0)` per lane, treating the input as INT32 (Imm12[11]=0).
   - `SFPSWAP` InstrMod=1 (`SFPSWAP_MOD1_VEC_MIN_MAX`): Conditional exchange where `VD` gets the smaller value and `VC` gets the larger, using INT32 comparison when Imm12[0]=0.
   - `SFPLZ` InstrMod=2 (`SFPLZ_MOD1_CC_NE0`): InstrMod[1]=1 enables CC update, setting `CC.Res = (MaskedVal != 0)`.
   - `SFPLOAD` InstrMod=4: SMAG32 format load from DEST; addr_mode=3 indicates 32-bit DEST addressing.
