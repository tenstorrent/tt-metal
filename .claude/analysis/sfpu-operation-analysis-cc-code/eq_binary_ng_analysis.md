# EQ (binary_ng) Implementation Analysis

## Overview

The EQ (equal) operation in the binary_ng framework compares two tensors element-wise, producing an output where each element indicates whether the corresponding elements of the inputs are equal. It is implemented through the unified `binary_ng` program factory at:

`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

The EQ operation has two distinct SFPU implementation strategies depending on data type:
1. **FLOAT32 path**: Uses a dedicated `eq_binary_tile` SFPU intrinsic (`SfpuBinaryOp::EQ`) that performs the comparison directly on the SFPU.
2. **INT32/UINT32/UINT16 path**: Decomposes into SUB (SFPU subtraction) followed by an `EQZ` (equal-to-zero) unary post-activation. The subtraction is performed via the SFPU, and the EQZ check is applied as a post-processing step in the compute kernel.

For non-SFPU-eligible data types (e.g., both inputs BF16), the FPU path is used instead.

## Path Selection: FPU vs SFPU

The path selection is determined by `utils::is_binary_sfpu_op()` in `binary_ng_device_operation.cpp` (line 15). For `BinaryOpType::EQ`, the function returns `true` (SFPU path) when both input data types are identical AND are one of: `FLOAT32`, `INT32`, `UINT32`, or `UINT16` (line 21-27).

When the SFPU path is selected (`is_sfpu == true`), the `OpConfig` constructor in `binary_ng_utils.cpp` (line 188-194) further branches:
- If `dtype == FLOAT32`: sets `binary_op = SfpuBinaryOp::EQ`, which maps to `eq_binary_tile_init()` / `eq_binary_tile()` -- a native FP32 equality comparison on the SFPU.
- Otherwise (INT32, UINT32, UINT16): sets `binary_op = SfpuBinaryOp::SUB` (default from constructor) and `postprocess = UnaryOpType::EQZ`. This subtracts the two inputs and then checks if the result equals zero.

When the FPU path is selected (e.g., both inputs BF16), the `OpConfig` uses `FpuBinaryOp::SUB` with `postprocess = EQZ`, leveraging the FPU's native subtraction followed by the same EQZ unary check.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `c.physical_volume() / tile_hw` (total output tiles) |
| **Loop structure** | Per-tile: reader reads 1 tile from A and 1 from B, compute processes 1 tile, writer writes 1 tile |

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|---------------|---------------|
| **Logical shape** | Up to 6D, collapsed beyond 5D | Up to 6D, collapsed beyond 5D |
| **Dimension convention** | [..., D, N, C, H, W] | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED (height/width/block) | INTERLEAVED or SHARDED (height/width/block) |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | FLOAT32, INT32, UINT32, or UINT16 (SFPU path) | Same as A (SFPU path requires matching types) |

### Output Tensor

| Property | Output Tensor C |
|----------|----------------|
| **Logical shape** | Broadcast-compatible output of A and B shapes |
| **Dimension convention** | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Configurable (commonly FLOAT32 for comparison results, or same as input) |

### Layout Transformations

No explicit tilize/untilize operations are performed within the kernels. Both inputs and outputs are expected to be in TILE_LAYOUT. When `a_dtype != c_dtype` (and not a quant op), a TYPECAST post-activation is automatically appended to handle the output data format conversion.

## Data Flow Pattern

For the two-tensor (non-scalar) case with no broadcast (`SubtileBroadcastType::NONE`):

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (tensor A) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `cb_push_back(c_0, 1)` |
| 1 | Reader | DRAM/L1 (tensor B) | CB c_1 | `cb_reserve_back(c_1, 1)`, `noc_async_read_page`, `cb_push_back(c_1, 1)` |
| 2 | Compute | CB c_0, CB c_1 | CB c_2 | `cb_wait_front(c_0, 1)`, `cb_wait_front(c_1, 1)`, `cb_reserve_back(c_2, 1)`, copy tiles to DST, `BINARY_SFPU_OP`, `PROCESS_POST_ACTIVATIONS`, pack, `cb_push_back(c_2, 1)`, `cb_pop_front(c_0, 1)`, `cb_pop_front(c_1, 1)` |
| 3 | Writer | CB c_2 | DRAM/L1 (tensor C) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

For the scalar case (B is a scalar value, not a tensor):
- The writer kernel fills CB c_1 with the packed scalar value once at the start.
- The reader only reads tensor A into CB c_0.
- The compute kernel reads the scalar from CB c_1 once and reuses it for all tiles.

## Circular Buffer Configuration

### FLOAT32 EQ (no pre/post activations needed -- native eq_binary_tile)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src_a | Input A staging | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) | Reader | Compute | Block |
| c_1 | cb_src_b | Input B staging | 2 tiles (interleaved, 2-tensor) or 1 tile (scalar) or shard volume (sharded) | 1 tile | Double (interleaved, 2-tensor) / Single (scalar) | Reader/Writer | Compute | Block (2-tensor) / Program (scalar) |
| c_2 | cb_out | Output staging | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) | Compute | Writer | Block |

### Non-FLOAT32 EQ (SUB + EQZ post-activation)

Same as above. The EQZ post-activation does not require additional intermediate CBs because it is a simple unary operation applied in-place on the DST register before packing. The `PROCESS_POST_ACTIVATIONS` macro applies EQZ directly on the DST tile index.

**Note on LHS/RHS activation CBs**: If `lhs_activations` or `rhs_activations` are non-empty (not the case for basic EQ), CB c_3 and/or c_4 would be created as single-tile intermediate buffers. For standard EQ, these are not allocated.

## Pipeline Pattern Summary

- **Interleaved mode**: All primary CBs (c_0, c_1, c_2) have capacity of 2 tiles with block size of 1 tile, enabling **double-buffering**. The reader can fill the next tile while compute processes the current one.
- **Sharded mode**: CB capacity equals the shard volume. All tiles are loaded at once, operating as a **bulk transfer** pattern rather than streaming.
- **Scalar mode**: CB c_1 is single-buffered (1 tile), loaded once at program start.

## Index Calculations

The reader kernel uses a 6-level nested loop structure to map logical tensor coordinates to physical tile offsets. The dimensions are (from outermost to innermost): `nD` (collapsed dims > 5), `D`, `N`, `C`, `Ht` (tile rows), `Wt` (tile columns).

For broadcasting, the input tile offset uses stride values that encode broadcasting behavior:
- `nD_stride = aHt * aWt * aC * aN * aD * (aND > 1)`: If `aND == 1`, the stride is 0, effectively broadcasting across the nD dimension.
- Similar logic for `d_stride`, `n_stride`, `c_stride`.

The `start_tile_id` for each core is computed from the output tensor's total tile count, distributed evenly across cores. The reader converts `start_tile_id` back to multi-dimensional coordinates using modular arithmetic against the output shape dimensions.

For sharded mode, the start tile ID is computed geometrically: `(core_index / num_shards_per_width) * (c_shard_height * cWt) + (core_index % num_shards_per_width) * c_shard_width`.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Tile-sequential within the innermost dimensions (Wt), with strided jumps at dimension boundaries. Each tile is read individually with `noc_async_read_page` followed by an immediate barrier. Both A and B tiles at the same logical position are read in the same inner loop iteration.
- **Sharded**: Bulk reserve/push of the entire shard. No NoC reads needed -- data is already in L1.

### Write Pattern
- **Interleaved**: Tile-sequential with `noc_async_write_page` per tile, immediate barrier after each write.
- **Sharded**: No explicit writes -- output CB is backed by the sharded output buffer in L1.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (or 1D depending on available cores) |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `compute_with_storage_grid.x * compute_with_storage_grid.y` (zero-start grid) or `all_device_cores.num_cores()` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two-group split: `core_group_1` gets `ceil(total_tiles / num_cores)` tiles, `core_group_2` gets one fewer. Cores outside both groups are assigned zero work (noop). |

For sharded mode, each core processes exactly its shard's tiles. The shard shape may vary for edge cores (last row/column of the core grid).

## Arguments

### Compile-Time Arguments

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Always 1. Number of output tiles produced per read-compute-write cycle. |

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs (A) | uint32_t[] | Tensor accessor compile-time args for input A |
| N+1..M | TensorAccessorArgs (B) | uint32_t[] | Tensor accessor compile-time args for input B |
| M+1 | has_sharding | uint32_t | 1 if any tensor is natively sharded, 0 otherwise |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs (C) | uint32_t[] | Tensor accessor compile-time args for output C |
| N+1 | has_sharding | uint32_t | 1 if any tensor is natively sharded, 0 otherwise |

### Runtime Arguments

#### Reader Kernel (two-tensor, no-broadcast case)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Buffer address of input tensor A |
| 1 | start_tile_id | uint32_t | Starting output tile ID for this core (c_start_id) |
| 2 | src_num_tiles | uint32_t | Number of A tiles in shard (sharded) or 0 (interleaved) |
| 3 | dst_num_tiles | uint32_t | Number of output tiles for this core (c_num_tiles) |
| 4 | dst_shard_width | uint32_t | Width of output shard in tiles (sharded) or 0 |
| 5 | nD_stride | uint32_t | Stride for collapsed dims > 5 (0 if broadcasting) |
| 6 | d_stride | uint32_t | Stride for D dimension |
| 7 | n_stride | uint32_t | Stride for N dimension |
| 8 | c_stride | uint32_t | Stride for C dimension |
| 9 | D | uint32_t | Output D dimension |
| 10 | N | uint32_t | Output N dimension |
| 11 | C | uint32_t | Output C dimension |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Collapsed dimensions > 5 for output |
| 15 | src_addr_b | uint32_t | Buffer address of input tensor B |
| 16 | nD_stride_b | uint32_t | Stride for B collapsed dims > 5 |
| 17 | d_stride_b | uint32_t | Stride for B D dimension |
| 18 | n_stride_b | uint32_t | Stride for B N dimension |
| 19 | c_stride_b | uint32_t | Stride for B C dimension |
| 20 | b_num_tiles | uint32_t | Number of B tiles in shard (sharded) or 0 |

#### Writer Kernel (two-tensor case)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Buffer address of output tensor C |
| 1 | start_tile_id | uint32_t | Starting output tile ID (c_start_id) |
| 2 | dst_num_tiles | uint32_t | Number of output tiles for this core |
| 3 | dst_shard_width | uint32_t | Width of output shard in tiles (sharded) or 0 |
| 4 | D | uint32_t | Output D dimension |
| 5 | N | uint32_t | Output N dimension |
| 6 | C | uint32_t | Output C dimension |
| 7 | Ht | uint32_t | Output height in tiles |
| 8 | Wt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Collapsed dimensions > 5 |
| 10 | (unused) | uint32_t | Reserved (set to 0) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total tiles to process on this core |
| 1 | freq | uint32_t | Broadcast frequency (1 for NONE broadcast) |
| 2 | counter | uint32_t | Starting counter for broadcast cycling (0 for NONE) |
| 3 | compute_scalar_value | uint32_t | Unused for EQ (set to 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 (A, B) | CB c_0, CB c_1 | Read A and B tiles via NoC |
| Compute | MATH+PACK (RISCV_2) | N/A | CB c_0, CB c_1 | CB c_2 | copy_tile to DST, BINARY_SFPU_OP (eq_binary_tile or sub+EQZ), pack_tile |
| Writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 (C) | Write output tiles via NoC |

### Reader Kernel (Two-Tensor, No Broadcast)

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` |
| Assigned cores | All worker cores in `all_device_cores` |

**Key Logic**:
- For **sharded** inputs: performs a single `cb_reserve_back` / `cb_push_back` for the entire shard volume on each CB. No NoC reads are needed since data resides in L1.
- For **interleaved** inputs: iterates through a 6-level nested loop (nD, D, N, C, Ht, Wt) using the output tile ordering. For each tile position, reads both A and B tiles in the same iteration.
- Uses `TensorAccessor` for page-to-bank address resolution, enabling correct interleaved memory access across DRAM banks.
- Both A and B reads share a single `noc_async_read_barrier()` call per tile pair, amortizing synchronization overhead.
- Broadcasting is encoded in the stride arguments: a stride of 0 means the dimension is broadcast (the input tile offset does not advance for that dimension).
- **Synchronization**: Produces tiles into CB c_0 and CB c_1 via `cb_reserve_back` / `cb_push_back`. The compute kernel consumes from these CBs.

### Compute Kernel (SFPU, No Broadcast)

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` |
| Assigned cores | All worker cores in `all_device_cores` |

**Key Logic**:
- Initializes with `unary_op_init_common(cb_post_lhs, cb_out)` to set up unpack/pack pipelines.
- For FLOAT32 EQ: `BINARY_SFPU_INIT` expands to `eq_binary_tile_init()` and `BINARY_SFPU_OP` expands to `eq_binary_tile(i*2, i*2+1, i*2)`.
- For non-FLOAT32 EQ (e.g., INT32): `BINARY_SFPU_INIT` expands to `sub_int_tile_init()`, `BINARY_SFPU_OP` expands to `sub_int_tile<DataFormat::Int32>(i*2, i*2+1, i*2)`, and `PROCESS_POST_ACTIVATIONS(i*2)` applies the EQZ unary operation.
- Tile processing per iteration:
  1. `cb_wait_front(cb_post_lhs, 1)` and `cb_wait_front(cb_post_rhs, 1)` -- wait for input tiles.
  2. `cb_reserve_back(cb_out, 1)` -- reserve output space.
  3. `tile_regs_acquire()` -- acquire DST register file.
  4. `copy_tile(cb_post_lhs, 0, 0)` -- unpack LHS tile to DST[0].
  5. `copy_tile(cb_post_rhs, 0, 1)` -- unpack RHS tile to DST[1].
  6. `BINARY_SFPU_OP(0, 1, 0)` -- perform SFPU comparison/subtraction, result in DST[0].
  7. `PROCESS_POST_ACTIVATIONS(0)` -- apply EQZ if needed (no-op for FLOAT32 path).
  8. `tile_regs_commit()` / `tile_regs_wait()` -- pipeline synchronization between math and pack.
  9. `pack_tile(0, cb_out)` -- pack DST[0] to output CB.
  10. `tile_regs_release()` -- release DST registers.
  11. `cb_push_back(cb_out, 1)`, `cb_pop_front(cb_post_lhs, 1)`, `cb_pop_front(cb_post_rhs, 1)`.
- Uses `UnpackToDestMode::UnpackToDestFp32` for all source CBs (since EQ is not POWER op), ensuring FP32 precision in DST registers.
- **Synchronization**: Consumes from CB c_0 and CB c_1 via `cb_wait_front` / `cb_pop_front`. Produces to CB c_2 via `cb_reserve_back` / `cb_push_back`.

### Writer Kernel (Two-Tensor, No Broadcast)

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` |
| Assigned cores | All worker cores in `all_device_cores` |

**Key Logic**:
- For **sharded** output: the writer kernel body is entirely skipped (`#if !DST_SHARDED`). The output CB is already backed by the sharded L1 buffer, so packing into the CB is sufficient.
- For **interleaved** output: iterates through the same 6-level nested loop as the reader, writing one tile at a time with `noc_async_write_page` + immediate `noc_async_write_barrier`.
- Uses `TensorAccessor` for output page-to-bank resolution.
- Handles shard-width adjustment: when sharding is active, the output tile offset skips `(Wt - dst_shard_width)` tiles at the end of each tile row to account for partial width coverage.
- **Synchronization**: Consumes from CB c_2 via `cb_wait_front` / `cb_pop_front`. This is the terminal consumer in the pipeline.

## Implementation Notes

- **Program factory variants**: There is a single `BinaryNgDeviceOperation::ProgramFactory` that handles all binary_ng operations. The factory is always selected; there are no alternative factories. The specific operation behavior is controlled entirely through compile-time defines and runtime arguments.
- **Type-based operation variants**: For EQ on the SFPU path, FLOAT32 uses the native `eq_binary_tile` intrinsic (calls `llk_math_eltwise_binary_sfpu_eq_fp32`). INT32/UINT32/UINT16 use SUB + EQZ decomposition. The specific sub variant is selected by integer data format templating (e.g., `sub_int_tile<DataFormat::Int32>`).
- **UnpackToDestFP32 mode**: Always enabled for SFPU EQ (since `op_type != BinaryOpType::POWER`). All four source/intermediate CBs (c_0, c_1, c_3, c_4) use `UnpackToDestMode::UnpackToDestFp32`, ensuring tiles are unpacked to FP32 format in DST registers regardless of the source data format.
- **Broadcast type selection**: The no-broadcast case (`SubtileBroadcastType::NONE`) uses the `eltwise_binary_sfpu_no_bcast.cpp` compute kernel and the `reader_interleaved_no_bcast.cpp` (from `kernels_ng`) reader. Other broadcast types (SCALAR, ROW, COL, SCALAR_A/B, ROW_A/B_COL_B/A) are supported through different kernel file selections and stride-based broadcasting in the reader arguments.
- **Sharding support and constraints**: Height, width, and block sharding are all supported. Native L1 sharding requires: (1) both inputs same shape, same memory config, same shard grid as output, (2) all in L1 (not DRAM), (3) evenly sharded output. If any condition fails, the operation falls back to the interleaved (tensor accessor) path even if tensors are technically sharded.
- **FP32 dest accumulation**: Enabled when output format is UInt32/Int32/Float32, or when both inputs are Float32/Int32/UInt32. For EQ with FLOAT32 inputs, this is always enabled. The `fp32_dest_acc_en` flag is passed to `ComputeConfig`.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the FLOAT32 EQ path (the native `eq_binary_tile` intrinsic).

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binary_comp.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_comp.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `eq_binary_tile(idst0, idst1, odst)` (defined in `eltwise_binary_sfpu.h`), which wraps the call in the `MATH(...)` macro ensuring it only executes on the math RISC-V.
2. This calls `llk_math_eltwise_binary_sfpu_eq_fp32<APPROX>(idst0, idst1, odst)` (in `llk_math_eltwise_binary_sfpu_binary_comp.h`), which passes the core SFPU function `ckernel::sfpu::calculate_binary_comp_fp32<APPROXIMATE, 8, SfpuType::eq>` to the params dispatch.
3. The params dispatch `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(sfpu_func, ...)` (in `llk_math_eltwise_binary_sfpu_params.h`) sets up the DEST write address, stalls until the SFPU is ready, then iterates over tile faces calling the SFPU function once per face (4 times for `VectorMode::RC`), advancing the DEST read/write counter by 16 rows between faces via `TTI_SETRWC`.
4. The core function `calculate_binary_comp_fp32<APPROXIMATE, 8, SfpuType::eq>` (in `ckernel_sfpu_binary_comp.h`) iterates 8 times per face (one iteration per row of 16-wide SIMD vector), loading two FP32 values from DEST, comparing them, and writing the result (1.0f or 0.0f) back to DEST.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the 32x32 tile are processed -- the params dispatch loops `face = 0..3`, calling the SFPU function once per face.
- **Operation invocation**: For each face, `calculate_binary_comp_fp32` is called once. Internally it loops 8 iterations (ITERATIONS=8), processing one row of 16 elements per iteration, covering the full 16x8=128 elements of one face (but stored as 8 rows of the SFPU's native 16-wide SIMD).
- **DEST address progression**: Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice, advancing the DEST counter by 16 rows total (2 x 8). Within the SFPU function, `sfpi::dst_reg++` increments the DEST address by `SFP_DESTREG_STRIDE` (=2) after each of the 8 iterations, covering 16 rows per face.

### Annotated SFPU Kernel Source

This kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, `v_if`/`v_elseif`/`v_endif`), so Style A (inline-commented source code) is used.

**Wormhole B0 variant:**

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_comp.h

template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_fp32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // APPROXIMATION_MODE=APPROX (compile-time), ITERATIONS=8, RELATIONAL_OP=SfpuType::eq
    static_assert(RELATIONAL_OP == SfpuType::eq, "Supported operation types: eq ");
    constexpr uint dst_tile_size_sfpi = 32; // SFPI stride: 64 DEST rows / SFP_DESTREG_STRIDE(2) = 32

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // Load two FP32 values from DEST at tile offsets for input0 and input1
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = 0.0f; // Default: not equal

        if constexpr (RELATIONAL_OP == SfpuType::eq) {
            sfpi::vInt in0_bits = sfpi::reinterpret<sfpi::vInt>(in0); // Bitwise reinterpret for special value handling
            sfpi::vInt in1_bits = sfpi::reinterpret<sfpi::vInt>(in1);

            // Standard float comparison (handles normal values and NaN correctly)
            v_if(in0 == in1) { result = 1.0f; }
            // Special handling for infinity: bitwise comparison AND check for inf exponent
            v_elseif((in0_bits == in1_bits) && ((in0_bits & 0x7FFFFFFF) == 0x7F800000)) { result = 1.0f; }
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // Store result to output tile in DEST
        sfpi::dst_reg++; // Advance DEST row pointer by SFP_DESTREG_STRIDE(2)
    }
}
```

**Blackhole variant** (differences from Wormhole annotated):

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_comp.h

template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_fp32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // APPROXIMATION_MODE=APPROX (compile-time), ITERATIONS=8, RELATIONAL_OP=SfpuType::eq
    static_assert(RELATIONAL_OP == SfpuType::eq, "Supported operation types: eq ");
    constexpr uint dst_tile_size_sfpi = 32;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = 0.0f;

        if constexpr (RELATIONAL_OP == SfpuType::eq) {
            sfpi::vInt in0_bits = sfpi::reinterpret<sfpi::vInt>(in0);
            sfpi::vInt in1_bits = sfpi::reinterpret<sfpi::vInt>(in1);
            sfpi::vInt in0_abs = in0_bits & 0x7FFFFFFF; // BH needs explicit abs for +0/-0 handling
            sfpi::vInt in1_abs = in1_bits & 0x7FFFFFFF;

            // BH: (-0.0 == 0.0) returns false in hardware FP compare, so explicitly check both-zero
            v_if((in0 == in1) || (in0_abs == 0 && in1_abs == 0)) { result = 1.0f; }
            // Special handling for infinity (same logic, uses precomputed abs)
            v_elseif((in0_bits == in1_bits) && ((in0_abs == 0x7F800000))) { result = 1.0f; }
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}
```

### SFPU Instructions Used

The SFPI abstractions in `calculate_binary_comp_fp32` compile down to the following SFPU instructions:

| Instruction / Intrinsic | Description |
|------------------------|-------------|
| `SFPLOAD` (via `dst_reg[]` read) | Loads a 16-wide vector from DEST register into an SFPU LREG. Used to read `in0` and `in1` from their respective tile offsets in DEST. |
| `SFPSTORE` (via `dst_reg[] =`) | Stores a 16-wide vector from an SFPU LREG back to DEST register. Used to write the `result` to the output tile offset in DEST. |
| `SFPLOADI` (via `result = 0.0f`, `result = 1.0f`) | Loads an immediate constant into an SFPU LREG. Used for the comparison result values 0.0f and 1.0f. |
| `SFPSETCC` / `SFPENCC` (via `v_if`, `v_elseif`, `v_endif`) | Condition code manipulation. `v_if(in0 == in1)` compiles to a float comparison that sets the CC register per lane. `v_elseif` complements CC and sets a new condition. `v_endif` restores CC to all-enabled. |
| `SFPIADD` (via integer comparison `in0_bits == in1_bits`) | Integer subtraction used for bitwise equality check on the reinterpreted integer representations. Sets CC based on result. |
| `SFPAND` (via `in0_bits & 0x7FFFFFFF`) | Bitwise AND to mask off the sign bit, producing the absolute value for infinity detection. |
| `SFPMOV` (via intermediate assignments) | Moves values between SFPU LREGs as needed by the compiler for register allocation. |
| `SFPCOMPC` (via `v_elseif`) | Complements the current condition code, enabling the "else" branch of a conditional. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST[idst0 * 32 + row]** | Source: input tile A. Read via `sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]`. Each iteration reads one row (16 elements). |
| **DEST[idst1 * 32 + row]** | Source: input tile B. Read via `sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]`. |
| **DEST[odst * 32 + row]** | Output: comparison result. Written via `sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result`. For EQ in binary_ng, `odst == idst0` (result overwrites input A's tile). |
| **LREG0-LREG3** | Temporary SFPU local registers used by the compiler to hold `in0`, `in1`, `in0_bits`, `in1_bits`, `result`, and intermediate values. Exact allocation is compiler-determined from the SFPI abstractions. |
| **LREG7** | May be used by the compiler for additional temporaries (e.g., the `in0_abs`/`in1_abs` values in the Blackhole variant). |
| **CC register** | Condition code register, set per-lane by `v_if`/`v_elseif` comparisons to guard conditional stores of 1.0f. |

### Address Mode Configuration

The init function `_llk_math_eltwise_binary_sfpu_init_<SfpuType::eq>()` configures the address mode via `eltwise_binary_sfpu_configure_addrmod<SfpuType::eq>()`.

For `SfpuType::eq`, the configuration is the default (no special-case branch):

| Field | ADDR_MOD_7 Value |
|-------|-----------------|
| `srca.incr` | 0 |
| `srcb.incr` | 0 |
| `dest.incr` | 0 |

This is the same on **both Wormhole B0 and Blackhole** -- the `eltwise_binary_sfpu_configure_addrmod` function is identical across architectures. The `ADDR_MOD_6` variant (with `dest.incr = 2`) is only configured for `mul_int32`, `mul_uint16`, `max`, `min`, and their integer variants -- not for `eq`.

Since `ADDR_MOD_7` has all increments set to zero, DEST address progression is handled entirely by `sfpi::dst_reg++` within the SFPU kernel (which increments by `SFP_DESTREG_STRIDE = 2` per iteration) and by the `TTI_SETRWC` instructions in the params dispatch (which advance by 16 rows between faces). The address mode itself does not auto-increment.

Note: The `SfpuType::eq` init also calls `sfpu::_init_sfpu_config_reg()` and `math::reset_counters(p_setrwc::SET_ABD_F)` to initialize the SFPU configuration register and reset the A/B/D/F read-write counters.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary_ng operation work? What is the architecture of binary_ng program factory, and how does it select between FPU and SFPU paths?"
   **Reason**: Needed to understand the overall binary_ng architecture and kernel file organization before diving into source code.
   **Key Findings**: Confirmed the `is_sfpu_op` flag drives path selection, identified the kernel file naming conventions (sfpu vs non-sfpu variants), and understood the `OpConfig` mechanism that maps `BinaryOpType` to specific FPU/SFPU operations with optional pre/post-processing.

2. [SFPU] **Query**: "How is eq_binary_tile implemented? What is the call chain from the compute API eq_binary_tile() down to the LLK and ckernel SFPU implementation?"
   **Reason**: Needed to trace the full SFPU call chain from the compute API through LLK dispatch to the core SFPU kernel for binary FP32 equality comparison.
   **Key Findings**: Confirmed the chain: `eq_binary_tile` -> `llk_math_eltwise_binary_sfpu_eq_fp32` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_binary_comp_fp32`. Identified that the core implementation lives in `ckernel_sfpu_binary_comp.h` with architecture-specific variants.

3. [SFPU] **Query**: "How is the binary SFPU equality operation (eq_binary, eq_fp32) implemented in the LLK layer? What is the call chain from llk_math_eltwise_binary_sfpu_eq_fp32 down to the ckernel_sfpu level?"
   **Reason**: Needed LLK-level details on the params dispatch mechanism, vector mode iteration, and face-level processing for binary SFPU operations.
   **Key Findings**: Confirmed that `_llk_math_eltwise_binary_sfpu_params_` handles face iteration (4 faces for VectorMode::RC), SETRWC-based DEST address advancement between faces, and that ITERATIONS=8 processes all rows within a face.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp`
   **Reason**: Needed to understand how `BinaryOpType::EQ` maps to specific SFPU init/op functions.
   **Key Information**: EQ with FLOAT32 maps to `eq_binary_tile_init()` / `eq_binary_tile()`. Non-FLOAT32 SFPU EQ decomposes to SUB + EQZ postprocess.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp`
   **Reason**: Needed to understand the `is_binary_sfpu_op` conditions for EQ.
   **Key Information**: EQ uses SFPU when `a == b && (a == FLOAT32 || a == INT32 || a == UINT32 || a == UINT16)`.

3. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: Needed to verify the SFPU intrinsic implementation for `eq_binary_tile`.
   **Key Information**: `eq_binary_tile` calls `llk_math_eltwise_binary_sfpu_eq_fp32<APPROX>()`, confirming it is an FP32-specific equality comparison at the LLK level.

### Confluence References
No Confluence pages were consulted for this analysis. The SFPI-based kernel uses high-level abstractions that did not require ISA-level instruction documentation.

### Glean References
No Glean searches were performed for this analysis.
