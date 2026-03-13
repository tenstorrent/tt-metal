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

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng operation work? What is the binary_ng program factory structure, how does it select between FPU and SFPU paths, and what kernels does it use?"
   **Reason**: Needed initial architectural understanding of the binary_ng operation framework before reading source code.
   **Key Findings**: The `is_sfpu` flag controls path selection. SFPU kernels are located under `kernels/compute/eltwise_binary_sfpu*.cpp`. The `OpConfig` class maps `BinaryOpType` to SFPU init/op function pairs. Reader reads both inputs when B is a tensor.

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
