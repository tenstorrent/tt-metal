# MAXIMUM (binary_ng) Implementation Analysis

## Overview

The MAXIMUM operation computes the element-wise maximum of two input tensors: `c[i] = max(a[i], b[i])`. It is implemented as an SFPU-only operation within the `binary_ng` framework, which is the next-generation binary element-wise infrastructure in TTNN. The `binary_ng` framework provides a unified program factory that handles many binary operations through compile-time macro specialization, supporting broadcasting, sharding, and pre/post-activation fusion.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

## Path Selection: FPU vs SFPU

MAXIMUM is an **SFPU-only** operation. The path selection occurs in two places:

1. **`is_binary_sfpu_op()`** in `binary_ng_device_operation.cpp` (line 56-57): For `BinaryOpType::MAXIMUM`, this function unconditionally returns `true`, meaning the SFPU path is always selected regardless of input data types.

2. **`OpConfig` constructor** in `binary_ng_utils.cpp` (lines 309-315): When the operation is constructed with `std::in_place_type<OpConfig::SfpuBinaryOp>`, the `MAXIMUM` case sets `binary_op = SfpuBinaryOp::MAXIMUM`. If the FPU path were ever attempted (it cannot be for MAXIMUM), it would throw `"Unsupported binary op for FPU"`.

The program factory (line 530, 578) reads `operation_attributes.is_sfpu` and constructs `OpConfig` accordingly. For MAXIMUM, `is_sfpu` is always true, so the factory always selects SFPU kernel files and configures UnpackToDestFP32 mode.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `c.physical_volume() / tile_hw` (total output tiles) |
| **Loop structure** | Outer loop over `num_tiles` assigned to core; inner loop processes `num_tiles_per_cycle` (= 1) tiles per iteration |

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|---------------|---------------|
| **Logical shape** | Up to rank 6+ (collapsed for higher ranks) | Up to rank 6+ (broadcastable against A) |
| **Dimension convention** | [..., D, N, C, H, W] | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED (height/width/block) | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32 | Same as A (or any type -- MAXIMUM always uses SFPU) |

### Output Tensor

| Property | Output Tensor C |
|----------|----------------|
| **Logical shape** | Broadcast-compatible output of A and B |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Matches input or user-specified |

### Layout Transformations

No explicit tilize/untilize is performed within the kernels. All tensors are expected in TILE_LAYOUT. If data type casting is needed (e.g., a_dtype differs from c_dtype), a `TYPECAST` post-activation is appended automatically by the program factory (line 604-609).

## Data Flow Pattern

The data flow depends on whether B is a tensor or a scalar. For the **two-tensor** (no-broadcast) case with SFPU:

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader (RISCV_0) | DRAM/L1 (A buffer, B buffer) | CB c_0 (A tiles), CB c_1 (B tiles) | reserve_back, noc_async_read_page, push_back |
| 2 | Compute (RISCV_2) | CB c_0 (A), CB c_1 (B) | CB c_2 (output) | wait_front on c_0/c_1, copy_tile to DEST, BINARY_SFPU_OP, pack_tile, push_back c_2, pop_front c_0/c_1 |
| 3 | Writer (RISCV_1) | CB c_2 (output) | DRAM/L1 (C buffer) | wait_front c_2, noc_async_write_page, pop_front c_2 |

For the **scalar B** case: the writer kernel fills a single tile in CB c_1 with the scalar value, then only writes output. The reader reads only A tiles into CB c_0.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src_a / cb_pre_lhs | Input A staging | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_src_b / cb_pre_rhs | Input B staging | 2 tiles (interleaved, tensor B) or 1 tile (scalar B) or shard volume (sharded) | 1 tile | Double (interleaved) / Single (scalar/sharded) | Reader or Writer (scalar) | Compute | Program |
| c_2 | cb_out | Output staging | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Compute | Writer | Program |
| c_3 | cb_post_lhs | LHS intermediate (only if LHS activations present) | 1 tile | 1 tile | Single | Compute (preprocess) | Compute (main) | Block |
| c_4 | cb_post_rhs | RHS intermediate (only if RHS activations present) | 1 tile | 1 tile | Single | Compute (preprocess) | Compute (main) | Block |

For MAXIMUM with no pre/post activations, only c_0, c_1, and c_2 are used. c_3 and c_4 are created only when `PROCESS_LHS_ACTIVATIONS` or `PROCESS_RHS_ACTIVATIONS` macros expand to non-empty code.

## Pipeline Pattern Summary

In the interleaved (non-sharded) case, CBs c_0, c_1, and c_2 each have capacity = 2 tiles with block size = 1 tile, enabling **double-buffering** between the reader and compute, and between compute and writer. This allows the reader to fill the next tile while compute processes the current one.

In the sharded case, all tiles for the shard are pushed at once, so the CBs operate as **single-buffered** bulk transfers.

## Index Calculations

The reader kernel computes a multi-dimensional tile offset from a flat `start_tile_id` using the output tensor's shape dimensions (cND, D, N, C, Ht, Wt). The key index decomposition:

```
tiles_per_nd = D * tiles_per_d
tiles_per_d  = N * tiles_per_n
tiles_per_n  = C * HtWt
HtWt         = Ht * Wt
```

For input A, stride parameters (`nD_stride`, `d_stride`, `n_stride`, `c_stride`) encode broadcasting: if a dimension has size 1, its stride is 0 (because the condition `(aDim > 1)` evaluates to 0, zeroing the stride product). This means repeated access to the same tile when broadcasting.

For input B, the same stride-based mechanism applies via `nD_stride_b`, `d_stride_b`, `n_stride_b`, `c_stride_b`.

The writer uses `TensorAccessor` for mapping the flat output tile index to the physical DRAM page via `noc_async_write_page`.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile-by-tile reads with `noc_async_read_page` followed by `noc_async_read_barrier`. Each tile is read individually. The innermost loop iterates over width tiles (tw), then height tiles (th), then channels (c), batch (n), depth (d), and collapsed dimensions (nd). Broadcasting is achieved through stride zeroing.
- **Sharded**: Bulk `cb_reserve_back` / `cb_push_back` for the entire shard -- tiles are already in L1.

### Write Pattern
- **Interleaved**: Sequential tile-by-tile writes with `noc_async_write_page` / `noc_async_write_barrier`. Same nested loop structure as reader but over the output address space.
- **Sharded**: No explicit writes needed -- output CB is backed by the sharded output buffer in L1.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (prefers rectangular grid starting at (0,0)) |
| **Grid dimensions** | Device-dependent (`compute_with_storage_grid` or `worker_grid`) |
| **Total cores** | `grid.x * grid.y` (zero-start) or `all_device_cores.num_cores()` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two-group split: `core_group_1` gets `ceil(total_tiles / num_cores)` tiles, `core_group_2` gets the remainder. Cores outside both groups are assigned zero-work (noop). |

The `split_work_to_cores` utility divides `c_num_tiles` across available cores. For sharded tensors, the core grid is determined by the shard spec, and each core processes exactly its shard's tiles.

## Arguments

### Compile-Time Arguments

**Compute kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Number of output tiles produced per compute cycle (always 1) |

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs (A) | uint32_t[] | Tensor accessor compile-time args for input A |
| N+1..M | TensorAccessorArgs (B) | uint32_t[] | Tensor accessor compile-time args for input B |
| M+1 | has_sharding | uint32_t | 1 if any tensor is sharded, 0 otherwise |

**Writer kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs (C) | uint32_t[] | Tensor accessor compile-time args for output C |
| N+1 | has_sharding | uint32_t | 1 if any tensor is sharded, 0 otherwise |

### Runtime Arguments

**Reader kernel (21 args per core, two-tensor case):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input A buffer address |
| 1 | c_start_id | uint32_t | Starting output tile ID for this core |
| 2 | a_num_tiles | uint32_t | Number of A shard tiles (sharded) or 0 |
| 3 | c_num_tiles | uint32_t | Number of output tiles for this core |
| 4 | c_current_shard_width | uint32_t | Shard width in tiles (sharded) or 0 |
| 5 | nD_stride | uint32_t | A's collapsed-dim stride (0 if broadcasting) |
| 6 | d_stride | uint32_t | A's D-dim stride |
| 7 | n_stride | uint32_t | A's N-dim stride |
| 8 | c_stride | uint32_t | A's C-dim stride |
| 9 | cD | uint32_t | Output D dimension |
| 10 | cN | uint32_t | Output N dimension |
| 11 | cC | uint32_t | Output C dimension |
| 12 | cHt | uint32_t | Output height in tiles |
| 13 | cWt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Output collapsed dims > 5 |
| 15 | src_addr_b | uint32_t | Input B buffer address |
| 16 | nD_stride_b | uint32_t | B's collapsed-dim stride |
| 17 | d_stride_b | uint32_t | B's D-dim stride |
| 18 | n_stride_b | uint32_t | B's N-dim stride |
| 19 | c_stride_b | uint32_t | B's C-dim stride |
| 20 | b_num_tiles | uint32_t | Number of B shard tiles (sharded) or 0 |

**Compute kernel (4 args per core):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | c_num_tiles | uint32_t | Number of output tiles this core produces |
| 1 | freq | uint32_t | Broadcast frequency (1 for NONE/ROW, Wt for COL, Ht*Wt for SCALAR) |
| 2 | counter | uint32_t | Starting tile offset within the broadcast cycle |
| 3 | compute_scalar_value | uint32_t | Unused for MAXIMUM (set to 0) |

**Writer kernel (11 args per core, two-tensor case):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output C buffer address |
| 1 | c_start_id | uint32_t | Starting output tile ID |
| 2 | c_num_tiles | uint32_t | Number of output tiles |
| 3 | c_current_shard_width | uint32_t | Shard width in tiles (or 0) |
| 4 | cD | uint32_t | Output D dimension |
| 5 | cN | uint32_t | Output N dimension |
| 6 | cC | uint32_t | Output C dimension |
| 7 | cHt | uint32_t | Output height in tiles |
| 8 | cWt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Output collapsed dims |
| 10 | reserved | uint32_t | Set to 0 |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | RISCV_0 | NOC0 | DRAM/L1 (A, B buffers) | CB c_0, CB c_1 | Read A and B tiles via NoC |
| Compute | RISCV_2 | N/A | CB c_0, CB c_1 | CB c_2 | copy_tile to DEST, binary_max_tile SFPU op, pack_tile |
| Writer | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 (C buffer) | Write output tiles via NoC |

### Reader Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` (two-tensor, no broadcast case) |
| Assigned cores | All cores in `all_device_cores` grid |

**Key Logic:**
- Reads both input A (into CB c_0) and input B (into CB c_1) in the same kernel, one tile at a time
- Uses `TensorAccessor` for address computation with `noc_async_read_page`
- Multi-dimensional loop nest: nd -> d -> n -> c -> th -> tw, with stride-based broadcasting for each input independently
- For sharded inputs, simply does `cb_reserve_back` / `cb_push_back` on the entire shard volume (tiles already in L1)
- Conditional compilation via `SRC_SHARDED` and `SRC_SHARDED_B` defines controls whether each input uses NoC reads or direct L1 access
- **Synchronization**: Produces to CB c_0 and CB c_1 via `cb_reserve_back(onetile)` -> `noc_async_read_page` -> `noc_async_read_barrier()` -> `cb_push_back(onetile)`. The read barrier ensures data is in L1 before making it visible to compute.

### Compute Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` (no-broadcast case) |
| Assigned cores | All cores in `all_device_cores` grid |

**Key Logic:**
- For MAXIMUM, `BINARY_SFPU_INIT` expands to `binary_max_tile_init();` (or `binary_max_int32_tile_init()` / `binary_max_uint32_tile_init()` for integer types)
- `BINARY_SFPU_OP(i*2, i*2+1, i*2)` expands to `binary_max_tile(i*2, i*2+1, i*2)` -- computes max of DEST registers at positions i*2 (LHS) and i*2+1 (RHS), stores result at i*2
- The tile processing sequence per iteration:
  1. `cb_wait_front(cb_post_lhs, 1)` -- wait for LHS tile
  2. `cb_wait_front(cb_post_rhs, 1)` -- wait for RHS tile
  3. `cb_reserve_back(cb_out, 1)` -- reserve output space
  4. `tile_regs_acquire()` -- acquire DEST registers
  5. `copy_tile(cb_post_lhs, 0, 0)` -- unpack LHS to DEST[0]
  6. `copy_tile(cb_post_rhs, 0, 1)` -- unpack RHS to DEST[1]
  7. `binary_max_tile(0, 1, 0)` -- SFPU max: DEST[0] = max(DEST[0], DEST[1])
  8. `tile_regs_commit()` + `tile_regs_wait()` -- synchronize math/pack pipeline
  9. `pack_tile(0, cb_out)` -- pack result from DEST[0] to output CB
  10. `tile_regs_release()` -- release DEST registers
  11. `cb_push_back(cb_out, 1)` + `cb_pop_front` on both inputs
- When LHS/RHS/POST activations are present, `PREPROCESS` macro applies unary operations before the binary op, using intermediate CBs c_3/c_4
- For broadcast variants (SCALAR, COL), uses `eltwise_binary_sfpu.cpp` with freq/counter-based iteration to reuse the broadcast operand across multiple tiles
- **Synchronization**: Consumes CB c_0 and c_1 (or c_3/c_4 if activations); produces to CB c_2. Pattern: `cb_wait_front` -> process -> `cb_pop_front` / `cb_push_back`.

### Writer Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` (two-tensor case) |
| Assigned cores | All cores in `all_device_cores` grid |

**Key Logic:**
- Writes output tiles from CB c_2 to the output buffer using `TensorAccessor` and `noc_async_write_page`
- Same multi-dimensional loop nest as reader (nd -> d -> n -> c -> th -> tw) to map flat tile index to physical output location
- For sharded output (`DST_SHARDED`), the entire write loop is compiled out -- output is already in the correct L1 location via the sharded CB
- Handles shard width adjustments: when sharded, `dst_tile_offset` is adjusted by `(Wt - dst_shard_width)` at the end of each tile row
- **Synchronization**: Consumes CB c_2 via `cb_wait_front(cb_id_dst, onetile)` -> read + NoC write -> `noc_async_write_barrier()` -> `cb_pop_front(cb_id_dst, onetile)`.

## Implementation Notes

- **Program factory variants**: There is a single `BinaryNgDeviceOperation::ProgramFactory`. The factory handles all binary_ng operations (ADD, SUB, MUL, MAXIMUM, etc.) through compile-time macro specialization. The SFPU vs FPU path is selected via `operation_attributes.is_sfpu`. MAXIMUM always takes the SFPU path.

- **Type-based operation variants**: MAXIMUM supports BFLOAT16 (via `binary_max_tile`), FLOAT32 (via `binary_max_tile`), INT32 (via `binary_max_int32_tile`), and UINT32 (via `binary_max_uint32_tile`). The dispatch is in `get_sfpu_init_fn()` in `binary_ng_utils.cpp` (lines 442-448).

- **UnpackToDestFP32 mode**: For SFPU binary ops (excluding POWER), `UnpackToDestMode::UnpackToDestFp32` is enabled on CBs c_0, c_1, c_3, c_4 (program factory lines 741-745). This unpacks input data to FP32 precision in DEST registers regardless of the input data format, ensuring higher precision for the SFPU computation.

- **Broadcast type selection**: The `SubtileBroadcastType` determines which reader and compute kernels are selected. For `NONE` (equal shapes), uses `ReaderNoBcastNg` + `ComputeNoBcast`. For `SCALAR_A/B`, `COL_A/B`, `ROW_A/B`, and mixed `ROW_A_COL_B`/`ROW_B_COL_A`, different reader kernels and `ComputeBcast` are used. Row broadcast with BF16-only inputs can additionally trigger LLK-level broadcast optimization (but this is rare for MAXIMUM which is SFPU-only).

- **Sharding support and constraints**: Height, width, and block sharding are all supported. Native L1 sharding (avoiding tensor accessor) is used when: both inputs have the same shape and memory config, are in L1, grids match, and shards are evenly sized. Uneven shards or mismatched grids fall back to interleaved tensor accessor mode. The `is_native_L1_sharding()` function (lines 652-703 in utils) encodes these constraints.

- **FP32 dest accumulation**: Enabled when output format is UInt32, Int32, or Float32, or when both inputs are Float32, Int32, or UInt32 (program factory lines 727-731). This ensures the DEST accumulator operates at full 32-bit precision for these types.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng operation work? What are its program factory variants, SFPU vs FPU paths, and kernel files?"
   **Reason**: Needed to understand the overall architecture of the binary_ng operation before reading source code.
   **Key Findings**: Confirmed single ProgramFactory, SFPU/FPU path selection via `is_sfpu` flag, `SubtileBroadcastType`-based kernel selection, and kernel file naming conventions (`eltwise_binary_sfpu_*.cpp` for SFPU path).

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp`
   **Reason**: Needed to understand how MAXIMUM maps to SFPU init/op functions and how OpConfig is constructed.
   **Key Information**: MAXIMUM maps to `binary_max_tile_init()` / `binary_max_tile` (float), `binary_max_int32_tile_init()` / `binary_max_int32_tile` (INT32), or `binary_max_uint32_tile_init()` / `binary_max_uint32_tile` (UINT32).

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp`
   **Reason**: Needed to understand the `is_binary_sfpu_op` function to confirm MAXIMUM is always SFPU.
   **Key Information**: `BinaryOpType::MAXIMUM` returns `true` unconditionally in `is_binary_sfpu_op()`.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp` and `eltwise_utils_common.hpp`
   **Reason**: Needed to understand the PREPROCESS macro, activation macros, and BCAST_INPUT logic.
   **Key Information**: PREPROCESS conditionally applies unary activations to an input CB before the main binary operation. The macro system uses preprocessor token pasting to detect empty activation lists.
