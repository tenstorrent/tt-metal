# ADD (binary_ng) Implementation Analysis

## Overview

The ADD operation is one of several binary operations implemented through the `binary_ng` ("next generation") program factory. It computes `c = a + b` element-wise, supporting tensor-tensor addition, tensor-scalar addition, and various broadcasting modes (row, column, scalar, and mixed row-column). The `binary_ng` framework is a unified infrastructure that handles all binary element-wise operations (ADD, SUB, MUL, DIV, etc.) through a single configurable program factory, using compile-time defines to specialize kernel behavior.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

For ADD specifically:
- **FPU path** (default for BFLOAT16): Uses `FpuBinaryOp::ADD`, which maps to `add_tiles` via the FPU matrix engine.
- **SFPU path** (for FLOAT32, INT32, UINT32, UINT16): Uses `SfpuBinaryOp::ADD`, which maps to `add_binary_tile` (or `add_int_tile` for integer types) via the SFPU vector engine.

The SFPU path is selected when both operands share a dtype of FLOAT32, INT32, UINT32, or UINT16 (determined by `is_binary_sfpu_op` in `binary_ng_device_operation.cpp`, line 19-27).

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `c.physical_volume() / tile_hw` (total output tiles) |
| **Loop structure** | 6-deep nested loop over collapsed-ND, D, N, C, Ht, Wt dimensions; 1 tile processed per iteration of the innermost loop |

The compute kernel processes exactly `num_tiles_per_cycle = 1` output tile per read-compute-write cycle (compile-time arg 0, set at line 760 of the program factory).

## Tensor Format and Layout

### Input Tensor A

| Property | Value |
|----------|-------|
| **Logical shape** | Up to 6D: [..., D, N, C, H, W] |
| **Dimension convention** | Last 5 dims extracted as [D, N, C, Ht, Wt]; dims beyond 5 collapsed into `cND` |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED or SHARDED (HEIGHT, WIDTH, BLOCK) |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 |

### Input Tensor B (optional -- may be a scalar)

| Property | Value |
|----------|-------|
| **Logical shape** | Same rank as A (broadcast-compatible), or absent (scalar mode) |
| **Dimension convention** | Same as A |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as A for FPU path; BFLOAT16 default for scalar FPU; same-type for SFPU |

### Output Tensor C

| Property | Value |
|----------|-------|
| **Logical shape** | Broadcast-compatible output of A and B shapes |
| **Dimension convention** | Same as A |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Configurable; defaults to input dtype |

### Layout Transformations

No explicit tilize/untilize within the program factory. Inputs must already be in TILE_LAYOUT. The `invoke_binary_ng` entry point (in `binary.cpp`) converts inputs to TILE layout and BFLOAT16 if necessary before dispatching to the device operation.

## Data Flow Pattern

### Tensor-Tensor Mode (b is a tensor)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader (BRISC) | DRAM/L1 (a) + DRAM/L1 (b) | CB c_0 (a tiles) + CB c_1 (b tiles) | reserve_back, noc_async_read_page, push_back |
| 2 | Compute (TRISC) | CB c_0, CB c_1 | CB c_2 | wait_front, binary op, pack_tile, push_back, pop_front |
| 3 | Writer (NCRISC) | CB c_2 | DRAM/L1 (c) | wait_front, noc_async_write_page, pop_front |

### Tensor-Scalar Mode (b is a scalar)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader (BRISC) | DRAM/L1 (a) | CB c_0 | reserve_back, noc_async_read_page, push_back |
| 1b | Writer (NCRISC) | scalar arg | CB c_1 | fill_with_val, reserve_back, push_back (once) |
| 2 | Compute (TRISC) | CB c_0, CB c_1 | CB c_2 | wait_front(rhs, once), then loop: wait_front(lhs), binary op, push_back, pop_front(lhs) |
| 3 | Writer (NCRISC) | CB c_2 | DRAM/L1 (c) | wait_front, noc_async_write_page, pop_front |

Key design: In scalar mode, the writer kernel fills CB c_1 with the scalar value once (line 29-37 of `writer_interleaved_scalar.cpp`), then the compute kernel reads it once and keeps it resident for all tiles. The reader kernel only reads tensor A.

In tensor-tensor mode, the reader kernel reads BOTH tensors A and B (the `_ng` reader variants handle both inputs). The writer only writes the output.

## Circular Buffer Configuration

### Interleaved (non-sharded) Mode

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src_a | Input A tiles | 2 tiles | 1 tile | Double | Reader | Compute | Block |
| c_1 | cb_src_b | Input B tiles (or scalar) | 2 tiles (tensor) / 1 tile (scalar) | 1 tile | Double / Single | Reader or Writer | Compute | Block / Program |
| c_2 | cb_out | Output tiles | 2 tiles | 1 tile | Double | Compute | Writer | Block |
| c_3 | cb_lhs_interim | LHS activation intermediate | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_4 | cb_rhs_interim | RHS activation intermediate | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_5 | cb_bcast_a | Row broadcast buffer for A | 2 tiles | 1 tile | Double | Reader | Compute | Block |
| c_6 | cb_bcast_b | Row broadcast buffer for B | 2 tiles | 1 tile | Double | Reader | Compute | Block |

Notes:
- CB c_3 and c_4 are only created when LHS/RHS pre-activations are present (lines 644-670). For plain ADD, these are not used.
- CB c_5 and c_6 are only created for row broadcast types (ROW_A, ROW_B, ROW_A_COL_B, ROW_B_COL_A) (lines 672-679).
- In scalar mode, CB c_1 capacity is 1 tile (the scalar is filled once and never popped until the end).
- In sharded mode, CB capacities are set to the shard volume (number of tiles per shard) instead of 2.

### Sharded Mode

When sharding is active, CB capacities become the shard tile count (e.g., `a_num_tiles_per_shard`, `b_num_tiles_per_shard`, `c_num_tiles_per_shard`). The buffers are backed directly by the sharded tensor's L1 memory via `UpdateDynamicCircularBufferAddress`.

## Pipeline Pattern Summary

For interleaved mode:
- **CB c_0 (Input A)**: Double-buffered (capacity=2, block=1) -- allows overlap of read and compute.
- **CB c_1 (Input B)**: Double-buffered for tensor mode; Single-buffered for scalar mode (filled once).
- **CB c_2 (Output)**: Double-buffered (capacity=2, block=1) -- allows overlap of compute and write.
- **CB c_3, c_4**: Single-buffered intermediates (only when activations present).

For sharded mode, all data is resident in L1 from the start, so there is no pipelining -- the reader simply marks the entire shard as available.

## Index Calculations

The reader and writer kernels use a 6-level nested dimension traversal (cND, D, N, C, Ht, Wt) to map output tile IDs to input tile positions. This supports arbitrary broadcasting.

### Reader Input Tile Offset (from `reader_interleaved_no_bcast.cpp`)

The key insight is that input A and B can have different shapes from the output. Broadcasting is implemented through **stride compression**: each dimension's stride is set to 0 when the input has size 1 in that dimension (achieved by multiplying by `(dim > 1)` in the program factory, lines 395-399).

```
tile_offset = start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride + start_th * Wt
```

Where strides are:
- `nD_stride = Ht * Wt * C * N * D * (nD > 1)` -- zero when only 1 collapsed dim
- `d_stride = Ht * Wt * C * N * (D > 1)` -- zero when D=1 (broadcast)
- `n_stride = Ht * Wt * C * (N > 1)`
- `c_stride = Ht * Wt * (C > 1)`

The `start_tile_id` is decomposed into (nd, d, n, c, th, tw) coordinates using modular arithmetic against the OUTPUT shape dimensions, then those coordinates are used with INPUT strides to compute the actual read position.

### Writer Output Tile Offset

For interleaved non-sharded mode, the writer simply increments a linear tile counter (`dst_tile_offset + num_tiles_written`). For sharded mode, the offset accounts for shard position: `c_start_id = (core_index / num_shards_per_width) * (c_shard_height * cWt) + (core_index % num_shards_per_width) * c_shard_width`.

### TensorAccessor Usage

Both reader and writer kernels use `TensorAccessor` and `TensorAccessorArgs` for physical address resolution. The TensorAccessor maps a logical tile index to the physical DRAM/L1 address, accounting for interleaved bank distribution. Compile-time args encode the accessor configuration, and common runtime args encode the tensor shape for bank mapping.

## Memory Access Patterns

### Read Pattern

- **Interleaved**: Sequential tile reads within each row (tw loop), then advances rows (th), channels (c), batches (n), etc. Each tile read uses `noc_async_read_page` followed by `noc_async_read_barrier` (one tile at a time). The pattern is effectively row-major within each 2D slice.
- **Sharded**: No reads needed -- the reader kernel simply marks the shard as available via `cb_reserve_back / cb_push_back` for the entire shard at once.

### Write Pattern

- **Interleaved**: Sequential tile writes mirroring the read order. Each tile is written with `noc_async_write_page` + `noc_async_write_barrier`.
- **Sharded**: No writes needed -- output is already in L1 shard memory.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (worker grid) |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` or shard grid |
| **Total cores** | `compute_with_storage_grid.x * compute_with_storage_grid.y` (zero-start) or `all_device_cores.num_cores()` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two-group split: group 1 gets `ceil(total_tiles / num_cores)` tiles, group 2 gets `floor(total_tiles / num_cores)` tiles |

### Work Splitting Logic

1. **Sharded mode**: Each core processes its shard. The shard shape determines the tile count per core. Edge cores may have smaller shards (handled by `ShardShapeGenerator`).

2. **Interleaved, zero-start grid** (single rectangular grid starting at (0,0)): Uses `split_work_to_cores(compute_with_storage_grid, c_num_tiles, row_major)` which divides tiles into two groups for balanced distribution.

3. **Interleaved, arbitrary grid**: Uses `split_work_to_cores(all_device_cores, c_num_tiles, row_major)` for the general case.

Cores not assigned any tiles receive zero-filled runtime args and effectively no-op (lines 314-317).

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs (A) | uint32_t[] | Bank mapping and accessor config for tensor A |
| N+1..M | TensorAccessorArgs (B) | uint32_t[] | Bank mapping and accessor config for tensor B |
| M+1 | has_sharding | uint32_t | 1 if any tensor is sharded, 0 otherwise |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs (C) | uint32_t[] | Bank mapping and accessor config for output tensor C |
| N+1 | has_sharding | uint32_t | 1 if any tensor is sharded, 0 otherwise |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 -- tiles produced per compute cycle |

### Runtime Arguments

#### Reader Kernel (21 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Base address of tensor A buffer |
| 1 | start_tile_id | uint32_t | Starting output tile ID for this core (c_start_id) |
| 2 | src_num_tiles | uint32_t | Number of A tiles in shard (sharded only, else 0) |
| 3 | dst_num_tiles | uint32_t | Number of output tiles this core processes (c_num_tiles) |
| 4 | dst_shard_width | uint32_t | Shard width in tiles (sharded only, else 0) |
| 5 | nD_stride | uint32_t | A's collapsed >5D stride (0 if dim=1) |
| 6 | d_stride | uint32_t | A's D-dimension stride (0 if D=1) |
| 7 | n_stride | uint32_t | A's N-dimension stride (0 if N=1) |
| 8 | c_stride | uint32_t | A's C-dimension stride (0 if C=1) |
| 9 | D | uint32_t | Output D dimension |
| 10 | N | uint32_t | Output N dimension |
| 11 | C | uint32_t | Output C dimension |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Output collapsed dims > 5 |
| 15 | src_addr_b | uint32_t | Base address of tensor B buffer (0 if scalar) |
| 16 | nD_stride_b | uint32_t | B's collapsed >5D stride |
| 17 | d_stride_b | uint32_t | B's D-dimension stride |
| 18 | n_stride_b | uint32_t | B's N-dimension stride |
| 19 | c_stride_b | uint32_t | B's C-dimension stride |
| 20 | src_num_tiles_b | uint32_t | Number of B tiles in shard (sharded only) |

#### Writer Kernel -- Tensor-Tensor Mode (11 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Base address of output buffer C |
| 1 | start_tile_id | uint32_t | Starting output tile ID (c_start_id) |
| 2 | dst_num_tiles | uint32_t | Number of output tiles for this core |
| 3 | dst_shard_width | uint32_t | Shard width in tiles |
| 4 | D | uint32_t | Output D dimension |
| 5 | N | uint32_t | Output N dimension |
| 6 | C | uint32_t | Output C dimension |
| 7 | Ht | uint32_t | Output height in tiles |
| 8 | Wt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Output collapsed dims > 5 |
| 10 | (reserved) | uint32_t | Set to 0 |

#### Writer Kernel -- Scalar Mode (11 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar | uint32_t | Scalar value packed as bfloat16x2 or float32 bits |
| 1 | dst_addr | uint32_t | Base address of output buffer C |
| 2 | start_tile_id | uint32_t | Starting output tile ID |
| 3 | dst_num_tiles | uint32_t | Number of output tiles |
| 4 | dst_shard_width | uint32_t | Shard width in tiles |
| 5 | D | uint32_t | Output D dimension |
| 6 | N | uint32_t | Output N dimension |
| 7 | C | uint32_t | Output C dimension |
| 8 | Ht | uint32_t | Output height in tiles |
| 9 | Wt | uint32_t | Output width in tiles |
| 10 | cND | uint32_t | Output collapsed dims > 5 |

#### Compute Kernel (4 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total tiles to process on this core |
| 1 | freq | uint32_t | Broadcast frequency (1 for no-bcast, Ht*Wt for scalar, Wt for col) |
| 2 | counter | uint32_t | Initial broadcast counter (starting offset within broadcast cycle) |
| 3 | compute_scalar_value | uint32_t | Quantization zero point (0 for non-quant ops like ADD) |

## Kernel Implementations

### FPU Path (BFLOAT16 default)

#### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 (A, B) | CB c_0, CB c_1 | Read A and B tiles via noc_async_read_page |

- **File (tensor-tensor)**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`
- **File (scalar, reads only A)**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Uses TensorAccessor for address resolution. Traverses 6 nested dimension loops matching output tile order. For broadcast variants (row, col, scalar), different reader kernels apply tile duplication (fill_tile_with_first_row, fill_tile_with_first_column, fill_tile_with_first_element).

#### Compute Kernel (FPU, no-broadcast)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | TRISC (RISCV_2) | N/A | CB c_0, CB c_1 | CB c_2 | `add_tiles` via FPU |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_no_bcast.cpp`
- **Key Logic**: For ADD, the define `BINARY_OP` resolves to `add_tiles` and `BINARY_OP_TYPE` to `EltwiseBinaryType::ELWADD`. The kernel calls `binary_op_init_common`, then loops tile-by-tile: `cb_wait_front` on both inputs, `tile_regs_acquire`, `BINARY_OP(cb_lhs, cb_rhs, 0, 0, 0)`, optional `PROCESS_POST_ACTIVATIONS`, `pack_tile`, `cb_push_back`, `cb_pop_front`.

#### Compute Kernel (FPU, scalar)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_scalar.cpp`
- **Key Logic**: Same as no-broadcast but waits for RHS only once (before the loop). The scalar tile stays resident in CB c_1 and is reused for all LHS tiles. RHS is popped only after the loop ends.

#### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 (C) | Write output tiles via noc_async_write_page |

- **File (tensor-tensor)**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
- **File (scalar, also fills B)**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp`
- **Key Logic**: In scalar mode, the writer first fills CB c_1 with the packed scalar value using `fill_with_val` / `fill_with_val_bfloat16`, then writes output tiles. In tensor-tensor mode, the writer only handles output.

### SFPU Path (FLOAT32 / INT32 / UINT32 / UINT16)

#### Compute Kernel (SFPU, no-broadcast)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | TRISC (RISCV_2) | N/A | CB c_0, CB c_1 | CB c_2 | `add_binary_tile` or `add_int_tile` via SFPU |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`
- **Key Logic**: SFPU path copies both input tiles to dest registers using `copy_tile`, then calls `BINARY_SFPU_OP(dst_a, dst_b, dst_out)`. For FLOAT32 ADD, this resolves to `add_binary_tile` with init `add_binary_tile_init()`. For INT32, it resolves to `add_int_tile<DataFormat::Int32>`. Uses `UnpackToDestMode::UnpackToDestFp32` for all non-POWER SFPU ops.

The reader and writer kernels are identical between FPU and SFPU paths.

## Implementation Notes

### Broadcast Type Selection

The `SubtileBroadcastType` enum (determined by `get_subtile_broadcast_type`) controls which kernel variants are selected. For ADD, the relevant cases are:
- **NONE**: Equal shapes, no broadcast. Uses `ComputeNoBcast`.
- **SCALAR_A/B**: One operand is 1x1. Uses `ComputeBcast` with `bcast_input=0` or `1`.
- **ROW_A/B**: One operand has H=1. Uses `ComputeNoBcast` (the reader handles the row duplication). When all dtypes are BFLOAT16, an LLK-level broadcast optimization is applied (`ComputeRowBcastNg`).
- **COL_A/B**: One operand has W=1. Uses `ComputeBcast`.
- **ROW_A_COL_B / ROW_B_COL_A**: Mixed broadcast. Uses `ComputeBcast`, or `ComputeRowColBcastNg` when all BFLOAT16.

### Program Caching

The operation uses `override_runtime_arguments` to update tensor addresses and tile counts without recompiling kernels. The program hash includes op type, dtypes, memory config, broadcast type, activations, and compute config -- but NOT logical shapes or scalar values, enabling cache hits across different sizes.

### Sharding Support

Native L1 sharding is supported only when:
1. Both inputs have the same shape and memory config (no broadcasting)
2. Neither input nor output uses DRAM
3. Shard grids are identical across all tensors
4. No uneven sharding (all shards divide evenly into the tensor)

When these conditions are not met, the operation falls back to the interleaved path (using TensorAccessor for address resolution) even if tensors are sharded.

### Stride-Based Broadcasting

Broadcasting is elegantly implemented by setting dimension strides to 0 when the input size is 1. For example, if tensor A has shape [1, 1, 1, 32, 32] and B has [4, 8, 16, 32, 32], A's strides become `[0, 0, 0, Wt, 1]` (effectively `nD_stride=0, d_stride=0, n_stride=0, c_stride=0`), causing the reader to re-read the same A tile for every (d, n, c) combination.

### Where Operation Integration

The binary_ng framework also handles WHERE operations (conditional selection), which use specialized compute kernels (`eltwise_where_*`) and fill operations for scalar branches. For plain ADD, this path is not taken.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng operation work? What is its architecture, what kernels does it use, and how does it handle different subtypes?"
   **Reason**: Initial reconnaissance to understand the overall architecture and locate kernel files.
   **Key Findings**: binary_ng uses a unified ProgramFactory with SubtileBroadcastType-driven kernel selection. Kernels are in `kernels/` and `kernels_ng/` subdirectories. The operation supports FPU and SFPU paths, program caching, and multiple broadcast modes.

### Documentation References

1. **Source**: `binary_ng_utils.hpp` and `binary_ng_utils.cpp`
   **Reason**: Understanding how BinaryOpType::ADD maps to kernel defines and SFPU/FPU functions.
   **Key Information**: ADD maps to `FpuBinaryOp::ADD` (FPU: `add_tiles`) or `SfpuBinaryOp::ADD` (SFPU: `add_binary_tile` / `add_int_tile`).

2. **Source**: `binary_ng_device_operation.hpp` and `binary_ng_device_operation.cpp`
   **Reason**: Understanding SubtileBroadcastType enum and SFPU path selection criteria.
   **Key Information**: SFPU is used for ADD when both operands are FLOAT32, INT32, UINT32, or UINT16. The operation_attributes_t structure holds all configuration.

3. **Source**: Kernel files in `kernels/compute/` and `kernels_ng/dataflow/`
   **Reason**: Understanding the actual data flow, CB usage, and compute logic.
   **Key Information**: FPU path uses `binary_op_init_common` + `BINARY_OP` macro; SFPU path uses `copy_tile` to dest + `BINARY_SFPU_OP`. Reader kernels use TensorAccessor for address resolution with stride-based broadcasting.

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to. The ADD binary_ng operation has two SFPU sub-paths depending on data type: (1) a **floating-point path** using `add_binary_tile` for FLOAT32, and (2) an **integer path** using `add_int_tile<DataFormat>` for INT32, UINT32, and UINT16.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` (float), `tt_metal/hw/inc/api/compute/add_int_sfpu.h` (integer) |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` (float), `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_add_int.h` (integer) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_binary.h` (float), `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_add_int.h` (integer) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

**Floating-point path (FLOAT32)**:
1. The compute kernel calls `add_binary_tile(i*2, i*2+1, i*2)` (defined in `eltwise_binary_sfpu.h`), which gates the call behind the `MATH(...)` macro ensuring it only executes on TRISC_MATH.
2. This calls `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>(idst0, idst1, odst)` (in `llk_math_eltwise_binary_sfpu_binop.h`), which forwards to `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>()` passing `calculate_sfpu_binary<APPROX, ADD, 8, false>` as the callable.
3. `_llk_math_eltwise_binary_sfpu_params_` (in `llk_math_eltwise_binary_sfpu_params.h`) sets up the DST write address, stalls SFPU until math completes, then iterates over 4 tile faces (in RC mode), calling the SFPU function once per face and advancing the DST pointer by 16 rows between faces via `TTI_SETRWC`.
4. `calculate_sfpu_binary` (in the metal-repo `ckernel_sfpu_binary.h`) is a thin wrapper that delegates to `_calculate_sfpu_binary_<APPROX, ADD, 8>` (in `tt_llk/.../ckernel_sfpu_binary.h`), which executes the core SFPU add loop.

**Integer path (INT32/UINT32/UINT16)**:
1. The compute kernel calls `add_int_tile<DataFormat::Int32>(i*2, i*2+1, i*2)` (defined in `add_int_sfpu.h`).
2. This calls `llk_math_eltwise_binary_sfpu_add_int<APPROX, 8, DataFormat::Int32, false>()` (in `llk_math_eltwise_binary_sfpu_add_int.h`), which forwards to `_llk_math_eltwise_binary_sfpu_params_` passing `_add_int_<APPROX, 8, INSTRUCTION_MODE, false>` as the callable.
3. The params dispatch and face iteration are identical to the float path.
4. `_add_int_` (in `tt_llk/.../ckernel_sfpu_add_int.h`) uses raw `TT_SFPLOAD` / `TTI_SFPIADD` / `TT_SFPSTORE` instructions to perform integer addition.

### Annotated SFPU Kernel Source

#### Floating-Point Path

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
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr std::uint32_t dst_tile_size_sfpi = 32;
        sfpi::vFloat in0                           = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // Load row from input tile 0
        sfpi::vFloat in1                           = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // Load row from input tile 1
        sfpi::vFloat result                        = 0.0f;

        if constexpr (BINOP == BinaryOp::ADD)
        {
            result = in0 + in1; // SFPU vector float add: compiles to SFPADDI or SFPMAD
        }
        // SUB, MUL, DIV, RSUB, POW, XLOGY branches elided -- not taken for ADD

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // Store result row to output tile
        sfpi::dst_reg++; // Advance DST row pointer by 1 (SFP_DESTREG_STRIDE)
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{ // For ADD: no initialization needed -- ADD has no special LUT or reciprocal setup
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW)
    {
        _init_sfpu_reciprocal_<false>();
    }
    else if constexpr (BINOP == BinaryOp::XLOGY)
    {
        _init_log_<APPROXIMATION_MODE>();
    }
    // ADD, SUB, MUL, RSUB: no init required
}
```

#### Integer Path

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_add_int.h
// (Blackhole version is identical)

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _add_int_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{ // For Int32: INSTRUCTION_MODE=INT32 (instr_mod=4), SIGN_MAGNITUDE_FORMAT=false
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // sfpload_instr_mod selects data interpretation; INT32=4 for 2's complement 32-bit integers
    constexpr int sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? INT32_2S_COMP : to_underlying(INSTRUCTION_MODE);

    constexpr std::uint32_t dst_tile_size = 64; // Raw Dest row stride (not halved like SFPI)

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, dst_index_in0 * dst_tile_size); // Load operand A into LREG0
        TT_SFPLOAD(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_3, dst_index_in1 * dst_tile_size); // Load operand B into LREG1
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4); // Integer ADD: LREG0 = LREG0 + LREG1; mod=4 (32-bit 2's complement)
        TT_SFPSTORE(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, dst_index_out * dst_tile_size); // Store result from LREG0 to Dest
        sfpi::dst_reg++; // Advance row pointer
    }
}
```

#### Parameters Dispatch (shared by both paths)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h
// (Blackhole version is identical)

template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_binary_sfpu_params_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_out,
    int vector_mode = static_cast<int>(VectorMode::RC),
    Args&&... args)
{
    LLK_ASSERT((dst_index_in0 < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "dst_index_in0 exceeds max dest tiles");
    LLK_ASSERT((dst_index_in1 < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "dst_index_in1 exceeds max dest tiles");
    LLK_ASSERT((dst_index_out < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "dst_index_out exceeds max dest tiles");

    _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(0); // Set DST write addr to tile 0, stall SFPU until math done

    VectorMode mode = static_cast<VectorMode>(vector_mode);

    if (mode == VectorMode::R)
    {
        // Row vector: 2 faces (top-left, top-right), skip bottom 2
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_out, std::forward<Args>(args)...);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D); // Advance DST addr by 8 rows
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D); // +8 more = 16 rows per face
        }
        // Skip the bottom 2 faces (4 x 8-row increments)
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }
    else if (mode == VectorMode::C)
    {
        // Column vector: 2 faces (top-left, bottom-left), each full 16 rows
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_out, std::forward<Args>(args)...);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    }
    else if (mode == VectorMode::RC)
    {
        // Full tile: all 4 faces, 8 rows per SFPU call (ITERATIONS=8), 2 SETRWC per face = 16-row advance
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++)
        {
            std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_out, std::forward<Args>(args)...);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D); // +8 rows
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D); // +8 rows = 16 total per face
        }
    }
    else
    {
        std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_out, std::forward<Args>(args)...);
    }
    _llk_math_eltwise_binary_sfpu_done_(); // Clear DST addr, wait for SFPU config completion
}
```

### SFPU Instructions Used

**Floating-point path (`_calculate_sfpu_binary_` with ADD)**:

| Instruction / Intrinsic | Description |
|-------------------------|-------------|
| `sfpi::dst_reg[idx]` (load) | SFPLOAD -- loads a vector of 32 float elements from a DST register row into an SFPU LREG. The index selects the row within the tile. |
| `in0 + in1` (vFloat operator+) | Compiles to SFPMAD or SFPADDI -- performs element-wise floating-point addition across the 32-wide SIMD lane. The SFPI compiler selects the optimal instruction. |
| `sfpi::dst_reg[idx] = result` (store) | SFPSTORE -- stores the 32-wide vector result from an SFPU LREG back to a DST register row. |
| `sfpi::dst_reg++` | SFPINCRWC (implicit) -- increments the SFPU's internal DST row counter by SFP_DESTREG_STRIDE (=2), moving to the next pair of rows. |
| `TTI_SETRWC` | SETRWC -- sets/increments the read-write counter for the DST register address. Used between faces to advance by 8 rows (2 calls of 8 = 16 rows per face). |
| `TTI_STALLWAIT` | STALLWAIT -- stalls the SFPU pipeline until the FPU math engine completes, ensuring DST register coherency before SFPU reads. |

**Integer path (`_add_int_`)**:

| Instruction / Intrinsic | Description |
|-------------------------|-------------|
| `TT_SFPLOAD(LREG, mod, addr_mod, offset)` | SFPLOAD -- loads a vector from DST into the specified LREG. The `mod` parameter (4 = INT32 2's complement) controls data interpretation. |
| `TTI_SFPIADD(imm, src, dst, mod)` | SFPIADD -- integer addition: `dst = dst + src`. The `mod=4` selects 32-bit 2's complement mode. The immediate operand is 0 (not used for register-register add). |
| `TT_SFPSTORE(LREG, mod, addr_mod, offset)` | SFPSTORE -- stores the LREG contents back to DST at the specified offset. Same `mod` as SFPLOAD for consistent data format. |
| `sfpi::dst_reg++` | Advances the implicit DST row counter. |

### SFPU Register Usage

**Floating-point path**:
- **DST register tiles**: Two input tiles are loaded into adjacent DST tile slots by `copy_tile` before the SFPU kernel runs. With `num_tiles_per_cycle=1`, tile A occupies DST slot `i*2` (=0) and tile B occupies DST slot `i*2+1` (=1). The output overwrites DST slot `i*2` (=0).
- **SFPI vFloat registers (LREGs)**: `in0`, `in1`, and `result` are `sfpi::vFloat` variables that the SFPI compiler maps to LREG0-LREG3. For a simple ADD, two LREGs are needed for inputs and one for the result (which may alias one of the inputs in the compiler's register allocation).
- **DST row addressing**: Each tile occupies 32 SFPI-rows (64 raw DST rows / SFP_DESTREG_STRIDE=2). The `dst_tile_size_sfpi=32` constant is multiplied by the tile index to compute the row offset. The SFPU processes 8 rows per `_calculate_sfpu_binary_` call (ITERATIONS=8), and the params dispatch calls it 4 times (once per face) for a total of 32 rows = 1 full tile.

**Integer path**:
- **LREGs**: LREG0 holds operand A and the result; LREG1 holds operand B. Only two of the four available LREGs are used.
- **DST row addressing**: Uses `dst_tile_size=64` (raw DST stride, not halved) for offset calculation. The SFPLOAD/SFPSTORE instructions address DST directly.
- **ADDR_MOD_3**: Used for all load/store operations in the integer path. This address modifier likely provides no auto-increment (the explicit `dst_reg++` handles row advancement).

### Address Mode Configuration

The binary SFPU init function (`_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>`) configures the following address mode:

**ADDR_MOD_7** (used for all binary SFPU ops including ADD):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
```
All increments are zero. The SFPU binary add kernel manages DST addressing explicitly through `sfpi::dst_reg++` (which increments by SFP_DESTREG_STRIDE) and `TTI_SETRWC` calls between faces, rather than relying on hardware auto-increment.

**ADDR_MOD_6** is configured only for mul_int32, mul_uint16, max, min, max_int32, min_int32, max_uint32, min_uint32 operations (with `.dest.incr = 2`). It is NOT configured for ADD.

This configuration is identical across Wormhole B0 and Blackhole -- both architectures use the same `eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>()` function with the same ADDR_MOD_7 values.

The init also calls:
- `sfpu::_init_sfpu_config_reg()` -- initializes the SFPU configuration register.
- `math::reset_counters(p_setrwc::SET_ABD_F)` -- resets all read-write counters (A, B, D, and F) to their initial state.

For ADD specifically, `_sfpu_binary_init_()` is a no-op since ADD requires no LUT or reciprocal initialization (those are only needed for DIV, POW, and XLOGY).

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the binary SFPU add operation work? What is the call chain from add_binary_tile through the LLK layers to the core SFPU implementation?"
   **Reason**: Needed to understand the full abstraction layer hierarchy and locate the core SFPU implementation files for the binary add operation.
   **Key Findings**: Confirmed the call chain: `add_binary_tile` -> `llk_math_eltwise_binary_sfpu_binop<APPROX, ADD>` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_sfpu_binary` -> `_calculate_sfpu_binary_`. The core implementation lives in architecture-specific `ckernel_sfpu_binary.h` files within the tt_llk submodule.

2. **Query**: "How does add_binary_tile work? What is the call chain from the compute API through llk_math_eltwise_binary_sfpu down to the ckernel_sfpu_add implementation?"
   **Reason**: Needed detailed information about the LLK dispatch layer, VectorMode face iteration, and SETRWC instructions used in the params dispatch function.
   **Key Findings**: The params dispatch iterates over 4 faces in RC mode, calling the SFPU function once per face. Between faces, two `TTI_SETRWC` instructions advance the DST pointer by 16 rows. The init function configures ADDR_MOD_7 with zero increments. For ADD, `_sfpu_binary_init_` is a no-op.

### Confluence References
Not consulted for this analysis -- the ADD operation uses straightforward SFPU instructions (load, add, store) that are well-documented through the source code and DeepWiki.

### Glean References
Not consulted for this analysis.
