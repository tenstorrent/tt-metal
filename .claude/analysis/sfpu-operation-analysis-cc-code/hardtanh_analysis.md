# HARDTANH Implementation Analysis

## Overview

HARDTANH is a unary element-wise activation function that clamps each element of the input tensor to a specified range `[min_val, max_val]`. Formally: `hardtanh(x) = max_val if x > max_val, min_val if x < min_val, else x`. This is equivalent to `clamp(x, min_val, max_val)`.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Path Selection: FPU vs SFPU

HARDTANH is exclusively an SFPU operation. There is no FPU alternative path. In `get_compute_kernel_path()` (line 958 of `unary_op_utils.cpp`), HARDTANH falls through to the `default` case which returns `"eltwise_sfpu.cpp"`. The macro define `SFPU_OP_HARDTANH_INCLUDE` (line 94 of `unary_op_utils.cpp`) gates the inclusion of `api/compute/eltwise_unary/hardtanh.h` in the compute kernel. The operation is parametrized (returns `true` from `is_parametrized_type()` at line 96 of `unary_op_utils.hpp`), requiring two float parameters (min_val and max_val) that are baked into the `SFPU_OP_CHAIN_0` macro as hex-encoded `uint32_t` literals.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `input.buffer()->num_pages()` |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles (always 1) |

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | Arbitrary (any rank) | Same as input |
| **Dimension convention** | N/A (element-wise) | N/A |
| **Tensor layout** | TILE_LAYOUT (or ROW_MAJOR) | Same as input |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32 | Same as input |

### Layout Transformations

None. The operation is purely element-wise and preserves the input layout, shape, and data type.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `noc_async_read_barrier`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_reserve_back(c_2, 1)`, `cb_wait_front(c_0, 1)`, `copy_tile`, SFPU op chain, `pack_tile`, `cb_pop_front(c_0, 1)`, `cb_push_back(c_2, 1)` |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `noc_async_writes_flushed`, `cb_pop_front(c_2, 1)` |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | src0 | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | output | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

Note: CB c_1 (tmp0) is NOT allocated for HARDTANH. It is only created for HARDSHRINK, CBRT, or LOGIT operations.

## Pipeline Pattern Summary

Both CB c_0 and CB c_2 have capacity = 2 * block_size, enabling double-buffering. The reader can write the next tile into CB c_0 while compute processes the current tile, and compute can write to CB c_2 while the writer drains the previous tile. This allows overlap across all three pipeline stages.

## Index Calculations

The reader and writer use `TensorAccessor` to map a linear page index to the physical memory address. Each core is assigned a contiguous range of page indices starting from `start_id` (a runtime argument). The linear page index `i` is passed to `noc_async_read_page(i, s, l1_write_addr)` and `noc_async_write_page(i, s, l1_read_addr)`, where the `TensorAccessor` object `s` handles bank mapping (interleaved pages distributed across DRAM banks or L1 banks).

## Memory Access Patterns

### Read Pattern
Sequential page access. Each core reads a contiguous range of pages `[start_id, start_id + num_pages)` from the source buffer. Pages are read one at a time with a barrier after each read (`noc_async_read_barrier`).

### Write Pattern
Sequential page access. Each core writes a contiguous range of pages `[start_id, start_id + num_pages)` to the destination buffer. Pages are written one at a time with a flush after each write (`noc_async_writes_flushed`), and a final write barrier at the end.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size` (device-dependent) |
| **Total cores** | Determined by `split_work_to_cores()` |
| **Work per core** | `num_pages / num_cores` tiles (group 1 gets ceil, group 2 gets floor) |
| **Load balancing** | Two-group split: core_group_1 gets `num_pages_per_core_group_1` tiles, core_group_2 gets `num_pages_per_core_group_2` tiles |

Core linearization: `core = {i / num_cores_y, i % num_cores_y}` (column-major ordering within the grid).

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Tensor accessor parameters for source buffer (bank mapping, page size, etc.) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer index (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Tensor accessor parameters for destination buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks to process on this core |
| 1 | per_core_block_size | uint32_t | Tiles per block (always 1 for this factory) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM/L1 address |
| 1 | num_pages | uint32_t | Number of pages (tiles) this core processes |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM/L1 address |
| 1 | num_pages | uint32_t | Number of pages (tiles) this core processes |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Not used for HARDTANH (remains 0) |
| 1 | packed_scalar2 | uint32_t | Not used for HARDTANH (remains 0) |

Note: HARDTANH does not match any of the special cases in the runtime arg packing switch statement (lines 128-152 of the program factory). The min_val and max_val parameters are instead embedded directly into the `SFPU_OP_CHAIN_0` macro as compile-time hex literals via `get_op_init_and_func_parameterized()` (line 525: `hardtanh_tile({idst}, {param0_hex}u, {param1_hex}u)`).

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 src_buffer | CB c_0 | Read tiles via TensorAccessor |
| Compute | TRISC (math) | N/A | CB c_0 | CB c_2 | SFPU hardtanh (clamp to [min, max]) |
| Writer | BRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 dst_buffer | Write tiles via TensorAccessor |

### Reader Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` |
| **Assigned cores** | all_cores (both core_group_1 and core_group_2) |

**Key Logic**:
- Iterates from `start_id` to `start_id + num_pages`, reading one page per iteration
- Uses `TensorAccessor` constructed from compile-time args for bank-aware address resolution
- Gets CB page size dynamically from `get_local_cb_interface(cb_id_in0).fifo_page_size`
- Supports both forward (`start_id` to `end_id`) and backward (`BACKWARDS` define) iteration, though HARDTANH uses forward only
- **Synchronization**: `cb_reserve_back(c_0, 1)` before writing, `cb_push_back(c_0, 1)` after NoC read completes. Blocks on `noc_async_read_barrier()` per page.

### Compute Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` |
| **Assigned cores** | core_group_1 (with `num_pages_per_core_group_1`), core_group_2 (with `num_pages_per_core_group_2`) |

**Key Logic**:
- Calls `init_sfpu(c_0, c_2)` once at startup to initialize SFPU pipeline between input and output CBs
- Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles (always 1)
- Per tile: `tile_regs_acquire()` -> `cb_wait_front(c_0, 1)` -> `copy_tile(c_0, 0, 0)` to load tile into DST register -> execute `SFPU_OP_CHAIN_0` macro -> `tile_regs_commit()` -> `tile_regs_wait()` -> `pack_tile(0, c_2)` -> `cb_pop_front(c_0, 1)` -> `tile_regs_release()`
- The `SFPU_OP_CHAIN_0` macro expands to `hardtanh_tile_init(); hardtanh_tile(0, {min_hex}u, {max_hex}u);`
- **SFPU hardtanh algorithm** (in `ckernel_sfpu_hardtanh.h`):
  - Loads `min_val` into LREG2 and `max_val` into LREG3 (each via two `SFPLOADI` for lower/upper 16 bits)
  - Iterates 8 times (ITERATIONS=8, processing 8 datum rows per tile face): loads element from DST into LREG0, copies min_val to LREG1 via `SFPMOV`, uses `SFPSWAP` with mode 1 (smaller to LREG0) to compute `max(x, min_val)` and stores result. Then repeats with max_val from LREG3 to compute `min(result, max_val)` and stores final clamped value.
  - `dst_reg++` advances to the next row of the tile face
- **Synchronization**: Waits on CB c_0 (`cb_wait_front`), pops from c_0 (`cb_pop_front`), reserves c_2 (`cb_reserve_back` at block level), pushes to c_2 (`cb_push_back` at block level)

### Writer Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` |
| **Assigned cores** | all_cores (both core_group_1 and core_group_2) |

**Key Logic**:
- Iterates from `start_id` to `start_id + num_pages`, writing one page per iteration
- Uses `TensorAccessor` constructed from compile-time args for bank-aware address resolution
- Gets CB page size dynamically from `get_local_cb_interface(cb_id_out).fifo_page_size`
- Supports `OUT_SHARDED` mode (single `cb_wait_front` for all pages) but HARDTANH via this factory uses the non-sharded interleaved path
- **Synchronization**: `cb_wait_front(c_2, 1)` before reading, `cb_pop_front(c_2, 1)` after NoC write is flushed. Calls `noc_async_write_barrier()` after all pages are written.

## Implementation Notes

- **Program factory variants**: Two factories can run HARDTANH: `UnaryProgramFactory` (standard interleaved path, analyzed here) and `UnarySubCoreGridProgramFactory` (for sub-core-grid cases). A `UnaryShardedProgramFactory` also exists for sharded tensors (defined elsewhere). The factory is selected based on whether the input tensor is sharded and whether `sub_core_grids` is specified.
- **Type-based operation variants**: Supports BFLOAT16, FLOAT32, INT32, and UINT32 input types. Type-specific defines (`INP_FLOAT32`, `INP_INT32`, `INP_UINT32`, `INP_FLOAT`) are set but the SFPU kernel itself operates on the raw bit pattern since `SFPSWAP` performs floating-point comparison. Integer types may produce unexpected results if the bit patterns are not IEEE-754 compatible.
- **UnpackToDestFP32 mode**: Enabled when `args.preserve_fp32_precision` is true. Sets `UnpackToDestMode::UnpackToDestFp32` for CB c_0 and CB c_1 (though c_1 is not used by HARDTANH).
- **Broadcast type selection**: N/A. HARDTANH is a pure unary element-wise operation with no broadcasting.
- **Sharding support and constraints**: The `UnaryProgramFactory` analyzed here handles only interleaved tensors. Sharded inputs use a separate `UnaryShardedProgramFactory`.
- **FP32 dest accumulation**: Controlled by `args.fp32_dest_acc_en`. When enabled, the DEST register uses FP32 precision for accumulation, which is relevant for maintaining precision in the clamp comparison.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary program factory work for SFPU operations? What is the structure of unary_program_factory.cpp and how does it set up kernels, circular buffers, and core distribution for SFPU unary ops?"
   **Reason**: Initial architectural understanding of the program factory pattern for unary SFPU operations.
   **Key Findings**: The factory creates three kernels (reader, compute, writer), splits work across cores using `split_work_to_cores()`, sets up CB c_0 (input, double-buffered) and CB c_2 (output, double-buffered), and dynamically selects the compute kernel path based on op type. Runtime arguments are updatable without recompilation via `override_runtime_arguments`.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` (lines 520-529)
   **Reason**: Understand how HARDTANH's init/func pair is generated for the SFPU_OP_CHAIN macro.
   **Key Information**: Generates `hardtanh_tile_init()` and `hardtanh_tile({idst}, {min_hex}u, {max_hex}u)` where min/max are bit-cast float-to-uint32 hex literals.

2. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Understand the actual SFPU instruction sequence for the hardtanh computation.
   **Key Information**: Uses `SFPLOADI` to load min/max params into LREG2/LREG3, then for each of 8 iterations: loads element, uses `SFPSWAP` (mode 1 = smaller to LREG0) twice to implement `clamp(x, min, max)` as `min(max(x, min_val), max_val)`.

3. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h`
   **Reason**: Verify the compute API wrapper that bridges the ckernel function call to the LLK layer.
   **Key Information**: `hardtanh_tile()` calls `llk_math_eltwise_unary_sfpu_hardtanh<APPROX>()` which dispatches to `_llk_math_eltwise_unary_sfpu_params_` with `calculate_hardtanh` as the SFPU function.
