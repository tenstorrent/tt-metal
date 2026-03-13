# LOGICAL_NOT_UNARY Implementation Analysis

## Overview

LOGICAL_NOT_UNARY is an element-wise unary operation that computes the logical NOT of each element in a tensor. For each element, it returns 1 if the element is zero, and 0 if the element is non-zero. This is the tensor equivalent of the C/C++ `!` operator applied element-wise.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

The operation is implemented as a pure SFPU operation. LOGICAL_NOT_UNARY falls into the `default` case of `get_compute_kernel_path()`, which routes it to the generic `eltwise_sfpu.cpp` compute kernel. The SFPU kernel define `SFPU_OP_LOGICAL_NOT_NOTI_INCLUDE` is set to include the logical-not-specific SFPU code.

## Path Selection: FPU vs SFPU

LOGICAL_NOT_UNARY is exclusively an SFPU operation. There is no FPU path for this operation. In `get_compute_kernel_path()` (line 958 of `unary_op_utils.cpp`), LOGICAL_NOT_UNARY is not listed as a special case and falls through to the `default` branch, which returns `"eltwise_sfpu.cpp"`. The program factory selection is determined by `UnaryDeviceOperation::select_program_factory()` in `unary_device_operation.cpp` (line 54): sharded tensors use `UnaryShardedProgramFactory`, tensors with `sub_core_grids` use `UnarySubCoreGridProgramFactory`, and all other tensors (the common interleaved case analyzed here) use `UnaryProgramFactory`.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `num_pages` = total number of tiles (or rows for ROW_MAJOR) in the input tensor |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles (always 1 for this factory) |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Any shape (flattened to pages) |
| **Dimension convention** | N/A (treated as flat page sequence) |
| **Tensor layout** | TILE_LAYOUT or ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, or UINT16 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | Same as input |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (or specified output dtype) |

### Layout Transformations

No explicit tilize/untilize or format conversions are performed within the program factory. The CB page sizes are set based on whether the layout is TILE (using `tile_size(cb_data_format)`) or ROW_MAJOR (using `buffer->page_size()`). The SFPU compute kernel always operates on tiles in the DST register.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `noc_async_read_barrier`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_reserve_back(c_2, 1)`, `cb_wait_front(c_0, 1)`, `copy_tile`, SFPU op chain, `pack_tile`, `cb_pop_front(c_0, 1)`, `cb_push_back(c_2, 1)` |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `noc_async_writes_flushed`, `cb_pop_front(c_2, 1)` |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | src0 | Input staging | 2 pages | 1 page | Double | Reader | Compute | Program |
| c_2 | output | Output staging | 2 pages | 1 page | Double | Compute | Writer | Program |

**Note**: CB c_1 (tmp0) is only allocated for HARDSHRINK, CBRT, or LOGIT operations. It is **not** allocated for LOGICAL_NOT_UNARY.

## Pipeline Pattern Summary

Both CB c_0 and CB c_2 have capacity = 2 pages and block size = 1 page, resulting in **double-buffering**. This allows the reader to fill one slot in c_0 while compute processes the other, and similarly compute can fill one slot in c_2 while the writer drains the other. This enables full overlap of read, compute, and write stages.

## Index Calculations

The program factory uses `TensorAccessor` for both reader and writer kernels. The `TensorAccessorArgs` are passed as compile-time arguments and encode the buffer's bank mapping (interleaved layout). At runtime, `noc_async_read_page(i, s, l1_write_addr)` and `noc_async_write_page(i, s, l1_read_addr)` translate the linear page index `i` to the correct DRAM bank and offset via the TensorAccessor `s`.

Each core receives a `start_id` (the first page index) and `num_pages` (how many pages to process). Pages are processed sequentially from `start_id` to `start_id + num_pages - 1`.

## Memory Access Patterns

### Read Pattern

Sequential page reads. The reader iterates linearly from `start_id` to `end_id`, reading one page at a time via `noc_async_read_page`. Each read is followed by `noc_async_read_barrier` (blocking until the read completes) before pushing to CB c_0.

### Write Pattern

Sequential page writes. The writer iterates linearly from `start_id` to `end_id`, writing one page at a time via `noc_async_write_page`. A `noc_async_writes_flushed` call ensures each write is dispatched before popping from CB c_2. A final `noc_async_write_barrier` at the end ensures all writes complete.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size` (device-dependent) |
| **Total cores** | Determined by `split_work_to_cores` based on `num_pages` |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` pages |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_pages / num_cores)` pages, group 2 gets `floor(num_pages / num_cores)` pages |

Cores are indexed linearly as `core = {i / num_cores_y, i % num_cores_y}`, filling columns first (column-major ordering). The `split_work_to_cores` utility divides pages across cores, creating two core groups to handle the remainder. Group 2 may be empty if pages divide evenly.

## Arguments

### Compile-Time Arguments

**Reader kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Encoded tensor accessor parameters for source buffer (bank mapping, page structure) |

**Writer kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer index (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encoded tensor accessor parameters for destination buffer |

**Compute kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tiles) this core processes |
| 1 | per_core_block_size | uint32_t | Tiles per block (always 1 in this factory) |

### Runtime Arguments

**Reader kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address |
| 1 | num_pages | uint32_t | Number of pages this core reads |
| 2 | start_id | uint32_t | First page index for this core |

**Writer kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address |
| 1 | num_pages | uint32_t | Number of pages this core writes |
| 2 | start_id | uint32_t | First page index for this core |

**Compute kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for LOGICAL_NOT_UNARY (set to 0) |
| 1 | packed_scalar2 | uint32_t | Unused for LOGICAL_NOT_UNARY (set to 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 src_buffer | CB c_0 | Sequential page reads via TensorAccessor |
| Compute | TRISC (math RISCV) | N/A | CB c_0 | CB c_2 | SFPU logical_not_unary: if element == 0 then 1, else 0 |
| Writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 dst_buffer | Sequential page writes via TensorAccessor |

### Reader Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` |
| **Assigned cores** | all_cores (both core_group_1 and core_group_2) |

**Key Logic**:
- Reads runtime args: `src_addr`, `num_pages`, `start_id`
- Instantiates a `TensorAccessor` from compile-time `TensorAccessorArgs<0>()` and runtime `src_addr`
- Gets page size from the CB interface: `get_local_cb_interface(cb_id_in0).fifo_page_size`
- Iterates from `start_id` to `start_id + num_pages`, reading one page per iteration
- **Synchronization**: `cb_reserve_back(c_0, 1)` blocks until CB c_0 has space; after `noc_async_read_barrier()` confirms the read, `cb_push_back(c_0, 1)` signals the compute kernel
- Supports optional `BACKWARDS` define for reverse iteration (not used for LOGICAL_NOT_UNARY)

### Compute Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` |
| **Assigned cores** | core_group_1 and core_group_2 (with different `per_core_block_cnt` compile-time args) |

**Key Logic**:
- Compile-time args: `per_core_block_cnt` (number of tiles), `per_core_block_dim` (always 1)
- Calls `init_sfpu(c_0, c_2)` to initialize the SFPU pipeline with input/output CBs
- Outer loop iterates `per_core_block_cnt` times (one tile per iteration since `per_core_block_dim` = 1)
- For each tile:
  1. `tile_regs_acquire()` -- acquire DST register file
  2. `cb_wait_front(c_0, 1)` -- wait for reader to provide a tile
  3. `copy_tile(c_0, 0, 0)` -- unpack tile from CB c_0 into DST register 0
  4. Execute `SFPU_OP_CHAIN_0` -- the preprocessor-injected SFPU operation chain, which expands to `logical_not_unary_tile_init(); logical_not_unary_tile(0);`
  5. `tile_regs_commit()` -- signal tile is ready for packing
  6. `tile_regs_wait()` -- wait for packer readiness
  7. `pack_tile(0, c_2)` -- pack DST register 0 into CB c_2
  8. `cb_pop_front(c_0, 1)` -- free the consumed input tile
  9. `tile_regs_release()` -- release DST registers
- `cb_reserve_back(c_2, per_core_block_dim)` is called before the inner loop; `cb_push_back(c_2, per_core_block_dim)` after the inner loop
- **SFPU operation**: The `logical_not_unary_tile(idst)` function invokes `calculate_logical_not_unary<sfpi::vFloat, float>` which iterates 8 times (processing 8 datum rows, covering the 32x32 tile in 8 SFPU vector passes of 4 elements each = 32 rows x 32 cols). For each datum: reads from `dst_reg[0]`, if value == 0 writes 1, else writes 0. Uses SFPI conditional (`v_if`/`v_else`/`v_endif`) for branchless vector execution.

### Writer Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` |
| **Assigned cores** | all_cores (both core_group_1 and core_group_2) |

**Key Logic**:
- Reads runtime args: `dst_addr`, `num_pages`, `start_id`
- Compile-time arg 0: `cb_id_out` (= 2, i.e., c_2)
- Instantiates a `TensorAccessor` from compile-time `TensorAccessorArgs<1>()` and runtime `dst_addr`
- Iterates from `start_id` to `start_id + num_pages`, writing one page per iteration
- **Synchronization**: `cb_wait_front(c_2, 1)` blocks until compute has produced a tile; after `noc_async_write_page` and `noc_async_writes_flushed`, `cb_pop_front(c_2, 1)` frees the consumed output slot
- Final `noc_async_write_barrier()` ensures all writes are committed before kernel exits
- Supports `OUT_SHARDED` define (not used in this interleaved factory) and `BACKWARDS` define

## Implementation Notes

- **Program factory variants**: Three program factories can initiate this operation: `UnaryProgramFactory` (default, for interleaved tensors), `UnarySubCoreGridProgramFactory` (when `sub_core_grids` is specified), and `UnaryShardedProgramFactory` (for sharded tensors). Selection is in `UnaryDeviceOperation::select_program_factory()`. This analysis covers `UnaryProgramFactory`.

- **Type-based operation variants**: LOGICAL_NOT_UNARY supports five data types with distinct SFPU implementations:
  - **BFLOAT16 / FLOAT32**: Uses `logical_not_unary_tile()` which calls `calculate_logical_not_unary<sfpi::vFloat, float>` -- comparison and assignment via SFPI `v_if`/`v_else` with float types.
  - **INT32**: Uses `logical_not_unary_tile_int32()` which calls `calculate_logical_not_unary<sfpi::vInt, int16_t>` -- same algorithm with integer vector type.
  - **UINT32**: Uses `logical_not_unary_tile_uint32()` which calls `calculate_logical_not_unary<sfpi::vUInt, uint16_t>`.
  - **UINT16**: Uses `logical_not_unary_tile_uint16()` which calls a completely different implementation `calculate_logical_not_unary_uint16` using raw TTI instructions (SFPLOAD, SFPMOV, SFPSETCC, SFPLOADI, SFPENCC, SFPSTORE) for 16-bit unsigned integer handling. This variant uses `LO16` addressing mode for 16-bit load/store.
  - The data type selection is done by `get_op_init_and_func_parameterized()` based on `input_dtype`, and the corresponding `INP_FLOAT32`, `INP_INT32`, `INP_UINT32`, or `INP_FLOAT` define is set in the program factory.

- **UnpackToDestFP32 mode**: Enabled when `args.preserve_fp32_precision` is true. Sets `UnpackToDestMode::UnpackToDestFp32` for CB c_0 and c_1 (c_1 is unused for this operation).

- **Broadcast type selection**: N/A. LOGICAL_NOT_UNARY is a pure unary operation with no broadcasting.

- **Sharding support and constraints**: Sharded tensors are routed to `UnaryShardedProgramFactory` (not analyzed here). The interleaved factory does not support sharded inputs.

- **FP32 dest accumulation**: Controlled by `args.fp32_dest_acc_en` and passed to `ComputeConfig`. When enabled, the DST register file uses FP32 format for intermediate results, providing higher precision for the comparison operation.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary program factory work for SFPU operations? What is the structure of unary_program_factory.cpp and how does it select between FPU and SFPU paths?"
   **Reason**: Needed to understand the overall architecture of the unary program factory and how it dispatches to different compute kernels.
   **Key Findings**: Confirmed that `UnaryProgramFactory` is primarily SFPU-based, the compute kernel path is determined by `get_compute_kernel_path()`, and the factory selection depends on sharding and sub_core_grids properties. LOGICAL_NOT_UNARY falls into the default SFPU path via `eltwise_sfpu.cpp`.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Contains `get_compute_kernel_path()`, `get_block_defines()`, `get_op_init_and_func_parameterized()`, and `get_macro_definition()` which determine kernel selection and SFPU operation chain generation.
   **Key Information**: LOGICAL_NOT_UNARY uses macro `SFPU_OP_LOGICAL_NOT_NOTI_INCLUDE`, compute kernel `eltwise_sfpu.cpp`, and has four type-specific tile function variants.

2. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_logical_not_noti.h`
   **Reason**: Contains the actual SFPU implementation of the logical NOT operation.
   **Key Information**: The core algorithm iterates 8 times over DST register datums, using SFPI conditional execution to write 1 where input is 0 and 0 where input is non-zero. The UINT16 variant uses raw TTI instructions for 16-bit operations.

3. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/logical_not_noti.h`
   **Reason**: Contains the HLK-level tile API wrappers that connect the compute kernel's `SFPU_OP_CHAIN_0` to the actual SFPU implementations.
   **Key Information**: Five functions defined: `logical_not_unary_tile` (float), `logical_not_unary_tile_int32`, `logical_not_unary_tile_uint32`, `logical_not_unary_tile_uint16`, and `logical_not_unary_tile_init`. Each dispatches to the appropriate template instantiation of `calculate_logical_not_unary`.

4. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Contains the `SFPU_UNARY_KERNEL_THREE_TEMPLATE_ARGS_FN` macro used by the logical_not_unary_tile functions.
   **Key Information**: The macro wraps the SFPU function call with `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` which handles DST indexing and vector mode (RC mode = row-column processing).
