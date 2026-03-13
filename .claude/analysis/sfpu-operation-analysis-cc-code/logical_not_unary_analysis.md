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

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel functions that the compute kernel dispatches to. LOGICAL_NOT_UNARY has two distinct kernel implementations: an SFPI-based kernel for float/int32/uint32 types and a raw TTI-based kernel for uint16.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/logical_not_noti.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (macro-based dispatch via `SFPU_UNARY_KERNEL_THREE_TEMPLATE_ARGS_FN` and `SFPU_UNARY_NO_PARAM_KERNEL_FN`) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_logical_not_noti.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `logical_not_unary_tile_init()` which expands via `SFPU_UNARY_KERNEL_INIT` to `llk_math_eltwise_unary_sfpu_init<SfpuType::logical_not_unary, APPROX>()`, configuring SFPU state and address modes.
2. The compute kernel calls `logical_not_unary_tile(0)` (for float/int32/uint32 types) which expands via `SFPU_UNARY_KERNEL_THREE_TEMPLATE_ARGS_FN` to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_logical_not_unary<V, T>, 0, VectorMode::RC)`.
3. For uint16, `logical_not_unary_tile_uint16(0)` expands via `SFPU_UNARY_NO_PARAM_KERNEL_FN` to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_logical_not_unary_uint16<APPROX>, 0, VectorMode::RC)`.
4. `_llk_math_eltwise_unary_sfpu_params_` sets the DST write address, stalls until SFPU is ready, then calls the SFPU function once per face (4 faces for RC mode), advancing the DEST read/write pointer by `DEST_FACE_WIDTH` (16 rows) between faces via `TTI_SETRWC`.
5. The core SFPU function (`calculate_logical_not_unary` or `calculate_logical_not_unary_uint16`) executes 8 iterations per face, processing 4 datums (one row of 32 elements across the SFPU's vector width) per iteration.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` for all type variants. This processes all 4 faces of the 32x32 tile (face 0 = top-left 16x16, face 1 = top-right 16x16, face 2 = bottom-left 16x16, face 3 = bottom-right 16x16).
- **Operation invocation**: The params function loops 4 times (once per face). Each iteration calls the SFPU function, then executes two `TTI_SETRWC` instructions to advance the DEST read/write pointer by 16 rows (2 x 8 rows per SETRWC call) to reach the next face.
- **DEST address progression**: The initial DEST address is set by `math::set_dst_write_addr<Tile32x32>(dst_index)`. Between faces, the pointer advances by `DEST_FACE_WIDTH` (16 rows) via two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` calls (each advancing by 8). Within the SFPU function, `dst_reg++` (SFPI) or `dst_reg++` (TTI variant) advances by 1 row per iteration, covering 8 rows per function call. After all 4 faces, `math::clear_dst_reg_addr()` resets the pointer.

### Annotated SFPU Kernel Source

This operation has two kernel implementations analyzed separately below.

#### Variant 1: SFPI-based kernel (float / int32 / uint32) -- Style A

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_logical_not_noti.h

namespace ckernel::sfpu {

template <typename V, typename T>
inline void calculate_logical_not_unary() { // V=sfpi::vFloat/vInt/vUInt, T=float/int16_t/uint16_t
#pragma GCC unroll 0 // Prevent compiler unrolling to reduce code size
    for (int d = 0; d < 8; d++) {
        V v = sfpi::dst_reg[0]; // Load current DEST row into SFPU vector register
        v_if(v == 0) { sfpi::dst_reg[0] = T(1); } // If element is zero, write 1
        v_else { sfpi::dst_reg[0] = T(0); }        // If element is non-zero, write 0
        v_endif;
        sfpi::dst_reg++; // Advance DEST pointer by 1 row (emits SETRWC)
    }
}

}  // namespace ckernel::sfpu
```

#### Variant 2: TTI-based kernel (uint16) -- Style B

The Wormhole and Blackhole implementations are identical except for the ADDR_MOD index used with SFPLOAD/SFPSTORE (Wormhole uses `ADDR_MOD_3`, Blackhole uses `ADDR_MOD_7`). The Wormhole variant is shown below:

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_logical_not_noti.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_logical_not_unary_uint16() {
    for (int d = 0; d < ITERATIONS; d++) {
        // full tile size
        constexpr int tile_size = 64;
        // load in conditional uint16 value
        TTI_SFPLOAD(p_sfpu::LREG0, LO16, ADDR_MOD_3, 0);
        // initially put 0 into output
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        // if (REG0 == 0)
        TTI_SFPSETCC(0, 0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        // load in (int) 1
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0x0001);

        // TTI_SFPENCC(IMM12_MATH, LREG_C, LREG_DEST, INSTR_MOD1);
        // IMM12_MATH: optional immediate value for math operations
        // LREG_C: unused
        // LREG_DEST: unused
        // INSTR_MOD1: 0 => condition code enable reg is not modified.
        TTI_SFPENCC(0, 0, 0, 0);
        // store result
        TTI_SFPSTORE(p_sfpu::LREG1, LO16, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}
```

**CC State Machine diagram for `calculate_logical_not_unary_uint16`:**

```
calculate_logical_not_unary_uint16 -- CC State Transitions (per iteration)
================================================================

  CC State: ALL_ENABLED                   <-- initial state
       |
       |  SFPLOAD LREG0, LO16, ADDR_MOD_3, 0   (no CC effect) -- load uint16 from DEST into LREG0
       |  SFPMOV LCONST_0 -> LREG1              (no CC effect) -- LREG1 = 0 (default output)
       |
       v
  +-------------------------------------------+
  | SFPSETCC  SFPSETCC_MOD1_LREG_EQ0 (=6)    |
  |                                           |
  | CC <- (LREG0 == 0)                        |
  +--------------------+----------------------+
                       |
                       v
  CC State: ENABLED where LREG0 == 0
       |
       |  SFPLOADI LREG1, USHORT, 0x0001    (CC-guarded: only LREG0==0 lanes)
       |                                     -- overwrites LREG1 with 1 for zero-valued lanes
       v
  +-------------------------------------------+
  | SFPENCC  INSTR_MOD1=0                     |
  |                                           |
  | CC <- ALL_ENABLED                         |
  +--------------------+----------------------+
                       |
                       v
  CC State: ALL_ENABLED
       |
       |  SFPSTORE LREG1, LO16, ADDR_MOD_3, 0   (all lanes) -- write result to DEST
       |  dst_reg++                               (advance DEST pointer by 1 row)
       v
```

The logic is: LREG1 starts at 0 for all lanes. SFPSETCC enables only lanes where the input (LREG0) is zero. The CC-guarded SFPLOADI then overwrites LREG1 with 1 only for those zero-valued lanes. SFPENCC re-enables all lanes. SFPSTORE writes the final LREG1 (0 for non-zero inputs, 1 for zero inputs) back to DEST. This achieves `output = (input == 0) ? 1 : 0` using predicated execution.

### SFPU Instructions Used

**SFPI variant (float/int32/uint32):**
The SFPI abstraction compiles to underlying SFPU instructions. The key operations are:
- `dst_reg[0]` read -- compiles to `SFPLOAD` to read a row from DEST into an SFPU LREG
- `v_if(v == 0)` -- compiles to `SFPSETCC` with EQ0 modifier, setting CC based on zero comparison
- `dst_reg[0] = T(1)` / `dst_reg[0] = T(0)` -- compile to `SFPLOADI` + `SFPSTORE` (or `SFPMOV` from LCONST) to write constant values, guarded by CC
- `v_else` / `v_endif` -- compile to `SFPCOMPC` (complement CC) and `SFPENCC` (re-enable all lanes)
- `dst_reg++` -- compiles to `SETRWC` to advance the DEST read/write pointer

**TTI variant (uint16):**

| Instruction | Description |
|-------------|-------------|
| `TTI_SFPLOAD(LREG0, LO16, ADDR_MOD_3, 0)` | Load 16-bit unsigned integer from current DEST row into LREG0. `LO16` mode reads the lower 16 bits. |
| `TTI_SFPMOV(0, LCONST_0, LREG1, 0)` | Move hardware constant 0 (LCONST_0 = register 9) into LREG1. Initializes output to 0 for all lanes. |
| `TTI_SFPSETCC(0, 0, 0, SFPSETCC_MOD1_LREG_EQ0)` | Set condition code: CC enabled for lanes where LREG0 == 0. INSTR_MOD1=6 selects EQ0 comparison mode. |
| `TTI_SFPLOADI(LREG1, SFPLOADI_MOD0_USHORT, 0x0001)` | Load immediate unsigned short value 1 into LREG1. CC-guarded: only executes for lanes where CC is enabled (input was zero). |
| `TTI_SFPENCC(0, 0, 0, 0)` | End conditional code region. Resets CC to ALL_ENABLED. INSTR_MOD1=0 means the CC enable register is not modified. |
| `TTI_SFPSTORE(LREG1, LO16, ADDR_MOD_3, 0)` | Store LREG1 as 16-bit unsigned integer back to current DEST row. `LO16` mode writes the lower 16 bits. |

### SFPU Register Usage

**SFPI variant:**
- `dst_reg[0]` (aliased to DEST row at current pointer): Read input, write output. The SFPI compiler manages LREG allocation internally.
- The SFPI runtime uses LREG0-LREG3 as working registers (compiler-managed), plus LCONST_0 (=0.0) and LCONST_1 (=1.0) as hardware-provided constants.

**TTI variant:**

| Register | Role | Description |
|----------|------|-------------|
| `LREG0` (p_sfpu::LREG0 = 0) | Input | Holds the uint16 value loaded from DEST |
| `LREG1` (p_sfpu::LREG1 = 1) | Output | Initialized to 0 via SFPMOV from LCONST_0, then conditionally overwritten with 1 by SFPLOADI under CC guard |
| `LCONST_0` (p_sfpu::LCONST_0 = 9) | Constant | Hardware-provided constant 0, source for SFPMOV to initialize LREG1 |
| `CC` (condition code register) | Control flow | Set by SFPSETCC (EQ0 test on LREG0), reset by SFPENCC |
| `DEST` (destination register file) | I/O | Source and sink for tile data; accessed via SFPLOAD/SFPSTORE with `dst_reg` pointer |

### Address Mode Configuration

The SFPU init function (`_llk_math_eltwise_unary_sfpu_init_`) calls `eltwise_unary_sfpu_configure_addrmod<SfpuType::logical_not_unary>()`. Since `logical_not_unary` does not match any special-cased SfpuType (topk_local_sort, typecast, unary_max/min), only the default ADDR_MOD_7 is configured:

**Wormhole B0 and Blackhole (identical):**
```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

This zero-increment address mode means that SFPLOAD/SFPSTORE do not auto-increment the DEST pointer -- the DEST pointer advancement is handled explicitly by `dst_reg++` (which emits a `SETRWC` instruction) at the end of each iteration.

**Note on the uint16 variant**: The Wormhole implementation uses `ADDR_MOD_3` in its SFPLOAD/SFPSTORE instructions, while the Blackhole implementation uses `ADDR_MOD_7`. On Wormhole, the params dispatch calls `math::set_addr_mod_base()` which sets the ADDR_MOD base register to 1, shifting effective indices from range 0..3 to 4..7. This means `ADDR_MOD_3` in the source code accesses physical `ADDR_MOD_7` (3 + base 4 = 7). On Blackhole, the source code directly references `ADDR_MOD_7` and there is no base offset shift in the params dispatch. Both resolve to the same {0, 0, 0} configuration.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary program factory work for SFPU operations? What is the structure of unary_program_factory.cpp and how does it select between FPU and SFPU paths?"
   **Reason**: Needed to understand the overall architecture of the unary program factory and how it dispatches to different compute kernels.
   **Key Findings**: Confirmed that `UnaryProgramFactory` is primarily SFPU-based, the compute kernel path is determined by `get_compute_kernel_path()`, and the factory selection depends on sharding and sub_core_grids properties. LOGICAL_NOT_UNARY falls into the default SFPU path via `eltwise_sfpu.cpp`.

2. [SFPU] **Query**: "How does the unary compute kernel dispatch SFPU operations for logical_not? What is the call chain from the compute kernel through LLK to the ckernel SFPU implementation?"
   **Reason**: Needed to trace the complete SFPU dispatch path from the compute kernel API through LLK macros to the core SFPU implementation.
   **Key Findings**: Confirmed the call chain: `logical_not_unary_tile()` -> `SFPU_UNARY_KERNEL_THREE_TEMPLATE_ARGS_FN` macro -> `_llk_math_eltwise_unary_sfpu_params_()` -> `calculate_logical_not_unary<V,T>()`. The macro system bridges the API layer to the hardware-specific ckernel implementation.

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

5. [SFPU] **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Contains the `_llk_math_eltwise_unary_sfpu_params_` function that orchestrates SFPU function dispatch with DST addressing and vector mode iteration.
   **Key Information**: In RC mode, the function loops 4 times (one per face), calling the SFPU function each time and advancing the DEST pointer by 16 rows between faces via two `TTI_SETRWC` calls. Stalls ensure SFPU pipeline synchronization.

6. [SFPU] **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Contains `eltwise_unary_sfpu_configure_addrmod` which sets up ADDR_MOD_7 with zero increments for the default SFPU case.
   **Key Information**: ADDR_MOD_7 is configured with {srca=0, srcb=0, dest=0} for all standard SFPU operations. Special cases (topk, typecast, min/max) configure additional ADDR_MOD_6.

7. [SFPU] **Source**: `runtime/sfpi/include/sfpi_constants.h`
   **Reason**: Contains SFPU instruction modifier constants used in the TTI-based kernel.
   **Key Information**: `SFPSETCC_MOD1_LREG_EQ0 = 6` (test LREG == 0), `SFPLOADI_MOD0_USHORT = 2` (load unsigned 16-bit immediate).
