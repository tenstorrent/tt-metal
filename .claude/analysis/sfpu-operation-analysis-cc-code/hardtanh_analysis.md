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

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. The compute kernel's `SFPU_OP_CHAIN_0` macro expands to `hardtanh_tile_init(); hardtanh_tile(0, {min_hex}u, {max_hex}u);`.
2. `hardtanh_tile_init()` (in `hardtanh.h`) calls `llk_math_eltwise_unary_sfpu_hardtanh_init<APPROX>()`, which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::hardtanh, APPROX>()` to configure SFPU config registers and address modes.
3. `hardtanh_tile(idst, param0, param1)` (in `hardtanh.h`) calls `llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1)`.
4. `llk_math_eltwise_unary_sfpu_hardtanh` (in `llk_math_eltwise_unary_sfpu_hardtanh.h`) calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_hardtanh<APPROX, 8>, dst_index, VectorMode::RC, param0, param1)`.
5. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) sets up DEST addressing, stalls for SFPU readiness, then calls `calculate_hardtanh(param0, param1)` once per face (4 times for `VectorMode::RC`), advancing the DEST address between faces.
6. `calculate_hardtanh` (in `ckernel_sfpu_hardtanh.h`) executes the raw SFPU instruction sequence that performs the clamp operation.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the 32x32 tile are processed. The dispatch loop iterates `face = 0..3`, calling the SFPU function once per face.
- **Operation invocation**: The core SFPU function `calculate_hardtanh` is called 4 times (once per face). Each invocation internally loops 8 times (`ITERATIONS=8`), processing one datum row per iteration. This covers all 8 rows of a 16x16 face.
- **DEST address progression**: Between faces, the dispatch function advances the DEST write address by 16 rows. On Wormhole, this is done via two `TTI_SETRWC(CLR_NONE, CR_D, 8, ...)` calls (each incrementing by 8). On Blackhole, this is done via two `inc_dst_addr<8>()` calls. Within a face, the SFPU kernel uses `sfpi::dst_reg++` to advance one row at a time. The SFPLOAD/SFPSTORE instructions use `ADDR_MOD_3` (Wormhole) or `ADDR_MOD_7` (Blackhole) with `dest.incr = 0`, so there is no auto-increment on load/store -- `dst_reg++` handles all intra-face row advancement.

### Annotated SFPU Kernel Source

The kernel uses raw `TTI_` instructions but has no condition code manipulation (no `SFPSETCC`, `SFPENCC`, or implicit CC side effects). `SFPSWAP`, `SFPLOAD`, `SFPSTORE`, `SFPMOV`, and `SFPLOADI` do not set CC. This is Style A.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h
// (Blackhole version is identical except ADDR_MOD_7 replaces ADDR_MOD_3)

namespace ckernel::sfpu {

// Hardtanh(x) = max_val if x > max_val, min_val if x < min_val, else x
// Equivalent to: clamp(x, min_val, max_val) = min(max(x, min_val), max_val)
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_hardtanh(uint param0, uint param1) { // APPROXIMATION_MODE is unused, ITERATIONS=8
    // Load min_val (param0) into LREG2 as a full 32-bit float, lower 16 bits then upper 16 bits
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, param0 & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, param0 >> 16);
    // Load max_val (param1) into LREG3 as a full 32-bit float
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, param1 & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, param1 >> 16);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // === Phase 1: x = max(x, min_val) ===
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0); // Load x from DEST into LREG0
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0); // Copy min_val from LREG2 to LREG1
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1); // mod1=1: smaller value -> LREG0, larger -> LREG1
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0); // Store max(x, min_val) from LREG1

        // === Phase 2: x = min(result, max_val) ===
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0); // Reload max(x, min_val) from DEST
        TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG1, 0); // Copy max_val from LREG3 to LREG1
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1); // mod1=1: smaller value -> LREG0, larger -> LREG1
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0); // Store min(result, max_val) from LREG0

        sfpi::dst_reg++; // Advance to the next row in the current tile face
    }
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `TT_SFPLOADI` | Loads a 16-bit immediate value into the lower or upper half of an SFPU local register. Used here to construct 32-bit float constants (min_val, max_val) in LREG2 and LREG3 from two 16-bit halves. The `SFPLOADI_MOD0_LOWER` and `SFPLOADI_MOD0_UPPER` modifiers select which half to write. |
| `TTI_SFPLOAD` | Loads a vector of values from the DEST register file into an SFPU local register (LREG0). `InstrModLoadStore::DEFAULT` loads in the default floating-point format. The address offset field (0) combined with the current DEST row pointer determines which row is read. |
| `TTI_SFPMOV` | Copies the contents of one SFPU local register to another. Used to duplicate the constant (min_val or max_val) into LREG1 before the swap, preserving the original in LREG2/LREG3 for reuse across iterations. |
| `TTI_SFPSWAP` | Conditionally swaps values between two SFPU local registers based on magnitude comparison. With `mod1=1`, places the smaller value in VD (LREG0) and the larger value in VC (LREG1). This single instruction implements both `max()` and `min()` -- after the swap, LREG1 holds the max and LREG0 holds the min. Does not modify condition codes. |
| `TTI_SFPSTORE` | Stores a vector of values from an SFPU local register back to the DEST register file. Uses the same address mode and offset as SFPLOAD to write back to the same DEST row. |
| `sfpi::dst_reg++` | Increments the DEST register row pointer by 1, advancing to the next row within the current tile face. This is an SFPI abstraction that translates to an internal DEST address counter increment. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Working register. Loaded with the current element vector from DEST via `SFPLOAD`. After `SFPSWAP` in Phase 1, holds `min(x, min_val)` (discarded). After `SFPSWAP` in Phase 2, holds `min(max(x, min_val), max_val)` -- the final clamped result, stored back to DEST. |
| **LREG1** | Temporary register. Receives a copy of the comparison constant via `SFPMOV`. After `SFPSWAP` in Phase 1, holds `max(x, min_val)` (stored to DEST). After `SFPSWAP` in Phase 2, holds the larger of the two values (discarded). |
| **LREG2** | Holds `min_val` (param0) throughout the entire kernel execution. Loaded once before the loop via two `TT_SFPLOADI` instructions (lower 16 bits, then upper 16 bits). Never modified during the loop. |
| **LREG3** | Holds `max_val` (param1) throughout the entire kernel execution. Loaded once before the loop via two `TT_SFPLOADI` instructions. Never modified during the loop. |
| **DEST** | The destination register file. Each row contains a vector of elements from the tile face. Read via `SFPLOAD` and written back via `SFPSTORE`. The row pointer advances by 1 each iteration via `dst_reg++`. |

### Address Mode Configuration

The `hardtanh` SFPU kernel uses `ADDR_MOD_3` on Wormhole and `ADDR_MOD_7` on Blackhole for all `SFPLOAD` and `SFPSTORE` instructions.

**Wormhole B0**:
- `ADDR_MOD_7` is explicitly configured during init (`eltwise_unary_sfpu_configure_addrmod<SfpuType::hardtanh>()`) to `{ srca.incr=0, srcb.incr=0, dest.incr=0 }`.
- The dispatch function `_llk_math_eltwise_unary_sfpu_params_` calls `math::set_addr_mod_base()`, which executes `TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 1)`, shifting the effective ADDR_MOD base from slot 0..3 to slots 4..7. When the kernel references `ADDR_MOD_3`, the hardware resolves it as physical slot `3 + 4 = 7`, which is the slot configured with `dest.incr=0`.
- With `dest.incr=0`, SFPLOAD/SFPSTORE do not auto-increment the DEST address. Row advancement within a face is handled entirely by `sfpi::dst_reg++`.

**Blackhole**:
- `ADDR_MOD_7` is explicitly configured during init to `{ srca.incr=0, srcb.incr=0, dest.incr=0 }`.
- Blackhole does not use the ADDR_MOD base offset mechanism -- `_llk_math_eltwise_unary_sfpu_start_` does NOT call `set_addr_mod_base()`. The kernel directly references `ADDR_MOD_7`, which is the explicitly configured slot. Behavior is identical: `dest.incr=0`, no auto-increment.

**Between faces** (managed by the params dispatch, not the SFPU kernel itself):
- On Wormhole: two `TTI_SETRWC(CLR_NONE, CR_D, 8, ...)` calls advance DEST by 16 rows (8+8) to the next face.
- On Blackhole: two `math::inc_dst_addr<8>()` calls achieve the same 16-row advance.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary program factory work for SFPU operations? What is the structure of unary_program_factory.cpp and how does it set up kernels, circular buffers, and core distribution for SFPU unary ops?"
   **Reason**: Initial architectural understanding of the program factory pattern for unary SFPU operations.
   **Key Findings**: The factory creates three kernels (reader, compute, writer), splits work across cores using `split_work_to_cores()`, sets up CB c_0 (input, double-buffered) and CB c_2 (output, double-buffered), and dynamically selects the compute kernel path based on op type. Runtime arguments are updatable without recompilation via `override_runtime_arguments`.

2. [SFPU] **Query**: "How is the hardtanh (relu_min/relu_max) SFPU operation implemented? What compute kernel does it use and how does it dispatch to the SFPU?"
   **Reason**: Understanding the full SFPU call chain from compute kernel through LLK to ckernel implementation.
   **Key Findings**: `hardtanh_tile()` dispatches through `llk_math_eltwise_unary_sfpu_hardtanh` to `_llk_math_eltwise_unary_sfpu_params_` with `calculate_hardtanh` as the functor. The core implementation uses `SFPSWAP` with mod1=1 to implement min/max operations. The metal `hw/ckernels/` implementation uses raw TTI_ instructions (different from the `tt_llk` version which uses SFPI abstractions with a subtraction-based algorithm).

3. [SFPU] **Query**: "How is relu_max or hardtanh implemented in the LLK layer?" (asked to `tenstorrent/tt-llk`)
   **Reason**: Understand the LLK dispatch mechanism and whether hardtanh reuses relu_max/relu_min kernels.
   **Key Findings**: Hardtanh has its own dedicated `calculate_hardtanh` function separate from `_relu_max_`/`_relu_min_`. The LLK dispatch uses `_llk_math_eltwise_unary_sfpu_params_` (the parameterized variant) rather than the basic `_llk_math_eltwise_unary_sfpu_` dispatch, because hardtanh requires two parameters (min, max).

4. [SFPU] **Query**: "What does the SFPSWAP instruction do? What are the modes (especially mode 1)? Does SFPSWAP modify the condition code?" (asked to `tenstorrent/tt-isa-documentation`)
   **Reason**: Understand the core comparison/swap instruction used by the hardtanh kernel.
   **Key Findings**: SFPSWAP with mod1=1 performs a conditional swap placing the smaller value in VD (LREG0) and the larger in VC (LREG1). It does NOT modify the condition code. Latency is 2 cycles. On Blackhole, SFPSWAP has a read-after-write hazard that may require manual SFPNOP insertion when preceded by SFPMAD.

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

4. [SFPU] **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the params dispatch function that manages VectorMode iteration and DEST address progression between tile faces.
   **Key Information**: For `VectorMode::RC`, iterates 4 faces, calling the SFPU functor once per face, with `TTI_SETRWC` advancing DEST by 16 rows (2x8) between faces.

5. [SFPU] **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand ADDR_MOD configuration and `set_addr_mod_base()` behavior for Wormhole SFPU operations.
   **Key Information**: The init function sets `ADDR_MOD_7` to `{dest.incr=0}` for all unary SFPU ops. `set_addr_mod_base()` shifts the base so kernel-referenced `ADDR_MOD_3` maps to physical slot 7.

### Confluence References

[SFPU] No Confluence page sections were consulted for this analysis. The SFPSWAP instruction details were sufficiently covered by DeepWiki's `tenstorrent/tt-isa-documentation` repository.

### Glean References

[SFPU] No Glean searches were needed for this analysis.
