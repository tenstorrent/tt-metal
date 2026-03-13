# LTZ (Less Than Zero) Implementation Analysis

## Overview

LTZ is a unary element-wise comparison operation that tests whether each element of a tensor is less than zero. For each element, the output is `1.0` (true) if `x < 0`, and `0.0` (false) otherwise. It supports both floating-point types (via SFPU float comparison) and INT32 (via SFPU integer comparison). The operation is implemented through the shared unary SFPU program factory.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Path Selection: FPU vs SFPU

LTZ is exclusively an SFPU operation. There is no FPU path for this operation. The `get_compute_kernel_path` function in `unary_op_utils.cpp` maps `UnaryOpType::LTZ` to the `default` case, which returns `"eltwise_sfpu.cpp"`. This is the generic SFPU compute kernel shared by many unary operations. The operation-specific behavior is injected via preprocessor defines (`SFPU_OP_CHAIN_0`) that expand to `ltz_tile_init()` and `ltz_tile(0)` calls. The include guard `SFPU_OP_UNARY_COMP_INCLUDE` is set to `1`, which pulls in `api/compute/eltwise_unary/comp.h` containing the `ltz_tile` / `ltz_tile_init` / `ltz_tile_int32` functions.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `num_pages` = total number of tiles in the input tensor |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles (always 1) |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary (flattened to tiles) |
| **Dimension convention** | N/A (operates on flat tile stream) |
| **Tensor layout** | TILE_LAYOUT (or ROW_MAJOR) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, or INT32 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | Same as input |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input |

### Layout Transformations

No tilize/untilize or format conversions are performed. The operation is applied in-place on tiles in DEST registers.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `noc_async_read_barrier`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_reserve_back(c_2, 1)`, `cb_wait_front(c_0, 1)`, `copy_tile`, `ltz_tile(0)`, `pack_tile`, `cb_pop_front(c_0, 1)`, `cb_push_back(c_2, 1)` |
| 3 | Writer | CB c_2 | DRAM/L1 (dst buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `noc_async_writes_flushed`, `cb_pop_front(c_2, 1)` |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | src0 | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | output | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

Note: CB c_1 (tmp0) is NOT allocated for LTZ. It is only allocated for HARDSHRINK, CBRT, or LOGIT operations.

## Pipeline Pattern Summary

Both CB c_0 and CB c_2 are double-buffered (capacity = 2 tiles, block size = 1 tile). This allows the reader to fill one tile slot while the compute kernel processes another, and similarly the compute kernel can write to one output slot while the writer drains the other. This enables overlapped execution across all three pipeline stages.

## Index Calculations

Index calculations use the `TensorAccessor` abstraction. The reader and writer kernels receive `TensorAccessorArgs` as compile-time arguments, which encode the buffer's interleaving scheme (page-to-bank mapping). At runtime, `noc_async_read_page(i, s, l1_write_addr)` and `noc_async_write_page(i, s, l1_read_addr)` translate a logical page index `i` to a physical NoC address using the accessor `s`. The page index starts at `start_id` (a runtime argument) and increments sequentially for `num_pages` iterations.

## Memory Access Patterns

### Read Pattern
Sequential page-by-page reads. Each core reads a contiguous range of pages starting at `start_id`, advancing by 1 per iteration. Pages are read via NoC0 using `noc_async_read_page`, with a barrier after each read to ensure completion before pushing to the CB.

### Write Pattern
Sequential page-by-page writes. Each core writes a contiguous range of pages starting at `start_id`, advancing by 1 per iteration. Pages are written via NoC1 using `noc_async_write_page`, with a flush after each write. A final `noc_async_write_barrier` is issued after the loop.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major iteration) |
| **Grid dimensions** | `compute_with_storage_grid_size.x` x `compute_with_storage_grid_size.y` |
| **Total cores** | Determined by `split_work_to_cores` |
| **Work per core** | `num_pages / num_cores` tiles (approximately) |
| **Load balancing** | Two-group: core_group_1 gets `num_pages_per_core_group_1` tiles, core_group_2 gets `num_pages_per_core_group_2` tiles (one fewer) |

The `split_work_to_cores` utility distributes `num_pages` tiles across available cores. If tiles do not divide evenly, some cores (core_group_1) handle one extra tile compared to others (core_group_2). Core iteration order is column-major: `core = {i / num_cores_y, i % num_cores_y}`.

## Arguments

### Compile-Time Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Encoded interleaving parameters for the source buffer |

**Writer kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (always 2 / `c_2`) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encoded interleaving parameters for the destination buffer |

**Compute kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks to process on this core |
| 1 | per_core_block_size | uint32_t | Tiles per block (always 1) |

### Runtime Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address |
| 1 | num_pages | uint32_t | Number of pages (tiles) to read |
| 2 | start_id | uint32_t | Starting page index for this core |

**Writer kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address |
| 1 | num_pages | uint32_t | Number of pages (tiles) to write |
| 2 | start_id | uint32_t | Starting page index for this core |

**Compute kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Always 0 for LTZ (no scalar parameter) |
| 1 | packed_scalar2 | uint32_t | Always 0 for LTZ (no scalar parameter) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | RISC_V 0 | NOC0 | DRAM/L1 src buffer | CB c_0 | Read tiles via TensorAccessor |
| Compute | RISC_V 2 (MATH) | N/A | CB c_0 | CB c_2 | `copy_tile` + `ltz_tile` (SFPU less-than-zero comparison) |
| Writer | RISC_V 1 | NOC1 | CB c_2 | DRAM/L1 dst buffer | Write tiles via TensorAccessor |

### Reader Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` |
| **Assigned cores** | all_cores (both core_group_1 and core_group_2) |

**Key Logic:**
- Iterates from `start_id` to `start_id + num_pages`, reading one page per iteration
- Uses `TensorAccessor` constructed from compile-time `TensorAccessorArgs` and runtime `src_addr`
- Page size is obtained dynamically from the CB interface (`get_local_cb_interface(cb_id_in0).fifo_page_size`)
- Supports optional `BACKWARDS` mode (not used for LTZ) for reverse iteration
- **Synchronization**: Calls `cb_reserve_back(c_0, 1)` to wait for space, performs async NoC read with barrier, then `cb_push_back(c_0, 1)` to signal compute

### Compute Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` |
| **Assigned cores** | core_group_1 and core_group_2 (separate kernel instances with different `per_core_block_cnt`) |

**Key Logic:**
- Calls `init_sfpu(c_0, c_2)` which initializes unpack (A2D datacopy), math, and pack hardware
- Outer loop iterates `per_core_block_cnt` times (one tile per block since `per_core_block_dim = 1`)
- Inner loop (1 iteration): acquires DEST registers, waits for input tile, copies tile from CB c_0 to DEST via `copy_tile`
- Executes `SFPU_OP_CHAIN_0` macro which expands to `ltz_tile_init(); ltz_tile(0);`
  - `ltz_tile_init()` expands to `llk_math_eltwise_unary_sfpu_init<SfpuType::less_than_zero, false>()`
  - `ltz_tile(0)` expands to `SFPU_ZERO_KERNEL(less_than_zero, RC, false, 0)` which calls `_llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::calculate_comp<false, SfpuType::less_than_zero>, 0, VectorMode::RC, 8)`
- For INT32 input: `ltz_tile_int32(0)` uses `SFPU_ZERO_KERNEL_TYPE(calculate_comp_int, less_than_zero, ...)` instead
- The SFPU `calculate_comp<false, SfpuType::less_than_zero>` function: iterates 8 times (8 vectors of 4 elements = 32 rows per face, processing all faces), reads `dst_reg[0]`, applies `v_if(v >= 0.0f) { v = 0.0f; } v_else { v = 1.0f; }`, writes back to `dst_reg[0]`, increments `dst_reg`
- For INT32: `calculate_comp_int` works similarly but with integer comparison `v_if(v < 0) { v = 1; } v_else { v = 0; }`
- After SFPU operation: commits DEST, waits for pack, packs tile from DEST to CB c_2, pops input from CB c_0, releases DEST
- **Synchronization**: `cb_wait_front(c_0, 1)` waits for reader; `cb_pop_front(c_0, 1)` frees reader slot; `cb_reserve_back(c_2, 1)` waits for writer to drain; `cb_push_back(c_2, 1)` signals writer

### Writer Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` |
| **Assigned cores** | all_cores (both core_group_1 and core_group_2) |

**Key Logic:**
- Iterates from `start_id` to `start_id + num_pages`, writing one page per iteration
- Uses `TensorAccessor` constructed from compile-time args and runtime `dst_addr`
- Page size obtained dynamically from CB interface
- Supports `OUT_SHARDED` mode (not used in this interleaved factory) where it simply waits for all pages
- Calls `noc_async_writes_flushed()` after each write for ordering, then `noc_async_write_barrier()` after the loop for completion
- **Synchronization**: `cb_wait_front(c_2, 1)` waits for compute to produce a tile; `cb_pop_front(c_2, 1)` frees the slot for compute

## Implementation Notes

- **Program factory variants**: Two program factories can run LTZ: `UnaryProgramFactory` (standard, uses `split_work_to_cores` for automatic grid sizing) and `UnarySubCoreGridProgramFactory` (uses caller-specified `sub_core_grids` with uniform tile distribution requiring even divisibility). Both use the same reader/compute/writer kernels. The factory is selected based on whether `sub_core_grids` is provided in the operation parameters.

- **Type-based operation variants**: Supports BFLOAT16, FLOAT32, and INT32. For BFLOAT16/FLOAT32, `ltz_tile()` uses `calculate_comp<false, SfpuType::less_than_zero>` which performs float comparison (`v >= 0.0f`). For INT32, `ltz_tile_int32()` uses `calculate_comp_int<false, SfpuType::less_than_zero>` which performs integer comparison (`v < 0`). Input dtype defines are set: `INP_FLOAT32` for FLOAT32, `INP_INT32` for INT32, `INP_UINT32` for UINT32, `INP_FLOAT` for BFLOAT16.

- **UnpackToDestFP32 mode**: Enabled when `args.preserve_fp32_precision` is true. Sets `UnpackToDestMode::UnpackToDestFp32` for CB c_0 and CB c_1 (though c_1 is unused for LTZ). This causes the unpacker to convert data to FP32 in DEST registers before the SFPU operates on it.

- **Broadcast type selection**: N/A. LTZ is a unary operation with no broadcasting.

- **Sharding support and constraints**: The `UnaryProgramFactory` analyzed here handles only interleaved memory layout. A separate `UnaryShardedProgramFactory` (not analyzed) handles sharded tensors.

- **FP32 dest accumulation**: Controlled by `args.fp32_dest_acc_en`. Passed to `ComputeConfig.fp32_dest_acc_en`. When enabled, DEST registers operate in FP32 mode, providing higher precision for the comparison result before packing.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/comp.h` |
| **LLK Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Wormhole) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_comp.h` (Wormhole) / `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_comp.h` (Blackhole) |
| **Parameters Dispatch** | Same as LLK Dispatch (the `_llk_math_eltwise_unary_sfpu_params_` function is in `llk_math_eltwise_unary_sfpu_params.h`) |

### Call Chain

1. The compute kernel calls `ltz_tile(0)` (defined in `comp.h`), which wraps `MATH(SFPU_ZERO_KERNEL(less_than_zero, RC, APPROX, idst))`.
2. The `SFPU_ZERO_KERNEL` macro (in `llk_math_eltwise_unary_sfpu_macros.h`) expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_comp<APPROX, SfpuType::less_than_zero>, 0, (int)VectorMode::RC, 8)`.
3. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) sets DEST write address, activates addr_mod base, stalls for SFPU readiness, then loops over 4 faces in `VectorMode::RC`, calling `calculate_comp<false, SfpuType::less_than_zero>(8)` once per face and advancing the DEST pointer by 16 rows (2x `SETRWC` with offset 8) between faces.
4. `calculate_comp` (in `ckernel_sfpu_comp.h`) runs 8 iterations per face invocation, each processing one 4-element vector row: it loads from `dst_reg[0]`, applies the less-than-zero comparison via SFPI conditionals, writes the result back to `dst_reg[0]`, and increments `dst_reg` (stride 2).

For INT32: the call chain is identical except `ltz_tile_int32(0)` uses the `SFPU_ZERO_KERNEL_TYPE` macro to dispatch to `calculate_comp_int<false, SfpuType::less_than_zero>`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the 32x32 tile are processed (face 0, 1, 2, 3). Each face is 16 rows x 16 columns, with the SFPU processing 4-element vectors per row.
- **Operation invocation**: The params dispatch function loops `for (int face = 0; face < 4; face++)`, calling `calculate_comp(8)` once per face. The `8` argument is passed through as `exponent_size_8` (but is not used by the `less_than_zero` branch; it is consumed only by the `equal_zero` / `not_equal_zero` branches for `_sfpu_is_fp16_zero_`). Inside `calculate_comp`, there is an inner loop of 8 iterations, processing 8 vector rows per face (8 rows x 4 elements = 32 elements per face width, but actually 8 rows of the 16-row face since `dst_reg` stride is 2).
- **DEST address progression**: The params dispatch sets the initial DEST write address via `math::set_dst_write_addr<Tile32x32>(dst_index)`. Between faces, Wormhole uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice (advancing DEST by 8+8=16 rows). Blackhole uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice (same 16-row advance). Within a face, `dst_reg++` advances the SFPU's internal DEST pointer by stride 2 per iteration.

### Annotated SFPU Kernel Source

This kernel uses SFPI abstractions (`vFloat`, `vInt`, `v_if`, `v_else`, `v_endif`, `dst_reg`), so Style A (inline-commented source code) is used.

**Float variant** (`calculate_comp` with `COMP_MODE = SfpuType::less_than_zero`):

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_comp.h

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp(uint exponent_size_8) { // APPROXIMATION_MODE=false, COMP_MODE=SfpuType::less_than_zero, ITERATIONS=8
    const vFloat zero = 0.0f;
    const vFloat one = 1.0f;
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0]; // Load 4-element vector from current DEST row into LREG

        // a[i] < 0
        if constexpr (COMP_MODE == SfpuType::less_than_zero) {
            v_if(v >= 0.0f) { v = zero; } // Comparison compiles to SFPXFCMPS with GTE condition; lanes where v >= 0 get 0.0
            v_else { v = one; }            // Remaining lanes (v < 0) get 1.0
            v_endif;
        }

        dst_reg[0] = v; // Store result back to same DEST row (SFPSTORE)
        dst_reg++;       // Advance DEST pointer by stride 2 to next vector row
    }
}
```

**INT32 variant** (`calculate_comp_int` with `COMP_MODE = SfpuType::less_than_zero`):

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_comp.h

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_int() { // APPROXIMATION_MODE=false, COMP_MODE=SfpuType::less_than_zero, ITERATIONS=8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = dst_reg[0]; // Load 4-element int vector from current DEST row
        vInt zero = 0;

        // a[i] < 0
        if constexpr (COMP_MODE == SfpuType::less_than_zero) {
            v_if(v < zero) { v = 1; } // Integer comparison via SFPIADD; lanes where v < 0 get 1
            v_else { v = zero; }       // Remaining lanes (v >= 0) get 0
            v_endif;
        }

        dst_reg[0] = v; // Store result back to same DEST row
        dst_reg++;       // Advance DEST pointer by stride 2
    }
}
```

Note: The Wormhole and Blackhole implementations of `calculate_comp` and `calculate_comp_int` are identical in source code. The only difference is in the `calculate_comp_uint16` and `calculate_eqz_uint32`/`calculate_nez_uint32` variants (which use raw TTI_ instructions and differ in `ADDR_MOD_3` vs `ADDR_MOD_7`), but those are not used by LTZ.

### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|--------------------------|-------------|
| `SFPLOAD` (via `dst_reg[0]` read) | Loads a 4-element vector from the DEST register file into an SFPU local register (LREG). Generated by the `vFloat v = dst_reg[0]` assignment. |
| `SFPXFCMPS` (via `v >= 0.0f`) | Scalar float comparison: compares each lane of a vFloat against a scalar float constant (0.0f) with GTE condition. Sets the condition code (CC) per lane. Generated by the `v_if(v >= 0.0f)` expression in the float variant. |
| `SFPSETCC` / `SFPPUSHC` / `SFPCOMPC` / `SFPPOPC` (via `v_if`/`v_else`/`v_endif`) | CC stack management instructions. `v_if` pushes the comparison result onto the CC stack, `v_else` complements the CC, `v_endif` pops the CC stack. These control predicated execution of the assignment instructions. |
| `SFPLOADI` (via `v = 0.0f` / `v = 1.0f`) | Loads an immediate constant into an LREG. Used for the `zero` (0.0f) and `one` (1.0f) constants, and for integer constants `0` and `1` in the INT32 variant. |
| `SFPSTORE` (via `dst_reg[0] = v`) | Stores a 4-element vector from an SFPU local register back to the DEST register file. Generated by `dst_reg[0] = v`. |
| `SFPIADD` (via `v < zero` in INT32 variant) | Integer addition/comparison. In the INT32 variant, `v < zero` compiles to an SFPIADD-based comparison that sets the CC based on the sign of the result. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST register file** | Input/output storage. The tile data resides in DEST after `copy_tile` moves it from CB c_0. Each face occupies 16 rows of DEST (rows 0-15 for face 0, 16-31 for face 1, etc.). The SFPU reads from and writes back to the same DEST location. |
| **LREG0** | Implicitly used as the primary working register. `vFloat v = dst_reg[0]` loads into LREG0. The comparison result and constant assignments also target LREG0. `dst_reg[0] = v` stores LREG0 back to DEST. |
| **LREG1-LREG3** | May be used transiently by the compiler for holding intermediate values (e.g., the `zero` and `one` constants), but the SFPI abstraction manages register allocation transparently. |
| **CC (Condition Code) register** | Used by `v_if`/`v_else`/`v_endif` to track which lanes satisfy the comparison condition. Each lane has an independent CC bit controlling predicated execution. |
| **CC Stack** | The `v_if` pushes the current CC state, `v_else` complements it (via `SFPCOMPC`), and `v_endif` pops (via `SFPPOPC`), restoring the prior CC state. |

### Address Mode Configuration

The SFPU init function (`_llk_math_eltwise_unary_sfpu_init_<SfpuType::less_than_zero>`) calls `eltwise_unary_sfpu_configure_addrmod<SfpuType::less_than_zero>()`, which configures:

**ADDR_MOD_7** (the default SFPU addr_mod, selected because ADDR_MOD_0 and ADDR_MOD_2 are used by the A2D datacopy):

| Field | Value | Description |
|-------|-------|-------------|
| `srca.incr` | 0 | No auto-increment for source A |
| `srcb.incr` | 0 | No auto-increment for source B |
| `dest.incr` | 0 | No auto-increment for DEST |

This configuration is identical on both Wormhole and Blackhole. Since `dest.incr = 0`, the SFPU does not auto-increment the DEST address between SFPLOAD/SFPSTORE pairs within a single iteration. Instead, DEST address progression within a face is handled by `dst_reg++` (which advances the SFPU's internal read/write pointer by stride 2), and progression between faces is handled by the params dispatch layer using explicit `SETRWC` instructions (Wormhole) or `math::inc_dst_addr<8>()` calls (Blackhole).

The `less_than_zero` SfpuType does not match any of the special-case addr_mod configurations (topk_local_sort, typecast, unary_max/min), so only ADDR_MOD_7 with all-zero increments is set.

The params dispatch function also calls `math::set_addr_mod_base()` before SFPU execution, which sets `ADDR_MOD_SET_Base_ADDR32 = 1`. This shifts the effective addr_mod index range from 0-3 to 4-7, so that `ADDR_MOD_3` in SFPU instructions actually refers to hardware addr_mod slot 7 (where the zero-increment configuration was programmed). After SFPU completes, `math::clear_addr_mod_base()` restores the base to 0.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary SFPU program factory work? What kernels does it use for SFPU operations like ltz?"
   **Reason**: Needed to understand the overall architecture of the unary program factory and how LTZ fits in.
   **Key Findings**: Confirmed that LTZ uses `eltwise_sfpu.cpp` as compute kernel, preprocessor defines inject operation-specific behavior via `SFPU_OP_CHAIN_0`, and the `SFPU_ZERO_KERNEL` macro is used for compare-with-zero operations.

2. [SFPU] **Query**: "How does the LTZ (less than zero) SFPU kernel work? What is the call chain from ltz_tile() through the LLK layer to the ckernel SFPU implementation?"
   **Reason**: Needed to trace the full call chain from the compute API through LLK dispatch to the core SFPU function, and identify the source files at each layer.
   **Key Findings**: Confirmed the call chain: `ltz_tile()` -> `SFPU_ZERO_KERNEL` macro -> `_llk_math_eltwise_unary_sfpu_params_` -> `calculate_comp<APPROX, SfpuType::less_than_zero>`. The core implementation is in `ckernel_sfpu_comp.h` for both Wormhole and Blackhole.

3. [SFPU] **Query**: "How do v_if, v_else, v_endif work in SFPI? How does dst_reg indexing work? What SFPU instructions do vFloat comparisons compile down to?"
   **Reason**: Needed to understand the SFPI abstractions used in the LTZ kernel to accurately document which hardware instructions are generated.
   **Key Findings**: `v_if`/`v_else`/`v_endif` manage a CC stack using `SFPSETCC`/`SFPPUSHC`/`SFPCOMPC`/`SFPPOPC`. Float comparisons like `v >= 0.0f` compile to `__builtin_rvtt_sfpxfcmps` (SFPXFCMPS instruction). `dst_reg[i]` accesses the DEST register file with stride 2, and `dst_reg++` advances by `SFP_DESTREG_STRIDE = 2`.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to trace how LTZ's compute kernel path, macro defines, and init/func pairs are generated.
   **Key Information**: LTZ maps to `SFPU_OP_UNARY_COMP_INCLUDE` define, uses `eltwise_sfpu.cpp` kernel (default case), and generates `ltz_tile_init()` / `ltz_tile(0)` calls (or `ltz_tile_int32` for INT32).

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/comp.h`
   **Reason**: Needed to understand the high-level API for `ltz_tile` and how it dispatches to the SFPU.
   **Key Information**: `ltz_tile(idst)` expands to `SFPU_ZERO_KERNEL(less_than_zero, RC, APPROX, idst)`. `ltz_tile_int32(idst)` expands to `SFPU_ZERO_KERNEL_TYPE(calculate_comp_int, less_than_zero, RC, APPROX, idst)`.

3. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_comp.h`
   **Reason**: Needed to understand the actual SFPU implementation of the less-than-zero comparison.
   **Key Information**: `calculate_comp<APPROX, SfpuType::less_than_zero>` iterates 8 times over DEST rows, performs `v_if(v >= 0.0f) { v = 0.0f; } v_else { v = 1.0f; }` using SFPI conditional instructions. The INT32 variant `calculate_comp_int` uses `v_if(v < 0) { v = 1; } v_else { v = 0; }`.

4. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Needed to understand the `SFPU_ZERO_KERNEL` macro expansion.
   **Key Information**: `SFPU_ZERO_KERNEL(OP, MODE, APPROXIMATE, DST_IDX)` expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_comp<APPROXIMATE, SfpuType::OP>, DST_IDX, (int)VectorMode::MODE, 8)` -- the `8` is the iteration count (8 vectors x 4 elements = 32 rows, covering all tile faces).
