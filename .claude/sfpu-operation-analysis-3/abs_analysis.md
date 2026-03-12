# ABS Implementation Analysis

## Overview
The ABS operation computes the element-wise absolute value of each element in an input tensor. It is implemented as a unary SFPU operation that processes tiles through the standard unary program factory pipeline: reader kernel fetches tiles from DRAM into L1 circular buffers, the compute kernel applies the SFPU abs function to each tile in the DST register, and the writer kernel writes results back to DRAM.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Path Selection: FPU vs SFPU

ABS is a pure SFPU operation -- there is no FPU path. The program factory does not distinguish between FPU and SFPU at the factory level; instead, the distinction is embedded in the compute kernel selection. The function `utils::get_compute_kernel_path()` maps `UnaryOpType::ABS` to the `default` case, which returns `"eltwise_sfpu.cpp"`. This is the generic SFPU compute kernel shared by many unary operations. The specific SFPU function (`abs_tile`) is injected via preprocessor defines: `SFPU_OP_CHAIN_0` expands to `abs_tile_init(); abs_tile(0);`. The macro `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` is also defined (ABS falls into the `default` case of `get_macro_definition`), which controls the header inclusion path in the split-include mechanism.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `num_pages` = total tiles in the input tensor |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles (always 1 for the default factory) |

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | Arbitrary (any rank) | Same as input |
| **Dimension convention** | N/A (element-wise) | N/A |
| **Tensor layout** | TILE_LAYOUT or ROW_MAJOR | Same as input |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32 | Same as input |

### Layout Transformations
None. The operation preserves input layout and data type. For INT32 inputs, a separate SFPU function (`abs_tile_int32`) is used that operates on integer data directly using SFPU load/abs/store instructions instead of the SFPI `abs()` intrinsic.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `noc_async_read_barrier`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_reserve_back(c_2, per_core_block_dim)`, `cb_wait_front(c_0, 1)`, `copy_tile`, `abs_tile`, `pack_tile`, `cb_pop_front(c_0, 1)`, `cb_push_back(c_2, per_core_block_dim)` |
| 3 | Writer | CB c_2 | DRAM (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `noc_async_writes_flushed`, `cb_pop_front(c_2, 1)` |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | src0 | Input staging | 2 pages | 1 page | Double | Reader | Compute | Program |
| c_2 | output | Output staging | 2 pages | 1 page | Double | Compute | Writer | Program |

Note: CB c_1 (tmp0) is NOT created for ABS. It is only allocated for HARDSHRINK, CBRT, or LOGIT operations.

## Pipeline Pattern Summary
Both input and output circular buffers are double-buffered (capacity = 2 pages, block size = 1 page). This allows the reader to fill one slot while compute processes the other, and similarly compute can produce into one output slot while the writer drains the other. This enables full overlap between the three pipeline stages under steady-state conditions.

## Index Calculations
The reader and writer kernels use `TensorAccessor` for index-to-address mapping. The `TensorAccessorArgs` are passed as compile-time arguments and encode the buffer's memory layout (interleaved bank mapping). The `noc_async_read_page(i, s, l1_write_addr)` call translates page index `i` into the correct DRAM bank address and offset using the tensor accessor `s`. Page indices are sequential starting from `start_id` (a per-core runtime argument) up to `start_id + num_pages`.

## Memory Access Patterns

### Read Pattern
Sequential page access. Each core reads a contiguous range of page indices `[start_id, start_id + num_pages_per_core)`. Pages are read one at a time with a NoC read barrier after each page, ensuring ordering.

### Write Pattern
Sequential page access matching the read pattern. Each core writes the same contiguous range of page indices. Writes use `noc_async_writes_flushed()` (non-blocking flush) per page, with a final `noc_async_write_barrier()` after the loop to ensure all writes complete.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size` (device-dependent) |
| **Total cores** | Determined by `split_work_to_cores()` |
| **Work per core** | `num_pages / num_cores` tiles (group 1 gets ceil, group 2 gets floor) |
| **Load balancing** | Two-group split: core_group_1 gets `num_pages_per_core_group_1` tiles, core_group_2 gets `num_pages_per_core_group_2` tiles |

Cores are enumerated in column-major order: `core = {i / num_cores_y, i % num_cores_y}`. The `split_work_to_cores` utility divides total pages across available cores, with remainder pages distributed to core_group_1 (each core in group 1 processes one more tile than cores in group 2).

## Arguments

### Compile-Time Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Encoded tensor accessor parameters for source buffer (bank mapping, page size, etc.) |

**Writer kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | Output circular buffer ID (c_2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encoded tensor accessor parameters for destination buffer |

**Compute kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tiles) this core processes |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1 for ABS in the default factory) |

### Runtime Arguments

**Reader kernel (per core):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_pages | uint32_t | Number of pages this core reads |
| 2 | start_id | uint32_t | Starting page index for this core |

**Writer kernel (per core):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM |
| 1 | num_pages | uint32_t | Number of pages this core writes |
| 2 | start_id | uint32_t | Starting page index for this core |

**Compute kernel (per core):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for ABS (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for ABS (always 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM | CB c_0 | Read tiles via TensorAccessor |
| Compute | TRISC (RISCV_2) | N/A | CB c_0 | CB c_2 | copy_tile, abs_tile (SFPU), pack_tile |
| Writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM | Write tiles via TensorAccessor |

### Reader Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` |
| **Assigned cores** | all_cores (both core_group_1 and core_group_2) |

**Key Logic:**
- Iterates sequentially over page indices from `start_id` to `start_id + num_pages`
- For each page: reserves 1 slot in CB c_0, reads page from DRAM via `noc_async_read_page`, waits for completion with `noc_async_read_barrier`, then pushes 1 page to CB c_0
- Page size is obtained dynamically from the CB interface (`get_local_cb_interface(cb_id_in0).fifo_page_size`), making the kernel layout-agnostic
- Supports optional `BACKWARDS` define for reverse iteration (not used for ABS)
- **Synchronization**: Produces into CB c_0 using `cb_reserve_back` / `cb_push_back`. Blocks on `cb_reserve_back` if the CB is full (both slots occupied by unconsumed data).

### Compute Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` |
| **Assigned cores** | core_group_1 gets kernel with `per_core_block_cnt = num_pages_per_core_group_1`; core_group_2 (if non-empty) gets separate kernel instance with `per_core_block_cnt = num_pages_per_core_group_2` |

**Key Logic:**
- Calls `init_sfpu(c_0, c_2)` to initialize unpack, math, and pack pipelines
- Outer loop iterates `per_core_block_cnt` times (one iteration per tile for ABS)
- Inner loop iterates `per_core_block_dim` times (always 1 for ABS default factory)
- Per tile: acquires DST registers (`tile_regs_acquire`), waits for input tile in CB c_0, copies tile from CB c_0 to DST register 0 via `copy_tile`, executes SFPU operation chain (`SFPU_OP_CHAIN_0` which expands to `abs_tile_init(); abs_tile(0);`), commits DST registers, waits for pack availability, packs tile 0 from DST to CB c_2, pops input tile from CB c_0, releases DST registers
- The `abs_tile_init()` call configures the SFPU for the abs operation; `abs_tile(0)` applies abs to DST register index 0
- At the SFPU microcode level (`calculate_abs`), the operation iterates 8 times (ITERATIONS=8, processing 8 datum rows of the 32x32 tile), reading `dst_reg[0]`, applying `sfpi::abs(v)`, writing back, and advancing the dst_reg pointer
- For INT32 data type, `abs_tile_int32` is used instead, which uses explicit SFPU instructions: `SFPLOAD` to load from DST to SFPU register, `SFPABS` to compute absolute value, `SFPSTORE` to store back
- **Synchronization**: Consumes from CB c_0 (`cb_wait_front` / `cb_pop_front`), produces into CB c_2 (`cb_reserve_back` / `cb_push_back`). The `cb_reserve_back` for CB c_2 is issued at the start of the outer loop for the entire block (1 tile), and `cb_push_back` at the end.

### Writer Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` |
| **Assigned cores** | all_cores (both core_group_1 and core_group_2) |

**Key Logic:**
- Iterates sequentially over page indices from `start_id` to `start_id + num_pages`
- For each page: waits for 1 page in CB c_2, reads L1 address from CB, writes page to DRAM via `noc_async_write_page`, flushes with `noc_async_writes_flushed`, then pops 1 page from CB c_2
- After the loop, issues `noc_async_write_barrier()` to ensure all writes are globally visible
- Supports `OUT_SHARDED` define (not used for interleaved path) which simply waits for all pages in the output CB without writing to DRAM
- Supports `BACKWARDS` define for reverse iteration (not used for ABS)
- **Synchronization**: Consumes from CB c_2 using `cb_wait_front` / `cb_pop_front`. Blocks on `cb_wait_front` if the CB is empty.

## Implementation Notes

- **Program factory variants**: Three program factories exist: `UnaryProgramFactory` (default, for interleaved tensors), `UnarySubCoreGridProgramFactory` (when `sub_core_grids` is specified), and `UnaryShardedProgramFactory` (for sharded inputs). Selection is in `UnaryDeviceOperation::select_program_factory()`: sharded input selects sharded factory, `sub_core_grids.has_value()` selects sub-core-grid factory, otherwise the default factory is used.
- **Type-based operation variants**: ABS supports BFLOAT16, FLOAT32, INT32, and UINT32 data types. For INT32, `abs_tile_int32` is used (explicit SFPU load/abs/store instructions). For floating-point types, `abs_tile` uses the SFPI `abs()` intrinsic. Data-type-specific preprocessor defines (`INP_FLOAT32`, `INP_INT32`, `INP_UINT32`, `INP_FLOAT`) are set based on input dtype.
- **UnpackToDestFP32 mode**: Enabled when `args.preserve_fp32_precision` is true. This sets `UnpackToDestMode::UnpackToDestFp32` for CB c_0 and c_1, causing the unpacker to write FP32 values directly to the DST register.
- **Broadcast type selection**: N/A. ABS is a unary element-wise operation with no broadcasting.
- **Sharding support and constraints**: Sharded inputs are handled by `UnaryShardedProgramFactory` (a separate factory not analyzed in depth here). The default `UnaryProgramFactory` handles only interleaved tensors.
- **FP32 dest accumulation**: Controlled by `args.fp32_dest_acc_en` and passed to `ComputeConfig`. When enabled, the DST register operates in FP32 mode for higher precision accumulation.

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_abs.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Wormhole) / `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu.h` (Blackhole, contains `_start_`/`_done_`/`_inc_dst_face_addr_`) |

### Call Chain

1. The compute kernel calls `abs_tile(0)`, defined in `compute_kernel_api.h`, which dispatches via the `MATH()` macro to `llk_math_eltwise_unary_sfpu_abs<APPROX>(idst)`.
2. `llk_math_eltwise_unary_sfpu_abs` (in `llk_math_eltwise_unary_sfpu_abs.h`) calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>` with `ckernel::sfpu::calculate_abs<APPROXIMATE>` as the callable and `dst_index` as the target tile, using `VectorMode::RC` (default).
3. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) sets the DST write address, stalls until SFPU is ready, then loops over 4 faces calling `calculate_abs` once per face, advancing the DST face address by 16 rows between faces.
4. `calculate_abs` (in `ckernel_sfpu_abs.h`) iterates 8 times over datum rows within one face, reading from `dst_reg[0]`, applying `sfpi::abs(v)`, writing back, and incrementing `dst_reg` (advancing by `SFP_DESTREG_STRIDE=2` rows per iteration).
5. For the init path: `abs_tile_init()` calls `llk_math_eltwise_unary_sfpu_abs_init<APPROX>()`, which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::abs, APPROXIMATE>()`, which calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::abs>()` to initialize the SFPU config register, configure ADDR_MOD_7, and reset counters.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the 32x32 tile are processed. Each face is a 16x16 sub-tile.
- **Operation invocation**: The dispatch loops 4 times (once per face). Each iteration calls `calculate_abs()` which processes 8 datum rows (the face's 16 rows, with the SFPU processing 2 rows per iteration due to `SFP_DESTREG_STRIDE=2`). Between faces, DST address is advanced by 16 rows.
- **DEST address progression**:
  - **Wormhole**: Uses explicit `TTI_SETRWC` instructions to advance the DST write counter by 8 twice per face (total +16 rows per face). After the SFPU function completes, `math::clear_dst_reg_addr()` resets addressing. A final `TTI_STALLWAIT(STALL_CFG, WAIT_SFPU)` and `math::clear_addr_mod_base()` are issued.
  - **Blackhole**: Uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice per face (total +16 rows per face). The `_start_` / `_done_` functions handle stalling, DST address setup, and cleanup.

### Annotated SFPU Kernel Source

**Floating-point variant** (SFPI-based -- Style A):

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h
// (Identical implementation on Wormhole B0)

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs() { // APPROXIMATION_MODE is unused; ITERATIONS=8
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];       // SFPLOAD: load 2 rows from DST into LREG
        dst_reg[0] = sfpi::abs(v);   // SFPABS(SFPABS_MOD1_FLOAT): clear sign bit, write back to DST
        dst_reg++;                   // SETRWC: advance DST pointer by SFP_DESTREG_STRIDE=2 rows
    }
}
```

**INT32 variant** (TT_/TTI_-based -- Style A, simple CC logic):

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() { // APPROXIMATION_MODE is unused; ITERATIONS=8
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(1, 12, ADDR_MOD_7, 0);  // Load LREG1 from DST row 0, instr_mod0=12 (INT32_2S_COMP), addr_mode=ADDR_MOD_7
        TTI_SFPABS(0, 1, 0, 0);             // LREG0 = abs(LREG1), imm12_math=0, instr_mod1=0 (integer mode)
        TTI_SFPSTORE(0, 12, ADDR_MOD_7, 0); // Store LREG0 to DST row 0, instr_mod0=12 (INT32_2S_COMP), addr_mode=ADDR_MOD_7
        dst_reg++;                           // SETRWC: advance DST pointer by 2 rows
    }
}
```

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h
// (Wormhole B0 INT32 variant -- differs in SFPLOAD/SFPSTORE encoding)

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() { // APPROXIMATION_MODE is unused; ITERATIONS=8
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(1, 4, 3, 0);     // Load LREG1 from DST row 0, instr_mod0=4 (INT32), sfpu_addr_mode=3
        TTI_SFPABS(0, 1, 0, 0);     // LREG0 = abs(LREG1), imm12_math=0, instr_mod1=0 (integer mode)
        TTI_SFPSTORE(0, 4, 3, 0);   // Store LREG0 to DST row 0, instr_mod0=4 (INT32), sfpu_addr_mode=3
        dst_reg++;                   // SETRWC: advance DST pointer by 2 rows
    }
}
```

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| **SFPLOAD** (implicit via `dst_reg[0]` read) | Loads a vector of elements from the current DST register row into an SFPU local register (LREG). For the float path, this is generated by the SFPI compiler from `vFloat v = dst_reg[0]`. For INT32, `TT_SFPLOAD` is called explicitly with `instr_mod0=INT32` (WH) or `INT32_2S_COMP` (BH). |
| **SFPABS** (via `sfpi::abs(v)` or `TTI_SFPABS`) | Computes absolute value. With `instr_mod1=SFPABS_MOD1_FLOAT` (=1), clears the sign bit of IEEE 754 floats. With `instr_mod1=SFPABS_MOD1_INT` (=0), computes integer absolute value. Takes source from `lreg_c` and writes result to `lreg_dest`. |
| **SFPSTORE** (implicit via `dst_reg[0]` write) | Stores a vector from an SFPU local register back to the current DST register row. For float, generated by SFPI from `dst_reg[0] = ...`. For INT32, `TTI_SFPSTORE` is called explicitly. |
| **SETRWC** (via `dst_reg++`) | Increments the DST register write counter by `SFP_DESTREG_STRIDE` (=2), advancing to the next pair of datum rows. Compiles to `__builtin_rvtt_ttincrwc(0, 2, 0, 0)`. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DST register** | The tile data resides in DST after `copy_tile` moves it from CB c_0. The SFPU reads from and writes back to DST in-place. Each `dst_reg[0]` access targets the current row pair (2 rows of 16 elements = 32 elements per SFPU vector operation). |
| **LREG0** | In the INT32 path, used as the destination for `SFPABS` output. In the float path, the SFPI compiler allocates LREGs automatically. |
| **LREG1** | In the INT32 path, used as the load target for `SFPLOAD` and source for `SFPABS`. |
| **LREG (compiler-managed)** | In the float path, `sfpi::abs(v)` uses compiler-allocated local registers. The SFPI `abs()` intrinsic maps to `__builtin_rvtt_sfpabs(v.get(), SFPABS_MOD1_FLOAT)`, which the compiler lowers to an `SFPABS` instruction with appropriate LREG allocation. |

### Address Mode Configuration

**ADDR_MOD_7** is configured during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::abs>()` via `eltwise_unary_sfpu_configure_addrmod<SfpuType::abs>()`. Since `SfpuType::abs` does not match any special-case `constexpr if` branches (topk_local_sort, reciprocal, typecast, etc.), only the default ADDR_MOD_7 is set.

The configuration is identical on both Wormhole B0 and Blackhole:

```
ADDR_MOD_7:
  srca.incr = 0    (no SRC A auto-increment)
  srcb.incr = 0    (no SRC B auto-increment)
  dest.incr = 0    (no DST auto-increment via addr_mod)
```

ADDR_MOD_7 is set to all-zero increments because the SFPU kernel manages DST address progression explicitly via `dst_reg++` (SETRWC instruction) rather than relying on hardware auto-increment through the address mode. The INT32 path references ADDR_MOD_7 in the `sfpu_addr_mode` parameter of SFPLOAD/SFPSTORE on Blackhole (value 7), while on Wormhole B0 the `sfpu_addr_mode` field is only 2 bits so it uses value 3 instead. In both cases, the addr_mode selects ADDR_MOD_7's configuration (zero increments) for the load/store operations, with actual row advancement handled by the explicit `dst_reg++`.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the unary program factory work for SFPU operations? What is the structure of unary_program_factory.cpp and how does it select between FPU and SFPU paths?"
   **Reason**: Needed to understand the overall architecture of the unary program factory and how kernel selection works.
   **Key Findings**: The factory selection is based on memory layout (sharded vs interleaved), not FPU vs SFPU. The FPU/SFPU distinction is made at the kernel level through compile-time defines and kernel path selection. Three factories exist: UnaryProgramFactory, UnarySubCoreGridProgramFactory, and UnaryShardedProgramFactory.

2. [SFPU] **Query**: "How does the ABS (absolute value) unary SFPU operation work? Trace the call chain from the compute kernel API through LLK dispatch down to the ckernel SFPU implementation."
   **Reason**: Needed to understand the full SFPU call chain for ABS and identify which instructions are used.
   **Key Findings**: The call chain is `abs_tile()` -> `llk_math_eltwise_unary_sfpu_abs()` -> `_llk_math_eltwise_unary_sfpu_params_()` -> `calculate_abs()`. The float path uses `sfpi::abs(v)` which maps to `SFPABS` with `SFPABS_MOD1_FLOAT`. The INT32 path uses explicit `SFPLOAD`/`SFPABS`/`SFPSTORE` instructions.

3. [SFPU] **Query**: "How is the abs SFPU operation implemented in the LLK layer? Show the call chain from llk_math_eltwise_unary_sfpu through to the ckernel implementation."
   **Reason**: Needed LLK-level detail on the params dispatch, vector mode handling, and face iteration.
   **Key Findings**: The params dispatch in `_llk_math_eltwise_unary_sfpu_params_` handles VectorMode::RC by iterating 4 faces, calling the SFPU function once per face. DST address advances by 16 rows per face via `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_` (BH) or `TTI_SETRWC` (WH). The `SFPABS` instruction is the core hardware instruction.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: To trace the define generation and kernel path selection for ABS.
   **Key Information**: ABS maps to `abs_tile_init()/abs_tile(0)` for the SFPU operation chain, uses `eltwise_sfpu.cpp` as the compute kernel (default case), and `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` as the macro definition (default case).

2. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h`
   **Reason**: To understand the SFPU microcode implementation of abs.
   **Key Information**: The `calculate_abs` function iterates 8 times over dst_reg rows, applying `sfpi::abs(v)` to each vFloat. The INT32 variant uses explicit SFPLOAD/SFPABS/SFPSTORE instructions.

3. **Source**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h`
   **Reason**: To verify the high-level compute API for abs_tile.
   **Key Information**: `abs_tile(idst)` calls `llk_math_eltwise_unary_sfpu_abs<APPROX>(idst)` on the math RISC-V. `abs_tile_init()` calls the corresponding init function.

4. [SFPU] **Source**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: To trace the SFPI `abs()` intrinsic to its underlying hardware instruction.
   **Key Information**: `sfpi::abs(vFloat v)` maps to `__builtin_rvtt_sfpabs(v.get(), SFPABS_MOD1_FLOAT)` where `SFPABS_MOD1_FLOAT=1`. `sfpi::abs(vInt v)` maps to `__builtin_rvtt_sfpabs(v.get(), SFPABS_MOD1_INT)` where `SFPABS_MOD1_INT=0`.

5. [SFPU] **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` and `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: To understand the parameters dispatch differences between Blackhole and Wormhole.
   **Key Information**: Both use the same face-iteration pattern for VectorMode::RC (4 faces), but differ in DST address advancement mechanism: BH uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (abstracted), WH uses explicit `TTI_SETRWC` instructions. WH also includes `math::set_addr_mod_base()` / `math::clear_addr_mod_base()` around the SFPU execution.

### Confluence References
No Confluence references were needed for this analysis. The SFPABS instruction is straightforward (clear sign bit for float, negate-if-negative for int) and was fully documented via source code and DeepWiki.

### Glean References
No Glean references were needed for this analysis.
