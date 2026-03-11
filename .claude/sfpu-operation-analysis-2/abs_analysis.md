# ABS (Absolute Value) Implementation Analysis

## Overview

The ABS operation computes the element-wise absolute value of an input tensor. It is implemented as a unary SFPU operation using the shared `UnaryProgramFactory`, which provides a generic framework for all unary element-wise operations. The ABS-specific behavior is injected via preprocessor defines that expand to `abs_tile_init()` and `abs_tile(idst)` calls in the compute kernel.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Work Unit Definition

One work unit is **one tile** (32x32 elements) for TILE layout, or **one row-page** for ROW_MAJOR layout. Pages are distributed across cores and processed sequentially within each core. The compute kernel processes one tile at a time (`per_core_block_size = 1`), iterating over the total number of tiles assigned to that core (`per_core_block_cnt`).

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|---|---|
| Dimension Convention | N-dimensional (flattened to pages) |
| Tensor Layout | TILE (32x32) or ROW_MAJOR |
| Memory Layout | Interleaved |
| Buffer Type | DRAM or L1 |
| Data Type | BFLOAT16, FLOAT32, BFLOAT8_B, BFLOAT4_B |

### Output Tensor

| Property | Value |
|---|---|
| Dimension Convention | Same as input |
| Tensor Layout | Same as input |
| Memory Layout | Interleaved |
| Buffer Type | DRAM or L1 |
| Data Type | Same as input (or may differ for chained ops) |

### Layout Transformations

No layout transformations are performed. The input and output share the same tensor layout and memory layout. The operation is purely element-wise.

## Data Flow Pattern

1. **Reader kernel** reads one page at a time from DRAM/L1 via NoC into CB c_0 (input circular buffer).
2. **Compute kernel** waits for one tile in CB c_0, copies it to DST register via `copy_tile`, executes `abs_tile_init()` then `abs_tile(0)` on the SFPU, packs the result from DST into CB c_2 (output circular buffer).
3. **Writer kernel** waits for one page in CB c_2, writes it back to DRAM/L1 via NoC, then pops the page.

This is a straightforward single-tile-at-a-time pipeline: Reader -> Compute -> Writer.

## Circular Buffer Configuration

| CB ID | Index | Purpose | Page Size | Num Pages | Total Size | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|---|
| c_0 | 0 | Input tiles | tile_size(input_dtype) | 2 | 2 * page_size | Double | Reader | Compute |
| c_2 | 2 | Output tiles | tile_size(output_dtype) | 2 | 2 * page_size | Double | Compute | Writer |

**Notes**:
- CB c_1 (index 1) is only allocated for HARDSHRINK and LOGIT operations, NOT for ABS.
- Both CBs have capacity for 2 pages but are produced/consumed 1 page at a time, enabling double-buffering.
- For TILE layout, page size equals `tile_size(data_format)`. For ROW_MAJOR, page size equals `buffer->page_size()`.

## Pipeline Pattern Summary

Both c_0 and c_2 are configured with 2 pages capacity and 1-page block size. This is a **double-buffered** configuration, allowing overlap between:
- Reader writing the next tile into c_0 while Compute processes the current tile
- Compute writing the next result into c_2 while Writer drains the current result

## Index Calculations

Index mapping uses the `TensorAccessor` abstraction. The reader and writer kernels both use a linear page index starting from `start_id` (a runtime argument). Each core processes a contiguous range of pages: `[start_id, start_id + num_pages)`.

The `TensorAccessor` is constructed from `TensorAccessorArgs` (compile-time) plus the buffer address and page size (runtime). It handles the mapping from a linear page index to the physical DRAM bank address via `noc_async_read_page(page_id, accessor, l1_addr)`.

## Memory Access Patterns

### Read Pattern

- **Sequential**: Pages are read in order from `start_id` to `start_id + num_pages - 1`.
- **Granularity**: One page per NoC read transaction.
- **Synchronization**: `noc_async_read_barrier()` is called after each page read (no pipelining of NoC reads within the reader).
- **Source**: Interleaved DRAM or L1, accessed via TensorAccessor which resolves bank-interleaved addressing.

### Write Pattern

- **Sequential**: Pages are written in the same order as read.
- **Granularity**: One page per NoC write transaction.
- **Synchronization**: `noc_async_writes_flushed()` after each page, `noc_async_write_barrier()` at the end.
- **Destination**: Interleaved DRAM or L1, accessed via TensorAccessor.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Column-major iteration over `compute_with_storage_grid_size` |
| Work Splitting | `split_work_to_cores(grid_size, num_pages)` |
| Core Group 1 | Cores with `ceil(num_pages / num_cores)` pages each |
| Core Group 2 | Cores with `floor(num_pages / num_cores)` pages each (may be empty) |
| Remainder Handling | Extra pages distributed to core_group_1; core_group_2 gets one fewer page |
| Core Indexing | `core = {i / num_cores_y, i % num_cores_y}` (column-major) |

The two core groups get separate compute kernel instances with different `per_core_block_cnt` compile-time arguments, but identical kernel code and defines.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0+ | TensorAccessorArgs | uint32_t[] | Tensor accessor parameters for input buffer (bank mapping, interleaving info) |

#### Writer Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_out | uint32_t | Output circular buffer index (always 2 / c_2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Tensor accessor parameters for output buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | per_core_block_cnt | uint32_t | Number of tiles (pages) this core must process |
| 1 | per_core_block_size | uint32_t | Tiles per block (always 1 for standard unary) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src_addr | uint32_t | Source buffer DRAM/L1 address |
| 1 | num_pages | uint32_t | Number of pages this core processes |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Destination buffer DRAM/L1 address |
| 1 | num_pages | uint32_t | Number of pages this core processes |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Compute Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | packed_scalar1 | uint32_t | Unused for ABS (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for ABS (always 0) |

## Kernel Implementations

### Reader Kernel

| Property | Value |
|---|---|
| File | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` |
| Type | ReaderDataMovementConfig |
| Assigned Cores | all_cores |

**Key Logic**: Simple sequential page reader. For each page in `[start_id, end_id)`: reserves space in CB c_0, issues a NoC async read, waits for completion, then pushes the page to the consumer (compute kernel). Supports a `BACKWARDS` mode via preprocessor define (not used for ABS).

### Compute Kernel

| Property | Value |
|---|---|
| File | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` |
| Type | ComputeConfig |
| Assigned Cores | core_group_1 and core_group_2 (separate kernel handles) |

**Key Logic**: Generic SFPU eltwise kernel. For each block (tile):
1. `tile_regs_acquire()` - acquire DST register file
2. `cb_wait_front(c_0, 1)` - wait for input tile
3. `copy_tile(c_0, 0, 0)` - unpack tile from CB to DST register 0
4. Execute `SFPU_OP_CHAIN_0` macro which expands to `abs_tile_init(); abs_tile(0);`
5. `tile_regs_commit()` / `tile_regs_wait()` - synchronize math pipeline
6. `pack_tile(0, c_2)` - pack result from DST to output CB
7. `cb_pop_front(c_0, 1)` - release input tile

**Compute Config**:
- `math_fidelity`: HiFi4
- `math_approx_mode`: false (ABS returns false from `get_op_approx_mode`)
- `fp32_dest_acc_en`: configurable via operation attributes

**Defines injected**:
- `SFPU_OP_CHAIN_0_INIT_0` = `abs_tile_init();`
- `SFPU_OP_CHAIN_0_FUNC_0` = `abs_tile(0);`
- `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` = `1` (ABS uses the default compute_kernel_api include)
- `SFPU_OP_CHAIN_0` = `SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0`
- One of: `INP_FLOAT32`, `INP_INT32`, `INP_UINT32`, or `INP_FLOAT` depending on input dtype

### Writer Kernel

| Property | Value |
|---|---|
| File | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` |
| Type | WriterDataMovementConfig |
| Assigned Cores | all_cores |

**Key Logic**: Sequential page writer. For each page in `[start_id, end_id)`: waits for compute to push a page into CB c_2, reads the L1 address, issues a NoC async write, flushes, then pops the page. Final `noc_async_write_barrier()` ensures all writes complete.

## Implementation Notes

1. **ABS is a non-parameterized operation**: It requires no scalar parameters (`packed_scalar1` and `packed_scalar2` are both 0). The runtime args to the compute kernel are unused.

2. **SFPU execution**: `abs_tile` and `abs_tile_init` map to LLK functions `llk_math_eltwise_unary_sfpu_abs` and `llk_math_eltwise_unary_sfpu_abs_init`, which execute on the SFPU (vector engine) of each Tensix core.

3. **Program caching**: The factory supports `override_runtime_arguments` which only updates buffer addresses, enabling efficient program reuse when tensor shapes remain the same but buffer locations change.

4. **Two program factory variants**: `UnaryProgramFactory` (analyzed here) uses full device grid with `split_work_to_cores`. `UnarySubCoreGridProgramFactory` allows specifying a subset of cores but requires uniform tile distribution (num_tiles must be divisible by num_cores).

5. **No intermediate buffer**: ABS does not require CB c_1 (the temporary buffer), unlike HARDSHRINK or LOGIT which need intermediate storage.

6. **INT32 variant**: A separate `UnaryOpType::ABS_INT32` exists that calls `abs_tile_int32()` instead, but uses the same `abs_tile_init()`.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary operation program factory work in TTNN? What kernels does it use for reader, compute, and writer? How are circular buffers configured for unary elementwise operations?"
   **Reason**: Needed architectural understanding of the unary program factory pattern before reading source code.
   **Key Findings**: Confirmed three-kernel architecture (reader/compute/writer), three CB indices (c_0, c_1, c_2), and the existence of three factory variants (standard, sub-core-grid, sharded). Confirmed that compute kernel path is dynamically selected based on operation type.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to understand how ABS maps to SFPU functions and which compute kernel file is used.
   **Key Information**: ABS maps to `abs_tile_init()` / `abs_tile(idst)`, uses default compute kernel `eltwise_sfpu.cpp`, uses `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` macro, and `get_op_approx_mode` returns false.

2. **Source**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h`
   **Reason**: Needed to confirm the hardware-level implementation of `abs_tile` and `abs_tile_init`.
   **Key Information**: `abs_tile(idst)` calls `llk_math_eltwise_unary_sfpu_abs<APPROX>(idst)` and `abs_tile_init()` calls `llk_math_eltwise_unary_sfpu_abs_init<APPROX>()`.

3. **Source**: `tt_metal/api/tt-metalium/work_split.hpp`
   **Reason**: Needed to understand core distribution strategy.
   **Key Information**: `split_work_to_cores` returns two core groups - one doing more work and one doing less when work cannot be evenly divided. Column-major core ordering by default.

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the ABS (absolute value) operation.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_abs.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `abs_tile(idst)` from `compute_kernel_api.h`, which wraps the call in the `MATH((...))` macro to execute on the math RISC-V processor: `MATH((llk_math_eltwise_unary_sfpu_abs<APPROX>(idst)))`.
2. `llk_math_eltwise_unary_sfpu_abs<APPROXIMATE>` (in `llk_math_eltwise_unary_sfpu_abs.h`) calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_abs<APPROXIMATE>, dst_index, vector_mode)`, passing the SFPU kernel as a function pointer.
3. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) sets the DEST write address, stalls until SFPU is ready, then iterates over tile faces (4 faces in RC mode), calling the SFPU kernel function once per face and advancing the DEST face address between calls.
4. `calculate_abs<APPROXIMATION_MODE, 8>()` (in `ckernel_sfpu_abs.h`) loops 8 iterations: each iteration loads a vector from `dst_reg[0]` (implicit SFPLOAD), applies `sfpi::abs()` which emits the `SFPABS` instruction with `SFPABS_MOD1_FLOAT` mode, stores the result back via `dst_reg[0] =` (implicit SFPSTORE), and increments the DEST pointer via `dst_reg++` (INCRWC by stride 2).

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs() { // APPROXIMATION_MODE is unused for abs (no approximation needed)
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];       // SFPLOAD: load 32 elements from DEST into LReg
        dst_reg[0] = sfpi::abs(v);   // SFPABS(Mod1=SFPABS_MOD1_FLOAT=1): clear sign bit; then SFPSTORE result back to DEST
        dst_reg++;                   // INCRWC: advance DEST read/write counter by SFP_DESTREG_STRIDE=2
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() { // APPROXIMATION_MODE is unused
    // SFPU microcode — int32 variant uses explicit instruction macros
    for (int d = 0; d < ITERATIONS; d++) {
        // Wormhole: TT_SFPLOAD(1, 4, 3, 0)  — Mod0=4 (BOB32: bag-of-bits 32-bit), AddrMod=ADDR_MOD_3, DEST addr=0, into LReg1
        // Blackhole: TT_SFPLOAD(1, 12, ADDR_MOD_7, 0) — Mod0=12 (BOB32 on BH), AddrMod=ADDR_MOD_7, DEST addr=0, into LReg1
        TT_SFPLOAD(1, /*mod0_fmt*/, /*addr_mod*/, 0);
        TTI_SFPABS(0, 1, 0, 0);     // SFPABS: src=LReg1, dst=LReg0, Mod1=0 (SFPABS_MOD1_INT: two's complement abs)
        TTI_SFPSTORE(0, /*mod0_fmt*/, /*addr_mod*/, 0); // SFPSTORE: LReg0 back to DEST with same format/addr_mod
        dst_reg++;                   // INCRWC: advance DEST pointer by stride 2
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

**Note on `calculate_abs_int32` arch differences**: On Wormhole B0, the SFPLOAD/SFPSTORE use `Mod0=4` (`SFPLOAD_MOD0_FMT_BOB32`) and `AddrMod=3`; on Blackhole, they use `Mod0=12` (`SFPLOAD_MOD0_FMT_BOB32` on BH, which has a different constant value) and `AddrMod=ADDR_MOD_7`. Both achieve the same semantic: load/store raw 32-bit integers without format conversion.

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| **SFPLOAD** | Moves 32 datums from DEST register file into an SFPU LReg, with optional data type conversion. Used implicitly by `vFloat v = dst_reg[0]` (float path) and explicitly by `TT_SFPLOAD` (int32 path). |
| **SFPABS** | Computes lanewise absolute value on LReg data. With `Mod1=1` (SFPABS_MOD1_FLOAT), clears the sign bit of IEEE 754 floats (preserves -NaN). With `Mod1=0` (SFPABS_MOD1_INT), performs two's complement negation for negative integers. Executes in 1 cycle on the SFPU simple sub-unit. |
| **SFPSTORE** | Moves 32 datums from an SFPU LReg back to the DEST register file, with optional data type conversion. Used implicitly by `dst_reg[0] = ...` (float path) and explicitly by `TTI_SFPSTORE` (int32 path). |
| **INCRWC** | Increments the DEST read-write counter by `SFP_DESTREG_STRIDE` (=2). Emitted by `dst_reg++` via `__builtin_rvtt_ttincrwc(0, 2, 0, 0)`. Advances the SFPU's view of which DEST rows to access next. |
| **SETRWC** | Sets/resets read-write counters. Used in `_llk_math_eltwise_unary_sfpu_params_` between faces to advance the DEST address by 16 rows (two `SETRWC` calls incrementing by 8 each) to move to the next 16x16 face. |
| **STALLWAIT** | Stalls the math pipeline until the SFPU is ready (before kernel execution) and waits for SFPU completion (after kernel execution). Used in the params dispatcher. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST register file** | Source and destination for tile data. The SFPU reads from and writes back to DEST. A 32x32 tile occupies 4 faces of 16x16 elements each. The SFPU processes one face per `calculate_abs` invocation (8 iterations x 32 lanes / stride-2 addressing = 16 rows x 16 columns). |
| **LReg0** | Used as destination for `SFPABS` in the int32 path. In the float path, the compiler allocates LRegs automatically via SFPI builtins. |
| **LReg1** | Used as source for `SFPABS` in the int32 path (`TT_SFPLOAD` loads into LReg1, `TTI_SFPABS` reads from LReg1). |
| **dst_reg pointer** | Virtual register file accessor. `dst_reg[0]` reads/writes at the current DEST offset; `dst_reg++` advances the offset by `SFP_DESTREG_STRIDE=2` rows. Over 8 iterations, this covers 16 rows of one 16x16 face. |

### Address Mode Configuration

The ABS operation uses `ADDR_MOD_7`, configured during initialization via `eltwise_unary_sfpu_configure_addrmod<SfpuType::abs>()`.

**ADDR_MOD_7 configuration** (same for both Wormhole B0 and Blackhole):
```cpp
addr_mod_t {
    .srca = {.incr = 0},   // No auto-increment for source A
    .srcb = {.incr = 0},   // No auto-increment for source B
    .dest = {.incr = 0},   // No auto-increment for DEST
}.set(ADDR_MOD_7);
```

The DEST increment is 0 because the SFPU kernel manages DEST advancement explicitly via `dst_reg++` (INCRWC instructions). The address mode simply avoids conflicting with ADDR_MOD_0 and ADDR_MOD_2, which are used by the A2D (Accumulate to DEST) pipeline.

The ABS operation does not fall into any special-case address mode configurations (no ADDR_MOD_6 setup) -- those are reserved for operations like `typecast`, `topk_local_sort`, and min/max operations.

**Minor arch difference in `_llk_math_eltwise_unary_sfpu_start_`**: On Wormhole B0, this function calls both `math::set_dst_write_addr` and `math::set_addr_mod_base()` before the stall; on Blackhole, it only calls `math::set_dst_write_addr` and the stall (no explicit `set_addr_mod_base`). Similarly, `_llk_math_eltwise_unary_sfpu_done_` on Wormhole includes `clear_addr_mod_base()` and a STALLWAIT, while Blackhole only clears the DEST register address.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the ABS unary operation's compute kernel invoke the SFPU? Trace the call chain from abs_tile through LLK dispatch down to the ckernel SFPU implementation."
   **Reason**: Needed to understand the full call chain and identify all files involved in the SFPU abstraction layers.
   **Key Findings**: Identified the 4-layer abstraction (API -> LLK dispatch -> params dispatcher -> ckernel SFPU). Confirmed `calculate_abs` uses `sfpi::abs()` and `calculate_abs_int32` uses `TTI_SFPABS`.

2. **Query**: "How is the SFPU abs operation implemented in the LLK/ckernel layer? What SFPU instructions does it use?" (tenstorrent/tt-llk)
   **Reason**: Needed LLK-specific details on initialization, face iteration, and instruction dispatch.
   **Key Findings**: Confirmed the `_llk_math_eltwise_unary_sfpu_params_` face iteration pattern and the `SFPABS` hardware instruction.

3. **Query**: "How is sfpi::abs() implemented for vFloat? What SFPU instruction does it use? Also explain dst_reg and dst_reg++." (tenstorrent/sfpi)
   **Reason**: Needed to understand the SFPI library implementation of abs and the DEST register access mechanism.
   **Key Findings**: `sfpi::abs(vFloat)` calls `__builtin_rvtt_sfpabs` with `SFPABS_MOD1_FLOAT=1`. `dst_reg[0]` triggers SFPLOAD/SFPSTORE. `dst_reg++` emits INCRWC with stride `SFP_DESTREG_STRIDE=2`.

4. **Query**: "What is the SFPABS instruction? What are its operands? Also describe SFPLOAD and SFPSTORE." (tenstorrent/tt-isa-documentation)
   **Reason**: Needed authoritative ISA-level details on the instructions used.
   **Key Findings**: SFPABS takes VC (src LReg), VD (dst LReg), Mod1 (int vs float mode). Float mode clears sign bit, preserves -NaN. SFPLOAD/SFPSTORE move data between DEST and LRegs with format conversion. All execute in 1 cycle.

### Confluence References
Not consulted -- DeepWiki and ISA documentation provided sufficient detail for this simple operation.

### Glean References
Not consulted -- no confidential specifications were needed beyond what DeepWiki and source code provided.
