# ABS (Absolute Value) Implementation Analysis

## Overview

The ABS operation computes the element-wise absolute value of a tensor. It is implemented as a unary SFPU operation within the generic unary eltwise program factory. For floating-point types, it uses the SFPI `abs()` intrinsic; for INT32, it uses the dedicated `SFPABS` instruction.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

The ABS operation is dispatched through `UnaryProgramFactory::create()`, which is a shared program factory for all unary eltwise operations. The specific operation is selected via preprocessor defines (`SFPU_OP_CHAIN_0`) injected at compute kernel compile time.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `num_pages` = total number of tiles in the input tensor |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles (always 1 for standard factory) |

In the standard `UnaryProgramFactory`, `per_core_block_dim` is hardcoded to 1, so each iteration of the outer loop processes exactly one tile. The total number of outer-loop iterations equals the number of tiles assigned to the core.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary (any rank) |
| **Dimension convention** | N-dimensional (flattened to pages) |
| **Tensor layout** | TILE_LAYOUT (or ROW_MAJOR) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | Same as input |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (or may differ for type-casting variants) |

### Layout Transformations

No layout transformations are performed. The operation is a pure element-wise computation -- data enters and exits in the same layout format. For TILE_LAYOUT, the CB page size equals the tile size. For ROW_MAJOR, the CB page size equals the buffer's page size.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile`, ABS SFPU op, `pack_tile`, `cb_pop_front(c_0, 1)`, `cb_push_back(c_2, per_core_block_dim)` |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

**Step-by-step flow**:
1. The **reader** kernel iterates over its assigned pages starting from `start_id`. For each page, it reserves space in CB c_0, performs a NoC async read to copy one page from the source buffer into L1, waits for the read to complete, then pushes the page.
2. The **compute** kernel acquires tile registers, waits for a tile in CB c_0, copies it into the DST register using `copy_tile`, executes `abs_tile_init()` followed by `abs_tile(0)` (injected via `SFPU_OP_CHAIN_0` define), commits tile registers, waits for pack availability, packs the result into CB c_2, pops the consumed tile from CB c_0, and releases tile registers.
3. The **writer** kernel waits for a tile in CB c_2, performs a NoC async write to copy the page to the destination buffer, then pops the tile from CB c_2.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | src0 | Input staging | 2 pages | 1 page | Double | Reader | Compute | Program |
| c_2 | output | Output staging | 2 pages | 1 page | Double | Compute | Writer | Program |

**Notes**:
- CB c_1 (tmp0) is only allocated for HARDSHRINK, CBRT, or LOGIT operations -- it is NOT used for ABS.
- "Pages" means tiles for TILE_LAYOUT or rows for ROW_MAJOR layout.
- Capacity of 2 pages with block size of 1 page enables double-buffering.

## Pipeline Pattern Summary

Both CB c_0 and CB c_2 have capacity = 2 x block_size, which is the **double-buffered** pattern. This allows the reader to write the next tile into CB c_0 while the compute kernel processes the current tile, and similarly allows compute to produce into CB c_2 while the writer drains the previous tile. This creates a 3-stage pipeline (reader -> compute -> writer) with overlap between adjacent stages.

## Index Calculations

The program factory uses `TensorAccessor` for mapping page indices to physical memory addresses. The reader and writer kernels both use a simple linear index scheme:

- `start_id`: The first page index assigned to this core (cumulative sum of pages assigned to prior cores).
- `end_id = start_id + num_pages`: The exclusive upper bound.
- Each page index `i` in `[start_id, end_id)` is passed to `noc_async_read_page(i, s, l1_write_addr)` or `noc_async_write_page(i, s, l1_read_addr)`.

The `TensorAccessor` object (constructed from `TensorAccessorArgs` compile-time args) translates the linear page index into the correct physical address accounting for interleaved bank distribution.

## Memory Access Patterns

### Read Pattern
- **Sequential**: Pages are read in strictly increasing order from `start_id` to `end_id - 1`.
- **Granularity**: One page per NoC transaction.
- **Barrier**: A `noc_async_read_barrier()` is issued after each single-page read, meaning reads are not pipelined within the reader kernel itself. The double-buffering in CB c_0 provides overlap between the reader and compute kernels instead.

### Write Pattern
- **Sequential**: Pages are written in strictly increasing order from `start_id` to `end_id - 1`.
- **Granularity**: One page per NoC transaction.
- **Flush**: `noc_async_writes_flushed()` is called after each write (not a full barrier), with a single `noc_async_write_barrier()` at the end of the loop. This allows limited write pipelining.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (flattened to 1D enumeration) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g. 8x8) |
| **Total cores** | min(grid_size.x * grid_size.y, num_pages) |
| **Work per core** | `num_pages / num_cores` or `num_pages / num_cores + 1` |
| **Load balancing** | Two-group split via `split_work_to_cores` |

**Core enumeration**: Cores are indexed as `core = {i / num_cores_y, i % num_cores_y}`, which is a column-major traversal of the 2D grid.

**Two-group strategy**:
- `core_group_1`: Contains `num_pages % num_cores` cores, each processing `floor(num_pages / num_cores) + 1` pages.
- `core_group_2`: Contains the remaining cores, each processing `floor(num_pages / num_cores)` pages.
- If pages divide evenly, `core_group_2` is empty.

Each core group gets its own compute kernel instance with different `per_core_block_cnt` compile-time arguments. The reader and writer kernels are shared across all cores, with per-core differences handled by runtime arguments.

## Arguments

### Compile-Time Arguments

**Reader kernel** (`reader_unary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Packed tensor accessor parameters for source buffer (memory layout, bank info, etc.) |

**Writer kernel** (`writer_unary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer index (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Packed tensor accessor parameters for destination buffer |

**Compute kernel** (`eltwise_sfpu.cpp`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tiles) to process on this core |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1 in standard factory) |

### Runtime Arguments

**Reader kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages this core must read |
| 2 | start_id | uint32_t | Starting page index for this core |

**Writer kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages this core must write |
| 2 | start_id | uint32_t | Starting page index for this core |

**Compute kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Packed scalar parameter (unused for ABS, set to 0) |
| 1 | packed_scalar2 | uint32_t | Packed scalar parameter (unused for ABS, set to 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM/L1 | CB c_0 | Read pages via TensorAccessor |
| compute | RISCV_2 (SFPU) | N/A | CB c_0 | CB c_2 | `copy_tile` -> `abs_tile` -> `pack_tile` |
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 | Write pages via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Uses `TensorAccessor` constructed from compile-time args to translate linear page indices to physical addresses. Reads one page at a time with a full read barrier between each read. Supports both forward and backward iteration (controlled by `BACKWARDS` define, not used in standard ABS).

### Compute Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **Key Logic**: Generic SFPU compute kernel. The actual operation is injected via the `SFPU_OP_CHAIN_0` preprocessor define, which for ABS expands to `abs_tile_init(); abs_tile(0);`. The kernel follows the standard tile register acquire/commit/wait/release protocol. It operates on one tile at a time (`per_core_block_dim = 1`), with `per_core_block_cnt` iterations.

### SFPU Implementation (ABS for float types)
- **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h`
- **Key Logic**: The `calculate_abs<APPROXIMATE>()` function iterates 8 times (one per face-row pair in a tile), loading from `dst_reg[0]`, applying `sfpi::abs(v)`, and writing back. The `sfpi::abs()` is a native SFPI intrinsic that clears the sign bit of floating-point values.

### SFPU Implementation (ABS for INT32)
- **File**: Same as above.
- **Key Logic**: Uses explicit SFPU instructions: `SFPLOAD` to load from DST, `SFPABS` to compute absolute value of integer, `SFPSTORE` to write back.

## Implementation Notes

1. **Math approximation mode**: For ABS, `get_op_approx_mode()` returns `false` (default case), so exact computation is used. This is expected since ABS is a simple sign-bit operation that does not benefit from approximation.

2. **Math fidelity**: Set to `MathFidelity::HiFi4` (highest fidelity). Since ABS only involves sign manipulation, fidelity does not materially affect the result.

3. **Data type handling**: The program factory detects the input dtype and sets corresponding defines (`INP_FLOAT32`, `INP_INT32`, `INP_UINT32`, or `INP_FLOAT`). For INT32 inputs, the operation dispatches to `ABS_INT32` which uses `abs_tile_int32()` with hardware integer ABS instructions.

4. **No scalar parameters**: ABS does not use any scalar parameters. The `packed_scalar1` and `packed_scalar2` runtime args are passed as 0 and ignored.

5. **Op chaining**: The unary program factory supports chaining multiple SFPU operations (`SFPU_OP_CHAIN_0` can contain multiple init/func pairs). For standalone ABS, the chain contains a single operation.

6. **Cached program**: The factory returns a `cached_program_t` that stores kernel handles and core count, enabling `override_runtime_arguments()` to update only buffer addresses on subsequent calls without recreating the program.

7. **BITCAST special case**: The input CB data format is overridden to match the output format when the operation is BITCAST. This does not apply to ABS.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary eltwise operation program factory work? What kernels does it use?"
   **Reason**: Needed to understand the overall architecture of the unary program factory before reading source code.
   **Key Findings**: Confirmed three-kernel architecture (reader/compute/writer), double-buffered CBs, and the `split_work_to_cores` distribution strategy.

2. **Query**: "How does split_work_to_cores work?"
   **Reason**: Needed to understand the core distribution and remainder handling mechanism.
   **Key Findings**: Returns a tuple of (num_cores, all_cores, core_group_1, core_group_2, units_per_group_1, units_per_group_2). Group 1 gets `floor(N/C)+1` units for `N%C` cores; group 2 gets `floor(N/C)` units for the remaining cores.

### Documentation References

1. **Source**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h`
   **Reason**: Needed to confirm `abs_tile()` and `abs_tile_init()` function signatures and behavior.
   **Key Information**: `abs_tile(idst)` calls `llk_math_eltwise_unary_sfpu_abs<APPROX>(idst)` on the MATH RISC-V.

2. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h`
   **Reason**: Needed to understand the actual SFPU microcode for ABS.
   **Key Information**: Float ABS uses `sfpi::abs(v)` intrinsic in an 8-iteration loop over dst_reg. INT32 ABS uses `SFPLOAD`/`SFPABS`/`SFPSTORE` instruction sequence.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to determine compute kernel path, define generation, and approx mode for ABS.
   **Key Information**: ABS maps to default kernel path `eltwise_sfpu.cpp`, generates defines `abs_tile_init()` and `abs_tile(0)`, and uses exact mode (not approximate).

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_abs.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `abs_tile_init()` which invokes `llk_math_eltwise_unary_sfpu_abs_init<APPROX>()`, which in turn calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::abs>()` to initialize the SFPU config register, configure ADDR_MOD_7, and reset counters.
2. The compute kernel calls `abs_tile(0)` which invokes `llk_math_eltwise_unary_sfpu_abs<APPROX>(dst_index)`.
3. This calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(calculate_abs<APPROX>, dst_index, VectorMode::RC)` in the params dispatch layer.
4. The params dispatch function sets the DST write address for the target tile, stalls until the SFPU is ready, then enters a loop over 4 faces (for `VectorMode::RC`). On each face iteration, it calls `calculate_abs<APPROX>()` and then advances the DST address by 16 rows (two increments of 8).
5. Inside `calculate_abs`, an 8-iteration loop reads from `dst_reg[0]`, applies `sfpi::abs(v)`, writes the result back to `dst_reg[0]`, and increments `dst_reg++`. Each iteration processes one row of 32 elements across all SFPU lanes.
6. After all 4 faces are processed, the params dispatch clears the DST register address and (on Wormhole) stalls until the SFPU is idle before clearing the address mode base.

For INT32, step 2 uses `abs_tile_int32(0)` which dispatches `calculate_abs_int32<APPROX>` through the same params path.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h
// NOTE: Wormhole B0 and Blackhole implementations are identical for calculate_abs().
//       They differ only in calculate_abs_int32() for the SFPLOAD/SFPSTORE addr_mode argument.

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs() { // APPROXIMATION_MODE=false (unused, ABS is exact)
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0]; // implicit SFPLOAD from DST register at current address
        dst_reg[0] = sfpi::abs(v); // SFPABS with InstrMod[0]=1 (FP32 mode), then implicit SFPSTORE
        dst_reg++; // advance DST register pointer by 1 row
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() { // APPROXIMATION_MODE=false (unused)
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        // --- Wormhole B0 variant ---
        // TT_SFPLOAD(lreg_dest=1, instr_mod=4(SMAG32), addr_mode=3, addr=0)
        //   Loads 32-bit value from DST into LREG[1] as SMAG32 format
        // TTI_SFPABS(imm12=0, lreg_c=1, lreg_dest=0, instr_mod1=0)
        //   instr_mod1=0 selects INT32 mode; reads LREG[1], writes abs to LREG[0]
        // TTI_SFPSTORE(lreg_ind=0, instr_mod=4(SMAG32), addr_mode=3, addr=0)
        //   Stores LREG[0] back to DST as SMAG32 format

        // --- Blackhole variant ---
        // TT_SFPLOAD(1, 12, ADDR_MOD_7, 0)  -- instr_mod=12 (reserved/arch-specific)
        // TTI_SFPABS(0, 1, 0, 0)             -- same as Wormhole
        // TTI_SFPSTORE(0, 12, ADDR_MOD_7, 0) -- instr_mod=12 (reserved/arch-specific)

        TT_SFPLOAD(1, 4, 3, 0);   // Wormhole: load DST[current] into LREG[1], SMAG32 format
        TTI_SFPABS(0, 1, 0, 0);   // abs(LREG[1]) -> LREG[0], INT32 mode (InstrMod[0]=0)
        TTI_SFPSTORE(0, 4, 3, 0); // store LREG[0] to DST[current], SMAG32 format
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### SFPU Instructions Used

| Instruction | Opcode | Description |
|-------------|--------|-------------|
| **SFPABS** | 0x7D | Computes the absolute value of `RG[VC]` and stores to `RG[VD]`. When `InstrMod[0]=1` (FP32 mode), clears the sign bit of the floating-point value (NaN passes through unchanged, Inf gets sign cleared). When `InstrMod[0]=0` (INT32 mode), negates negative integers (overflow saturates `INT32::MAX_NEG` to `INT32::MAX_POS`). IPC=1, latency=1. Sets exception flags (NaN, Inf, Overflow) but does not set condition codes. |
| **SFPLOAD** | 0x70 | Loads a value from the Destination register file into an SFPU local register (LREG). The `InstrMod` field selects the source data format and conversion. In `calculate_abs()`, this is implicit via `dst_reg[0]` read. In `calculate_abs_int32()`, explicit `TT_SFPLOAD(1, 4, 3, 0)` loads into LREG[1] with SMAG32 format (InstrMod=4). IPC=1, latency=1. |
| **SFPSTORE** | 0x72 | Stores a value from an SFPU local register back to the Destination register file. In `calculate_abs()`, this is implicit via `dst_reg[0] = ...` write. In `calculate_abs_int32()`, explicit `TTI_SFPSTORE(0, 4, 3, 0)` stores LREG[0] with SMAG32 format. IPC=1, latency=2 (SrcS) or 3 (Dest). |

For the float path (`calculate_abs`), the SFPI compiler emits `SFPLOAD` (implied format), `SFPABS` (with `SFPABS_MOD1_FLOAT=1`), and `SFPSTORE` (implied format) from the `dst_reg[0]` read, `sfpi::abs(v)` call, and `dst_reg[0]` write respectively. These are not visible as explicit instruction macros because the SFPI C++ abstraction layer generates them through `__builtin_rvtt_sfpabs`.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DST (Destination Register File)** | Source and sink for tile data. The SFPU reads from and writes back to the same DST tile slot. Each `dst_reg[0]` access addresses the current row (32 elements wide) in the DST register. The `dst_reg++` operation advances the internal pointer by one row. |
| **LREG[0]** (float path) | Implicitly used by `sfpi::abs()`. The `vFloat v = dst_reg[0]` loads into an LREG, the `SFPABS` instruction operates on it, and the result is stored back from an LREG. The SFPI compiler manages LREG allocation. |
| **LREG[1]** (INT32 path) | Explicitly used as the source register in `TT_SFPLOAD(1, ...)`. Holds the loaded integer value from DST before the `SFPABS` instruction operates on it. |
| **LREG[0]** (INT32 path) | Explicitly used as the destination register in `TTI_SFPABS(0, 1, 0, 0)` and source in `TTI_SFPSTORE(0, ...)`. Holds the absolute value result. |

The SFPU has 8 local registers (LREG[0]..LREG[7]). For ABS, only LREG[0] and LREG[1] are used. The float path uses compiler-managed LREG allocation via the SFPI `vFloat` type, while the INT32 path uses explicit LREG indices in the instruction macros.

### Address Mode Configuration

The init function `_llk_math_eltwise_unary_sfpu_init_<SfpuType::abs>()` calls `eltwise_unary_sfpu_configure_addrmod<SfpuType::abs>()`, which configures:

**ADDR_MOD_7** (used by all standard unary SFPU operations):

| Field | Value | Description |
|-------|-------|-------------|
| `srca.incr` | 0 | No auto-increment on SRC A register address |
| `srcb.incr` | 0 | No auto-increment on SRC B register address |
| `dest.incr` | 0 | No auto-increment on DST register address |

This configuration is identical for both Wormhole B0 and Blackhole. ADDR_MOD_7 is chosen specifically to avoid conflicts with ADDR_MOD_0 and ADDR_MOD_2, which are used by the A2D (Accumulate-to-Destination) pipeline that runs concurrently.

The `dest.incr = 0` setting means the SFPU does not auto-advance through DST rows via the address mode hardware. Instead, row advancement is handled explicitly by `dst_reg++` within the SFPU kernel loop (which compiles to `SFPINCRWC` or equivalent counter manipulation) and by `TTI_SETRWC` calls in the params dispatch layer between faces.

ABS does not use ADDR_MOD_6 (that is only configured for specific operations like `typecast`, `topk_local_sort`, `unary_max`, `unary_min`, and on Blackhole additionally `reciprocal`).

**Wormhole B0 vs Blackhole differences in params dispatch**: The Wormhole B0 params dispatch (`llk_math_eltwise_unary_sfpu_params.h`) calls `math::set_addr_mod_base()` before stalling and `math::clear_addr_mod_base()` after the SFPU completes. It also uses `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU)` after clearing the DST address. The Blackhole params dispatch omits these calls, using only `_llk_math_eltwise_unary_sfpu_start_` / `_llk_math_eltwise_unary_sfpu_done_` helpers and `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` for face advancement (which calls `math::inc_dst_addr<8>()` twice per face). The Wormhole B0 variant instead uses inline `TTI_SETRWC` instructions with `p_setrwc::CR_D` and increment 8 for the same purpose.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the ABS (absolute value) SFPU kernel work? What is the call chain from the compute kernel API through LLK down to the ckernel SFPU implementation?"
   **Reason**: Needed to identify the full call chain and file paths for all abstraction layers.
   **Key Findings**: Confirmed the chain: `abs_tile()` -> `llk_math_eltwise_unary_sfpu_abs<APPROX>()` -> `_llk_math_eltwise_unary_sfpu_params_()` -> `calculate_abs()`. Identified both Wormhole B0 and Blackhole file paths.

2. **Query**: "How is the abs SFPU operation implemented in the LLK layer? What SFPU instructions does it use?" (tenstorrent/tt-llk)
   **Reason**: Needed LLK-specific implementation details and instruction-level behavior.
   **Key Findings**: Confirmed `SFPABS` instruction usage, and the dispatch through `_llk_math_eltwise_unary_sfpu_params_` with face iteration.

3. **Query**: "How does SFPI handle absolute value? What instructions or register manipulations are used?" (tenstorrent/sfpi)
   **Reason**: Needed to understand the SFPI intrinsic layer and how `sfpi::abs()` maps to hardware instructions.
   **Key Findings**: `sfpi::abs(vFloat)` compiles to `__builtin_rvtt_sfpabs(v.get(), SFPABS_MOD1_FLOAT)` which is the `SFPABS` instruction with InstrMod[0]=1. For `vInt`, it uses `SFPABS_MOD1_INT=0`.

### Confluence References

1. **Page**: Tensix SFPU Instruction Set Architecture (Page ID: 1170505767)
   **Sections consulted**: SFPABS, SFPLOAD, SFPSTORE
   **Key findings**: SFPABS opcode 0x7D, O2 encoding, IPC=1, latency=1. InstrMod[0] selects INT32 (0) vs FP32 (1) mode. For FP32 mode, clears sign bit (NaN passes through, Inf gets sign cleared). For INT32 mode, negates negative values with overflow saturation (MAX_NEG -> MAX_POS, sets Overflow flag). SFPLOAD opcode 0x70 with InstrMod=4 (SMAG32) loads 32-bit signed-magnitude from DST. SFPSTORE opcode 0x72 with InstrMod=4 stores back as SMAG32.

### Glean References

No Glean queries were needed for this analysis. The SFPABS instruction is simple and fully documented in the Confluence SFPU ISA page.
