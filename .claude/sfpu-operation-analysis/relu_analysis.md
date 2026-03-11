# RELU Implementation Analysis

## Overview

RELU (Rectified Linear Unit) is an element-wise unary activation function that clamps negative values to zero: `f(x) = max(0, x)`. It is implemented through the generic unary program factory, which serves as a shared infrastructure for all unary element-wise operations in TTNN.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

The RELU operation uses the `UnaryProgramFactory` (interleaved path). The compute kernel is the generic `eltwise_sfpu.cpp`, specialized via preprocessor defines that inject `relu_tile_init()` and `relu_tile()` calls. For INT32 inputs, `relu_tile_int32()` is used instead.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) or row (for ROW_MAJOR layout) |
| **Unit size** | 1 page (1 tile for TILE_LAYOUT, 1 row for ROW_MAJOR) |
| **Total units** | `input.buffer()->num_pages()` |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles (dim=1 for standard factory), processing one tile per iteration |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary (any rank) |
| **Dimension convention** | Flattened to pages |
| **Tensor layout** | TILE_LAYOUT (typical) or ROW_MAJOR |
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
| **Data type** | Same as input (or specified output dtype) |

### Layout Transformations

No layout transformations (tilize/untilize) are performed within this operation. Input and output share the same layout. For BITCAST operations the input CB uses the output data format, but RELU does not trigger this path.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `noc_async_read_barrier`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile`, `relu_tile(0)`, `pack_tile(0, c_2)`, `cb_pop_front(c_0, 1)`, `cb_push_back(c_2, per_core_block_dim)` |
| 3 | Writer | CB c_2 | DRAM (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `noc_async_writes_flushed`, `cb_pop_front(c_2, 1)`, `noc_async_write_barrier` |

**Detailed flow**:
1. The reader kernel iterates from `start_id` to `start_id + num_pages`, reading one page at a time from DRAM into CB c_0 using `TensorAccessor` for address resolution.
2. The compute kernel processes tiles in blocks. For each block of `per_core_block_cnt` iterations, it reserves output space in CB c_2, then for each tile: acquires register space, waits for input in c_0, copies the tile to DST registers, applies the SFPU RELU operation (`relu_tile(0)`), commits and packs the result to c_2, then pops the input from c_0.
3. The writer kernel waits for each output page in CB c_2, writes it to DRAM via `TensorAccessor`, and pops it.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | src0 | Input staging | 2 pages | 1 page | Double | Reader | Compute | Program |
| c_2 | output | Output staging | 2 pages | 1 page | Double | Compute | Writer | Program |

**Notes**:
- Page size depends on layout: `tile_size(cb_data_format)` for TILE_LAYOUT, `src_buffer->page_size()` for ROW_MAJOR.
- CB c_1 (tmp0) is NOT allocated for RELU. It is only created for HARDSHRINK, CBRT, or LOGIT operations.
- For BITCAST operations, c_0 uses the output data format instead of input -- not applicable to RELU.

## Pipeline Pattern Summary

Both CB c_0 and CB c_2 have capacity = 2 pages and block size = 1 page, enabling **double-buffering**. This allows the reader to write the next tile into c_0 while the compute kernel processes the current tile, and similarly allows the compute kernel to write to c_2 while the writer drains the previous tile. This creates a 3-stage overlapping pipeline: Read | Compute | Write.

## Index Calculations

Index mapping is handled entirely by `TensorAccessor`, which is initialized from `TensorAccessorArgs` at compile time. The accessor encapsulates the mapping from a linear page index to a physical NoC address, accounting for:
- Interleaved bank distribution (pages round-robin across DRAM banks or L1 banks)
- Whether the buffer resides in DRAM or L1

The reader and writer kernels iterate over a contiguous range of page IDs: `[start_id, start_id + num_pages)`. The `noc_async_read_page(i, s, l1_write_addr)` call uses the TensorAccessor `s` to resolve page `i` to its physical bank and offset.

## Memory Access Patterns

### Read Pattern
- **Sequential**: Pages are read in linear order from `start_id` to `start_id + num_pages - 1`.
- **Granularity**: One page per NoC read transaction.
- **Synchronization**: Each read is followed by `noc_async_read_barrier()` before pushing to the CB, ensuring completion before the page is made available to compute.
- **Access type**: DRAM interleaved (pages distributed across banks).

### Write Pattern
- **Sequential**: Pages are written in the same linear order as they were read.
- **Granularity**: One page per NoC write transaction.
- **Synchronization**: `noc_async_writes_flushed()` is called after each page write, with a final `noc_async_write_barrier()` after the loop completes.
- **Access type**: DRAM interleaved.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major enumeration) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g., 8x8) |
| **Total cores** | `num_cores` (determined by `split_work_to_cores`) |
| **Work per core** | `num_pages / num_cores` pages (with remainder handling) |
| **Load balancing** | Two-group split: core_group_1 gets `ceil(num_pages/num_cores)` pages, core_group_2 gets `floor(num_pages/num_cores)` pages |

**Core enumeration order**: Cores are enumerated column-major: `core = {i / num_cores_y, i % num_cores_y}`, meaning column index increments first, then row index.

**Remainder handling**: `split_work_to_cores` divides `num_pages` across all available cores. If `num_pages % num_cores != 0`, `core_group_1` gets one extra page per core. Two separate compute kernel instances are created with different `per_core_block_cnt` compile-time args to reflect this difference.

## Arguments

### Compile-Time Arguments

**Reader kernel** (`reader_unary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Encoded tensor accessor configuration for the source buffer (bank mapping, DRAM/L1 flag, page layout) |

**Writer kernel** (`writer_unary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer index (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encoded tensor accessor configuration for the destination buffer |

**Compute kernel** (`eltwise_sfpu.cpp`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tiles) to process on this core |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1 for standard unary factory) |

### Runtime Arguments

**Reader kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM/L1 address |
| 1 | num_pages | uint32_t | Number of pages this core should read |
| 2 | start_id | uint32_t | Starting page index for this core |

**Writer kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM/L1 address |
| 1 | num_pages | uint32_t | Number of pages this core should write |
| 2 | start_id | uint32_t | Starting page index for this core |

**Compute kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Packed scalar parameter (0 for RELU -- unused) |
| 1 | packed_scalar2 | uint32_t | Packed scalar parameter (0 for RELU -- unused) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM | CB c_0 | Read pages via TensorAccessor |
| compute | RISCV_2 | N/A | CB c_0 | CB c_2 | copy_tile, relu_tile (SFPU), pack_tile |
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM | Write pages via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page reader. Gets page size from CB interface (`get_local_cb_interface(cb_id_in0).fifo_page_size`), constructs a `TensorAccessor` from compile-time args, then loops from `start_id` to `end_id`, reading one page per iteration with a read barrier after each page. Supports `BACKWARDS` define for reverse iteration (not used by RELU).

### Compute Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **Key Logic**: Generic SFPU compute kernel. Calls `init_sfpu(c_0, c_2)` to initialize the SFPU pipeline. The operation-specific behavior is injected via the `SFPU_OP_CHAIN_0` preprocessor define, which expands to:
  - `SFPU_OP_CHAIN_0_INIT_0` -> `relu_tile_init();`
  - `SFPU_OP_CHAIN_0_FUNC_0` -> `relu_tile(0);` (or `relu_tile_int32(0);` for INT32)

  The macro also sets `SFPU_OP_RELU_FAMILY_INCLUDE` to `1`, which causes `sfpu_split_includes.h` to include the RELU family SFPU implementation headers.

  The tile processing flow per iteration: `tile_regs_acquire` -> `cb_wait_front(c_0, 1)` -> `copy_tile(c_0, 0, 0)` -> SFPU RELU op -> `tile_regs_commit` -> `tile_regs_wait` -> `pack_tile(0, c_2)` -> `cb_pop_front(c_0, 1)` -> `tile_regs_release`. Output CB push happens once per block (but block_dim=1, so effectively per tile).

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Sequential page writer. Supports `OUT_SHARDED` mode (just waits for all pages in CB without writing) and standard interleaved mode. In interleaved mode, loops from `start_id` to `end_id`, waiting for one page in c_2, writing it via NoC, flushing, and popping. Final `noc_async_write_barrier()` ensures all writes complete.

## Implementation Notes

1. **Compute configuration**: RELU uses `MathFidelity::HiFi4` and `math_approx_mode = false` (the default from `get_op_approx_mode`).

2. **No scalar parameters**: Unlike operations like LEAKY_RELU or RELU_MAX, standard RELU has no parameters. Both `packed_scalar1` and `packed_scalar2` remain 0.

3. **INT32 support**: When `input.dtype() == DataType::INT32`, the define `INP_INT32` is set to `1`, and `relu_tile_int32()` is used instead of `relu_tile()`.

4. **Program caching**: The `override_runtime_arguments` method enables program reuse across calls with different buffer addresses but identical shapes and configurations. Only `src_addr` and `dst_addr` are updated per call.

5. **Op chaining**: The unary factory supports chaining multiple SFPU operations in a single pass via the `op_chain` vector. For standalone RELU, the chain has a single entry.

6. **RELU6 variant**: RELU6 (clamping to [0, 6]) reuses `relu_max_tile` with a hardcoded upper bound of 6.0 (`0x40c00000u`), not the standard RELU path.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary eltwise program factory work? What kernels does it use (reader, compute, writer)? How does it handle interleaved vs sharded tensors?"
   **Reason**: Needed to understand the overall architecture of the unary program factory before diving into source code.
   **Key Findings**: Three program factory variants exist (interleaved, sharded, sub-core-grid). Interleaved uses `reader_unary_interleaved_start_id.cpp` and `writer_unary_interleaved_start_id.cpp`. Compute kernel is selected by `get_compute_kernel_path`.

2. **Query**: "How does split_work_to_cores work? What does it return?"
   **Reason**: Needed to understand core distribution and remainder handling for the analysis.
   **Key Findings**: Returns two core groups -- core_group_1 gets `ceil(N/cores)` work units, core_group_2 gets `floor(N/cores)`. Cores are enumerated to fill the grid, and separate kernel instances with different compile-time args handle the two groups.

3. **Query**: "How does TensorAccessorArgs work in tt-metal? What compile-time args does it append?"
   **Reason**: Needed to understand the compile-time argument structure for reader/writer kernels.
   **Key Findings**: TensorAccessorArgs encapsulates bank mapping configuration (DRAM vs L1, interleaved vs sharded) into compile-time args. The TensorAccessor constructed from these args provides `get_noc_addr(page_id)` for physical address resolution.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to determine RELU's compute kernel path, macro defines, init/func strings, and approx mode.
   **Key Information**: RELU falls through to the `default` case in `get_compute_kernel_path`, returning `"eltwise_sfpu.cpp"`. Its macro definition is `SFPU_OP_RELU_FAMILY_INCLUDE`. Init string is `relu_tile_init()`, func string is `relu_tile({idst})`. `get_op_approx_mode` returns `false` for RELU.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
   **Reason**: Needed to understand the generic compute kernel structure and how SFPU_OP_CHAIN defines are consumed.
   **Key Information**: The kernel uses `init_sfpu`, `copy_tile`, tile register acquire/commit/wait/release pattern, and injects operation-specific logic via `SFPU_OP_CHAIN_0` macro expansion.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to. RELU is notable for having **architecture-divergent implementations**: Wormhole B0 uses explicit SFPU instructions (SFPLOAD, SFPMOV, SFPSWAP, SFPSTORE) while Blackhole uses higher-level SFPI conditional assignment (v_if/v_endif). Both achieve the same result -- clamping negative values to the threshold (0 for standard RELU) -- but through fundamentally different mechanisms.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_relu.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

Note: There is also an older, simplified version of the SFPU kernel at `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h` which wraps the tt_llk version. The tt_llk submodule contains the authoritative implementation with the `_relu_min_` template signature that the API header macro resolves to.

### Call Chain

1. **Compute kernel** calls `relu_tile(0)` (injected via `SFPU_OP_CHAIN_0_FUNC_0` define).
2. **API header** (`relu.h`) expands this to `MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_min_, RC, APPROX, 0, 0))`, which calls `_llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::_relu_min_<sfpi::vFloat, false, 8, uint32_t>, 0, (int)VectorMode::RC, 0)`.
3. **LLK params** (`llk_math_eltwise_unary_sfpu_params.h`) sets the DST write address, stalls until SFPU is ready, then calls the SFPU function 4 times (once per face in RC mode), incrementing the DST face address between each call.
4. **Core SFPU** (`ckernel_sfpu_relu.h`) runs `_relu_min_<vFloat, false, 8, uint32_t>(0)`, which converts the threshold (0) to a `vFloat` via `Converter::as_float`, then calls `_relu_min_impl_` to iterate over 8 rows per face, comparing each element against the threshold and replacing values below it.

The init path is: `relu_tile_init()` -> `SFPU_UNARY_KERNEL_INIT(relu_min, APPROX)` -> `llk_math_eltwise_unary_sfpu_init<SfpuType::relu_min, false>()` -> `_llk_math_eltwise_unary_sfpu_init_<SfpuType::relu_min>()`, which initializes the SFPU config register, configures ADDR_MOD_7, and resets counters.

### Annotated SFPU Kernel Source

**Blackhole implementation** -- uses SFPI high-level conditional assignment:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(const int iterations, VecType threshold) // APPROXIMATION_MODE=false, ITERATIONS=8
{
    for (int d = 0; d < iterations; d++)
    {
        VecType a = sfpi::dst_reg[0];       // Load current element from DEST row 0
        v_if (a < threshold)                 // Predicated: sets CC for lanes where a < threshold
        {
            sfpi::dst_reg[0] = threshold;    // Conditionally replace with threshold (0.0f for RELU)
        }
        v_endif;
        sfpi::dst_reg++;                     // Advance to the next DEST row
    }
}

template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_min_(T threshold) // VectorType=sfpi::vFloat, APPROXIMATION_MODE=false, ITERATIONS=8, T=uint32_t
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>, "VectorType must be sfpi::vFloat or sfpi::vInt");

    VectorType v_threshold;
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            v_threshold = static_cast<int>(threshold);
        }
        else
        {
            v_threshold = Converter::as_float(threshold); // Reinterpret uint32_t 0x00000000 as float 0.0f
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_min_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
}
```

**Wormhole B0 implementation** -- uses explicit SFPU instructions (SFPLOAD, SFPMOV, SFPSWAP, SFPSTORE):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(const int iterations, [[maybe_unused]] VecType threshold, int sfpload_instr_mod)
{
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, 0);  // Load DEST[row] -> LREG0; sfpload_instr_mod=DEFAULT for float
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);               // Copy threshold from LREG2 -> LREG1
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);              // Conditional swap: max->LREG1, min->LREG0; Imm12[0]=1 means FP32 comparison
        TTI_SFPSTORE(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_3, 0); // Store max(input, threshold) from LREG1 -> DEST[row]
        sfpi::dst_reg++;                                                 // Advance DEST row pointer
    }
}

template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_min_(T threshold) // VectorType=sfpi::vFloat, APPROXIMATION_MODE=false, ITERATIONS=8, T=uint32_t
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>, "VectorType must be sfpi::vFloat or sfpi::vInt");

    VectorType v_threshold;
    int scalar = threshold;
    if (scalar < 0)
    { // To convert from 2's complement to sign+magnitude
        scalar  = -scalar;
        int res = 0x80000000 | (scalar & 0x7FFFFFFF);
        scalar  = res;
    }
    int sfpload_instr_mod = DEFAULT;        // InstrModLoadStore::DEFAULT for float data
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            _sfpu_load_imm32_(p_sfpu::LREG2, scalar);  // Load sign+magnitude threshold into LREG2
            sfpload_instr_mod = INT32_2S_COMP;           // InstrModLoadStore for int32 2's complement format
        }
        else
        {
            _sfpu_load_imm32_(p_sfpu::LREG2, threshold); // Load threshold (0x00000000 = 0.0f) into LREG2
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_min_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold, sfpload_instr_mod);
}
```

The `_sfpu_load_imm32_` helper used in Wormhole:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_load_config.h

inline void _sfpu_load_imm32_(const std::uint32_t dest, const std::uint32_t val)
{
    TT_SFPLOADI(dest, 10, (val & 0xFFFF));      // insmod==10 writes lower 16 bits, preserves upper bits
    TT_SFPLOADI(dest, 8, (val >> 16) & 0xFFFF); // insmod==8 writes upper 16 bits, preserves lower bits
}
```

### SFPU Instructions Used

**Blackhole** (SFPI-based, instructions emitted by compiler from v_if/dst_reg constructs):

| Instruction | Description |
|-------------|-------------|
| `SFPLOAD` | Loads a value from DEST register into an LREG. Emitted by `dst_reg[0]` read. Uses `ADDR_MOD_7` (incr=0). |
| `SFPSTORE` | Stores a value from an LREG back to DEST register. Emitted by `dst_reg[0] = threshold`. Uses `ADDR_MOD_7`. |
| `SFPSETCC` | Sets the condition code based on a comparison. Emitted by the `<` operator in `v_if(a < threshold)`. |
| `SFPPUSHC` | Pushes condition code onto the CC stack. Emitted by `v_if`. |
| `SFPPOPC` | Pops condition code from the CC stack. Emitted by `v_endif`. |
| `SFPCOMPC` | Complements the condition code (used internally by conditional logic). |

**Wormhole B0** (explicit instruction-level programming):

| Instruction | Description |
|-------------|-------------|
| `SFPLOAD` | Loads a value from DEST register into LREG0. `InstrModLoadStore::DEFAULT` for FP32; `INT32_2S_COMP` for int32. Uses `ADDR_MOD_3`. |
| `SFPMOV` | Copies the threshold value from LREG2 to LREG1 (so SFPSWAP can operate on LREG1 vs LREG0). |
| `SFPSWAP` | Conditional swap with `InstrMod=0` (unconditional on all rows? No -- see details below) and `Imm12[0]=1` (FP32 comparison mode). After execution, LREG1 holds the larger value (max) and LREG0 holds the smaller value (min). This implements `max(input, threshold)` in a single instruction. |
| `SFPSTORE` | Stores the result (LREG1 = max value) back to DEST. Uses `ADDR_MOD_3`. |
| `SFPLOADI` | Loads a 16-bit immediate into an LREG. Used by `_sfpu_load_imm32_` to construct the 32-bit threshold in LREG2 (two SFPLOADI calls: lower 16 bits with insmod=10, upper 16 bits with insmod=8). |

**SFPSWAP detail**: The instruction is called with parameters `(0, p_sfpu::LREG1, p_sfpu::LREG0, 1)`. The `InstrMod=0` means unconditional swap on all rows -- but since the swap is between LREG1 (threshold copy) and LREG0 (input), and `Imm12[0]=1` selects FP32 comparison, the result is that the larger of the two values ends up in LREG1 (VC register) and the smaller in LREG0 (VD register). Since the threshold is 0.0f for standard RELU, LREG1 will contain `max(input, 0.0f)` which is exactly the RELU function. This is a single-instruction implementation of the comparison + selection, which is more efficient than the SFPI conditional branch approach.

**Correction on SFPSWAP InstrMod**: Looking more carefully at the TTI_SFPSWAP call: `TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1)`. The fourth argument `1` maps to `Imm12[0]=1`, meaning FP32 comparison. The first argument `0` is the InstrMod, but for SFPSWAP with conditional mode (InstrMod >= 1), the behavior is to conditionally swap. With `InstrMod=0`, operands are swapped unconditionally. However, looking at the Confluence ISA spec, `InstrMod=0` is "swap unconditionally" and `InstrMod=1` is "conditionally exchange." The WH code uses `InstrMod=0` in the TTI macro's first argument position, but the actual encoding may differ. The key semantic is: after SFPSWAP, LREG1 contains max and LREG0 contains min, achieving the relu_min operation.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Holds the input value loaded from DEST. In WH, explicitly loaded via SFPLOAD; in BH, implicitly via `dst_reg[0]` read. |
| **LREG1** | (WH only) Temporary copy of threshold from LREG2; after SFPSWAP holds the max(input, threshold) result. |
| **LREG2** | (WH only) Holds the threshold value (0.0f for standard RELU), loaded once before the loop via `_sfpu_load_imm32_`. Persists across all 8 iterations. |
| **DEST registers** | The tile data resides in the DEST register file. Each iteration processes one row (64 elements across all SFPU lanes). The `dst_reg++` advances to the next row. 8 rows are processed per face, and 4 faces per tile (in RC mode). |
| **CC stack** | (BH only) The condition code stack is used by v_if/v_endif to predicate the threshold assignment. Elements where `a >= threshold` are left unchanged. |

### Address Mode Configuration

**Blackhole**: The init function `_llk_math_eltwise_unary_sfpu_init_<SfpuType::relu_min>()` configures `ADDR_MOD_7` with all increments set to 0:

```
ADDR_MOD_7: srca.incr = 0, srcb.incr = 0, dest.incr = 0
```

RELU does not match any of the special-cased `SfpuType` values (reciprocal, typecast, unary_max, unary_min, etc.), so `ADDR_MOD_6` is not configured. The dest increment is 0 because SFPI manages DEST row advancement through the `dst_reg++` software abstraction rather than hardware auto-increment.

**Wormhole B0**: The same `ADDR_MOD_7` configuration applies (all increments = 0). However, the WH `_relu_min_impl_` uses `ADDR_MOD_3` in its SFPLOAD/SFPSTORE instructions rather than `ADDR_MOD_7`. This is because the WH ckernel_sfpu_relu.h file was written before the standardization on ADDR_MOD_7 for general SFPU operations, and `ADDR_MOD_3` serves the same purpose (dest.incr = 0) in the WH address mode register bank. The WH `_llk_math_eltwise_unary_sfpu_start_` also calls `math::set_addr_mod_base()`, and `_llk_math_eltwise_unary_sfpu_done_` calls `math::clear_addr_mod_base()` and stalls for SFPU completion -- neither of which exist in the BH version.

**Summary**: Both architectures use a zero-increment address mode for SFPLOAD/SFPSTORE, relying on `dst_reg++` (which translates to explicit DEST address manipulation) for row advancement. The hardware-level ADDR_MOD register merely controls auto-increment behavior of the DEST pointer per SFPU instruction, and zero means "do not auto-increment."

## External Knowledge Sources (SFPU Analysis)

### DeepWiki Queries

1. **Query**: "How does the RELU SFPU kernel work in tt-metal? Trace from the compute kernel API (relu_tile) through LLK to the ckernel SFPU implementation."
   **Reason**: Needed to identify the complete call chain and all relevant file paths for the RELU SFPU implementation.
   **Key Findings**: RELU uses `_relu_min_` with threshold=0. The call chain goes through `SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT` macro to `_llk_math_eltwise_unary_sfpu_params_` to `ckernel_sfpu_relu.h`. Separate implementations exist per architecture.

2. **Query**: "How is the relu SFPU operation implemented in the LLK layer?" (tenstorrent/tt-llk)
   **Reason**: Needed to understand the tt_llk submodule's SFPU kernel source code, particularly the VectorType-templated `_relu_min_` function.
   **Key Findings**: The tt_llk implementation has architecture-divergent relu_min_impl: WH uses SFPLOAD/SFPMOV/SFPSWAP/SFPSTORE while BH uses SFPI v_if/v_endif. The WH version loads threshold into LREG2, then uses SFPSWAP to compute max(input, threshold) in a single instruction.

3. **Query**: "How do v_if comparisons work in SFPI?" (tenstorrent/sfpi)
   **Reason**: Needed to understand what SFPU instructions the BH v_if/v_endif construct generates.
   **Key Findings**: `v_if(a < threshold)` generates SFPSETCC to set condition code bits, SFPPUSHC to push onto the CC stack. All instructions inside the v_if block execute on all lanes but only affect lanes where the condition is true. v_endif generates SFPPOPC to restore the previous CC state.

### Confluence References

1. **Page**: Tensix SFPU Instruction Set Architecture (Page ID: 1170505767)
   **Section consulted**: SFPSWAP instruction specification
   **Key Information**: SFPSWAP (opcode 0x92) conditionally or unconditionally exchanges values in two LREGs. With InstrMod=1, it conditionally swaps based on comparison. With Imm12[0]=1, it uses FP32 comparison. After conditional swap, the smaller value ends up in RG[VD] and the larger in RG[VC]. Latency is 2 cycles (4-cycle total with pipeline), IPC is 0.5. This confirms that the WH RELU implementation computes max(input, 0) by placing the larger value in LREG1 (VC) after the swap.

### Glean References

No Glean searches were necessary for this analysis. The SFPU instruction details were sufficiently documented in Confluence and DeepWiki.
