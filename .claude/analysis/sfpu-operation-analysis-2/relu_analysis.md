# RELU Implementation Analysis

## Overview

RELU (Rectified Linear Unit) computes `max(0, x)` element-wise on a tensor. It is implemented as a unary SFPU operation that routes through the shared `UnaryProgramFactory` in the interleaved case. RELU has no parameters (it is a non-parameterized unary op), uses the generic `eltwise_sfpu.cpp` compute kernel, and maps to the `relu_tile()` / `relu_tile_init()` SFPU functions on the hardware.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) or row-major page |
| **Unit size** | 1 page (tile or row-major row) |
| **Total units** | `input.buffer()->num_pages()` |
| **Loop structure** | Outer loop over blocks (`per_core_block_cnt`), inner loop over tiles within block (`per_core_block_dim` = 1 for RELU). Effectively a flat loop of 1 tile per iteration. |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|-------------|
| **Logical shape** | Arbitrary (any rank) |
| **Dimension convention** | Flattened to pages |
| **Tensor layout** | TILE_LAYOUT (typical) or ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, or INT32 |

### Output Tensor

| Property | Output Tensor |
|----------|--------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | Same as input |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (or as specified by output dtype) |

### Layout Transformations

No explicit tilize/untilize or reshard occurs within the program factory. The factory supports both TILE_LAYOUT and ROW_MAJOR layouts transparently -- the page size is derived from the CB interface (`get_local_cb_interface(cb_id).fifo_page_size`) and tile size vs buffer page size is selected at CB configuration time (line 58-59 of the program factory).

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `noc_async_read_barrier`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile(c_0, 0, 0)`, `relu_tile_init()`, `relu_tile(0)`, `pack_tile(0, c_2)`, `cb_pop_front(c_0, 1)`, `cb_push_back(c_2, per_core_block_dim)` |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `noc_async_writes_flushed`, `cb_pop_front(c_2, 1)`, `noc_async_write_barrier` |

**Detailed flow**: The reader fetches one page at a time from DRAM via NoC, writing it to CB c_0. The compute kernel waits for a tile in c_0, copies it to the destination register file, executes the SFPU RELU operation (`relu_tile`), packs the result into CB c_2, and pops c_0. The writer waits for a tile in c_2, writes it to the output buffer via NoC, and pops c_2. For INT32 input, `relu_tile_int32()` is used instead.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input staging | 2 pages | 1 page | Double | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 pages | 1 page | Double | Compute | Writer | Program |

**Notes**:
- CB c_1 (tmp0) is NOT allocated for RELU. It is only allocated for HARDSHRINK, CBRT, or LOGIT operations.
- Page size depends on layout: for TILE_LAYOUT it is `tile_size(cb_data_format)`, for ROW_MAJOR it is `src_buffer->page_size()`.
- For RELU, `cb_data_format_for_input` equals `cb_data_format` (the BITCAST special case does not apply).

## Pipeline Pattern Summary

Both CB c_0 and c_2 have capacity = 2 pages and block size = 1 page, enabling **double-buffering**. This allows the reader to fill one slot while compute processes the other, and similarly compute can fill one output slot while the writer drains the other. This provides a classic 3-stage pipelined execution pattern.

## Index Calculations

The program factory uses `TensorAccessor` for index-to-physical-address mapping. The reader and writer kernels use:

```cpp
const auto s = TensorAccessor(src_args, src_addr, page_bytes);
noc_async_read_page(i, s, l1_write_addr);  // i = page index
```

The `TensorAccessor` handles the mapping from a linear page index `i` to the correct DRAM bank and offset within that bank, abstracting away the interleaved memory layout. Compile-time args from `TensorAccessorArgs(*src_buffer)` encode the buffer's memory configuration (number of banks, page size alignment, etc.).

Each core receives a contiguous range of page indices: `[start_id, start_id + num_pages_per_core)`. The `start_id` is accumulated as cores are assigned work.

## Memory Access Patterns

### Read Pattern
- **Sequential**: Pages are read in linear order from `start_id` to `start_id + num_pages - 1`.
- **Granularity**: One page per NoC read transaction.
- **Synchronization**: `noc_async_read_barrier()` after each page (no batching).
- **Source**: DRAM (interleaved across banks) or L1.

### Write Pattern
- **Sequential**: Pages are written in linear order from `start_id` to `start_id + num_pages - 1`.
- **Granularity**: One page per NoC write transaction.
- **Synchronization**: `noc_async_writes_flushed()` after each page, `noc_async_write_barrier()` at the end.
- **Destination**: DRAM (interleaved across banks) or L1.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major traversal) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g., 8x8) |
| **Total cores** | Determined by `split_work_to_cores` based on num_pages |
| **Work per core** | `num_pages / num_cores` (group 1) or remainder tiles (group 2) |
| **Load balancing** | Two-group split: core_group_1 gets `ceil(num_pages/num_cores)` pages each, core_group_2 gets one fewer page |

**Core indexing**: Cores are traversed in column-major order: `core = {i / num_cores_y, i % num_cores_y}`. This means the grid is filled column by column.

**Two core groups**: `split_work_to_cores` divides work into two groups to handle remainders. Group 1 cores get `num_pages_per_core_group_1` pages, group 2 cores get `num_pages_per_core_group_2` pages (one fewer). Each group gets its own compute kernel instance with the appropriate `per_core_block_cnt` compile-time arg.

## Arguments

### Compile-Time Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Encoded buffer layout parameters for source tensor (bank count, page alignment, etc.) |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer index (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encoded buffer layout parameters for destination tensor |

#### Compute Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tiles) this core processes |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1 for standard unary) |

### Runtime Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages this core reads |
| 2 | start_id | uint32_t | First page index for this core |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages this core writes |
| 2 | start_id | uint32_t | First page index for this core |

#### Compute Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for RELU (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for RELU (always 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM/L1 | CB c_0 | Read pages sequentially via TensorAccessor |
| compute | RISCV_2 (unpack+math+pack) | N/A | CB c_0 | CB c_2 | Unpack tile, apply RELU SFPU op, pack tile |
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 | Write pages sequentially via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page reader. Uses `TensorAccessor` for address resolution. Supports `BACKWARDS` define for reverse iteration (not used by RELU). Reads one page per iteration with `noc_async_read_page` and a per-page barrier.

### Compute Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **Key Logic**: Generic SFPU compute kernel shared by most unary operations. The operation-specific behavior is injected via preprocessor defines:
  - `SFPU_OP_RELU_FAMILY_INCLUDE=1` -- enables the RELU family include header
  - `SFPU_OP_CHAIN_0` -- expands to `SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0`
  - `SFPU_OP_CHAIN_0_INIT_0` -- expands to `relu_tile_init();`
  - `SFPU_OP_CHAIN_0_FUNC_0` -- expands to `relu_tile(0);` (or `relu_tile_int32(0);` for INT32)
- **Execution pattern**: For each block, reserves output CB space, then for each tile: acquires registers, waits for input tile in c_0, copies to DST registers, executes the SFPU op chain, commits registers, waits for pack, packs to c_2, pops c_0, releases registers. Pushes the block to c_2 after all tiles in the block are packed.
- **Math configuration**: `math_fidelity = HiFi4`, `math_approx_mode = false` (RELU returns false from `get_op_approx_mode`).

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Sequential page writer. Uses `TensorAccessor` for address resolution. Supports `OUT_SHARDED` define for sharded output (not used in interleaved path). Writes one page per iteration with `noc_async_write_page`, flushes after each write, and issues a final barrier.

## Implementation Notes

1. **RELU is non-parameterized**: Unlike `RELU_MAX`, `RELU_MIN`, or `LEAKY_RELU`, plain RELU has no scalar parameters. The `packed_scalar1` and `packed_scalar2` runtime args are always 0.

2. **SFPU implementation**: RELU is implemented as `_relu_min_` with a lower limit of 0, effectively clamping all negative values to zero. This is a hardware-level SFPU operation on the vector engine.

3. **Shared compute kernel**: RELU uses the same `eltwise_sfpu.cpp` kernel as most other unary SFPU operations. Operation-specific behavior is entirely controlled through preprocessor defines generated by `get_block_defines()`.

4. **INT32 support**: When the input dtype is INT32, the define `INP_INT32=1` is set and the SFPU function switches to `relu_tile_int32()`.

5. **Program caching**: The factory implements `override_runtime_arguments` which only updates buffer addresses (src and dst). This enables program caching -- when the same operation is called again with different tensors of the same shape/config, only the buffer addresses are patched without recreating the program.

6. **No BITCAST interaction**: The special BITCAST logic (using output format for input CB) does not apply to RELU.

7. **Block size of 1**: The `per_core_block_dim` is always 1 for the standard `UnaryProgramFactory`, meaning tiles are processed one at a time. The `per_core_block_cnt` equals the number of pages assigned to that core.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary eltwise program factory work? What kernels does it use for reader, compute, and writer? How does it handle interleaved vs sharded tensors?"
   **Reason**: Needed to understand the overall architecture of the unary program factory before diving into source code.
   **Key Findings**: Confirmed three program factories (Interleaved, SubCoreGrid, Sharded), identified kernel file paths, and learned the selection logic.

2. **Query**: "How does the RELU unary operation work in TTNN? What compute kernel does it use? How does get_compute_kernel_path resolve the kernel path for RELU? What SFPU function implements relu?"
   **Reason**: Needed to understand RELU-specific kernel path resolution and SFPU function mapping.
   **Key Findings**: RELU maps to `eltwise_sfpu.cpp` (default case in `get_compute_kernel_path`), uses `relu_tile()` / `relu_tile_init()` SFPU functions, and the actual HW implementation is `_relu_min_` with lower limit 0.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to verify the exact defines generated for RELU, the compute kernel path, macro definitions, and approx mode.
   **Key Information**: RELU generates `SFPU_OP_RELU_FAMILY_INCLUDE=1`, maps to `eltwise_sfpu.cpp` via default case, uses `relu_tile_init()` / `relu_tile(0)` functions, and `get_op_approx_mode` returns false for RELU.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
   **Reason**: Needed to understand the compute kernel execution pattern shared by RELU.
   **Key Information**: Generic SFPU kernel with `SFPU_OP_CHAIN_0` macro expansion for operation-specific behavior. Uses `init_sfpu`, `copy_tile`, tile register acquire/commit/wait/release pattern.

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h` |
| **LLK Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (params dispatch) and `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h` (init/start/done) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h` (Wormhole) and `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h` (Blackhole) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

Note: There is also a secondary `ckernel_sfpu_relu.h` at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h` (and the Blackhole equivalent) which contains simplified wrapper functions (`relu_min`, `relu_max`, `calculate_lrelu`) that are distinct from the core `_relu_min_` implementation in the tt_llk submodule. The core tt_llk version is the one actually invoked for RELU via the macro chain.

### Call Chain

1. **Compute kernel** calls `relu_tile_init()` and `relu_tile(0)` (defined via preprocessor macro `SFPU_OP_CHAIN_0`).
2. **`relu_tile(idst)`** (in `relu.h`) expands to `MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_min_, RC, APPROX, idst, 0))`.
3. **`SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT`** (in `llk_math_eltwise_unary_sfpu_macros.h`) resolves to `_llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::_relu_min_<sfpi::vFloat, false, 8, uint32_t>, 0, (int)VectorMode::RC, 0)`.
4. **`_llk_math_eltwise_unary_sfpu_params_`** (in `llk_math_eltwise_unary_sfpu_params.h`) sets the DST write address, stalls until SFPU is ready, then iterates over 4 tile faces (RC mode), calling `_relu_min_<vFloat, false, 8, uint32_t>(0)` for each face and advancing the DST pointer by 16 rows (2x `SETRWC` with increment 8) between faces.
5. **`_relu_min_`** (in the tt_llk `ckernel_sfpu_relu.h`) loads the threshold into LREG2 via `_sfpu_load_imm32_`, then calls `_relu_min_impl_` which executes the hardware SFPU instructions (SFPLOAD, SFPMOV, SFPSWAP, SFPSTORE) in a loop over 8 iterations per face.

For init: `relu_tile_init()` expands to `llk_math_eltwise_unary_sfpu_init<SfpuType::relu_min, false>()` which calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::relu_min>()`. This initializes the SFPU config register, configures ADDR_MOD_7 with `{.dest = {.incr = 0}}`, and resets math counters.

### Annotated SFPU Kernel Source

The Wormhole and Blackhole implementations differ significantly. The Wormhole version uses explicit SFPU instructions (SFPLOAD/SFPMOV/SFPSWAP/SFPSTORE) for a hardware-optimized max operation, while the Blackhole version uses higher-level SFPI conditional constructs (`v_if`/`v_endif`).

#### Wormhole B0

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h

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
    int sfpload_instr_mod = DEFAULT; // InstrModLoadStore::DEFAULT for float path
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            _sfpu_load_imm32_(p_sfpu::LREG2, scalar); // load sign+magnitude int threshold into LREG2
            sfpload_instr_mod = INT32_2S_COMP; // use 2's complement load mode for int32
        }
        else
        {
            _sfpu_load_imm32_(p_sfpu::LREG2, threshold); // load float-as-uint32 threshold into LREG2 (for RELU: threshold=0 -> 0x00000000)
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_min_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold, sfpload_instr_mod);
}

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(const int iterations, [[maybe_unused]] VecType threshold, int sfpload_instr_mod)
// VecType=sfpi::vFloat, APPROXIMATION_MODE=false, ITERATIONS=8
{
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, 0);  // load input from DST into LREG0
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);                // copy threshold from LREG2 to LREG1
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);               // conditional swap: max->LREG1, min->LREG0; InstrMod=1 (conditional on all rows), Imm12[0]=0 (INT32 comparison, works for sign+magnitude floats)
        TTI_SFPSTORE(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_3, 0);  // store max(input, threshold) back to DST
        sfpi::dst_reg++;
    }
}
```

#### Blackhole

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h

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
            v_threshold = Converter::as_float(threshold); // reinterpret uint32 bits as float (for RELU: 0 -> 0.0f)
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_min_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
}

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(const int iterations, VecType threshold)
// VecType=sfpi::vFloat, APPROXIMATION_MODE=false, ITERATIONS=8
{
    for (int d = 0; d < iterations; d++)
    {
        VecType a = sfpi::dst_reg[0]; // load input from DST register
        v_if (a < threshold)          // conditional: if input < threshold (0.0f for RELU)
        {
            sfpi::dst_reg[0] = threshold; // clamp to threshold (0.0f)
        }
        v_endif;
        sfpi::dst_reg++;              // advance to next DST row
    }
}
```

### SFPU Instructions Used

**Wormhole B0 implementation** (explicit hardware instructions):

| Instruction | Description |
|-------------|-------------|
| `SFPLOADI` | Loads a 16-bit immediate value into a local register. Used twice in `_sfpu_load_imm32_` to load the full 32-bit threshold: first the lower 16 bits (insmod=10), then the upper 16 bits (insmod=8). For RELU with threshold=0, both halves are 0. |
| `SFPLOAD` | Loads a 32-bit value from the Destination Register file into a specified local register (LREG0). The `InstrModLoadStore` field controls the data interpretation (DEFAULT for float, INT32_2S_COMP for int32). Uses ADDR_MOD_3 for dest address control. |
| `SFPMOV` | Copies the contents of one local register to another. Here it copies LREG2 (threshold) to LREG1 each iteration, since SFPSWAP modifies both operands in-place. |
| `SFPSWAP` | Conditionally exchanges two local registers based on a comparison. With InstrMod=1 and Imm12[0]=0, it performs a conditional exchange using INT32 comparison on all rows. After execution, the larger value (in sign+magnitude integer ordering) goes to LREG1 (VC) and the smaller to LREG0 (VD). For IEEE 754 floats in sign+magnitude form, integer comparison gives correct magnitude ordering, making this a branchless max operation. Latency: 4 cycles (higher than the typical 3-cycle SFPU instruction). |
| `SFPSTORE` | Stores a 32-bit value from a local register (LREG1, containing max result) back to the Destination Register file. Uses the same `InstrModLoadStore` and ADDR_MOD as the corresponding SFPLOAD. |

**Blackhole implementation** (SFPI high-level constructs):

The Blackhole version uses SFPI abstractions (`dst_reg[]`, `v_if`/`v_endif`) which the compiler translates to SFPU instructions. The generated instructions would typically be:
- `SFPLOAD` to load from DST into an LREG
- `SFPSETCC` to set condition codes based on the comparison `a < threshold`
- `SFPLOADI`/`SFPMOV` to load the threshold value
- Conditional `SFPSTORE` (via `SFPENCC`/condition code mechanism) to write the clamped value back

The Blackhole version does not use `SFPSWAP` -- it relies on the condition code mechanism instead, which is a fundamentally different approach to the same problem.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Holds the input value loaded from the Destination Register file. After SFPSWAP, contains the minimum of (input, threshold). |
| **LREG1** | Holds a copy of the threshold (from LREG2). After SFPSWAP, contains the maximum of (input, threshold), which is the RELU result stored back to DST. |
| **LREG2** | Holds the threshold value (0 for plain RELU) loaded once before the iteration loop via `_sfpu_load_imm32_`. Preserved across all iterations as a constant. |
| **LREG3-LREG7** | Not used by the RELU kernel. |
| **DST (Destination Register File)** | The tile data resides here. Each SFPU iteration reads one row (one element per lane across all columns) from DST, computes max(input, 0), and writes the result back. The `dst_reg++` / `SETRWC` calls advance through all 8 rows of a 16x16 face, and the params dispatch layer advances across the 4 faces of a 32x32 tile. |
| **CC Result Register** | Not directly used by Wormhole relu_min (SFPSWAP does not set condition codes). Used by the Blackhole version via `v_if`. |

### Address Mode Configuration

The SFPU init for RELU (`_llk_math_eltwise_unary_sfpu_init_<SfpuType::relu_min>()`) configures:

**ADDR_MOD_7** (set explicitly by `eltwise_unary_sfpu_configure_addrmod`):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
```
This address mode has zero increment on all address counters. It is used by the params dispatch layer (`_llk_math_eltwise_unary_sfpu_params_`) but not directly by the relu kernel itself.

**ADDR_MOD_3** (used by the Wormhole SFPLOAD/SFPSTORE instructions):
ADDR_MOD_3 is **not** explicitly configured by the RELU init path. It uses whatever configuration was set by prior operations or the default state. In the Wormhole `_relu_min_impl_`, the SFPLOAD and SFPSTORE instructions reference ADDR_MOD_3. Since the kernel manages row advancement explicitly via `sfpi::dst_reg++` (which emits `SETRWC` instructions to increment the DST address), the ADDR_MOD_3 setting for dest increment is not critical -- the explicit `dst_reg++` handles the addressing.

**Blackhole note**: The Blackhole version uses ADDR_MOD_7 (same configuration) set in `eltwise_unary_sfpu_configure_addrmod`. The Blackhole `_llk_math_eltwise_unary_sfpu_init_` is identical to Wormhole for this operation since `SfpuType::relu_min` does not match any of the special-cased types (topk, typecast, unary_max/min).

**Cross-face advancement**: Between faces, the params dispatch layer (`_llk_math_eltwise_unary_sfpu_params_`) issues `TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D)` twice (incrementing DST address by 16 rows total) to advance to the next 16x16 face.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How is the RELU SFPU kernel implemented? Trace the call chain from the compute kernel API through LLK dispatch down to the ckernel SFPU implementation."
   **Reason**: Needed to identify file paths and understand the full abstraction layer stack for RELU.
   **Key Findings**: Confirmed the 4-layer architecture: API header (`relu.h`) -> macro dispatch (`llk_math_eltwise_unary_sfpu_macros.h`) -> LLK params/init (`llk_math_eltwise_unary_sfpu_params.h` / `llk_math_eltwise_unary_sfpu.h`) -> ckernel SFPU (`ckernel_sfpu_relu.h`). Identified that `_relu_min_` is the actual function invoked, not `relu_min`.

2. **Query**: "How is relu implemented in the LLK/ckernel layer? What SFPU instructions does the relu ckernel use?"
   **Reason**: Needed detail on the SFPU instruction-level implementation.
   **Key Findings**: Learned that the Wormhole version uses SFPLOAD/SFPMOV/SFPSWAP/SFPSTORE instructions and that SFPSWAP performs a conditional min/max exchange. Also learned about the `_calculate_lrelu_` function using SFPSETCC/SFPMUL/SFPENCC for leaky relu.

### Confluence References
1. **Page**: Tensix SFPU Instruction Set Architecture (Page ID: 1170505767)
   **Sections consulted**: SFPSWAP instruction specification
   **Key Information**: SFPSWAP (opcode 0x92) exchanges values in RG[VC] and RG[VD] conditionally. With InstrMod=1, it conditionally exchanges on all rows, placing the smaller value in RG[VD] and larger in RG[VC]. Imm12[0]=0 selects INT32 comparison mode. Latency is 4 cycles (vs typical 3 for most SFPU instructions). The SWAP_INVERT bit in the SFPU control register can reverse the exchange direction.

### Glean References
No Glean queries were needed for this analysis. The SFPU instruction details were sufficiently covered by Confluence.
