# Typecast (Interleaved) Implementation Analysis

## Overview

The typecast operation converts tensor data between different data types on device (e.g., BFLOAT16 to FLOAT32, FLOAT32 to BFLOAT8_B, etc.). The interleaved program factory (`TypecastProgramFactory`) handles non-sharded tensors in both TILE and ROW_MAJOR layouts. It uses SFPU-based type conversion via the `typecast_tile` LLK, which invokes SFPU instructions (SFPCAST, SFPSHFT2, SFPSWAP, etc.) to perform the actual data format transformation in the DEST registers.

**Program factory path**: `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.cpp`

## Path Selection: FPU vs SFPU

The typecast operation is entirely SFPU-based. There is no FPU path for this operation. The `select_program_factory` method in `typecast_device_op.cpp` selects among four program factories based on tensor properties:

1. **TypecastShardedProgramFactory** -- selected when input is sharded AND meets optimization criteria (matching tile sizes, both buffers in L1, L1-aligned shard dimensions, no sub_core_grids).
2. **TypecastProgramFactory** (this analysis) -- selected as the general-purpose fallback for interleaved TILE layout inputs, or for sharded inputs that fail the optimized factory's criteria.
3. **TypecastSubgridProgramFactory** -- selected when `sub_core_grids` is specified (also in the same source file).
4. **TypecastRowMajorChunkedProgramFactory** -- selected when the input is ROW_MAJOR layout and not sharded.

All four factories use the same SFPU-based compute kernel (`eltwise_typecast.cpp`). The `TypecastProgramFactory` is reached when: (a) the input is not sharded, (b) no `sub_core_grids` are specified, and (c) the input layout is TILE. It is also a fallback for sharded inputs that cannot use the optimized sharded factory. The compute kernel calls `init_sfpu`, `copy_tile`, and the SFPU LLK `typecast_tile` -- there is no FPU math path.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Page (tile for TILE layout, row for ROW_MAJOR layout) |
| **Unit size** | 1 page |
| **Total units** | `num_pages` = total pages in the input buffer |
| **Loop structure** | Outer loop over `per_core_block_cnt` pages, inner loop over `per_core_block_dim` (always 1) |

Each core processes a contiguous range of pages. The inner dimension (`per_core_block_dim`) is always 1, so the compute kernel processes one page at a time.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | Any (arbitrary) | Same as input |
| **Dimension convention** | N/A (generic) | N/A (generic) |
| **Tensor layout** | TILE or ROW_MAJOR | Same as input |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | Any supported (BFLOAT16, FLOAT32, BFLOAT8_B, BFLOAT4_B, UINT16, UINT32, INT32, etc.) | Any supported (different from input) |

### Layout Transformations

No tilize/untilize or reshard operations are performed. The operation preserves the tensor layout (TILE or ROW_MAJOR) and logical shape. Only the data type changes. The CB data formats are derived from the input and output data types via `datatype_to_dataformat_converter`. The hardware unpacker reads data in the input format, the SFPU typecast LLK converts it, and the packer writes in the output format.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `noc_async_read_barrier`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile`, `TYPECAST_LLK`, `pack_tile`, `cb_pop_front(c_0, 1)`, `cb_reserve_back(c_2, 1)`, `cb_push_back(c_2, 1)` |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `noc_async_writes_flushed`, `cb_pop_front(c_2, 1)` |

Data flows page-by-page: the reader fetches one page from DRAM into CB c_0, the compute kernel unpacks it to DEST, applies the SFPU typecast operation, packs the result to CB c_2, and the writer drains one page from CB c_2 to DRAM.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input staging | 2 pages | 1 page | Double | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 pages | 1 page | Double | Compute | Writer | Program |

- Page size for c_0 = `single_tile_size_input` (for TILE) or `src_buffer->page_size()` (for ROW_MAJOR).
- Page size for c_2 = `single_tile_size_output` (for TILE) or `dst_buffer->page_size()` (for ROW_MAJOR).
- Both CBs are double-buffered (capacity = 2 pages, block size = 1 page), enabling overlap between reader/compute and compute/writer stages.

## Pipeline Pattern Summary

Both circular buffers are double-buffered (capacity = 2x block size). This allows:
- The reader to fill one page in c_0 while the compute kernel processes another.
- The compute kernel to fill one page in c_2 while the writer drains another.
- Classification: **Double-buffered pipeline** for both input and output stages.

## Index Calculations

The reader and writer kernels use `TensorAccessor` for index-to-address mapping. The `TensorAccessorArgs` are passed as compile-time arguments, encoding the buffer's interleaved layout properties (number of banks, bank stride, page size alignment). At runtime, `noc_async_read_page(i, s, l1_write_addr)` translates page index `i` into a physical NoC address using the accessor `s`.

Pages are processed sequentially starting from `start_id` (a runtime argument set per core). Each core's `start_id` is the cumulative sum of pages assigned to all preceding cores.

## Memory Access Patterns

### Read Pattern
Sequential page reads. Each core reads a contiguous range of page indices `[start_id, start_id + num_pages)`. Pages are read one at a time with a NoC barrier after each read (`noc_async_read_barrier`), ensuring the page is fully in L1 before being pushed to the compute kernel.

### Write Pattern
Sequential page writes. Each core writes its output pages to the same contiguous page index range `[start_id, start_id + num_pages)`. Writes are flushed after each page (`noc_async_writes_flushed`) with a final `noc_async_write_barrier` after all pages.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (uses full compute grid) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g., 8x8) |
| **Total cores** | min(num_pages, available_cores) |
| **Work per core** | `num_pages / num_cores` (group 1 gets +1 if remainder) |
| **Load balancing** | Two-group: group 1 = ceil(num_pages/num_cores) pages, group 2 = floor(num_pages/num_cores) pages |

The `split_work_to_cores` utility handles distribution. For TILE layout, cores are assigned column-wise (top-to-bottom, left-to-right). For ROW_MAJOR layout, the `is_row_major` flag causes row-wise core ordering. When `num_pages` does not divide evenly, `core_group_1` gets one extra page per core relative to `core_group_2`.

## Arguments

### Compile-Time Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Encodes src_buffer interleaved layout (bank count, stride, alignment) |

**Writer kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | Output CB ID (c_2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encodes dst_buffer interleaved layout |

**Compute kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of pages this core processes |
| 1 | per_core_block_dim | uint32_t | Always 1 (one page per inner iteration) |
| 2 | src0_cb_index | uint32_t | Input CB ID (c_0) |
| 3 | output_cb_index | uint32_t | Output CB ID (c_2) |

### Runtime Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM/L1 address |
| 1 | num_pages | uint32_t | Number of pages for this core |
| 2 | start_id | uint32_t | Starting page index for this core |

**Writer kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM/L1 address |
| 1 | num_pages | uint32_t | Number of pages for this core |
| 2 | start_id | uint32_t | Starting page index for this core |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 (src_buffer) | CB c_0 | Read pages via TensorAccessor |
| Compute | TRISC (RISCV_2) | N/A | CB c_0 | CB c_2 | SFPU typecast (copy_tile + typecast_tile + pack_tile) |
| Writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 (dst_buffer) | Write pages via TensorAccessor |

### Reader Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` |
| **Assigned cores** | `all_cores` (union of core_group_1 and core_group_2) |

**Key Logic:**
- Iterates sequentially from `start_id` to `start_id + num_pages`.
- For each page: reserves one slot in CB c_0 (`cb_reserve_back(c_0, 1)`), reads the page from DRAM/L1 using `noc_async_read_page(i, accessor, l1_write_addr)`, waits for the read to complete (`noc_async_read_barrier`), then pushes the page (`cb_push_back(c_0, 1)`).
- Page size is obtained dynamically from the CB interface (`get_local_cb_interface(cb_id_in0).fifo_page_size`), making it work for both TILE and ROW_MAJOR layouts.
- Supports optional `BACKWARDS` define for reverse iteration (not used in this factory).
- **Synchronization**: Produces into CB c_0. Blocks on `cb_reserve_back` if c_0 is full (compute hasn't consumed).

### Compute Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp` |
| **Assigned cores** | `core_group_1` (and `core_group_2` if non-empty, with separate compile args) |

**Key Logic:**
- Calls `init_sfpu(input_cb, output_cb)` to configure unpacker/packer for the input and output data formats.
- Outer loop iterates `per_core_block_cnt` times (one per page).
- Reserves output space at block granularity: `cb_reserve_back(output_cb, per_core_block_dim)` (always 1).
- Inner loop (always 1 iteration since `per_core_block_dim = 1`):
  - `tile_regs_acquire()` -- acquires DEST registers for unpack+math.
  - `cb_wait_front(input_cb, 1)` -- waits for one page from reader.
  - `copy_tile(input_cb, 0, 0)` -- unpacks the page from CB c_0 into DEST register 0.
  - `TYPECAST_LLK_INIT()` -- macro expanding to `typecast_tile_init<input_df, output_df>()`, initializes SFPU for the specific type conversion.
  - `TYPECAST_LLK(0)` -- macro expanding to `typecast_tile<input_df, output_df>(0)`, performs the SFPU typecast on DEST register 0.
  - `tile_regs_commit()` -- hands DEST to packer.
  - `tile_regs_wait()` -- waits for packer readiness.
  - `pack_tile(0, output_cb)` -- packs DEST register 0 to CB c_2.
  - `cb_pop_front(input_cb, 1)` -- frees the input page.
  - `tile_regs_release()` -- releases DEST for next iteration.
- `cb_push_back(output_cb, per_core_block_dim)` -- publishes the output page after the inner loop.
- The `TYPECAST_LLK_INIT` and `TYPECAST_LLK` macros are defined via preprocessor defines in the program factory, parameterized with the numeric data format IDs of input and output types. This enables the LLK to select the correct SFPU instruction sequence for the given type conversion pair.
- **Synchronization**: Consumes from CB c_0 (blocks on `cb_wait_front`), produces into CB c_2 (blocks on `cb_reserve_back`).

### Writer Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` |
| **Assigned cores** | `all_cores` (union of core_group_1 and core_group_2) |

**Key Logic:**
- Iterates sequentially from `start_id` to `start_id + num_pages`.
- For each page: waits for one page in CB c_2 (`cb_wait_front(c_2, 1)`), reads the L1 pointer, writes the page to DRAM/L1 using `noc_async_write_page(i, accessor, l1_read_addr)`, flushes (`noc_async_writes_flushed`), then frees the page (`cb_pop_front(c_2, 1)`).
- After all pages, calls `noc_async_write_barrier()` to ensure all writes complete.
- Supports `OUT_SHARDED` define for sharded output (not used in this factory) and `BACKWARDS` for reverse iteration (not used).
- **Synchronization**: Consumes from CB c_2. Blocks on `cb_wait_front` if c_2 is empty (compute hasn't produced).

## Implementation Notes

- **Program factory variants**: The typecast operation has four program factories: `TypecastProgramFactory` (this analysis, for interleaved TILE or fallback sharded), `TypecastShardedProgramFactory` (optimized sharded), `TypecastSubgridProgramFactory` (sub_core_grids), and `TypecastRowMajorChunkedProgramFactory` (interleaved ROW_MAJOR). Selection logic is in `select_program_factory()` in `typecast_device_op.cpp`.
- **Type-based operation variants**: All data type pairs are handled by the same compute kernel. The specific SFPU conversion routine is selected via the `TYPECAST_LLK` and `TYPECAST_LLK_INIT` preprocessor macros, which encode the input and output data format IDs as template parameters to `typecast_tile<input_df, output_df>`.
- **UnpackToDestFP32 mode**: Enabled when `preserve_fp32_precision` is true. Sets `UnpackToDestMode::UnpackToDestFp32` on CB c_0, causing the unpacker to convert data to FP32 in DEST regardless of the input format. This preserves precision during type conversion.
- **Broadcast type selection**: N/A. Typecast is a unary elementwise operation with no broadcasting.
- **Sharding support and constraints**: This specific factory (`TypecastProgramFactory`) primarily targets interleaved tensors. It can also be used as a fallback for sharded inputs that fail the optimized sharded factory's criteria. The validation requires input and output memory layouts to match.
- **FP32 dest accumulation**: Controlled by `fp32_dest_acc_en` parameter, passed directly to `ComputeConfig`. When enabled, DEST registers use FP32 precision. This is separate from `preserve_fp32_precision` (which controls unpack mode). Both can be independently set.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel functions that the compute kernel dispatches to. The typecast operation is unique among SFPU operations because it is not a single SFPU kernel but rather a **family of 13 distinct SFPU kernels**, each handling a specific source-to-destination data format conversion. The LLK dispatch layer selects the appropriate kernel at compile time based on the `IN_DTYPE` and `OUT_DTYPE` template parameters. Some format pairs (e.g., Float16_b to Float32, Bfp8_b to Float16_b) require no SFPU kernel at all -- the unpacker/packer hardware handles the conversion directly.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_typecast.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `TYPECAST_LLK(0)`, which is a preprocessor macro expanding to `typecast_tile<IN_DTYPE, OUT_DTYPE>(0)`.
2. `typecast_tile<IN_DTYPE, OUT_DTYPE>(idst)` (in `typecast.h`) calls `llk_math_eltwise_unary_sfpu_typecast<APPROX, IN_DTYPE, OUT_DTYPE>(idst)` wrapped in the `MATH()` macro (only active on TRISC_MATH RISC-V).
3. `llk_math_eltwise_unary_sfpu_typecast` (in `llk_math_eltwise_unary_sfpu_typecast.h`) uses a compile-time `if constexpr` chain to select the correct `_calculate_typecast_*` function based on the format pair, and calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_func, dst_index, vector_mode)`.
4. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) sets up the DEST write address, stalls until SFPU is ready, then loops over tile faces (4 faces for `VectorMode::RC`) calling the selected `calculate_typecast_*` function once per face and advancing the DEST address by `DEST_FACE_WIDTH` (16 rows) between faces.
5. The `calculate_typecast_*` wrapper (in the metal-level `ckernel_sfpu_typecast.h`) directly delegates to the `_calculate_typecast_*` core implementation in the tt_llk-level `ckernel_sfpu_typecast.h`.

Similarly, `TYPECAST_LLK_INIT()` expands to `typecast_tile_init<IN_DTYPE, OUT_DTYPE>()` which calls `llk_math_eltwise_unary_sfpu_typecast_init`, which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::typecast, APPROXIMATE>` (configuring address modes) and then a format-specific `_init_typecast_*` function (if the format pair requires SFPLOADMACRO setup).

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (the default). All 4 faces of a 32x32 tile are processed (faces 0-3).
- **Operation invocation**: The params dispatch loops 4 times (`for (int face = 0; face < 4; face++)`), calling the selected `calculate_typecast_*<APPROXIMATE, 8>` function once per face. Each function internally iterates 8 times (ITERATIONS=8) over the rows within a face, processing one row of 32 elements per iteration.
- **DEST address progression**: Between faces, the DEST read/write pointer is advanced by `DEST_FACE_WIDTH` (16 rows = 2x `inc_dst_addr<8>()` calls). Within a face, the `_calculate_typecast_*` functions use `ADDR_MOD_6` (dest increment = 2) on SFPLOAD/SFPSTORE/SFPLOADMACRO instructions to advance through the 8 row pairs, or use explicit `dst_reg++` for the SFPI-based `_calculate_typecast_fp32_to_int32_`. The Wormhole and Blackhole params dispatch implementations differ slightly: Wormhole uses explicit `TTI_SETRWC` instructions while Blackhole uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()`, but the net DEST advancement is the same.

### Annotated SFPU Kernel Source

The typecast operation has **13 distinct `_calculate_typecast_*` functions** plus **11 `_init_typecast_*` functions**. The kernel style is predominantly **Style B** (raw `TT_`/`TTI_` instructions with SFPLOADMACRO-based pipelining), with one exception: `_calculate_typecast_fp32_to_int32_` uses SFPI abstractions on Wormhole (Style A) and raw TTI instructions on Blackhole (Style B). Due to the complexity of the CC manipulation in the Blackhole `fp32_to_int32` and `fp32_to_uint32` kernels, CC State Machine diagrams are provided for those.

The source code below is from the **Wormhole** implementation (`tt_llk_wormhole_b0`). Differences with Blackhole are noted where they exist; the key difference is that Wormhole uses `ADDR_MOD_6` and `ADDR_MOD_7` while Blackhole uses `ADDR_MOD_2` and `ADDR_MOD_3` for the same logical purpose (the address mode values are identical across architectures).

#### Conversion Kernel Categorization

The 13 conversion kernels fall into three categories based on their SFPU implementation strategy:

**Category 1: SFPLOADMACRO-pipelined kernels** -- These use `SFPLOADMACRO` to orchestrate SFPU sub-unit pipelines (Load, Simple, MAD, Round, Store) for high throughput. The `_init_typecast_*` function configures instruction templates and macro definitions. The `_calculate_*` function merely issues SFPLOADMACRO and a few explicit instructions per iteration:
- `_calculate_typecast_fp32_to_uint16_` (2 cycles/row)
- `_calculate_typecast_uint16_to_fp16b_` (1 cycle/row)
- `_calculate_typecast_int32_to_fp16b_` (4 cycles/row)
- `_calculate_typecast_fp32_to_fp16b_` (3 cycles/row)
- `_calculate_typecast_uint16_to_fp32_` (1 cycle/row)
- `_calculate_typecast_int32_to_fp32_` (4 cycles/row)
- `_calculate_typecast_uint32_to_fp16b_` (3 cycles/row)
- `_calculate_typecast_uint32_to_fp32_` (3 cycles/row)
- `_calculate_typecast_uint16_to_uint32_` (1 cycle/row)
- `_calculate_typecast_uint32_to_uint16_` (2 cycles/row)
- `_calculate_typecast_int32_to_uint16_` (3 cycles/row)

**Category 2: Raw TTI kernels with CC manipulation** -- These use explicit SFPLOAD/SFPSTORE and CC-modifying instructions:
- `_calculate_typecast_fp32_to_int32_` (Blackhole: TTI-based with CC; Wormhole: SFPI-based)
- `_calculate_typecast_fp32_to_uint32_`

**Category 3: SFPI-based kernels** -- These use SFPI abstractions (`vFloat`, `vInt`, `dst_reg`, `v_if`):
- `_calculate_typecast_fp32_to_int32_` (Wormhole only)

#### Core SFPU Source Code (Wormhole)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_typecast.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint16_()
{
    // Implementation notes, see the original file for more details
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1; // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, v >> 2);
        TTI_SFPNOP;
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp16b_()
{
    // Implementation notes, see the original file for more details
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1; // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_6, v >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp16b_()
{
    // Implementation notes, see the original file for more details

    constexpr int t = p_sfpu::LREG4;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00); // -2**31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // alternate between p_sfpu::LREG2 and p_sfpu::LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_6, v >> 2);
        TT_SFPABS(0, v, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5); // SFPSHFT2_MOD1_SHFT_LREG: L7 = t >> 31 (extract sign)
        TTI_SFPCAST(t, t, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// SFPI implementation of FP32 to INT32 typecast (Wormhole).
// Converts IEEE 754 single-precision float to two's complement int32.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_int32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in = sfpi::dst_reg[0];

        // Extract exponent (debiased: exp=0 means value is 1.something)
        sfpi::vInt exp = sfpi::exexp(in);

        // Extract mantissa with implicit 1 bit at position 23
        sfpi::vUInt man = sfpi::exman8(in);

        // Compute shift amount: (exp - 23)
        sfpi::vInt shift_amt = exp - 23;

        // Compute result = mantissa << shift_amt
        sfpi::vInt result = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(man, shift_amt));

        // Handle overflow: exp >= 31 means |value| >= 2^31
        v_if (exp >= 31) {
            result = 0x80000000;  // INT_MIN
        }
        v_endif;

        // Handle underflow: exp < 0 means |value| < 1
        v_if (exp < 0) {
            result = 0;
        }
        v_endif;

        // Handle negative input: apply two's complement negation
        v_if (in < 0.0f) {
            result = ~result + 1;
        }
        v_endif;

        // Store result
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0); // Load from DEST, no addr increment
        // result = 0
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0);

        // LaneEnabled = in >= 0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
        // exp = in.Exp (LaneEnabled = exp >= 0)
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // result = 0xffffffff
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, 0xffff);
        // exp -= 32 (LaneEnabled = exp < 32)
        TTI_SFPIADD(-32 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        // exp += 9
        TTI_SFPIADD(9, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // result = exman8(in) << (exp - 23)
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_6, 0); // Store to DEST, addr += 2
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_fp16b_()
{
    // Implementation notes, see the original file for more details

    constexpr int b = p_sfpu::LREG2;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int a = d & 1; // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (a & 3), 0, ADDR_MOD_7, a >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), 0, ADDR_MOD_6, b >> 2);
        TT_SFPAND(0, p_sfpu::LREG12, a, 0); // a &= 1 (extract rounding bit)
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp32_()
{
    // Implementation notes, see the original file for more details

    constexpr int v = p_sfpu::LREG0;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_6, v >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp32_()
{
    // Implementation notes, see the original file for more details

    constexpr int t = p_sfpu::LREG4;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00); // -2**31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // alternate between p_sfpu::LREG2 and p_sfpu::LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_6, v >> 2);
        TT_SFPABS(0, v, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5); // SFPSHFT2_MOD1_SHFT_LREG
        TTI_SFPCAST(t, t, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp16b_()
{
    // Implementation notes, see the original file for more details

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00); // 2**31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // alternate between p_sfpu::LREG2 and p_sfpu::LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_6, v >> 2);
        TT_SFPSHFT2(v, p_sfpu::LREG12, p_sfpu::LREG7, 5); // SFPSHFT2_MOD1_SHFT_LREG
        TT_SFPSETSGN(0, v, v, 1); // SFPSETSGN_MOD1_ARG_IMM: clear sign bit
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp32_()
{
    // Implementation notes, see the original file for more details

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00); // 2**31

    constexpr int a  = p_sfpu::LREG2;
    constexpr int b  = p_sfpu::LREG3;
    constexpr int L7 = p_sfpu::LREG7;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_7, a >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_7, b >> 2);
        TTI_SFPLOADMACRO((2 << 2) | (L7 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, L7 >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_uint32_()
{
    // Implementation notes, see the original file for more details
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | 0, InstrModLoadStore::LO16, ADDR_MOD_6, 0);
    }
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_uint16_()
{
    // Implementation notes, see the original file for more details

    constexpr int a = p_sfpu::LREG0;
    constexpr int b = p_sfpu::LREG1;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::LO16, ADDR_MOD_7, a >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_6, b >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_uint16_()
{
    // Implementation notes, see the original file for more details
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int a = d & 1; // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_6, a >> 2);
        TT_SFPCAST(a, a, 0); // Cast int32 to fp32
        TTI_SFPNOP;
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
```

#### CC State Machine: `_calculate_typecast_fp32_to_uint32_` (Wormhole)

```
_calculate_typecast_fp32_to_uint32_ — CC State Transitions (per iteration)
════════════════════════════════════════════════════════════════

  CC State: ALL_ENABLED                   <-- initial state

       |
       |  SFPLOAD  L0 = DEST[addr]       (no CC effect)
       |  SFPLOADI L1 = 0                 (no CC effect)
       |
       v
  +-------------------------------------------+
  | SFPSETCC  LREG_GTE0                       |
  |                                           |
  | CC <- (L0 >= 0)                           |
  +-------------------+-----------------------+
                      |
                      v
  CC State: ENABLED where input >= 0
       |
       v
  +-------------------------------------------+
  | SFPEXEXP  SET_CC_SGN_EXP | SET_CC_COMP_EXP|
  |                                           |
  | CC <- CC_prev AND (unbiased_exp >= 0)     |
  |    = (input >= 0) AND (exp >= 0)          |
  +-------------------+-----------------------+
                      |
                      v
  CC State: ENABLED where input >= 0 AND exp >= 0
       |
       |  SFPLOADI L1 = 0xFFFFFFFF       (CC-guarded: only input>=0 AND exp>=0 lanes)
       |
       v
  +-------------------------------------------+
  | SFPIADD  -32, CC_LT0                      |
  |                                           |
  | CC <- CC_prev AND (exp - 32 < 0)          |
  |    = (input >= 0) AND (0 <= exp < 32)     |
  +-------------------+-----------------------+
                      |
                      v
  CC State: ENABLED where input >= 0 AND 0 <= exp < 32
       |
       |  SFPIADD  +9, CC_NONE           (CC-guarded; CC_NONE = no CC update)
       |  SFPEXMAN L0 -> L1              (CC-guarded: extract mantissa)
       |  SFPSHFT  L1 <<= L2             (CC-guarded: shift mantissa)
       |
       v
  +-------------------------------------------+
  | SFPENCC                                   |
  |                                           |
  | CC <- ALL_ENABLED                         |
  +-------------------+-----------------------+
                      |
                      v
  CC State: ALL_ENABLED
       |
       |  SFPSTORE L1 -> DEST[addr], addr += 2   (all lanes)
       v
```

**CC logic summary**: Negative inputs get result=0 (initial value). Inputs with exp >= 32 get result=0xFFFFFFFF (UINT_MAX). Valid positive inputs (0 <= exp < 32) get the mantissa shifted to produce the integer magnitude.

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `SFPLOADMACRO` | Macro-scheduled instruction that orchestrates Load, Simple, MAD, Round, and Store SFPU sub-units in a pipelined fashion. The macro index and configuration (set during init) determine which sub-unit operations execute. This is the primary instruction for high-throughput typecast kernels. |
| `SFPLOAD` | Loads a 32-element row from DEST into an SFPU local register. Used in non-SFPLOADMACRO kernels (fp32_to_int32, fp32_to_uint32). |
| `SFPSTORE` | Stores an SFPU local register back to a DEST row. Used in non-SFPLOADMACRO kernels. |
| `SFPLOADI` | Loads an immediate value into an SFPU local register. Used to set up constants (0, -2^31, 2^31, 0xFFFFFFFF, etc.). |
| `SFPCAST` | Casts an unsigned integer value in a local register to IEEE FP32 format. Used in int-to-float conversions. |
| `SFPABS` | Computes absolute value of a local register. Used in signed int-to-float conversions to separate magnitude from sign. |
| `SFPSHFT2` | Bitwise shift operation on a local register. `MOD1_SHFT_LREG` shifts by a value in another register; `MOD1_SHFT_IMM` shifts by an immediate. Used for sign extraction (shift right 31) and rounding bit extraction. |
| `SFPSHFT` | Barrel shift of a local register by the amount in another register. Used to shift mantissa by the exponent offset in int conversions. |
| `SFPEXEXP` | Extracts the unbiased exponent from a float in a local register. With `SET_CC_SGN_EXP | SET_CC_COMP_EXP`, also sets CC based on sign and exponent value. |
| `SFPEXMAN` | Extracts the mantissa with implicit leading 1 bit at position 23. Used in float-to-integer conversions. |
| `SFPSETCC` | Sets the condition code (lane enable mask) based on a register comparison. `LREG_GTE0` enables lanes where the register >= 0; `LREG_LT0` enables lanes where < 0. |
| `SFPENCC` | Resets condition code to ALL_ENABLED (all lanes active). |
| `SFPIADD` | Integer add with immediate or two's complement negation. With `CC_LT0`/`CC_GTE0`, also narrows the CC mask. With `ARG_2SCOMP_LREG_DST`, computes two's complement negation. |
| `SFPAND` | Bitwise AND between two local registers. Used in fp32_to_fp16b for rounding bit extraction. |
| `SFPSETSGN` | Sets or clears the sign bit of a float. `MOD1_ARG_IMM` with imm=0 clears the sign bit. Used in unsigned int-to-float conversions. |
| `SFPSWAP` | Min/max swap operation. With `LCONST_0` and mod=0xf, computes `max(0, L[VD])`. Used to clamp negative values to zero in uint16 conversions. |
| `SFP_STOCH_RND` | Stochastic rounding / format conversion. `MOD1_FP32_TO_FP16B` rounds FP32 to FP16B format. `MOD1_FP32_TO_UINT16` converts FP32 to unsigned 16-bit with clamping to [0, 65535]. |
| `SFPMAD` | Multiply-accumulate: `VA * VB + VC`. With `MOD1_INDIRECT_VA`, VA is selected indirectly via LREG7 (used for sign-dependent correction: L[L7]*1.0 + v). |
| `SFPCONFIG` | Configures SFPLOADMACRO instruction templates, macro definitions, and miscellaneous settings (store format, delay kind). Only used in `_init_typecast_*` functions. |
| `SFPNOP` | No-operation. Used for pipeline drainage between SFPLOADMACRO sequences to ensure all pipelined operations complete. |
| `SFPOR` | Bitwise OR between two local registers. Used in uint32_to_uint16 conversion to combine high and low 16-bit halves. |
| `SFPGT` | Greater-than comparison. Used in Blackhole's uint32_to_uint16 init to set VD = (VD > 0) ? -1 : 0. |

### SFPU Register Usage

**DEST registers**: The SFPU reads input data from and writes output data to DEST rows. Each face has 16 rows of 32 elements. The SFPU processes rows in pairs (even/odd) using ADDR_MOD_6 (dest increment = 2) to step through rows 0, 2, 4, ..., 14 in 8 iterations.

**SFPU Local Registers (LREG0-LREG7, LREG12, LREG13)**:
- **LREG0, LREG1**: Dual-purpose. In SFPLOADMACRO kernels, used as alternating VD registers for double-buffered pipeline processing. In non-SFPLOADMACRO kernels, LREG0 holds the input value and LREG1 holds the result.
- **LREG2, LREG3**: Used as alternating VD registers in kernels that need LREG0/1 for constants (e.g., int32_to_fp32 where L0=0.0, L1=-2^31). Also used as the exponent register in fp32_to_uint32.
- **LREG4**: Temporary register (`t`) for intermediate values in signed int-to-float conversions (abs value, cast result).
- **LREG7**: Sign bit storage. Stores the extracted sign bit (0 or 1) via `SFPSHFT2 >> 31`. Used as an indirect index for SFPMAD with `MOD1_INDIRECT_VA` to select L0 or L1 based on sign.
- **LREG12**: Programmable constant register (`vConstIntPrgm0`). Loaded with format-specific values: 1 (for rounding bit mask in fp32_to_fp16b), -31 (for exponent adjustment in int/uint conversions).
- **LREG13**: Programmable constant register (`vConstIntPrgm1`). Loaded with 0x7FFF in fp32_to_fp16b for rounding bias.
- **LCONST_0**: Hardware constant register containing 0.0. Used as the comparison operand in SFPSWAP for clamping to zero.
- **LCONST_1**: Hardware constant register containing 1.0. Used as VB in SFPMAD for `L[L7] * 1.0 + v`.

### Address Mode Configuration

Two address modes are configured for the typecast operation. The configuration is identical across Wormhole and Blackhole architectures (same field values, same ADDR_MOD indices).

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| **ADDR_MOD_7** | 0 | 0 | 0 | No DEST address increment. Used for SFPLOAD/SFPLOADMACRO instructions that should not advance the DEST pointer (e.g., loading input without moving to next row, or the first of two macro invocations in a double-load pattern). |
| **ADDR_MOD_6** | 0 | 0 | 2 | DEST address increments by 2 rows. Used for SFPSTORE/SFPLOADMACRO instructions that should advance to the next row pair after processing. With 8 iterations per face and increment=2, this covers rows 0-14 (16 rows per face). |

These address modes are set by `eltwise_unary_sfpu_configure_addrmod<SfpuType::typecast>()` during `typecast_tile_init()`. The choice of ADDR_MOD_6 vs ADDR_MOD_7 per instruction determines whether DEST advances after that instruction. In SFPLOADMACRO-pipelined kernels, the store step uses ADDR_MOD_6 (advance) while load steps may use either depending on whether they need to advance (most use ADDR_MOD_6 for load+advance, some use ADDR_MOD_7 for load-without-advance in double-load patterns like fp32_to_fp16b and uint32_to_fp32).

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the typecast operation work in TTNN? What are the different program factory paths?"
   **Reason**: Needed to understand the overall operation architecture and factory selection before reading code.
   **Key Findings**: Four program factories exist (sharded optimized, general, subgrid, row-major chunked). The general factory serves as fallback. All paths use SFPU-based compute.

2. **Query**: "What does typecast_tile and typecast_tile_init do in the LLK? Does it use SFPU or FPU?"
   **Reason**: Needed to confirm the compute path is SFPU-based and understand the LLK template parameters.
   **Key Findings**: `typecast_tile` and `typecast_tile_init` are SFPU operations in the `ckernel::sfpu` namespace. They use SFPU instructions (SFPCAST, SFPSHFT2, SFPSWAP, etc.) for data format conversion. Template parameters are `APPROXIMATION_MODE` and `ITERATIONS`.

3. **Query**: "What does init_sfpu do? What does copy_tile do? How does tile_regs synchronization work?"
   **Reason**: Needed to understand the compute kernel's synchronization pattern and data flow through DEST registers.
   **Key Findings**: `init_sfpu` configures unpacker/packer for input/output CB data types. `copy_tile` unpacks a tile from CB to DEST. The acquire/commit/wait/release pattern manages DEST register ownership between unpacker, math core (SFPU), and packer.

4. **Query**: "How does split_work_to_cores work in tt-metal?"
   **Reason**: Needed to understand core distribution strategy.
   **Key Findings**: Returns a 6-tuple with two core groups. Group 1 gets ceil(work/cores) units, group 2 gets floor(work/cores). Supports row-wise or column-wise grid ordering.

5. [SFPU] **Query**: "How does the typecast_tile SFPU kernel work? What is the call chain from typecast_tile through the LLK layers to the ckernel SFPU implementation? What SFPU instructions does it use?"
   **Reason**: Needed to understand the full call chain, file locations, and SFPU instruction set used by the typecast SFPU kernels.
   **Key Findings**: Typecast uses a family of `_calculate_typecast_*` and `_init_typecast_*` functions in `ckernel_sfpu_typecast.h`, one per format pair. Most use SFPLOADMACRO for pipelined throughput. The call chain goes: `typecast_tile` -> `llk_math_eltwise_unary_sfpu_typecast` -> `_llk_math_eltwise_unary_sfpu_params_` -> `calculate_typecast_*` -> `_calculate_typecast_*`. Address mode ADDR_MOD_6 has dest.incr=2.

### Documentation References

1. **Source**: `typecast_device_op.cpp` (lines 56-78)
   **Reason**: Understanding factory selection logic.
   **Key Information**: Selection priority: sharded optimized > general sharded fallback > subgrid > row-major chunked > general interleaved.

2. **Source**: `typecast_device_op_types.hpp`
   **Reason**: Understanding operation parameters.
   **Key Information**: `TypecastParams` includes `input_dtype`, `output_dtype`, `fp32_dest_acc_en`, `preserve_fp32_precision`, `bfp8_pack_precise`, and optional `sub_core_grids`.
