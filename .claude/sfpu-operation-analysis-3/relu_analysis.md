# RELU Implementation Analysis

## Overview

RELU (Rectified Linear Unit) computes `f(x) = max(0, x)` element-wise on each element of the input tensor. It is implemented as an SFPU unary operation using the generic `eltwise_sfpu.cpp` compute kernel with the `SFPU_OP_RELU_FAMILY_INCLUDE` macro, which pulls in the `relu.h` header. The underlying SFPU implementation uses the `_relu_min_` kernel with a threshold of 0, which leverages hardware SFPSWAP instruction to select the maximum of the input and zero.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Path Selection: FPU vs SFPU

RELU is a pure SFPU operation -- there is no FPU path. The `get_compute_kernel_path` function in `unary_op_utils.cpp` falls through to the `default` case, which returns `"eltwise_sfpu.cpp"`. The SFPU path is therefore always selected for RELU regardless of data type or configuration.

The factory selection is based on memory layout, not FPU vs SFPU:
- **Sharded input**: `UnaryShardedProgramFactory` (in `unary_sharded_program_factory.cpp`)
- **Sub-core grids specified**: `UnarySubCoreGridProgramFactory` (in `unary_program_factory.cpp`, second factory)
- **Default (interleaved)**: `UnaryProgramFactory` (in `unary_program_factory.cpp`, first factory)

This analysis covers the default `UnaryProgramFactory` (interleaved) SFPU path.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) or row (for ROW_MAJOR layout) |
| **Unit size** | 1 page (1 tile in TILE_LAYOUT, 1 row in ROW_MAJOR) |
| **Total units** | `input.buffer()->num_pages()` |
| **Loop structure** | Outer loop over blocks (per_core_block_cnt), inner loop processes 1 tile per iteration (per_core_block_size = 1) |

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Any shape (arbitrary rank) |
| **Dimension convention** | Flattened to pages |
| **Tensor layout** | TILE_LAYOUT or ROW_MAJOR |
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
| **Data type** | Same as input (or different for bitcast, but not applicable to RELU) |

### Layout Transformations

No layout transformations are performed. Input and output share the same tensor layout and memory layout.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `noc_async_read_barrier`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_reserve_back(c_2, 1)`, `cb_wait_front(c_0, 1)`, `copy_tile`, SFPU op, `pack_tile`, `cb_pop_front(c_0, 1)`, `cb_push_back(c_2, 1)` |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `noc_async_writes_flushed`, `cb_pop_front(c_2, 1)` |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | src0 | Input staging | 2 pages | 1 page | Double | Reader | Compute | Program |
| c_2 | output | Output staging | 2 pages | 1 page | Double | Compute | Writer | Program |

Note: CB c_1 (tmp0) is only allocated for HARDSHRINK, CBRT, or LOGIT operations -- not for RELU.

Page size depends on layout: for TILE_LAYOUT it is `tile_size(cb_data_format)` (e.g., 2048 bytes for BF16); for ROW_MAJOR it is the buffer's page size.

## Pipeline Pattern Summary

Both c_0 and c_2 have capacity = 2 pages with block size = 1 page, giving **double-buffered** configuration. This allows the reader to write the next tile while compute processes the current one, and similarly compute can produce the next output tile while the writer drains the current one. All three kernels can overlap execution.

## Index Calculations

The program factory uses `TensorAccessor` for both reader and writer kernels to map page indices to physical memory locations. The reader and writer each receive a `start_id` (the first page index assigned to that core) and iterate sequentially through `num_pages` pages.

The `TensorAccessor` handles the mapping from logical page index to the correct DRAM bank and offset, abstracting away interleaved bank mapping. The page index `i` is passed to `noc_async_read_page(i, s, l1_write_addr)` and `noc_async_write_page(i, s, l1_read_addr)`.

## Memory Access Patterns

### Read Pattern

Sequential page reads. Each core reads a contiguous range of pages starting from `start_id` through `start_id + num_pages - 1`. Pages are read one at a time with `noc_async_read_page` followed by `noc_async_read_barrier` (blocking per page). This is a simple sequential access pattern across interleaved DRAM banks.

### Write Pattern

Sequential page writes. Each core writes pages in the same order they were read. Pages are written one at a time with `noc_async_write_page` followed by `noc_async_writes_flushed` (non-blocking flush per page). A final `noc_async_write_barrier` ensures all writes complete before the kernel finishes.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major traversal) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g., 8x8) |
| **Total cores** | Determined by `split_work_to_cores` based on `num_pages` |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` pages |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_pages / num_cores)` pages, group 2 gets `floor(num_pages / num_cores)` pages |

Core indexing uses column-major order: `core = {i / num_cores_y, i % num_cores_y}`. The `split_work_to_cores` utility divides work into two core groups to handle remainder pages. Cores in group 1 process one more page than cores in group 2. Two separate compute kernels are created with different `per_core_block_cnt` compile-time arguments for the two groups.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Tensor accessor parameters for src_buffer (bank mapping, page size, etc.) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer index (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Tensor accessor parameters for dst_buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (pages) this core processes |
| 1 | per_core_block_dim | uint32_t | Number of tiles per block (always 1) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages this core reads |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages this core writes |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Packed scalar parameter (0 for RELU -- no parameter needed) |
| 1 | packed_scalar2 | uint32_t | Packed scalar parameter (0 for RELU -- unused) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 | CB c_0 | Read pages via TensorAccessor |
| compute | TRISC (RISCV_2) | N/A | CB c_0 | CB c_2 | `copy_tile` to DST, `relu_tile(0)` SFPU op, `pack_tile` to CB |
| writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 | Write pages via TensorAccessor |

### Reader Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` |
| **Assigned cores** | all_cores (both core groups) |

**Key Logic**:
- Receives `src_addr`, `num_pages`, and `start_id` as runtime arguments
- Constructs a `TensorAccessor` from compile-time `TensorAccessorArgs` with the source address and page size
- Page size is obtained from the CB interface: `get_local_cb_interface(cb_id_in0).fifo_page_size`
- Iterates sequentially from `start_id` to `start_id + num_pages`
- Per-page loop body: `cb_reserve_back(c_0, 1)` -> get write pointer -> `noc_async_read_page(i, s, l1_write_addr)` -> `noc_async_read_barrier()` -> `cb_push_back(c_0, 1)`
- Supports optional `BACKWARDS` define for reverse iteration (not used by RELU)
- **Synchronization**: Produces to CB c_0. Waits on `cb_reserve_back` for space, signals data availability with `cb_push_back`

### Compute Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` |
| **Assigned cores** | core_group_1 and core_group_2 (separate kernel instances with different compile-time args) |

**Key Logic**:
- Receives `per_core_block_cnt` and `per_core_block_dim` (always 1) as compile-time arguments
- Calls `init_sfpu(c_0, c_2)` to initialize SFPU with input/output CB indices
- Outer loop iterates `per_core_block_cnt` times (one iteration per page)
- Calls `cb_reserve_back(c_2, per_core_block_dim)` at the start of each outer iteration
- Inner loop (single iteration since `per_core_block_dim = 1`):
  - `tile_regs_acquire()` -- acquire DST register for exclusive use
  - `cb_wait_front(c_0, 1)` -- wait for input tile from reader
  - `copy_tile(c_0, 0, 0)` -- unpack tile from CB c_0 into DST register 0
  - `SFPU_OP_CHAIN_0` macro expands to: `relu_tile_init(); relu_tile(0);`
  - `tile_regs_commit()` -- release DST for packer
  - `tile_regs_wait()` -- wait for packer readiness
  - `pack_tile(0, c_2)` -- pack DST register 0 into CB c_2
  - `cb_pop_front(c_0, 1)` -- free the input tile in CB c_0
  - `tile_regs_release()` -- release DST register
- `cb_push_back(c_2, per_core_block_dim)` after inner loop completes
- **SFPU implementation detail**: `relu_tile(0)` calls `SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_min_, RC, APPROX, 0, 0)` which invokes `_relu_min_` with threshold = 0. The LLK implementation uses SFPLOAD to load from DST, SFPMOV to copy threshold (0) to LREG1, SFPSWAP to find maximum of input and threshold, then SFPSTORE to write result back to DST. For each row of the tile, this effectively computes `max(x, 0)`.
- **Synchronization**: Consumes from CB c_0 (`cb_wait_front` / `cb_pop_front`), produces to CB c_2 (`cb_reserve_back` / `cb_push_back`). Uses `tile_regs_acquire/commit/wait/release` for DST register synchronization between unpacker, SFPU, and packer.

### Writer Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` |
| **Assigned cores** | all_cores (both core groups) |

**Key Logic**:
- Receives `dst_addr`, `num_pages`, and `start_id` as runtime arguments
- Output CB index `cb_id_out` is a compile-time argument (value 2, i.e., c_2)
- Constructs a `TensorAccessor` from compile-time `TensorAccessorArgs<1>()` (index 1 to skip the cb_id_out arg)
- Iterates sequentially from `start_id` to `start_id + num_pages`
- Per-page loop body: `cb_wait_front(c_2, 1)` -> get read pointer -> `noc_async_write_page(i, s, l1_read_addr)` -> `noc_async_writes_flushed()` -> `cb_pop_front(c_2, 1)`
- Final `noc_async_write_barrier()` ensures all writes complete
- Supports optional `OUT_SHARDED` define (waits for all pages at once, no write loop) and `BACKWARDS` define -- neither used for default RELU
- **Synchronization**: Consumes from CB c_2. Waits on `cb_wait_front` for data, frees space with `cb_pop_front`

## Implementation Notes

- **Program factory variants**: Three program factories can initiate RELU: `UnaryProgramFactory` (interleaved, default), `UnarySubCoreGridProgramFactory` (interleaved with explicit sub-core grids), and `UnaryShardedProgramFactory` (sharded inputs). Factory selection is based on `input.is_sharded()` and `args.sub_core_grids.has_value()` in `UnaryDeviceOperation::select_program_factory`.

- **Type-based operation variants**: RELU supports BFLOAT16, FLOAT32, INT32, and UINT32 data types. The program factory sets type-specific defines (`INP_FLOAT32`, `INP_INT32`, `INP_UINT32`, or `INP_FLOAT`). For INT32 inputs, the SFPU_OP_CHAIN expands to `relu_tile_init(); relu_tile_int32(0);` which uses `SFPU_UNARY_ONE_PARAM_KERNEL_FN_INT` instead of the float variant, using INT32_2S_COMP load/store instruction modifier.

- **UnpackToDestFP32 mode**: Enabled when `args.preserve_fp32_precision` is true. Sets `UnpackToDestMode::UnpackToDestFp32` for CB c_0 and c_1 (tmp), causing the unpacker to convert input data to FP32 in the DST register regardless of the input format.

- **Broadcast type selection**: N/A. RELU is a purely element-wise unary operation with no broadcasting.

- **Sharding support and constraints**: Sharded inputs are routed to `UnaryShardedProgramFactory` (not analyzed here). The interleaved factory does not support sharded memory layout.

- **FP32 dest accumulation**: Controlled by `args.fp32_dest_acc_en` in the `ComputeConfig`. When enabled, the DST register operates in FP32 mode, providing higher precision for intermediate results. This is independent of `preserve_fp32_precision` (which controls unpack mode).

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h` |
| **LLK Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Wormhole) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h` (Wormhole) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h` (Blackhole) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (same file as LLK Dispatch) |

### Call Chain

1. The compute kernel calls `relu_tile(0)` (from `relu.h`), which wraps the call in the `MATH()` macro to ensure it runs on the math RISC-V processor.
2. Inside `relu_tile`, the macro `SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_min_, RC, APPROX, 0, 0)` expands to call `_llk_math_eltwise_unary_sfpu_params_<APPROX>()` with the functor `ckernel::sfpu::_relu_min_<sfpi::vFloat, APPROX, 8, uint32_t>`, dst_index=0, VectorMode::RC, and param0=0.
3. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) sets the DST write address, configures the ADDR_MOD base, stalls until the SFPU is ready, then iterates over 4 faces (for VectorMode::RC), calling the functor once per face with the threshold parameter (0).
4. The functor `_relu_min_<sfpi::vFloat, APPROX, 8, uint32_t>(0)` (in `ckernel_sfpu_relu.h`) converts the threshold to the appropriate type, loads it into LREG2, then calls `_relu_min_impl_` which executes 8 iterations of the core SFPU instruction sequence per face.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- processes all 4 faces of the 32x32 tile (Face 0 through Face 3). Each face is a 16x16 sub-tile.
- **Operation invocation**: The functor is called once per face (4 calls total). Each call executes 8 iterations internally (ITERATIONS=8), processing one row of SFPU elements per iteration. Each SFPU "row" covers all 32 lanes in parallel, so 8 iterations x 4 faces = 32 rows = full 32x32 tile.
- **DEST address progression**: On Wormhole, after each functor call the params dispatch advances the DEST read/write base by 16 rows using two `TTI_SETRWC(..., CR_D, 8, ..., SET_D)` calls (8+8=16 rows = one face). Within the functor, `sfpi::dst_reg++` increments the DEST address by 1 row after each iteration. The ADDR_MOD used by SFPLOAD/SFPSTORE has `dest.incr = 0`, so DEST auto-increment is handled entirely by the explicit `dst_reg++` and `SETRWC` calls, not by the address mode hardware. On Blackhole, the params dispatch uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice to advance by 16 rows per face.

### Annotated SFPU Kernel Source

The RELU SFPU kernel has **architecture-specific implementations** that differ significantly between Wormhole and Blackhole. Both are documented below.

#### Wormhole B0 Implementation (TTI-based, Style B)

On Wormhole, `_relu_min_impl_` uses raw `TTI_` SFPU instructions with a conditional SFPSWAP to compute `max(input, threshold)`. This approach avoids branching and condition codes entirely -- SFPSWAP performs the comparison and exchange atomically.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(const int iterations, [[maybe_unused]] VecType threshold, int sfpload_instr_mod)
{
    for (int d = 0; d < iterations; d++)
    {
        // Load input tensor to lreg0
        TTI_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, 0);
        // Copy value param from lreg2 to lreg1
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // Swap and store maximum in lreg1, minimum in lreg0 (sign + magnitude format)
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);
        // Store the result
        TTI_SFPSTORE(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_min_(T threshold)
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
    int sfpload_instr_mod = DEFAULT;
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            _sfpu_load_imm32_(p_sfpu::LREG2, scalar);
            sfpload_instr_mod = INT32_2S_COMP;
        }
        else
        {
            _sfpu_load_imm32_(p_sfpu::LREG2, threshold);
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_min_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold, sfpload_instr_mod);
}
```

**Note on the Wormhole float path for RELU**: When called as `_relu_min_<sfpi::vFloat, APPROX, 8, uint32_t>(0)`, the threshold `T` is `uint32_t` and `VectorType` is `sfpi::vFloat`. This enters the `else` branch within the `uint32_t` constexpr block, calling `_sfpu_load_imm32_(p_sfpu::LREG2, 0)` which loads 0x00000000 into LREG2 via two SFPLOADI instructions. The `sfpload_instr_mod` remains `DEFAULT` (=0), meaning SFPLOAD/SFPSTORE interpret data in the native DST format (FP32 in sign+magnitude). The SFPSWAP instruction with Imm12[0]=1 then performs FP32 comparison.

Since SFPSWAP does not set CC (confirmed by the ISA: "Sets CC Result? N"), no CC State Machine diagram is needed for this kernel.

#### Blackhole Implementation (SFPI-based, Style A)

On Blackhole, `_relu_min_impl_` uses the SFPI abstraction layer with `v_if`/`v_endif` to conditionally replace values below the threshold.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(const int iterations, VecType threshold) // APPROXIMATION_MODE=true (unused), ITERATIONS=8
{
    for (int d = 0; d < iterations; d++)
    {
        VecType a = sfpi::dst_reg[0]; // Load current DEST row into vector register
        v_if (a < threshold) // CC set for lanes where a < threshold
        {
            sfpi::dst_reg[0] = threshold; // Write threshold to those lanes only
        }
        v_endif; // Reset CC to ALL_ENABLED
        sfpi::dst_reg++; // Advance to next DEST row
    }
}

template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_min_(T threshold) // VectorType=sfpi::vFloat, T=uint32_t, threshold=0
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
            v_threshold = static_cast<int>(threshold); // INT32 path
        }
        else
        {
            v_threshold = Converter::as_float(threshold); // Reinterpret uint32_t bits as float
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_min_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
}
```

**Note on the Blackhole float path for RELU**: `Converter::as_float(0)` reinterprets the 32-bit pattern `0x00000000` as float, which is `0.0f`. The `v_if (a < 0.0f)` comparison uses SFPI's built-in comparison which compiles to SFPSETCC or similar CC-setting instruction under the hood, and `v_endif` compiles to SFPENCC. Because the CC flow is fully managed by the SFPI abstractions (`v_if`/`v_endif`), no manual CC tracking is needed.

### SFPU Instructions Used

#### Wormhole B0

| Instruction | Description |
|-------------|-------------|
| `SFPLOADI` | Loads a 16-bit immediate into the upper or lower half of an LREG. Used twice to load the full 32-bit threshold into LREG2 (via `_sfpu_load_imm32_`). InstrMod=10 writes lower 16 bits; InstrMod=8 writes upper 16 bits. |
| `SFPLOAD` | Loads one row of data from the DEST register file into LREG0. InstrMod=DEFAULT (0) means native format (sign+magnitude FP32). |
| `SFPMOV` | Copies the contents of one LREG to another. Here copies LREG2 (threshold) to LREG1 so that SFPSWAP can operate on LREG0 and LREG1 without destroying the threshold. |
| `SFPSWAP` | Conditionally exchanges values between two LREGs. With InstrMod=1 and Imm12[0]=1: performs FP32 comparison, places the smaller value in VD (LREG0) and the larger in VC (LREG1). Opcode 0x92, latency 2 cycles, IPC 0.5. Does NOT set CC. |
| `SFPSTORE` | Stores one row of data from an LREG back to the DEST register file. Stores LREG1 (the maximum) back to DEST. |

#### Blackhole

The Blackhole implementation uses SFPI abstractions that compile to lower-level instructions. The effective instruction sequence per iteration is:

| SFPI Construct | Compiles To | Description |
|----------------|-------------|-------------|
| `dst_reg[0]` (read) | `SFPLOAD` | Load current DEST row into a vector register |
| `v_if (a < threshold)` | `SFPSETCC` or comparison + CC set | Set CC for lanes where value is less than threshold |
| `dst_reg[0] = threshold` | `SFPSTORE` (CC-guarded) | Store threshold to DEST, only for lanes where CC is enabled |
| `v_endif` | `SFPENCC` | Reset CC to ALL_ENABLED |
| `dst_reg++` | `SETRWC` | Advance DEST address by 1 row |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Wormhole: holds the input value loaded from DEST via SFPLOAD. After SFPSWAP, contains min(input, threshold). Blackhole: implicit via `vFloat a`. |
| **LREG1** | Wormhole: receives a copy of the threshold from LREG2 via SFPMOV. After SFPSWAP, contains max(input, threshold) -- this is the RELU result that gets stored back. Blackhole: not explicitly used. |
| **LREG2** | Wormhole: holds the threshold value (0 for RELU), loaded once before the iteration loop via `_sfpu_load_imm32_`. Persists across all 8 iterations as a constant. Blackhole: not explicitly used (threshold is in a `vFloat` variable). |
| **DEST** | Both: the source and destination for tile data. SFPLOAD reads from the current DEST row; SFPSTORE writes back to it. Each iteration processes one row (32 elements across all SFPU lanes). |

### Address Mode Configuration

The SFPU address mode for RELU is configured during `relu_tile_init()` which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::relu_min, APPROX>()`, which in turn calls `eltwise_unary_sfpu_configure_addrmod<SfpuType::relu_min>()`.

**Both Wormhole and Blackhole** configure the same address mode:

```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

This means SFPLOAD/SFPSTORE do NOT auto-increment the DEST address. All DEST address progression is handled explicitly:
- **Within the functor**: `sfpi::dst_reg++` advances by 1 row after each iteration (8 times per face).
- **Between faces**: The params dispatch advances by 16 rows (one face width) using `SETRWC` instructions (Wormhole) or `inc_dst_addr<8>()` calls (Blackhole).

**Why ADDR_MOD_7?** On Wormhole, the params dispatch calls `set_addr_mod_base()` which sets the ADDR_MOD base register to 1, remapping indices 0-3 to hardware slots 4-7. When the kernel references `ADDR_MOD_3`, the hardware actually uses ADDR_MOD_7 (3 + 4 = 7). On Blackhole, the params dispatch calls `_llk_math_eltwise_unary_sfpu_start_` which does NOT call `set_addr_mod_base()`, so the kernel's `dst_reg++` and SFPI-generated instructions use ADDR_MOD_7 directly (SFPI internally references the correct slot). The net effect is the same on both architectures: `dest.incr = 0`.

RELU does not use any other ADDR_MOD slots (e.g., ADDR_MOD_6 is only configured for `topk_local_sort`, `typecast`, or `unary_max/min` operations).

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary operation program factory work for SFPU operations like RELU? What is the path selection between FPU and SFPU for unary eltwise operations?"
   **Reason**: Needed to understand the overall architecture of how RELU is dispatched through the unary program factory and which kernels are involved.
   **Key Findings**: RELU uses `eltwise_sfpu.cpp` as the compute kernel, with the SFPU_OP_CHAIN_0 define dynamically generated to expand to `relu_tile_init(); relu_tile(0);`. The factory sets up reader, compute, and writer kernels with circular buffers for coordination. The `SFPU_OP_RELU_FAMILY_INCLUDE` macro includes the `relu.h` header.

2. [SFPU] **Query**: "How does the RELU SFPU kernel work in tt-metal? Trace the call path from the compute kernel API (like relu_tile) through LLK dispatch to the ckernel SFPU implementation."
   **Reason**: Needed to understand the full call chain from `relu_tile()` API through LLK macro dispatch to the core SFPU kernel function.
   **Key Findings**: The call chain is `relu_tile()` -> `SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT` macro -> `_llk_math_eltwise_unary_sfpu_params_()` -> `ckernel::sfpu::_relu_min_()`. The macro instantiates the functor with `sfpi::vFloat` as VectorType, APPROX mode, 8 ITERATIONS, and `uint32_t` as the threshold type.

3. [SFPU] **Query**: "How is the RELU operation implemented in the LLK layer? What is the SFPU kernel function for relu, what instructions does it use?"
   **Reason**: Needed to understand the low-level SFPU instruction sequence and architectural differences between Wormhole and Blackhole.
   **Key Findings**: Wormhole uses raw TTI instructions (SFPLOAD, SFPMOV, SFPSWAP, SFPSTORE) for an efficient branchless max(x, threshold) implementation. Blackhole uses SFPI abstractions (v_if, dst_reg) for a conditional replacement approach. Both architectures use ADDR_MOD_7 with dest.incr=0.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to understand how `get_compute_kernel_path` selects the kernel file and how `get_block_defines` constructs the SFPU_OP_CHAIN_0 macro.
   **Key Information**: RELU falls through to the `default` case returning `"eltwise_sfpu.cpp"`. The init/func pair is `{"relu_tile_init();", "relu_tile(0);"}` for non-INT32, or `{"relu_tile_init();", "relu_tile_int32(0);"}` for INT32.

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h`
   **Reason**: Needed to understand the high-level SFPU API for relu_tile.
   **Key Information**: `relu_tile(idst)` is a thin wrapper that calls `SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_min_, RC, APPROX, idst, 0)`. RELU is implemented as `relu_min` with threshold = 0, which computes `max(x, 0)` using the SFPSWAP hardware instruction.

3. **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h`
   **Reason**: Needed to understand the low-level SFPU instruction sequence for RELU.
   **Key Information**: The `_relu_min_impl_` function uses four SFPU instructions per element row: SFPLOAD (load from DST), SFPMOV (copy threshold to LREG1), SFPSWAP (swap to get max in LREG1, min in LREG0), SFPSTORE (store max back to DST). For RELU (threshold=0), LREG2 is loaded with 0, so SFPSWAP effectively computes max(x, 0).

4. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
   **Reason**: Needed to verify which header is included when `SFPU_OP_RELU_FAMILY_INCLUDE` is defined.
   **Key Information**: When `SFPU_OP_RELU_FAMILY_INCLUDE` is defined to 1, it includes `"api/compute/eltwise_unary/relu.h"`.

### Confluence References

1. [SFPU] **Page**: Tensix SFPU Instruction Set Architecture (Page ID: 1170505767)
   **Section**: SFPSWAP instruction specification
   **Reason**: Needed authoritative details on the SFPSWAP instruction behavior, particularly the conditional exchange semantics and format select bit.
   **Key Information**: SFPSWAP (opcode 0x92) with InstrMod=1 conditionally exchanges operands on all rows. With Imm12[0]=1, it performs FP32 comparison. Default behavior places the smaller operand in RG[VD] and the larger in RG[VC]. The instruction does NOT set CC Result or CC Enable. Latency is 2 cycles with IPC of 0.5.
