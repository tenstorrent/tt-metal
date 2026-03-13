# MAXIMUM (binary_legacy) Implementation Analysis

## Overview

The MAXIMUM operation computes the element-wise maximum of two input tensors: `y = max(a, b)`. It is implemented through the binary legacy SFPU program factory, which dispatches tile-level max comparisons on the SFPU (vector unit). Three data-type variants exist: floating-point (`binary_max_tile`), INT32 (`binary_max_int32_tile`), and UINT32 (`binary_max_uint32_tile`).

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Path Selection: FPU vs SFPU

The `BinaryDeviceOperation::select_program_factory` function (in `binary_device_operation.cpp`) decides between the FPU path (`ElementWiseMultiCore`) and the SFPU path (`ElementWiseMultiCoreSfpu`). When both input tensors have the same height and width (no broadcasting required), the function calls `utils::is_binary_sfpu_op(op, dtype1, dtype2)`. For `BinaryOpType::MAXIMUM`, this function unconditionally returns `true` regardless of the input data types — meaning MAXIMUM **always** routes to the SFPU program factory when tensor shapes are element-wise compatible.

If broadcasting is needed (height_b==1 or width_b==1), the operation is dispatched to a broadcast-specific program factory instead of either element-wise factory.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32 elements) |
| **Unit size** | `block_size` tiles (1 tile when interleaved, up to `max_block_size` when sharded) |
| **Total units** | `physical_volume / TILE_HW` tiles across all cores |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_size` tiles per block |

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|---------------|---------------|
| **Logical shape** | [N, C, H, W] | [N, C, H, W] (same H, W as A) |
| **Dimension convention** | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32 | BFLOAT16, FLOAT32, INT32, UINT32 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as determined by output config |

### Layout Transformations

No tilize/untilize operations. All tensors must already be in TILE_LAYOUT. No format conversion is performed within the operation.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src0_buffer, src1_buffer) | CB c_0, CB c_1 | reserve_back, push_back (per tile) |
| 2 | Compute | CB c_0 (as cb_inp0), CB c_1 (as cb_inp1) | CB c_2 | wait_front, copy_tile to DST, SFPU max, pack_tile, pop_front, push_back |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | wait_front, pop_front (per tile) |

For MAXIMUM, there is no pre-scaling step (no `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defines are set), so CBs c_3 and c_4 are not created. The compute kernel reads directly from c_0 and c_1.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input A staging | 2 tiles (interleaved) or `num_tiles_per_shard` (sharded) | 1 tile (interleaved) or `max_block_size` (sharded) | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_src1 | Input B staging | 2 tiles (interleaved) or `num_tiles_per_shard` (sharded) | 1 tile (interleaved) or `max_block_size` (sharded) | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 tiles (interleaved) or `num_tiles_per_shard` (sharded) | 1 tile (interleaved) or `max_block_size` (sharded) | Double (interleaved) / Single (sharded) | Compute | Writer | Program |

Note: For MAXIMUM, CBs c_3 and c_4 (interim buffers for pre-scaling) are **not allocated** because no `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defines are emitted.

Capacity formula (interleaved): `2 * max_block_size * tile_size`. With `max_block_size = 1`, this yields 2 tiles.

## Pipeline Pattern Summary

- **Interleaved path**: CB c_0, c_1, and c_2 each have capacity = 2 tiles with block_size = 1 tile, yielding **double-buffered** operation. The reader can fill the next tile while compute processes the current one.
- **Sharded path**: CB capacity equals `num_tiles_per_shard` and the CB is globally allocated to the tensor buffer. The entire shard is available at once (single-buffered bulk transfer).

## Index Calculations

The reader kernel uses `TensorAccessor` for DRAM-interleaved reads. Tile IDs are assigned sequentially per core via `start_id` (the first tile ID for that core) and `num_tiles` (count). For block/width sharded layouts, tile access follows a 2D pattern: `row_start_tile_id = start_id`, incrementing by `num_cores_y * block_width` per row, and tile_id increments by 1 within each row of width `block_width`.

The writer uses the same `TensorAccessor` pattern with sequential tile IDs for interleaved output, or simply waits on the output CB for sharded output (no write needed since the CB is backed by the output buffer).

## Memory Access Patterns

### Read Pattern

- **Interleaved**: Sequential tile reads via `noc_async_read_tile`. One tile at a time with a barrier after each tile (both inputs read before barrier). Tiles are read in order from `start_id` to `start_id + num_tiles - 1`.
- **Block/width sharded**: 2D access pattern iterating height then width within the shard, with tile ID stride of `num_cores_y * block_width` between rows.
- **Sharded input**: No DRAM reads; the CB is directly backed by the L1 shard buffer. The reader just does `cb_reserve_back` / `cb_push_back` to make the data available.

### Write Pattern

- **Interleaved**: Sequential single-tile writes via `noc_async_write_page` with `noc_async_writes_flushed` after each tile.
- **Sharded output**: No writes needed; the output CB is backed by the output tensor's L1 shard buffer. The writer kernel is `writer_unary_interleaved_start_id.cpp` with `OUT_SHARDED` defined, which just calls `cb_wait_front`.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (row-major) for interleaved; matches shard grid for sharded |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `all_device_cores.num_cores()` or grid area |
| **Work per core** | `num_tiles_per_core_group_1` for group 1 cores, `num_tiles_per_core_group_2` for group 2 (remainder cores) |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_tiles / num_cores)` tiles, group 2 gets one fewer tile. For sharded, all cores get the same number of tiles. |

The runtime args function (`set_eltwise_binary_runtime_args`) supports a `zero_start_grid` optimization when the core range is a single rectangle starting at (0,0), enabling faster work-splitting via `split_work_to_cores` with a `CoreCoord` rather than a `CoreRangeSet`.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_or_width_sharded | uint32_t | 1 if block or width sharded, 0 otherwise |
| 1+ | TensorAccessorArgs(src0) | varies | Tensor accessor args for input A (omitted if IN0_SHARDED) |
| N+ | TensorAccessorArgs(src1) | varies | Tensor accessor args for input B (omitted if IN1_SHARDED) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs(dst) | varies | Tensor accessor args for output buffer |

#### Compute Kernel

No explicit compile-time args. Configuration is via `#define` macros:

| Define | Value (for MAXIMUM) | Description |
|--------|-------------------|-------------|
| `BINOP_INIT` | `binary_max_tile_init();` (float) / `binary_max_int32_tile_init();` (INT32) / `binary_max_uint32_tile_init();` (UINT32) | Initializes the SFPU max operation |
| `BINARY_SFPU_OP` | `binary_max_tile(i*2, i*2+1, i*2);` (float) / `binary_max_int32_tile(...)` / `binary_max_uint32_tile(...)` | Executes the max comparison on two DST tiles |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | Address of input tensor A buffer |
| 1 | src1_addr | uint32_t | Address of input tensor B buffer |
| 2 | num_tiles | uint32_t | Number of tiles this core processes |
| 3 | start_id | uint32_t | Starting tile ID for this core |
| 4 | block_height | uint32_t | Shard block height in tiles (0 if not sharded) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 if not sharded) |
| 6 | num_cores_y | uint32_t | Number of shards per width (used for stride calculation) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process (outer loop count) |
| 1 | per_core_block_size | uint32_t | Tiles per block (inner loop count) |

#### Writer Kernel (interleaved output)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Address of output tensor buffer |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for output writes |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 src0, src1 | CB c_0, CB c_1 | Read input tiles via TensorAccessor |
| Compute | TRISC (RISCV_2) | N/A | CB c_0 (DST even slots), CB c_1 (DST odd slots) | CB c_2 | copy_tile to DST, binary_max_tile SFPU op, pack_tile |
| Writer | BRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 dst | Write output tiles via TensorAccessor |

### Reader Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` |
| **Assigned cores** | All worker cores (`all_device_cores`) |

**Key Logic**:
- If `IN0_SHARDED` is defined, the reader skips DRAM reads for input A and simply makes the sharded L1 data available via `cb_reserve_back` / `cb_push_back` for the full `num_tiles`.
- If `IN1_SHARDED` is defined, same logic applies to input B.
- For interleaved (non-sharded) inputs in the default path (not `block_or_width_sharded`): iterates tile-by-tile from `start_id` to `start_id + num_tiles - 1`, reading one tile at a time from each input buffer via `noc_async_read_tile`, calling `noc_async_read_barrier()` after each pair, then pushing both tiles.
- For `block_or_width_sharded` path: uses a 2D loop over `block_height` x `block_width`, with row stride of `num_cores_y * block_width` to compute tile IDs.
- **Synchronization**: Produces into CB c_0 and CB c_1 via `cb_reserve_back(cb, 1)` -> write -> `cb_push_back(cb, 1)`.

### Compute Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` |
| **Assigned cores** | All worker cores (`all_device_cores`) |

**Key Logic**:
- Outer loop: `per_core_block_cnt` iterations (blocks).
- For MAXIMUM, no pre-scaling step is active (no `SFPU_OP_INIT_PRE_IN0_0` / `SFPU_OP_INIT_PRE_IN1_0`), so `cb_inp0 = cb_in0 = c_0` and `cb_inp1 = cb_in1 = c_1`.
- Waits on both input CBs for `per_core_block_size` tiles, reserves output CB.
- Acquires DST registers via `tile_regs_acquire()` and `tile_regs_wait()`.
- Copies input A tiles to even DST slots (`i*2`) and input B tiles to odd DST slots (`i*2+1`) using `copy_tile` with proper data type initialization via `copy_tile_to_dst_init_short_with_dt`.
- Executes `BINOP_INIT` (e.g., `binary_max_tile_init()`) then `BINARY_SFPU_OP` (e.g., `binary_max_tile(i*2, i*2+1, i*2)`) per tile. The result overwrites the even DST slot (idst0).
- Packs result from DST slot `i*2` to the output CB via `pack_tile(i*2, cb_out0)`.
- Commits and releases DST registers, pops input CBs, pushes output CB.
- **Synchronization**: `cb_wait_front(cb_inp0, block_size)` and `cb_wait_front(cb_inp1, block_size)` to consume from reader. `cb_reserve_back(cb_out0, block_size)` then `cb_push_back(cb_out0, block_size)` to produce for writer. `cb_pop_front` on both inputs after processing.

### Writer Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (interleaved) or `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` (block/width sharded to interleaved) |
| **Assigned cores** | All worker cores (`all_device_cores`) |

**Key Logic**:
- For sharded output (`OUT_SHARDED` defined): simply calls `cb_wait_front(cb_id_out, num_pages)` to ensure compute is complete. No DRAM write needed since the output CB is backed by the output tensor's L1 buffer.
- For interleaved output: iterates from `start_id` to `start_id + num_pages`, writing one tile at a time via `noc_async_write_page`, flushing after each write, and a final `noc_async_write_barrier`.
- **Synchronization**: Consumes from CB c_2 via `cb_wait_front(cb_out, 1)` -> read -> `cb_pop_front(cb_out, 1)` per tile.

## Implementation Notes

- **Program factory variants**: The `ElementWiseMultiCoreSfpu::create` factory is selected when `is_binary_sfpu_op` returns true and tensor shapes are element-wise compatible (same H, W). For broadcast cases, separate factories handle height/width broadcasting.
- **Type-based operation variants**: Three SFPU function variants based on data types: `binary_max_tile` for floating-point types (BFLOAT16, FLOAT32), `binary_max_int32_tile` for INT32, and `binary_max_uint32_tile` for UINT32. The variant is selected at define-generation time in `get_defines_fp32`.
- **UnpackToDestFP32 mode**: Enabled for all CBs (c_0, c_1, c_3, c_4) since the op type is not POWER. This forces FP32 unpacking to DEST regardless of the input data format.
- **Broadcast type selection**: No broadcasting in this factory. MAXIMUM with different-shaped inputs is routed to broadcast-specific program factories.
- **Sharding support and constraints**: Supports HEIGHT_SHARDED, WIDTH_SHARDED, and BLOCK_SHARDED. Any of the three tensors (input A, input B, output) can be independently sharded or interleaved. The writer kernel selection changes based on whether block/width sharded output writes to interleaved DRAM.
- **FP32 dest accumulation**: Enabled when the output data format is Float32, Int32, or UInt32 (`fp32_dest_acc_en` flag in `ComputeConfig`).

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to. The MAXIMUM operation has two distinct SFPU kernel variants: a **floating-point** variant using `SFPSWAP` for IEEE 754 min/max, and an **integer** variant that uses `SFPSWAP` with sign-correction via `SFPSETCC`/`SFPENCC` condition code manipulation.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/binary_max_min.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_max_min.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel invokes `binary_max_tile(i*2, i*2+1, i*2)` (defined in `binary_max_min.h`), which is guarded by `MATH((...))` to run only on the math RISC-V.
2. `binary_max_tile` calls `llk_math_eltwise_binary_sfpu_binary_max<APPROX>(idst0, idst1, odst, vector_mode)` in `llk_math_eltwise_binary_sfpu_max_min.h`.
3. That function calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_binary_max_min<true>, dst_index0, dst_index1, odst, vector_mode)` in `llk_math_eltwise_binary_sfpu_params.h`.
4. The params dispatch function iterates over tile faces (4 faces for `VectorMode::RC`), calling `calculate_binary_max_min<true>(dst_index_in0, dst_index_in1, dst_index_out)` per face, then advancing the DEST read/write counter by 16 rows (two `SETRWC +8` calls) between faces.
5. For initialization, `binary_max_tile_init()` calls `llk_math_eltwise_binary_sfpu_binary_max_init<APPROX>()`, which calls `llk_math_eltwise_binary_sfpu_init<SfpuType::max, APPROXIMATE>(sfpu::binary_max_min_init<true>)`. This configures ADDR_MOD registers and then runs `binary_max_min_init<true>()` to set up SFPLOADMACRO instruction templates and macros.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces (16x16 sub-tiles) of the 32x32 tile are processed. For `VectorMode::R`, only faces 0 and 1 are processed (first row of faces). For `VectorMode::C`, only faces 0 and 2 are processed (first column of faces).
- **Operation invocation**: In `VectorMode::RC`, the core SFPU function is called 4 times in a loop (`for face = 0..3`). Each invocation processes one 16x16 face (8 SFPU iterations of 2 rows each = 16 rows). Between face invocations, two `TTI_SETRWC(CLR_NONE, CR_D, 8, ...)` calls advance the DEST pointer by 16 rows total.
- **DEST address progression**: The DEST read/write counter starts at 0 (reset in `_llk_math_eltwise_binary_sfpu_start_`). After each face, two `SETRWC +8` instructions advance it by 16 rows. The core function itself handles intra-face row addressing via absolute offsets encoded in `SFPLOADMACRO`/`SFPLOAD` instruction immediates, not via ADDR_MOD auto-increment.

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions with `SFPLOADMACRO` for pipelined scheduling. The floating-point variant has no condition code manipulation. The integer variant uses `SFPSETCC`/`SFPENCC` for sign handling. Both are documented below.

#### Floating-Point Variant: `calculate_binary_max_min`

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h

template <bool IS_MAX_OP = true, int ITERATIONS = 8>
inline void calculate_binary_max_min(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // IS_MAX_OP=true, ITERATIONS=8
    uint offset0 = (dst_index_in0 * 32) << 1;  // byte offset to input A in DEST
    uint offset1 = (dst_index_in1 * 32) << 1;  // byte offset to input B in DEST
    uint offset2 = (dst_index_out * 32) << 1;   // byte offset to output in DEST

    // Implementation notes, see the original file for more details

    constexpr int b = p_sfpu::LREG2;   // LREG2 holds input B row
    constexpr int c = p_sfpu::LREG3;   // LREG3 used for store-back (output)

#pragma GCC unroll 8
    for (int i = 0; i < ITERATIONS; ++i) {
        int a = i & 1;  // alternate LREG0/LREG1 for double-buffering loads of input A
        // Load input A row into LREG[a] via macro 0; ADDR_MOD_3(WH)/ADDR_MOD_7(BH) = dest.incr=0
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_3, offset0 | (a >> 2));  // WH: ADDR_MOD_3; BH: ADDR_MOD_7
        // Load input B row into LREG2; same addr_mod (no DEST auto-increment)
        TT_SFPLOAD(b, InstrModLoadStore::DEFAULT, ADDR_MOD_3, offset1);  // WH: ADDR_MOD_3; BH: ADDR_MOD_7
        // Store output via macro 1; ADDR_MOD_2(WH)/ADDR_MOD_6(BH) = dest.incr=2 (advance 2 rows per store)
        TT_SFPLOADMACRO((1 << 2) | (c & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_2, offset2 | (c >> 2));  // WH: ADDR_MOD_2; BH: ADDR_MOD_6
    }

    TTI_SFPNOP;  // pipeline drain: 3 NOPs to flush SFPLOADMACRO pipeline
    TTI_SFPNOP;
    TTI_SFPNOP;
}
```

**Note on architecture differences**: The Wormhole variant uses `ADDR_MOD_3` and `ADDR_MOD_2`, while Blackhole uses `ADDR_MOD_7` and `ADDR_MOD_6`. Both are configured with the same field values (dest.incr=0 for loads, dest.incr=2 for stores). The difference in ADDR_MOD slot indices is because Blackhole has more ADDR_MOD slots available (up to 7), while Wormhole uses lower-numbered slots. The init function (`eltwise_binary_sfpu_configure_addrmod`) configures ADDR_MOD_7 and ADDR_MOD_6 on both architectures; the Wormhole ckernel appears to rely on a mapping or separate configuration for ADDR_MOD_3/ADDR_MOD_2 with equivalent values.

#### Integer Variant: `calculate_binary_max_min_int32`

The integer variant is more complex because `SFPSWAP` compares IEEE 754 floating-point values, but integer data in DEST is reinterpreted as float. For signed integers, negative values have their sign bit set, which would cause incorrect float comparisons. The kernel corrects this by using `SFPSETCC` to test the sign of the XOR of the two inputs (loaded into alternating registers), and `SFPENCC` to enable all lanes after the conditional swap. For unsigned integers, `SFPSETCC_MOD1_LREG_GTE0` is used instead of `SFPSETCC_MOD1_LREG_LT0`.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false, int ITERATIONS = 8>
inline void calculate_binary_max_min_int32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // IS_MAX_OP=true, IS_UNSIGNED=false (signed int32) or IS_UNSIGNED=true (uint32)
    uint offset0 = (dst_index_in0 * 32) << 1;
    uint offset1 = (dst_index_in1 * 32) << 1;
    uint offset2 = (dst_index_out * 32) << 1;

    // Implementation notes, see the original file for more details

    constexpr int a0 = p_sfpu::LREG0;
    constexpr int b0 = p_sfpu::LREG1;
    constexpr int a1 = p_sfpu::LREG2;
    constexpr int b1 = p_sfpu::LREG3;
    constexpr int c = p_sfpu::LREG7;

    lltt::record<lltt::NoExec>(0, 10);  // BH: load_replay_buf(0, 10, [...]{...})

    // first iteration, with a0, b0, c
    TT_SFPLOADMACRO((0 << 2) | (a0 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset0 | (a0 >> 2));
    TT_SFPLOADMACRO((2 << 2) | (b0 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset1 | (b0 >> 2));
    TTI_SFPSETCC(0, a1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPENCC(0, 0, 0, 0);
    TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_2, offset2 | (c >> 2));

    // second iteration, with a1, b1, c
    TT_SFPLOADMACRO((1 << 2) | (a1 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset0 | (a1 >> 2));
    TT_SFPLOADMACRO((2 << 2) | (b1 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset1 | (b1 >> 2));
    TTI_SFPSETCC(0, a0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPENCC(0, 0, 0, 0);
    TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_2, offset2 | (c >> 2));

#pragma GCC unroll 4
    for (int i = 0; i < ITERATIONS / 2; ++i) {
        lltt::replay(0, 10);  // replay both iterations (10 instructions)
    }

    if constexpr (ITERATIONS & 1) {
        lltt::replay(0, 5);       // replay first iteration only
        TTI_SFPNOP;
        TTI_SFPNOP;
        lltt::replay(5 + 2, 2);   // replay SFPSETCC + SFPENCC from second iteration
    } else {
        TTI_SFPNOP;
        TTI_SFPNOP;
        lltt::replay(2, 2);       // replay SFPSETCC + SFPENCC to finalize CC state
    }

    TTI_SFPNOP;
}
```

**CC State Machine for `calculate_binary_max_min_int32` (signed, IS_MAX_OP=true, IS_UNSIGNED=false):**

Each pair of iterations (processing 2 rows) follows this CC pattern. The SFPLOADMACRO schedules `SFPSWAP` via instruction templates, and `SFPSETCC`/`SFPENCC` bracket the swap to handle sign correction.

```
calculate_binary_max_min_int32 (per iteration pair) — CC State Transitions
════════════════════════════════════════════════════════════════

  CC State: ALL_ENABLED                   <-- initial state

       |
       |  SFPLOADMACRO: load a0 from DEST[offset0]     (no CC effect)
       |  SFPLOADMACRO: load b0 from DEST[offset1]     (no CC effect)
       |
       v
  +------------------------------------------+
  | SFPSETCC  a1, SFPSETCC_MOD1_LREG_LT0    |
  |                                          |
  | CC <- ENABLED where LREG[a1] < 0        |
  | (a1 holds XOR of previous inputs;       |
  |  LT0 = sign bits differ = mixed-sign)   |
  +------------------+-----------------------+
                     |
                     v
  CC State: ENABLED where a1 < 0 (mixed-sign lanes)
       |
       v
  +------------------------------------------+
  | SFPENCC                                  |
  |                                          |
  | CC <- ALL_ENABLED                        |
  +------------------+-----------------------+
                     |
                     v
  CC State: ALL_ENABLED
       |
       |  SFPLOADMACRO: triggers SFPSWAP([a0], b0)     (CC-guarded by macro pipeline)
       |    -- SFPSWAP with mod1=9: VD=max, VC=min     (all lanes, after SFPENCC)
       |  SFPLOADMACRO: store c to DEST[offset2]       (no CC effect)
       |
       |  --- second half of pair ---
       |
       |  SFPLOADMACRO: load a1 from DEST[offset0]     (no CC effect)
       |  SFPLOADMACRO: load b1 from DEST[offset1]     (no CC effect)
       |
       v
  +------------------------------------------+
  | SFPSETCC  a0, SFPSETCC_MOD1_LREG_LT0    |
  |                                          |
  | CC <- ENABLED where LREG[a0] < 0        |
  +------------------+-----------------------+
                     |
                     v
  CC State: ENABLED where a0 < 0
       |
       v
  +------------------------------------------+
  | SFPENCC                                  |
  |                                          |
  | CC <- ALL_ENABLED                        |
  +------------------+-----------------------+
                     |
                     v
  CC State: ALL_ENABLED
       |
       |  SFPLOADMACRO: triggers SFPSWAP([a1], b1)     (all lanes)
       |  SFPLOADMACRO: store c to DEST[offset2]       (no CC effect)
       v
```

**Important note on the CC flow**: The `SFPSETCC` + `SFPENCC` pair in the integer variant works with the SFPLOADMACRO pipeline scheduling. The instruction templates (set in `binary_max_min_int32_init`) include `SFPSETCC` as template[2], which the macro scheduler uses to test the sign of the previously-loaded value. The `SFPENCC` immediately after re-enables all lanes before the `SFPSWAP` executes. The net effect is that for signed integers, when the two operands have different signs (one positive, one negative), the CC-guarded `SFPSWAP` correctly handles the comparison by treating the sign-differing case specially. For unsigned integers (`IS_UNSIGNED=true`), `SFPSETCC_MOD1_LREG_GTE0` is used instead, and the swap template uses `IS_MAX_OP ^ IS_UNSIGNED` to invert the swap direction to account for unsigned interpretation.

#### Init Function: `binary_max_min_init` (Floating-Point)

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h

template <bool IS_MAX_OP = true>
inline void binary_max_min_init() {
    // IS_MAX_OP=true for MAXIMUM
    constexpr int b = p_sfpu::LREG2;

    // InstructionTemplate[0]: SFPSWAP with mod1=9 (VD=max, VC=min) for max; SFPSWAP_MOD1_VEC_MIN_MAX for min
    TTI_SFPSWAP(0, b, 12, IS_MAX_OP ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // InstructionTemplate[1]: SFPSHFT2 for round/store pipeline stage
    TTI_SFPSHFT2(0, 0, 13, 6);  // SFPSHFT2_MOD1_SHFT_IMM

    // Macro 0: schedules load->swap->round pipeline
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (1 << 3) | 4;   // enable=1, template_idx=1, delay=4
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (3 << 3) | 5;    // enable=1, use_load_mod=1, lreg=3, delay=5
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);  // install as macro 0
    }

    // Macro 1: schedules store pipeline
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (2 << 3) | 3;    // use_load_mod=1, lreg=2, delay=3

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);  // install as macro 1
    }

    // Misc config: StoreMod0=DEFAULT, UsesLoadMod0ForStore={1,1}, UnitDelayKind={1,1}
    TTI_SFPCONFIG(0x330, 8, 1);
}
```

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `SFPLOADMACRO` | Macro-scheduled load from DEST into an LREG. Triggers pipelined execution of instruction templates (SFPSWAP, SFPSHFT2) at specified delays. Achieves 3-cycle-per-row throughput for float, 5-cycle-per-row for int32. |
| `SFPLOAD` | Direct load from DEST into a specified LREG. Used to load input B (non-macro-scheduled). |
| `SFPSWAP` | Compares two LREGs and swaps them so VD=max and VC=min (when mod1=9). This is the core comparison instruction. For min, `SFPSWAP_MOD1_VEC_MIN_MAX` reverses the assignment. |
| `SFPSHFT2` | Shift instruction used in the round/store pipeline stage of SFPLOADMACRO. With `SFPSHFT2_MOD1_SHFT_IMM` (mod1=6), performs an immediate shift for data format conversion during store-back. |
| `SFPLOADI` | Loads an immediate value into LREG0's lower or upper 16 bits. Used during init to configure SFPLOADMACRO macro bit fields. |
| `SFPCONFIG` | Configures SFPU internal state: installs instruction templates, macros, and miscellaneous settings (store mode, delay kinds). |
| `SFPSETCC` | Sets the condition code based on an LREG value. `SFPSETCC_MOD1_LREG_LT0` enables lanes where the LREG is negative (sign bit set). Used in the int32 variant for sign correction. |
| `SFPENCC` | Resets condition code to ALL_ENABLED. Used after `SFPSETCC` to re-enable all SFPU lanes before the `SFPSWAP` executes. |
| `SFPNOP` | No-operation. Used to drain the SFPLOADMACRO pipeline (3 NOPs for float, 1 NOP for int32 tail). |
| `SETRWC` | Sets the read/write counter for DEST addressing. Used between face iterations to advance the DEST pointer by 16 rows (two increments of 8). |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Float: alternating input A row (even iterations). Int32: input A row (first half of pair, `a0`). |
| **LREG1** | Float: alternating input A row (odd iterations). Int32: input B row (first half of pair, `b0`). |
| **LREG2** | Float: input B row (`b`). Int32: input A row (second half of pair, `a1`). |
| **LREG3** | Float: output staging register (`c`). Int32: input B row (second half of pair, `b1`). |
| **LREG7** | Int32 only: output staging register (`c`). Not used in float variant. |
| **DEST** | Tile data in the destination register file. Input A at `dst_index_in0 * 32` rows, input B at `dst_index_in1 * 32` rows, output written to `dst_index_out * 32` rows. Each row is 16 elements wide (one face width). |

**Double-buffering strategy**: The floating-point variant alternates LREG0 and LREG1 for input A loads (`a = i & 1`), allowing the SFPLOADMACRO pipeline to overlap the next load with the current SFPSWAP computation. The integer variant uses a similar strategy with two register pairs: `(a0=LREG0, b0=LREG1)` and `(a1=LREG2, b1=LREG3)`, processing two rows per replay iteration.

### Address Mode Configuration

Two ADDR_MOD slots are configured for this operation:

**ADDR_MOD_7 (Blackhole) / ADDR_MOD_3 (Wormhole)** — used for SFPLOAD and SFPLOADMACRO load instructions:
- `srca.incr = 0`
- `srcb.incr = 0`
- `dest.incr = 0`
- Purpose: No auto-increment after load. Row addressing is handled by the absolute offset encoded in the instruction immediate field, not by ADDR_MOD auto-increment.

**ADDR_MOD_6 (Blackhole) / ADDR_MOD_2 (Wormhole)** — used for SFPLOADMACRO store instructions:
- `srca.incr = 0`
- `srcb.incr = 0`
- `dest.incr = 2`
- Purpose: Auto-increments the DEST write pointer by 2 rows after each store. Since each SFPLOADMACRO store writes one row, the `dest.incr=2` advances past the current row and the next (interleaved face layout), ensuring consecutive output rows are written correctly.

Both Wormhole and Blackhole use identical field values; only the ADDR_MOD slot indices differ. The init function `eltwise_binary_sfpu_configure_addrmod<SfpuType::max>()` (in `llk_math_eltwise_binary_sfpu.h`) configures both slots. The slot index difference is because the ckernel source files are architecture-specific and reference their respective slot numbers.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary eltwise SFPU program factory work in ttnn? What kernels does it use and how does it handle broadcast modes?"
   **Reason**: Initial reconnaissance to understand the factory structure, kernel assignments, and broadcast handling.
   **Key Findings**: Confirmed three kernels (reader, compute, writer), that broadcast is handled by separate factories, and that `ElementWiseMultiCoreSfpu` is selected via `is_binary_sfpu_op`.

2. [SFPU] **Query**: "Where is the binary_max_tile compute API defined? What is the call chain from binary_max_tile through LLK to the core SFPU implementation for binary max/min operations?"
   **Reason**: Needed to trace the full SFPU call chain from the compute API through LLK dispatch to the core ckernel implementation.
   **Key Findings**: Confirmed the call chain: `binary_max_tile` -> `llk_math_eltwise_binary_sfpu_binary_max` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_binary_max_min`. Identified the core implementation file `ckernel_sfpu_binary_max_min.h` and key instructions (`SFPLOADMACRO`, `SFPSWAP`, `SFPLOAD`, `SFPSTORE`). Confirmed architecture-specific variants for Wormhole and Blackhole.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp` (lines 21-65)
   **Reason**: Needed to understand SFPU path selection logic.
   **Key Information**: `is_binary_sfpu_op` returns `true` unconditionally for MAXIMUM/MINIMUM (line 57-61).

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 184-552)
   **Reason**: Needed to understand how MAXIMUM maps to SFPU defines.
   **Key Information**: MAXIMUM sets `BINOP_INIT` and the `BINARY_SFPU_OP` define with `binary_max_tile(i*2, i*2+1, i*2)` (or int32/uint32 variants). No pre-scaling defines are emitted.

3. **Source**: `tt_metal/hw/inc/api/compute/binary_max_min.h`
   **Reason**: Needed to understand the SFPU compute API for max/min operations.
   **Key Information**: Three variants exist (float, int32, uint32). Each takes (idst0, idst1, odst) and computes element-wise max, writing result to odst in DST register buffer.

### Confluence References

No Confluence pages were consulted for this analysis. The SFPU instructions used (SFPLOADMACRO, SFPSWAP, SFPSETCC, SFPENCC) were sufficiently documented through DeepWiki and source code inspection.

### Glean References

No Glean searches were performed for this analysis.
