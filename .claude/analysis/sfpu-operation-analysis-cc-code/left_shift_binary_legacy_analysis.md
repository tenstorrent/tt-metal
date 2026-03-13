# LEFT_SHIFT (Binary Legacy) Implementation Analysis

## Overview
The LEFT_SHIFT operation performs element-wise bitwise left shift on two integer tensors (`A << B`), where each element of tensor A is shifted left by the corresponding element of tensor B. It is implemented through the binary legacy SFPU program factory, which routes integer shift operations through the SFPU (vector unit) rather than the FPU (matrix unit).

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Path Selection: FPU vs SFPU

The binary device operation uses `BinaryDeviceOperation::select_program_factory()` (in `binary_device_operation.cpp`) to choose between program factories. When both input tensors have the same shape (no broadcasting needed), the function calls `utils::is_binary_sfpu_op(op, dtype1, dtype2)` to decide between `ElementWiseMultiCore` (FPU) and `ElementWiseMultiCoreSfpu` (SFPU).

For `BinaryOpType::LEFT_SHIFT`, `is_binary_sfpu_op` returns `true` when:
- Both inputs are `INT32`, or
- Both inputs are `UINT32`

(Note: `UINT16` is also listed in the switch case alongside `GCD`, `LCM`, `RIGHT_SHIFT`, and `LOGICAL_RIGHT_SHIFT`.)

If the shapes differ (broadcasting needed), the operation is routed to broadcast-specific factories (`BroadcastHeightMultiCore`, `BroadcastWidthMultiCore`, etc.) instead. The SFPU path is only selected for element-wise (same-shape) cases.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32 elements) |
| **Unit size** | 1 tile per inner loop iteration; `per_core_block_size` tiles per block |
| **Total units** | `total_tiles = physical_volume / TILE_HW` |
| **Loop structure** | Outer loop: `per_core_block_cnt` blocks; inner loop: `per_core_block_size` tiles per block |

In the non-sharded case, `per_core_block_size = 1` and `per_core_block_cnt = num_tiles_per_core`. In the sharded case, `per_core_block_size = find_max_block_size(num_tiles_per_shard)` and `per_core_block_cnt = num_tiles_per_shard / per_core_block_size`.

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|---------------|---------------|
| **Logical shape** | Arbitrary (must match B) | Arbitrary (must match A) |
| **Dimension convention** | NHWC/any | NHWC/any |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | INT32 or UINT32 | INT32 or UINT32 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as inputs |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | INT32 or UINT32 |

### Layout Transformations
No tilize/untilize or format conversions are performed within the operation. Both inputs and the output must already be in TILE_LAYOUT.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src0_buffer) | CB c_0 | reserve_back, push_back |
| 1 | Reader | DRAM/L1 (src1_buffer) | CB c_1 | reserve_back, push_back |
| 2 | Compute | CB c_0, CB c_1 | CB c_2 | wait_front, pop_front, reserve_back, push_back |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | wait_front, pop_front |

**Interleaved path**: The reader reads one tile at a time from each input buffer using `noc_async_read_tile` via TensorAccessor, pushing tiles into CB c_0 and CB c_1. The compute kernel waits for `per_core_block_size` tiles in both input CBs, copies both inputs into DST registers (A into even slots, B into odd slots), executes the SFPU shift operation, packs the result to CB c_2, and pops both input CBs. The writer reads one tile at a time from CB c_2 and writes it to the output buffer via `noc_async_write_page`.

**Sharded path**: When an input is sharded, the reader simply does `cb_reserve_back / cb_push_back` for the full shard (the CB is backed by the sharded buffer directly). When the output is sharded, the writer just does `cb_wait_front` and the data is already in place.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input A staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile (interleaved) or max_block_size tiles (sharded) | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_src1 | Input B staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile (interleaved) or max_block_size tiles (sharded) | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile (interleaved) or max_block_size tiles (sharded) | Double (interleaved) / Single (sharded) | Compute | Writer | Program |

**Note**: CB c_3 and c_4 (interim buffers for pre-scale operations) are NOT created for LEFT_SHIFT because the operation does not define `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0`.

## Pipeline Pattern Summary

- **Interleaved**: Double-buffered on all three CBs (capacity = 2 * block_size). This allows the reader and compute (or compute and writer) to overlap -- while compute processes one block, the reader can fill the next.
- **Sharded**: Single-buffered (capacity = num_tiles_per_shard = total shard). The sharded CB is globally allocated on the tensor's buffer, so there is no streaming overlap -- the entire shard is available at once.

## Index Calculations

The reader kernel uses `TensorAccessor` to map linear tile IDs to physical memory addresses. For the non-sharded interleaved path, tile IDs are sequential starting from `start_id`. Each core processes a contiguous range of `num_tiles_per_core` tiles.

For the block/width-sharded path, tile traversal follows a 2D pattern: outer loop over `block_height` rows, inner loop over `block_width` columns. The `start_id` for each core is computed as:
```
start_id = (core_index / num_shards_per_width) * (block_height * block_width * num_shards_per_width)
         + (core_index % num_shards_per_width) * block_width
```
Row advancement uses stride: `row_start_tile_id += num_cores_y * block_width`.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads. Each tile is read individually via `noc_async_read_tile` with a barrier after each tile pair (one from each input). This is a tile-at-a-time pattern.
- **Sharded**: No reads needed; the CB is backed by the sharded L1 buffer directly.

### Write Pattern
- **Interleaved**: Sequential tile writes via `noc_async_write_page`, one tile at a time, with `noc_async_writes_flushed()` after each tile and `noc_async_write_barrier()` at the end.
- **Sharded**: No writes needed; the output CB is backed by the output sharded buffer.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (interleaved) or 2D (sharded, matching shard grid) |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `num_cores_total` from worker grid |
| **Work per core** | `num_tiles_per_core_group_1` (main group) or `num_tiles_per_core_group_2` (remainder group) |
| **Load balancing** | Two-group split: group 1 gets `ceil(total_tiles/num_cores)` tiles, group 2 gets `floor(total_tiles/num_cores)` tiles. For sharded: all cores get equal `num_tiles_per_shard`. |

Work splitting uses `tt::tt_metal::split_work_to_cores()` for the interleaved case. The function divides total tiles across available cores, creating two groups to handle remainder. Non-working cores (beyond `num_cores`) receive zero-tile arguments.

An optimization is applied when the grid is a single rectangle starting at (0,0) (`zero_start_grid` flag), using faster core enumeration algorithms.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_or_width_sharded | uint32_t | 1 if block or width sharded, 0 otherwise |
| 1+ | TensorAccessorArgs(src0) | varies | Accessor args for input A (omitted if IN0_SHARDED) |
| N+ | TensorAccessorArgs(src1) | varies | Accessor args for input B (omitted if IN1_SHARDED) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs(dst) | varies | Accessor args for output buffer |

#### Compute Kernel
No compile-time arguments. All configuration is via preprocessor defines:
- `SHIFT_INIT` = `binary_shift_tile_init();`
- `BINARY_SFPU_OP` = `binary_left_shift_tile<DataFormat::Int32>(i*2, i*2+1, i*2);` (or `UInt32`/`UInt16` variant)

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
| 6 | num_cores_y | uint32_t | Number of shards per width dimension (0 if not sharded) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_size | uint32_t | Number of tiles per block |

#### Writer Kernel (Interleaved, non-block-sharded output)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Address of output buffer |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for output writes |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 src0, src1 | CB c_0, CB c_1 | Read tiles from both input buffers |
| Compute | Tensix compute (RISCV_2) | N/A | CB c_0, CB c_1 | CB c_2 | Copy tiles to DST, execute SFPU left shift, pack result |
| Writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 dst | Write output tiles to destination buffer |

### Reader Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` |
| Assigned cores | All worker cores (`all_device_cores`) |

**Key Logic**:
- When inputs are sharded (`IN0_SHARDED` / `IN1_SHARDED` defined), the reader simply calls `cb_reserve_back(cb_id, num_tiles)` then `cb_push_back(cb_id, num_tiles)` to make the globally-allocated sharded buffer available to compute. No actual data movement occurs.
- For interleaved inputs, a `TensorAccessor` is constructed from compile-time args for each non-sharded input.
- In the `block_or_width_sharded` path (sharded input with interleaved other), tile traversal uses 2D indexing: outer loop over `block_height`, inner loop over `block_width`, with row stride = `num_cores_y * block_width`.
- In the standard interleaved path, tiles are read sequentially from `start_id` to `start_id + num_tiles`.
- Each tile read uses `noc_async_read_tile` followed by `noc_async_read_barrier` before pushing to the CB.
- **Synchronization**: Produces to CB c_0 and CB c_1 via `cb_reserve_back` / `cb_push_back` (one tile at a time for interleaved).

### Compute Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` |
| Assigned cores | All worker cores (`all_device_cores`) |

**Key Logic**:
- Outer loop iterates `per_core_block_cnt` times (one iteration per block).
- No pre-scale path is active for LEFT_SHIFT (no `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defines), so `cb_inp0 = cb_in0 = c_0` and `cb_inp1 = cb_in1 = c_1`.
- Waits for `per_core_block_size` tiles in both input CBs and reserves space in the output CB.
- Acquires tile registers (`tile_regs_acquire` + `tile_regs_wait`).
- Copies input A tiles to even DST slots (`i*2`) using `copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0)` then `copy_tile(cb_inp0, i, i*2)`.
- Copies input B tiles to odd DST slots (`i*2+1`) using `copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1)` then `copy_tile(cb_inp1, i, i*2+1)`.
- Executes `SHIFT_INIT` -> `binary_shift_tile_init()` once per inner tile.
- Executes `BINARY_SFPU_OP` -> `binary_left_shift_tile<DataFormat::Int32>(i*2, i*2+1, i*2)` which reads operands from DST[i*2] and DST[i*2+1], writes the result back to DST[i*2].
- Packs result from DST[i*2] into output CB c_2 via `pack_tile(i*2, cb_out0)`.
- After inner loop: `tile_regs_commit` / `tile_regs_release`, then pops both input CBs and pushes the output CB.
- **Synchronization**: Consumes CB c_0 and c_1 via `cb_wait_front` / `cb_pop_front`. Produces to CB c_2 via `cb_reserve_back` / `cb_push_back`.

### Writer Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (standard) or `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` (block/width sharded input, interleaved output) |
| Assigned cores | All worker cores (`all_device_cores`) |

**Key Logic**:
- For sharded output (`OUT_SHARDED` defined): simply calls `cb_wait_front(cb_id_out, num_pages)` -- the data is already in the output buffer since the CB is globally allocated on it.
- For interleaved output: iterates from `start_id` to `start_id + num_pages`, writing one tile per iteration. Each iteration: `cb_wait_front` for 1 tile, gets L1 read address, calls `noc_async_write_page`, flushes, then `cb_pop_front`.
- Final `noc_async_write_barrier()` ensures all writes complete.
- **Synchronization**: Consumes CB c_2 via `cb_wait_front` / `cb_pop_front`.

## Implementation Notes

- **Program factory variants**: The `BinaryDeviceOperation` has multiple program factories: `ElementWiseMultiCore` (FPU), `ElementWiseMultiCoreSfpu` (SFPU), and several broadcast variants. LEFT_SHIFT always uses `ElementWiseMultiCoreSfpu` since `is_binary_sfpu_op` returns true for all its supported dtype combinations. The factory is selected in `select_program_factory()` only when input shapes match (no broadcasting).
- **Type-based operation variants**: Supports INT32 x INT32, UINT32 x UINT32, and UINT16 x UINT16. The data format string (`"Int32"`, `"UInt32"`, or `"UInt16"`) is selected in `get_defines_fp32()` and templated into the SFPU call as `binary_left_shift_tile<DataFormat::XXX>`.
- **UnpackToDestFP32 mode**: Enabled for all CBs (c_0, c_1, c_3, c_4) because `op_type != BinaryOpType::POWER` -- all non-POWER SFPU binary ops use `UnpackToDestMode::UnpackToDestFp32`.
- **Broadcast type selection**: N/A. LEFT_SHIFT through this program factory requires both inputs to have identical shapes. No broadcasting is supported in this path.
- **Sharding support and constraints**: Supports height-sharded, width-sharded, and block-sharded memory layouts. Either or both inputs and/or the output can be sharded. When sharded, CBs are globally allocated on the tensor buffer (no data movement). A special writer kernel variant handles the case of block/width-sharded input with interleaved output.
- **FP32 dest accumulation**: Enabled when the output data format is Float32, Int32, or UInt32 (`fp32_dest_acc_en`). Since LEFT_SHIFT operates on INT32 or UINT32, this is always enabled for LEFT_SHIFT.

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/binary_shift.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_shift.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/common/inc/sfpu/ckernel_sfpu_shift.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `binary_shift_tile_init()` (API header), which expands via `MATH(...)` to `llk_math_eltwise_binary_sfpu_shift_init<APPROX>()`. This calls `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()`, which initializes the SFPU config register, configures ADDR_MOD_7, and resets RWC counters.

2. The compute kernel then calls `binary_left_shift_tile<DataFormat::Int32>(i*2, i*2+1, i*2)` (API header), which expands via `MATH(...)` to `llk_math_eltwise_binary_sfpu_left_shift<APPROX, DataFormat::Int32>(i*2, i*2+1, i*2)`.

3. The LLK dispatch function resolves `INSTRUCTION_MODE = InstrModLoadStore::INT32` (since DataFormat is Int32, not UInt16), then calls `_llk_math_eltwise_binary_sfpu_params_<APPROX>(calculate_binary_left_shift<APPROX, 8, INT32, false>, i*2, i*2+1, i*2, VectorMode::RC)`.

4. The params dispatch sets the DST write address, stalls until the SFPU is ready, then loops over 4 faces (RC mode), calling `calculate_binary_left_shift(...)` for each face with `TTI_SETRWC` to advance the DEST read/write pointer by 16 rows between faces.

5. `calculate_binary_left_shift` is a thin wrapper in `ckernel_sfpu_shift.h` (metal llk_api copy) that directly calls `_calculate_binary_left_shift_<APPROX, 8, INT32, false>(...)` in the tt_llk core SFPU implementation.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default, not overridden by the caller). All 4 faces of the 32x32 tile are processed.
- **Operation invocation**: The params dispatch loops `for (int face = 0; face < 4; face++)`, calling the SFPU function once per face. Each call internally loops `ITERATIONS=8` times, processing 8 rows (1 row = 1 SFPU vector of 32 elements). So 4 faces x 8 rows = 32 rows total, but each "row" in DEST is actually a pair of 16-element half-rows from the two 16x16 sub-faces, yielding all 1024 elements of the 32x32 tile.
- **DEST address progression**: Between faces, the params dispatch issues `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice per face (advancing DEST pointer by 8+8=16 rows). Inside the SFPU function, `sfpi::dst_reg++` advances the DEST pointer by 1 row after each of the 8 iterations. The SFPLOAD/SFPSTORE instructions use `ADDR_MOD_7` (Blackhole) or `ADDR_MOD_3` (Wormhole), both configured with `.dest = {.incr = 0}`, meaning the ADDR_MOD itself does not auto-increment; the explicit `dst_reg++` handles row advancement within a face.

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions with complex condition code manipulation (SFPSETCC, SFPIADD with CC, SFPCOMPC, SFPENCC), so Style B applies.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_shift.h
// (Wormhole version is identical except ADDR_MOD_3 replaces ADDR_MOD_7)

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _calculate_binary_left_shift_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    constexpr int sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? INT32_2S_COMP : to_underlying(INSTRUCTION_MODE);
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        // size of each tile in Dest is 64 rows
        constexpr std::uint32_t dst_tile_size = 64;
        // load
        TT_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, dst_index_in0 * dst_tile_size);
        TT_SFPLOAD(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_7, dst_index_in1 * dst_tile_size);
        // if (shift_amount < 0 OR shift_amount >= 32) -> result should be 0
        TTI_SFPSETCC(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0xFE0, p_sfpu::LREG1, p_sfpu::LREG2, 1); // 0xFE0 = -32
        TTI_SFPCOMPC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPENCC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        // shift left
        TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        // store result
        TT_SFPSTORE(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, dst_index_out * dst_tile_size);
        sfpi::dst_reg++;
    }
}
```

```
_calculate_binary_left_shift_ — CC State Transitions
════════════════════════════════════════════════════════════════

  CC State: ALL_ENABLED                   <-- initial state
       |
       |  SFPLOAD LREG0, DEST[in0]       (no CC effect)
       |  SFPLOAD LREG1, DEST[in1]       (no CC effect)
       |    LREG0 = value_to_shift
       |    LREG1 = shift_amount
       |
       v
  +-------------------------------------------+
  | SFPSETCC  mod1=4 (LREG_GTE0)             |
  |   VC = LREG1 (shift_amount)              |
  |                                           |
  | CC <- (shift_amount >= 0)                 |
  +--------------------+----------------------+
                       |
                       v
  CC State: ENABLED where shift_amount >= 0
       |
       v
  +-------------------------------------------+
  | SFPIADD  imm=0xFE0(-32), mod1=1          |
  |   mod1=1: ARG_IMM + CC_LT0 (default)     |
  |   LREG2 = shift_amount + (-32)            |
  |         = shift_amount - 32               |
  |                                           |
  | CC <- CC_prev AND (result < 0)            |
  |    = (shift_amount >= 0)                  |
  |      AND (shift_amount - 32 < 0)          |
  |    = (0 <= shift_amount < 32)             |
  +--------------------+----------------------+
                       |
                       v
  CC State: ENABLED where 0 <= shift_amount < 32
       |
       v
  +-------------------------------------------+
  | SFPCOMPC                                  |
  |                                           |
  | CC <- NOT(CC_prev)                        |
  |    = NOT(0 <= shift_amount < 32)          |
  |    = (shift_amount < 0 OR                 |
  |       shift_amount >= 32)                 |
  +--------------------+----------------------+
                       |
                       v
  CC State: ENABLED where shift_amount < 0 OR shift_amount >= 32
       |
       |  SFPMOV LREG0 = LCONST_0 (=0)   (CC-guarded: only out-of-bounds lanes)
       |    Sets result to 0 for invalid shift amounts
       |
       v
  +-------------------------------------------+
  | SFPENCC                                   |
  |                                           |
  | CC <- ALL_ENABLED                         |
  +--------------------+----------------------+
                       |
                       v
  CC State: ALL_ENABLED
       |
       |  SFPSHFT LREG0 = LREG0 << LREG1  (all lanes)
       |    For out-of-bounds lanes: LREG0 was set to 0, so 0 << anything = 0
       |    For in-bounds lanes: LREG0 = value << shift_amount
       |
       |  SFPSTORE LREG0 -> DEST[out]      (no CC effect)
       |  dst_reg++                         (advance DEST pointer by 1 row)
       v
```

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `TT_SFPLOAD` | Loads a 32-element vector from a DEST register row into an SFPU local register (LREG). The `sfpload_instr_mod` parameter controls data interpretation: `INT32` treats data as 32-bit two's complement integers, `LO16` treats data as 16-bit unsigned integers. |
| `TTI_SFPSETCC` | Sets per-lane condition code flags based on a comparison of LREG[VC]. With mod1=4 (`LREG_GTE0`), CC is enabled for lanes where the register value is >= 0. |
| `TTI_SFPIADD` | Performs lanewise 32-bit integer addition. With mod1=1 (`ARG_IMM`), it adds the sign-extended 12-bit immediate to the source register. By default (mod1 bit2=0), it also updates CC with `CC_prev AND (result < 0)`. |
| `TTI_SFPCOMPC` | Complements the current condition code: `CC <- NOT(CC_prev)`. This provides the "else" branch for the preceding SFPSETCC/SFPIADD conditional chain. |
| `TTI_SFPMOV` | Moves data between SFPU local registers. CC-guarded: only executes on lanes where CC is enabled. Used here to zero out LREG0 for out-of-bounds shift amounts. |
| `TTI_SFPENCC` | Resets the condition code to ALL_ENABLED, ending the conditional execution block. All subsequent instructions execute on all lanes unconditionally. |
| `TTI_SFPSHFT` | Performs a lanewise bitwise shift. With mod1=0, the shift amount comes from LREG[VC] (LREG1). Positive values shift left, negative values shift right. `LREG[VD] = LREG[VB] << LREG[VC]`. |
| `TT_SFPSTORE` | Stores a 32-element vector from an SFPU local register (LREG) back to a DEST register row. Uses the same instruction mode as SFPLOAD for data format consistency. |

### SFPU Register Usage

| Register | Role | Notes |
|----------|------|-------|
| **LREG0** | Value to shift (input A), then result | Loaded from `DEST[dst_index_in0]`. Zeroed by SFPMOV for out-of-bounds lanes. After SFPSHFT, contains the final shifted result. Stored to `DEST[dst_index_out]`. |
| **LREG1** | Shift amount (input B) | Loaded from `DEST[dst_index_in1]`. Used as the shift amount operand for both the bounds-checking SFPIADD and the final SFPSHFT. Not modified. |
| **LREG2** | Scratch for bounds check | Receives the result of `shift_amount - 32` from SFPIADD. Used only to set CC flags; the value itself is discarded. |
| **LCONST_0** | Constant zero | Hardware constant register, always 0. Used as source for SFPMOV to zero out LREG0 on out-of-bounds lanes. |
| **DEST registers** | Input/output tile data | `DEST[dst_index_in0 * 64 + row]` holds the value to shift; `DEST[dst_index_in1 * 64 + row]` holds the shift amount; `DEST[dst_index_out * 64 + row]` receives the result. The `dst_tile_size = 64` accounts for the DEST layout where each tile occupies 64 rows. |
| **dst_reg** | DEST row pointer | SFPI software counter tracking the current DEST row. Incremented by 1 after each iteration. Reset between faces by `TTI_SETRWC` in the params dispatch. |

### Address Mode Configuration

Both Blackhole and Wormhole configure the same `addr_mod_t` structure for this operation, but assign it to different ADDR_MOD slots:

**Blackhole** (uses `ADDR_MOD_7`):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}.set(ADDR_MOD_7);
```
The SFPLOAD/SFPSTORE instructions reference `ADDR_MOD_7`. Since `.dest.incr = 0`, there is no automatic DEST address increment from the address mode. Instead, DEST row advancement is handled explicitly by `sfpi::dst_reg++` after each iteration (increments the SFPU's internal DEST pointer by 1 row).

**Wormhole B0** (uses `ADDR_MOD_3`):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}.set(ADDR_MOD_7);
```
The same zero-increment configuration is stored in `ADDR_MOD_7` during init, but the Wormhole `ckernel_sfpu_shift.h` references `ADDR_MOD_3` in its SFPLOAD/SFPSTORE instructions. This discrepancy means `ADDR_MOD_3` is not explicitly configured by the shift init code and relies on whatever default or prior configuration exists. The functional effect is the same -- zero auto-increment -- because the Wormhole SFPU implementation also uses explicit `sfpi::dst_reg++` for row advancement. Additionally, Wormhole's `_llk_math_eltwise_binary_sfpu_start_` calls `math::set_addr_mod_base()`, and `_llk_math_eltwise_binary_sfpu_done_` calls `math::clear_addr_mod_base()`, which Blackhole does not. This relates to Wormhole's address modifier base offset mechanism.

**Between faces**: The params dispatch issues `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice between faces (advancing DEST by 8+8=16 rows), which aligns the DEST pointer to the start of the next 16x16 face. The `sfpi::dst_reg` is not explicitly reset between faces; the SETRWC instructions directly manipulate the hardware DEST read/write counter.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary element-wise SFPU program factory work in ttnn? What kernels does it use and how does it handle different binary operations like left_shift?"
   **Reason**: Needed to understand the overall SFPU binary factory architecture and kernel selection before reading source code.
   **Key Findings**: Confirmed the three-kernel structure (reader, compute, writer), the use of preprocessor defines to parameterize the compute kernel, and that LEFT_SHIFT uses `binary_shift_tile_init()` and `binary_left_shift_tile<DataFormat::Int32>`. Also confirmed that `is_binary_sfpu_op` controls factory selection.

2. [SFPU] **Query**: "How is binary_left_shift_tile implemented? What is the call chain from the compute API through LLK to the ckernel SFPU implementation?"
   **Reason**: Needed to trace the full call chain from the API header through LLK dispatch to the core SFPU function, and locate all relevant file paths.
   **Key Findings**: Confirmed the call chain: `binary_left_shift_tile` -> `llk_math_eltwise_binary_sfpu_left_shift` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_binary_left_shift` -> `_calculate_binary_left_shift_`. Identified file locations for all abstraction layers.

3. [SFPU] **Query**: "How is the binary left shift SFPU operation implemented in LLK? What SFPU instructions does ckernel_sfpu_shift use?" (tenstorrent/tt-llk)
   **Reason**: Needed detailed understanding of the SFPU instruction sequence, register usage, and condition code flow in the core implementation.
   **Key Findings**: Confirmed the instruction sequence (SFPLOAD, SFPSETCC, SFPIADD, SFPCOMPC, SFPMOV, SFPENCC, SFPSHFT, SFPSTORE) and the bounds-checking logic. Identified that Blackhole uses ADDR_MOD_7 while Wormhole uses ADDR_MOD_3.

4. [SFPU] **Query**: "What does SFPSETCC do with different modifier values (especially mod=4)? What does SFPCOMPC do? What does SFPSHFT do?" (tenstorrent/tt-isa-documentation)
   **Reason**: Needed precise semantics of each SFPU instruction's condition code behavior and shift mechanics.
   **Key Findings**: Confirmed SFPSETCC mod1=4 corresponds to LREG_GTE0, SFPCOMPC complements CC, SFPSHFT performs lanewise bitwise shifts with positive values shifting left.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp`
   **Reason**: Needed to understand `select_program_factory` logic and `is_binary_sfpu_op` conditions.
   **Key Information**: LEFT_SHIFT routes to SFPU when both inputs are INT32 or UINT32; same-shape requirement for element-wise path.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp`
   **Reason**: Needed to understand which preprocessor defines are generated for LEFT_SHIFT.
   **Key Information**: Defines `SHIFT_INIT` = `binary_shift_tile_init();` and `BINARY_SFPU_OP` = `binary_left_shift_tile<DataFormat::XXX>(i*2, i*2+1, i*2);` with format selected based on input dtypes.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp`
   **Reason**: Needed to understand runtime argument setup and core distribution logic.
   **Key Information**: Two-group work splitting for interleaved; shard-based distribution for sharded; `zero_start_grid` optimization for rectangular grids starting at (0,0).

4. [SFPU] **Source**: `runtime/sfpi/include/sfpi_constants.h`
   **Reason**: Needed to map numeric SFPSETCC and SFPIADD modifier values to their symbolic constants.
   **Key Information**: `SFPSETCC_MOD1_LREG_GTE0 = 4`, `SFPIADD_MOD1_ARG_IMM = 1`, `SFPIADD_MOD1_CC_LT0 = 0`, `SFPIADD_MOD1_CC_NONE = 4`.
