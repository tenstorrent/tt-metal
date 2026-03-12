# Max Pool With Indices (MPWI) Implementation Analysis

## Overview

Max Pool With Indices (MPWI) computes 2D max pooling over a sliding window while simultaneously tracking the index of the maximum value within each pooling window. It produces two outputs: the max-pooled values and a tensor of indices indicating which input element was the maximum for each output position. This is the `return_indices=true` path of the generic Pool2D operation.

**Program factory path**: `ttnn/cpp/ttnn/operations/pool/generic/device/pool_multi_core_program_factory.cpp`

The operation uses SFPU-based `max_reduce_with_indices` for the core reduction logic and SFPU `add_int_tile` for integer index arithmetic, making it an SFPU-centric operation.

## Path Selection: FPU vs SFPU

The program factory `pool2d_multi_core_sharded_with_halo_v2_impl_new` serves both standard Pool2D (max/avg pooling) and Max Pool With Indices. The path is selected by the `return_indices` boolean parameter:

- **When `return_indices == false`**: The factory selects `reader_pool_2d.cpp` (reader) and `compute_pool_2d.cpp` (compute). The compute kernel uses FPU-based tilize/reduce/untilize operations (`reduce_h` with `PoolType::MAX` or `PoolType::SUM`). This is the standard pool path.
- **When `return_indices == true`**: The factory selects `reader_mpwi.cpp` (reader) and `compute_mpwi.cpp` (compute). The compute kernel uses SFPU-based `max_reduce_with_indices` for the reduction and `add_int_tile` for integer index manipulation. The SFPU path is exclusively used for Max Pool With Indices and requires `Pool2DType::MAX_POOL2D` (enforced by `static_assert(REDUCE_OP == PoolType::MAX)`). Additionally, when `return_indices` is true, `split_reader` is always true, `MAX_TILES_PER_REDUCTION` is forced to 1, and the `in_cb_sz` uses `TILE_HEIGHT` as the height multiplier (full tile) rather than `num_tilized_rows`.

All remaining sections analyze the SFPU (MPWI) path exclusively.

## Work Unit Definition

One work unit is the complete max-pooling reduction of a single output spatial position (one "stick") across all channel blocks. For each output position, the reader assembles all `kernel_h * kernel_w` input sticks from the sliding window, the compute kernel reduces them via SFPU max-with-indices, and the writer (reader1) copies the packed result to the output shard. When channels exceed one tile width (`in_nblocks_c > 1`), the work unit iterates over channel blocks, processing one tile-width of channels at a time.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|---|---|
| Dimension Convention | NHWC (flattened to [1, 1, N*H*W, C]) |
| Tensor Layout | ROW_MAJOR |
| Memory Layout | Sharded (HEIGHT_SHARDED, BLOCK_SHARDED, or WIDTH_SHARDED) |
| Buffer Type | L1 |
| Data Type | BF16 (block float formats converted to BF16 for internal processing) |
| Shard Shape | [max_in_nhw_per_core, C_per_shard] |

The input arrives pre-processed by the halo operation, which inserts padding and halo regions into the shard so that each core's shard contains all input data needed for its assigned output positions.

### Output Tensor (Values)

| Property | Value |
|---|---|
| Dimension Convention | NHWC (flattened to [1, 1, N*out_H*out_W, C]) |
| Tensor Layout | ROW_MAJOR (default) or TILE |
| Memory Layout | Sharded (same scheme as input) |
| Buffer Type | L1 |
| Data Type | BF16 |
| Shard Shape | [max_out_nhw_per_core, C_per_shard] |

### Output Tensor (Indices)

| Property | Value |
|---|---|
| Dimension Convention | NHWC (same as values output) |
| Tensor Layout | ROW_MAJOR |
| Memory Layout | Sharded (same scheme as input) |
| Buffer Type | L1 |
| Data Type | UINT16 (if in_h * in_w <= 65535) or UINT32 (otherwise) |
| Shard Shape | [max_out_nhw_per_core, C_per_shard] |

### Layout Transformations

- Input data is in ROW_MAJOR format in L1 shards. The reader assembles sticks into the input CB in a raw stick layout.
- The compute kernel uses `copy_tile_to_dst` operations to load data into DST registers where the SFPU operates.
- Output is packed from DST back to CBs using `pack_tile<true>` and then copied to the output shard by reader1.
- No explicit tilize/untilize is performed in the MPWI path; data moves as raw sticks and individual face-width segments.

## Data Flow Pattern

1. **Initialization (Reader0)**:
   - Fills `clear_value_cb` with `-inf` (BF16) for max initialization.
   - Pre-clears the input CB tiles using `clear_out_tiles`.
   - Fills `in_scalar_cb_id_0` with the scalar value (1.0 for max pool).
   - Computes initial index values based on `start_row`/`start_col` and fills `in_idx_cb_id`.
   - Fills increment CBs (`right_inc`, `down_left_wrap_inc`, `up_left_wrap_inc`, and optionally `intra_kernel_right_inc`, `intra_kernel_down_left_wrap_inc` for large kernels).

2. **Per-output-stick loop** (iterates over all output positions assigned to this core):

   a. **Reader0**: Reads `kernel_h * kernel_w` input sticks from the L1 input shard into `in_cb_id`. Sticks are read using `noc_async_read_one_packet` from the local L1 shard (no DRAM access). For large kernels, data is chunked into groups of `sticks_per_chunk`.

   b. **Compute**: For each channel block `c_i`:
      - Acquires tile registers.
      - Loads current index tile from `in_idx_cb_id` (first iteration) or `compute_tmp_idx_cb_id` (subsequent).
      - For large kernels, clears accumulation DST registers.
      - Iterates over chunks: waits for `in_cb_id`, copies input data to DST, computes index increments, calls `max_reduce_with_indices` SFPU operation.
      - Packs results: `data_dst_idx` to `pack_tmp_cb_id`, `index_dst_idx` to `pack_idx_tmp_cb_id`, updated indices to `compute_tmp_idx_cb_id`.
      - Releases tile registers.

   c. **Reader1** (writer role): After compute finishes each channel block:
      - Waits on `pack_tmp_cb_id` and `pack_idx_tmp_cb_id`.
      - Copies max values from `pack_tmp_cb_id` to `out_cb_id` (output shard).
      - Copies max indices from `pack_idx_tmp_cb_id` to `out_idx_cb_id` (index output shard).
      - Uses `noc_async_read_one_packet` for the L1-to-L1 copy.

3. **Cleanup**: Compute pops all persistent CBs (scalar, increment CBs) to prevent stale data in cached programs.

## Circular Buffer Configuration

All CBs below are allocated only when `return_indices == true` (MPWI path). CB IDs are assigned sequentially starting from `c_0`.

| CB ID (sequential) | Purpose | Page Size | Num Pages | Data Format | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|
| 0 (`in_scalar_cb_id_0`) | Scalar coefficient (1.0 for max pool) | tile_size(BF16) | 2 (multi_buffering) | BF16 | Double | Reader0 | Compute |
| 1 (`clear_value_cb_id`) | Clear/init value (-inf for max pool) | tile_size(BF16) | 1 | BF16 | Single | Reader0 | Compute (large kernel), Reader0 (clear) |
| 2 (`raw_in_cb_id`) | Input shard (halo-processed) | in_nbytes_c | max_in_nhw_per_core | BF16 | Backed by input buffer | Input tensor | Reader0 |
| 3 (`in_reader_indices_cb_id`) | Reader indices (sliding window metadata) | reader_indices_size | 1 | UInt16 | Single | Sharded/DRAM buffer | Reader0/Reader1 |
| 4 (`in_cb_id_0`) | Assembled input sticks for reader0 | in_cb_pagesize | 2 (multi_buffering) | BF16 | Double | Reader0 | Compute |
| 5 (`in_cb_id_1`) | Assembled input sticks for reader1 | in_cb_pagesize | 2 (multi_buffering) | BF16 | Double | Reader1 | Compute |
| 6 (`in_idx_cb_id`) | Initial index tile (per-window-element indices) | index_nbytes * TILE_HW | 1 | UInt16/UInt32 | Single | Reader0 | Compute |
| 7 (`pack_tmp_cb_id`) | Temp: packed max values from compute | nbytes * TILE_HW | 1 | BF16 | Single | Compute | Reader1 |
| 8 (`pack_idx_tmp_cb_id`) | Temp: packed max indices from compute | index_nbytes * TILE_HW | 1 | UInt16/UInt32 | Single | Compute | Reader1 |
| 9 (`right_inc_cb_id`) | Index increment for moving right | index_nbytes * TILE_HW | 1 | UInt16/UInt32 | Single | Reader0 | Compute |
| 10 (`down_left_wrap_inc_cb_id`) | Index increment for down-left wrap | index_nbytes * TILE_HW | 1 | UInt16/UInt32 | Single | Reader0 | Compute |
| 11 (`up_left_wrap_inc_cb_id`) | Index increment for up-left wrap (batch boundary) | index_nbytes * TILE_HW | 1 | UInt16/UInt32 | Single | Reader0 | Compute |
| 12 (`compute_tmp_idx_cb_id`) | Temp: incremented indices between iterations | index_nbytes * TILE_HW | 1 | UInt16/UInt32 | Single | Compute | Compute (self-loop) |
| 13+ (`out_cb_id`) | Output values shard | face_width * nbytes | out_nhw * out_ntiles_c | BF16 | Backed by output buffer | Reader1 | Output tensor |
| 14+ (`out_idx_cb_id`) | Output indices shard | face_width * index_nbytes | out_nhw * out_ntiles_c | UInt16/UInt32 | Backed by output[1] buffer | Reader1 | Output tensor |

**Large kernel only** (additional CBs when `window_size_hw > max_rows_for_reduction`):

| CB ID | Purpose | Page Size | Num Pages | Data Format | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|
| `intra_kernel_right_inc_cb_id` | Intra-kernel right increment | index_nbytes * TILE_HW | 1 | UInt16/UInt32 | Single | Reader0 | Compute |
| `intra_kernel_down_left_wrap_inc_cb_id` | Intra-kernel down-left wrap increment | index_nbytes * TILE_HW | 1 | UInt16/UInt32 | Single | Reader0 | Compute |

## Pipeline Pattern Summary

- **in_cb_id_0 / in_cb_id_1**: 2 pages, double-buffered -- allows reader to fill next chunk while compute processes current.
- **in_scalar_cb_id_0**: 2 pages, double-buffered but filled once at initialization and consumed once at cleanup.
- **pack_tmp_cb_id / pack_idx_tmp_cb_id**: 1 page, single-buffered -- synchronization point between compute and reader1 (writer).
- **compute_tmp_idx_cb_id**: 1 page, single-buffered -- self-loop within compute for carrying incremented indices across iterations.
- All increment CBs (right_inc, down_left_wrap, up_left_wrap, intra_kernel variants): 1 page, single-buffered -- filled once during initialization, consumed persistently.

## Index Calculations

The MPWI operation tracks indices in the **flattened H*W space** of the original (unpadded) input tensor. The key index calculation strategy:

1. **Initial index**: Computed from `start_row` and `start_col` (the top-left corner of the first pooling window assigned to this core). The formula depends on padding position:
   - If in top padding: `init_index = -pad_l - pad_t * in_w + start_col + start_row * in_w`
   - If in left padding: `init_index = (start_row - pad_t) * in_w - pad_l + start_col`
   - Otherwise (valid region): `init_index = (start_row - pad_t) * in_w + (start_col - pad_l)`

2. **Per-element indices within a kernel window**: For each element at position `(h, w)` in the kernel, the index is `init_index + h * column_stride + ... ` with `column_stride = dilation_w` and `row_stride = dilation_h * in_w - eff_kernel_w - (dilation_w - 1)`.

3. **Sliding window movement** (compute kernel tracks current position via `current_idx_col`/`current_idx_row`):
   - **Right**: `right_inc = stride_w`
   - **Down-left wrap** (end of row): `down_left_wrap_inc = in_w * stride_h + (1 - out_w) * stride_w`
   - **Up-left wrap** (end of batch): `up_left_wrap_inc = (1 - out_h) * stride_h * in_w + (1 - out_w) * stride_w`

4. **Large kernel corrections**: When the kernel window exceeds `max_rows_for_reduction`, the window is processed in chunks. The inter-position increments are adjusted by subtracting `index_correction = last_top_left_kernel_index - first_top_left_kernel_index` to account for the intra-kernel traversal offset.

5. **Unsigned overflow for negatives**: Negative indices (from padding regions) are allowed to wrap as unsigned integers. Since padding positions never produce the maximum value, the validity of padding indices is unimportant -- only their increment behavior matters.

## Memory Access Patterns

### Read Pattern

- **Input shard**: Reader0 reads sticks from the local L1 input shard using `noc_async_read_one_packet`. Access is strided: for a kernel window at position `ind`, the reader accesses `ind + w*dilation_w + h*dilation_h*in_w_padded` for each `(h, w)` in the kernel. This is a scatter-gather pattern within L1.
- **Reader indices**: Read once at kernel start from sharded L1 (or DRAM if `config_in_dram`). Contains segment descriptors `[start, end]` pairs that define which input positions to process.
- **Increment CBs**: Filled once during initialization; remain resident in L1 throughout execution.

### Write Pattern

- **Output**: Reader1 copies packed results from `pack_tmp_cb_id`/`pack_idx_tmp_cb_id` to `out_cb_id`/`out_idx_cb_id` using `noc_async_read_one_packet` (L1-to-L1 local copy). Writes are sequential face-by-face (1 or 2 faces per channel block) into the output shard.
- Output shard is backed by the output tensor buffer, so writes go directly to the final tensor location.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Matches input tensor's shard grid (`all_cores = input.shard_spec().grid`) |
| Work Splitting | Each core processes `max_out_nhw_per_core` output positions (spatial dimension split) |
| Sharding Modes | HEIGHT_SHARDED: each core gets unique NHW slice; BLOCK_SHARDED: cores in same column share NHW, split channels; WIDTH_SHARDED: all cores share NHW, split channels |
| Load Balancing | Last core may have fewer output elements (`out_nhw_this_core = min(max_out_nhw_per_core, remaining)`) |
| Remainder Handling | `remaining_out_nhw` computed per core; cores beyond total output produce zero sticks |
| Reader Assignment | Reader0 on RISCV_0 (NoC0), Reader1 on RISCV_1 (NoC1); both run the same `reader_mpwi.cpp` kernel |

For block-sharded layouts, `core_nhw_index` is based on `core_y_i` (row index), and multiple columns share the same NHW range but process different channel shards.

## Arguments

### Compile-Time Arguments (Reader Kernel - `reader_mpwi.cpp`)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | max_out_sticks_per_core | uint32_t | Maximum output sticks any core processes (0 = use runtime arg) |
| 1 | kernel_h | uint32_t | Pooling kernel height |
| 2 | kernel_w | uint32_t | Pooling kernel width |
| 3 | pad_w | int32_t | Total width padding (pad_l + pad_r) |
| 4 | in_nbytes_leftover | uint32_t | Bytes in last partial channel tile |
| 5 | in_w | int32_t | Input width (unpadded) |
| 6 | in_c | uint32_t | Channels per shard (ceil-aligned) |
| 8 | reader_id | uint32_t | 0 for reader0, 1 for reader1 |
| 9 | bf16_scalar | uint32_t | BF16 scalar (1.0 for max pool, packed in upper 16 bits) |
| 10 | bf16_init_value | uint32_t | BF16 init value (-inf for max pool) |
| 11 | in_nblocks_c | uint32_t | Number of channel blocks per reduction |
| 12 | in_cb_sz | uint32_t | Input CB size in elements |
| 13 | max_sticks_for_reduction | uint32_t | Max sticks fitting in one reduction chunk |
| 14 | ceil_pad_w | uint32_t | Ceiling-mode extra padding width |
| 15-16 | in_cb_id_0/1 | uint32_t | Input CB IDs for reader0/reader1 |
| 17 | in_shard_cb_id | uint32_t | Raw input shard CB ID |
| 18 | in_reader_indices_cb_id | uint32_t | Reader indices CB ID |
| 19-20 | in_scalar_cb_id_0/1 | uint32_t | Scalar CBs |
| 21 | clear_value_cb_id | uint32_t | Clear value CB ID |
| 22 | is_avg_pool | uint32_t | Pool type (cast from Pool2DType) |
| 23 | one_scalar_per_core | uint32_t | Whether single scalar suffices |
| 24 | config_cb_id | uint32_t | Config tensor CB ID |
| 25 | in_nbytes_c | uint32_t | Bytes per channel row |
| 26 | shard_width_bytes | uint32_t | Shard width in bytes |
| 27 | multi_buffering_factor | uint32_t | Double-buffering factor (2) |
| 28 | stride_w | uint32_t | Stride width |
| 29-30 | dilation_h/w | uint32_t | Dilation factors |
| 31 | zero_pages | uint32_t | Whether to zero partial pages |
| 32 | config_in_dram | uint32_t | Config tensor location flag |
| 33-34 | config_dram_addr/page_size | uint32_t | Config tensor DRAM address/page size |
| 35-36 | reader_dram_addr/page_size | uint32_t | Reader indices DRAM address/page size |
| 37 | in_idx_cb_id | uint32_t | Index CB ID |
| 38 | pack_tmp_cb_id | uint32_t | Packed max values temp CB |
| 39 | pack_idx_tmp_cb_id | uint32_t | Packed max indices temp CB |
| 40-42 | right/down_left/up_left inc CB IDs | uint32_t | Increment CB IDs |
| 43-44 | pad_t/pad_l | uint32_t | Top/left padding amounts |
| 45-47 | right/down_left/up_left inc values | uint32_t | Index increment magnitudes |
| 48-49 | intra_kernel_right/down_left inc values | uint32_t | Large kernel intra-kernel increments |
| 50-51 | out_cb_id/out_idx_cb_id | uint32_t | Output value/index CB IDs |
| 52-53 | intra_kernel inc CB IDs | uint32_t | Large kernel increment CB IDs |
| 54 | indexes_32_bit | uint32_t | Whether to use 32-bit indices |
| 55+ | TensorAccessorArgs | uint32_t[] | Tensor accessor args for reader indices (and config tensor) |

### Compile-Time Arguments (Compute Kernel - `compute_mpwi.cpp`)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | in_ntiles_c | uint32_t | Number of tiles in channel dimension |
| 1 | window_size_hw | uint32_t | kernel_h * kernel_w |
| 3 | max_out_sticks_per_core | uint32_t | Max output sticks (0 = use runtime) |
| 4 | in_c | uint32_t | Channels per shard |
| 5 | in_nblocks_c | uint32_t | Channel blocks for wide reduction |
| 6 | max_sticks_for_reduction | uint32_t | Max sticks per reduction chunk |
| 7 | in_cb_id_0 | uint32_t | Input CB for reader0 |
| 9 | in_scalar_cb_id_0 | uint32_t | Scalar CB |
| 11 | out_cb_id | uint32_t | Output values CB |
| 12 | one_scalar_per_core | bool | Single scalar flag |
| 13 | pre_tilize_cb_id | uint32_t | Pre-tilize CB (unused in MPWI ROW_MAJOR) |
| 14 | is_output_tiled | bool | Output layout flag |
| 15 | is_output_block_format | bool | Block float output flag |
| 16 | in_idx_cb_id | uint32_t | Initial index CB |
| 17-18 | pack_tmp/pack_idx_tmp CB IDs | uint32_t | Temp pack CBs |
| 19-21 | right/down_left/up_left inc CB IDs | uint32_t | Increment CBs |
| 22 | out_idx_cb_id | uint32_t | Output index CB |
| 23-24 | stride_h/stride_w | uint32_t | Stride dimensions |
| 25-26 | in_h_padded/in_w_padded | uint32_t | Padded input dimensions |
| 27-28 | eff_kernel_h/eff_kernel_w | uint32_t | Effective kernel (with dilation) |
| 29 | pad_l | uint32_t | Left padding |
| 30-31 | intra_kernel inc CB IDs | uint32_t | Large kernel increment CBs |
| 32 | compute_tmp_idx_cb_id | uint32_t | Temp index CB for inter-iteration carry |
| 33 | clear_value_cb_id | uint32_t | Clear value CB |
| 34-35 | kernel_h/kernel_w | uint32_t | Kernel dimensions (separate from effective) |
| 36 | indexes_32_bit | uint32_t | 32-bit index flag |

### Runtime Arguments (per core)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | out_nhw_this_core | uint32_t | Number of output sticks this core processes |
| 1 | core_nhw_index | uint32_t | Core's NHW index (for reader indices lookup) |
| 2 | start_row | uint32_t | Starting row in padded input for first window |
| 3 | start_col | uint32_t | Starting column in padded input for first window |

## Kernel Implementations

| Kernel | File | Processor | NoC | Role |
|---|---|---|---|---|
| Reader0 | `reader_mpwi.cpp` | RISCV_0 | NoC0 | Reads input sticks, initializes index/increment CBs |
| Reader1 (Writer) | `reader_mpwi.cpp` | RISCV_1 | NoC1 | Copies packed compute results to output shards |
| Compute | `compute_mpwi.cpp` | Compute (SFPU) | N/A | SFPU max reduction with index tracking |

### Reader0 (Data Movement - RISCV_0)

| Property | Value |
|---|---|
| File | `ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/reader_mpwi.cpp` |
| Assigned Cores | All cores in shard grid |

**Key Logic:**
- **Initialization phase** (runs once):
  - Fills `clear_value_cb` with `-inf` BF16 values, then clears all input CB pages to `-inf` using `clear_out_tiles`.
  - Calls `initialize_return_indices_data` which: computes `init_index` from `start_row`/`start_col` and padding, fills the `in_idx_cb_id` with per-window-element indices (each element gets the flat H*W index), fills all increment CBs with constant increment values (replicated across `fill_c` channel elements per window position).
  - Fills `in_scalar_cb_id_0` with the BF16 scalar (1.0 for max pool).
  - Optionally loads reader indices from DRAM using `TensorAccessor`.
- **Main loop**: Iterates over segments from reader indices. Each segment defines a contiguous range of input top-left positions. For each position:
  - Calls `read_kernel_with_top_left_index` which iterates over `kernel_h * kernel_w` sticks.
  - Reads each stick from L1 input shard using `noc_async_read_one_packet` (address computed as `in_l1_read_base_addr + stick_offset * shard_width_bytes + c_i * MAX_BYTES_PER_REDUCTION`).
  - For large kernels, chunks sticks into groups of `sticks_per_chunk` and pushes intermediate chunks.
  - After the entire kernel window: pushes the final input chunk to `in_cb_id`.
- **Synchronization**: Produces to `in_cb_id` via `cb_reserve_back`/`cb_push_back`. Does NOT interact with output CBs (that is reader1's job). Uses `noc_async_read_barrier` before pushing.

### Reader1 (Writer - Data Movement - RISCV_1)

| Property | Value |
|---|---|
| File | `ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/reader_mpwi.cpp` (same file, `reader_id == 1`) |
| Assigned Cores | All cores in shard grid |

**Key Logic:**
- **Split reader pattern**: Reader1 runs the same kernel file but with `reader_id = 1`. The `if constexpr (reader_id == 0)` / `else` branches route the two readers to different duties.
- Reader1 does NOT read input sticks (all `noc_async_read_one_packet` calls for input data are guarded by `reader_id == 0`).
- **Output copy**: When `kernel_complete` is true (all window sticks processed), reader1:
  1. Computes `output_faces` (1 or 2 depending on whether last channel tile is partial).
  2. Reserves space in `out_cb_id` and `out_idx_cb_id`.
  3. Waits on `pack_tmp_cb_id` (compute has packed max values).
  4. Copies max values from `pack_tmp_cb_id` to `out_cb_id` via `noc_async_read_one_packet`.
  5. Waits on `pack_idx_tmp_cb_id` (compute has packed max indices).
  6. Copies indices from `pack_idx_tmp_cb_id` to `out_idx_cb_id`.
  7. Issues `noc_async_read_barrier`, pops both pack CBs, pushes both output CBs.
- **Synchronization**: Consumes from `pack_tmp_cb_id` and `pack_idx_tmp_cb_id` via `cb_wait_front`/`cb_pop_front`. Produces to `out_cb_id` and `out_idx_cb_id` via `cb_reserve_back`/`cb_push_back`.

### Compute (SFPU)

| Property | Value |
|---|---|
| File | `ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_mpwi.cpp` |
| Assigned Cores | All cores in shard grid |

**Key Logic:**
- **DST register allocation** (fixed slots):
  - `data_dst_idx = 0`: Current input data tile
  - `data_accum_dst_idx = 1`: Accumulated max values (large kernel only)
  - `index_dst_idx = 2`: Current index tile
  - `index_accum_dst_idx = 3`: Accumulated max indices (large kernel)
  - `inc_dst_idx = 4`: Current increment tile
  - `index_temp_dst_idx = 5`: Index backup for large kernel inter-chunk restore
  - `index_scratch_out_dst_idx = 6`: Scratch output for incremented indices

- **Initialization**: Calls `unary_op_init_common` and `max_reduce_with_indices_init<ROW_MAJOR>()`. Waits on scalar and all increment CBs (they are persistent throughout execution).

- **Main loop** (`num_out_sticks_this_core` iterations, nested with `in_nblocks_c`):
  1. `tile_regs_acquire()`.
  2. Loads index tile: first iteration from `in_idx_cb_id`, subsequent from `compute_tmp_idx_cb_id`.
  3. For large kernels: clears `data_accum_dst_idx` and backs up indices to `index_temp_dst_idx` via `copy_dest_values`.
  4. **Chunk loop** (1 chunk for normal kernels, `w_chunks * kernel_h` for large):
     - Waits on `in_cb_id`, copies input data to `data_dst_idx` via `copy_tile`.
     - **Index increment logic**: Determines which increment to apply based on current position:
       - Last chunk of last C block: applies inter-position increment (right, down-left-wrap, or up-left-wrap depending on boundary conditions).
       - Non-last chunk in large kernel: applies intra-kernel increment (right or down-left-wrap within the kernel window).
     - Applies increment via `add_int_tile<copy_format>(index_dst_idx, inc_dst_idx, index_scratch_out_dst_idx)` -- this is an **SFPU integer addition** operation.
     - Calls `max_reduce_with_indices<max_mpwi_kernel_size, ROW_MAJOR, is_large_kernel>(data_dst_idx, index_dst_idx, chunk)` -- the core **SFPU max-reduction-with-index-tracking** operation. Template parameter `max_mpwi_kernel_size` is 9 for small kernels (<=9), 32 otherwise.
     - Pops input CB.
  5. **Pack phase**: `tile_regs_commit()` / `tile_regs_wait()`, then:
     - Packs `data_dst_idx` to `pack_tmp_cb_id` (max values).
     - Packs `index_dst_idx` to `pack_idx_tmp_cb_id` (max indices).
     - If not the last iteration, packs `index_scratch_out_dst_idx` to `compute_tmp_idx_cb_id` (carry incremented indices to next iteration).
  6. `tile_regs_release()`.

- **Cleanup**: Pops all persistent CBs (scalar, all increment CBs) to prevent stale data with program caching.

- **Synchronization**:
  - Consumes: `in_cb_id_0` (from reader0), `in_idx_cb_id` (from reader0, first iter), `compute_tmp_idx_cb_id` (self, subsequent iters), all increment CBs (from reader0, persistent).
  - Produces: `pack_tmp_cb_id` (to reader1), `pack_idx_tmp_cb_id` (to reader1), `compute_tmp_idx_cb_id` (to self).

## Implementation Notes

- **Program factory variants**: There is a single program factory (`pool2d_multi_core_sharded_with_halo_v2_impl_new`) that handles both standard Pool2D and MPWI. The `return_indices` boolean selects between `reader_pool_2d.cpp`/`compute_pool_2d.cpp` (FPU path) and `reader_mpwi.cpp`/`compute_mpwi.cpp` (SFPU path). There is no separate writer kernel; the split-reader pattern uses reader1 as the writer.

- **Type-based operation variants**: Input data is BF16 (block float formats are converted). Index format is dynamically chosen: UINT16 when `in_h * in_w <= 65535`, UINT32 otherwise. The `indexes_32_bit` flag propagates to all index-related CBs, memory operations, and SFPU operations (via the `copy_format` template parameter).

- **UnpackToDestFP32 mode**: FP32 dest accumulation is enabled when `indexes_32_bit` is true or when `is_avg_pool && is_large_kernel` (the latter does not apply to MPWI since it is max pool only). For MPWI with 16-bit indices, FP32 dest is not enabled by default.

- **Broadcast type selection**: N/A. The MPWI operation does not use broadcast. The scalar CB is used only for initialization (the scalar 1.0 is applied as a coefficient in the FPU path but is not used in the SFPU max reduction).

- **Sharding support and constraints**: All three sharding modes are supported: HEIGHT_SHARDED, BLOCK_SHARDED, WIDTH_SHARDED. The input must be sharded in L1 with halo regions already inserted. `max_in_nhw_per_core` must fit in uint16_t. The operation requires `split_reader = true` (enforced by a `TT_FATAL`).

- **FP32 dest accumulation**: Enabled only when 32-bit indices are used (`indexes_32_bit`). Also enables `dst_full_sync_en` for large kernels with indices or 32-bit indices. Math fidelity is HiFi4 with `math_approx_mode = false`.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel functions that the compute kernel dispatches to. The MPWI operation uses three distinct SFPU operations: **max_reduce_with_indices** (the core max-pooling reduction), **add_int_tile** (integer index addition for sliding window movement), and **copy_dest_values** (tile-to-tile copy within DST for index backup/restore).

### SFPU Abstraction Layers

**Operation 1: max_reduce_with_indices**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_max_pool_indices.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_max_pool_indices.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

**Operation 2: add_int_tile**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/add_int_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_add_int.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_add_int.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` (shared) |

**Operation 3: copy_dest_values**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/copy_dest_values.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_copy_dest_values.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_copy_dest_values.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` (shared) |

### Call Chain

All three SFPU operations share the same binary SFPU dispatch mechanism:

**max_reduce_with_indices** call chain:
1. `max_reduce_with_indices<9, ROW_MAJOR, false>(data_dst_idx, index_dst_idx, chunk)` in `compute_mpwi.cpp` calls via `MATH(...)` macro.
2. `llk_math_eltwise_binary_sfpu_max_pool_with_indices<true, DST_ACCUM_MODE, 9, 8, ROW_MAJOR, false>(dst_index, idx_index, chunk)` passes the SFPU function as a callable and the three DST indices to the params dispatcher.
3. `_llk_math_eltwise_binary_sfpu_params_<true>(calculate_max_pool_with_indices<...>, values_tile_idx, indices_tile_idx, chunk)` sets the DEST write address, stalls until SFPU is available, then dispatches the callable. Since vector_mode defaults to `RC` but the max pool function is NOT called in the face-iteration loop (see Parameters Dispatch Summary), the params dispatcher invokes the function once.
4. `calculate_max_pool_with_indices<true, false, 9, 8, ROW_MAJOR, false>(values_tile_idx, indices_tile_idx, chunk)` selects `_calculate_max_pool_with_indices_` (for num_rows <= 9) or `_calculate_max_pool_with_indices_generic_` (for num_rows > 9).

**add_int_tile** call chain:
1. `add_int_tile<UInt16>(index_dst_idx, inc_dst_idx, index_scratch_out_dst_idx)` in `compute_mpwi.cpp`.
2. `llk_math_eltwise_binary_sfpu_add_int<APPROX, 8, UInt16, false>(idst0, idst1, odst)` resolves `INSTRUCTION_MODE = LO16` for UInt16 (or `INT32` for UInt32).
3. `_llk_math_eltwise_binary_sfpu_params_<APPROX>(_add_int_<...>, dst_index0, dst_index1, odst, VectorMode::RC)` dispatches across 4 faces with `SETRWC` dest pointer advancement between faces.
4. `_add_int_<APPROX, 8, LO16, false>(dst_index_in0, dst_index_in1, dst_index_out)` executes SFPLOAD/SFPIADD/SFPSTORE in a loop of 8 iterations.

**copy_dest_values** call chain:
1. `copy_dest_values<UInt16>(index_dst_idx, index_temp_dst_idx)` in `compute_mpwi.cpp`.
2. `llk_math_eltwise_binary_sfpu_copy_dest_values<UInt16>(dst_index_in, dst_index_out)` passes to the params dispatcher with VectorMode::RC.
3. `_llk_math_eltwise_binary_sfpu_params_<false>(copy_dest_value<UInt16, false>, dst_index_in, dst_index_out, 0)` dispatches across 4 faces.
4. `copy_dest_value<UInt16, false, 8>(dst_index_in, dst_index_out, 0)` executes SFPLOAD/SFPSTORE in a loop of 8 iterations.

### Parameters Dispatch Summary

All three SFPU operations use the shared `_llk_math_eltwise_binary_sfpu_params_` dispatcher, but with importantly different dispatch behaviors:

- **Vector mode**: `VectorMode::RC` for all three operations. This processes all 4 faces of a tile (Face 0, Face 1, Face 2, Face 3), iterating 4 times with DEST pointer advancement between each face.

- **Operation invocation**:
  - For **add_int_tile** and **copy_dest_values**: The SFPU function is called once per face (4 calls total for RC mode). Each call runs 8 iterations (ITERATIONS=8), processing 4 rows per iteration = 32 rows per face. Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice to advance the DEST read/write pointer by 16 rows (one face height).
  - For **max_reduce_with_indices**: Despite using the same params dispatcher with `VectorMode::RC`, the function's behavior is special. The max pool function is called 4 times (once per face), but the function itself manages its own DEST addressing via explicit absolute offsets (`values_tile_idx * 64 + offset`). The SETRWC between face calls advances the DEST base pointer, but the SFPU kernel uses absolute addressing that is independent of this base pointer. The function processes even-column and odd-column groups of the ROW_MAJOR data in a single invocation, covering both Face 0 and Face 1 data interleaved in DEST.

- **DEST address progression**: Between face iterations, the params dispatcher advances the DEST pointer by 16 rows via two `TTI_SETRWC` calls (each advances by 8). However, the max pool indices kernel ignores this auto-increment because it uses absolute DEST offsets from the tile base. The add_int and copy_dest_values kernels rely on the `dst_reg++` SFPI primitive within their iteration loops, which is additive with the SETRWC face advancement.

### Annotated SFPU Kernel Source

This operation uses three SFPU kernels: `_calculate_max_pool_with_indices_` (small kernel path, <= 9 rows), `_calculate_max_pool_with_indices_generic_` (large kernel path, <= 32 rows), `_add_int_`, and `copy_dest_value`. All use raw `TT_`/`TTI_` instructions but without complex condition code logic (no CC-modifying instructions at all -- SFPSWAP, SFPTRANSP, SFPLOAD, SFPSTORE, SFPIADD with mod=4 do not set CC). Therefore, **Style A** (inline-commented source code) is used.

#### SFPU Kernel 1: max_reduce_with_indices (small kernel, <= 9 rows)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_max_pool_indices.h

template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    int ITERATIONS             = 8,
    ckernel::DataLayout layout = ckernel::DataLayout::TILE,
    bool accumulate            = false>
inline void _calculate_max_pool_with_indices_(const std::uint32_t values_tile_idx, const std::uint32_t indices_tile_idx, const std::uint32_t chunk)
{ // APPROXIMATION_MODE=true, is_fp32_dest_acc_en=false (or true when indexes_32_bit), ITERATIONS=8, layout=ROW_MAJOR, accumulate=false
    constexpr std::uint32_t dst_tile_size   = 64; // each tile occupies 64 DEST rows
    const std::uint32_t values_tile_offset  = values_tile_idx * dst_tile_size;   // absolute DEST offset for values (tile 0 = row 0)
    const std::uint32_t indices_tile_offset = indices_tile_idx * dst_tile_size;  // absolute DEST offset for indices (tile 2 = row 128)
    constexpr std::uint32_t face_offset    = 16;
    constexpr std::uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16; // controls 16-bit vs 32-bit DEST load/store for indices

    if constexpr (layout == ckernel::DataLayout::ROW_MAJOR)
    {
        // Implementation notes, see the original file for more details
        // ROW_MAJOR data layout: rows interleave Face0/Face1, so consecutive DEST rows
        // alternate between even-column (Face0) and odd-column (Face1) data.
        // Row N even cols at offset N*2, row N odd cols at offset N*2+2.

        auto process_columns = [values_tile_offset, indices_tile_offset](const std::uint32_t col_offset) __attribute__((always_inline))
        {
            // Load 4 pairs of (value, index) rows into LREG0-3 (values) and LREG4-7 (indices)
            TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 0 + col_offset);  // rows 0-1
            TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 4 + col_offset);  // rows 2-3
            TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 8 + col_offset);  // rows 4-5
            TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 12 + col_offset); // rows 6-7
            // index -- LO16 mode loads lower 16 bits as unsigned int, INT32 loads full 32 bits
            TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 0 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 4 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 8 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 12 + col_offset);

            // Execute replay buffer (8 instructions): reduces 4 LREGs to max in LREG0
            // With Dest Index Tracking Mode enabled, LREG4 tracks the index of the max value
            lltt::replay(0, 7);

            // Load the 5th row (row 8) for final comparison
            TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 16 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 16 + col_offset);

            // Final swap: LREG0 gets max of all 9 rows, LREG4 gets corresponding index
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

            // Store results back: max value to row 0, max index to row 0 of indices tile
            TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 0 + col_offset);
            TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 0 + col_offset);
        };

        // Process even columns (col_offset=0) then odd columns (col_offset=2)
        constexpr int even_column_offset = 0;
        constexpr int odd_column_offset  = 2;
        process_columns(even_column_offset);
        process_columns(odd_column_offset);
    }
    // TILE layout path omitted -- not used by MPWI (ROW_MAJOR only)
}
```

#### SFPU Kernel 1b: Initialization and Replay Buffer Programming

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_max_pool_indices.h

template <ckernel::DataLayout layout = ckernel::DataLayout::TILE>
inline void _init_max_pool_with_indices_()
{
    // Enable Destination Index Tracking Mode via SFPU_CONTROL_REG bit[2]:
    // When enabled, LREGs 4-7 are treated as indices for values in LREGs 0-3.
    // During SFPSWAP(ALL_ROWS_MAX), when values are swapped, their indices follow automatically.
    _sfpu_load_config32_(0xF, 0x0, 0x4);

    if constexpr (layout == ckernel::DataLayout::ROW_MAJOR)
    {
        // Record 8 instructions into the replay buffer (indices 0-7)
        lltt::record(0, 7);

        // Tournament-style max reduction: 8 rows -> 4 -> 2 -> 1 max
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // max(row0-1, row2-3) -> LREG0
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX); // max(row4-5, row6-7) -> LREG2
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // max of 4 pairs -> LREG0

        TTI_SFPTRANSP(0, 0, 0, 0); // transpose LREG lanes: puts per-column elements into separate LREGs

        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // cross-lane max reduction
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0); // transpose back: final max in LREG0, index in LREG4
        // Total: 8 instructions recorded. lltt::replay(0,7) replays these.
    }
    // TILE layout replay buffer programming omitted -- not used by MPWI
}
```

#### SFPU Kernel 1c: max_reduce_with_indices (large kernel, <= 32 rows)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_max_pool_indices.h

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS, bool accumulate = false>
inline void _calculate_max_pool_with_indices_generic_(const std::uint32_t values_tile_idx, const std::uint32_t indices_tile_idx, const std::uint32_t chunk)
{ // APPROXIMATION_MODE=true, is_fp32_dest_acc_en depends on indexes_32_bit, ITERATIONS=8, accumulate=true (is_large_kernel)
    constexpr std::uint32_t dst_tile_size         = 64;
    const std::uint32_t values_tile_offset        = values_tile_idx * dst_tile_size;        // DST tile 0 base
    const std::uint32_t indices_tile_offset       = indices_tile_idx * dst_tile_size;       // DST tile 2 base
    const std::uint32_t values_accum_tile_offset  = (values_tile_idx + 1) * dst_tile_size;  // DST tile 1 (accumulator)
    const std::uint32_t indices_accum_tile_offset = (indices_tile_idx + 1) * dst_tile_size; // DST tile 3 (accumulator)
    constexpr std::uint32_t eight_row_offset   = 16;  // 8 rows = 16 DEST row units (2 units per row in ROW_MAJOR)
    constexpr std::uint32_t sixteen_row_offset = 32;
    constexpr std::uint8_t instr_mod_index     = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    // Implementation notes, see the original file for more details

    // Reduces 8 rows to max in LREG0/LREG4, optionally stores result.
    auto reduce_8_rows = [instr_mod_index](const std::uint32_t val_base, const std::uint32_t idx_base, const bool store_result) __attribute__((always_inline))
    {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base + 0);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base + 4);
        TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base + 8);
        TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base + 12);
        TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, idx_base + 0);
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, idx_base + 4);
        TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, idx_base + 8);
        TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, idx_base + 12);

        lltt::replay(0, 7); // replays the 8-instruction sort/reduction from init

        if (store_result)
        {
            TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base + 0);
            TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, idx_base + 0);
        }
    };

    // Reduces 16 rows by reducing two 8-row blocks and swapping their maxima.
    auto process_16_rows = [&reduce_8_rows, values_tile_offset, indices_tile_offset, eight_row_offset, instr_mod_index](
                               const std::uint32_t base_offset, const std::uint32_t col_offset, const bool store_result) __attribute__((always_inline))
    {
        const std::uint32_t val_base_first  = values_tile_offset + base_offset + col_offset;
        const std::uint32_t idx_base_first  = indices_tile_offset + base_offset + col_offset;
        const std::uint32_t val_base_second = values_tile_offset + eight_row_offset + base_offset + col_offset;
        const std::uint32_t idx_base_second = indices_tile_offset + eight_row_offset + base_offset + col_offset;

        reduce_8_rows(val_base_first, idx_base_first, true);   // first 8 rows: store result
        reduce_8_rows(val_base_second, idx_base_second, false); // second 8 rows: keep in LREGs

        // Load first block's max into LREG1/LREG5 for final comparison
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base_first);
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, idx_base_first);

        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // max of 16 rows -> LREG0

        if (store_result)
        {
            TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, idx_base_first);
            TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base_first);
        }
    };

    // Combines max of rows 0-15 with max of rows 16-31, handles cross-chunk accumulation
    auto final_swap = [values_tile_offset, indices_tile_offset, values_accum_tile_offset, indices_accum_tile_offset, instr_mod_index, chunk](
                          const std::uint32_t col_offset) __attribute__((always_inline))
    {
        const std::uint32_t val_first = values_tile_offset + col_offset;
        const std::uint32_t idx_first = indices_tile_offset + col_offset;

        // LREG0/LREG4 already holds max(R16-31) from previous process_16_rows
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_first); // max(R0-15)
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, idx_first);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // max(R0-31) -> LREG0

        if constexpr (accumulate)
        {
            const std::uint32_t val_accum = values_accum_tile_offset + col_offset;
            const std::uint32_t idx_accum = indices_accum_tile_offset + col_offset;
            if (chunk > 0)
            { // load running max from accumulation tile and compare
                TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_accum);
                TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, idx_accum);
                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
            }
            // store running max to accumulation tile
            TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, idx_accum);
            TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_accum);
        }

        // store final result to input tile positions (tile 0 for data, tile 2 for indices)
        TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, idx_first);
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_first);
    };

    // Process even columns then odd columns, each: 16 rows -> 16 rows -> final swap
    constexpr int even_column_offset = 0;
    constexpr int odd_column_offset  = 2;

    process_16_rows(0, even_column_offset, true);
    process_16_rows(sixteen_row_offset, even_column_offset, false); // keep max(R16-31) in LREGs
    final_swap(even_column_offset);

    process_16_rows(0, odd_column_offset, true);
    process_16_rows(sixteen_row_offset, odd_column_offset, false);
    final_swap(odd_column_offset);
}
```

#### SFPU Kernel 2: add_int (integer index addition)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_add_int.h

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _add_int_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{ // APPROXIMATION_MODE=true, ITERATIONS=8, INSTRUCTION_MODE=LO16 (UInt16) or INT32 (UInt32), SIGN_MAGNITUDE_FORMAT=false
    // INSTRUCTION_MODE determines SFPLOAD/SFPSTORE data width: LO16 for 16-bit unsigned, INT32 for 32-bit
    constexpr int sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? INT32_2S_COMP : to_underlying(INSTRUCTION_MODE);

    constexpr std::uint32_t dst_tile_size = 64; // 64 DEST rows per tile

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) // 8 iterations x 4 rows/iter = 32 rows per face
    {
        TT_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, dst_index_in0 * dst_tile_size); // load index value
        TT_SFPLOAD(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_3, dst_index_in1 * dst_tile_size); // load increment value
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4); // LREG0 += LREG1; imod=4 means CC_NONE (no CC update)
        TT_SFPSTORE(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, dst_index_out * dst_tile_size); // store result
        sfpi::dst_reg++; // advance DEST pointer by 4 rows (one SFPU processing width)
    }
}
```

#### SFPU Kernel 3: copy_dest_values (tile copy within DST)

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_copy_dest_values.h

template <DataFormat DATA_FORMAT, bool APPROXIMATION_MODE, int ITERATIONS = 8>
void copy_dest_value(const uint dst_index_in, const uint dst_index_out, const uint /* unused */) {
    // DATA_FORMAT=UInt16 or UInt32 (from copy_format); resolves instr_mod for SFPLOAD/SFPSTORE
    constexpr uint8_t instr_mod_index = GetSfpLoadStoreInstrMod<DATA_FORMAT>();
    constexpr uint dst_tile_size = 64;
    for (int d = 0; d < ITERATIONS; d++) { // 8 iterations x 4 rows = 32 rows per face
        TT_SFPLOAD(p_sfpu::LREG0, instr_mod_index, ADDR_MOD_3, dst_index_in * dst_tile_size);
        TT_SFPSTORE(p_sfpu::LREG0, instr_mod_index, ADDR_MOD_3, dst_index_out * dst_tile_size);
        dst_reg++; // advance DEST pointer
    }
}
```

### SFPU Instructions Used

| Instruction | Description |
|---|---|
| `SFPLOAD` (`TT_SFPLOAD`) | Loads 4 rows (one SFPU processing width) from a DEST register tile into an LREG. The `instr_mod` parameter controls data format: `DEFAULT` for BF16/FP32 values, `LO16` for unsigned 16-bit integers, `INT32` for 32-bit integers. Uses absolute DEST offset addressing. |
| `SFPSTORE` (`TT_SFPSTORE`) | Stores 4 rows from an LREG back to a DEST register tile. Same format modes as SFPLOAD. |
| `SFPSWAP` (`TTI_SFPSWAP`) | Conditionally swaps values between two LREGs. With `ALL_ROWS_MAX` mode, compares corresponding elements and places the maximum into the destination LREG. When Dest Index Tracking Mode is enabled (bit[2] of SFPU_CONTROL_REG), the corresponding index LREGs (4-7) automatically track the movement of values in LREGs 0-3. This is the core max-finding instruction. |
| `SFPTRANSP` (`TTI_SFPTRANSP`) | Transposes the 4 lanes across the 4 LREGs (LREG0-3). Used in the replay buffer to reorganize data so that elements from different DEST rows within the same column are placed into separate LREGs, enabling cross-row comparisons via SFPSWAP. |
| `SFPIADD` (`TTI_SFPIADD`) | Integer addition: `LREG_dest += LREG_c + imm`. With `imod=4` (`CC_NONE`), performs addition without updating the condition code. Used for index increment computation. |
| `lltt::replay(0, 7)` | Replays 8 pre-recorded instructions from the replay buffer (programmed during `_init_max_pool_with_indices_`). The replay buffer contains the SFPSWAP/SFPTRANSP tournament reduction sequence that reduces 8 rows to a single maximum. |
| `_sfpu_load_config32_(0xF, 0x0, 0x4)` | Configures the SFPU control register: sets bit[2] to enable Destination Index Tracking Mode. In this mode, LREGs 4-7 automatically mirror the swap operations applied to LREGs 0-3, so indices are tracked without explicit index manipulation instructions. |

### SFPU Register Usage

| Register | Usage |
|---|---|
| **LREG0** | Primary value accumulator. After reduction, holds the maximum value. During SFPSWAP tournament, receives the winner (larger value) from pairwise comparisons. In `_add_int_`, holds the first operand and receives the addition result. |
| **LREG1** | Secondary value register. Loaded with comparison candidates during reduction. In `_add_int_`, holds the second operand (increment). |
| **LREG2** | Tertiary value register for 4-way reduction in replay buffer. Holds rows 4-5 data during 8-row reduction. |
| **LREG3** | Quaternary value register for 4-way reduction. Holds rows 6-7 data. |
| **LREG4** | Primary index tracking register. With Dest Index Tracking Mode enabled, automatically mirrors LREG0's swap movements. After reduction, holds the index of the maximum value. |
| **LREG5** | Secondary index register, mirrors LREG1 swaps. |
| **LREG6** | Tertiary index register, mirrors LREG2 swaps. |
| **LREG7** | Quaternary index register, mirrors LREG3 swaps. |
| **DST tiles 0-6** | Seven DST tile slots used simultaneously: tile 0 (data input), tile 1 (data accumulator, large kernel), tile 2 (index input), tile 3 (index accumulator, large kernel), tile 4 (increment), tile 5 (index backup, large kernel), tile 6 (scratch output for incremented indices). Each tile is 64 DEST rows. |
| **Replay buffer slots 0-7** | 8 instructions recorded during init: 3x SFPSWAP + SFPTRANSP + 2x SFPSWAP + SFPTRANSP for ROW_MAJOR layout. Replayed via `lltt::replay(0, 7)` during each 8-row reduction. |

### Address Mode Configuration

The SFPU kernels in this operation use ADDR_MOD_3 (Wormhole) or ADDR_MOD_7 (Blackhole) for all SFPLOAD/SFPSTORE instructions. However, these address modes are **not explicitly configured** by the max_pool_with_indices init path. The binary SFPU init (`eltwise_binary_sfpu_configure_addrmod<SfpuType::max_pool_with_indices>`) only configures ADDR_MOD_7 with all-zero increments:

```
ADDR_MOD_7: srca.incr=0, srcb.incr=0, dest.incr=0
```

The SFPU kernels do not rely on ADDR_MOD auto-increment at all. Instead, they use **absolute DEST offset addressing** -- every SFPLOAD/SFPSTORE specifies the complete offset from the DEST base (e.g., `values_tile_idx * 64 + row_offset`). The ADDR_MOD parameter in these instructions is effectively a no-op because the kernel manually computes all addresses.

For the `_add_int_` and `copy_dest_value` kernels, the `dst_reg++` SFPI primitive advances the SFPU's internal DEST read/write pointer by 4 rows per iteration. This is combined with the params dispatcher's `TTI_SETRWC` face advancement (2x `inc_dst_addr<8>` = +16 rows between faces). These two mechanisms together cover all 64 rows of a tile across 4 faces x 8 iterations.

**Wormhole vs Blackhole difference**: The only difference is the ADDR_MOD slot used -- Wormhole uses ADDR_MOD_3, Blackhole uses ADDR_MOD_7. Both are configured identically (all-zero increments). Blackhole also uses `load_replay_buf()` lambda syntax instead of `lltt::record()` for programming the replay buffer, but the recorded instructions are identical.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How is the generic pool operation implemented in ttnn? Specifically, I'm looking at pool_multi_core_program_factory.cpp. What kernels does it use, how does it handle max pooling with indices, and what is the SFPU vs FPU path selection?"
   **Reason**: Initial reconnaissance to understand the operation structure and kernel selection.
   **Key Findings**: Confirmed that `return_indices` selects between `reader_mpwi.cpp`/`compute_mpwi.cpp` (SFPU) and `reader_pool_2d.cpp`/`compute_pool_2d.cpp` (FPU). Learned about split reader pattern and index format configuration.

2. **Query**: "What is max_reduce_with_indices in tt-metal? How does the SFPU max reduction with index tracking work?"
   **Reason**: Needed to understand the core SFPU operation used by the compute kernel.
   **Key Findings**: Query failed (502 error). Fell back to source code inspection of `compute_kernel_api.h` which revealed the `max_reduce_with_indices` template calling `llk_math_eltwise_binary_sfpu_max_pool_with_indices` with parameters for num_rows, layout, accumulation mode, and DST accumulation mode.

3. [SFPU] **Query**: "How does max_reduce_with_indices work in the compute kernel API? What SFPU function does it dispatch to? Also, how does add_int_tile work?" (asked to `tenstorrent/tt-metal`)
   **Reason**: Needed to trace the SFPU dispatch chain from the compute API down to the core SFPU functions.
   **Key Findings**: Confirmed that `max_reduce_with_indices` dispatches to `llk_math_eltwise_binary_sfpu_max_pool_with_indices`, which calls either `_calculate_max_pool_with_indices_` (num_rows <= 9) or `_calculate_max_pool_with_indices_generic_` (num_rows <= 32). `add_int_tile` dispatches to `_add_int_` with `INSTRUCTION_MODE` determined by the data format.

4. [SFPU] **Query**: "How is llk_math_eltwise_binary_sfpu_max_pool_with_indices implemented? What is the SFPU kernel function it calls? Also, how is add_int_tile / llk_math_eltwise_unary_sfpu_add_int implemented?" (asked to `tenstorrent/tt-llk`)
   **Reason**: Needed detailed understanding of the SFPU kernel internals, especially the Dest Index Tracking Mode, replay buffer contents, and the SFPSWAP-based tournament reduction algorithm.
   **Key Findings**: Learned about the `_sfpu_load_config32_` call that enables Dest Index Tracking Mode (LREGs 4-7 mirror LREGs 0-3 during SFPSWAP). Confirmed the replay buffer contains SFPSWAP/SFPTRANSP instructions for 8-row tournament reduction. Identified that the generic version uses nested `reduce_8_rows`/`process_16_rows`/`final_swap` lambdas with cross-chunk accumulation via DST tiles 1 and 3.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/pool/pool_utils.hpp` and `pool_utils.cpp`
   **Reason**: Understanding `FactoryParameters`, `get_factory_parameters`, index data type selection, and pool defines.
   **Key Information**: `MAX_TILES_PER_REDUCTION = 1` for MPWI, `get_defines` maps MAX_POOL2D to `PoolType::MAX`, index type selected by `in_h * in_w` threshold. `split_reader` is always true and enforced by fatal assertion.

2. **Source**: `ttnn/cpp/ttnn/operations/pool/generic/device/pool_op.hpp`
   **Reason**: Understanding operation structure and shared variables.
   **Key Information**: Single `MultiCore` program factory variant. Shared variables include all CB handles for runtime argument override. Operation attributes include `return_indices_` flag.

3. **Source**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 615-637)
   **Reason**: Understanding `max_reduce_with_indices` SFPU API signature.
   **Key Information**: Template parameters: `num_rows` (default 9, max 32), `layout` (ROW_MAJOR used), `accumulate` flag, `ITERATIONS` (default 8). Internally calls `llk_math_eltwise_binary_sfpu_max_pool_with_indices` with `APPROXIMATE=true`.

4. **Source**: `ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp`
   **Reason**: Understanding shared utility functions used by reader kernels.
   **Key Information**: `fill_with_val` fills L1 buffers, `clear_out_tiles` copies init values across CB pages, `load_config_tensor_if_in_dram` handles DRAM-resident config tensors via TensorAccessor, `zero_out_page` zeros pages for block float format padding.
