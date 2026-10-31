# DFB imbalance audit: UNTILIZE + UNTILIZE_WITH_UNPADDING (mainline + experimental/quasar)

Paths below are relative to `ttnn/cpp/ttnn/operations/`.
- Mainline = `data_movement/{untilize,untilize_with_unpadding}`
- Quasar = `experimental/quasar/{untilize,untilize_with_unpadding}`

## Dispatch notes (what can run on Quasar)

- **Mainline `ttnn.untilize`** (`data_movement/untilize/untilize.cpp:215`): the codegen path is skipped on Quasar, so it always takes `untilize_native` and then `prim::untilize`. `select_program_factory` (`untilize_device_operation.cpp:308-385`) picks one of these factories:
  - SingleCore
  - SubCoreGrids
  - Block (`!enough_space_height`, or a wide interleaved tensor)
  - ShardSpecIdentical
  - NDIdentical (excluded on QUASAR at `:338`)
  - ParallelizeColumn (pf=0: interleaved, row too wide, 1 tile tall)
  - SingleCore (pf=1)
  - NDShardInput
  - MultiCore
- **Mainline `ttnn.untilize_with_unpadding` (uwu)**: there is no direct test call. It is reached through mainline `ttnn.to_layout(ROW_MAJOR)` when the padding changes (`core/to_layout/to_layout_op.cpp:181`). The selector is at `untilize_with_unpadding_device_operation.cpp:66-118`:
  - sharded input → Sharded or NDSharded
  - output sharded → MultiCoreInterleaved
  - `!use_multicore` → SingleCore
  - `!enough_space_height` or wide → BlockInterleaved
  - otherwise MultiCoreInterleaved
- **Quasar `untilize`**: the selector is the same as mainline, except that NDIdentical is allowed and uses the quasar metal2 kernels. It is reached by `ttnn.experimental.quasar.untilize/to_layout`. Quasar `to_layout_op.cpp:129` calls untilize when the padding is unchanged, and `:178` calls uwu otherwise.
- **Quasar uwu**: the selector is at `untilize_with_unpadding_device_operation.cpp:19-65` and picks Sharded, NDSharded, SingleCore, BlockInterleaved or MultiCoreInterleaved. The **ColInterleaved factory is never selected**, so it is unreachable.
- **Quasar `to_layout(ROW_MAJOR)` on a SHARDED input whose padding changes** sets `output_memory_config` to the input's sharded config (`to_layout_op.cpp:166-169`). That routes uwu to the Sharded factory's **out_sharded** branch (see F1).

### Test exposure

- **Resnet:**
  - The final `quasar.untilize_with_unpadding` runs on a WIDTH_SHARDED fc output with `memory_config=L1_MEMORY_CONFIG` (`ttnn_functional_resnet50.py:1537`). That is the Sharded factory with the interleaved-blocks writer, which is balanced.
  - `test_untilize_with_unpadding.py` uses DRAM interleaved inputs: (1,1,32|64,1024)→W1000 and (1,1,64,64)→(30,62), on single_core and multi_core. Those run the SingleCore and MultiCoreInterleaved factories (W=32 tiles does not exceed the threshold of 32, so the Block factory is not used). Both are balanced, including the full-padding tile row in `unpad_both`.
  - `test_to_layout.py` uses aligned DRAM interleaved tensors, so it goes through untilize (balanced).
- **Llama and qwen** (`ttnn.untilize`, `use_multicore`, DRAM, `[1,1,32,N]` with N up to 128256/37984): these hit the ParallelizeColumn or Block factory (1 tile tall). Both are balanced. `sub_core_grids` is always None.

## Findings

### F1: CONFIRMED (self-loop DFB, push without pop). Sharded uwu with a sharded output

- **Files:**
  - Quasar: `experimental/quasar/untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_batch_rows_sharded.cpp` (reserve_back(N) at top, push_back(N) at end; no wait/pop)
  - Quasar: `.../writer_unary_unpad_width_16_sharded.cpp` (same pattern)
  - Mainline: `data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_batch_rows_sharded.cpp:31,55` and `writer_unary_unpad_width_16_sharded.cpp:36,106`
- **Factories:** quasar `untilize_with_unpadding_multi_core_sharded_program_factory.cpp` (the `out_sharded` writer bindings) and the mainline factory of the same name (`:234-258`). Each binds `out_sharded`, which is borrowed from OUTPUT, to the **writer as both PRODUCER and CONSUMER**.
- **DFB:** `out_sharded` (legacy c_17). Role: writer DM (self-loop).
- **Trigger:** uwu with a sharded input and a sharded output. That includes quasar `to_layout(ROW_MAJOR)` of any sharded tensor whose logical shape is not equal to its padded shape (the default memcfg is the input's sharded memcfg).
- **Counts:** pushes = `num_unpadded_output_rows`, pops = 0. The DM producer's `finish()` waits on read_acked == read_posted forever.
- **Exercised by tests?** Not in the 4 dirs as far as traced. The resnet final uwu uses an L1 interleaved output. Resnet `to_layout(ROW_MAJOR)` sites only fire for batch 1 or 20 on aligned 12544-row tensors, which take the untilize path.
- **Fix:** after `push_back(n)` add `cb_out.wait_front(n); cb_out.pop_front(n);`, the same handshake `writer_unary_sharded_metal2.cpp` uses. Alternatively, drop the self-loop binding and use the output buffer address directly.

### F2: CONFIRMED (writer skips pop for fully padded tile rows, so compute→writer is unbalanced). Block (WH) untilize-with-unpadding writer

- **Files:**
  - Quasar: `experimental/quasar/untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_wh_multicore.cpp`. The guard `if (start_row_id + tile_height > total_num_rows) this_block_num_rows = total_num_rows - start_row_id;` is followed by `if (this_block_num_rows > 0) write_block(...)`, and wait/pop live inside `write_block` (scratch lines 597-602; source ≈ kernel_main dim3/b/m loop).
  - Mainline: `data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_wh_multicore_metal2.cpp:79-84` (same code).
- **Factories:**
  - quasar `untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp` (`total_num_rows = output.logical_shape()[-2]`, `:180`)
  - mainline `untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp:160`
  - The same writer is shared by both untilize Block factories. For plain untilize, `total_num_rows` is the input logical H, which is never a full tile below padded H, so it is safe there.
- **DFB:** `out`. Producer: compute `untilize_wh(_metal2)`, which pushes every padded input tile row: third_dim × block_col × block_row. Consumer: writer.
- **Trigger:** the uwu BlockInterleaved factory (input wider than 32 tiles with `ncores < ncores_wh`, or `!enough_space_height`) **and** the output trims 32 or more rows of height, so at least one tile row assigned to a core starts at or after the output's logical H.
  - If `start_row_id == total_num_rows`, num_rows is 0, so there is no wait/pop: compute pushes `sub_block` tiles that are never popped (a finish() hang, or a pack stall once the ring fills).
  - If `start_row_id > total_num_rows`, the uint32 underflow gives a huge num_rows, so the writer writes garbage rows. This is a pre-existing correctness bug.
- **Counts per affected tile row:** compute pushes `block_row` tiles, the writer pops 0.
- **Exercised by tests?** No. The resnet uwu cases are ≤32 tiles wide (MultiCoreInterleaved/SingleCore). The llama/qwen to_layout tensors are small or narrow.
- **Fix:** clamp with `rows = start_row_id >= total ? 0 : min(tile_height, total - start_row_id)`, and always do `wait_front(single_sub_block_size_row_arg)` / `pop_front(...)` for every sub-block. Only the NoC-write loop should depend on rows > 0. In practice, remove the `if (this_block_num_rows > 0)` and use `wait_front(single_block_size)` without the `*has_rows`.

### F3: INFO / SUSPECT (category 1: push without reserve on a borrowed DFB)

- **Files:**
  - `eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp`
  - quasar `untilize/.../reader_unary_sharded_metal2.cpp`
  - quasar `untilize_with_unpadding/.../reader_unary_sharded.cpp`
- **What they do:** `push_back(num_tiles_per_core)` with no `reserve_back`. This is the standard idiom for a borrowed input shard.
- **Counts:** these are balanced against compute pops in every untilize/uwu factory that uses them (MultiCore sharded non-block-reader, ShardSpecIdentical, NDIdentical, uwu Sharded).
- **Exercised:** yes, by the resnet final uwu (WIDTH_SHARDED input). It currently passes. Flag it only if finish() also checks reserve/push symmetry. finish_impl only checks posted vs acked, so this should be fine.

### F4: SUSPECT (pre-existing, not test-exercised). SubCoreGrids untilize

- **File:** mainline and quasar `untilize_multi_core_sub_core_grids_program_factory.cpp`.
- **Counts:** the reader pushes `ntiles_per_core`. The writer (`writer_unary_stick_layout_split_rows_interleaved_parallel_columns*.cpp`) pops `(num_sticks/32) × num_tiles_per_core = ntiles_per_column × ntiles_per_core`. These only agree when the tensor is 1 tile tall, and nothing in the factory or validation enforces that. Taller inputs leave the writer waiting forever (a hang even before finish()).
- **Tests:** every test passes `sub_core_grids=None`.
- **Fix:** gate it in validate (`num_tiles_per_col == 1`) or loop the writer per core over its own tiles. ParallelizeColumn has the same writer, but its pf=0 selection requires `num_tiles_per_col == 1`, so it is balanced.

### F5: SUSPECT (low). Single-core untilize with an uneven width-sharded output

- **File:** mainline and quasar `untilize_single_core_program_factory.cpp`.
- **Counts:** `num_columns_of_blocks = W/shard_w` is floored, and `num_tiles_per_column_row = W/num_columns/32`. The reader pushes `num_tiles` (all tiles), but compute and writer handle `num_columns × tiles_per_column_row × H_tiles`. These differ if `shard_w ∤ W`, which leaves leftover input tiles in `src`.
- **Status:** it is likely rejected by validate (`tensor_width % output_shard_width` checks around `untilize_device_operation.cpp:246-260`), but that was not proven for the pf=1 route. No test hits it.

### F6: SUSPECT (low). ND-shard-input factories: reader vs compute block counts

- **Files:** mainline and quasar `untilize_multi_core_nd_shard_input_program_factory.cpp`, and uwu `*_nd_sharded_program_factory.cpp`.
- **The two counts:**
  - Compute `per_core_block_cnt` comes from `page_mapping.core_host_page_indices[core]`, counting non-PADDING block starts.
  - Reader and writer iterate `shard_pages(shard_id)` for `shard_id = start_shard_id; += num_cores`, where `start_shard_id` is the index in `ordered_cores_with_data`.
- **Status:** the counts match if `ordered_cores_with_data[i]` owns exactly shards i, i+N, …, which is the round-robin core group order. The two mappings were not cross-verified. With uneven `shards_per_core` and a different core order, compute's count could differ by one shard's blocks. No test uses ND-sharded untilize.

## Balanced kernels and factories checked

- `kernel_lib/untilize_helpers.inl` `untilize<>`: per block, wait W / pop W / reserve W / push W in the fast, block-based and standard paths. WaitUpfront waits for the total and pops per block. The LLK wrappers (`api/compute/pack_untilize.h`, `experimental/fast_untilize.h`) make no DFB calls.
- Compute kernels:
  - mainline `untilize_metal2.cpp`, `untilize_variable_num_blocks_metal2.cpp` (returns early at 0, which matches readers that push 0), `untilize_wh_metal2.cpp`
  - quasar `untilize_metal2.cpp`, `untilize_variable_num_blocks_metal2.cpp`
  - quasar uwu `untilize.cpp`, `untilize_metal2.cpp`, `untilize_variable_num_blocks.cpp`, `untilize_wh.cpp`, `eltwise_copy.cpp` (1:1 wait/pop/reserve/push)
- Readers:
  - `reader_unary_start_id_metal2` (mainline and quasar)
  - `reader_unary_interleaved_start_id_metal2` (eltwise and quasar)
  - `reader_unary_interleaved_wh_multicore(_metal2)`
  - `reader_unary_sharded_blocks(_metal2)`
  - `reader_unary_nd_sharded_blocks(_metal2)`
  - quasar uwu `reader_unary_interleaved_start_id` and `reader_unary_nd_sharded_blocks`
  - All are 1:1 reserve/push.
- Writers:
  - `writer_unary_stick_layout_split_rows_{single_core,multi_core,multi_core_nd_shard}` (mainline and quasar `_metal2`): 1:1 wait/pop per block. Counts match compute.
  - `writer_unary_stick_layout_split_rows_interleaved_parallel_columns(_metal2)`: balanced for ParallelizeColumn because it is 1 tile tall (see F4 for SubCoreGrids).
  - `writer_unary_sharded_metal2`: wait N, pop N.
  - uwu `writer_unary_stick_layout_split_rows_multicore` (mainline and quasar): BlockRep pops = Σ times·(n_data + [n_mixed>0] + n_pads) = Σ block_count, which equals the reader's `num_pages/tiles_per_row`. `distribute_work` gives each core exactly `nblocks_per_core` (or the cliff count).
  - uwu `writer_unary_unpad_dims_split_rows` (single core): ceil(oy/32) + floor((iy−oy)/32) = iy/32 tile rows, and pops per row = `num_blocks_w_input`. This matches compute's `num_tiles/num_tiles_per_block`.
  - uwu `writer_unary_stick_layout_interleaved_blocks` (quasar) and `kernel/dataflow/writer_unary_stick_layout_interleaved_blocks_metal2` (mainline): pops block_h×block_w, which equals the shard tiles, even for cores with 0 unpadded rows (`break` only skips writes).
  - mainline uwu `writer_unary_unpad_sharded_to_interleaved` and `writer_unary_unpad_cross_sharded`: pop the full padded tile rows, which matches compute.
  - uwu `writer_unary_stick_layout_split_rows_multicore_nd_sharded` (mainline and quasar): skipped out-of-tensor blocks still wait and pop.
  - The `out` DFB of the `writer_unary_unpad_{batch_rows,width_16}_sharded` writers is balanced (batch×ntiles_per_batch or 8·k+rem = compute pushes). Only `out_sharded` is broken (F1).
- Factories with consistent reader/compute/writer runtime args:
  - mainline and quasar untilize: SingleCore (interleaved), MultiCore (interleaved, sharded, block-reader, cliff), ParallelizeColumn, Block (full / cliff_row / cliff_col / cliff_col_row: reader = compute = writer = col×row tiles per dim), ShardSpecIdentical, NDIdentical (quasar), NDShardInput (F6 caveat)
  - uwu: SingleCore, MultiCoreInterleaved, Sharded (non-sharded-output branch), NDSharded
- Not reachable: quasar uwu `MultiCoreColInterleaved` (no selector path; not audited in depth). Mainline NDIdentical is excluded on Quasar.
