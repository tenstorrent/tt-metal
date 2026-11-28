# DFB balance audit — sub-family TILIZE / TILIZE_WITH_VAL_PADDING / EMBEDDING

Paths below are relative to ttnn/cpp/ttnn/operations/ unless they start with ttnn/ or tt_metal/.

## Findings

### CONFIRMED (traced counts)

**T1. Embedding Fused factory, SHARDED output: the compute self-loop pushes N and pops 0 (category 5)**
- embedding/device/embeddings_fused_program_factory.cpp:396-404. When `output_sharded`, the compute kernel is bound to OUTPUT as both PRODUCER and CONSUMER, and no writer is created (:441, :469).
- The compute kernel is `ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp` or `embedding/device/kernels/compute/tilize_chunked.cpp`. Both go through `compute_kernel_lib::tilize` (ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl:182/190), which does `out.reserve_back`/`push_back` per block and never calls `wait_front`/`pop_front` on `dfb::out`.
- Counts for the OUTPUT DFB: pushes = per_core_block_cnt * num_tiles_per_block; pops = 0.
- Under finish(), the PACK/UNPACK drain waits for the TC to empty, so the kernel hangs at kernel end.
- Exercised by tests: NO. Every test and model call uses DRAM interleaved output (llama embedding_1d.py:201/238 uses DRAM_MEMORY_CONFIG; the qwen3_vl test_embedding cases are DRAM).
- Fix: under an `OUT_SHARDED` define, add `DataflowBuffer(dfb::out).wait_front(total); pop_front(total);` at the end of compute. Alternatively, make the output a borrowed DFB with a trivial wait+pop DM writer (`writer_unary_sharded_metal2.cpp`).

**T2. Embedding RM factory, SHARDED output: the reader self-loop pushes N and pops 0 (category 5)**
- embedding/device/embeddings_rm_program_factory.cpp:229-243 binds OUTPUT to the reader as PRODUCER and CONSUMER, with no writer (:339, :383).
- embedding/device/kernels/dataflow/embeddings.cpp:67-75 does reserve_back(1)/push_back(1) per row per chunk and never pops.
- Counts: pushes = num_rows * num_chunks; pops = 0. The DM finish waits for read_acked == read_posted, which never happens, so it hangs.
- Gen2 ValidateProgramSpec already rejects DM self-loop DFBs ("Self-loop DFBs not supported for DM kernels on Gen2"), so on Quasar this config most likely FATALs before it can hang.
- Exercised: NO.
- Fix: after the loop, under OUT_SHARDED, do `dfb_in0.wait_front(n); dfb_in0.pop_front(n);`. Better: use the borrowed buffer with no DFB.

**T3. Embedding TilizedIndices factory: not ported to Quasar; one dangling self-loop and one dangling reserve (categories 4/5)**
- embedding/device/kernels/dataflow/embedding_ind_tilized.cpp:52 `dfb_in1.reserve_back(1)` and :134 `dfb_in1.push_back(1)`. INDEX_SCRATCH is a DM self-loop (factory :167-183) and is never popped.
  - Counts: 1 push, 0 pops, so the finish producer wait never completes.
- `prepare_local_cache(noc, dfb::local_cache, ...)` at :42/:44 is passed the DFB token rather than `scratch::local_cache`. On ARCH_QUASAR, embeddings_common_metal2.hpp builds a `Scratchpad` from that token, so this probably does not even compile on Quasar.
  - On WH/BH it does reserve_back(1 or 2) with no push.
- Factory has no `index_as_scratchpad` handling (the Fused and RM factories do). Selected only when the indices tensor is TILE layout (embedding_device_operation.cpp:19).
- Exercised: NO. All test indices are ROW_MAJOR uint32.
- Fix: port it like Fused/RM, with INDEX_SCRATCH and WEIGHT_CACHE as ScratchpadSpecs on Quasar, `scratch::in1` and `scratch::local_cache`, and no push.

**T4. Quasar tilize / val_padding writers: dead `OUT_SHARDED` branch does wait_front without pop_front (category 2)**
- experimental/quasar/tilize/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp, `#ifdef OUT_SHARDED`: `cb.wait_front(num_pages);` with no pop.
- experimental/quasar/tilize_with_val_padding/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp:27-28: same.
- No tilize or val_padding factory (mainline or quasar) defines OUT_SHARDED (grep is empty), so the path is unreachable today.
- The mainline counterpart, eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp, already pops ("Pop to leave the DFB balanced").
- Exercised: NO.
- Fix: add `cb.pop_front(num_pages);` to both.

### SUSPECT (needs runtime/HW knowledge)

**S1. val_padding readers issue zero-entry `reserve_back(0)`/`push_back(0)`, which advances the DM `tc_idx` round-robin**
- Files:
  - data_movement/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_dims_split_rows_multicore.cpp, `read_block()`
  - the experimental/quasar copy of the same file
  - reader_unary_pad_multicore_both_dims.cpp (`has_rows` pattern)
- Trigger: `read_block(page_id, n_mixed)` is called once per `times` iteration even when n_mixed == 0. That happens when the input height is a multiple of 32 and the output pads extra tile-rows (for example H 32→64, or Z/W padding).
- With has_rows == false the call is reserve_back(0) then push_back(0). Entry counts stay balanced: push = 0, compute's per_core_block_cnt excludes it via `BlockRep::block_count()`.
- The risk: DM `push_back_impl` (tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:206) always does `tc_idx = (tc_idx+1) % num_tcs_to_rr`, even for 0 entries.
  - If the DFB has num_tcs_to_rr > 1 (several consumer engines striped), the producer's TC rotation drifts from the consumers'. Per-TC posted/acked totals then mismatch, and finish() hangs or tiles land on the wrong engine.
  - If num_tcs_to_rr == 1 it is harmless.
- Exercised: likely, by llama tests/debug_ops/test_quasar_tilize_with_val_padding.py and qwen/llama tilize_with_val_padding calls when the input H is 32-aligned and the output H is larger. I did not confirm shapes against TC count.
- Fix: guard the call with `if (n_mixed) read_block(...)`, or early-return in read_block when !has_rows. Ideally also make `push_back_impl`/`pop_front_impl` no-ops for num_entries == 0.

**S2. Tilize Block reader skips a push when `this_block_num_rows == 0` (category 3, latent)**
- data_movement/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_multicore_both_dims.cpp, the `if (this_block_num_rows > 0)` guard.
- When start_row_id == total_num_rows the reader pushes nothing for that column block, but compute (tilize_wh.cpp) still pops block_size_col * third_dim blocks.
- If start_row_id > total_num_rows, the unsigned subtraction wraps instead, so the push still happens but rows are garbage.
- The host split (tilize_multi_core_block_program_factory.cpp:393-466 and val_padding block_interleaved :387-470) derives column blocks from ceil(rows/32), so this should not occur. I did not prove it for third_dim > 1 with total_num_rows = logical rows.
- Exercised: the mainline tilize picker can select Block on Quasar for wide rows (tilize_device_operation.cpp:326-372 has no Quasar guard on this branch). For example, resnet tests/ops/test_tilize_wh_control.py uses ttnn.tilize multicore.
- Fix: compute `this_block_num_rows` with a saturating subtraction and always push (fill pad rows).

**S3. Quasar tilize Sharded / WidthSharded: `per_core_block_cnt` uses floor division (category 3)**
- experimental/quasar/tilize/device/tilize_multi_core_sharded_program_factory.cpp:46-47,126 and tilize_multi_core_width_sharded_program_factory.cpp:43-44,123.
- per_core_block_cnt = num_tiles_per_shard / num_tiles_per_row, while the reader pushes num_tiles_per_shard and the writer waits/pops num_tiles_per_shard.
- If shard height % 32 != 0 then compute pops fewer than were pushed. Example: shard 48x64 gives tiles = 3, tpr = 2, so compute pops 2 and the reader pushed 3.
- The WIDTH path is protected (it requires output shard height == 32). The HEIGHT path relies on the TILE TensorSpec rejecting non-tile-aligned shard shapes. Same pattern in the mainline sharded factory (:161).
- Exercised: resnet tests/ops/test_tilize_width_quasar.py (WIDTH_SHARDED, so protected).
- Fix: TT_FATAL that shard[0] % 32 == 0 in can_use_sharded_optimized_factories.

**S4. Retile: aliased `mid_view` DFB is read with `evil_set_read_ptr` and never waited or popped (category 4)**
- data_movement/tilize/device/kernels/compute/retile.cpp:104-125.
- The `mid` self-loop itself is balanced: per iteration, pushes (via the untilize helper plus fill_zeros_pages) = block_pages, and pops = block_pages.
- The early return/break paths are consistent with the reader (factory :337-350 asserts both clamps are zero together, and the shrink/grow ceil identity holds).
- `mid_view` has no producer. Under finish() its TC should be empty unless it shares a TC with `mid`, which I have not verified.
- Exercised: NO. There are no tiny or retile tiles in the tests.

## Balanced kernels (checked)

Mainline tilize:
- ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl: per block, wait/pop(in) = block_width (or min(32, left) when asymmetric) and reserve/push(out) = block_width. WaitUpfront does one wait(total) plus per-block pops. Balanced.
- ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp: thin wrapper; Default, SingleCore and Sharded factories match reader/writer counts.
- data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_multicore.cpp: pushes (num_rows/th) * ntpb; matches compute nblocks * ntpb and the writer (Default factory, full and cliff).
- data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_singlecore.cpp: num_sticks/32 * num_full_blocks * tpb = num_tiles; matches.
- eltwise/unary/.../writer_unary_interleaved_start_id_metal2.cpp: 1-page wait/pop loop; the OUT_SHARDED branch also pops.
- eltwise/unary/.../reader_unary_sharded_metal2.cpp: push(n) with no reserve on a borrowed DFB. Matches compute pops.
- data_movement/sharded/.../writer_unary_sharded_metal2.cpp: wait(n) + pop(n).
- data_movement/tilize/device/kernels/compute/tilize_wh.cpp with reader_unary_pad_multicore_both_dims.cpp and writer_unary_interleaved_start_id_wh.cpp (Block): full = sbs², cliff_col = cc*sbs, cliff_row = sbs*cr, cliff_col_row = cc*cr. Reader, compute and writer agree; the staging buffer is a Scratchpad, not a DFB.
- data_movement/tilize/device/kernels/compute/retile.cpp (Retile and ShardedRetile): balanced apart from S4.

Quasar tilize:
- experimental/quasar/tilize/device/kernels/compute/tilize.cpp: same helper; balanced.
- experimental/quasar/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_multicore.cpp and _singlecore.cpp: same as mainline; factory args match (Default :198-238, SingleCore :163/:181-189).
- experimental/quasar/tilize/device/kernels/dataflow/reader_unary_sharded.cpp (push n) and writer_unary_sharded.cpp (wait + pop n): balanced, subject to S3.
- experimental/quasar/tilize/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp: non-sharded path is balanced (see T4 for the dead branch).

Mainline tilize_with_val_padding:
- reader_unary_pad_dims_split_rows_multicore.cpp (Default): pushes sum of block_count() = nblocks_per_core_local; compute uses nblocks_per_core from the same split_blocks_for_tilize → distribute_work. I verified the BlockRep / FullRep / split_at arithmetic block-for-block. Writer = ntpr * nblocks_local. Balanced, apart from S1.
- reader_unary_pad_dims_split_rows.cpp (SingleCore): per tile-row, n_blocks_w_input + leftover + w_diff = num_blocks_w_output, plus Y/Z/W pad blocks; compute num_tiles/tpb. Balanced.
- reader_unary_pad_height_width_sharded.cpp (Sharded, WIDTH only): num_batches * ntiles_per_batch = ntiles_per_core = compute nblocks * ntpb = writer num_units. Balanced. The pad buffer is a Scratchpad.
- Block-interleaved: same kernels as the tilize Block factory; balanced apart from S2.

Quasar tilize_with_val_padding:
- reader_unary_pad_dims_split_rows_multicore.cpp, reader_unary_pad_dims_split_rows.cpp, compute/tilize_metal2.cpp, writer_unary_interleaved_start_id_metal2.cpp (non-sharded path): same structure as mainline; balanced apart from S1 and T4.

Embedding:
- embedding/device/kernels/dataflow/embeddings.cpp with embeddings_rm_writer_chunked.cpp / ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp (RM, interleaved output): reader num_rows * num_chunks = writer num_sticks * num_chunks. Index and weight cache are Scratchpads on Quasar. Balanced. (The non-Quasar `dfb_in1` reserve/push self-loop is WH/BH only.)
- embedding/device/kernels/dataflow/embeddings_tilize.cpp with tilize_chunked.cpp / tilize_metal2.cpp and writer_unary_interleaved_start_id_metal2.cpp (Fused, interleaved output): reader num_blocks * Σchunk = num_blocks * ntpb = compute = writer. Balanced. This is the path the tests use (indices 32-aligned with layout=TILE).

## Dispatch notes

- Mainline `TilizeDeviceOperation::select_program_factory` (data_movement/tilize/device/tilize_device_operation.cpp:303-372):
  - Retile / ShardedRetile when the tile shape differs.
  - SingleCore when use_low_perf, or !multicore.
  - Sharded or Default for sharded input.
  - **Block** for !enough_space_height and for the wide-row heuristic. There is NO Quasar guard on this branch, even though the memory note said one was added; Block is Metal2 now (:302 Quasar branch).
  - Default otherwise.
- Quasar `qsr::TilizeDeviceOperation` (experimental/quasar/tilize/device/tilize_device_operation.cpp:199-250): SingleCore, WidthSharded, Sharded, Default. Block is guarded out (!enough_space_height → SingleCore, wide → Default), so `reader_unary_pad_multicore_both_dims_metal2.cpp` is unreachable.
- Mainline val_padding picker (tilize_with_val_padding_device_operation.cpp:43-81): Sharded (WIDTH only), Default, BlockInterleaved, SingleCore. All are Metal2 with a Quasar branch.
- Quasar val_padding picker: only Default and SingleCore; Sharded and BlockInterleaved are not selected. quasar tilize_with_zero_padding is the val_padding op with 0 → zero-fill path.
- experimental/quasar/to_layout/to_layout_op.cpp routes to quasar tilize / tilize_with_val_padding / untilize / untilize_with_unpadding. Mainline to_layout and from_torch(TILE) go to mainline tilize / val_padding.
- Embedding (embedding/embedding.cpp:45-66, embedding_device_operation.cpp:19-25):
  - TILE indices → TilizedIndices.
  - layout=TILE with 32-aligned indices and weight width → Fused.
  - Otherwise RM, followed by a to_layout tilize.
  - Test configs: qwen case 00 (1 index) → RM. qwen case 01, llama prefill and rope (32-multiple seq) → Fused. All outputs are DRAM interleaved.
- Test usage is mostly DRAM-interleaved tilize (Default / SingleCore), resnet WIDTH_SHARDED quasar tilize (WidthSharded factory), and val_padding through the llama debug_ops tests plus from_torch/to_layout.
