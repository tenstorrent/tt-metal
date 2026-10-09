# DFB imbalance audit — family: layout_dm

## EXECUTIVE SUMMARY (exercised-by-tests first)
EXERCISED by tests (will hang under #57646 finish()):
- E1 qsr pad writer_pad_tiled.cpp:53/62 — cb_pad_val reserve(1)+push(1), never popped; writer bound producer+consumer (pad_tile_multicore_program_factory.cpp:148). Hit by llama32_1b_quasar/tests/prototype_ops/test_pad.py (quasar.pad TILE). Fix: Scratchpad (as mainline writer_pad_tiled.cpp:119) or wait/pop(1) at end.
- E2 qsr transpose_wh_rm_sharded.cpp:93/106/110 (+ narrow :53-63) — borrowed output DFB, compute is producer+consumer, no writer (ht<=8); pushes n_blocks*Wt*Ht, wait_front w/o pop, pops=0. Hit by resnet test_fold_transpose.py (b1/b2_wh_4x256x224_aligned, b1_wh_out16_narrow, b1_wh_out8_narrow) / resnet fold transpose(2,3). Fix: pop_front after each push / replace :110 wait with wait+pop.
CONFIRMED but not exercised by the 4 test dirs: s2i RM partial last shard (mainline+qsr), qsr untilize_with_unpadding sharded-output writers (unpad_batch_rows / width_16), untilize wh_multicore writer skipping fully-padded tile rows, mainline pad RM sharded height-only, mainline slice RM sharded reader, mainline concat S2S tiled writer, qsr writer_unary_pad_dims_interleaved dangling reserve, qsr transpose HC sharded generic path, mainline typecast sharded (no consumer), embedding fused/rm sharded output, embedding tilized_indices (unported).
LATENT: many `#ifdef OUT_SHARDED wait_front(no pop)` writers (never defined). SUSPECT: tilize_val_padding reader push_back(0) with multi-TC DFB (likely exercised; depends on whether zero-entry push advances tc_idx), SubCoreGrids untilize, nd-shard untilize ordering, pad width-only shard heights, slice stride.

Per-subfamily details follow (each was traced reader/compute/writer with factory runtime args).


---

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

---

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

---

# DFB imbalance audit — sub-family: sharding / reshard / to_memory_config / move / reallocate / copy / clone

Paths relative to ttnn/cpp/ttnn/operations/ unless noted.

## Dispatch notes (Quasar)
- `ttnn.to_memory_config` (mainline, core/to_memory_config/to_memory_config_op.cpp) -> mainline prim::interleaved_to_sharded / prim::sharded_to_interleaved / ttnn::reshard / prim::copy. No Quasar reroute.
- `ttnn.experimental.quasar.to_memory_config` (experimental/quasar/to_memory_config/to_memory_config_op.cpp) -> prim::qsr::sharded_to_interleaved, quasar::interleaved_to_sharded, quasar::reshard, and the fallback is mainline `ttnn::prim::copy` (with or without dtype).
- `ttnn.interleaved_to_sharded` / `ttnn.sharded_to_interleaved` / `ttnn.reshard` -> mainline data_movement/sharded/** Metal2 factories (they have explicit is_quasar branches). The `experimental.quasar.*` variants -> experimental/quasar/{interleaved_to_sharded,sharded_to_interleaved,reshard}.
- `ttnn.experimental.quasar.reallocate` -> quasar::move -> MoveDeviceOperation: MULTI_CORE_SHARDED (sharded input) -> MoveShardedProgramFactory (no DFB); MULTI_CORE_OVERLAP -> MoveOverlapProgramFactory (Metal2); MULTI_CORE -> reuses mainline `CopyDeviceOperation::SameMemoryConfig::create_descriptor` (legacy ProgramDescriptor + CBDescriptor, NOT ported to Metal2).
- `ttnn.reallocate` (yolo test_reallocate) -> mainline data_movement/move (all three factories are legacy ProgramDescriptor; whether they build/run on Quasar is outside this audit's scope; counted anyway).
- `prim::copy` select: SameMemoryConfig (legacy descriptor) if in/out mem configs equal and not ND; else DefaultRowMajor (RM) / DefaultTilized (TILE), both Metal2.
- `ttnn.clone` -> data_movement/clone (Metal2).
- `experimental.quasar.to_device` -> host `tensor.to_device()`; no device kernels.

## Findings

### F1 CONFIRMED (cross-kernel, cat. 3): mainline sharded_to_interleaved, ROW_MAJOR, partial last height shard
- Factory: data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp
  - reader args: line ~244-248 `num_tiles_per_core = num_units_per_shard` (= shard_spec.shape[0] for RM) on every used core.
  - writer RM args: line ~268-282 `block_height = shard_height = min(shard_h, tensor_h - h0)`.
- Kernels:
  - eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp:25 `dfb.push_back(num_tiles_per_core)` (DFB `in`, borrowed from INPUT, producer).
  - data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2.cpp:38/51 `wait_front(block_height)` / `pop_front(block_height)` (consumer).
- Trigger: ROW_MAJOR input, HEIGHT_SHARDED or BLOCK_SHARDED, and tensor_h % shard_h != 0 → the last-height-shard core(s): pushed = shard_h, popped = tensor_h - h0 < shard_h. finish() on the reader waits forever for (shard_h - shard_height) acks.
- Same imbalance on OUT DFB if convert_df (compute eltwise_copy_metal2 pushes per_core_tile_cnt = num_units_per_shard CTA; writer pops shard_height) — RM+convert is unusual.
- TILE path is balanced (writer uses block_num_tiles = num_units_per_shard for wait/pop and only iterates the unpadded tiles).
- Exercised? Not by the captured tests I found: llama/qwen graph_ops s2i cases are TILE WIDTH_SHARDED; yolo clone path uses TILE. SUSPECT for model runs with non-divisible RM height shards.
- Minimal fix: writer RM `wait_front/pop_front(num_units_per_shard)` (pass a separate `block_num_units` runtime arg = num_units_per_shard, keep the loop over `block_height`) — mirror of the TILE writer. Alternatively make reader push `shard_height` for that core (requires per-core reader args).

### F2 CONFIRMED (cross-kernel, cat. 3): experimental/quasar sharded_to_interleaved, ROW_MAJOR, partial last shard
- Factory: experimental/quasar/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp
  - reader: line ~222 `reader_run.runtime_arg_values["num_units"][core] = num_units_per_shard` for all used cores.
  - writer RM: lines ~282-305: `shard_height = num_units_per_shard_height_last` for end core (HEIGHT_SHARDED) / end row (BLOCK_SHARDED), passed as `block_height`.
  - compute (convert_df): `num_units = num_units_per_shard` (line ~333).
- Kernels: experimental/quasar/sharded_to_interleaved/device/kernels/dataflow/reader_unary_sharded.cpp:17 `cb_in0.push_back(num_units)`; writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:29/42 `wait_front/pop_front(block_height)`.
- Trigger: RM, HEIGHT/BLOCK sharded, num_units_height % shard_h != 0 → end core pushes shard_h, pops shard_h_last. Hang in reader finish().
- Exercised? resnet tests/ops/test_sharded_to_interleaved.py (shard_h 255/256/1568/6272 × 1-2 cores) and test_to_memory_config sharded→interleaved use exact multiples → NOT triggered there. Real path: resnet conv2d_DRAM uses quasar.to_memory_config(RM HEIGHT_SHARDED → DRAM). If a conv activation's shard height was rounded up (e.g. NHW not divisible by num_cores*32-ish), it WILL hang under #57646. SUSPECT for model runs.
- Minimal fix: same as F1 — writer RM wait/pop the full `num_units_per_shard` (new RT arg) while writing only `block_height` sticks; or set reader `num_units` to shard_height for the last core.

### F3 SUSPECT/legacy (cat. 3): copy SameMemoryConfig, RM sharded + dtype conversion
- data_movement/copy/device/copy_same_memory_config_program_factory.cpp:~180-198 + 205-220: RM sharded reader (`reader_unary_stick_start_id.cpp`) pushes `num_units_per_core * num_shards` (`full_input_row/input_unit_size`), but compute eltwise_copy CTA = `num_units_per_core` → if num_shards>1 and convert_dtype, compute pops fewer and writer waits more than compute pushes. Already a hang pre-#57646; legacy ProgramDescriptor path (reached on Quasar via quasar move MULTI_CORE only without dtype change), so not exercised. No action needed for #57646.

### Notes (cat. 4, benign for finish())
- Push-without-reserve on borrowed shard DFBs: reader_unary_sharded_metal2.cpp:25 (mainline) and experimental/quasar/sharded_to_interleaved/.../reader_unary_sharded.cpp:17. Push count matches consumer pops (except F1/F2), so finish() is satisfied; no reserve needed since the borrowed DFB is sized to the shard.
- Reshard (generic, generic diff_width, same_width, same_height; mainline + quasar) kernels use only `get_write_ptr/get_read_ptr` on the shard DFB and scratch DFB — zero reserve/push/wait/pop → posted==acked==0, finish() trivially satisfied.
- nd_reshard_copy_local_shards.cpp (mainline + quasar): no DFB at all.
- copy/device/kernels/writer_unary_start_id.cpp:19-20 `#ifdef OUT_SHARDED wait_front(num_tiles)` with no pop — dangling wait, but OUT_SHARDED is never defined by any factory that uses this kernel (SameMemoryConfig only adds BACKWARDS). Dead code; would be a cat. 2 imbalance if ever enabled. Fix (hygiene): add `pop_front(num_tiles)`.

## Balanced (checked)
- mainline interleaved_to_sharded factory: reader_unary_sharded_blocks_interleaved_start_id_metal2 (reserve/push block_num_tiles=curr_num_units_per_shard) ↔ writer_unary_sharded_metal2 / writer_unary_sharded_blocks_start_id_metal2 (wait/pop num_units / block_width_padded_num_tiles = same value) ↔ eltwise_copy_metal2 (per_core_tile_cnt same). RM: reader_unary_stick_layout_..._metal2 (reserve/push block_height) ↔ writer num_units / block_height (curr_num_units_per_shard updated together with shard_height). Balanced.
- experimental/quasar interleaved_to_sharded: same structure (reader_unary_sharded_blocks_interleaved_start_id, reader_unary_stick_layout_..., writer_unary_sharded, writer_unary_sharded_blocks_start_id, writer_unary_sharded_stick_layout_start_id, compute eltwise_copy) — balanced.
- mainline sharded_to_interleaved TILE path (reader_unary_sharded_metal2 ↔ writer_unary_sharded_blocks_interleaved_start_id_metal2, block_num_tiles = num_units_per_shard; ttnn/kernel/compute/eltwise_copy_metal2 CTA same) — balanced.
- experimental/quasar sharded_to_interleaved TILE path (reader_unary_sharded ↔ writer_unary_sharded_blocks_interleaved_start_id, block_num_tiles = num_units_per_shard; compute eltwise_copy num_units same) — balanced.
- nd_reshard_copy_pages_reader/_writer (mainline + quasar): reserve/push and wait/pop 1 per page over identical [start_page,end_page) — balanced.
- reshard generic / same_width / same_height (mainline + quasar): no counter ops — balanced (0/0).
- quasar move overlap: move_interleaved_with_overlap(.cpp/_writer.cpp), move_stick_layout_interleaved_with_overlap(.cpp/_writer.cpp): reserve/push num_pages ↔ wait/pop num_pages, same RT args per core — balanced.
- quasar move sharded: reader_unary_local_l1_copy_backwards.cpp — no DFB.
- mainline move (legacy): overlap kernels self-loop reserve/push then wait/pop num_tiles — balanced; sharded backwards copy — no DFB.
- copy DefaultTilized: reader_unary_interleaved_start_id_metal2 ↔ writer_unary_interleaved_start_id_metal2 (non-OUT_SHARDED branch) ↔ sharded/.../compute/eltwise_copy_metal2 (per_core_tile_cnt) — same num_pages per core — balanced.
- copy DefaultRowMajor parallel: redistribute_pages_row_major_parallel_reader/_writer, 1 per unit over identical ranges — balanced.
- copy DefaultRowMajor non-parallel: redistribute_pages_row_major_reader (one reserve at output-subblock start, one push at output-subblock end; traced both branches) ↔ redistribute_pages_row_major_writer (one wait/pop per output subblock) — balanced.
- copy SameMemoryConfig (legacy): reader_unary_start_id ↔ writer_unary_start_id (tile), reader_unary_stick_start_id ↔ writer_unary_stick_start_id (RM sharded, num_shards equal for same mem config), ttnn/kernel/dataflow/reader/writer_unary_stick_layout_interleaved_start_id (RM interleaved), ttnn/kernel/compute/eltwise_copy (CTA per group) — balanced except F3.
- clone: read_kernel/write_kernel (+_rm, _sharded, _rm_sharded) per-unit reserve/push ↔ wait/pop with identical per-core counts; compute_kernel.cpp uses compute_kernel_lib::copy PerTile wait/pop + reserve/push for num_tiles CTA = same per-group count (convert only on TILE, validated) — balanced.

## Coverage
Factories checked: 17 (i2s mainline+quasar, s2i mainline+quasar, reshard 5 mainline + 5 quasar, quasar move overlap/sharded/multicore→copy, mainline move 3, copy 3, clone 1). Kernels checked: ~45. Findings: 2 CONFIRMED (F1, F2 — same RM partial-last-shard bug in mainline and quasar s2i; not hit by current per-op test shapes), 1 SUSPECT legacy (F3), 1 dead-code dangling wait (OUT_SHARDED).

---

# DFB imbalance audit: pad / slice / concat / split / repeat-expand (sub-family of layout_dm)

Paths: M = ttnn/cpp/ttnn/operations/data_movement, Q = ttnn/cpp/ttnn/operations/experimental/quasar

## Dispatch notes (which factory is picked on Quasar, and which tests reach it)
- **ttnn.pad (M)**: `pad_device_operation.cpp:70`. RM+sharded → WidthOnly / HeightOnly / MultiCoreDefault; RM interleaved → MultiCoreDefault (v2 kernels) by default (`use_multicore=True`); TILE → PadTileMulticore (single-core tile/RM only when use_multicore=False).
  - yolo `test_pad.py` (RM, DRAM, pad on dim1) → M PadRmReaderWriterMultiCoreDefault (v2).
  - llama rope pad / `tests/ops/test_pad.py` / e2e `test_llama_e2e.py:1126` / `test_quasar_nlp_concat_heads_decode.py` (TILE, interleaved) → M PadTileMulticore.
- **ttnn.experimental.quasar.pad (Q)**: same selector (`Q/pad/device/pad_device_operation.cpp:73`). llama `tests/prototype_ops/test_pad.py:52` (TILE, DRAM, seq96→128 / 128→256 / 500→512) → **Q PadTileMulticore**.
- **ttnn.slice (M)**: `slice_device_operation.cpp:313`. TILE → SliceTile; RM → SliceRm (SliceRmSharded only if HS in+out, no step; SliceRmStride if step≠1); use_tensor_args → SliceTileTensorArgs. Tests: llama slices are TILE (SliceTile) or RM-DRAM (`test_quasar_slice_rm_boundary.py` → SliceRm); yolo slice → SliceTile/SliceRm. `ttnn.experimental.quasar.slice` (llama prototype test_slice, TILE) → Q SliceTile.
- **ttnn.concat (M)**: `concat_device_operation.cpp:36`. Interleaved → ConcatProgramFactory (ConcatTiledUnaligned is hard-gated off on Quasar at `concat_tiled_unaligned_program_factory.cpp:80`). Sharded: S2I / BlockSharded / S2SMulti are **legacy ProgramDescriptor** factories (not DFB/Metal2 → not Quasar-buildable); S2SRM and S2STiled are Metal2. yolo/llama concats (interleaved) → ConcatProgramFactory; yolo `test_concat_neck_sharded` (RM HEIGHT_SHARDED, 2 inputs, dim=-1) → S2SRM if output stays sharded L1 (else S2I = legacy).
- **ttnn.split (M)**: `split.cpp` — native TILE kernel (split_program_factory) only for equal last-dim N-way TILE splits fitting the grid; everything else → N × ttnn.slice. llama `test_split.py` (dim=3 halves, TILE) → native split factory; yolo `test_split.py` (dim=1) → slice.
- **ttnn.expand (M)** → ttnn.repeat. `repeat.cpp:622`: codegen path (`repeat/codegen/*`, legacy ProgramDescriptor + CircularBuffer — not a DFB factory; Quasar reachability not gated, flag for owner) else native RepeatProgramFactory{LastDim,HigherDim} (Metal2). qwen3_vl `test_expand.py`.

## Findings

### CONFIRMED (traced counts)

1. **Q pad tile-multicore writer — self-loop DFB pushed, never popped (cat 4/5)** — EXERCISED
   - `Q/pad/device/kernels/dataflow/writer_pad_tiled.cpp:53` `cb_pad_val.reserve_back(1)` … `:62` `cb_pad_val.push_back(1)`; no wait_front/pop_front anywhere. Factory binds writer as both `ProducerOf(CB_PAD,"cb_pad_val")` and `ConsumerOf(CB_PAD,"cb_pad_val")` (`Q/pad/device/pad_tile_multicore_program_factory.cpp:148`).
   - Counts per core: pushed=1, popped=0 → finish() producer drain (posted 1 ≠ acked 0) spins forever on every writer core.
   - Exercised by `models/experimental/llama32_1b_quasar/tests/prototype_ops/test_pad.py` (ttnn.experimental.quasar.pad, TILE, DRAM, all 3 params). (`cb_input` in reader_pad_tiled.cpp:61/64 vs writer :77/:82 is balanced — identical within_input_region walk.)
   - Fix: do what mainline did — make the pad tile a `Scratchpad` (`M/pad/device/kernels/dataflow/writer_pad_tiled.cpp:119` + `flush_l2_cache_range` on Quasar) and drop the DFB; or minimal: delete the reserve/push (use get_write_ptr only) or add `cb_pad_val.wait_front(1); … cb_pad_val.pop_front(1);` at kernel end. (Mainline comment notes Quasar rejects DM self-loops anyway.)

2. **Q pad tile single-core writer — dangling reserve on scratch DFB (cat 4)** — NOT exercised (needs use_multicore=False)
   - `Q/pad/device/kernels/dataflow/writer_unary_pad_dims_interleaved.cpp:31` `cb_out1.reserve_back(1)` ("not pushing anything, just using the space"), no push. reserves=1, pushes=0. cb_out0 is balanced (reader pushes `num_unpadded_tiles`, writer pops W·Z·Yt·Xt unpadded = same; factory `pad_tile_program_factory.cpp:186`).
   - Fix: switch `dfb::pad` to a Scratchpad (as mainline `M/pad/.../writer_unary_pad_dims_interleaved.cpp:28`) or drop reserve_back and use get_write_ptr only.

3. **M pad RM height-only sharded — cross-kernel: producer pushes, bound consumer never pops (cat 3)** — NOT exercised (RM sharded input, width unchanged)
   - Reader `M/pad/device/kernels/dataflow/reader_pad_dims_rm_sharded.cpp:32` `dfb_out0_exp.reserve_back(num_sticks_padded)` / `:70` `push_back(num_sticks_padded)`.
   - Writer `M/pad/device/kernels/dataflow/writer_pad_dims_rm_sharded.cpp:74` binds `out_shard` as CONSUMER (`pad_rm_sharded_height_only_program_factory.cpp:386-392`) but only calls `get_write_ptr()` (`:86`); no wait_front/pop_front. pushed=num_sticks_padded(shard_height_padded), popped=0 → reader finish() hangs.
   - Fix: either drop the reserve/push in the reader (Q version writes the output shard in place via TensorAccessor with no DFB — `Q/pad/.../reader_pad_dims_rm_sharded.cpp:101-106`), or add `dfb_out0_exp.wait_front(num_sticks_padded); dfb_out0_exp.pop_front(num_sticks_padded);` at end of the writer.

4. **M slice RM height-sharded — self-loop DFB pushed, never popped (cat 5)** — NOT exercised (RM HS input AND output, no step)
   - `M/slice/device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp:41` `dfb_out.reserve_back(num_sticks_unpadded)` / `:90` `dfb_out.push_back(num_sticks_unpadded)`; single-kernel factory binds reader as PRODUCER+CONSUMER of `out` (`slice_program_factory_rm_sharded.cpp:324-345`); nothing pops. pushed=shard_height_unpadded, popped=0.
   - Fix: remove reserve/push (Q copy `Q/slice/.../slice_reader_unary_unpad_dims_rm_sharded.cpp` has no FIFO ops), or append `dfb_out.wait_front(n); dfb_out.pop_front(n);`.

5. **M concat S2S-tiled writer — self-loop output DFB pushed, never popped (cat 5)** — NOT exercised (needs two TILE height-sharded inputs, width concat, sharded output)
   - `M/concat/device/kernels/dataflow/writer_height_sharded_width_concat_two_tensors_tiled.cpp:39` `output_dfb.reserve_back(Wt0+Wt1)` / `:57` `output_dfb.push_back(Wt0+Wt1)` per row × input0_num_tiles_height; writer is both PRODUCER and CONSUMER of `output` (`concat_s2s_tiled_program_factory.cpp:276-283`), no wait/pop. pushed=Ht·(Wt0+Wt1), popped=0.
   - Other DFBs of this trio balanced: input0/input1 (reader push Ht·Wt without reserve, `reader_…_tiled.cpp:48-49` — push-without-reserve but totals match compute pops Ht·Wt via `transpose<>` helper), input{0,1}_transpose (compute push Wt/row, reader wait/pop Wt/row), concat (reader push Wout/row, compute pop Wout/row), output_transpose (compute push Wout/row, writer pop Wout/row).
   - Fix: drop reserve/push on output_dfb and advance a local write cursor from get_write_ptr(), or add `output_dfb.wait_front(N); output_dfb.pop_front(N);` at end.

### SUSPECT (need runtime/config knowledge)

S1. **Pad RM width-only sharded (stickwise), M and Q** — writer pushes `padded_shard_height`, reader pops `unpadded_shard_height`:
   - M: `writer_pad_dims_rm_sharded_stickwise.cpp:55` reserve(padded_shard_height), `:81` push_back(1)×padded_shard_height; `reader_pad_dims_rm_sharded_stickwise.cpp:34/52` wait/pop(1)×unpadded_shard_height. Factory `pad_rm_sharded_width_only_program_factory.cpp:57/61/155/183` (input vs output shard_spec heights).
   - Q: same at `Q/pad/.../writer_pad_dims_rm_sharded_stickwise.cpp:58/77`, `reader_…_stickwise.cpp:37/44`; factory `Q/pad/.../pad_rm_sharded_width_only_program_factory.cpp:64/68`.
   - Balanced only when output shard height == input shard height (normal for width-only since total H is equal, but nothing enforces it). If padded > unpadded, writer finish() hangs. Not exercised. Fix: have reader pop the remaining `padded - unpadded` entries at end (or have writer push only unpadded and fill tail without FIFO ops).

S2. **Slice RM stride 4d/nd (M & Q)** — reader pushes one row per processed row bounded by `rows_processed < num_rows_for_this_core`, writer pops exactly `num_rows_for_this_core` (`reader_multicore_slice_4d.cpp:115-176`, `writer_multicore_slice_4d.cpp:77-90`; nd analog). Balanced as long as the host's per-core row count never exceeds what the reader's iteration space yields; mismatch would already deadlock pre-#57646. Only reached with step≠1 (not in tests).

### Latent (compiled-out branches)
- `#ifdef OUT_SHARDED` branch: `dfb.wait_front(num_pages)` with **no pop_front** in `M/slice/.../writer_unary_interleaved_start_id.cpp:25-26` and `Q/slice/.../writer_unary_interleaved_start_id.cpp:21-22` (also in Q copies `Q/tilize_with_val_padding/.../writer_unary_interleaved_start_id_metal2.cpp:27-28` and `Q/reduction/generic/.../writer_unary_interleaved_start_id_metal2.cpp:28-29` — outside this sub-family). No slice/concat factory defines OUT_SHARDED, so dormant; mainline eltwise `writer_unary_interleaved_start_id_metal2.cpp:37-38` already does wait+pop — copy that.

## Balanced (checked)
- M `pad/.../reader_pad_tiled.cpp` + `writer_pad_tiled.cpp` (PadTileMulticore; pad tile is Scratchpad) — EXERCISED by llama TILE pads; balanced.
- M `pad/.../reader_pad_dims_rm_interleaved_v2.cpp` + `writer_pad_dims_rm_interleaved_v2.cpp` (MultiCoreDefault; same num_sticks_per_core/per_barrier per core, factory :269/:282) — EXERCISED by yolo test_pad; balanced.
- M `pad/.../reader_pad_dims_rm_interleaved.cpp` + `writer_pad_dims_rm_interleaved.cpp` (single-core RM) — balanced (identical W/Z/Y loops).
- M `pad/.../writer_unary_pad_dims_interleaved.cpp` + eltwise `reader_unary_interleaved_start_id_metal2.cpp` (single-core tile) — balanced (num_unpadded_tiles both sides).
- M `pad/.../writer_pad_dims_rm_sharded.cpp` `pad` DFB — no FIFO ops (address-only), OK; (`out_shard` → finding 3).
- Q `pad/.../reader_pad_tiled.cpp` `cb_input` vs writer — balanced (cb_pad_val → finding 1).
- Q `pad/.../reader_pad_dims_rm_interleaved_v2.cpp` + `writer_…_v2.cpp` — cb_in0 balanced; cb_pad / cb_pad_align address-only (no FIFO ops) OK.
- Q `pad/.../reader_pad_dims_rm_interleaved_sc.cpp` + `writer_…_sc.cpp`, and `reader/writer_pad_dims_rm_interleaved.cpp` — balanced.
- Q `pad/.../reader_pad_dims_rm_sharded.cpp` + `writer_pad_dims_rm_sharded.cpp` (height-only) — cb_pad reserve/push 1 vs wait/pop 1; output written in place; balanced.
- Q `pad/.../reader_unary_interleaved_start_id.cpp` — balanced vs cb_out0 consumer.
- M/Q `slice/.../reader_unary_unpad_dims_interleaved_start_id.cpp` + `writer_unary_interleaved_start_id.cpp` (SliceTile; reader num_tiles == writer num_pages per core) — EXERCISED (llama/yolo TILE slices, Q prototype slice); balanced.
- M/Q `slice/.../reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp` (+ writer) — balanced; M's `dfb_tensor` self-loop does reserve/push/wait/pop ×2 each.
- M/Q `slice/.../slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` + `slice_writer_unary_stick_layout_interleaved_start_id.cpp` (SliceRm, incl. M chunked path with identical batch math) — EXERCISED (test_quasar_slice_rm_boundary.py); balanced.
- Q `slice/.../slice_reader_unary_unpad_dims_rm_sharded.cpp` — no FIFO ops; OK.
- M `concat/.../reader_concat_interleaved_start_id.cpp` / `reader_concat_stick_layout_interleaved_start_id.cpp` (ublock=1) + eltwise `writer_unary_interleaved_start_id_metal2.cpp` / `writer_unary_stick_layout_interleaved_start_id_metal2.cpp` (ConcatProgramFactory; num_pages_per_core both sides, factory :383-408) — EXERCISED (yolo/llama concats); balanced.
- M `concat/.../reader_height_sharded_width_concat_two_tensors.cpp` (S2SRM, two instances, no FIFO ops) — balanced; likely exercised by yolo test_concat_neck_sharded.
- M `concat/.../compute/height_sharded_width_concat_two_tensors.cpp` + `reader_…_tiled.cpp` — balanced (writer output_dfb → finding 5).
- M `split/.../reader_tm_tile_layout_split_two_chunks.cpp` (out_num_tensors=1) + `writer_split_n_chunks_tile.cpp` — same z/y/x CTAs per core; balanced. EXERCISED by llama test_split.
- M `repeat/device/kernels/*.cpp` (5 native kernels) — no DFB FIFO ops at all; balanced. (qwen3_vl expand.)
- M `repeat/codegen/kernels/reader_repeat_{higherdim,last_dim}_rm.cpp` + `writer_repeat_rm.cpp` — legacy CircularBuffer; reader pushes num_out_pages in batches, writer pops prime+steady+drain = num_pages; balanced. Note: legacy ProgramDescriptor factory, not Quasar-gated.
- Not Quasar-reachable (skipped): M ConcatTiledUnaligned (gated off), ConcatS2I / BlockSharded / S2SMulti (legacy ProgramDescriptor), strided_slice_* kernels (no factory references them).

## Coverage counts
- Kernels checked: pad M 11 + Q 14; slice M 10 (+2 unused strided) + Q 10 (+2 unused); concat 5 (+compute 1); split 2; repeat 5 native + 3 codegen.
- CONFIRMED: 5 (1 exercised by tests: Q pad tile multicore). SUSPECT: 2. Latent: OUT_SHARDED wait-without-pop (4 files).

---

# DFB imbalance audit — layout_dm / sub-family TRANSPOSE + PERMUTE + RESHAPE + TYPECAST + FULL/FILL

Paths below are relative to `ttnn/cpp/ttnn/operations/` unless absolute.
"Quasar-buildable" = Metal2 ProgramSpec/DataflowBufferSpec factory. ProgramDescriptor/CBDescriptor (legacy) factories are noted as not Quasar-reachable (DataMovementKernel FATAL, cf. reshape_device_operation.cpp:15 comment).

## Dispatch notes (what the tests hit)

| Test call | Config | Factory on Quasar |
|---|---|---|
| llama `tests/ops/test_transpose.py` `ttnn.transpose(x,1,2)` | TILE [1,1,1or32,64] interleaved | mainline `TransposeHCTiledInterleavedProgramFactory` (C=1 -> NEEDS_PADDING) |
| llama `tests/prototype_ops/test_transpose.py` `ttnn.experimental.quasar.transpose(x,1,2)` | same | qsr `TransposeHCTiledInterleavedProgramFactory` |
| resnet `tests/ops/test_fold_transpose.py` `experimental.quasar.transpose(.,2,3)` | RM HEIGHT_SHARDED [n,4,256,224], [1,128,16,256], [1,128,8,256] | qsr `TransposeWHShardedRMProgramFactory` (ht<=8 in all configs -> borrowed-output self-loop path; narrow-row for H=16/8) |
| resnet same, `(.,1,2)` | RM HEIGHT_SHARDED [1,4,224,256] | qsr `TransposeHCShardedProgramFactory` (special-case path very likely; generic path only if shard geometry fails the divisibility test) |
| yolo `test_permute.py` RM (0,2,3,1), (0,3,1,2) | RM interleaved | permute.cpp -> `prim_permute` `MultiCoreBlockedGeneric` |
| yolo RM (0,1,3,2), (0,2,1) | RM | permute.cpp -> `ttnn::transpose(-2,-1)` -> mainline `TransposeWHProgramFactory` RM path |
| yolo TILE (0,1,3,2), (0,2,1) | TILE L1 interleaved | mainline `TransposeWHProgramFactory` tiled path |
| yolo TILE (0,3,1,2) | TILE | `prim_permute` `MultiCoreTileRowInvariant` (needs_padding when output_H%32) |
| llama/qwen `ttnn.reshape` / `experimental.quasar.reshape` | RM + TILE | mainline `ReshapeViewRM/TiledProgramFactory` (Metal2, scratchpad+DFB) / qsr `ReshapeView{RM,Tiled}MetalV2ProgramFactory` (Quasar-routed, reshape_device_operation.cpp:18,26) |
| llama `tests/ops/test_typecast.py` `ttnn.typecast` | TILE interleaved [1,1,32,2048] bf16<->fp32, ->bfp8 | mainline `TypecastProgramFactory` |
| llama sampling `ttnn.typecast(..., sub_core_grids=)` | interleaved | mainline `TypecastSubgridProgramFactory` |
| llama model attention typecasts on sharded heads | only if tile sizes equal + L1 (e.g. same-size dtypes) | mainline `TypecastShardedProgramFactory` — not hit by fp32<->bf16/bfp8 (tile sizes differ) |
| llama `tests/prototype_ops/test_typecast.py` `experimental.quasar.typecast` | interleaved | qsr typecast factories are **ProgramDescriptor (legacy)** — not Metal2 |
| qwen `ttnn.zeros_like` | | full_like_impl -> `prim::full` (full/device factories) |

## Findings

### F1 CONFIRMED — qsr transpose WH sharded-RM: borrowed output DFB self-loop pushed, never popped (cat 4/5)
- Factory: `experimental/quasar/transpose/device/transpose_wh_sharded_rm_program_factory.cpp:113-119` (CB_OUT0 borrowed_from OUTPUT when `ht<=8`), bindings `:173-177` bind compute as BOTH PRODUCER and CONSUMER of CB_OUT0; no writer kernel in the work unit when `ht<=8` (`:211-232`). (Header comment of the kernel says "PRODUCER-only" — stale; the factory binds the self-loop.)
- Kernel: `experimental/quasar/transpose/device/kernels/compute/transpose_wh_rm_sharded.cpp`
  - non-narrow: `:93 reserve_back(Ht)` / `:106 push_back(Ht)` / `:110 wait_front(Ht)` per w — **no pop_front anywhere**.
  - narrow (`last_output_row_num_datums<32`, i.e. H%32!=0): `:53/57` and `:59/63` reserve/push `pack_num_pages_last_row_col` / `pack_num_pages_last_col` per w — **no wait/pop**.
- Counts per core: pushes = num_hw_blocks_per_core * Wt * Ht (non-narrow) or num_hw_blocks * ((Wt-1)*pack_num_pages_last_col + pack_num_pages_last_row_col) (narrow); pops = 0. With finish(): TRISC pack/unpack drain waits for posted==0 -> hang at kernel end.
- Exercised: YES — `models/demos/vision/classification/resnet50/quasar/tests/ops/test_fold_transpose.py` configs b1/b2_wh_4x256x224_aligned (ht=8, Wt=7, non-narrow), b1_wh_out16_narrow (H=16, narrow, Wt=8), b1_wh_out8_narrow (H=8). This is also the real resnet fold transpose(2,3) path.
- Fix (minimal): pop what was pushed, right after each push (data is already in the borrowed output shard; pop only advances the ring): non-narrow replace `cb_out_buf.wait_front(Ht);` with `cb_out_buf.wait_front(Ht); cb_out_buf.pop_front(Ht);`; narrow add `wait_front(n); pop_front(n)` after each of the two push_backs. Alternative: drop the CONSUMER self-binding and add a DM writer that does `wait_front(total); pop_front(total)` (mirrors `writer_unary_sharded.cpp`).

### F2 SUSPECT (latent, pre-existing) — qsr transpose WH sharded-RM, `ht>8` + narrow-row: cross-kernel count mismatch (cat 3)
- Compute narrow path pushes per block `(Wt-1)*pack_num_pages_last_col + pack_num_pages_last_row_col` entries to CB_OUT_STAGE (`transpose_wh_rm_sharded.cpp:53-63`), writer `writer_unary_transpose_wh_sharded_rm.cpp` (Ht>8 branch) waits/pops `Ht` per w => Wt*Ht per block. E.g. W%32==0,H%32!=0: pack_num_pages_last_col=1 -> pushes Wt vs pops Wt*Ht (Ht>=9). Writer would hang even without finish().
- Exercised: NO (all test H <= 256 -> ht<=8). Triggers only for H>256 with H%32!=0.
- Fix: in the ht>8 case force the non-narrow compute path (or make the writer wait per-w for the narrow page count).

### F3 CONFIRMED — qsr transpose HC sharded-RM generic path: reader self-loop cb_out pushed, never popped (cat 4/5)
- Factory: `experimental/quasar/transpose/device/transpose_hc_sharded_program_factory.cpp:503-511` binds reader as Producer+Consumer of CB_IN and CB_OUT (both borrowed shards), single kernel.
- Kernel: `experimental/quasar/transpose/device/kernels/dataflow/reader_unary_transpose_hc_sharded_rm.cpp:123 reserve_back(num_sticks_per_core)`, `:177 push_back(num_sticks_per_core)`; no wait/pop. CB_IN: no ops (0/0, fine).
- Counts: CB_OUT posted = num_sticks_per_core, acked = 0 -> DM finish() (read_acked != read_posted) spins.
- Exercised: only when `is_special_case` is false (`:337-341`: shard_height vs H/C divisibility or shard_height > C*H). test_fold_transpose b1_hc_4x224x256 (H=224,C=4) is special-case for typical core counts (shard rows dividing 896) -> likely NOT hit; SUSPECT for odd core counts. Special-case path (`#ifdef USE_SPECIAL_CASE`) does no DFB ops at all -> balanced.
- Fix: after `cb_out.push_back(num_sticks_per_core);` add `cb_out.wait_front(num_sticks_per_core); cb_out.pop_front(num_sticks_per_core);`.

### F4 CONFIRMED (code) / not exercised by tests — mainline typecast sharded: compute self-loop output DFB pushed, never popped (cat 4/5)
- Factory: `copy/typecast/device/typecast_sharded_program_factory.cpp:108-116` (OUT borrowed_from OUTPUT), `:184-189` compute bound PRODUCER+CONSUMER of OUT, no writer kernel (`spec.kernels = {reader, compute}`).
- Kernel: `copy/typecast/device/kernels/compute/eltwise_typecast.cpp` — `ckl::unary` with `ReservePolicy::PerOuter/PushPolicy::PerOuter`, per_core_block_cnt=1, per_core_block_dim=num_tile_per_core -> pushes num_tile_per_core into OUT; nothing pops.
- Counts: OUT posted=num_tile_per_core, popped=0 -> hang in finish().
- Exercised: requires sharded L1 input with equal input/output tile sizes (`typecast_device_op.cpp:16-40`). llama test_typecast is interleaved; model casts fp32<->bf16 / bf16->bfp8 have different tile sizes -> falls to `TypecastProgramFactory`. NOT exercised by these tests (latent; would hit e.g. bf16<->uint16 on sharded).
- Fix: add a DM consumer (e.g. `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp`, which does wait_front(n)+pop_front(n)) bound as CONSUMER of OUT and drop the compute CONSUMER self-binding; or append `DataflowBuffer o(dfb::out); o.wait_front(n); o.pop_front(n);` at the end of the compute kernel for the sharded variant (needs an RTA/CTA for n).
- Same pattern in the legacy qsr copy `experimental/quasar/typecast/device/typecast_sharded_program_factory.cpp` (ProgramDescriptor; out CB pushed by `experimental/quasar/typecast/device/kernels/compute/eltwise_typecast.cpp:21/42`, no consumer) — not Metal2, so not Quasar-reachable as-is.

### F5 LATENT (dead define) — qsr transpose writer OUT_SHARDED wait without pop (cat 2)
- `experimental/quasar/transpose/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` `#ifdef OUT_SHARDED: cb.wait_front(num_pages);` with no pop_front (mainline metal2 fork has the pop). No qsr transpose factory defines OUT_SHARDED -> unreachable. Fix if ever enabled: add `cb.pop_front(num_pages);`.

### Note (not an imbalance) — producer-only scratch DFB with get_write_ptr and no reserve/push
- qsr/mainline `TransposeHCTiledProgramFactory` (partitioned) SCRATCH_CB (`experimental/quasar/transpose/device/transpose_hc_tiled_program_factory.cpp:130`, kernel `reader_unary_transpose_hc_interleaved_partitioned.cpp:55-59`): 0 pushes / 0 pops -> counters balanced. Factory is not selected by either select_program_factory (MULTI_CORE_HC -> sharded / RM / TiledInterleaved only), i.e. dead.
- Sharded readers `reader_unary_sharded*.cpp` push_back(n) without reserve_back on borrowed input DFBs; consumer pops n -> counters balanced (push-without-reserve is intentional for borrowed shards).

## Balanced (checked)

Transpose (mainline, Metal2):
- `TransposeWHProgramFactory` tiled: reader_unary_transpose_wh_interleaved_start_id.cpp push num_tiles / transpose_wh_metal2.cpp wait+pop 1, reserve+push 1 x NHtWt(=num_tiles) / writer_unary_interleaved_start_id_metal2.cpp pop num_pages(=num_tiles). OK.
- `TransposeWHProgramFactory` RM: reader_..._wh_..._rm.cpp push Wt x Ht per block / transpose_wh_rm_metal2.cpp tilize helper (wait/pop Wt x Ht, cb_tilize push Wt x Ht) + cb_tilize self-loop wait/pop HtWt + out push Ht x Wt / writer_..._wh_..._rm.cpp pop Ht x Wt. OK.
- `TransposeWHShardedProgramFactory`: reader_unary_sharded_metal2 push NHtWt / transpose_wh_sharded_metal2 wait/pop NHtWt, reserve/push NHtWt (=N*Ht*Wt, factory :146-156) / writer_unary_sharded_metal2 wait/pop num_units(=NHtWt). OK.
- `TransposeHCTiledInterleavedProgramFactory`: reader ..._padding_aware_metal2 push num_tiles + 1 pad (NEEDS_PADDING) / writer ..._padding_aware pop (end-start)=num_tiles + 1 pad. OK (both unconditional per core).
- `TransposeHCRMProgramFactory`: reader/writer partitioned_rm / start_id_rm iterate num_sticks_per_core_read x num_read_per_barrier with identical RTAs. OK.
- `TransposeCNProgramFactory`: reader/writer cn num_pages each. OK.
- `TransposeHCShardedProgramFactory`, `TransposeWHShardedRMProgramFactory` (mainline): ProgramDescriptor legacy -> not Quasar-reachable (not audited further).
Transpose (experimental/quasar):
- `TransposeWHProgramFactory` tiled + RM (transpose_wh.cpp, transpose_wh_rm.cpp + its reader/writer): same counts as mainline. OK.
- `TransposeWHShardedProgramFactory` (reader_unary_sharded / transpose_wh_sharded / writer_unary_sharded): NHtWt each. OK.
- `TransposeHCTiledInterleavedProgramFactory` (qsr reader/writer padding_aware): num_tiles + 1 pad each side. OK.
- `TransposeHCRMProgramFactory`, `TransposeCNProgramFactory`: OK.
- `TransposeHCShardedProgramFactory` special-case path: no DFB ops. OK.
Permute (mainline, Metal2):
- `MultiCoreRowInvariant`: reader/writer row_invariant push/pop 1 per row over [start_row,end_row). OK.
- `MultiCoreBlockedGeneric`: reader push x_block_size per block / transpose_xw_rm_single_tile_size: tilize asymmetric (wait/pop x_block_size, cb_tilize push 1), cb_tilize wait/pop 1, out push w_block_size / writer wait/pop w_block_size per block. OK.
- `MultiCoreTileInvariant`: reader_permute_..._tiled_invariant push 1 per tile / (swap_hw) transpose_wh_metal2 NHtWt / writer_unary_interleaved_start_id_metal2 num_pages — all = num_tiles_per_core. OK.
- `MultiCoreTileRowInvariant`: mainline padding_aware_metal2 reader (num_tiles + 1 pad) / writer_permute_..._tiled_row_invariant (end-start + 1 pad). OK.
- `MultiCoreTiledGeneric`: reader_permute_..._tiled_generic push 1 per block + 1 Y-pad / transpose_xw_tiled (tilize 1/1, self-loop 1/1, out push 1 per block) / writer_permute_..._tiled_generic pop 1 per block + 1 Y-pad. OK.
Reshape:
- mainline `ReshapeViewTiledProgramFactory` reader_reshape_tiled.cpp / writer_reshape_tiled.cpp and qsr `ReshapeViewTiledMetalV2ProgramFactory` reader/writer_reshape_tiled_metal2.cpp: mapping 1/1 per output page; input tiles: reader pushes on first segment of each output page + each index change, writer pops with identical dedup logic (previous reset to MAX per page) + final pop gated on !first. OK.
- mainline `ReshapeViewRMProgramFactory` rm_reshape_interleaved.cpp and qsr `ReshapeViewRMMetalV2ProgramFactory` rm_reshape_interleaved_metal2.cpp: Scratchpads only, no DFB. OK. (ttnn.view is host-only.)
Typecast (mainline, Metal2):
- `TypecastProgramFactory` (interleaved and non-optimized sharded): reader/writer num_pages = per_core_block_cnt per group. OK.
- `TypecastSubgridProgramFactory`: uniform ntiles_per_core on all three kernels. OK.
- `TypecastRowMajorChunkedProgramFactory`: reader/writer num_rows*(full+partial) chunks, compute per_core_block_cnt = rows*chunks_per_row_total (factory :186,249,254). OK incl. tail chunk.
- `TypecastShardedProgramFactory`: IN 1/1 OK; OUT -> F4.
- experimental/quasar typecast: ProgramDescriptor legacy factories (interleaved/subgrid/rm_chunked balanced by inspection; sharded = same issue as F4) — not Metal2.
Full / fill:
- full/device writer_full.cpp, writer_full_sharded.cpp, writer_full_nd_sharded.cpp: self-loop `value` DFB reserve/push 1 then wait/pop 1, no early return between. OK (used by zeros_like via full_like_impl -> prim::full).
- fill_rm_interleaved.cpp: Scratchpads only. OK.
Kernel-lib helpers checked: `kernel_lib/tilize_helpers.inl` (WaitBlock/WaitUpfront: pops sum to waited total; push = reserve = block_width per block), `kernel_lib/untilize_helpers.inl` (all three paths reserve/push and wait/pop matched).

---

## (a) COLLECTED: ops used by tests (call counts)
```
== models/demos/vision/classification/resnet50/quasar
    120 ttnn.from_torch
     28 ttnn.experimental.quasar.to_memory_config
     18 ttnn.experimental.quasar.reallocate
     11 ttnn.experimental.quasar.reshape
      9 ttnn.experimental.quasar.to_layout
      8 ttnn.experimental.quasar.tilize
      6 ttnn.experimental.quasar.untilize_with_unpadding
      6 ttnn.experimental.quasar.to_device
      5 ttnn.tilize
      2 ttnn.experimental.quasar.transpose
== models/experimental/llama32_1b_quasar
    204 ttnn.from_torch
     56 ttnn.typecast
     56 ttnn.to_memory_config
     51 ttnn.reshape
     38 ttnn.tilize
     35 ttnn.untilize
     35 ttnn.slice
     35 ttnn.interleaved_to_sharded
     34 ttnn.concat
     29 ttnn.embedding
     25 ttnn.sharded_to_interleaved
     22 ttnn.pad
     19 ttnn.to_layout
     19 ttnn.to_device
     18 ttnn.experimental.quasar.tilize
     16 ttnn.transpose
     14 ttnn.split
     12 ttnn.fill_cache
      9 ttnn.copy_host_to_device_tensor
      7 ttnn.experimental.quasar.interleaved_to_sharded
      6 ttnn.experimental.quasar.untilize
      3 ttnn.experimental.quasar.to_memory_config
      2 ttnn.experimental.quasar.to_device
      2 ttnn.experimental.quasar.tilize_with_val_padding
      1 ttnn.zeros
      1 ttnn.untilize_with_unpadding
      1 ttnn.tilize_with_val_padding
      1 ttnn.experimental.quasar.typecast
      1 ttnn.experimental.quasar.transpose
      1 ttnn.experimental.quasar.tilize_with_zero_padding
      1 ttnn.experimental.quasar.slice
      1 ttnn.experimental.quasar.sharded_to_interleaved
      1 ttnn.experimental.quasar.reshape
      1 ttnn.experimental.quasar.pad
      1 ttnn.clone
== models/experimental/ops/quasar/tests/qwen3_vl_ops
     24 ttnn.reshape
     21 ttnn.to_memory_config
      8 ttnn.typecast
      8 ttnn.to_device
      7 ttnn.to_layout
      7 ttnn.embedding
      6 ttnn.concat
      5 ttnn.untilize
      5 ttnn.split
      5 ttnn.interleaved_to_sharded
      4 ttnn.zeros_like
      4 ttnn.transpose
      4 ttnn.slice
      4 ttnn.sharded_to_interleaved
      4 ttnn.expand
      3 ttnn.pad
      2 ttnn.from_torch
      1 ttnn.untilize_with_unpadding
      1 ttnn.tilize_with_val_padding
      1 ttnn.tilize
== models/experimental/ops/quasar/tests/yolo_ops
     10 ttnn.to_memory_config
     10 ttnn.split
     10 ttnn.slice
     10 ttnn.permute
     10 ttnn.interleaved_to_sharded
     10 ttnn.concat
      9 ttnn.reshape
      8 ttnn.sharded_to_interleaved
      8 ttnn.clone
      6 ttnn.to_layout
      5 ttnn.reshard
      5 ttnn.from_torch
      4 ttnn.pad
      3 ttnn.reallocate
```

## Metal2/DFB program factories in scope and their kernels (Quasar-buildable)
```
## data_movement/clone/device/clone_program_factory.cpp
"data_movement/clone/device/kernels/compute_kernel.cpp"
"data_movement/clone/device/kernels/read_kernel.cpp"
"data_movement/clone/device/kernels/read_kernel_rm.cpp"
"data_movement/clone/device/kernels/read_kernel_rm_sharded.cpp"
"data_movement/clone/device/kernels/read_kernel_sharded.cpp"
"data_movement/clone/device/kernels/write_kernel.cpp"
"data_movement/clone/device/kernels/write_kernel_rm.cpp"
"data_movement/clone/device/kernels/write_kernel_rm_sharded.cpp"
"data_movement/clone/device/kernels/write_kernel_sharded.cpp"
## data_movement/split/device/split_program_factory.cpp
"reader_tm_tile_layout_split_two_chunks.cpp"
"writer_split_n_chunks_tile.cpp"
## data_movement/untilize_with_unpadding/device/factories/untilize_with_unpadding_single_core_program_factory.cpp
"reader_unary_interleaved_start_id_metal2.cpp"
"data_movement/untilize/device/kernels/compute/untilize_metal2.cpp"
"writer_unary_unpad_dims_split_rows.cpp"
## data_movement/fill_rm/device/fill_rm_program_factory.cpp
"data_movement/fill_rm/device/kernels/dataflow/fill_rm_interleaved.cpp"
## data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_block_interleaved_program_factory.cpp
"reader_unary_pad_multicore_both_dims.cpp"
"data_movement/tilize/device/kernels/compute/tilize_wh.cpp"
"writer_unary_interleaved_start_id_wh.cpp"
## data_movement/untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_nd_sharded_program_factory.cpp
"reader_unary_nd_sharded_blocks_metal2.cpp"
"untilize_variable_num_blocks_metal2.cpp"
"writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp"
## data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_default_program_factory.cpp
"reader_unary_pad_dims_split_rows_multicore.cpp"
"ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp"
"writer_unary_interleaved_start_id_metal2.cpp"
## data_movement/pad/device/pad_rm_reader_writer_multi_core_default_program_factory.cpp
"reader_pad_dims_rm_interleaved_v2.cpp"
"writer_pad_dims_rm_interleaved_v2.cpp"
## data_movement/untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp
"reader_unary_interleaved_wh_multicore_metal2.cpp"
"data_movement/untilize/device/kernels/compute/untilize_wh_metal2.cpp"
"writer_unary_stick_layout_wh_multicore_metal2.cpp"
## data_movement/pad/device/pad_rm_reader_writer_multi_core_program_factory.cpp
"reader_pad_dims_rm_interleaved.cpp"
"writer_pad_dims_rm_interleaved.cpp"
## data_movement/untilize/device/factories/untilize_multi_core_sub_core_grids_program_factory.cpp
"data_movement/untilize/device/kernels/compute/untilize_metal2.cpp"
"eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp"
"writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp"
## data_movement/pad/device/pad_tile_program_factory.cpp
"reader_unary_interleaved_start_id_metal2.cpp"
"writer_unary_pad_dims_interleaved.cpp"
## data_movement/pad/device/pad_rm_sharded_height_only_program_factory.cpp
"data_movement/pad/device/kernels/dataflow/reader_pad_dims_rm_sharded.cpp"
"data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_sharded.cpp"
## data_movement/pad/device/pad_tile_multicore_program_factory.cpp
"data_movement/pad/device/kernels/dataflow/reader_pad_tiled.cpp"
"data_movement/pad/device/kernels/dataflow/writer_pad_tiled.cpp"
## data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp
"data_movement/untilize/device/kernels/dataflow/reader_unary_sharded_blocks.cpp"
"data_movement/untilize/device/kernels/dataflow/reader_unary_start_id_metal2.cpp"
"eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp"
"untilize_variable_num_blocks_metal2.cpp"
"writer_unary_stick_layout_split_rows_multi_core.cpp"
## data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_sharded_program_factory.cpp
"reader_unary_pad_height_width_sharded.cpp"
"ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp"
"writer_unary_sharded_metal2.cpp"
## data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_single_core_program_factory.cpp
"reader_unary_pad_dims_split_rows.cpp"
"ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp"
"writer_unary_interleaved_start_id_metal2.cpp"
## data_movement/slice/device/slice_program_factory_tile.cpp
"reader_unary_unpad_dims_interleaved_start_id.cpp"
"writer_unary_interleaved_start_id.cpp"
## data_movement/pad/device/pad_rm_sharded_width_only_program_factory.cpp
"reader_pad_dims_rm_sharded_stickwise.cpp"
"writer_pad_dims_rm_sharded_stickwise.cpp"
## data_movement/untilize/device/factories/untilize_multi_core_block_program_factory.cpp
"reader_unary_interleaved_wh_multicore_metal2.cpp"
"data_movement/untilize/device/kernels/compute/untilize_wh_metal2.cpp"
"writer_unary_stick_layout_wh_multicore_metal2.cpp"
## data_movement/permute/device/permute_rm_program_factory.cpp
"reader_permute_interleaved_rm_blocked_generic.cpp"
"reader_permute_interleaved_rm_row_invariant.cpp"
"transpose_xw_rm_single_tile_size.cpp"
"writer_permute_interleaved_rm_blocked_generic.cpp"
"writer_permute_interleaved_rm_row_invariant.cpp"
## data_movement/untilize/device/factories/untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.cpp
"data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp"
"data_movement/untilize/device/kernels/compute/untilize_metal2.cpp"
"eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp"
## data_movement/untilize/device/factories/untilize_multi_core_nd_shard_input_program_factory.cpp
"reader_unary_nd_sharded_blocks_metal2.cpp"
"untilize_variable_num_blocks_metal2.cpp"
"writer_unary_stick_layout_split_rows_multi_core_nd_shard.cpp"
## data_movement/pad/device/pad_rm_reader_writer_program_factory.cpp
"reader_pad_dims_rm_interleaved.cpp"
"writer_pad_dims_rm_interleaved.cpp"
## data_movement/slice/device/slice_program_factory_tile_tensor_args.cpp
"reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp"
"writer_unary_interleaved_start_id_metal2.cpp"
## data_movement/permute/device/permute_tiled_program_factory.cpp
"reader_permute_interleaved_tiled_generic.cpp"
"reader_permute_interleaved_tiled_invariant.cpp"
"reader_unary_transpose_hc_interleaved_tiled_padding_aware_metal2.cpp"
"data_movement/permute/device/kernels/compute/transpose_xw_tiled.cpp"
"data_movement/transpose/device/kernels/compute/transpose_wh_metal2.cpp"
"writer_permute_interleaved_tiled_generic.cpp"
"writer_permute_interleaved_tiled_row_invariant.cpp"
"writer_unary_interleaved_start_id_metal2.cpp"
## data_movement/slice/device/slice_program_factory_rm_sharded.cpp
"slice_reader_unary_unpad_dims_rm_sharded.cpp"
## data_movement/untilize/device/factories/untilize_single_core_program_factory.cpp
"data_movement/untilize/device/kernels/compute/untilize_metal2.cpp"
"data_movement/untilize/device/kernels/dataflow/reader_unary_start_id_metal2.cpp"
"writer_unary_stick_layout_split_rows_single_core.cpp"
## data_movement/untilize/device/factories/untilize_multi_core_parallelize_column_program_factory.cpp
"data_movement/untilize/device/kernels/compute/untilize_metal2.cpp"
"eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp"
"writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp"
## data_movement/copy/device/copy_default_row_major_program_factory.cpp
"data_movement/copy/device/kernels/redistribute_pages_row_major_parallel_reader.cpp"
"data_movement/copy/device/kernels/redistribute_pages_row_major_parallel_writer.cpp"
"data_movement/copy/device/kernels/redistribute_pages_row_major_reader.cpp"
"data_movement/copy/device/kernels/redistribute_pages_row_major_writer.cpp"
## data_movement/slice/device/slice_program_factory_rm.cpp
"slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp"
"slice_writer_unary_stick_layout_interleaved_start_id.cpp"
## data_movement/transpose/device/transpose_hc_tiled_program_factory.cpp
"reader_unary_transpose_hc_interleaved_partitioned.cpp"
"writer_unary_interleaved_start_id_metal2.cpp"
## data_movement/transpose/device/transpose_hc_tiled_interleaved_program_factory.cpp
"reader_unary_transpose_hc_interleaved_tiled_padding_aware_metal2.cpp"
"writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp"
## data_movement/transpose/device/transpose_hc_rm_program_factory.cpp
"reader_unary_transpose_hc_interleaved_partitioned_rm.cpp"
"writer_unary_transpose_hc_interleaved_start_id_rm.cpp"
## data_movement/sharded/reshard/device/reshard_program_factory_same_width.cpp
"data_movement/sharded/device/kernels/dataflow/reshard_same_width_reader.cpp"
"data_movement/sharded/device/kernels/dataflow/reshard_same_width_writer.cpp"
## data_movement/transpose/device/transpose_wh_sharded_program_factory.cpp
"transpose_wh_sharded_metal2.cpp"
"data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp"
"eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp"
## data_movement/tilize/device/tilize_single_core_program_factory.cpp
"reader_unary_stick_layout_split_rows_singlecore.cpp"
"ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp"
"writer_unary_interleaved_start_id_metal2.cpp"
## data_movement/tilize/device/tilize_multi_core_retile_program_factory.cpp
"reader_unary_start_id_metal2.cpp"
"data_movement/tilize/device/kernels/compute/retile.cpp"
"writer_unary_interleaved_start_id_metal2.cpp"
## data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp
"ttnn/cpp/ttnn/kernel/compute/eltwise_copy_metal2.cpp"
"eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp"
"writer_unary_sharded_blocks_interleaved_start_id_metal2.cpp"
"writer_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2.cpp"
## data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_local.cpp
"data_movement/sharded/reshard/device/kernels/nd_reshard_copy_local_shards.cpp"
## data_movement/repeat/device/repeat_program_factory_last_dim.cpp
"data_movement/repeat/device/kernels/repeat_last_dim_rm_interleaved.cpp"
"data_movement/repeat/device/kernels/repeat_last_dim_rm_sharded.cpp"
## data_movement/concat/device/concat_s2s_tiled_program_factory.cpp
"height_sharded_width_concat_two_tensors.cpp"
"reader_height_sharded_width_concat_two_tensors_tiled.cpp"
"writer_height_sharded_width_concat_two_tensors_tiled.cpp"
## data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_pages.cpp
"data_movement/sharded/reshard/device/kernels/nd_reshard_copy_pages_reader.cpp"
"data_movement/sharded/reshard/device/kernels/nd_reshard_copy_pages_writer.cpp"
## data_movement/tilize/device/tilize_multi_core_block_program_factory.cpp
"reader_unary_pad_multicore_both_dims.cpp"
"data_movement/tilize/device/kernels/compute/tilize_wh.cpp"
"writer_unary_interleaved_start_id_wh.cpp"
## data_movement/sharded/reshard/device/reshard_program_factory_generic.cpp
"data_movement/sharded/device/kernels/dataflow/reshard_reader.cpp"
"data_movement/sharded/device/kernels/dataflow/reshard_reader_diff_width.cpp"
## experimental/quasar/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp
"eltwise_copy.cpp"
"reader_unary_sharded_blocks_interleaved_start_id.cpp"
"reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp"
"writer_unary_sharded.cpp"
"writer_unary_sharded_blocks_start_id.cpp"
"writer_unary_sharded_stick_layout_start_id.cpp"
## experimental/quasar/pad/device/pad_tile_program_factory.cpp
"reader_unary_interleaved_start_id.cpp"
"writer_unary_pad_dims_interleaved.cpp"
## experimental/quasar/move/device/move_overlap_program_factory.cpp
"move_interleaved_with_overlap.cpp"
"move_interleaved_with_overlap_writer.cpp"
"move_stick_layout_interleaved_with_overlap.cpp"
"move_stick_layout_interleaved_with_overlap_writer.cpp"
## experimental/quasar/pad/device/pad_rm_reader_writer_program_factory.cpp
"reader_pad_dims_rm_interleaved_sc.cpp"
"writer_pad_dims_rm_interleaved_sc.cpp"
## data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp
"reader_unary_sharded_blocks_interleaved_start_id_metal2.cpp"
"reader_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2.cpp"
"data_movement/sharded/device/kernels/compute/eltwise_copy_metal2.cpp"
"writer_unary_sharded_blocks_start_id_metal2.cpp"
"writer_unary_sharded_metal2.cpp"
"writer_unary_sharded_stick_layout_start_id_metal2.cpp"
## experimental/quasar/reshape_view/device/reshape_tiled_metal2_program_factory.cpp
"reader_reshape_tiled_metal2.cpp"
"writer_reshape_tiled_metal2.cpp"
## data_movement/repeat/device/repeat_program_factory_higher_dim.cpp
"data_movement/repeat/device/kernels/repeat_higher_dim_rm_interleaved.cpp"
"data_movement/repeat/device/kernels/repeat_higher_dim_rm_sharded.cpp"
"data_movement/repeat/device/kernels/repeat_higher_dim_tile.cpp"
## data_movement/tilize/device/tilize_multi_core_sharded_retile_program_factory.cpp
"data_movement/tilize/device/kernels/compute/retile.cpp"
"eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp"
"writer_unary_interleaved_start_id_metal2.cpp"
"writer_unary_sharded_metal2.cpp"
## data_movement/concat/device/concat_s2s_rm_program_factory.cpp
"reader_height_sharded_width_concat_two_tensors.cpp"
## experimental/quasar/move/device/move_sharded_program_factory.cpp
"reader_unary_local_l1_copy_backwards.cpp"
## experimental/quasar/pad/device/pad_tile_multicore_program_factory.cpp
"experimental/quasar/pad/device/kernels/dataflow/reader_pad_tiled.cpp"
"experimental/quasar/pad/device/kernels/dataflow/writer_pad_tiled.cpp"
## experimental/quasar/reshard/device/reshard_program_factory_same_width.cpp
"experimental/quasar/reshard/device/kernels/dataflow/reshard_same_width_reader.cpp"
"experimental/quasar/reshard/device/kernels/dataflow/reshard_same_width_writer.cpp"
## experimental/quasar/pad/device/pad_rm_sharded_width_only_program_factory.cpp
"reader_pad_dims_rm_sharded_stickwise.cpp"
"writer_pad_dims_rm_sharded_stickwise.cpp"
## experimental/quasar/tilize/device/tilize_multi_core_block_program_factory.cpp
"reader_unary_pad_multicore_both_dims_metal2.cpp"
"experimental/quasar/tilize/device/kernels/compute/tilize_wh.cpp"
"writer_unary_interleaved_start_id_wh.cpp"
## experimental/quasar/reshape_view/device/reshape_rm_metal2_program_factory.cpp
"rm_reshape_interleaved_metal2.cpp"
## experimental/quasar/pad/device/pad_rm_sharded_height_only_program_factory.cpp
"experimental/quasar/pad/device/kernels/dataflow/reader_pad_dims_rm_sharded.cpp"
"experimental/quasar/pad/device/kernels/dataflow/writer_pad_dims_rm_sharded.cpp"
## experimental/quasar/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp
"eltwise_copy.cpp"
"reader_unary_sharded.cpp"
"writer_unary_sharded_blocks_interleaved_start_id.cpp"
"writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp"
## experimental/quasar/reshard/device/reshard_program_factory_generic.cpp
"reshard_reader.cpp"
"reshard_reader_diff_width.cpp"
## experimental/quasar/tilize/device/tilize_single_core_program_factory.cpp
"reader_unary_stick_layout_split_rows_singlecore.cpp"
"experimental/quasar/tilize/device/kernels/compute/tilize.cpp"
"writer_unary_interleaved_start_id.cpp"
## experimental/quasar/slice/device/slice_program_factory_tile_tensor_args.cpp
"reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp"
"writer_unary_interleaved_start_id.cpp"
## experimental/quasar/tilize/device/tilize_multi_core_sharded_program_factory.cpp
"experimental/quasar/tilize/device/kernels/compute/tilize.cpp"
"experimental/quasar/tilize/device/kernels/dataflow/reader_unary_sharded.cpp"
"experimental/quasar/tilize/device/kernels/dataflow/writer_unary_sharded.cpp"
## experimental/quasar/tilize/device/tilize_device_operation.cpp
## experimental/quasar/pad/device/pad_rm_reader_writer_multi_core_default_program_factory.cpp
"reader_pad_dims_rm_interleaved_v2.cpp"
"writer_pad_dims_rm_interleaved_v2.cpp"
## experimental/quasar/transpose/device/transpose_wh_sharded_rm_program_factory.cpp
"reader_unary_transpose_wh_sharded_rm.cpp"
"transpose_wh_rm_sharded.cpp"
"writer_unary_transpose_wh_sharded_rm.cpp"
## experimental/quasar/slice/device/slice_program_factory_tile.cpp
"reader_unary_unpad_dims_interleaved_start_id.cpp"
"writer_unary_interleaved_start_id.cpp"
## experimental/quasar/transpose/device/transpose_hc_tiled_interleaved_program_factory.cpp
"reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp"
"writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp"
## experimental/quasar/reshard/device/reshard_program_factory_same_height.cpp
"experimental/quasar/reshard/device/kernels/dataflow/reshard_same_height_reader.cpp"
"experimental/quasar/reshard/device/kernels/dataflow/reshard_same_height_writer.cpp"
## experimental/quasar/tilize_with_val_padding/device/factories/tilize_with_val_padding_single_core_program_factory.cpp
"reader_unary_pad_dims_split_rows.cpp"
"tilize_metal2.cpp"
"writer_unary_interleaved_start_id_metal2.cpp"
## experimental/quasar/transpose/device/transpose_hc_sharded_program_factory.cpp
"reader_unary_transpose_hc_sharded_rm.cpp"
"writer_unary_transpose_hc_sharded_rm.cpp"
## experimental/quasar/tilize/device/tilize_multi_core_default_program_factory.cpp
"reader_unary_stick_layout_split_rows_multicore.cpp"
"experimental/quasar/tilize/device/kernels/compute/tilize.cpp"
"writer_unary_interleaved_start_id.cpp"
## experimental/quasar/slice/device/slice_program_factory_rm.cpp
"slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp"
"slice_writer_unary_stick_layout_interleaved_start_id.cpp"
## experimental/quasar/transpose/device/transpose_hc_tiled_program_factory.cpp
"reader_unary_transpose_hc_interleaved_partitioned.cpp"
"writer_unary_interleaved_start_id.cpp"
## experimental/quasar/untilize/device/factories/untilize_multi_core_sub_core_grids_program_factory.cpp
"reader_unary_interleaved_start_id_metal2.cpp"
"experimental/quasar/untilize/device/kernels/compute/untilize_metal2.cpp"
"writer_unary_stick_layout_split_rows_interleaved_parallel_columns_metal2.cpp"
## experimental/quasar/tilize/device/tilize_multi_core_width_sharded_program_factory.cpp
"experimental/quasar/tilize/device/kernels/compute/tilize.cpp"
"experimental/quasar/tilize/device/kernels/dataflow/reader_unary_sharded.cpp"
"experimental/quasar/tilize/device/kernels/dataflow/writer_unary_sharded.cpp"
## experimental/quasar/untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp
"eltwise_copy.cpp"
"reader_unary_sharded.cpp"
"untilize.cpp"
"writer_unary_stick_layout_interleaved_blocks.cpp"
"writer_unary_unpad_batch_rows_sharded.cpp"
"writer_unary_unpad_width_16_sharded.cpp"
## experimental/quasar/reshard/device/nd_reshard_program_factory_copy_local.cpp
"experimental/quasar/reshard/device/kernels/nd_reshard_copy_local_shards.cpp"
## experimental/quasar/tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_default_program_factory.cpp
"reader_unary_pad_dims_split_rows_multicore.cpp"
"tilize_metal2.cpp"
"writer_unary_interleaved_start_id_metal2.cpp"
## embedding/device/embeddings_fused_program_factory.cpp
"ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp"
"embedding/device/kernels/compute/tilize_chunked.cpp"
"embedding/device/kernels/dataflow/embeddings_tilize.cpp"
"writer_unary_interleaved_start_id_metal2.cpp"
## experimental/quasar/untilize/device/factories/untilize_single_core_program_factory.cpp
"reader_unary_start_id_metal2.cpp"
"experimental/quasar/untilize/device/kernels/compute/untilize_metal2.cpp"
"writer_unary_stick_layout_split_rows_single_core.cpp"
## experimental/quasar/untilize/device/factories/untilize_multi_core_parallelize_column_program_factory.cpp
"reader_unary_interleaved_start_id_metal2.cpp"
"experimental/quasar/untilize/device/kernels/compute/untilize_metal2.cpp"
"writer_unary_stick_layout_split_rows_interleaved_parallel_columns_metal2.cpp"
## experimental/quasar/untilize/device/factories/untilize_multi_core_program_factory.cpp
"reader_unary_sharded_blocks_metal2.cpp"
"reader_unary_sharded_metal2.cpp"
"reader_unary_start_id_metal2.cpp"
"untilize_variable_num_blocks_metal2.cpp"
"writer_unary_stick_layout_split_rows_multi_core_metal2.cpp"
## experimental/quasar/transpose/device/transpose_hc_rm_program_factory.cpp
"reader_unary_transpose_hc_interleaved_partitioned_rm.cpp"
"writer_unary_transpose_hc_interleaved_start_id_rm.cpp"
## full/device/full_program_factory_interleaved.cpp
"full/device/kernels/writer_full.cpp"
## experimental/quasar/untilize/device/factories/untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.cpp
"compute/untilize_variable_num_blocks_metal2.cpp"
"dataflow/reader_unary_sharded_metal2.cpp"
"dataflow/writer_unary_sharded_metal2.cpp"
## data_movement/reshape_view/device/reshape_tiled_program_factory.cpp
"data_movement/reshape_view/device/device/dataflow/reader_reshape_tiled.cpp"
"data_movement/reshape_view/device/device/dataflow/writer_reshape_tiled.cpp"
## data_movement/reshape_view/device/reshape_rm_program_factory.cpp
"data_movement/reshape_view/device/device/rm_reshape_interleaved.cpp"
## experimental/quasar/slice/device/slice_program_factory_rm_sharded.cpp
"slice_reader_unary_unpad_dims_rm_sharded.cpp"
## experimental/quasar/untilize/device/factories/untilize_multi_core_nd_shard_input_program_factory.cpp
"compute/untilize_variable_num_blocks_metal2.cpp"
"dataflow/reader_unary_nd_sharded_blocks_metal2.cpp"
"dataflow/writer_unary_stick_layout_split_rows_multi_core_nd_shard_metal2.cpp"
## experimental/quasar/untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_interleaved_program_factory.cpp
"reader_unary_interleaved_start_id.cpp"
"untilize_metal2.cpp"
"writer_unary_stick_layout_split_rows_multicore.cpp"
## data_movement/untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp
"ttnn/cpp/ttnn/kernel/compute/eltwise_copy_metal2.cpp"
"ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks_metal2.cpp"
"data_movement/untilize/device/kernels/compute/untilize_metal2.cpp"
"eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp"
"writer_unary_unpad_batch_rows_sharded.cpp"
"writer_unary_unpad_cross_sharded.cpp"
"writer_unary_unpad_sharded_to_interleaved.cpp"
"writer_unary_unpad_width_16_sharded.cpp"
## experimental/quasar/untilize_with_unpadding/device/factories/untilize_with_unpadding_single_core_program_factory.cpp
"reader_unary_interleaved_start_id.cpp"
"untilize_metal2.cpp"
"writer_unary_unpad_dims_split_rows.cpp"
## data_movement/untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_interleaved_program_factory.cpp
"reader_unary_interleaved_start_id_metal2.cpp"
"data_movement/untilize/device/kernels/compute/untilize_metal2.cpp"
"writer_unary_stick_layout_split_rows_multicore.cpp"
## experimental/quasar/untilize/device/factories/untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.cpp
"compute/untilize_metal2.cpp"
"dataflow/reader_unary_sharded_metal2.cpp"
"dataflow/writer_unary_sharded_metal2.cpp"
## experimental/quasar/transpose/device/transpose_wh_program_factory.cpp
"reader_unary_transpose_wh_interleaved_start_id.cpp"
"reader_unary_transpose_wh_interleaved_start_id_rm.cpp"
"transpose_wh_rm.cpp"
"experimental/quasar/transpose/device/kernels/compute/transpose_wh.cpp"
"writer_unary_interleaved_start_id.cpp"
"writer_unary_transpose_wh_interleaved_start_id_rm.cpp"
## experimental/quasar/untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_col_interleaved_program_factory.cpp
"reader_unary_interleaved_col_multicore.cpp"
"untilize_w.cpp"
"writer_unary_stick_layout_col_multicore.cpp"
## embedding/device/embeddings_rm_program_factory.cpp
"embeddings_rm_writer_chunked.cpp"
"embedding/device/kernels/dataflow/embeddings.cpp"
"writer_unary_stick_layout_interleaved_start_id_metal2.cpp"
## copy/typecast/device/typecast_rm_chunked_program_factory.cpp
"copy/typecast/device/kernels/compute/eltwise_typecast.cpp"
"copy/typecast/device/kernels/dataflow/reader_typecast_rm_chunked.cpp"
"copy/typecast/device/kernels/dataflow/writer_typecast_rm_chunked.cpp"
## data_movement/slice/device/slice_program_factory_rm_stride.cpp
"data_movement/slice/device/kernels/dataflow/reader_multicore_slice_4d.cpp"
"data_movement/slice/device/kernels/dataflow/reader_multicore_slice_nd.cpp"
"data_movement/slice/device/kernels/dataflow/writer_multicore_slice_4d.cpp"
"data_movement/slice/device/kernels/dataflow/writer_multicore_slice_nd.cpp"
## full/device/full_program_factory_sharded.cpp
"full/device/kernels/writer_full_sharded.cpp"
## copy/typecast/device/typecast_sharded_program_factory.cpp
"copy/typecast/device/kernels/compute/eltwise_typecast.cpp"
"eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp"
## embedding/device/embeddings_tilized_indices_program_factory.cpp
"ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp"
"embedding/device/kernels/dataflow/embedding_ind_tilized.cpp"
## data_movement/copy/device/copy_default_tilized_program_factory.cpp
"data_movement/sharded/device/kernels/compute/eltwise_copy_metal2.cpp"
"eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp"
"eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp"
## data_movement/transpose/device/transpose_cn_program_factory.cpp
"reader_unary_transpose_cn_interleaved_start_id.cpp"
"writer_unary_transpose_cn_interleaved_start_id.cpp"
## data_movement/transpose/device/transpose_wh_program_factory.cpp
"reader_unary_transpose_wh_interleaved_start_id.cpp"
"reader_unary_transpose_wh_interleaved_start_id_rm.cpp"
"data_movement/transpose/device/kernels/compute/transpose_wh_metal2.cpp"
"data_movement/transpose/device/kernels/compute/transpose_wh_rm_metal2.cpp"
"writer_unary_interleaved_start_id_metal2.cpp"
"writer_unary_transpose_wh_interleaved_start_id_rm.cpp"
## copy/typecast/device/typecast_program_factory.cpp
"copy/typecast/device/kernels/compute/eltwise_typecast.cpp"
"eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp"
"eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp"
## data_movement/sharded/reshard/device/reshard_program_factory_same_height.cpp
"data_movement/sharded/device/kernels/dataflow/reshard_same_height_reader.cpp"
"data_movement/sharded/device/kernels/dataflow/reshard_same_height_writer.cpp"
## data_movement/tilize/device/tilize_multi_core_default_program_factory.cpp
"reader_unary_stick_layout_split_rows_multicore.cpp"
"ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp"
"writer_unary_interleaved_start_id_metal2.cpp"
## data_movement/tilize/device/tilize_multi_core_sharded_program_factory.cpp
"ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp"
"eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp"
"writer_unary_interleaved_start_id_metal2.cpp"
"writer_unary_sharded_metal2.cpp"
## data_movement/concat/device/concat_program_factory.cpp
"reader_concat_interleaved_start_id.cpp"
"reader_concat_stick_layout_interleaved_start_id.cpp"
"ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp"
"writer_unary_interleaved_start_id_metal2.cpp"
## experimental/quasar/untilize/device/factories/untilize_multi_core_block_program_factory.cpp
"reader_unary_interleaved_wh_multicore.cpp"
"untilize_wh.cpp"
"writer_unary_stick_layout_wh_multicore.cpp"
## experimental/quasar/pad/device/pad_rm_reader_writer_multi_core_program_factory.cpp
"reader_pad_dims_rm_interleaved_sc.cpp"
"writer_pad_dims_rm_interleaved_sc.cpp"
## experimental/quasar/reshard/device/nd_reshard_program_factory_copy_pages.cpp
"experimental/quasar/reshard/device/kernels/nd_reshard_copy_pages_reader.cpp"
"experimental/quasar/reshard/device/kernels/nd_reshard_copy_pages_writer.cpp"
## experimental/quasar/reshape_view/device/reshape_device_operation.cpp
## experimental/quasar/slice/device/slice_program_factory_rm_stride.cpp
"experimental/quasar/slice/device/kernels/dataflow/reader_multicore_slice_4d.cpp"
"experimental/quasar/slice/device/kernels/dataflow/reader_multicore_slice_nd.cpp"
"experimental/quasar/slice/device/kernels/dataflow/writer_multicore_slice_4d.cpp"
"experimental/quasar/slice/device/kernels/dataflow/writer_multicore_slice_nd.cpp"
## experimental/quasar/transpose/device/transpose_wh_sharded_program_factory.cpp
"reader_unary_sharded.cpp"
"transpose_wh_sharded.cpp"
"writer_unary_sharded.cpp"
## experimental/quasar/transpose/device/transpose_cn_program_factory.cpp
"reader_unary_transpose_cn_interleaved_start_id.cpp"
"writer_unary_transpose_cn_interleaved_start_id.cpp"
## experimental/quasar/untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_nd_sharded_program_factory.cpp
"reader_unary_nd_sharded_blocks.cpp"
"untilize_variable_num_blocks.cpp"
"writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp"
## full/device/full_program_factory_nd_sharded.cpp
"full/device/kernels/writer_full_nd_sharded.cpp"
## experimental/quasar/untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp
"reader_unary_interleaved_wh_multicore.cpp"
"untilize_wh.cpp"
"writer_unary_stick_layout_wh_multicore.cpp"
```
