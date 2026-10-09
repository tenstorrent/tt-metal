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
