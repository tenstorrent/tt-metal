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
