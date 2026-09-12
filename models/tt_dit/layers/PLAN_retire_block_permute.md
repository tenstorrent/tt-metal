# Plan: retire `block_permute.py` by moving stages 2-5 onto the bricked neighborhood attention

Status: PHASE 4 VERIFIED; TIER 3 DONE; one torch oracle; GNA even-stride leader aligned with NATTEN, 2026-09-12 00:30. PAUSED. Branch `na-integration`. Owner: James Lee.
Phase 0 done; Phase 1 B2 done; D1 priced (axis swap rejected); Phase 4 (deletion) done; every gate green.
Block order was found to be unused in production (Phase 1 notes). Remaining, optional: Phases 1 (B1),
2, 3 and 5 = the "bricked deterministic stages" speed project; everything is uncommitted in the tree.

## Deletion verification (read 2026-09-11 20:45; jobs 424-428 ran after the session ended)

- 424 unit (neighborhood_sdpa, permute, bricked executor, tests/unit/test_na3d.py, vae/test_na3d.py): **90 passed, 5 skipped**.
- 425 block arms, baseline now bricked: **19 passed** (5 arms x 3 stages + 4 stage-1 arms).
- 428 production pipeline: PASSED, ANOMALIES none, **VAE decode 12.31 s** (12.35 s before the deletion).
- 427 gate runner: 29 passed, 1 skipped, **1 failed**: `test_stage5_bricked_matches_upstream_at_production_width
  [w480_h272]` timed out at pytest's 300 s. Pre-existing, not the deletion: the same param FAILED on
  2026-08-31 (jobs 876/885, PCC 1.6-1.8 %). The w480_h64 param passes at 99.9936 %. Ledger: every bricked gate
  is "new, not baselined"; no baselined entry moved.
- 426 (gates + shard equivalence + stage-5 parity): reaped by the broker while the h272 case above sat
  silent for 300 s. Before that: the full-decode bricked gate (tp_off, tp4), the stage-5 bricked gates,
  shard equivalence, `test_stage5_parity_w_sharded_bricked`, the h64 production-width gate and the ported
  `test_stage5_gna_parity_w_sharded[t12_stride111]` all PASSED; **`[t12_stride122]` FAILED** (traceback not
  captured; single-case rerun is job 430).
- Pre-commit on the deletion commit: isort/autoflake fixups committed as cb8e1ffb2dd.

## GNA even-stride leader now matches NATTEN (2026-09-12 00:30, DONE, device-verified; read this first)

**Why.** Consolidating the two torch oracles (section below) exposed that they disagreed at even GNA strides,
and James asked which one the reference implementation uses. Checked against the source of truth: NATTEN's
reference mask (`csrc/include/natten/cuda/reference/mask.hpp`) elects the group leader as
`min((index / stride) * stride + stride / 2, length - 1)`, and the GNA paper (arXiv 2504.16922, section 3)
states the default: the centre-most query, biased to the RIGHT for an even-sized group, so that it cancels an
even window's left bias and stride == window can become perfectly block-sparse. That was the rule of the deleted
general op's `nbr_shift_start` and of the old `na3d.window_bounds`. The bricked op's `window_origin_on_axis`
(and its Python transcriptions) used `first + (last - first) / 2`, i.e. biased LEFT -- a latent disagreement
with NATTEN since the op was written, pinned by nothing: both search oracles (gtest and Python) ran at stride 1
only. Odd strides and stride 1 were never affected; upstream LTX passes no stride to NATTEN, and production runs
(1,1,1), so no shipped output changes.

**What changed (one token, three places, plus tests).** The centre is now `first + (last - first + 1) / 2`,
which equals NATTEN's formula for full and truncated tail groups alike, in:
- `kernels/neighborhood_window_rule.hpp::window_origin_on_axis` (host planner AND device mask; rebuild + JIT);
- `layers/neighborhood_reference.py::context_window_origin` (the dense oracle, hence `na3d.window_bounds`/`na3d_torch`);
- `layers/neighborhood_attention.py::_window_origin` (the host regime-mask builder -- a third copy, easy to miss).
Brick snapping sits downstream of the centre and was left alone; the stride-equals-brick gtests still pass.
Tests: `test_neighborhood_reference.py::test_window_origin_matches_natten_leader_at_every_stride` (strides 1-8
incl. even, tail groups, 240 params) and gtest `NeighborhoodContextWindow.MatchesNattenLeaderAtEveryStride`
(same sweep, `brick=0`); both use the existing search oracle centred on NATTEN's leader. Host: `window_bounds`
equals NATTEN's start formula for every length < 30, kernel < 14, dividing stride (0 mismatches); Python suite
234 passed; gtest 11 passed (standalone build, recipe in memory note `gtest-build-disabled-in-build-release`).

**Device (jobs 448/449, after `./build_metal.sh --release`).** `test_neighborhood_reference.py` +
`test_neighborhood_sdpa.py` + `test_na3d_bricked_w_sharded.py` + `unit/test_na3d.py`: **307 passed, 76 skipped**
(JIT 1315/1358 hits -- the 43 misses are the mask kernels recompiling on the header change). Stage-5 parity
99.9936 % + GNA parity 5/5, **6 passed**. GNA rows vs the stride-1 upstream reference, before -> after:
(1,2,2) 0.999925 -> 0.999925, (1,4,4) 0.999912 -> 0.999912, (2,4,8) 0.999884 -> 0.999890. The unchanged
rows are expected: against an all-centred reference, "one member centred, one shifted left" and "one centred,
one shifted right" are mirror images, so the PCC magnitude is the same; only the snapped (2,4,8) row moved.

**Paused here at James's request.** Not done: the production pipeline was not re-run (stride 1 is provably
unchanged: bit-identical `window_bounds`, and the stride-1 oracle tests are untouched); `NEIGHBORHOOD_ATTENTION.md`
has no sentence on the leader rule yet -- section 2.1 (`neighborhood_window_rule.hpp`) is where one belongs.

## na3d.py / neighborhood_attention.py boundary + one torch oracle (2026-09-11 23:59, DONE)

James asked whether the two modules could be combined. Answer given and accepted: no merge (two unrelated
executors, and the gather backend is stage 1's only executor plus the sharded tests' replicated oracle), but
(1) cut the one cross-import and (2) consolidate the two torch oracles. Both done:

1. `na3d.neighborhood_attention_3d` serves only `backend="gather"`; the `"bricked"` branch and its import of
   `neighborhood_attention.py` are gone. Stage 5's module-level wrapper dispatches `"bricked"` straight to
   `neighborhood_attention_3d_bricked` (module-level import; a first attempt put the name in the W-sharded
   match arm's local import, autoflake dropped it as unused there, and `test_decode_timing[gather+bricked5]`
   failed with NameError -- job 444). `na3d.py` no longer imports `neighborhood_attention.py`.
2. ONE window rule in Python: `na3d.window_bounds` now wraps `neighborhood_reference.context_window_origin`
   (the bricked op's `window_origin_on_axis`). Measured before the change: identical at stride 1 for every
   (length < 40, kernel < 16); at even GNA strides the old `window_bounds` placed the group leader one site to
   the RIGHT (the deleted general op's `nbr_shift_start` rule) while the shipped op centres it biased left
   (139 of 331 stride > 1 cases differed, none at odd stride). `na3d_torch` is therefore now the tiled form of
   the dense reference, and `test_neighborhood_reference.py` holds them equal (`test_window_bounds_is_the_reference_rule`,
   `test_tiled_reference_matches_dense_reference`, 10 volume/stride cases each incl. even stride and stride == kernel).
   Deleted as the duplicated test: `tests/unit/test_neighborhood_3d_geometry.py` (220 lines) -- "Step 1" of the
   general op's 3D-neighborhood generalization; its `in_axis` longhand encoded the deleted kernel's biased-right
   rule and would now fail by design; everything else it pinned is covered by the two new tests.

Device (jobs 444/446/447): reference + gather + bricked-vs-`na3d_torch` units **120 passed, 5 skipped**
(`test_neighborhood_reference.py`, `unit/test_na3d.py`, `vae/test_na3d.py`, `test_na3d_bricked_w_sharded.py`,
`test_neighborhood_sdpa.py::test_matches_torch_reference`); `test_decode_timing[s16-gather+bricked5]` PASSED;
stage-5 parity + GNA parity 6 passed. Job 444 also showed a pre-existing hazard: after the NameError mid-decode
the pytest process never exited (device teardown hang), so run risky blocks as separate broker jobs.

Remaining duplicates worth a look later: `unit/test_na3d.py` (gather vs tiled oracle with chunking/sharding)
and `vae/test_na3d.py::test_na3d_op_vs_gather` (same comparison at decoder volumes plus a perf print) overlap
in purpose but not in geometry; not deleted.

## Tier 3 -- the general SDPA op's neighborhood mode excised (2026-09-11, DONE and device-verified)

James asked "what else can be deleted" after the Tier-2 verification closed, and approved items 1-3
of the answer (item 4, the ~50 MB of untracked scratch at the repo root, was NOT approved and is left).

1. **General op's neighborhood mode (C++).** No caller remained but its own test and one probe test.
   `sdpa.cpp/.hpp`, `sdpa_nanobind.cpp`, `sdpa_device_operation.cpp/.hpp`, `sdpa_device_operation_types.hpp`,
   `sdpa_program_factory.cpp`, `ring_distributed_sdpa_program_factory.cpp`, `sdpa_interleaved_cb_ids.hpp`,
   `reader_interleaved.cpp`, `writer_interleaved.cpp`, `windowed_mask_gen.hpp`, `windowed_loop_geometry.hpp`
   were reset to the merge base with main (7370e30262af, 2026-09-10): every hunk in them was neighborhood
   mode (checked hunk by hunk; the two non-neighborhood hunks were clang-format one-liners). Deleted:
   `kernels/dataflow/neighborhood_gather.hpp` and `tests/ttnn/unit_tests/operations/sdpa/test_neighborhood_3d_sdpa.py`;
   `vae/test_na3d.py::test_fused_q_offset` (the probe of the general op's `neighborhood_gather`) removed.
   KEPT: `compute_common.hpp::matmul_blocks(mask_subblock_stride)` -- the bricked op's compute kernel uses it.
   The bricked op's kernels include none of the reverted headers (`neighborhood_point3.hpp` only named
   `NeighborhoodBox` in a comment; reworded). Upstream's windowed / sliding-window modes (Qwen2.5-VL,
   Gemma 4, GPT-OSS) are untouched by construction: those files are now byte-identical to the merge base.
2. **Bricked-op probe knobs.** `DIFFVAE_NA_SKIP_KV`, `DIFFVAE_NA_MASK_MEMSET_ONLY`, `DIFFVAE_NA_TABLE_ALWAYS`
   (the 2026-09-10 mask-lever probes, all measured no-ops) and the `DIFFVAE_NA_PER_BRICK_MASK` override are
   gone from `neighborhood_kernel_args.hpp` (three `reader_arg` slots), `neighborhood_sdpa_program_factory.cpp`
   (four `getenv`s) and `neighborhood_reader.cpp` (the branches). `per_brick_mask` is now purely derived
   (chunk wider than the stride). `DIFFVAE_NA_UNSAFE_CHUNK` stays: `test_neighborhood_sdpa.py` and
   `na_sdpa_cost.py` depend on it. Reader compile-arg indices shifted, so a rebuild + device run is required.
3. **Dead script / docs.** Untracked `run_ltx25_window.sh` deleted (it exported `DIFFVAE_S5_KERNEL`, which
   nothing reads; the stash-only reader is described in memory note `s5-window-sweep-2026-09-10`).
   `NEIGHBORHOOD_ATTENTION.md`: diagnostics table and "Their SDPA kernels" section rewritten.

Verification plan (after `./build_metal.sh --release` and `import ttnn`):
- `tests/ttnn/unit_tests/operations/sdpa/test_windowed_sdpa.py` -- upstream's windowed mode still works on
  the reverted reader/writer (this is the belt for item 1).
- `models/tt_dit/tests/unit/test_neighborhood_sdpa.py test_neighborhood_permute.py test_na3d_bricked_w_sharded.py`
  + `tests/models/vae/test_na3d.py` -- bricked op after the probe removal (JIT recompiles the reader).
- `test_diffvae_decoder.py -k 'bricked_matches_replicated or shard_equivalence'`, stage-5 parity
  `test_diffvae_stage5.py -k parity_w_sharded_bricked` (expect 99.9936 %), then the production pipeline
  (expect VAE decode ~12.3 s).
Results (2026-09-11 23:06, host rebuilt with `./build_metal.sh --release`, `_ttnncpp.so` 50 KB smaller,
`import ttnn` shows the general op without the neighborhood kwargs and with `sliding_window_size` intact):
- Job 441 (units): **144 passed** -- `test_windowed_sdpa.py` (upstream's windowed mode on the reverted
  reader/writer), `test_neighborhood_sdpa.py`, `test_neighborhood_permute.py`, `test_na3d_bricked_w_sharded.py`,
  `vae/test_na3d.py`.
- Job 442 (gates): decoder `bricked_matches_replicated` + `shard_equivalence` **6 passed** at 99.9924-99.9958 %
  (identical to pre-excision); stage-5 `parity_w_sharded_bricked` 99.9936 % + GNA parity 5/5, **6 passed**.
- Job 443 (production pipeline, SLAB 78): PASSED, ANOMALIES none, **VAE decode 12.20 s** (12.31 s before),
  output `~/ltx25_diffvae_1080p.mp4`.
Kernel-directory notes (2026-09-11 23:30, James asked, then approved): `NOTES_windowed_loop_geometry.md`,
`dataflow/NOTES_neighborhood_gather_wrun.md`, `dataflow/NOTES_windowed_mask.md`, both `*_figs/` directories and
the two root diagram scripts that drew them were deleted -- they annotated the excised neighborhood mode.
`NA_SDPA_PROBES_2026-09-10.md` moved to `models/tt_dit/layers/` beside `SDPA_FUSED_VS_NEIGHBORHOOD.md` (both
untracked historical records). `MASK_PERSISTENCE.md` stays (live reader mechanism); its flag line now says the
`PER_BRICK_MASK` override is gone. James separately removed the 44 MB `h` dump and `models/demos/t3000/llama2_70b/`.
**Tier 3 DONE.** Still untouched by choice: item 4 (untracked scratch: root diagrams, `NA_SDPA_KV_COALESCING.patch`), `DIFFVAE_STAGES_WSP` (live fallback), the three untracked
tests worth committing (`test_brick_activation.py`, `test_halo_exchange_geometry.py`, `test_gemma4_cache_roundtrip.py`).

## PICKUP 2026-09-11 21:25 -- deletion fully verified, nothing open (superseded by the Tier 3 note above)

The two items left open at 20:55 both closed on device:
1. `test_stage5_gna_parity_w_sharded` (job 439): **5 passed**. Stride-1 rows at 99.9936 % / 99.9932 % PCC;
   strides (1,2,2) / (1,4,4) / (2,4,8) logged end-to-end PCC 0.999925 / 0.999912 / 0.999884. The
   (2,4,8) row replaces the (11,4,8) row that a (2,4,4) brick cannot represent.
2. `test_stage5_bricked_matches_upstream_at_production_width[w480_h272]` (job 440): **PASSED at 99.9936 %**
   (RMSE/sigma 1.1 %), call time 501 s. The "timeout" in jobs 426/427 was pytest.ini's 300 s default
   against the ~10 min host reference (which the test's own comment already priced), not a hang. The row
   now carries `pytest.mark.timeout(1800)`. The 2026-08-31 PCC failure (jobs 876/885) predates the
   Phase-0 HiFi2/exact-exp numerics fix and does not reproduce.

Broker note for this gate: a job silent for 300 s is reaped, so run it with a stdout heartbeat
(`(while true; do echo heartbeat; sleep 60; done) & ...; kill $!`) and a `timeout_sec` of 1500.

Still to decide (James): the bricked gates are all "new, not baselined" in the ledger. A
`RECORD_BASELINE=1 run_diffvae_gates.sh` run would pin them; it needs >= 1500 s because of the h272 row.

Next levers (optional, unchanged): Phase 5 per-stage brick hoist; stage 1 (index 0, replicated gather
backend, ~1000 ms per decode).

## PICKUP 2026-09-11 20:55 -- deletion verified except two items (superseded by the note above)

Commits on na-integration, oldest first: ddc59cd71ed (bricked deterministic stages), 396f54b6956 (misc),
ee209de41d5 (Tier-2 deletion of the general-SDPA executors), 955898dba2b (pickup note), cb8e1ffb2dd
(pre-commit fixups), then the commit carrying this note and the GNA-parity test fix.

Open:
1. `test_stage5_gna_parity_w_sharded` at W=64 (job 431): 4 of 5 PASSED (both stride-1 rows at
   PCC >= 0.999, strides (1,2,2) and (1,4,4) logged). `t22_stride11_4_8` FAILED with the planner's
   "a multi-brick query chunk must equal the stride exactly": a T stride of 11 is not a whole number of
   (2,4,4) bricks, a bricked-op constraint, not the deletion. The row is now stride (2,4,8)
   (`t22_stride2_4_8`), **unverified on device**: run
   `pytest models/tt_dit/tests/models/vae/test_diffvae_stage5.py -k gna_parity_w_sharded` (needs
   `LTX_CORE_SRC=/home/noblewoodall/LTX-2/packages/ltx-core/src`) to close it.
2. `test_stage5_bricked_matches_upstream_at_production_width[w480_h272]` times out at pytest's 300 s
   (jobs 426/427). PRE-EXISTING: the same case failed on 2026-08-31 at PCC 1.6-1.8 % (jobs 876/885).
   Not caused by the deletion; either investigate (first-run JIT for that geometry vs a real hang --
   it went silent right after mesh creation) or mark it xfail with that history.

Everything else passed after the deletion: unit 90, arms 19, decoder + stage-5 gates (all bricked
params), shard equivalence, production pipeline VAE decode 12.31 s, ANOMALIES none.

## PICKUP 2026-09-11 17:35 (superseded by the note above)

Commits on na-integration: ddc59cd71ed (bricked deterministic stages), 396f54b6956 (misc: decode-tree live
lines, perf-table breakdown, accessor fix), then the WIP deletion commit (pre-commit hooks skipped with -n;
run `python_env/bin/pre-commit run --files $(git diff --name-only HEAD~1 HEAD)` and amend if it reformats).
Broker jobs queued at commit time (logs under /var/log/tt-device-broker/ and generated/del_*.log):
424 unit (neighborhood_sdpa, permute, bricked executor, tests/unit/test_na3d.py, vae/test_na3d.py),
425 block arms (baseline now bricked), 426 decoder gates + shard equivalence + stage-5 parity/GNA
parity (ported to bricked) + production-width gate, 427 run_diffvae_gates.sh, 428 production pipeline
(expect VAE decode ~12.35 s). If any fail, the deletion commit is the one to fix or revert. Plan and
inventory: ~/.claude/plans/right-now-the-deterministic-sparkling-pike.md.

## Bricked deterministic stages -- in progress 2026-09-11 (Phases 1-B1, 2, 3)

Scope (James, 2026-09-11): stages 2-4 only; stage 1 stays replicated on the gather backend. Per-stage
default by measurement: a stage where the bricked executor is slower keeps `op_sp_w_sharded`.
Working plan with the geometry table and gates: `~/.claude/plans/right-now-the-deterministic-sparkling-pike.md`.

Done in the tree (uncommitted):
- B1 via brick width 1. `_choose_sharded_brick` searches odd widths; the non-hoisted K/V path bricks
  the OWNED columns first and halo-exchanges on `W_br` (the hoisted path's exchange, shared as
  `exchange(...)`), so the natural-order W-fold and its 128 B stick hazard are gone. Host planner
  picks: stage 2 (8,4,1)/63, stage 3 (16,2,1)/45, stage 4 (8,2,2)/27; stage 5 unchanged (8,2,2)/147.
- Flat `(1, heads, sites, hd)` input is transposed to site-major inside the executor when heads > 1
  (the old reshape was a view only at one head per chip).
- `DIFFVAE_NA_BRICK` accepts a volume-keyed form `T,H,W:bt,bh,bw;...`.
- `NeighborhoodAttention` has a `bricked_sp_w_sharded` arm (flat and volume paths);
  `DeterministicStages` takes a name or a per-stage `{1: .., 2: .., 3: ..}`; `DIFFVAE_STAGES_BACKEND`
  (adapter, timing test, run scripts; default `op_sp_w_sharded`).
- Tests: chooser pins for the six deterministic geometries; width-1 3-shard op case; executor cases at
  W_local 15/30 with 2/4 heads per chip and a flat input; `bricked` / `bricked_volume` arms;
  `test_decode_full_bricked_matches_replicated` (latent (2,8,16), TP on/off).

### Device results so far (2026-09-11, jobs 415-417)

- Unit (job 415): 102 passed -- chooser pins for all six deterministic geometries, the width-1 3-shard
  op case, and every new executor case (W_local 15 with 2/4 heads per chip, flat head-major input).
- Arms correctness (job 416): 28 passed; `bricked` / `bricked_volume` at 99.997-99.998 % PCC vs the
  strided baseline on stages 2/3/4 (same band as `flat_seq`).
- Arms timing (job 417, ITERS=10, ms/block x depth at the arms geometry (6,68,120)/(11,68,120)/(21,136,240)):

| arm | stage 2 | stage 3 | stage 4 | total |
|---|---|---|---|---|
| baseline (strided, no flags) | 37.9 x6 = 227 | 35.3 x4 = 141 | 267.8 x2 = 536 | 904 |
| flat_seq (strided, production flags) | 27.9 x6 = 167 | 27.1 x4 = 108 | 202.8 x2 = 406 | 681 |
| bricked (flat handoff) | 11.1 x6 = 67 | 10.3 x4 = 41 | 47.7 x2 = 95 | 203 |
| bricked_volume | 10.8 x6 = 65 | 10.0 x4 = 40 | 47.6 x2 = 95 | 200 |

  Bricked wins every stage by 2.6-4.3x per block. The B3 worry (2-2.7x more keys gathered per
  query) is outweighed: the strided executor's full-W K/V gather + retile + fused SDPA over the
  W-outer sequence costs far more than the bricked op's small gather.

- Decoder gate (job 418, latent (2,8,16)): `tp_off` PASSED at 99.9924 % PCC (RMSE/sigma 1.3 %); `tp4`
  FAILED on a reshape volume mismatch at the out-proj: the gate runs without the production flags, so
  the unfused projection handed the bricked executor every head and the TP head all-gather returned
  `tp * dim` channels. Fixed by partitioning the `(tokens, dim)` projections over `tp_axis` on the
  bricked backend (`own_heads` in `NeighborhoodAttention.forward`); re-run pending.
- Decode `s34x60` (jobs 419/420, pipeline flags, TP4, SLAB 73, 145 frames), bricked vs strided
  deterministic stages, stage 5 bricked in both:

| row | bricked (ms) | strided (ms) |
|---|---|---|
| decode total | **13356** | 13826 |
| det stage 0 (replicated, gather) | 1000 | 997 |
| det stage 1 (21,68,120) | 213 | 288 |
| det stage 2 (41,68,120) | 149 | 205 |
| det stage 3 (81,136,240) | 423 | 746 |

  Every W-sharded stage wins (-75 / -56 / -323 ms); -470 ms per decode (-3.4 %). Stage 1 (index 0,
  replicated on the gather backend, 1000 ms) is now the largest deterministic stage by far.

- Decoder gate re-run (job 421) after the `own_heads` fix: `tp_off` and `tp4` both PASSED at 99.9924 %.
  **D2 decided: bricked on all three stages.** Default flipped in `stages_backend_from_env`
  (`bricked_sp_w_sharded`); the run scripts' `DIFFVAE_STAGES_BACKEND` default follows.

- Profiled pipeline (job 422, `PROFILE=1`, SLAB 73, BLOCK_PROF): PASSED, ANOMALIES none, video at
  `generated/profile/20260911_064903/ltx25_1080p.mp4`. VAE decode 13.83 s under deep profiling (not
  comparable to the 12.58 s production number; the plain run is job 423). Stage-2 breakdown (6 blocks,
  BLOCK_PROF-inflated): attention 278 ms = neighborhood-sdpa 90, halo+brick-permute (k,v) 83, q-to-seq 29,
  head-unflatten 19, head-allgather 11, unbrick 8. Layout work (~120 ms) now exceeds the op (~90 ms):
  that is the case for the optional Phase 5 per-stage brick hoist (brick once at stage entry, unbrick
  before the upsample, bricked RoPE tables), which would remove the per-block brick/unbrick permutes.
  Stage 1 (index 0, replicated gather backend) at ~1000 ms is the largest deterministic stage and is out
  of this round's scope.

- Production-config pipeline (job 423, SLAB 78, untraced, bricked deterministic stages): PASSED, ANOMALIES
  none, **VAE decode 12.35 s** vs the 12.58 s baseline (job 413), output `generated/bricked_det_stages.mp4`.

**Bricked deterministic stages: DONE 2026-09-11.** Defaults flipped everywhere (`stages_backend_from_env`,
both run scripts). Nothing committed. Next levers: Phase 5 per-stage brick hoist; stage 1 (index 0).

## Pickup -- state as of 2026-09-11 00:40 (read this first)

**Where things stand.** Phase 0 (bricked executor is the stage-5 pipeline default, numerics fixed),
Phase 1 B2 (multi-head TP permute in the bricked executor) and Phase 4 (block-permute path deleted,
Python + C++) are done and device-verified. D1 was priced and the axis swap rejected. Nothing is
committed. Phases 1 (B1), 2, 3 and 5 are optional: they migrate the deterministic stages onto the
bricked executor for speed only, since both paths run stride-1 attention today.

**Uncommitted tree, this work (26 files; `neighborhood_sdpa_nanobind.cpp` joined for follow-up 2):**
```
M models/tt_dit/experimental/scripts/run_ltx25_diffvae.sh
M models/tt_dit/experimental/scripts/run_ltx25_pipeline.sh
M models/tt_dit/layers/NEIGHBORHOOD_ATTENTION.md
D models/tt_dit/layers/block_permute.py
M models/tt_dit/layers/na3d.py
M models/tt_dit/layers/neighborhood_attention.py
D models/tt_dit/tests/models/vae/ab_gna_decode.py
D models/tt_dit/tests/models/vae/ab_gna_stage5.py
M models/tt_dit/tests/models/vae/test_decode_timing.py
M models/tt_dit/tests/models/vae/test_diffvae_decoder.py
D models/tt_dit/tests/unit/test_block_permute.py
D models/tt_dit/tests/unit/test_block_permute_device.py
D models/tt_dit/tests/unit/test_block_sdpa_op.py
M models/tt_dit/tests/unit/test_na3d_op_sp.py
M ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp
M ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/windowed_mask_gen.hpp
M ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp
M ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/windowed_loop_geometry.hpp
M ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp
M ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.hpp
M ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation_types.hpp
M ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp
M ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp
M ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.hpp
M ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp
```
Also modified but NOT this work (James's, pre-existing): `models/tt_dit/pipelines/ltx/pipeline_ltx_distilled.py`, `models/tt_dit/utils/ltx.py`, `ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/strided_all_gather_common.hpp`.
Untracked scratch left alone: root-level diagram scripts/pngs, `NOTES_*.md` (the kernel geometry
notes had their block section retired), `neighborhood_gather_figs/`, `windowed_loop_geometry_figs/`.

**Suggested commit split (all on `na-integration`; pre-commit's clang-format may reformat once):**
1. `neighborhood_attention.py` compute config (HiFi2/exact exp) + decoder gate `latent_w`/dump +
   `run_ltx25_pipeline.sh` stage-5 defaults -- "Phase 0: bricked stage 5 by default, matched numerics".
2. `neighborhood_attention.py` TP permute + `test_na3d_op_sp.py::test_bricked_w_sharded_tp_matches_host`
   + `test_decode_timing.py` axis knobs -- "bricked executor: TP with >1 head per chip".
3. Everything else (deletions, `na3d.py`, scripts, SDPA C++/kernels, docs) -- "retire block_permute".
   A rebuild is required for this one: `./build_metal.sh --release` (~15 min here; it installs).

**Re-verify (device, via the broker; workspace `/home/jameslee`, prefix `cd /home/jameslee/tt-metal &&`):**
```
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
# kernels (129 tests, ~4 min; keep stdout flowing -- the broker reaps 300 s of silence)
python_env/bin/python -u -m pytest tests/ttnn/unit_tests/operations/sdpa/test_windowed_sdpa.py tests/ttnn/unit_tests/operations/sdpa/test_neighborhood_3d_sdpa.py models/tt_dit/tests/unit/test_na3d_op_sp.py -q --timeout=0
# stage-5 gates (~5 min): bricked (4 params, expect >= 99.995 pct) and strided op_sp_w_sharded (expect 99.996 pct)
python_env/bin/python -u -m pytest models/tt_dit/tests/models/vae/test_diffvae_decoder.py -k 'bricked_matches_replicated or stage5_wsp_matches_replicated' -q --timeout=0
# decode timing (expect ~13.8-14.0 s, deterministic stages ~2.3 s)
eval "$(grep '^export DIFFVAE_' models/tt_dit/experimental/scripts/run_ltx25_pipeline.sh | grep -v PROFILE)"; DIFFVAE_SLAB_FRAMES=73 python_env/bin/python -u -m pytest models/tt_dit/tests/models/vae/test_decode_timing.py -k 'test_decode_wsp_timing and s34x60' -s -q --timeout=0
# pipeline (~11 min; expect VAE decode ~12.6 s at the script's SLAB 78)
LTX_VAE_TIME=1 bash models/tt_dit/experimental/scripts/run_ltx25_pipeline.sh
```
Stage-5-only parity against upstream needs `LTX_CORE_SRC=/home/noblewoodall/LTX-2/packages/ltx-core/src`
(`test_diffvae_stage5.py -k parity_w_sharded_bricked`, expect 99.9936 pct).

**Open follow-ups, in priority order:**
1. DONE 2026-09-11 01:32 (job 413): full 1080p pipeline on the rebuilt library + kernels, VAE decode
   **12.58 s** (12.56 s before the C++ step), output `generated/phase4_final.mp4`.
2. DONE 2026-09-11 01:40 (rebuilt; job 414: `test_neighborhood_sdpa.py` + `test_neighborhood_permute.py`
   pass calling the op without a config, stage-5 parity 99.9936 %, 4-way decoder gate 99.9946-99.9958 %):
   `neighborhood_sdpa_nanobind.cpp` now resolves the default via `init_device_compute_kernel_config(arch,
   cfg, HiFi2, approx=false, fp32_acc=false, l1_acc=false)` like `sdpa.cpp`, instead of
   `DeviceComputeKernelConfig{}` (LoFi/approx). Host unit syntax-checks clean.
3. `test_na3d_op_sp_w_sharded_matches_host[dims2-kernel2-sp_cols]` (12-token shard) fails at 94 pct
   under the pipeline's exported `DIFFVAE_*` flags, on old and new code alike; passes clean. Either
   the flags are invalid for tiny shards (likely `DIFFVAE_PAD_GATHER` / `DIFFVAE_SDPA_KCHUNK=256`) or a
   real bug in `op_sp_w_sharded` at sub-tile shards. Not production geometry.
4. `DIFFVAE_S5_KERNEL` window-override reader exists only in `git stash@{0}` while
   `run_ltx25_window.sh` still exports it (memory note `s5-window-sweep-2026-09-10`).
5. Weight cache dir is keyed by mesh only (`CP1_0_TP4_0_SP8_1_mesh4x8_bf16`), so a TP-off run
   collides with the TP-on layout; workaround used: `TT_DIT_CACHE_DIR=~/.cache/tt-dit-tpoff`
   (hardlink copy, deletable).
6. Optional project, Phases 1-3/5: B1 via brick width 1 in `_choose_sharded_brick` (the axis swap
   costs 2.2 s/decode, rejected), then wire `bricked_sp_w_sharded` into `NeighborhoodAttention`
   for stages 2-4 and measure against the strided executor at GNA=0 (~290 ms of K/V gather +
   retile on the table vs small-window gather waste).

**Gotchas learned this round:**
- Kernel sources are JIT-only: the unity host syntax check never sees them. After a kernel edit, run
  the exact JIT command (printed on a compile failure, or via `TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1`)
  from the cached kernel dir with `-fsyntax-only` before touching the device.
- Do not export the pipeline's `DIFFVAE_*` set into unit tests; it changes executor paths.
- A `grep | tail` filter on a broker job hides errors AND goes silent: tee the full log to a file.
- Both stage-5 executors fuse the gather into the kernel; say "block-permute executor" (older,
  now deleted) and "bricked executor" (newer), never "fused".

## Goal

Delete `models/tt_dit/layers/block_permute.py` and everything that exists only to serve it. Today it
and `neighborhood_permute.py` implement the same 3-D token permutation twice (same 8-D reshape,
same permute order `(0,1,3,5,2,4,6,7)`, same ceil-padding-by-concat, same index formula with
`bt*bh*bw` in place of the literal 32). Rather than unify the two, retire the older one: every
consumer of block order moves to the bricked op, and the block path is removed end to end.

## Naming

- **block-permute executor** -- the OLDER implementation: `neighborhood_attention_3d_op_sp_w_sharded`
  (`layers/na3d.py`, kernel name `op_sp_w_sharded`): full-W K/V all-gather + `wrow` retile, Q in block
  order via `block_permute.py`, runs the general `ttnn.transformer.scaled_dot_product_attention` with
  its neighborhood arguments.
- **bricked executor** -- the NEWER implementation: `neighborhood_attention_3d_bricked_w_sharded`
  (`layers/neighborhood_attention.py`, kernel name `bricked_sp_w_sharded`): halo exchange, bricked order
  via `neighborhood_permute.py`, runs the dedicated `ttnn.transformer.neighborhood_scaled_dot_product_attention`.

"Fused" is not used for either: both fuse the gather into the kernel, and the word already names
`DIFFVAE_SP_FUSED`, fused RoPE/qkv and the `fused-sdpa` timing-tree row.

## Where block order is live today

`block_permute` has exactly one production caller, `neighborhood_attention_3d_op_sp_w_sharded`
in `layers/na3d.py`, gated by `DIFFVAE_BLOCK=1 && DIFFVAE_SP_FUSED=1`. Both are exported by
`experimental/scripts/run_ltx25_pipeline.sh`. That executor is used by:

| Consumer | Selected by | Uses block order in production? |
|---|---|---|
| Deterministic stages 2-4 (`diffvae_ltx.py::DeterministicStages`, `block_backend(stage)`) | `DIFFVAE_STAGES_WSP=1` | **no** -- `_pick_block` finds no legal block at W_local 15/30 (found 2026-09-10 evening), so the executor runs its strided mode; see Phase 1 notes |
| Stage 5 (`diffvae_ltx_stage5.py`, `NAKernel "op_sp_w_sharded"`) | was the default when `DIFFVAE_STAGE5_BACKEND` was unset; Phase 0 made `bricked_sp_w_sharded` the script default | was reachable, but the executor OOMs in the full pipeline (Phase 0 notes) -- only decode-only harnesses ever ran it |
| Deterministic stage 1 (index 0) | replicated `gather` backend (W=60 does not divide the size-8 mesh axis) | no, untouched by this plan |

Downstream of the Python, the block dims travel as `neighborhood_block` through
`ttnn.transformer.scaled_dot_product_attention` -> `sdpa.cpp` -> `sdpa_program_factory.cpp` ->
`windowed_loop_geometry.hpp` (`BlockCoord`, `block_index_of_chunk`, `neighborhood_box_block`,
`block_query_coord`). All of that becomes dead once no caller sets `DIFFVAE_BLOCK`.

## Deterministic-stage geometry the migration must fit (1080p, mesh 4x8, SP on the size-8 axis, TP4 on the size-4 axis)

From `diffvae_bricked_timing_tree.txt` and `tests/models/vae/test_det_nablock_arms.py`:

| Stage | Volume in (T,H,W) | dim / heads (head_dim 64) | heads per chip @TP4 | Kernel | W_local @sp=8 |
|---|---|---|---|---|---|
| 2 | 21, 68, 120 | 1024 / 16 | 4 | (3,7,7) | 15 |
| 3 | 41, 68, 120 | 512 / 8 | 2 | (3,5,5) | 15 |
| 4 | 81, 136, 240 | 512 / 8 | 2 | (3,5,5) | 30 |
| 5 (reference) | ~25 latent -> 272x480 | 256 / 4 | 1 | (11,11,11) | 60 |

Deterministic-stage attention cost per decode today (ms, from the same tree; row labels as in the tree): fused-sdpa 410, kv-wrow
retile 203, kv-allgather 90, attn-unflatten 81. Deterministic stages total 1767 ms = 13.5% of the decode.
This is a code-deletion migration, not a speed one; the gate is "no regression".

## Blockers found

### B1. Shard width is not brick-aligned at stages 2 and 3

`neighborhood_plan.cpp` requires `shard_origin` to be brick-aligned. `_choose_sharded_brick`
only tries even brick widths (odd widths were excluded for a `neighbor_pad` stick-size hang in
natural order). With W_local = 15, `index * 15 - halo` is odd for odd shard indices, so every
candidate is rejected and the executor cannot build a plan. Stage 4 (W_local 30) is fine.

Options, preferred first:

1. **Shard the deterministic stages over the size-4 axis** (W_local 30 at stages 2-3), TP over heads on the
   size-8 axis. Also lands stages 3-4 at one head per chip (see B2). Changes the mesh layout
   the deterministic stages run on; `_wshard`/`_wgather`/`mesh_partition` of cos/sin all take `sp_axis`, so
   it is a config change, but `DIFFVAE_TP_HEADS` / `tp_axis` plumbing in `NeighborhoodAttention`
   and `SwiGLU` must be checked for hard-coded axes.
2. **Allow brick width 1** in `_choose_sharded_brick`. In bricked order the halo stick is
   `32 * channels` wide, so the 128 B hang that motivated the exclusion may not apply. Needs a
   device check of `_halo_exchange` at brick width 1, and the gather waste at (t,h,1) bricks.
3. Teach the planner non-aligned shard origins. Kernel work; do not start here.

### B2. The bricked TP-over-heads path asserts one head per chip

`neighborhood_attention_3d_bricked_w_sharded` (TP block, ~line 975) asserts `head_count == 1`
because it reshapes the site-major output `(sites, heads*hd)` to `(heads, sites, hd)` as a view.
Stages 2-4 have 4/2/2 heads per chip at TP4. Fix: permute site-major -> head-major before the
head all-gather when `head_count > 1` (the block-permute executor's `attn-unflatten` does exactly this).
Python only. The op itself already takes `head_count > 1`.

### B3. Performance at small windows is unmeasured

(3,7,7) = 147 sites and (3,5,5) = 75 sites; the op gathers whole 32-site bricks, so the waste
factor is several times worse than at 11^3 (where gather is 147 bricks for 1331 sites). Against
that, the bricked path drops the full-W K/V all-gather and the `kv-wrow` retile (~290 ms).
Net sign unknown until measured (Phase 3).

### B0 (precondition). Bricked stage 5 has two open PCC gate failures on baseline

Memory note `bricked-stage5-gates-fail-on-baseline`: two stage-5 PCC misses predate the recent
work. Making bricked the production default for stage 5 (Phase 0) requires closing or
explicitly waiving them.

## Phases

### Phase 0 -- bricked becomes the stage-5 production default
- [x] Reproduce the two failing stage-5 bricked gates on baseline (2026-09-10, broker jobs 353/354):
  - `test_decode_wsp_shard_equivalence` now PASSES at stride (1,1,1) (PCC 1.000000, 0.14 % mean
    diff, different bricks per arm) and at stride (1,2,2) (bit-identical, same brick both arms).
    The memory note's 0.9779 failure no longer reproduces post main-merge. Closed.
  - `test_decode_stage5_bricked_matches_replicated` still misses: 99.8999 % (t_at_window) and
    99.8882 % (t_clear_of_window) vs 99.9 %, RMSE/sigma ~5 %. History from the broker logs: the
    same test shape and the same brick (8,2,2) scored 99.9935 % on 2026-08-31 17:55 (job 869) and
    99.9909 % at 21:31 (job 907); the first miss is 2026-09-01 16:26 (job 944). The only commit
    in that window touching the path is 5f5039bef1a (`_halo_split` sub-column halo exchange,
    `_release_intermediates`, chooser degenerate-brick skip). Calibration: the PRODUCTION block-permute
    executor with `DIFFVAE_GNA=1 DIFFVAE_BLOCK=1` scores 99.8436 % on the sibling test (job 943), so
    bricked is already closer to the replicated reference than what ships today.
  - Not the brick chooser: forcing (2,4,4) cannot even plan at local width 8 (shards gather
    120 vs 150 bricks; job 356).
  - Ruled out by the `latent_w` param + pixel dump (job 360): local width 8 vs 16 makes no
    difference (99.886-99.901 % across all four), and the error is flat across columns and frames
    (band-edge/interior ratio 0.94-1.08) -- a uniform ~5 % RMSE/sigma floor, not a seam.
  - **Root cause (job 369/370):** `neighborhood_scaled_dot_product_attention` was called without a
    `compute_kernel_config`, so the binding's `value_or(DeviceComputeKernelConfig{})` gave it
    **LoFi** matmuls and the **approximate exp** (`ComputeKernelConfig` defaults). The general
    `scaled_dot_product_attention` op that the replicated reference and the block-permute executor run
    defaults to **HiFi2** (`init_device_compute_kernel_config`
    in `sdpa.cpp`) with exact exp (`SDPAProgramConfig(exp_approx_mode=False)`). Fix: pass a matching
    `WormholeComputeKernelConfig(HiFi2, math_approx_mode=False)` at both op call sites
    (`_compute_kernel_config()` in `neighborhood_attention.py`; `DIFFVAE_NA_FIDELITY` /
    `DIFFVAE_NA_APPROX_EXP` are A/B knobs). Stage-5-only parity vs ltx_core: 99.9899 -> 99.9936 %
    (block-permute executor: 99.9935 %). Gate A: 99.9946-99.9958 % on all four params, RMSE/sigma 0.9-1.1 %.
    Why the 08-31 runs passed with the same default is unexplained; the default was already LoFi then.
- [x] Fix the gate-A miss (compute config, above). Gate-test additions kept: `latent_w` param and
      `DIFFVAE_DUMP_PIXELS` dump of both arms.
- [x] Cost of HiFi2 vs LoFi on the bricked op (job 371, `test_decode_wsp_timing -k s34x60`, 4x8,
      TP4, W-SP deterministic stages): decode 13916 vs 13911 ms, stage-5 attention 1132-1139 vs
      1134-1146 ms per block. No measurable cost -- the op is gather-bound, not matmul-bound.
- [ ] Follow-up (C++, not blocking): make the op's own default HiFi2/exact like `sdpa.cpp` does via
      `init_device_compute_kernel_config(arch, cfg, HiFi2, ...)`, so no caller can fall into LoFi
      silently. Needs a ttnn rebuild (~25 min); the Python-side config covers production meanwhile.
- [x] `run_ltx25_pipeline.sh`: `DIFFVAE_STAGE5_BACKEND`, `DIFFVAE_S5_GNA_STRIDE=1,1,1` and
      `DIFFVAE_TP_HEADS=1` moved from the PROFILE block into the common exports (2026-09-10).
- [x] Pipeline timing, same script and tree, one untraced generation each (VAE decode row; the
      DiffVAE decode is untraced either way so it compares across trace modes). Summary: new
      default 12.56 s (SLAB 78) / 13.35 s (SLAB 73); fidelity fix free (12.58 s at LoFi); the block-permute
      executor (old default) OOMs in every configuration on today's tree (TP4 at 78 and 73, TP off at 73) and
      there is no record of it ever completing the 1080p pipeline on this branch since
      2026-08-20 (see arm B notes) -- the full-W K/V retile does not fit beside the resident
      transformer + encoder (~4.0 of 4.2 GB/bank). The evening `pipeline_ltx_distilled.py` /
      `utils/ltx.py` edits are reporting-only and are NOT the cause. **Phase 0 verdict: bricked default is faster or equal,
      fits where the block-permute executor no longer does, and passes every gate. Proceed to Phase 1.**
  - Arm A, new default (bricked, HiFi2, TP4, SLAB 78): **12.56 s** decode, job 372, PASSED.
  - Arm B, old default (block-permute executor, `op_sp_w_sharded`) at the script's SLAB 78: **OOM** -- 1.39 GB DRAM
    buffer in the block-permute executor's K/V retile (`na3d.py:1217 wrow`), 4.04 of 4.21 GB/bank
    allocated (job 373). The block-permute executor only ever ran the pipeline at SLAB 73; the bricked path
    fits at 78 because the halo exchange replaces the full-W K/V gather. Re-run as
    `DIFFVAE_STAGE5_BACKEND=op_sp_w_sharded DIFFVAE_TP_HEADS=0 DIFFVAE_SLAB_FRAMES=73` (job 376).
  - Arm C, bricked at the old op numerics (LoFi, approx exp), SLAB 78: **12.58 s**, job 374,
    PASSED. Fidelity is free at pipeline level too (vs arm A's 12.56 s).
  - Arm A73, new default at SLAB 73: **13.35 s**, job 377, PASSED (SLAB 78 -> 73 costs ~0.8 s,
    matching the decode-only note in [[diffvae-slab78-pipeline-ooms]]).
  - Arm B at SLAB 73 with TP4 on: **OOM again**, same site (`na3d.py:1217 wrow`), 1.29 GB with
    4.04 GB/bank allocated (job 379). On today's tree the block-permute executor does not fit the pipeline's
    DRAM budget with TP on at either slab size. CORRECTION: this morning's 13.29 s "baseline" (job 281) was the
    BRICKED executor (TP4, SLAB 73, traced) -- its launcher script says so; the memory note that
    called it the "standard prefix" was misread. There is NO full-pipeline run with the
    block-permute executor for stage 5 in the broker history since 2026-08-20 (decode 107-120 s,
    pre-W-sharding era); every 1080p pipeline run since then set `bricked_sp_w_sharded`. The
    block-permute executor was only ever exercised by decoder/timing tests (decode-only, no
    transformer/encoder resident), where its 1.3 GB (TP4) / 5.2 GB (TP off) full-W K/V retile fits.
    Traced mode does not rescue it either: traced, TP off, SLAB 73 OOMs on a 2.7 GB buffer at the
    same retile (`na3d.py:1213 wrow`, job 383). Hypothesis closed.
  - **Why the block-permute executor was never pipeline-checked for stage 5, and why we stop here
    (decided 2026-09-10, James):** the switch itself exists (`DIFFVAE_STAGE5_BACKEND`), but the
    executor's `wrow` step copies the full-W gathered K and V from TILE to ROW_MAJOR, ~1.3 GB per
    tensor per band at TP4 with the tiled original still alive, against ~170 MB free once the
    transformer + encoder are resident. It only ever ran in decode-only harnesses where nothing
    else is resident. Making it fit would take smaller bands (`DIFFVAE_SLAB_FRAMES` ~30-40, slower),
    gathering straight into ROW_MAJOR (`DIFFVAE_KV_RM_GATHER=1`, measured to inflate memory),
    freeing the transformer around the decode (~85 s of reloads per run), or giving it the halo
    exchange the bricked executor already has (kernel work that re-derives the bricked design).
    None of that serves a plan whose goal is to retire the executor, and the decode-only timing test
    already compares the two stage-5 executors cleanly. Not pursued.
  - Arm B with TP off (separate weight cache `TT_DIT_CACHE_DIR=~/.cache/tt-dit-tpoff`, a hardlink
    copy with the diffvae entries dropped): **OOM in warmup**, 5.2 GB K/V retile (4x the TP4
    buffer, as expected) at `na3d.py:1208 wrow` (job 381). So the block-permute executor cannot complete
    the pipeline on today's tree in any configuration reachable here; the bricked path completes
    in all of them.
  - Arm B with `DIFFVAE_TP_HEADS=0` cannot run from the shared weight cache: the cache dir
    (`CP1_0_TP4_0_SP8_1_mesh4x8_bf16`) is keyed by mesh only, was written by TP-on runs (column-parallel
    fused-qkv weights), and TP-off wants `stages.det_stages.*.attn.to_q.weight` (job 376). Pre-existing;
    not this plan's problem, but it means the "old default" is measured with TP on (job 379).
- [ ] Gate: `test_decode_stage5_bricked_matches_replicated`, `test_stage5_parity_w_sharded_bricked`,
      and a pipeline gen-0 output within the usual PCC of the current `op_sp_w_sharded` run.
- After this phase, stage 5 no longer needs `DIFFVAE_BLOCK`. Stages 2-4 still do.

### Phase 1 -- generalize the bricked W-sharded executor (`layers/neighborhood_attention.py`)
- [x] B2: multi-head TP path (2026-09-10). `neighborhood_attention_3d_bricked_w_sharded` now
      reshapes the site-major output to `(b, sites, heads, hd)` (a view) and permutes to
      `(b, heads, sites, hd)` before the head all-gather when heads per chip > 1; one head per chip
      keeps the old view. New test `test_na3d_op_sp.py::test_bricked_w_sharded_tp_matches_host`
      (4x8, W over the size-8 axis, heads presharded over the size-4 axis, 1 and 2 heads per chip,
      W_local 8 and 16, windows (3,3,3)/(3,5,5)) -- 4 passed, PCC 99.982/99.982 % and
      99.970/99.971 % (one vs two heads per chip, per geometry; job 385). Stage-5 parity control
      unchanged at 99.9936 %. The bricked executor does NOT slice heads itself under TP
      (`heads_presharded` is documentation only); the caller must hand it its own heads, which the
      column-parallel qkv does in production.
- [ ] B1: implement the chosen option (decision D1 below).
  - D1 pricing (2026-09-10, `test_decode_wsp_timing -k s34x60`, bricked stage 5, TP4, SLAB 73,
    deterministic stages on the block-permute executor; knobs `DIFFVAE_STAGES_SP_AXIS` /
    `DIFFVAE_STAGES_TP_AXIS` added to the test): today's axes (W over size-8, TP over size-4) =
    **13993 ms** at the script's `DIFFVAE_GNA=1`. The swapped axes (W over size-4, TP over size-8)
    **cannot run at GNA=1**: `_pick_block(t=81, h=136, w_local=30, gna=True)` picks a block with
    bh=8 and the executor sets the GNA stride to the block, but stage 4's kernel is (3,5,5), so the
    fused SDPA rejects "neighborhood_stride h=8 must not exceed the effective kernel h=5" (job 386).
    A latent bug in the block-permute executor -- `_pick_block`'s `kmax` defaults to 11 (the stage-5
    window) regardless of the stage's kernel -- that only surfaces at non-default shard widths.
    Re-priced at `DIFFVAE_GNA=0` for both arms (job 388) so the comparison isolates the axis swap:
    today's axes **13814 ms** (deterministic stages 2291 ms); swapped axes **15994 ms**
    (deterministic stages 4244 ms, of which `det -> replicated context gather` 1781 ms, plus a
    66 ms `stage5: context reshard`). **Option 1 costs ~2.2 s per decode (+16 %)**, almost all of
    it the lost same-axis W-sharded handoff, which re-materialises the replicated stage-5 context
    on every chip. Not viable as-is; it would need an axis-0 -> axis-1 reshard collective that
    moves 1/8 of the context per chip instead of gathering all of it. Option 2 (brick width 1) is
    now the recommended B1 fix.
  - **Block order is DEAD in production at 1080p (host check of `_pick_block`, job 388 logs).**
    At the production layout (W over the size-8 axis) `_pick_block` returns None for every
    deterministic stage -- (21,68,15), (41,68,15), (81,136,30) -- with GNA on or off: a legal block
    needs a volume that is a multiple of 32 and none of those shard widths carries the factors.
    So `op_sp_w_sharded` runs its STRIDED mode there, `DIFFVAE_GNA=1` is a no-op (no block, no
    stride), and the deterministic stages already run true stride-1 attention. The profiled decode
    trees (jobs 351/352, `diffvae_bricked_timing_tree.txt`) contain no `block-permute` span. Block
    order was reachable only for stage 4 at W over the size-4 axis ((9,8,4)) and for stage 5 on the
    block-permute executor -- which Phase 0 replaced and which cannot run in the pipeline anyway.
    **Consequence: retiring `block_permute.py` does not depend on migrating the deterministic
    stages.** Phase 4 can run now against the strided `op_sp_w_sharded`; Phases 1-3 become a
    separate "bricked deterministic stages" project whose case is quality-neutral (both stride 1)
    and speed-only (drop the full-W K/V gather + `wrow` retile, ~290 ms, against the small-window
    gather waste of B3). The earlier "GNA caveat" written here was wrong and has been removed.
  - **History check (git + every broker log):** block order DID work, for stage 5 only. Built
    2026-08-18 (`a10861f3815` .. `de5bda545c7`, "6s 35.7 s -> 25.0 s -> 22.3 s") for the 6-second
    1080p stage 5, where W_local 60 and H 272 admit blocks (6,8,10) / (11,8,4) / (4,8,4); those are
    the only non-None picks in the logs apart from the small decoder-gate grids (11,64,8/16) and
    today's axis-swap run. The picker's constraints (divisor dims, volume a multiple of 32 in
    [128, 512]) are unchanged since that first commit. Stage 5 moved to the bricked executor on
    2026-08-25/26 (`59873aa1713`, `5577f125079`), which is when block order stopped being exercised.
    The deterministic stages were W-sharded from 2026-08-17 but have never produced a block
    (>3300 `_blk=None` log lines at (21,68,15), (41,68,15), (81,136,30); zero non-None).
- [ ] Tests in `tests/unit/test_neighborhood_sdpa.py`: windows (3,5,5) and (3,7,7) at stride 1;
      an odd shard index at W_local 30 (or W_local 15 if option 2); TP with two heads per chip.
- [ ] Gate: all existing `test_neighborhood_sdpa.py` and `test_neighborhood_permute.py` cases still pass.

### Phase 2 -- wire the deterministic stages (`models/vae/diffvae_ltx.py`)
- [ ] `NeighborhoodAttention.forward`: add a `bricked_sp_w_sharded` arm that calls the bricked
      executor with the flat `(1, heads_local, tokens, head_dim)` q/k/v the fused-rope path already
      produces (`heads_presharded=True`, `already_bricked=False`, `stride=(1,1,1)`). RoPE stays in
      natural order before the call; nothing else in the block changes. The executor returns a
      5-D volume (no TP) or `(1,1,sites,C)` TILE (TP); the existing `reshape -> TILE -> proj` tail
      handles both.
- [ ] `DeterministicStages.block_backend(stage)`: return `bricked_sp_w_sharded` for stages > 0
      when W-sharded (`_w_sharded` currently keys off the string `"op_sp_w_sharded"`; make it a
      set of W-sharded backend names).
- [ ] Selection: extend `stages_na3d_backend` in `vae_ltx.py` (currently hard-coded to
      `op_sp_w_sharded` when `DIFFVAE_STAGES_WSP=1`) with a `DIFFVAE_STAGES_BACKEND` override,
      default bricked once Phase 3 passes.
- [ ] Tests: a `bricked` arm in `test_det_nablock_arms.py` (PCC 0.999 vs baseline, plus the timing
      test) at all three STAGES; a `test_decode_full_bricked_matches_replicated` beside the stage-5
      one in `test_diffvae_decoder.py`.

### Phase 3 -- measure and decide
- [ ] `test_det_nablock_arm_timing` for `bricked` vs `flat_seq` at stages 2, 3, 4.
- [ ] One full 1080p decode with the decode tree (`PROFILE=1 run_ltx25_pipeline.sh`); compare
      deterministic-stage totals against the 1767 ms baseline and gen-0 output against Phase 0's.
- [ ] Decision D2: if deterministic-stage time regresses, either keep `op_sp_w_sharded` in strided mode
      (`DIFFVAE_BLOCK=0`, no block permute) for the small-window stages, or invest in the op (e.g. smaller effective
      brick / chunk tuning for small windows). Either outcome still frees block_permute.

### Phase 4 -- delete
Once no configuration sets `DIFFVAE_BLOCK`:
- [x] Python (2026-09-10): `git rm` of `layers/block_permute.py`, `tests/unit/test_block_permute.py`,
      `test_block_permute_device.py`, `test_block_sdpa_op.py`, and the GNA-from-block A/B harnesses
      `tests/models/vae/ab_gna_stage5.py` / `ab_gna_decode.py` (meaningless without a block). In
      `layers/na3d.py`: `_pick_block`, the `DIFFVAE_BLOCK`/`op_block` config, the `q-block-permute` /
      `unblock-permute` spans, the `neighborhood_block=` kwarg, the block-sized `q_chunk_size`, and the
      `DIFFVAE_GNA` block-stride branch removed; `DIFFVAE_GNA_STRIDE` / `gna_stride` (explicit stride)
      kept. `DIFFVAE_GNA=1` / `DIFFVAE_BLOCK=1` exports dropped from `run_ltx25_pipeline.sh` and
      `run_ltx25_diffvae.sh`. `NEIGHBORHOOD_ATTENTION.md` block_permute section marked retired.
- [x] C++ (2026-09-11; rebuilt, `test_windowed_sdpa.py` + `test_neighborhood_3d_sdpa.py` + `test_na3d_op_sp.py` = 129 passed, job 410; one JIT fix on the way -- a leftover ternary line in `windowed_mask_gen.hpp` that the unity host check cannot see, caught by the writer's own JIT command run offline): `neighborhood_block` removed
      from `sdpa.hpp/.cpp`, `sdpa_nanobind.cpp`, `sdpa_device_operation*.{hpp,cpp}` and the
      `operation_attributes_t`; the factory no longer pushes the {bt,bh,bw} slots (reader tail is now
      `w_origin, st, sh, sw`; writer slots 22-24 are the GNA stride, previously 25-27); the block
      section of `windowed_loop_geometry.hpp` (`BlockCoord`, `block_index_of_chunk`,
      `neighborhood_box_block`, `block_query_coord`, `neighborhood_box_k_chunk_range`) deleted; the
      `bt/bh/bw/hb/wb` parameters and `nb_bt != 0` branches removed from `windowed_mask_gen.hpp`
      (five functions), `reader_interleaved.cpp` and `writer_interleaved.cpp`; the packed mask's
      single-window shortcut now requires an unsharded W only. `compute_common.hpp` had no block use.
      Host units syntax-checked clean (unity recipe). `NOTES_windowed_loop_geometry.md` section 5 retired.
- [x] Python-step device checks (jobs 389/390/392/394/395): decode 13924 ms (vs 13814-13993),
      deterministic stages 2296 ms (vs 2291); stage-5 gate on `op_sp_w_sharded` 99.9961 % both axes;
      deterministic-stage arms 99.998 % x3; `test_na3d_op_sp.py` 32/33 -- the one miss
      (`test_na3d_op_sp_w_sharded_matches_host[dims2-kernel2-sp_cols]`, a 12-token shard) fails at
      94 % ONLY under the pipeline's exported `DIFFVAE_*` flags and fails identically on the pre-change
      file; passes with a clean env. Pre-existing, environment-induced, unrelated to Phase 4 (noted,
      not fixed).
- [x] Scripts/docs (2026-09-11): the `DIFFVAE_BLOCK=1` export was dropped with the Python step; the block figure
      went with `windowed_loop_geometry_diagrams.py` / `NOTES_windowed_loop_geometry.md` (deleted in Tier 3).
      Only the untracked root-level `block_permutation_diagram.py`/`.png` remain.
- [x] Gate (2026-09-11, job 411, rebuilt library + kernels): stage-5 gate on `op_sp_w_sharded`
      99.9961 % both axes (identical to pre-deletion); deterministic-stage arms 99.9984/99.9981/99.9981 %
      (identical); decode s34x60 **13859 ms**, deterministic stages 2301 ms (pre-deletion range
      13814-13993 / 2291-2296). Performance-neutral, as predicted. The full pipeline was not re-run
      after the C++ step: the Python step's pipeline is unchanged and the kernels are covered by the
      129-test SDPA suite plus the stage-5 gate; run it before committing if you want the belt too.
      **Phase 4 complete.** `block_permute.py` and the `neighborhood_block` path are gone.

### Phase 5 (optional) -- per-stage brick hoist for the deterministic stages
Brick once at stage entry as stage 5 does (`_brick_activation` / `_unbrick_activation`,
bricked RoPE tables via `rope_tables(..., brick=)`), passing `already_bricked=True`, and unbrick
before each `LinearPixelShuffleUpsample` (needs natural order). Only if Phase 3 shows the
per-block brick/unbrick permutes matter; at deterministic-stage sizes (stage 5 measured 0.48 ms per 52 MB) they
should not.

## Decisions

- **B2 fix (decided 2026-09-10, James):** keep TP over heads for the deterministic stages and
  generalize the bricked executor's head reassembly -- a real site-major -> head-major permute
  before the head all-gather when heads per chip > 1 (mirrors the block-permute executor's unflatten).
  Dropping TP was rejected: up to 4x attention compute per chip (~1.2 s upper bound) to save a
  10-line change.
- **D1 (B1 fix), still open:** swap SP/TP axes for the deterministic stages (recommended), allow
  brick width 1, or extend the planner. Owner: James. Note option 1 loses the W-sharded
  deterministic->stage-5 band handoff (stage 5 cannot move axes: 4 heads do not TP 8 ways), so
  it needs one timing run with `stages_sp_axis=0` to price the extra context gather + reshard.
- **D2 (Phase 3 fallback):** is keeping `op_sp_w_sharded` in strided mode for small-window stages
  acceptable if bricked is slower there, or must bricked win everywhere before deletion?

## Out of scope
- Stage 1 (index 0) stays on the replicated `gather` backend.
- (Superseded 2026-09-11.) `neighborhood_attention_3d_op_sp_w_sharded` and every other executor of
  the general SDPA op's neighborhood mode (`op`, `fused`, `op_sp`, `op_sp_w`, `op_sp_sharded`) were
  deleted once stages 2-5 all ran the bricked executor; `na3d.py` keeps the gather backend, the
  planner, `na3d_torch` and `window_bounds`. The general op's C++ neighborhood mode itself stays.
- The `DIFFVAE_S5_KERNEL` window-override reader is not in the tree (only in `git stash@{0}`);
  unrelated to this plan but will bite any window experiment run alongside it.

## Related notes
- `layers/SDPA_FUSED_VS_NEIGHBORHOOD.md`, `layers/NEIGHBORHOOD_MASK_GENERATION.md`
- `ttnn/.../sdpa/device/kernels/NOTES_windowed_loop_geometry.md` -- deleted 2026-09-11 with the mode it described;
  `MASK_PERSISTENCE.md` is the only note left in that directory
- `DIFFVAE_TIMING_ANALYSIS.md`, `diffvae_bricked_timing_tree.txt` (baseline numbers above)
