# Rebase onto llk_helper_library 107f838f379 — status

Worktree `/scratch/dnijemcevic/tt-metal-rebase` (on /scratch: / has 39 GB free, too little for a second build),
branch `dnijemcevic/model_bringup_rebased`, from `f2e9420851b` (= origin/dnijemcevic/model_bringup, X.3 PASS).
Nothing pushed. `~/repos/tt-metal` untouched (only used, read-only, to run the old build's baseline).
Logs: `/scratch/dnijemcevic/rebase/` (rebase.log, conflicts.log, build*.log, baseline/), tmux session `rebase`.

## STATUS 2026-11-06 05:00 — pushed as dnijemcevic/model_bringup_rebased (owner: 2x2 / 1x4 cases to the QuietBox)
The rebased model passes its end-to-end verification on the 8x4 (details under Verification and Log). I am not
pushing because one item matches a stop rule (a failure I cannot fix at its cause), and a few need your decision:

1. **Stop item — indexer_score's 7 committed 2x2 ring cases** (from the 2x2 O.1 capture) error at fabric init on this
   Galaxy ("Fabric Router Sync: Timeout ... Device 1" on a 2x2 submesh). The OLD build fails the same 7 the same way:
   box, not rebase. Options: waive them for this box (tests.fork_waivers), or give the cases a box guard like the
   other 2x2 cases (they skip with "exact match of requested num devices"). Your call.
2. **tt-d-gen**: the new base writes H2D stream descriptor v3; the spec's serving.server_repo engine (a9701ce6675)
   reads only v2. K.1R passes with tt-d-gen main (d8a194417) at /scratch/dnijemcevic/tt-d-gen-v3 (BRINGUP_SERVER_REPO)
   plus F255 (driver fix, committed). Point serving.server_repo at a tt-d-gen main for this branch? (spec change)
3. **Accuracy: all within limits; one move investigated to its root.** err_state_window_kv rose on layers 20-37 (up to
   37 % of its margin): the seed is ONE MoE routing weight (layer 0, token 10689, one bf16 ulp) that the new build
   rounds correctly and the old build rounded wrongly (float64 check); it spreads through the window and sparse top-k.
   Both builds are bit-deterministic. err_final_hidden 0.14481 (old 0.14476), top1/top5 1.0, logits tail pcc better.
4. **Speed**: warm chunk 226.0 ms (old 245.2), full prefill 56k 2503 ms (old 2891), device ms/chunk 165.0 (old 162.6).
5. Pending decisions from the sync: fork moe_grouped_topk vs upstream #60550 (decided 2026-11-06: fork retired, the
   stock op); prepare_chunk_recurrence unsynced
   (#63309; DS41 does not use it); re-check the stock combine_fabric2d and stock moe_grouped_topk merges (GLM-5.3 /
   Kimi paths, not run here); knowledge staleness edits uncommitted until you read
   /scratch/dnijemcevic/rebase/staleness_report.md.
6. Not rebase work: the Galaxy ledger's O.1 never ran; 8x4 fork cases for its 195 calls are new test work.
7. Running meanwhile: the full fork suite on the rebased build (/scratch/dnijemcevic/rebase/forksuite_new/), so you
   have the complete list; anything new it finds is added below.
Branch commits on the new base: 70075e3e7a1 906d3fbe4b7 3b9ed4635b1 020b0fb7556 47fc755acfe d8b010327de dd280fa68d9.

## For the QuietBox (2x2 Blackhole): tests the Galaxy cannot run (keep this list current)
1. The full fork suite (`python -m models.demos.common.bringup.testing.fork_suite`): every 2x2 / 1x4 case the
   Galaxy skips ("exact match of requested num devices", 110 in DS41's committed cases alone).
2. indexer_score's 7 2x2 ring cases (fabric init timeout on a Galaxy 2x2 submesh, old build too).
3. combine's ernie45 1x4 model case (the Galaxy cannot open a 1x4 mesh); say if the QuietBox cannot either.
4. DS41's committed fork cases: `fork_cases --capture models/demos/deepseek_v41_flash_d_p/bringup/results/fork_calls.json
   models/demos/deepseek_v41_flash_d_p/bringup/results/fork_calls_block.json --model deepseek_v41_flash_d_p --run-tests`.
5. moe_grouped_topk fork retired (owner 2026-11-06): DS41's router accuracy case against the STOCK op (the test that
   replaces the fork's case; path added when the retirement lands). Fails = fork the latest stock and re-apply only
   what the test shows missing.
6. Stock moe_grouped_topk's carried fp32 weight normalization (66e17107625, now behind its fp32_selection option;
   #60550 always selects in fp32): GLM-5.3's router component tests (tests/bringup/test_c_kda_moe_router.py,
   test_c_dsa_moe_router.py) and DS41's stock-router case (item 5) with fp32_selection on and off. Off passes = drop
   the carried change (stock = upstream), remove the option from the callers; only on passes = keep it (an upstream
   PR candidate) (owner 2026-11-06).
7. prepare_chunk_recurrence (GLM-5.3 only): GLM's KDA attention component tests (tests/bringup/test_c_kda_dense_attention.py,
   test_c_kda_moe_attention.py) with GLM_KDA_DECAY=kernel (the new stock op, #63309). Pass (worst head <= 0.05) =
   retire the fork, GLM on stock; fail = fork the latest stock and re-apply precise_gate_factors on #63309's batched
   pipeline (owner 2026-11-06).

8. fabric_all_gather fork retired if the 8x4 A/B passes (stock high_bw_all_gather = the same program, #61387):
   glm53_flash_d_p_lb's all-gather MoE (GLM_EXPERTS_MODE=ag) on stock; the fork kept an EMULE_FABRIC_2D routing
   path, stock routes via worker_routing_utils (owner 2026-11-06).

9. fabric_all_gather C++ fork retired (8x4 A/B: DS41's 7 calls bit-exact, stock +0.15 ms a chunk, owner accepted):
   its 19 bit-exact model cases now test the stock high_bw_all_gather:
   models/demos/deepseek_v41_flash_d_p/tests/test_all_gather_stock.py, models/demos/glm53_flash_d_p/tests/test_all_gather_stock.py.

## Tracked, not queued (owner 2026-11-06)
- Galaxy O.1: 8x4 fork cases for the model's 195 distinct fork calls (16 forks; mhc_pre_xing 81+2,
  update_padded_kv_cache 25, prepared_program 17+17, rms_norm 9, rope 8, fabric_all_gather 7, ...). Estimate: writing
  ~half a day to a day (agent, device iterations); one run ~1-1.5 h on the 8x4. Run as the ledger's O.1 through the
  orchestrator on the rebased branch when the owner says.

## Bases
- Old base: `6b80daf43e3` (old llk_helper_library tip; llk_helper_library was force-pushed since, so
  `git merge-base` gives an older main commit, 48444d1f8a1 — wrong).
- New base: `107f838f379` (origin/llk_helper_library, 2026-11-02; main as of 2026-10-30 + helper commits), 720
  commits ahead of 48444d1f8a1.
- Branch: 2473 commits on the old base; 29 files changed on both sides.

## Log
- 2026-11-05 23:15 — baseline fork suite (old build, `~/repos/tt-metal`) started; aborted after the first fork:
  **F254** `fork_suite` exits when a fork has no `tests/source.yaml` (7 forks have no upstream source: ckv_bfp4_ttnn,
  combine_rf, fabric_reduce_scatter_ttnn, mhc_post_ttnn, mhc_pre_ttnn, pair_pool_ttnn, prepared_program). Re-run
  with an in-memory patch (`/scratch/dnijemcevic/rebase/fork_suite_nomanifest.py`); fixed on the rebased branch.
- 23:20 — rebase replayed (2473 commits, ~4 min) taking the branch's side on each conflict, then:
  - five merge commits (lowbit, bfp4kv, topk_large_indices, owner's MoE work) are linearized by the rebase and their
    resolutions were lost on 22 branch-only files; restored from the old tip (commit "restore what the linearized
    merges resolved"). Check: every branch-only path now equals the old tip; every other path equals the new base.
  - the 29 files both sides changed: 3-way merge (old tip / old base / new base): 22 clean and identical to the
    replay, 1 clean (ADDING_A_PREFILL_MODEL.md), 6 conflicts:
    - `prefill_producer.py`: kept ernie45 on the GQA KV path + upstream's new `num_ranks` argument.
    - `deepseek_v3_d_p/tt/moe/init_helpers.py`: upstream docstring + the branch's TT_MOE_COMBINE_OVERLAP=0 switch.
    - stock `combine_fabric2d_program_factory.cpp` (not called by DS41; GLM-5.3/Kimi): upstream now takes the
      channel semaphores from the caller; the branch made them once per mesh. Kept both: the caller's when given,
      else the branch's per-mesh ones. **Owner re-check** (changed under us).
    - stock `moe_grouped_topk` (3 files; DS41 uses its own fork ttnn.bringup.moe_grouped_topk; GLM-5.3 uses the
      stock op with fp32_selection): upstream #60550 now always selects in full fp32; the branch's fp32 weight
      normalization (66e17107625) is not upstream. Being ported onto upstream's code. **Owner re-check**.
- 23:30 — fork sync to the new base (full sync, as the last rebase) started for the 13 forks whose source moved.
- 23:30 — build started (same options as the old tree: Release, no tests).
- 23:45 — committed: the shared-file merge (906d3fbe4b7), F254 (3b9ed4635b1), get_forwarding_link_indices takes the
  mesh device (#61503) in 6 Python callers (020b0fb7556; DS41 ccl.py, MoE routing_stream, matmul_reduce_scatter,
  combine_rf / fabric_all_gather_ttnn planners). Not ours: upstream's qwen38_flash_next gr_fold still calls the old
  signature. First build attempt failed on a half-cloned umd submodule (my killed first try); restarted.
- 00:10 — fork sync committed (47fc755acfe): 12 of 13 forks fully synced (sdpa 9 commits, topk_large_indices with
  upstream's split top-k race fix #62675 — may move DS41's index top-k ids where the race hit; rotary gains
  row_major_output, default off). **Owner decisions pending** (DS41 behaviour unchanged either way):
  - fork moe_grouped_topk: only #62567 taken; #60550 (upstream's own always-fp32 selection) conflicts with the
    fork's fp32_selection / fp32_activation kernels, which DS41 uses. Kept the fork's kernels.
  - prepare_chunk_recurrence (not used by DS41): unsynced; #63309 rewrote the kernels around its
    precise_gate_factors.
- 00:15 — ports committed (d8b010327de): removed compute init shims (#60881) in 5 forks' kernels;
  fabric_all_gather_ttnn C++ (#61503 / #61262).
- Baseline fork suite (old build) so far: one ERROR — combine's ernie45 1x4 model case cannot open a 1x4 mesh on
  the Galaxy (system mesh gives 2x1). Box mismatch, not the rebase; on the owner's list.
- 23:35 — build OK (Release, same options as the old tree); venv made (fresh uv cache on /scratch: ~/.cache/uv is
  damaged like the pre-commit cache was).
- 23:38 — baseline fork suite stopped after 3 of 26 forks (~25 min for 3: hours in all). The e2e model verification
  goes first; the baseline is re-run afterwards only for forks that fail on the rebased build. Verification driver
  `/scratch/dnijemcevic/rebase/verify/run.sh` (results and logs in that folder, progress.log): X.3's gate (ladder
  s56320, clean wall, profile, positions), O.1 (ladder at rung last + fork capture, fork cases --run-tests), K.1R's
  five contract tests. Then the full fork suite.

## Verification (rebased build, 8x4)
- X.3 ladder s56320: PASS, 491.6 s (old 492 s). 104 limited metrics, none out of limit. err_final_hidden 0.14481
  (old 0.14476), top1 / top5 1.0, err_layer_* x1.00-1.02, compress_kv / index_k / topk_idx x1.00.
  **Investigating**: err_state_window_kv rose on layers 20-37 (x1.16-1.48; L20 0.0994 -> 0.1414, 37 % of its margin;
  7 metrics move > 1/4 of their margin; encoder layers x0.98-1.10). Ruled out: torch (2.11.0 both), host bfp8/bfp4
  encode/decode (bit-identical old vs new build), weight caches (none rewritten). Next: the same ladder on the old
  build now (is the metric deterministic run to run?), then bisect the window-KV write path.
- X.3 clean wall: warm chunk 226.0 ms (old 245.2 ms), spread 15.6 ms (old 25.2).
- Knowledge staleness (step 8): 223 entries re-checked, 49 rewritten, 15 removed, 22 need a device check. Edits are
  in the worktree, **uncommitted until the owner reads** `/scratch/dnijemcevic/rebase/staleness_report.md`.
- X.3 profile + positions: PASS. full prefill 56k 2503 ms (old 2891), device ms/chunk 165.0 (old 162.6, +1.5 %),
  op rows 1290 = 1290, timeline ok; positions 0..204800: 219-252 ms (old 255-286), no new programs.
- O.1 capture (ladder rung last, golden prefix + last chunk): PASS, 351.5 s; 104 limited metrics in limit. Not
  comparable with X.3's s56320 rung (rung last starts from the golden prefix). Window-KV question still open: after
  the driver, re-run the s56320 ladder on the old build (deterministic?) and on the new build (reproducible?).
- 01:46 driver done.
  - K.1R contract + kv_layout: PASS.
  - acks: the waived check only (layer 18 window_kv last block 0.98980 < 0.99, the same value as P.23; matches
    tests.unit_waivers): PASS under the existing waiver.
  - runner contract / smoke: FAIL. Cause: tt-d-gen engine (a9701ce6675, nested tt-metal f5985995a23) expects
    H2DStreamServiceDescriptor v2; the new base writes v3 (D2H v4; bumped in #58175). tt-d-gen main (d8a194417,
    tt-metal pin 58afa857945) has v3: building it beside the old one (/scratch/dnijemcevic/tt-d-gen-v3), then re-run.
  - O.1 fork cases: all 195 calls uncovered (sigs changed under the new base): re-key in progress (step 5).
    indexer_score's 2x2 cases error at fabric init (router sync timeout, Device 1); to check on the old build.
- 01:50 window-KV A/B started: s56320 ladder on the old build, then again on the new build (verify/ab_*.log).
- 02:00 the "195 uncovered" is not the rebase: sigs are computed identically; the committed capture is the 2x2 O.1
  (Oct 27-29, 121 calls), the fresh one is the 8x4 with a month of model changes (P.5-P.32, FIX.14-22). The Galaxy
  ledger's O.1 never ran (next after X.3): new 8x4 cases are new test work, not part of the rebase (for the owner).
  The committed keys still cover the committed capture on the rebased tree (0 uncovered, lint 0). Rebase check: run
  those cases (--run-tests) on the new build, and on the old build for any that fail (verify/cases.sh).
- 02:10 tt-d-gen main built at /scratch/dnijemcevic/tt-d-gen-v3 (d8a194417; nested tt-metal 58afa857945: H2D v3 / D2H
  v4 = the new base). README: /scratch/dnijemcevic/tt-d-gen-v3-build/README.md (tt-blaze's new BFP packer needs
  OpenMP: clang-20 has no libomp here, built with g++-12; no source changed). Runner contract / smoke re-run with
  BRINGUP_SERVER_REPO=/scratch/dnijemcevic/tt-d-gen-v3 (the spec's serving.server_repo stays the old engine, which
  the old branch needs; a spec change for the rebased branch is the owner's call).
- 01:58 window-KV A/B, old build: bit-identical to X.3 on all 40 err_state_window_kv (old build deterministic). So the
  new build's rise (L04 on, up to +48 % at L31; L00-L03 identical) is real if ab_new reproduces it. Root-cause hunt
  started (code side: the window-KV write / readback path vs the base's changes; then an op micro-test old vs new).
- 02:20 window-KV analysis (code + the A/B jsons): the served window-KV copy is a bit-exact bf16 copy (no rounding on
  that path). Old vs new first differ inside layer 2's compressor (compress_kv_L02 0.11495087 -> 0.11495103; L00-L01
  and window_kv L00-L02 bit-identical): a ~1e-7 seed that grows through depth. err_layer averages 5120 rows (+1-2 %);
  err_state_window_kv samples 128 rows on one chip, so it shows the trajectory change more. pcc_logits_tail 0.99312 ->
  0.99351 (no worse). Seed candidates: pair_pool (first used at layer 2) via the rewritten BH SFPU exp (#58929,
  #62041, #58179, #60714, #61824, #62120), compressor matmul (#61818, #62351), bfp4 pack reconfig (#61006).
  Op micro-test old vs new (/scratch/dnijemcevic/rebase/wkv/) queued after the runner re-run.
- 02:06 ab_new: identical to the first new-build run (only chunk timings differ): the new build is deterministic too;
  the window-KV move is a reproducible build-dependent trajectory change, within every limit.
- 03:13 committed fork cases (2x2 O.1 capture, 121 calls, 11 forks, --run-tests) on the new build: 131 passed, 110
  skipped (every skip: "exact match of requested num devices" = 2x2 / 1x4 cases on the 32-chip Galaxy), 7 errors: all
  indexer_score 2x2 ring cases, fabric init "Fabric Router Sync: Timeout ... Device 1" on a 2x2 submesh. Same 7 on
  the old build queued (idx_old) to tell box from rebase.
- 03:25 runner contract / smoke with tt-d-gen main: the engine attaches (descriptor v3 matches) but our driver passed
  chunk_size, which tt-d-gen 1d4deb1a1 (#1766) dropped from device_prefill_pipeline (taken from RuntimeConfig; K=1
  keeps the single-chunk wire). F255: dgen_prefill_driver.device_pipeline_args passes chunk_size only to a build that
  declares it (selftest updated, passes). Re-run queued after the micro-test and idx_old.
- 03:40 op micro-test old vs new (1x1, model per-chip shapes, seeded): compressor projection matmul, pair_pool, exp
  chain, norms, ropes, wk matmul, ckv_bfp4 rows, bfp_round, kv norm / rope, bfp8 region write, to_rm: ALL
  bit-identical. Next: per-position dump of layers 0-2 states over all chunks (layers 0/1 are window-only, so a
  difference in an earlier chunk would not reach the last chunk's L00/L01 metrics but would reach compress_kv_L02).
- 03:28 indexer_score's 7 2x2 ring cases on the OLD build: the same fabric init timeout ("Fabric Router Sync:
  Timeout ... Device 1"). Box, not rebase (2x2 ring submesh on this Galaxy); on the owner's list with combine's 1x4.
- 03:50 K.1R complete on the rebased build with tt-d-gen main: runner smoke PASS (Paris through tt-d-gen, 696 s);
  runner contract: only the waived lines (engine header + layer 18 window_kv 0.98980 < 0.99, same value as the old
  branch) = PASS under the existing waiver. F255 committed (dd280fa68d9). Owner: the spec's serving.server_repo
  (/scratch/$USER/tt-d-gen, a9701ce6675) cannot serve the rebased branch; point it at tt-d-gen main (here
  /scratch/dnijemcevic/tt-d-gen-v3) when this branch replaces the old one.
- 03:50 per-position dump of layers 0-2 (old vs new, s56320) running to find where the builds first differ.
- 04:05 per-position dump (old vs new, s56320): bit-identical through chunk 1. The single seed: token 10689 (chunk 2)
  in layer 0's output (one row; window_kv_L00 identical, so the attention input is the same). It spreads through layer
  1's 128-token window and from layer 2 through the sparse top-k (topk_idx_L02 differs on 22k rows). Drilling into that
  token's stages in layer 0 (routing near-tie vs race vs dispatch/combine).
- 04:35 **window-KV move explained (resolved)**. Stage capture of layer 0, chunk 2: every step is bit-identical
  (attention, mHC, norms, shared expert, expert ids) except ONE routing weight from the MoE block's fused-head router:
  token 10689, expert 126: old 0.27539, new 0.27734 (one bf16 ulp; same 6 experts). The float64 host router on the
  identical input gives 0.2763730, above the bf16 midpoint 0.2763672: correctly rounded it is 0.27734 = the NEW build.
  The old build rounded it the wrong way. Cause: the router's fp32 SFPU chain (softplus / sqrt; upstream rewrote
  softplus_init / the fp32 body: #58179, #61824) moved a last bit across that rounding boundary. That one token's
  difference spreads through layer 1's 128-token window and the sparse top-k from layer 2 on, which is what
  err_state_window_kv (128 rows, one chip) shows; every limit holds, err_final_hidden x1.0003, pcc_logits_tail
  0.99312 -> 0.99351. Not a regression; no fix needed. Tools: /scratch/dnijemcevic/rebase/wkv/ (plugin, compare).
- 05:00 owner: ENGRAM_TABLE is mmap (= MADV_RANDOM, the former 'random') | resident; the read-ahead mapping removed;
  default resident, tests mmap; X.3's gate sets DS41_ENGRAM_TABLE=resident (its wall ran with tables from disk:
  245.2 old / 226.0 new, vs the P tasks' resident ~195-207). Committed 9610ac56304; CPU mode test passes. The
  resident X.3 gate is to be re-run on the 8x4 after the fork suite.
- 05:00 owner: 2x2 / 1x4 cases cannot run on the Galaxy; they will be run on the QuietBox (2x2 Blackhole). That was
  the only stop item: push authorized. Pushed dnijemcevic/model_bringup_rebased (new branch, no force).
- 09:14 full fork suite (8x4, rebased build): 3340 passed, 219 failed, 308 skipped; classification (box / rebase /
  pre-existing, old-build reruns) in progress.
- 09:39 **stock hybrid combine (combine_fabric2d, the merge kept the branch's per-mesh semaphores + upstream's #62312
  VC channels): test_hybrid_routed_expert_combine.py HANGS in its first case on the rebased build** (host waiting on
  a completion-queue event 19+ min; stopped). Not on DS41's default path (deepseek_v3_d_p MoE / gemma4). Old build
  run queued; merge under review. The push to dnijemcevic/model_bringup waits for this.
- 09:55 test_hybrid_routed_expert_combine.py hangs the same way on the OLD build (first case, completion-queue wait;
  stopped): pre-existing, not the merge. Not on DS41's default path.
- 10:05 hybrid combine: review: the merge is consistent (the only caller passes channel semaphores; the per-mesh
  path is unused); committed taking upstream's file (7e2f8146513, side worktree). The hang (old and new builds alike)
  stalls in the real run after the collect-only pass: next, the first case without the precompile plugin on both
  builds (harness vs op). Pre-existing failures are fixed or waived like any other (owner rule).
- 10:05 sdpa row_bfp4 H=32: SFPI 7.93.0 internal compiler error (H=64, DS41's 8x4 call, builds): a rebase
  regression for DS41 on a 2x2 (32 heads a chip); kernel workaround in progress, upstream report drafted for the owner.
- 10:45 fork-suite classification (219 = failed 47 + error 158 + not_run 7 + collect 7; old-build reruns):
  BOX 162 (fabric init timeout on 2x2 / 1x4 / 2x4 / 4x1 submeshes of the Galaxy, old build too; ernie45 1x4;
  indexer_score source grid-dependent; topk faster than its two-sided band), PRE-EXISTING 40 (missing `eval` module
  = tt_ops_code_gen submodule not checked out: 34 incl. 7 fake not-run from a fork_suite parsing bug; flat_routed_expert
  perf / planner 4; moe_ag perf 2), REBASE 17 (moe_grouped_topk 12: retired; rotary 1: test regex after #63498, patch;
  sdpa row_bfp4 H=32 4: SFPI 7.9x internal compiler error triggered by the new base's compute headers, not the fork's
  changes; -O levels don't avoid it). 70 source tests "missing" vs stale baselines (upstream renames): re-record.
  Patches / repro: /scratch/dnijemcevic/rebase/fsclass/.
