# deepseek-ai/DeepSeek-V4.1-Flash bring-up: breadcrumbs

Append-only log, one section per task attempt: what was done, decisions and why, gotchas, the re-run command, the verdict.

Older entries (R.2 reference, attempt 1 (2026-10-14) .. P.8 perf (attempt 2): re-verified accuracy with the change on; stopped at the first failing swap test) are in BREADCRUMBS_archive_2026-10.md (moved 2026-11-06: the repo's 500 KB file limit).

## P.9 perf (attempt 1): weighted combine with bf16 running sums, the chunk in one pass (owner comparison)
- Setting COMBINE_ACC {fp32, bf16} (tt/settings.py; default fp32 = today's path, as the owner's text says; F104 sets
  bf16 on accept), MAX_PRECISION fp32. Used only under EXPERTS_COMBINE=weighted.
- Fork combine_rf: `sparse_combine_rf_prepare(acc_dtype=None | ttnn.bfloat16)`: CB_ACC Float16_b with 2048 B tiles
  instead of fp32 4096 B; same kernels, same det order (the packer's L1 accumulation adds the weighted fp32 DEST row
  onto the bf16 sum). `accumulator_bytes(acc_dtype=)`, `acc_page()`, the arena refusal ("bf16 tiles", s_fit / a_min)
  size with it. Default None = fp32: programs unchanged. CHANGELOG / README / INDEX updated; Python only (no build).
- Model: tt/weighted_combine.py `WeightedCombine(acc_dtype=)` picks the fewest passes with it (DS41-69's sizing) and
  logs `EXPERTS_WEIGHTED S=1024 passes=1 x 1024 acc=bf16`; tt/experts.py `combine_acc` (build_experts arg / setting)
  asserts one pass for bf16 and logs the pass count per layer. 2x2: 9 accumulators, 114 tokens x 5 bf16 tiles =
  1140 KB each, arena 1314 KB (ring depth 4, local ring 8 kept: DS41-70).
- Tests: fork tests/unit/test_acc_dtype.py 4/4 (sizing: one call holds 1143 bf16 vs 567 fp32 tokens at 9
  accumulators / 1356 KB; refusal 1024 tokens in 1156 KB: bf16 "<= 963 or >= 10 accumulators", 963 builds, 972 refused,
  fp32 "<= 477"; standalone combine precision, 6 bf16 rows a token at 2^-6..2^6 vs the float64 sum: fp32 rel 7e-9, bf16
  rel 2.71e-3 / 2.79e-3 (one rounding of the exact sum 1.66e-3), bias -1.2e-5 / -1.5e-5, worst row 3.9e-3); whole fork
  suite 8/8 (test_column_planner 4/4 = baseline). Model tt/perf_experts.py: test_weighted_combine_routes_bf16 (S=1024,
  local / remote / random: one pass, completes) 3/3; test_combine_acc (layers 0 / 24, real weights, chunk 2048).
- Measured, test_combine_acc (synced experts call per layer; float64 on 128 tokens; bf16 vs fp32 on all 2048 rows):
  shipped glm precision (NUMERICS native, FFN_NORM_DTYPE bf16, CCL_IMPL fabric_bf16): fp32 2 x 512 5.74 ms, bf16
  1 x 1024 4.29 ms (L0 and L24); vs float64 L0 fp32 0.02015 / bf16 0.02018, L24 0.02059 / 0.02061; bf16 vs fp32
  0.0036 (bias -1.8e-4). Full precision (EXPERTS_PRECISION=full, CCL_IMPL=fabric): 6.09 / 6.05 -> 4.96 / 4.95 ms;
  vs float64 L0 fp32 0.00291 / bf16 0.00406 (bias +2e-5 / -2.0e-4), L24 0.00296 / 0.00413; bf16 vs fp32 0.00287 /
  0.00291. Determinism: bf16 twice 0 of 10.5M values differ (both layers).
- Per op, layer 0, glm (test_combine_acc ops): fp32 total 5372 µs = fused 2 calls 4107 + planners 807 (2 passes on
  2 cores); bf16 4236 µs = fused 1 call 2201 + planner 1573 (1024 tokens on one core). The planner is now the
  biggest non-expert piece (DS41-73: next lever).
- Accuracy, change on (DS41_COMBINE_ACC=bf16): the 16 frozen test files in gate order, each once, run one by one
  with scripts/run_safe_pytest.sh (the measure part `test_c_${t}_experts` is one 8-file command, ~19 min, over the
  10-minute foreground limit): all PASS; they run EXPERTS_COMBINE=separate (MAX_PRECISION), so COMBINE_ACC is not on
  their path (test_c_swa_experts vs CPU rel 0.00224936, the recorded value; swa_12 swap_ffn_hc WAIVED as before).
  Ladder `last` via `orchestrator measure --task P.9 --part last`: rc 0, min layer PCC 0.99680 (L03) (P.8 on: 0.9968).
- Plain profile via `measure --part profile` (the tool moved the foreground call to background at its 600 s limit; it
  ran to the end): device_ms_total 251.86 (P.8 on 258.1), device_ms_experts 38.91 (P.8 on 45.3; gate < 45.3),
  chunk_wall_ms 241.0, profiled_programs 11772, pcc_chunk_out 0.99977.
- Re-run: on `DS41_COMBINE_ACC=bf16`, off `DS41_COMBINE_ACC=fp32` in front of each gate command; A/B:
  `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  "models/demos/deepseek_v41_flash_d_p/tt/perf_experts.py::test_combine_acc" -k "not ops"` (ops: profiler env);
  fork: `scripts/run_safe_pytest.sh --run-all --no-precompile ttnn/ttnn/bringup/combine_rf/tests/unit`.

## P.10 perf (attempt 1): routed expert weights in bfp4 (the checkpoint's fp4 width) vs bfp8 (owner comparison)
- Setting EXPERTS_IMPL: default bfp8 -> bfp4 (agent rule 7 / F117: fp4 -> bfp4), MAX_PRECISION stays bfp8 (exact), so
  the component / swap tests are unchanged (they run bfp8). Description in tt/settings.py and tt/experts.py updated.
- What bfp4 broke: nothing on the model's path. The flat plan changes with bf4 tiles: `nu=3 np=0 mt=1 -> mt=2`,
  flat arena 1156 -> 948 KB, same 48 gate/up + 32 down cores, same 10 free cells / 9 accumulators, weighted combine
  still ONE pass x 1024 (arena 1314 KB, set by the bf16 accumulators). Logged per layer (EXPERTS_FLAT,
  EXPERTS_WEIGHTED). One latent bug fixed: tt/weighted_combine.py:weighted_combine cached its state by (mesh, kwargs)
  only, so a process building bfp8 then bfp4 experts (an A/B) would reuse the bfp8 plan's core layout; the key now
  includes the flat plan.
- Measured, tt/perf_experts.py::test_weights_bfp4 (new; synced whole experts call, random inputs, default path:
  flat + weighted + COMBINE_ACC bf16 + glm + native + ffn_norm bf16 + fabric_bf16; 2x2, chunk 2048):
  L00 bfp8 4.47 ms, bfp4 3.47 ms; L24 4.47 / 3.48 ms (-22 %). Load 37-44 s a layer either way.
  vs float64 with the exact fp4 weights (128 tokens): L00 bfp8 rel 0.0202 bias +0.0132, bfp4 rel 0.1843 bias +0.0373;
  L24 bfp8 0.0206 / +0.0133, bfp4 0.2050 / +0.0399. bfp4 vs float64 with bfp4-rounded weights (`_bfp4_round`)
  L00 rel 0.0202 bias +0.0131: the device path is as exact as bfp8's; the rest is bfloat4_b's rounding of the weights
  (float64 alone: bfp4-rounded vs exact weights rel 0.179, bias +0.024). bfp4 vs bfp8 all rows rel 0.180 / 0.200.
  Determinism: bfp4 twice 0 of 10.5M values differ. Per layer at load: 16.4-16.6 % of values changed, worst matrix
  rel 0.118-0.124.
- Final check `python -m models.demos.common.bringup.orchestrator measure --task P.10` (quick, change on): rc 0, worst
  layer PCC 0.99530 (L03; P.9 on 0.9968), pcc_chunk_out 0.99943, chunk_wall_ms 238.3 (P.9 237.1: the wall did not move),
  host_transfers_per_layer 0. Layer PCCs L00 0.99937, L01 0.99903, L02 0.99787, L03 0.99530, L14 0.99617, L20 0.99929,
  L21 0.99689, L24 0.99943. Not run by me: the plain profile (device_ms_experts gate < 38.9; P.9 38.91; expected about
  -1 ms a layer x 8 from the timed call) and the frozen tests (pinned to bfp8 by MAX_PRECISION).
- Re-run: on `DS41_EXPERTS_IMPL=bfp4`, off `DS41_EXPERTS_IMPL=bfp8`; A/B: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh
  --run-all --no-precompile "models/demos/deepseek_v41_flash_d_p/tt/perf_experts.py::test_weights_bfp4" -k "time"`
  (ops: with the profiler env of testing/profiler.py).

## P.11 perf (attempt 1): the weighted combine's planner on several cores (column multi-core planner) vs one core
- Setting COMBINE_PLANNER {one, multi} (tt/settings.py; default one = today's program, as the owner's text says; the
  orchestrator sets both sides). Not a precision switch (bit-identical tables), so not in MAX_PRECISION. Used only under
  EXPERTS_COMBINE=weighted: tt/experts.py `combine_planner` -> tt/weighted_combine.py `WeightedCombine(planner=,
  planner_cores=32)`; each pass's planner on its own 32 cores (disjoint: run_planners merges the passes' programs, now
  with their semaphores). Logged per layer: `COMBINE_PLANNER=multi (multi: 32 cores x 64 tokens a pass)`.
- Fork combine_rf: `combine_planner_prepare(axis=0, cores=[...], tile_in=True)` = new kernel `_PLANNER_MCC`: Marko's V2
  multi-core structure (counting sort into per-chunk buckets of final table words, count exchange, per-expert chunk
  bases, 1/M of the row tiles pulled per core, atomics onto core 0's plan) with the one-core planner's column / pass
  options (id_offset skip, tokens / token_base, uint32 or uint16 TILE ids read as faces 0 / 2, det). Chunks = 32-token
  multiples inside one home (`_mcc_geometry`: M = N ceil(S / CT)), so chunk order = the one-core (home, token, k) scan
  and every expert's rows land in the same order. det: the chunks of this chip's home tokens build their tokens' order
  lists themselves (a row's rank = the rows of the chunks before it, sent with the counts; source schedules from the
  totals), the accumulators' owners (core a % M) assemble the pages. The tails past the planned rows are written as
  zeros (the one-core planner writes the whole page). Default cores=None: the one-core program unchanged; the
  whole-mesh planners untouched. CHANGELOG / README / INDEX updated; Python + runtime-compiled kernel (no build).
- Tests: fork tests/unit/test_column_planner_mc.py 20 passed / 12 skipped by design: one-core vs multi-core, every
  output word (counts, regions, whole token index and table, plan, det pages) identical per chip (both columns), S 512 /
  1024 one pass, 1024 in passes of 512 and of 342, det / arrival, 32 / 8 cores (8 < 9 accumulators), uint32 / uint16
  ids, uniform / hot / one-column routings, outputs pre-filled with garbage. Fork suite tests/unit: 28 passed (the old
  8 = baseline, option off).
- Measured, tt/perf_experts.py::test_combine_planner (default path: bfp4, COMBINE_ACC bf16, glm, native, ffn_norm
  bf16, fabric_bf16; 2x2, chunk 2048; same random inputs): planner op 1572.7 -> 68.8 µs (device, profiler op mode, L00
  and L24); experts ops total 3427.5 -> 1912.4 µs (L00), 3425.5 -> 1927.1 µs (L24); synced experts call 3.66-3.68 ->
  2.10-2.16 ms; experts output multi vs one 0 of 10.5M values differ (both layers).
- Phases (run.tdbg, cycles at ~1.31 GHz, chip 0): chunk scan + sort ~20K, exchanges + slices ~15K, det ~50K on the 16
  home-chunk cores (count sums 10K, 2 schedules 10K, order lists ~20K, owner pages ~9K). DS41-75: scalar L1 accesses
  cost ~10-25 cycles (counters moved to local arrays, core 0's zeroing to local NoC copies: 104K -> 90K). Next lever
  if wanted: the schedules on idle cores (8 per det core on 8x4), not worth it on 2x2 (~7 µs).
- Final check `python -m models.demos.common.bringup.orchestrator measure --task P.11` (quick, change on): rc 0, layer
  PCCs equal to P.10 (L00 0.99937, L01 0.99903, L02 0.99787, L03 0.99530, L14 0.99617, L20 0.99929, L21 0.99689, L24
  0.99943), pcc_chunk_out 0.99943, chunk_wall_ms 232.1 (P.10 238.3), host_transfers_per_layer 0. Not run by me: the
  frozen tests (MAX_PRECISION runs EXPERTS_COMBINE=separate: the planner is not on their path) and the plain profile
  (device_ms_experts < 38.9; expected about -1.5 ms a layer x 8).
- Re-run: on `DS41_COMBINE_PLANNER=multi`, off `DS41_COMBINE_PLANNER=one`; A/B: `PYTHONPATH=$PWD
  scripts/run_safe_pytest.sh --run-all --no-precompile "models/demos/deepseek_v41_flash_d_p/tt/perf_experts.py::
  test_combine_planner" -k time` (ops: with TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
  TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000, -k ops); fork:
  `scripts/run_safe_pytest.sh --run-all --no-precompile ttnn/ttnn/bringup/combine_rf/tests/unit`.

## P.12 perf (attempt 1): the shared expert on the flat routed expert (bfp8, always-on) vs the ttnn.linear chain
- Setting SHARED_EXPERT_IMPL {linear, flat} (tt/settings.py). Default kept `linear` as the owner's text says (P.9 /
  P.11 precedent; the orchestrator sets both sides, F104 flips it on accept). MAX_PRECISION: linear (exact bf16 weights),
  so the component / swap tests are unchanged.
- flat (tt/shared_expert.py `_load_flat`, `_flat_partial`): Marko's K3_SE=flat with our TP layout: the chip's
  intermediate slice (2x2: 1152 columns of 2304) as SE_PARTS = 1 always-on expert of ttnn.bringup.flat_routed_expert
  (counts [1024], regions [0]), every own token routed to it; weights bfp8 as stored [N, K] (weights_nk, DS41-35), laid
  out once at load. The op refused I = 1152: se6_dcompute.cpp static_assert `kt_d % 8 == 0` (down K in blocks of 8
  tiles), so the slice is zero-padded to 1280 (DS41-76; 4x4 / 8x4 / 2x4: 576 -> 768). Precision follows the routed
  experts' EXPERTS_PRECISION (glm: LoFi, bfp8 x / h tiles; full: MATMUL_FIDELITY, bf16 tiles), fp32 DEST in gate/up and
  down, y bf16 TILE written in ccl.partial_dtype (bf16 under fabric_bf16, DS41-67: no typecast before moe_add).
  The down coordinators' done words are shared by every layer's shared expert (one L1 tensor, `_DONE`).
- x: under EXPERTS_COMBINE=weighted + NUMERICS=native the shared expert reads its own rows (r S + t) by a load-time token
  index from the routed experts' gathered row-major q(x) [R S, H] (TtExperts.keep_rows / take_rows; the block sets
  keep_rows, the shared_expert step takes and frees it): no layout conversion of its own (bit-identical to the own-x
  path, SHARED_ROWS). Without rows: own to_layout (56 us). NUMERICS=official: own q(x) rows + the op's fp8_act_quant
  with unit routing weights (that option refuses a token index); runs (7.9 ms, the composed fp8_qdq of x dominates).
- Hidden? Not folded, not overlapped (owner asked to prefer it only when no more work): folding into the routed
  experts' program needs one weight dtype for both (routed bfp4 since P.10, shared bfp8; the op takes one) and the
  combine planner to place extra ids; overlapping with the dispatch needs sub-devices. What K3 hides under the dispatch
  (the x untilize) costs nothing here: the gathered rows are reused.
- Measured, tt/perf_shared_expert.py::test_shared_flat_vs_linear (2x2, chunk 2048, real weights, random x, default
  path otherwise: native, glm, ffn_norm bf16, fabric_bf16, weighted, bfp4 routed; L00 / L24 identical to 0.01 ms):
  shared partial wall 1.449 -> 0.356 ms (own x) / 0.274 ms (rows); device per op (ops mode, L00): linear 1430.9 us
  (linear 801.7 + 445.1, slices, clamps, silu, multiply, typecast), flat 252.6 us (one op; + to_layout 55.9 own x).
  MoE block device (router .. reduce-scatter) 3727 -> 2529 us. Determinism: flat twice 0 values differ.
- Plain profile, by hand (gate env, BRINGUP defaults, K-58: generated/bringup_adhoc/P.12.json), per chunk of 8 layers:
  shared_expert linear 11.41 -> flat 2.01 ms (gate < 11.4: pass; programs 288 -> 32), moe_add 1.78 -> 1.58,
  experts 17.38 / 17.33, router 1.83 / 1.83; MoE block per layer 4.05 -> 2.84 ms; device total 230.28 -> 220.61 ms
  (-4.2 %); every other section within 0.06 ms.
- Accuracy, shared_out vs float64 with the exact fp8 weights (256 rows, native numerics): linear rel 0.00297 / 0.00301
  (L00 / L24), bias +0.0009; flat 0.02048 / 0.02177, bias +0.0132 / +0.0140, worst row 0.023 / 0.026; flat vs float64
  with bfp8-rounded weights 0.0197 / 0.0208 (the device path; the bfp8 weights alone: rel 0.0052 / 0.0062, bias 0;
  bfp8 changes 10.2-11.1 % of the fp8 values, ||dW||/||W|| 0.0035-0.0040). Variants (test_flat_variants, L00 / L24):
  LoFi + bf16 x / h 0.377 ms, rel 0.011 / 0.012, bias -0.8 %; LoFi x bfp8 h bf16 0.34 ms, 0.015 / 0.016, +0.9 %;
  LoFi x bf16 h bfp8 0.38 ms, 0.013 / 0.014, -0.4 %; stochastic packer 0.36 ms, bias +0.2-0.3 % but worst row
  0.05-0.075; HiFi2 + bf16 = LoFi + bf16 bit for bit (0.45 ms); HiFi4 + bf16 0.68 ms, rel 0.0060 / 0.0069, bias +0.1 %
  (DS41-77). Shipped flat = EXPERTS_PRECISION's glm choice (as the routed experts, which carry the same +1.3 % bfp8
  bias, DS41-74); the owner may prefer bf16 tiles (+15 us) or HiFi4.
- Final check `python -m models.demos.common.bringup.orchestrator measure --task P.12` (quick, change on): rc 0, worst
  layer L03 0.99519 (P.11 0.99530), pcc_chunk_out 0.99943, host_transfers_per_layer 0; chunk_wall_ms 232.1 (spread
  7.5; P.11 232.1) on the first run, 255.6 (spread 51.3, values 228-280: a noisy box) on the rerun after the last
  description edits. The device saves ~9.7 ms a chunk; the clean wall did not move by it (not isolated; the wall
  seems host-bound near 230 ms). Odd, not chased: in the perf test the linear side's MoE block wall is 8.0 ms vs 3.7
  ms device (flat 2.75 vs 2.53; routed alone 2.40 either way).
- Frozen tests not run (quick A/B; they run SHARED_EXPERT_IMPL=linear under MAX_PRECISION). Sanity:
  test_c_swa_shared_expert PASS (err 0.01754 <= 0.06646) after the hooks change (build_shared_expert gets the chunk).
- stock ttnn.to_layout instead of ttnn.bringup.combine_rf's untilize (layout_ops): only on the fallback own-x path (no
  gathered rows, or NUMERICS=official); the shipped path converts nothing. stock ttnn.add for the parts' sum instead
  of combine_rf sum_parts: only when SE_PARTS > 1 (it is 1).
- Re-run: on `DS41_SHARED_EXPERT_IMPL=flat`, off `DS41_SHARED_EXPERT_IMPL=linear`; A/B: `PYTHONPATH=$PWD
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tt/perf_shared_expert.py -k
  "not ops"` (ops: with the profiler env, -k ops).

## P.13 perf (attempt 1): realistic defaults (fp8 weights bfp8, HiFi2, fp32 accumulation where needed, stock window attention)
- Settings (tt/settings.py), defaults = change on: DENSE_WEIGHTS bfp8 (new), MATMUL_FIDELITY HiFi2, FP32_ACC needed (new;
  replaces the boolean FP32_DEST_ACC), WINDOW_SDPA source. MAX_PRECISION: bf16 / HiFi4 / all / high_precision, so the
  component / swap tests run the old path (test_c_swa_experts re-run: PASS).
- DENSE_WEIGHTS (tt/common.py:dense_dtype): the checkpoint's fp8 matrices (attn wq_a | wkv, wq_b, wo_a, wo_b, indexer
  wq_b, engram wkv, the shared expert's linear path) uploaded as bfloat8_b; the per-matrix change is logged at load
  (DENSE_BFP8, ttnn's own host packing round-trip): 10.0-10.3 % of values, ||dW||/||W|| 0.0034-0.0035 for attention;
  indexer wq_b up to 13.8 % / 0.0046 (L14); engram wkv 12.9 % / 0.0045 (L1), 21.9 % / 0.0051 (L14). Layout kept (W^T,
  groups along N): a host check of [N, K] vs [K, N] gives the same rounding except the engram wkv (0.0035 vs 0.0045).
  The compressor's wkv / wgate / wk and the indexer's weights_proj are BF16 in the checkpoint: kept bf16 (rule 7).
  The shared expert's default (flat) path was already bfp8 (P.12).
- FP32_ACC (tt/common.py:fp32_acc, compute_config(needed=), attn_compute_config(needed=)): measured every dense projection
  at its per-chip shape (tt/perf_precision.py::test_fp32_acc_matmuls, layers 0 / 1 / 2 / 20, real bfp8 weights, HiFi2,
  random inputs vs float64 with the same weights). A 16-bit DEST: q_a rel 0.0044 -> 0.034 (bias +2.6 %), wq_b 0.0041 ->
  0.018, wo_a 0.0041 -> 0.056 (+4.5 %), wo_b 0.0044 -> 0.026, indexer wq_b 0.0041 -> 0.018, weights_proj / compressor
  0.0066 -> 0.045, engram wkv 0.0041 -> 0.084 (+6.8 %), router 0.0041 -> 0.084; time 45-55 % less per matmul (wo_b
  0.60 -> 0.28 ms, engram 2.48 -> 1.39 ms). Probe (1 chip, bf16 W): the bias comes from the spilled K-block partials
  (DS41-78); packer_l1_acc=True makes it unbiased but still 2-3.5x fp32's error (K 1024 0.0086, K 4096 0.0144 vs
  0.0041). Where fp32 stays (all of it): the long-K projections above; the stock sparse_sdpa (softmax statistics: rel
  0.019-0.024 with vs 0.026-0.042 without, worst row 0.064 vs 0.278 at x4 spread); the indexer ring score (K-13); mHC
  hc_fn + Sinkhorn, norms, engram gate sums (fp32 SFPU data, DS41-27 / DS41-40); routed + shared experts (as today,
  GLM-5.3). So FP32_ACC=needed == all on today's code (no program differs); no site was turned off.
- WINDOW_SDPA=source: ttnn.transformer.sparse_sdpa with the attention config (HiFi2, fp32 DEST) for the window and the
  compressed layers. Op alone (perf_precision.py::test_window_sdpa): 0.355 -> 0.334 ms, rel vs float64 0.0073 -> 0.024
  (bias -0.45 % -> -1.1 %); x4 spread 0.0086 -> 0.019.
- stock ttnn.transformer.sparse_sdpa instead of ttnn.bringup.sparse_sdpa: owner rule F123 / the P.13 brief (the stock op
  at its normal settings; the fork's high_precision stays the max-precision value, WINDOW_SDPA=high_precision).
- Plain profile, 2x2, chunk 53248-55296, by hand (gate env + DS41_ overrides; K-58: generated/bringup_adhoc/P.13*.json),
  device total per chunk: off 220.79; DENSE_WEIGHTS=bfp8 alone 220.90 (no section moves: compute-bound at 1024 rows,
  DS41-79); MATMUL_FIDELITY=HiFi2 alone 219.52 (engram_kv 6.02 -> 4.87); WINDOW_SDPA=source alone 220.30 (attention
  114.93 -> 114.39); FP32_ACC=needed alone: same programs; all on 219.17 (-1.62 ms, -0.7 %: attention 114.40, engram_kv
  4.93, q_a 4.75, indexer 34.99, experts 17.34).
- Quick A/B (last chunk vs golden + clean wall): off L00 0.99936 L01 0.99899 L02 0.99782 L03 0.99519 L14 0.99614 L20
  0.99928 L21 0.99688 L24 0.99943, wall 231.7 ms; on (final `measure --task P.13`, rc 0) L00 0.99932 L01 0.99892 L02
  0.99768 L03 0.99486 L14 0.99612 L20 0.99927 L21 0.99682 L24 0.99941, pcc_chunk_out 0.99941, wall 243.0 (spread 27.6;
  first run 236.7), host_transfers_per_layer 0; WINDOW_SDPA=source alone: L03 0.99487 (all of the drop).
- Re-run: on `DS41_DENSE_WEIGHTS=bfp8 DS41_MATMUL_FIDELITY=HiFi2 DS41_FP32_ACC=needed DS41_WINDOW_SDPA=source`, off
  `bf16 HiFi4 all high_precision`; measurements `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_precision.py -s`; final `python -m models.demos.common.bringup.orchestrator
  measure --task P.13`.

## P.14 perf (attempt 1): fused mHC (MHC_FUSED on / off)
- Setting MHC_FUSED (tt/settings.py), default on (the change; the brief's details said "off (default)", the role's rule
  says the change is the default: the orchestrator pins both sides). Not a precision switch (no MAX_PRECISION entry):
  the same fp32 SFPU arithmetic, sums in another order.
- Fork changes, every default unchanged (programs, hashes, returns): mhc_pre_ttnn CHANGELOG 8 (`mhc_pre_xing(...,
  collapse_pre=)`: coefficients + a collapse with a given pre in one program; DM define MHC_XING_GIVEN_PRE) and 9
  (`mhc_pre_xing_pack(None, h, engram_keys=, engram_weight=)`: the 12 engram partials in one pass, own program);
  mhc_post_ttnn CHANGELOG 7 (`mhc_post(..., stats_weight=W)` -> (X', X' @ W with sum X'^2 in column 24); own program).
  New fork tests: mhc_pre tests/unit/test_xing_collapse_pre.py (4), test_engram_pack.py (7); mhc_post
  tests/unit/test_mhc_post_stats.py (8). Regression with the options off (2x2): mhc_pre unit + golden 332 passed / 1
  skipped (recorded 321 / 1 + 11 new), model cases 103 (= recorded); mhc_post unit + golden + cases 382 (= 164 + 208 +
  2 recorded + 8 new). No source-test selection in either fork (AI-generated ops).
- Model (MHC_FUSED=on): attn_hc / ffn_hc = (the partial row: from the previous residual's mhc_post pass when it
  emitted one, else matmul + pack) -> tp_sum -> ONE mhc_pre_xing(deepseek=True, collapse_pre=the pending / attn_hc
  mixes) (DS41-3: a sublayer collapses with the PREVIOUS sublayer's pre); attn_collapse / ffn_collapse hand the y on
  (0 programs). The block boundary carries the whole ffn_hc mixes [S/SP, 24] as the pending pre (layer 0: one-hot
  [S/SP, 24] at load; from_host pads [S, 4] to 24, to_host trims) and DevH.stats = (layer, row). Residuals:
  TtHcResidual.fused -> mhc_post(stats_weight=next hc's fn_t, MATMUL_FIDELITY via a ComputeConfigDescriptor):
  attn_residual -> this layer's ffn_hc; ffn_residual -> the next loaded consecutive non-engram layer's attn_hc
  (TtDS41Model.next_hc); the engram add (identity-comb mhc_post) -> its own layer's attn_hc. Here (layers 0-3, 14,
  20, 21, 24) attn_hc of L0, L20, L24 still runs matmul + pack (embedding / golden-gap run starts); in a full stack
  only L0. Final collapse: mhc_pre_xing(coefficients_given=True) on the 24-wide pre. Engram gates: one engram-mode pack
  -> tp_sum(exact=True) (DS41-82) -> the unchanged gate math; the add: mhc_post(value, h, gate, I).
- stock ttnn.matmul instead of ttnn.bringup.mhc_post(stats_weight=): TtHcMixes.partial_row, for an attn_hc whose
  input came from no residual pass (the embedding, a run start after a gap); the hc_fn projection at MATMUL_FIDELITY.
- Op timings (2x2 p150, [1024, 4 x 2560] fp32 a chip, wall, 30 synced calls): mhc_post 0.270 / with stats 0.327 /
  unfused post + matmul + pack 0.638 ms; mhc_pre_xing coef 0.028, + collapse_pre 0.164 vs the multiply / addcmul
  collapse 0.765 ms; engram partials 0.424 vs the op chain 2.51 ms. The stats and engram programs are DRAM-bound
  (variants without matmul / SFPU / copies <= 0.01-0.04 ms faster; without input reads 0.271): DS41-80.
- Plain profile, 2x2, chunk 53248-55296 (by hand, generated by test_profile.py; profiles P.14_20261020T154842 off,
  P.14_20261020T152421 on), device ms a chunk, off -> on: attn_hc 3.21 -> 2.54, attn_collapse 5.00 -> 0, attn_residual
  2.44 -> 2.94, ffn_hc 3.21 -> 1.44, ffn_collapse 5.00 -> 0, ffn_residual 3.11 -> 3.27 (mHC 21.96 -> 10.19), engram_add
  6.99 -> 1.78; device total 219.54 -> 202.30 (-17.24 ms, -7.9 %); wall (profiler on) 390.8 -> 317.6.
- Quick (BRINGUP_QUICK, on, by hand): worst layer L03 0.99488 (P.13: 0.99486), wall median 216-221 ms (P.13 243.0).
  Final `measure --task P.14` (rc 0): min layer L03 0.99488, pcc_chunk_out 0.99941, chunk_wall 213.7 ms (spread 22.1),
  host_transfers_per_layer 0.
- Follow-ups not done: W multicast to the cores of a column block in the stats program (~13 % of its reads); post /
  comb read from the packed mixes in the stats program (2 slices a residual); the engram value slice as an F column
  offset; the engram pass at ~200 GB/s (DM-bound like the stats one).
- Re-run: fork tests `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/mhc_post_ttnn/tests/unit/test_mhc_post_stats.py`
  and `.../mhc_pre_ttnn/tests/unit/test_{xing_collapse_pre,engram_pack}.py`; profile `DS41_MHC_FUSED=on|off` + the gate's
  profile command; final `python -m models.demos.common.bringup.orchestrator measure --task P.14`.

## P.15 perf (attempt 1): attention / RoPE glue folded into ops (ATTN_GLUE on / off)
- Setting ATTN_GLUE (tt/settings.py), default on (the change; the orchestrator pins both sides). Not a precision switch
  (no MAX_PRECISION entry): every fold is a data move or a cast moved into the op before it.
- (1) Pair interleave / de-interleave: tt/rope.py:PairPerm, a 0/1 [64, 64] bf16 matmul on the bf16 tail (data in in0),
  HiFi4 + fp32 DEST (DS41-83: HiFi2 / LoFi drop in0's last bit; DS41-16's one-product-per-dot idea, K-30: never on
  fp32 data). Sites: attention q (after the scale's bf16 multiply), attention output (bf16 sdpa output -> to_de, fp32
  out), indexer q, kv and compressor (rope_deinterleaved_bf16: the rope's add / subtract write bf16 = act_q / fp4_q's
  cast under native, DS41-84). Window layers now read the table as stored (instance INTERLEAVED=True when the q fold is
  on): no per-chunk table transform (glue report A17). Output re-interleave dropped: wo_a's input rows permuted at load
  to [nope | re | ro] (bfp8 groups run along N, so the weight rounds the same). NUMERICS=official keeps the fp32
  interleave before an fp8 / fp4 block (q has no block: its fold runs in both numerics).
- (2) q heads: ttnn.experimental.nlp_create_q_heads_split (attention: 16 heads, split 448; indexer: 32 heads, split
  64): bit-identical to reshape + transpose / permute (probe ds41_attn_glue/probe_001); RoPE then on the 64-wide tail only.
- (6) Output: the 448 nope columns stay bf16 (no bf16 -> fp32 -> bf16); after wo_a nlp_concat_heads (native: on the bf16
  act_q, K-19: 2 x 1024 columns fit; g_l = 2 on every listed box) instead of slices + concat; sum_cols(fp32=False) returns
  the bf16 reduce-scatter result. mhc_post takes bf16 F only without stats_weight (README limits), so under MHC_FUSED=on
  (default) tt/residual.py casts it to fp32 before the stats pass: the A12 cast moved, not removed (attn_residual +0.36
  ms); it is gone only under MHC_FUSED=off. Removing it needs a bf16 F CB in mhc_post's stats program (not done).
- Exactness (probes ds41_attn_glue/probe_001..010): q bf16 bit-identical after reordering the tail; rope_deinterleaved_bf16
  == typecast(rope_deinterleaved) (widths 512, 128); rope_heads(bf16) + PairPerm == typecast(rope_heads(interleave=True));
  PairPerm to_de == deinterleave_tail. Window attention L0 output path vs float64 (bf16 o input): rel 0.00736 off and
  on (DENSE_WEIGHTS / ATTN_FIDELITY / CCL max precision: 0.00136 both); on vs off 0.0019 (rounding order: permuted wo_a
  K rows, table order in sdpa). Whole window layer on random inputs: 11.00 -> 4.74 ms a call (wall, synced).
- Plain profile, 2x2, chunk 53248-55296 (profiles P.15_20261020T165333 off, P.15_20261020T164559 on), device ms a
  chunk off -> on: attention 114.36 -> 43.01, indexer 34.97 -> 25.15, kv 2.11 -> 1.24, compressor 2.38 -> 1.90,
  attn_residual 2.93 -> 3.29; device sum 202.19 -> 119.82 (-82.4 ms, -41 %). Programs (all chips) 9772 -> 8892:
  attention 3296 -> 2648, indexer 756 -> 660, kv 1024 -> 928, compressor 1120 -> 1048, attn_residual 224 -> 256.
- Quick A/B, on (`measure --task P.15`, rc 0): min layer L03 0.99489 (P.14: 0.99488), pcc_chunk_out 0.99941, chunk
  wall 201.8 ms (spread 13.6; P.14 213.7), host_transfers_per_layer 0; final run after the doc edits: chunk wall 196.0
  ms (spread 8.9), same PCCs. The wall falls far less than the device: the
  chunk is host-bound now (device ~120 ms vs 202 ms wall), so op count is the next lever.
- Frozen swap tests run by hand with ATTN_GLUE=on: test_swap_enc_source_14, swa_12, dec_reindex_13 pass (official
  numerics, device attention / indexer / compressor / kv in the block; swa_12's ffn_hc WAIVED as before).
- Re-run: probes `tests/ttnn/unit_tests/operations/ds41_attn_glue/probes/`; profile `DS41_ATTN_GLUE=on|off` + the gate's
  profile command; final `python -m models.demos.common.bringup.orchestrator measure --task P.15`.

## P.16 perf (attempt 1): indexer / shared-source glue (INDEXER_GLUE on / off; glue report items 3, 7, 8, 9, 10)
- Setting INDEXER_GLUE (tt/settings.py), default on (the change; the orchestrator pins both sides; the brief's
  "off (default)" vs "make the change the default": as P.14 / P.15). Not a precision switch: data moves, and the bf16
  partials q_a already writes under CCL_IMPL=fabric_bf16 (P.6).
- (3) Candidate block max (L20): fork option on ttnn.bringup.ring_indexer_score_dsa (indexer_score CHANGELOG "Block
  epilogue"): `block_keys=8, block_max_out=<load-time [1, 1, Sq, ceil32(T / 8)] fp32 RM buffer>, block_max_pin=True`;
  the writer takes the max of each 8 bf16 scores of its untilized strip (exact, widened to fp32) and stamps +inf at the
  query's own block (old: +1e30 in bf16; same selection). Then typecast to bf16 (exact) + topk_large_indices(2048,
  valid_length = kv_len / 8); below 2048 blocks a slice + -inf pad as before (DS41-52: the full-range slice is a
  wrapper over the persistent buffer, never freed). Replaces 8 strided slices, 7 maximum, to_layout, the 9 pin ops.
- (8) Re-index bias (L24): the same scatter (ROW_MAJOR, width ceil32(T / 8)) + to_layout TILE -> `block_bias` of the
  score op: the writer writes -inf over every key of a block whose bias is nonzero; no repeat_interleave, slices, add.
- Fork tests: tests/unit/test_block_epilogue.py (2x2, ratio 1, natural blocks, block-cyclic keys, T 6144, 6 starts,
  block_keys 8 / 4: bias, max, pin bit-exact vs the plain call; sentinel past kv_len; 7 refusals); negative controls by
  hand (corrupted max, a bias that skips a key) fail. Regression with the option off: unit + model cases 22 passed / 1
  skipped (= 17 / 1 + 2 recorded + 3 new); fork_source 134 tests, 0 regressions (the recorded
  test_indexer_score_rejects_fp32_dest_acc divergence only).
- Score op alone, L20 shapes (Sq 512 x 32 x 128, T 55296, HiFi2 + fp32 DEST, 30 synced calls; probe
  ds41_p16_epilogue, deleted): plain 1.68 ms; + max / pin 1.96 (first version 3.07; in the model's per-op profile
  6.68: DS41-85); + bias 1.69 (first, ROW_MAJOR bias rows: 2.16; DS41-86).
- (7) tt/sparse_attention.ChunkMemo on DS41DeviceState (memo): ("comp", kv source) the gathered compressed history
  and nb, ("index", idx source) the [window | top-k rows] index list, ("keys", kv source) the indexer's tiled index
  keys; each keyed by (start, cache id, TtServedRows.version: bumped by every _put); a new top-k of a layer drops its
  "index" slot (DS41DeviceState.put); TtDS41Model.memo_last drops each slot after its last loaded reader of the chunk
  (here: comp 2 after L3, 20 after L24; index 2 after L3, 20 after L21; keys 20 after L24). L3 / L21 / L24 no longer
  slice + tile + all_gather x 2 + untilize the history; L3 / L21 skip _topk_rows (~25 ops); L24 skips the key retile.
- (9) tt/rope.RowsMemo: RopeTable.rows(start, S[, tp_split]) and CompressRope.rows once per (start, length); callers
  release through `rope.free` (memoized rows stay; the latest start only). The indexer's TP split of the rows is a
  memo entry too. Router weights: with EXPERTS_COMBINE=weighted the router returns (None, idx, wts) (no zeros slice /
  2 to_layout / scatter) and experts._weighted takes wts (no to_layout + gather); bit-identical (dense was their
  scatter).
- (10) Indexer gates: linear writes ccl.partial_dtype (bf16 under fabric_bf16) -> tt/ccl.sum_rows = ONE
  fabric_reduce_scatter by row blocks: chip c's block is rows [c Sq, (c+1) Sq), what tp_sum + mesh_partition kept
  (no all_gather, mesh_partition, 3 typecasts). Compressor: its [wkv | wgate] partial written in partial_dtype (C1).
  Both only with HC_TP_SUM=gather_add; stock / untiled fall back to the old path.
- stock ttnn.slice instead of ttnn.bringup.ring_indexer_score_dsa: the first nbv < 2048 columns of the op's own
  block-max output before the -inf pad (a column slice, not a block pool).
- Plain profile, 2x2, chunk 53248-55296 (profiles P.15_20261020T164559 = off, P.16_20261020T183616 = on), device ms a
  chunk off -> on: indexer 25.15 -> 12.94, attention 43.01 -> 40.56, experts 17.18 -> 16.82, router 1.82 -> 1.47,
  compressor 1.90 -> 1.79, kv 1.24 -> 1.18; device sum 119.82 -> 104.28 (-15.5 ms, -13 %). Programs (4 chips) 8892
  -> 7800: indexer 660 -> 376, attention 2648 -> 2244, kv 928 -> 800, compressor 1048 -> 964, router 480 -> 384,
  experts 480 -> 384. Wall with the profiler on 367.1 -> 329.0 ms.
- Quick runs by hand: on: chunk wall 171.9 / 175.7 ms, L00 0.99933 L01 0.99892 L02 0.99769 L03 0.99490 L14 0.99612
  L20 0.99928 L21 0.99682 L24 0.99941; off: 197.0 ms, L03 0.99489 (P.15 196.0); host_transfers_per_layer 0.
  Ladder last (on, by hand): pcc_state_min 0.99712, overlap candidates L20 0.99757, top-k L24 0.99291.
- Frozen tests run by hand with INDEXER_GLUE=on (sanity, max precision): test_c_dec_candidate_indexer (failed first:
  DS41-52, the block-max buffer freed by a full-range slice; fixed), test_c_dec_reindex_indexer, enc_source_indexer,
  enc_source_compressor, dec_reuse_attention, swa_attention, dec_reindex_experts, test_swap_dec_reindex_13_experts:
  pass.
- Final `measure --task P.16` (on, rc 0): min layer L03 0.99490, pcc_chunk_out 0.99941, chunk wall 174.9 ms (spread
  6.9; P.15 196.0), host_transfers_per_layer 0.
- Not done: the 56 MiB concat [window | comp] per sparse layer (needs a two-KV sparse_sdpa option); a persistent
  natural-order history mirror (report item 7b).
- Re-run: fork `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/indexer_score/tests/unit/test_block_epilogue.py`;
  profile `DS41_INDEXER_GLUE=on|off` + the gate's profile command; final `python -m
  models.demos.common.bringup.orchestrator measure --task P.16`.

## FIX.5 (overseer fix, attempt 1): ChunkMemo cleared at chunk start: NOT DONE, blocked by scope
- The brief allows no model code: tasks.yaml FIX.5 has `paths: []`, so tt/ and hooks.py are read-only; the fix is
  in tt/model.py. No code changed, no device run.
- Patch for the retry (needs `paths: [models/demos/deepseek_v41_flash_d_p/tt]`): first lines of
  `TtDS41Model.begin_chunk` (tt/model.py:682), before the engram early return:
  `st = self.state` / `if st is not None and st.memo is not None: st.memo.clear()`. harness.begin_chunk calls it
  before every chunk (profile and timeline runs alike).
- Diagnosis note: memo keys already carry (start, cache id, cache version) (sparse_attention.py:251,
  indexer.py:265), so another chunk misses; the stale hit needs the same start and an unchanged source version
  (X.3 profile run -> timeline run of the same chunk). Clearing in begin_chunk covers both cases.
- Re-run: the X.3 gate command in the FIX.5 brief.

## FIX.6 (overseer fix, attempt 1): reference numerics switch official | native
- `DS41Reference(..., numerics="official" | "native")` (reference/ds41_ref.py, docstring "Numerics"), passed by
  `hooks.reference(spec, layers=None, dtype=None, numerics="official")`. One switch: `_q8` (fp8 act q-dq per 32) and
  `_q4` (fp4 q-dq) at every call site, `_fp8_linear` = mm(_q8(x), w). native: all are x.float(): fp8 inputs of q_a,
  kv, wq_b, wo_b, indexer wq_b, engram wkv, shared / routed experts x and h; window-KV fp8; attention output fp8
  before wo_a; compressed-KV, index-key and index-query fp4. Weights unchanged (exact dequant).
- Nothing else to remove: the reference has no TF32 rounding and casts no intermediate to bf16 (it computes in fp32).
  `kernels.attend`'s bf16 P only runs on bf16 inputs, which only the oracle passes. kernels.py is untouched: the
  oracle (hf_oracle.py) keeps running the faithful port, and official stays bit for bit (DS41-1: same tiled GEMMs).
- `with_numerics(n)`: shallow copy that shares weights, tables and hasher (the test loads the weights once).
- Gate (CPU, 35 min): R.2 every layer maxabs 0, logits pcc 1.0, top1 1.0; R.3 hidden / state pcc 1.000000000,
  graph replay 0 for every block type.
- reference/test_numerics.py (CPU): (1) at 512 tokens, the official forward calls the quantizers
  (act_quant_dequant 3472, fp4_quant_dequant 10 on the real checkpoint); the native forward runs with every
  kernels.py quantizer patched to raise. (2) At 4096 tokens, one-shot: rel of native vs official.
  rel = ||native - official|| / ||official||:
    | checkpoint | L0 | L1 | L2 | L3 | L14 | L20 | L21 | L24 | final_norm |
    |---|---|---|---|---|---|---|---|---|---|
    | real (BRINGUP_SCALED=0) | .0281 | .0270 | .0315 | .0360 | .0910 | .1411 | .0308 | .0201 | .2886 |
    | scaled quietbox_2x2 | .0285 | .0355 | .0525 | .0741 | .1205 | .1848 | .1724 | .1984 | .2145 |
  L0 steps (identical inputs): q_resid .0115 (one fp8 q-dq; DS41-72 had .0116), kv .0289, attn_out .043.
  L20: router row-rel median .51, top-k overlap .85 / .79 (real / scaled). Logits top-1 agreement over the last 512
  positions: .45 / .51. These are the partial model's layers (0-3, 14, 20, 21, 24, chained).
  Bounds: L0 q_resid in (.008, .016), kv in (.020, .040); layers < 0.3; final_norm < 0.4.
- Not ~1e-2 as the brief expected: one fp8 rounding alone is 1.15 %, and routing and top-k flips compound it (DS41-90).
- Heads-up (DS41-89): now that the hook takes `numerics`, component / swap tests get the native reference
  (defaults.yaml tests.step_numerics: native), but the device's MAX_PRECISION is still NUMERICS=official. Those tests
  were not rerun here; F128 should switch both together. tt/settings.py NUMERICS text ("the reference hook has no
  numerics switch") is now stale; settings.py was outside this step's paths.
- Gotcha: run the CPU test with `--no-precompile`. The wrapper's up-front collect pass gave other numbers (scaled
  L24 .334, 1562 vs 1328 quantizer calls) than the real pass, which matched a standalone script exactly.
- Re-run: the gate in the FIX.6 brief; `BRINGUP_SCALED=0|1 scripts/run_safe_pytest.sh --no-precompile --run-all
  models/demos/deepseek_v41_flash_d_p/reference/test_numerics.py` (scaled ~8 min, real ~12 min).

## FIX.6 (overseer fix, attempt 2): no code change; attempt 1's switch stands
- Attempt 1 passed the gate (R.2 maxabs 0 on all 25 layers, logits pcc 1.0; R.3 pcc 1.000000000, replay 0). Only the
  commit failed: pre-commit `check-large-files` refused bringup/state.json (502 KB then, 516 KB now, limit 500 KB).
  state.json is the orchestrator's file and outside this step's paths, so this attempt cannot shrink it. The
  orchestrator must trim it, or commit it apart from the hook, before FIX.6 can land.
- Reviewed attempt 1's diff again: every K.act_quant_dequant / K.fp4_quant_dequant call in ds41_ref.py goes through
  `_q8` / `_q4`. Only the module-level `fp8_linear` (ds41_ref.py:308) still calls K directly, and the per-layer
  steps no longer use it (they call `self._fp8_linear`). reference/test_numerics.py makes every K quantizer raise
  under native.
- Reran the gate unchanged (result in the summary below / results/FIX.6.json).

## FIX.7 (overseer fix, attempt 1): official numerics at the GPU's dtype (bf16 activations)
- `DS41Reference(numerics="official", dtype=torch.bfloat16)` now computes like DeepSeek's GPU code (generate.py's bf16
  default dtype): arithmetic stays fp32, and a value is rounded to bf16 exactly where model.py holds it in bf16
  (`_r`; ds41_ref.py docstring "Numerics" lists every point): streams at every unpack (so an fp32 golden input reads as
  the GPU's bf16 stream), engram rows, every fp8 / fp4 / bf16 GEMM output, every RMSNorm output, the hc_pre collapses,
  RoPE'd q / kv / index q / latent / index key, sparse_attn's bf16 P (kernels.attend on bf16 q) and output, the
  inverse-RoPE'd output, wo_a, hc_post outputs, the experts' h before w2, moe_add. fp32 stays where model.py computes
  in fp32: mHC mixes + Sinkhorn, ratio-2 compressor wkv / wgate / pooling (and compress_partial), router, the routed
  experts' fp32 sum, attention scores / softmax statistics, pending pre, logits. DS41-38's compressor bf16 points
  (bf16 latent before and after the norm, bf16 wk / k_norm, RoPE into bf16) are exactly these.
- `_mm16`: the GEMMs model.py runs on bf16 x and a bf16 weight (wo_a, ratio-1 compressor wkv, indexer wk /
  weights_proj) and the index-score einsum + relu x weights + head sum run on bf16 tensors (K.linear / K.grouped_out
  / K.index_score_tile tiles, DS41-1). First version used fp32 GEMMs rounded once: every step bit-exact except
  attention (99.9 % equal, rel 9e-5..1.6e-4), block outputs vs the oracle rel 3.6e-4 / 3.3e-3 / 7.7e-3 / 1.1e-2 at
  L0-L3. Cause: CPU bf16 einsum and fp32 + one rounding differ on 0.03 % of outputs (both rel 1.659e-3 vs fp64,
  probe of [256, 8, 4096] x [8, 1024, 4096]); fp8 q-dq flips grow it (DS41-91). With `_mm16` everything is bit-exact.
- native and official fp32 are untouched: `_r` is the identity there, `_mm16` is the old `mm(x.float(), w)` /
  `K.grouped_out(o, w)`. Checked against a copy of the pre-FIX.7 ds41_ref.py, real checkpoint, 512 tokens, spec layers:
  native 144 / 144 boundaries and official fp32 144 / 144 equal bit for bit. native refuses dtype bf16;
  `with_numerics("native")` is always fp32, `with_numerics("official", dtype)` picks the official dtype.
- Gate (CPU): R.2 official L00-L24 maxabs 0, logits pcc 1.0, top1 1.0; R.3 official and native chunked vs one-shot
  hidden pcc 1.000000000 / 0.999998146, state min 1.000000000, graph replay 0 for all 8 block types.
- reference/test_official_bf16.py (CPU, 4 tests, both checkpoints pass): (1) every step of layers 0-3 on the oracle's
  own inputs (official_taps.capture: build_oracle at bf16 = hooks.hf_model(spec, None)'s configuration, first 4
  blocks, one-shot 512 tokens) equals model.py at bf16: rel 0, equal 1.000000, indexer top-k overlap 1.0; control: the
  fp32 official misses L0 attn_norm by 1.68e-3 (0.86 bf16 roundings, bound 0.5). (2) block outputs end to end: rel 0
  at L0-L3 (bound 3 roundings). (3) native from the bf16 ref == native built at fp32 (bytes), native at bf16 refused.
  (4) report of official vs native at 4096 (below).
- The new margins, scaled quietbox_2x2, rel = ||official - native|| / ||native||:
  | 4096 one-shot | L0 | L1 | L2 | L3 | L14 | L20 | L21 | L24 | final_norm |
  |---|---|---|---|---|---|---|---|---|---|
  | official bf16 | .0286 | .0361 | .0528 | .0752 | .1213 | .1838 | .1718 | .1983 | .2165 |
  | official fp32 | .0285 | .0354 | .0523 | .0738 | .1204 | .1836 | .1703 | .1962 | .2145 |
  reference.drift.compute(s4096) (the framework's margin, chunked 2048 as the ladder, gapped runs fed from the native
  golden; run with the file write skipped, nothing written into the golden), layer_L*: bf16 / fp32
  L0 .0281 / .0273, L1 .0367 / .0355, L2 .0535 / .0526, L3 .0779 / .0762, L14 .0602 / .0622, L20 .0260 / .0254,
  L21 .0632 / .0629, L24 .0225 / .0218 (rel_res one bf16 rounding .00166); states: window_kv .029-.067,
  compress_kv .095-.104, index_k .116-.120, topk_idx overlap .886-.952, candidates_L20 1.0.
  bf16 activations move the margin by -3 % .. +4 % of itself; the fp8 / fp4 roundings dominate it (DS41-90).
- Not done here: drift files / official goldens are not regenerated (outside this step's paths); the next G task's
  `reference.drift --rung <rung>` and `generate_golden --numerics official` now run at the spec's shipped bf16.
- Gotcha: run_safe_pytest.sh only in the foreground (the real-checkpoint 4096 report runs > 10 min, so only the
  scaled one ran here; the 512-token tests run in 7.5 min on the real checkpoint).
- Re-run: the gate in the FIX.7 brief; `BRINGUP_SCALED=0|1 scripts/run_safe_pytest.sh --no-precompile --run-all
  models/demos/deepseek_v41_flash_d_p/reference/test_official_bf16.py -s` (`-k "not margins"` for the 512-token part).

## FIX.8 (overseer, 2026-10-21): onto F131-F142, shipped settings everywhere, parity code deleted

What changed (the tree did not import before: F137 removed `Settings(max_precision=...)`):
- (1) `tt/settings.py`: MAX_PRECISION and `max_precision=` gone; `Settings("DS41_", TABLE)`. Deleted settings:
  NUMERICS, COMPRESSOR_X (round_tf32), ENGRAM_KV_QDQ (qdq), ATTN_IN_MM, WINDOW_SDPA (source was the only shipped
  value; high_precision / exact gone), WINDOW_SDPA_OUT / _SCORES / _PROBS, WINDOW_PIECES (33 -> 24 settings). The
  "max precision" wording in the remaining descriptions rewritten. settings_lint: 33 files, 0 violations; no
  emulation default.
- (2) Deleted files: `tt/quant.py` (277 lines: fp8_qdq, fp4_qdq, round_tf32, native / act_q / fp4_q / tf32_q),
  `tt/exact_mm.py` (127), `tt/window_exact.py` (266), `tt/perf_attention.py` (81, timed only the deleted sdpa
  variants). `act_q` / `fp4_q` (their native branch) became `tt/common.py:cast` (x itself when already the dtype,
  else one typecast). Official branches removed: attention.py (fp32_out / fp32_scores / bf16_probs / fp32_q, the
  ttnn.bringup.sparse_sdpa call, the exact window path, the glue_out official re-interleave, the official wo_a
  concat order), sparse_attention.py (exact assert), q_a.py (ExactMatmul), kv.py / compressor.py / indexer.py
  (`native()` conditions, round_x, fp4 blocks), engram_kv.py (qdq), experts.py (`qdq`, the fork's
  `fp8_act_quant=` argument: default False now), shared_expert.py (the official flat path: `routed=` plan, unit
  routing weights, fp8_act_quant), norm.py (quant -> cast), weighted_combine.py (`acc_unpack_fp32=True` dropped: a
  no-op with COMBINE_ACC=bf16, bf16 values pass srcA exactly), perf_experts / perf_shared_expert (float64 references
  always native), perf_precision (fork variant dropped). tt/: +279 / -1332 lines. The fork options themselves stay
  (J2 removes routing_weights / fp8_act_quant); their READMEs' "used by" say the model no longer calls them.
- (3) Contract helpers from reports/fix8_dev_model_diffs.patch (its settings hunk ignored, (1) supersedes it):
  `server_rules.state_threshold()` returns a per-(config, layer) limit, the stricter of the server's 0.93 and
  `e2e_limits.pcc_limit(drift-s4096, state_<kind>_L<i>)`; `caches.pcc_failures` accepts it. On s4096: window L00
  0.9983, L03 0.9911, L24 0.9972; compress L02 0.9783; index L02 0.9709, L20 0.9728.
- Two K.1 failures the gate exposed, neither from this change (K.1 last passed 2026-10-17, before P.8 / F128):
  DS41-93 (integer engram_tail had no contract drift: the hook now checks it exactly, asserts, metric
  contract_state_exact_engram_tail = 1) and DS41-94 (runner fabric payload 4352 B vs the weighted combine's 10880 B:
  the runner hung in FlatWithProgramDeviceOperation; DS41Config.FABRIC_PAYLOAD_SIZE now follows EXPERTS_COMBINE).
- Kept: FIX.5's tt/model.py held wrappers + begin_chunk memo clear, tt/sparse_attention.py doc.

Measured (2x2, defaults):
- Plain profile, chunk 53248-55296 (test_profile, BRINGUP_TASK=FIX.8): device 104.31 ms a chunk vs P.16's 104.28,
  7800 programs both, every section within 0.02 ms (attention 40.57, experts 16.84, indexer 12.93); wall 331 ms.
- M.1 (s4096, native goldens, F135 limits): err_layer L00 .0273/.0561, L01 .0355/.0734, L02 .0547/.1071, L03
  .0847/.1558, L14 .0572/.1204, L20 .0307/.0519, L21 .0514/.1265, L24 .0309/.0449; err_state max window_kv_L03
  .0588/.1334; host_transfers_per_layer 0.
- L.last (53k prefix from the golden): err_layer L00 .0265/.0570, L01 .0346/.0720, L02 .0523/.1077, L03 .0818/.1587,
  L14 .0618/.1380, L20 .0306/.0507, L21 .0521/.1312, L24 .0281/.0418 (worst share of the margin: L24 67 %);
  err_state worst window_kv_L21 .0470/.1118.
- K.1: contract_checks_failed 0, acks_early 0; pcc_producer_kv window_kv .99840 >= .99164, compress_kv .99923 >=
  .97827, index_k .99977 >= .97088; pcc_contract_state_window_kv .99843 >= .99164; kv_layout, adapter_acks,
  runner_contract (185 s) pass; runner_smoke skipped (DS41-8).
- "Not numerically changed": no earlier err_* baseline exists (the tree did not import since F137); every deleted
  branch was off under the old defaults (NUMERICS=native, WINDOW_SDPA=source, ATTN_IN_MM=fpu), the dropped
  acc_unpack_fp32 / fp8_act_quant=False are no-ops on the shipped path, and the profile's program count and section
  times are the same.

Gotchas:
- The drift gives compress_partial an empty margin (DS41-92: ladder limit 1.8e256, contract PCC limit -3.0): the
  all-zero state passes anything; a framework fix (exact check of an all-zero state).
- Frozen per-step tests that still assume the official arithmetic were not touched (test role / J0): e.g.
  test_c_*_kv / test_c_enc_source_compressor check the fp8 / fp4 grid of the device output and will fail on the
  bf16 cast; the batched verify (F142) will list them.
- L.last's gate has no --no-precompile, so its body runs twice (K-52): 11.6 min, longer than one 10-minute call.

Re-run: the FIX.8 gate (brief), one command per call; profile: `BRINGUP_TASK=FIX.8 TT_METAL_DEVICE_PROFILER=1
TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000
scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_profile.py`.

## FIX.9 (overseer, 2026-10-22): one expert pipeline (precision_debt_report.md J2)

What changed (the flat routed expert with the weighted combine, and the flat shared expert, are the only path):
- `tt/experts.py` 1019 -> 383 lines: deleted EXPERTS_COMBINE=separate (dispatch / combine / K-slot fp32 adds,
  `_modules`, `_dispatch`, `_combine`), the routing-weight pieces (`_pieces`, `_weight_col`, `_routing_rows`, the
  selectors `sel` / `sel_all`, DS41-23), EXPERTS_KERNEL loop / unified (`_loop`, `_ffn`, `_unified`, `_flat`), the
  fp4_decode weights (`pack_planes`, `_load_fp4`, `_decode`) and the unused device tables (dispatch table, gidx).
  `partial` = the old `_weighted`; `_load_block` keeps the bfp8 exactness assert / bfp4 stats on the [N, K]
  matrices and feeds `_load_flat`. `build_experts` lost `combine=`.
- `tt/shared_expert.py`: deleted SHARED_EXPERT_IMPL=linear (the ttnn.linear chain, its bf16 / DENSE_WEIGHTS
  weights); `chunk` is required, `impl=` gone.
- `tt/settings.py`: deleted EXPERTS_KERNEL, EXPERTS_COMBINE, EXPERTS_DECODE, SHARED_EXPERT_IMPL; EXPERTS_IMPL is
  {bfp8, bfp4}; descriptions of EXPERTS_PRECISION / COMBINE_ACC / COMBINE_PLANNER / MATMUL_FIDELITY / DENSE_WEIGHTS
  no longer name the deleted paths (24 -> 20 settings).
- `bringup/hooks.py:device_params` and `tt/runners/adapter.py:_FabricPayload` always use the weighted combine's
  10880 B payload (DS41-94); `tt/model.py` keep_rows / router_wts no longer test `experts.weighted`;
  `tt/weighted_combine.py` docstring. tt/ + hooks: +238 / -1218 lines.
- `tt/perf_experts.py`: deleted test_time_experts (bfp8 vs fp4_decode), test_unified_vs_loop, test_kernels;
  test_weighted_vs_separate -> test_weighted (weighted only; also fixed its undefined `wround`).
  `tt/perf_shared_expert.py`: test_shared_flat_vs_linear -> test_shared_flat.
- Forks (no other model calls the options: grep of every model's code, the forks' tests/cases.py and tests):
  - unified_routed_expert_ffn: removed `routing_weights` / `fp8_act_quant` (attributes, input, CBs c_20 / c_21,
    reader staging, `calculate_routed_h`, UnpackToDestFp32); `weights_transposed` stays (no caller now). Program
    factory and kernels = pre-P.2 + WEIGHTS_NK. `tests/unit/test_ds41_h_options.py` deleted, its transposed check ->
    `tests/unit/test_weights_transposed.py`; `tests/reference.py` lost the fp8 helpers.
  - flat_routed_expert_ttnn: removed `routing_weights` / `fp8_act_quant` and the plan's `routed` (plan branch, 36 KB
    arena region, RH_* CBs / semaphores / args, `AddrSrc::RoutingWeights`, SE_RH_* kernel blocks, `se_rh.hpp`);
    `weights_nk` / `hidden` stay. `tests/unit/test_ds41_h_options.py` deleted -> `tests/unit/test_weights_nk.py`
    (nk bit-identity, V4.1 shapes with hidden, block_float_exact). Weight-cache keys never had `routed`.
  - READMEs, CHANGELOGs, INDEX.md updated. Build OK (warnings as errors).
Fork-side diff: 30 files, +131 / -1764.

Measured (2x2, defaults):
- Baseline first (untouched tree, 6 of the 8 gate component tests; the loop was cut at dec_reuse): every printed check
  line of the after-runs equals the before-runs (diff of the `ok` / `FAIL` / `note` lines: identical). err_experts
  L00 0.130168, L02 0.16062, L20 0.183528, L21 0.183309 (no before); err_shared_expert L00 0.0151197, L02 0.0179665,
  L20 0.0182662, L21 0.0176558.
- Gate: the 4 shared-expert tests PASS. The 4 experts tests FAIL, before and after, identically (DS41-95): err_experts
  is within its limit (e.g. 0.130 <= 0.337) but the frozen auto checks vs the CPU step (rel 0.13-0.18 > 0.06-0.08;
  second inputs' fixed 0.03) fail, and test_c_swa_experts.py KeyErrors on `component_col_factor`. These frozen tests
  (2026-10-16/17) predate F128 and the bfp4 / glm defaults; the brief's "pass today" is not so for them.
- L.last (run with --no-precompile to fit one 10-minute call, K-52): PASS; err_layer L00 .0265/.057, L01 .0346/.072,
  L02 .0523/.1077, L03 .0818/.1587, L14 .0618/.138, L20 .0306/.0507, L21 .0521/.1312, L24 .0281/.0418 (= FIX.8's);
  err_state max window_kv_L03 .0542; worst layer pcc 0.996652.
- Plain profile (chunk 53248-55296, BRINGUP_TASK=FIX.9): device 104.28 ms a chunk (FIX.8 104.31), 7800 programs,
  experts 16.77 (16.84), shared_expert 2.02, router 1.47, moe_add 1.58; host_transfers_per_layer 0; wall 355 ms.
- Forks, options off: unified fork_source 73 / 75 run and pass, 0 regressions; the 2 weight-cache tests are not
  collected (deepseek_v3_d_p/tests/conftest.py imports xing40's settings, which still pass the removed
  `max_precision=`; F137, not this change); model cases 5 passed / 1 skipped (xing40 4x2); test_weights_transposed
  3 passed. Flat tests/unit/ 19 passed / 13 failed = the recorded baseline (12 x `p.nsg == 1`, the MiMo perf band,
  DS41-63) + test_weights_nk 3 passed; GLM's test_flat_bench 9.556 / 23.725 ms (9.558 / 23.721).
- settings_lint 0 violations, ops_lint 0 problems.

Gotchas:
- Out of date now (pointers into deleted code): DS41-23, DS41-57, DS41-58, DS41-59, DS41-60, DS41-61, DS41-62
  (knowledge-use report FIX.9.overseer.1.json).
- A shell loop of run_safe_pytest calls longer than 10 minutes is moved to the background by the tool and then killed:
  run one test file per call.

Re-run: the FIX.9 gate (brief), one command per call (L.last with --no-precompile to fit 10 minutes); fork checks
`scripts/run_safe_pytest.sh --run-all --no-precompile ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/<file>.py`,
`python -m models.demos.common.bringup.testing.fork_source --fork unified_routed_expert_ffn`,
`scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/unified_routed_expert_ffn/tests/test_unified_routed_expert_ffn.py`.

## FIX.9 attempt 5 (overseer, 2026-10-22): the re-rendered experts tests and the devfmt diagnosis

State: attempt 1's deletions (section above) were still in the tree after the crash; kept unchanged (re-read the
diff; rebuilt with ./build_metal.sh, since some fork sources were newer than the library; fork option tests
test_weights_nk.py 3 passed, test_weights_transposed.py 3 passed).
Added `bringup/hooks.py:devfmt_component` (+ `_bfp_round`): the F137 diagnosis hook (testing/devfmt.py, never gated).
Without it the diagnosis said "a logic bug" (it modelled only bf16 activations). experts: the weights go through
EXPERTS_IMPL's block format on the stored [N, K] matrices (tt/experts.py:_bfp4_round), x / h / y as bfp8 tiles (glm),
the bf16 routing weight is applied after w2, and the sum is rounded to bf16. Other steps: the framework's default
(bf16 activations). No device path or setting changed.
Gate, run one command per call (overseer re-rendered tests, OV-27):
- 4 x shared_expert PASS: err L00 0.0151197, L02 0.0179665, L20 0.0182662, L21 0.0176558 (= the baseline).
- 4 x experts FAIL, identically to the baseline: err_experts L00 0.130168 / L02 0.16062 / L20 0.183528 / L21 0.183309,
  each within its golden limit (0.337 / 0.507 / 0.882 / 0.958). The vs-CPU check fails (rel > 0.0598 / 0.0720 / 0.0814 /
  0.0819, median row ratio +4.4 to +5.5 %), and so do the second inputs (0.18-0.20 > 0.03). Diagnosis with the hook: the
  CPU step at the device's formats is 0.128 / 0.158 / 0.181 / 0.181 from native, and the device is 0.0149 / 0.0171 /
  0.0184 / 0.0182 from it. Verdict: a precision gap (DS41-95).
- CPU only, random x, layer 0, the CPU step at the device's formats vs native: bfp4 + glm 0.1825 (ratio 1.047),
  bfp4 + full 0.1798, bfp8 + glm 0.0171 (1.006), bfp8 + full 0.0038. So the gap is bfp4 alone.
- L.last (--no-precompile, K-52) PASS: err_layer L00 .0265, L01 .0346, L02 .0523, L03 .0818, L14 .0618, L20 .0306,
  L21 .0521, L24 .0281; worst layer pcc 0.996652; err_state max .0542 (= FIX.8's).
Decision: no change to shipped numerics (the brief forbids it; rule 14: a precision gap is reported, not raised).
The four experts tests need an owner waiver (tests.waivers) for EXPERTS_IMPL=bfp4, or the owner picks bfp8.
Re-run: the gate, one test file per call; diagnosis lines: `grep "^note err_experts"` in the test output.

## P.17 perf attempt 1 (2026-10-22): ROPE_IMPL=on, one stock rotary op per RoPE site (precision-debt J3)

What: setting ROPE_IMPL (tt/settings.py, default on; off = the P.16 path, kept for the owner's A/B). On: every RoPE
site (attention q and output inverse, indexer q, window kv, compressor ckv / k: 34 a chunk) is ONE
`ttnn.experimental.deepseek_prefill.rotary_embedding_indexed` on bf16 heads (tt/rope.py:rope_stock). INDEX.md has no
bring-up rope fork, so the stock op is first in line (rule 16). K-11: rotary_offset ropes the 64-dim tail of a 512 /
128 head in place, so no weight row is permuted any more (wq_b, wkv, kv_norm, wo_a, compressor wkv / wgate / norm /
wk / k_norm, indexer wq_b all in checkpoint order); the Meta pair order with the trans-mat is the reference's
interleaved order; inverse = the transposed trans-mat.
Decisions:
- Rows: the model's layout is natural blocks, not the server's block-cyclic order the op's kv_actual_global math
  assumes, so each call gets this chip's rows pre-sliced (RopeTable.rows / CompressRope.rows, memoized per start as
  before) and kv_actual_global 0 (every chip reads from row 0; any 32-aligned start; one cached program).
- Tables bf16 [max_seq, 64] (each angle twice): the op refuses fp32 cos / sin (DS41-96, probe_001).
- Compute config: the op's default (HiFi4 + fp32 DEST). DS41-83: the trans-mat product is exact on bf16 only at
  HiFi3 / HiFi4; probe_002 tail vs float64: HiFi4 0.0037, HiFi2 0.0052 (one bf16 rounding 0.0017), 0.09-0.10 ms.
- q: the 512^-0.5 scale moved into the input's bf16 cast (multiply writing bf16: still one rounding, one op fewer);
  wq_b writes bf16 (DS41-84: noise); nlp_create_q_heads_split then rope_stock(tail, concat_prefix=nope). Indexer q
  likewise (no scale). Output: rope_stock(oh, trans_inv) on the bf16 heads, then the unchanged _out_proj.
- kv: kv_norm writes bf16 (TtDistributedRmsNorm(dtype=)), then rope_stock. Compressor: ckv = rope_stock of the
  latent cast to bf16 (wk still reads the fp32 latent); k_norm writes bf16, then rope_stock.
- Not deleted: PairPerm, the interleave helpers and the fp32 chains stay as the ROPE_IMPL=off path, because the
  orchestrator measures off as the old side. Once the owner accepts, delete with ROPE_IMPL: tt/rope.py PairPerm,
  deinterleave_perm, rope_deinterleaved(_bf16), interleave_tail, deinterleave_tail, rope_heads; the off branches in
  tt/attention.py (_q_glue, _q_heads, _out_glue, _out_heads, the table transform), tt/indexer.py (_query_glue and the
  permute path), tt/kv.py, tt/compressor.py; then rewrite DS41-15 / DS41-83.
Measured (2x2, chunk 53248-55296):
- Quick A/B side on (`measure --task P.17`): chunk wall 153.6 ms (spread 22.4), worst layer L03 pcc 0.99670,
  host_transfers_per_layer 0; every err_* within its limit: err_layer L00 .0261/.0570, L01 .0347/.0720, L02 .0520/.1077,
  L03 .0813/.1587, L14 .0605/.1380, L20 .0300/.0507, L21 .0507/.1312, L24 .0281/.0418; window_kv L00 .0042/.0585
  (FIX.9: .0034), state max .0541.
- Plain profile (on) vs FIX.5's profile of the same tree with the old path: programs (4 chips) 7800 -> 6140 (-415
  calls a chunk); device sum 104.33 -> 96.91 ms; attention 40.56 -> 34.50, indexer 12.93 -> 11.85, kv 1.18 -> 0.81,
  compressor 1.81 -> 1.65 (programs: attention 2244 -> 1476, indexer 376 -> 200, kv 800 -> 384, compressor 964 -> 664).
- Component steps vs golden, on / off (frozen tests run by hand, not this gate): kv L0 0.00417 / 0.00335 (limit
  0.0582), window attention L0 0.0214 / 0.0228 (0.0902), compressor L2 0.00361 / 0.00319 (0.217), compressed
  attention L2 0.0276 / 0.0298 (0.148); indexer overlap equal (L2 miss 0.00577 / 0.00576, L20 0.00340 / 0.00343).
Gotchas:
- Pre-existing, on and off alike (not this change): test_c_enc_source_indexer / test_c_dec_candidate_indexer fail
  their overlap limit (0.00065 / 0.00144); the gate's swap experts tests fail outside the waivers:
  test_swap_enc_source_14_experts err_swap_out 0.0296 on / 0.0294 off (<= 0.0159), swap_indexer match 0.0015 both,
  swap_router; test_swap_swa_12_experts err_swap_out 0.0248 on / 0.0267 off (<= 0.0185), swap_router_vs_golden_err
  0.149 on / 0.151 off (<= 0.142). test_c_swa_experts passes (waived).
Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.17`; probe
`scripts/tt-probe.sh ds41_rope_stock < tests/ttnn/unit_tests/operations/ds41_rope_stock/probes/probe_002.py`.

## P.18 perf attempt 1 (2026-10-22): NORM_IMPL=on, the RMS norms through ttnn.bringup.rms_norm (precision-debt J4)

What: setting NORM_IMPL (tt/settings.py, default on; off = the P.17 path, kept for the owner's A/B). On, every norm
(tt/norm.py:TtDistributedRmsNorm._fork) is ttnn.bringup.rms_norm at the fork's normal settings (HiFi4, fp32 DEST):
- column-split (attn_norm, ffn_norm, final norm): rms_norm(x, stats_out=True) -> ccl.all_gather of the [S/SP, 32 B]
  fp32 partials over TP (dim 3) -> rms_norm(x, partial_stats=, stats_width=H, weight=, dtype=): 3 calls (+ ffn_norm's
  output all_gather, as before). TP is read from the mesh; any TP works (galaxy 4x4 / 8x4: 4 partial sets gathered).
- q_norm / kv_norm: one call on the fused [wq_a | wkv] output p with input_window=(c0, c1), no slice (tt/q_a.py,
  tt/kv.py pass the window; the off path slices inside the norm, as before); kv_norm / k_norm write bf16 (dtype=);
  the compressor norm (fp32 in / out, whole rows) is the fork's default program.
Fork extension (rms_norm_ttnn CHANGELOG 6, README, INDEX): a second device operation behind the new keyword options
(stats_out, partial_stats, stats_width, input_window, dtype), routed to only when one is set; the default program is
untouched. Kernels kernels/rms_norm_split_*.cpp, host device/rms_norm_split_device_operation.*.
Decisions:
- Stats exchange as partials (K-12: stock rms_norm_pre_all_gather leaves junk in stats columns 1-31 on fp32 and post
  reduces the whole tile): stats_out writes a partial in EVERY element (packer-L1-accumulated x^2 tiles, fp32), so the
  apply reduces all columns of all gathered tiles. One gather of [1024, 64] fp32 per norm (0.04 ms) instead of an
  all_gather of the rows (10 MB fp32) + mesh_partition.
- The statistic is FPU-grade (DS41-44: x^2 and the row sum read fp32 at TF32); no exact-statistics option (brief).
  DS41-97: the weight multiply moved into DEST on the SFPU (v1 through an fp32 L1 intermediate: ratio -1.4e-3).
- attn_norm / final norm keep writing fp32 (consumers unchanged; ATTN_NORM_DTYPE is J5's), ffn_norm FFN_NORM_DTYPE.
- No stock norm op is called (rule 16): the INDEX line now lists rms_norm_pre/post_all_gather and slice + rms_norm
  under the fork's "Consider instead of".
Measured (2x2, chunk 53248-55296):
- Quick A/B side on (`measure --task P.18`): chunk wall 127.9 ms (spread 49.4; P.17 on: 153.6, spread 22.4),
  host_transfers_per_layer 0, worst layer L03 pcc 0.99678; every err_* within its limit, equal to P.17 within noise:
  err_layer L00 .0253/.0570 (P.17 .0261), L01 .0337/.0720, L02 .0509/.1077, L03 .0803/.1587, L14 .0603/.1380,
  L20 .0301/.0507, L21 .0512/.1312, L24 .0282/.0418 (.0281); window_kv L02 .0362/.0954 (.0330), L03 .0555/.1314.
- Plain profile (on; profiles/P.18_20261022T100716.json) vs P.17's: device 96.91 -> 93.18 ms a chunk; programs (4
  chips) 6140 -> 4580 (-390 calls a chip a chunk); attn_norm 1.83 -> 0.89 ms (programs 448 -> 96), ffn_norm 2.11 ->
  1.19 (480 -> 128), q_a 4.76 -> 3.79 (512 -> 192), kv 0.81 -> 0.22 (384 -> 64), compressor 1.65 -> 1.37 (664 -> 448);
  the norm-touched sections 11.16 -> 7.46 ms.
- Probe ds41_norm_split/probe_002 (bf16-valued weights, vs float64): split norm fp32 rows -> fp32 rel 6.1e-4, ratio
  -3.5e-4; -> bf16 rel 1.77e-3 (one bf16 rounding 1.66e-3); bf16-valued rows ratio -1.6e-4; deterministic.
- Swap test swa_12 by hand: attn_norm vs CPU rel 3.2e-4 (<= 3.3e-3; off 0), ffn_norm 1.77e-3 (<= 3.3e-3; off 1.66e-3),
  q_a 2.9e-3 (<= 0.024), kv 4.2e-3 (<= 0.058). The test still fails, identically with NORM_IMPL=off (pre-existing, P.17
  gotchas): err_swap_out 0.0260 on / 0.0248 off (<= 0.0185), swap_router_vs_golden_err 0.1457 on / 0.1489 off
  (<= 0.1425), swap_router vs CPU 7.9e-4 both (<= 3.6e-4).
- Fork regression with the options off: unit 1375 passed / 47 skipped (= 1327 + 48 new), fork_source 263 tests 0
  regressions, model cases 10 passed / 2 skipped (8-device cases).
Gotchas:
- unary_bcast of an fp32 CB unpacks to DEST (exact; srcB is 19 bits): the way to bring an fp32 operand into an SFPU
  step without TF32 truncation.
- O.1 should add the model's new calls (stats_out [1024, 2560] fp32, partial_stats [1024, 128], input_window q / kv
  of [1024, 1792], k_norm dtype bf16 [512, 128]) to the fork's cases.
- When the owner accepts: delete NORM_IMPL=off (TtDistributedRmsNorm.rstd, the mhc_pre_xing_pack statistic, the
  slice branch) and the norm-only exact sums in tt/ccl.py if nothing else uses them.
Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.18`; fork tests
`scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/rms_norm_ttnn/tests/unit/test_rms_norm_ttnn_split.py`;
probe `scripts/tt-probe.sh ds41_norm_split < tests/ttnn/unit_tests/operations/ds41_norm_split/probes/probe_002.py`.

## P.19 perf attempt 1 (2026-10-22): GLUE_FOLD=on, the rest of the fusion report (owner comparison)

What: setting GLUE_FOLD (tt/settings.py, default on; off = the P.18 path, kept for the owner's A/B). On:
- (a) mHC exchange (report item 7): attn_hc / ffn_hc send the chips' fp32 partial rows [S/SP, 32] through ONE
  ccl.all_gather on dim 1 -> [1, TP, S/SP, 32]; ttnn.bringup.mhc_pre_xing(partials=TP) adds them in mesh-column order
  on the SFPU (fork mhc_pre CHANGELOG 10). Was: typecast fp32 -> bf16, bf16 fabric reduce-scatter, all_gather,
  typecast back (DS41-82). Numerics: the exact fp32 sum (= CCL_IMPL=fabric's / stock gather_add), not the bf16 RS:
  closer to the reference's fp32 mixes; any TP (4x4 / 8x4: partials=4).
- (b) mhc_post (item 9): post / comb read from the packed [S/SP, 24] mixes (post_col=4, comb_col=8; both programs),
  and the stats pass takes the bf16 attention / MoE sums as F (fork mhc_post CHANGELOG 8); moe_add's
  experts.reduce(fp32=False). Removes 2 slices + 1 typecast per residual and the MoE _to_fp32 (bit-identical: bf16 ->
  fp32 is exact).
- (c) fabric_reduce_scatter(program_hash=True) (item 4; fork CHANGELOG "program_hash"; DS41-98).
- (d) window path (item 6): no pad mask when end - start == chunk; TtWindowCache.keep_rows keeps the chunk's kv rows
  when the chunk is whole, and the next chunk at start = end slices its carry from their last 128 rows (any other
  write start, a host load or free() drops them; the cache read stays the fallback); WindowTables.srv_row memoizes the
  server-order index row of the latest start (8 window + 6 compressed writes a chunk); tt/served.DeviceScalars(memo=)
  keeps the latest 8 sliced scalars (the r2 writes' slot / start / end).
- (e) deallocations through the binding (tt/ccl.py:dealloc, bound at load by ccl.configure; every tt/ forward
  module): ~1.3 us of the wrapper a call. The bring-up profiler's op sequences skip them on both of its runs alike.
- Not done: the moe_add fold into the reduce-scatter (report item 10): it removes one add a layer, measured ceiling
  0.09 ms host + 0.073 ms device a call = ~1.3 ms a chunk, against new reader code in every port / final kernel of
  the RS; the combine planner / zero / barrier generic_ops (24 a chunk, 0.09 ms host): their descriptors bake
  addresses per object, so a custom hash per object would add program-cache misses whenever addresses change; the
  engram rows (report item 8): ROW_MAJOR upload + device tilize = TILE upload (3.4 ms both, probe ds41_p19_fork), the
  host cost is the table lookup itself (DS41-99).
- stock ttnn.slice instead of ttnn.bringup.rms_norm: the carry is a row slice of the kept kv rows (no norm).
Measured (2x2, chunk 53248-55296):
- Final `measure --task P.19` (quick A/B side on, rc 0): chunk wall 118.8 ms (spread 39.1; P.18 127.9), worst layer
  L03 pcc 0.99680, pcc_chunk_out 0.99960, host_transfers_per_layer 0; every err_* within its limit: err_layer L00
  .0250/.0570, L01 .0337/.0720, L02 .0508/.1077, L03 .0800/.1587, L14 .0605/.1380, L20 .0301/.0507, L21 .0510/.1312,
  L24 .0282/.0418 (P.18: .0253 .. .0282); state worst window_kv_L03 .0525/.1314.
- Plain profile (on; profiles/P.19_20261022T114503.json) vs P.18's: device 93.18 -> 91.87 ms a chunk; programs (4
  chips) 4580 -> 3776 (-201 program calls a chunk); attn_residual 3.31 -> 2.52 ms (programs 256 -> 32), ffn_residual
  3.29 -> 2.88 (248 -> 56), moe_add 1.58 -> 1.22 (96 -> 64), attn_hc 2.56 -> 2.64 (184 -> 88), ffn_hc 1.46 -> 1.57
  (160 -> 64), attention 34.49 -> 34.32 (1476 -> 1380), compressor 1.37 -> 1.37 (448 -> 380).
- Host ms per section, off -> on (10 warm clean-wall runs, a FastOperation timer + profiler.signpost hook, a test
  under generated/ (not committed); calls = outermost ttnn calls, the on side's binding deallocations not counted):
  wall 130.3 -> 117.5 ms; in-layer host 102.3 -> 83.0 ms; attention 29.88 -> 26.18, attn_hc 6.27 -> 2.28, ffn_hc 4.53
  -> 1.39, attn_residual 3.25 -> 0.86, ffn_residual 2.91 -> 1.02, moe_add 4.55 -> 3.40, compressor 8.80 -> 7.91,
  experts 11.83 -> 9.74, router 6.97 / 7.04, q_a 5.27 / 5.40, indexer 4.85 / 4.82; outside the layers begin_chunk +
  embed 29.1 / 30.9, final drain 2.0 / 3.2. fabric_reduce_scatter 49 -> 33 calls, 0.305 -> 0.271 ms host a call.
  The wall is now the host dispatch (DS41-99).
- Ladder s4096 by hand (on; chunk 2 takes its carry from chunk 1's kept rows): pass, err_layer L00 .0267/.0561 ..
  L03 .0823/.1558 (FIX.8: .0273 .. .0847), state worst window_kv_L03 .0582/.1334.
- Fork tests: new tests/unit/test_xing_partials.py (13), test_mhc_post_mixes.py (19), test_fabric_rs_program_hash.py
  (9), all bit-identical; regressions with the options off: mhc_pre unit + golden 345 / 1 skipped (= 332 / 1 + 13),
  model cases 103; mhc_post unit + golden + cases 401 (= 382 + 19); fabric RS unit 19 / 13 skipped, cases 2. Build OK.
Gotchas:
- The hashed RS path needs its own descriptor cache entries (DS41-98): sharing the default path's dkey silently reuses
  an unhashed descriptor.
- The carry from the kept rows never hits in the repeated-chunk profile (same start every run, the prefix reload drops
  it); only consecutive chunks (ladder s4096 / s55296, serving) use it.
- When the owner accepts: delete GLUE_FOLD=off (ccl.tp_sum's bf16 path for the hc rows, the mhc_post slices / typecasts,
  the window mask for whole chunks), and the wrapper branch of ccl.dealloc.
Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.19`; fork tests
`scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/mhc_pre_ttnn/tests/unit/test_xing_partials.py`,
`.../mhc_post_ttnn/tests/unit/test_mhc_post_mixes.py`, `.../fabric_reduce_scatter_ttnn/tests/unit/test_fabric_rs_program_hash.py`.

## P.21 perf attempt 1 (2026-10-22): STATIC_TABLES=on, one persistent attention table per KV source; trace ceiling

What: setting STATIC_TABLES (tt/settings.py, default on; off = the P.19 path, kept for the owner's A/B). Fusion report
item 3. On:
- tt/compressor.AttnTable: per KV source and slot (allocated with the source's served compressed KV, `TtServedRows(
  attn_table=True)`; TtCompressCaches passes it, so the device model, the hybrid and the serving runtime all get one)
  ONE replicated bf16 ROW_MAJOR buffer [hist + 128 + S, 512], hist = the cache's token capacity / r: the history in
  natural order (group j at row j), then the window region at base = hist.
- TtServedRows.write also slice_writes the replicated natural-order rows it was given (the compressor's `rm`) at row
  start / r: one `ttnn.experimental.slice_write` a source a chunk; TtServedRows.load_host (golden prefix, migrated
  prefix, hybrid) writes the loaded rows the same way, so the table follows the served cache on every write path.
- TtSparseAttention._static: the layer's [carry | chunk] window rows slice_written into the region (2 slice_writes;
  the source's layers run in order on one queue and reuse it), index list = [WindowTables.index_lists(base)[class] |
  the top-k as it is] (mesh_partition + concat, + the same compaction gather at starts < 128), memoized for the
  shared top-k as before. Gone per chunk: the history slice + to_layout + 2 all_gathers + to_layout (56 MiB at r1),
  the [window | history] concat, the 27-op fp32 id remap (`_topk_rows`). The table's shape is the same in every
  chunk (no program-cache misses as the history grows).
Decisions:
- History first, window region after it: top-k ids need no transform (0xFFFFFFFF stays the sentinel); only the
  load-time window ids carry the offset, built per state at state creation (index_lists(base), 5 small lists).
- The table is kept in step by the served cache's own writers instead of being rebuilt from the cache: the cache has
  no other writer on a prefill node (serving_contract.md "Input": a follow-up remounts its own slot; migrated prefixes
  come in through state.load_served -> load_host).
- Window-only layers (L0, L1) keep their [carry | chunk] concat (2 MiB, already one shape per chunk).
- Stock ops only (slice_write, concat, gather, mesh_partition): no INDEX entry lists them (rule 16).
Measured (2x2, chunk 53248-55296):
- Final `measure --task P.21` (quick A/B side on, rc 0): chunk wall 103.5 ms (spread 6.1; P.19 on: 118.8, spread
  39.1), host_transfers_per_layer 0, worst layer L03 pcc 0.99680, pcc_chunk_out 0.99960; every err_* within its limit
  and equal to P.19: err_layer L00 .0250/.0570, L01 .0337/.0720, L02 .0508/.1077, L03 .0800/.1587, L14 .0605/.1380,
  L20 .0301/.0507, L21 .0510/.1312, L24 .0282/.0418; worst state window_kv_L21 .0458/.1118.
- Ladder s4096 by hand (on; chunk 0 at start 0 takes the compaction path, chunk 1 a grown history): pass, err_layer
  L00 .0267/.0561, L01 .0353/.0734, L02 .0541/.1071, L03 .0823/.1558, L14 .0551/.1204, L20 .0303/.0519, L21
  .0508/.1265, L24 .0305/.0449 (P.19: L00 .0267 .. L03 .0823).
- Trace ceiling (owner; tt/perf_trace.py, standalone, not the model's path; STATIC_TABLES=on): the 8-layer stack at
  start 53248, inputs made before the capture: traced replay 87.0 ms a chunk (10 replays 87.0-87.1), untraced
  layer stack median 103.7 ms (87.9-199.0), begin_chunk (engram host input) 38-40 ms warm outside it; replay output
  vs untraced PCC 1.000000, vs golden 0.999605 (= untraced). The forward captures as it is; what blocked the first
  capture: the flat expert's address-keyed descriptor (DS41-100: settle until no new key, 6 runs).
- Not run: the plain profile (the orchestrator runs it with the change on and off).
Memory (rule 15): a replicated copy of each source's compressed KV per slot. 2x2 at 57,344: r2 28 MiB (L2, L14), r1
56 MiB (L20) + 2.1 MiB of window region each a chip. Full model: 4 sources (L2, L8, L14 r2, L20 r1): per chip per
slot 1.25 GiB at 524,288 and 2.5 GiB at 1,048,576 on every box (replicated: the chip count does not divide it;
the served caches themselves are 384 + 256 MiB at 1M before dedup).
What still blocks a chunk-to-chunk trace (code reading, printed by perf_trace.py): host per-start values baked into
the commands (rope row slices, srv_row / update_padded_kv_cache scalars / DeviceScalars.at, the window carry slices,
pad-mask scalar and start class, the ring score's chunk_start_idx / kv length and the candidate widths, the top-k /
candidate stores keyed by start), the address-keyed flat-op / planner descriptors (DS41-100), and the per-chunk
engram rows / ids uploads into new tensors.
Gotchas:
- K-52: ran perf_trace.py with --no-precompile (a model-sized test).
- When the owner accepts: delete STATIC_TABLES=off (sparse_attention._compressed, _topk_rows, _index, TOPK_LIMIT,
  the ChunkMemo "comp" slot and model.memo_last's ("comp", src) entries, the [window | comp] concat).
Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.21`; trace ceiling
`PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tt/perf_trace.py`.

## FIX.10 overseer (attempt 1, 2026-10-22): "remove P.21's second copy of the compressed history": DISAGREE, no code change

Request: make the served compressed cache and P.21's AttnTable one storage, to recover "+1.25 GiB per chip per slot at
524,288 (+2.5 at 1M)" on galaxy_4x4. Answer: disagree (knowledge-use report, overseer FIX.10). Nothing in tt/,
hooks.py or the forks changed.
Why (DS41-101):
- The served copy is not a second full copy. 1_compress_kv_r2 / 3_compress_kv_r1 are TP-dedup and SP-sharded: each
  chip holds 1 / (SP x TP) of a source's history (tt/compressor.py TtServedRows: owners = sp x tp, rows =
  ceil(max_seq / chunk) x W / TP). The AttnTable is the whole history on every chip, because sparse_sdpa reads only its
  own chip's table and a top-k id may name any earlier group. The 1.25 GiB is that table.
- Measured (probe tests/ttnn/unit_tests/operations/ds41_fix10_mem/probes/probe_001.py, 2x2, DRAM allocated per chip
  via ttnn.get_memory_view, one TtServedRows(dedup=True, attn_table=True), STATIC_TABLES off / on), MiB per chip:
  max_seq 55,296: r2 6.87 / 37.12 (served slab 6.75, table 29.12 = 27,648 history + 128 + 2,048 window rows);
  r1 13.52 / 70.88 (served 13.50, table 56.12). max_seq 57,344: r2 7.13 / 38.38; r1 14.02 / 73.38. This run's three
  sources (L2, L14 r2; L20 r1) at 55,296: served 27.0 MiB, tables 114.4 MiB a chip.
- Full model, 4 sources (L2, L8, L14 r2; L20 r1), 1 KiB a row, chunk 4096 (window region 4,224 rows = 4.1 MiB each),
  per chip per slot: table 1.266 GiB at 524,288 and 2.516 GiB at 1,048,576 on every box. Served dedup copy:
  galaxy_4x4 80 MiB (524k) / 160 MiB (1M); galaxy_8x4 40 / 80 MiB. Today (P.21): 4x4 1.344 / 2.672 GiB, 8x4 1.305 /
  2.594 GiB. One storage readable by sparse_sdpa must still hold the whole history per chip: 1.266 / 2.516 GiB on both
  boxes. The merge saves 80 MiB (4x4, 524k), 40 MiB (8x4, 524k), 160 / 80 MiB at 1M: 6 % of the 1.25 GiB.
- What the merge would cost: the served configs replicated over SP and TP in natural order (serving_contract.md
  "Placement" shards the sequence over SP rows; the KV Manager would write each migrated record to all 16 chips), and a
  sparse_sdpa fork option to read an ND-sharded engine tensor [slots x layers, 1, rows, 512] at a (slot, layer) base row
  with the window region inside it. Not worth 80 MiB.
- Budget at 524k on galaxy_4x4 (plan_boxes.json gb_at_deploy_seq 26.49 GiB, which counts ONE r1-sized table, 0.5 GiB,
  plus the 80 MiB served copy): P.21 adds 3 r2 tables + 3 window regions = +0.76 GiB -> ~27.25 of 27.2; the merge
  would give ~27.17. But 26.49 predates two shipped settings: dense fp8 weights as bfp8 (DENSE_WEIGHTS, P.13; plan.md:
  bf16 dense costs +1.5 GiB on 4x4) and routed experts as bfp4 (0.5625 B a value vs the plan's packed 0.531: about
  +1.0 GiB). Estimate with both: ~26.7 GiB with P.21 as it is. The plan check should be re-run with the shipped dtypes
  and the per-source tables before anyone trades speed for memory.
- If memory is needed, options that do cut it (not done; a perf / plan decision): (a) the plan's one shared r1-sized
  table refilled per source per chunk (0.5 GiB at 524k; pays the per-chunk history gather P.21 removed); (b) keep the
  persistent table only for L20 (read by layers 20-39) and one shared r2 buffer refilled per r2 source per chunk
  (0.77 GiB at 524k, -0.5 GiB vs P.21; 3 r2 history gathers per chunk).
Gate: not run (no code changed since P.21's passing gate; agent rule 8). Chunk wall / device time / err_* unchanged:
P.21's quick wall 103.5 ms, err_* as in the P.21 section.
Re-run the measurement: `PYTHONPATH=$PWD scripts/tt-probe.sh ds41_fix10_mem < tests/ttnn/unit_tests/operations/ds41_fix10_mem/probes/probe_001.py`.

## FIX.10 overseer (attempt 3, 2026-10-22): disagree kept, no code change; the gate's failures are the indexer swap checks (DS41-102)

- FIX.10: answer stays `disagree` (attempt 1's evidence, accepted by the overseer 14:20; DS41-101). Nothing in tt/,
  hooks.py or the forks changed. Memory, chunk wall (quick 103.5 ms), device time and err_* are P.21's.
- OV-31: done. The owner waivers are in the spec (5fb7d75926d); swap swa_12 and swa_engram_15 now pass (orchestrator log
  FIX.10.log, 14:46-14:54).
- Gate status on P.21 code (the first full run of P.21's frozen tests; P.21 was accepted on a quick A/B):
  - test_c_*_experts (8): pass (FIX.10.log).
  - swap enc_source_14: FAIL, swap_indexer_vs_cpu match 0.0015 (>= 0.922), vs_golden 0.0015 (>= 0.830), near ties
    flips 1967 unexplained 1471 recall 0.9942 (FIX.10.log). Every other failed check there is waived.
  - swap enc_source_engram_17 (my run): FAIL, the same indexer checks (match 0.0016 >= 0.860; unexplained 218, recall
    0.9979) and pytest-timeout 300 s (the test ran 300.65 s; 291 s on the earlier run).
  - dec_candidate_14 and dec_reindex_13 run an indexer through the same checks: expect the same. enc_reuse_12,
    dec_reuse_12, the P.21 ladder and the profile: not reached.
  - K.1 contract tests (my runs, P.21 code): test_contract PASS (691 s; 335 s at K.1), test_kv_layout PASS,
    test_adapter_acks PASS, test_runner_contract PASS, test_runner_smoke PASS.
  - Ladder L.last: not finished. It runs > 20 min, beyond the agent's tool limit (10 min foreground + 10 background),
    and was stopped by SIGTERM at 1200 s (card marked dirty, run_safe_pytest resets it on the next run).
- Indexer diagnosis (CPU only, /tmp/fix10diag/idx_ties.py, layer 2, golden inputs, NF.flips, details in
  known_issues_detail.md DS41-102): native scores rounded to bf16 (topk_large_indices is bf16-only) give 1277
  unexplained / recall 0.9946 (device 1471 / 0.9942); the official bf16 arithmetic gives 2048 / 0.9638. The match is
  order: the reference returns ids ascending, the device by score (the native set by score: 0.0014). K-42 says index
  outputs need topk_overlap. Not a logic bug, not from P.21: a framework / frozen-test or waiver decision.
- No fix in the model: sorting the device ids would add a device sort only for the test and still not pass (a flip shifts
  positions), and fp32 top-k scores would raise precision above the official code (rule 7).

## P.20 perf attempt 1 (2026-10-22): BF16_ACTIVATIONS=on, activations in bf16 where the GPU code keeps bf16 (J5)

What: setting BF16_ACTIVATIONS (tt/settings.py, default on; off = the P.21 path, kept for the owner's A/B). On, per
bringup/official_taps.yaml:
- the four mHC streams bf16 (block boundary and h_mid): embed concats the bf16 embedding rows (no typecast), the
  residuals (mhc_post, plain and stats_weight) and the engram add write bf16, tt/layout.streams_to_device (harness
  boundary, from_host) uploads bf16 (tt/common.stream_dtype);
- attn_collapse / ffn_collapse / final collapse bf16 (mhc_pre_xing writes y in the streams' dtype);
- attn_norm and the final norm write bf16 (build_norm / TtDistributedRmsNorm dtype=act_dtype());
- the [wq_a | wkv] exchange stays bf16 (ccl.tp_sum(fp32=False): the fabric_bf16 RS + gather result without the
  typecast back), q_norm writes bf16 (q_resid), kv_norm already bf16;
- wo_a writes bf16 (no cast after it); engram wkv writes bf16 (the engram add reads it as is);
- the compressor's norm writes the latent bf16 (the rope reads it as is, no typecast), wk writes bf16.
fp32 kept where the reference is fp32: the hc mixes / Sinkhorn, every norm statistic, the engram gate sums (fp32 SFPU
on the exact bf16 values), the compressor's projection + pair pooling, the router; fp32 accumulation in every matmul.
Requires MHC_FUSED, GLUE_FOLD, ROPE_IMPL, NORM_IMPL on (asserted in TtDS41Model).
Fork extensions (bring-up ops, rule 16; option = bf16 inputs, fp32 programs unchanged):
- ttnn.bringup.mhc_pre_xing / mhc_pre_xing_pack: bf16 streams and engram keys (CHANGELOG 11; defines MHC_XING_X16 /
  MHC_ENGRAM_X16; srcA reconfigured between the fp32 / bf16 CBs; UnpackToDestFp32 on the bf16 CBs: exact);
- ttnn.bringup.mhc_post(stats_weight=): bf16 X with bf16 F (CHANGELOG 9; MHC_POST_STATS_X16; X' and the matmul's
  X' bf16, the sum of squares from the fp32 mix).
- The hybrid's collapse with a bare [R, 4] pre on bf16 streams pads the pre to 24 columns and uses
  mhc_pre_xing(coefficients_given=True) (tt/collapse.py; harness only).
Measured (2x2, chunk 53248-55296):
- Final `measure --task P.20` (quick A/B side on, rc 0): every err_* within its limit and equal to P.21 within noise:
  err_layer L00 .0248/.0570 (P.21 .0250), L01 .0338/.0720, L02 .0514/.1077, L03 .0805/.1587, L14 .0605/.1380,
  L20 .0299/.0507, L21 .0513/.1312, L24 .0280/.0418; worst state window_kv_L03 .0543/.1314 (P.21 .0525);
  compress_kv_L02 .0382/.2095 (.0390), index_k_L02 .0205/.2404 (.0210); worst err/limit 0.67; min layer PCC L03
  0.99676 (P.21 0.99680), pcc_chunk_out 0.99961, host_transfers_per_layer 0.
- Wall: chunk_wall_ms 142.1 (spread 27.5) on a loaded box (a golden generation at ~7 cores + a selftest, load
  13-19); the off side run by hand right after, same command and load (DS41_BF16_ACTIVATIONS=off BRINGUP_QUICK=1
  BRINGUP_PROFILE_WALL=1 test_profile.py): 161.3 (spread 57.3). P.21's 103.5 was on an idle box: compare only the
  orchestrator's A/B pair.
- Plain profile (on; profiles/P.20_20261022T202929.json) vs FIX.10's (P.21 code, off): device 88.33 -> 83.96 ms a
  chunk (-4.4); programs (4 chips) 3328 -> 3216; mHC / residual / norm / engram sections: attn_hc 2.65 -> 1.75,
  ffn_hc 1.57 -> 1.12, attn_residual 2.50 -> 1.80, ffn_residual 2.87 -> 2.44, engram_add 1.79 -> 1.34, engram_kv
  4.94 -> 4.62, attn_norm 0.89 -> 0.60, ffn_norm 1.17 -> 1.00, q_a 3.79 -> 3.12 (programs 192 -> 128), attention
  30.72 -> 30.38, indexer 11.86 -> 11.73; moe_add 1.52 -> 1.96 (its code is unchanged: the RS waits on the slowest
  chip of the row).
- Memory per chip (computed from the shapes, not measured; weights unchanged): a stream copy [1024, 4 x 2560] 40 ->
  20 MiB (2x2; 4x4 [1024, 4 x 1280]: 20 -> 10 MiB), about 3 live a block (in, h_mid, out); collapse / attn_norm
  [1024, 2560] 10 -> 5 MiB, q_resid 5 -> 2.5, the [wq_a | wkv] output 7 -> 3.5, engram kv [1024, 12800] 50 -> 25
  MiB: peak activations about -80 MiB a chip at 2x2.
- Hybrid path: test_swap_swa_engram_15_experts passes (every failure waived, as before); attn_hc vs CPU rel 0.0021 ->
  0.0012, ffn_hc 0.0027 -> 0.0016 (the FPU reads bf16 exactly, fp32 at TF32: DS41-104); err_swap_out 0.0235 (0.0234).
- Fork tests: tests/unit/test_xing_bf16_streams.py (10) and test_mhc_post_bf16_streams.py (7): bf16 paths
  bit-identical to the fp32 programs on the same values up to the packer's one rounding (to nearest, ties away:
  DS41-103); regressions with fp32 inputs: mhc_pre unit + golden 355 / 1 skipped (= 345 / 1 + 10), model cases 103;
  mhc_post unit + golden + cases 408 (= 401 + 7). Build OK.
Gotchas:
- The packer's fp32 -> bf16 rounds ties away from zero (DS41-103): a bit-exact test against torch's .bfloat16() fails
  on ~1e-5 of values.
- A background orchestrator `measure` killed at the tool's limit leaves its pytest running orphaned; stop it before
  re-running (the next safe run resets the cards).
- When the owner accepts: delete BF16_ACTIVATIONS=off with the fp32 stream paths that only it uses (embed's typecast,
  tp_sum's fp32 default for q_a, the compressor's latent cast); the fp32 fork programs stay (other models).
Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.20`; fork tests
`scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/mhc_pre_ttnn/tests/unit/test_xing_bf16_streams.py` and
`.../mhc_post_ttnn/tests/unit/test_mhc_post_bf16_streams.py`; profile (on)
`TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000 scripts/run_safe_pytest.sh --run-all --no-precompile
models/demos/common/bringup/tests/test_profile.py`.

## P.22 perf attempt 1 (2026-10-22): router through the fused moe_grouped_topk (ROUTER_IMPL=fused, owner comparison)

What: setting ROUTER_IMPL (tt/settings.py, default fused; ops = the P.21 chain). fused = the gate matmul (unchanged:
bf16 x, bf16 gate, HiFi2 + fp32 DEST, fp32 logits) then ONE ttnn.bringup.moe_grouped_topk(n_groups=1,
score_func="sqrtsoftplus", top 6, route_scale 1.5, eps 1e-20, stable_sort, fp32_selection=True,
fp32_activation=True) instead of softplus, sqrt, add, topk, gather, sum, add, multiply, div (tt/router.py:_fused).
Outputs: weights bf16 TILE, ids uint16 TILE: what the weighted experts read (tt/weighted_combine.py:routing now skips
the bf16 typecast when the weights are bf16; the uint32 -> uint16 typecast of the old topk ids is gone too). The
recentred fp32 bias is a load-time [max_rows, E] tensor (the op takes a full-shape bias). The dense routing (hybrid /
swap tests only) is the scatter of those weights typecast to fp32.
Decisions:
- The source commits are already in this tree (71062e76a38 / 570d95b6e7d = 5e47659815e / 27c5c2a16ae + the rebase fix
  9780c18f0d0), not only on Marko's branch: stock ttnn.experimental.deepseek_prefill.moe_grouped_topk has
  fp32_selection here. I forked it anyway (owner instruction, and it needed a change):
- DS41-105: stock moe_grouped_topk builds its compute kernel without INP_FLOAT32, so sqrt(softplus) runs the bf16-grade
  polynomial (degree 6, 0 below -5) even with fp32_selection: keys max error 8.2e-2 vs float64 on the same logits
  (the chain: 2.5e-5), and 0-13 of 2048 rows a layer picked other experts than the chain (scaled golden s4096 chunk
  1). Fork option fp32_activation (default off = the source program; define INP_FLOAT32 on the compute kernel only):
  keys bit-identical to stock ttnn.sqrt(ttnn.softplus) on fp32. Setting ROUTER_SOFTPLUS {fp32 (default), bf16}.
- Weights bf16 (the op has no fp32 output): the reference's own dtype (model.py Gate: weights.type_as(x)); the experts
  read bf16 weights in both paths. Router step checks now sit at one bf16 rounding (DS41-106, waived).
- No fork of the experts side and no uint32 ids from the gate (one typecast left per layer, 0.05 ms host).
Measured (2x2, scaled model, chunk 53248-55296 unless noted):
- Routing vs the old path on the golden (s4096 chunk 1, golden ffn_norm as bf16 input, 8 layers, 2048 rows each;
  probes tests/ttnn/unit_tests/operations/ds41_router/probes/probe_005.py / probe_006.py): selected experts identical
  on 2048 / 2048 rows of every layer; weights rel 1.53e-3..1.65e-3 vs the chain's fp32 weights (= one bf16 rounding);
  on same-set rows vs float64 on the device's logits rel = one bf16 rounding, bias <= 2.5e-5; both paths 25-45 rows a
  layer differ from float64 on the host's logits (the matmul's TF32 partials, waived); deterministic. Without
  fp32_activation: 0-13 rows differ from the chain.
- One router call (layer 0, 1024 rows a chip, 50 calls): host 0.583 -> 0.070 ms, async wall 0.598 -> 0.131 ms.
- Real-time profile with op rows (BRINGUP_PROFILE_OPS=1, both sides by hand, same load; profiles
  P.22_20261022T221748.json (fused) / P.22_20261022T222358.json (ops)): host calls per chunk 777 -> 697 (router 80 ->
  16, experts 96 -> 80); router section host 7.21 -> 1.23 ms, device 1.50 -> 1.04 ms a chunk, programs (4 chips)
  384 -> 64; device total 83.52 -> 82.97 ms; host dispatch (op mode) 459.5 -> 419.2 ms.
- Quick A/B side on (`measure --task P.22`, rc 0): every err_* within its limit and equal to P.20 to 4 digits
  (worst L03 0.08050 / 0.15875, min layer PCC 0.99676, pcc_chunk_out 0.99961), host_transfers_per_layer 0,
  chunk_wall_ms 102.0 (spread 30.9; box loaded by a golden generation; the orchestrator's off side is the comparison).
- Gate units passed with --check: swa (C experts), all 8 swap *_experts units (router step WAIVED as before: vs CPU rel
  1.75e-3..2.44e-3, 0 unexplained flips, recall_tie 1.0). Not run by me: the 7 other C.*.experts units (golden
  routing, the router is not in them), the ladder and the profile.
Fork: ttnn/ttnn/bringup/moe_grouped_topk (CHANGELOG, README, INDEX entry); source tests: the op's 5 nightly files,
203 / 203 pass on the original and the fork (tests/source_baseline.json); unit tests tests/unit/test_fp32_activation.py
(7 pass). Build OK.
Gotchas:
- ttnn.topk on fp32 returns UINT32 ids (the experts typecast them to uint16 then back to uint32); the fused gate gives
  uint16.
- tt/experts.py: routing may hand its inputs through (bf16 weights, a one-row mesh); the frees skip them now.
Stock calls added (F113):
- stock ttnn.typecast instead of ttnn.bringup.mhc_post: the fused gate's bf16 weights -> fp32 for the dense routing
  scatter (hybrid / swap tests only; no mHC step there).
- stock ttnn.slice instead of ttnn.bringup.rms_norm: the load-time bias rows cut to a row count other than the chunk's
  (once per count, kept; no norm there).
Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.22`; fork tests
`scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/moe_grouped_topk/tests/unit` and
`python -m models.demos.common.bringup.testing.fork_source --fork moe_grouped_topk`; router section A/B
`[DS41_ROUTER_IMPL=ops] BRINGUP_PROFILE_OPS=1 scripts/run_safe_pytest.sh --run-all --no-precompile
models/demos/common/bringup/tests/test_profile.py`.

## P.22 perf attempt 2 (2026-10-22): the F113 failure of attempt 1 fixed; no code path changed

What: attempt 1 failed only on the F113 lint: the ROUTER_IMPL help text in tt/settings.py named the stock
`ttnn.experimental.deepseek_prefill.moe_grouped_topk(...)`, though tt/router.py calls the fork
`ttnn.bringup.moe_grouped_topk` (DS41-107: the lint matches `<op>(` inside strings too). The text now names the fork.
Nothing else changed; `./build_metal.sh` re-run (no-op, the fork was built).
Checked: `ops_lint.violations` and `settings_lint.violations` on every changed file: none. Gate units with --check: all
16 component / swap units PASS on this code (ladder and profile left to the orchestrator, as the brief says).
`measure --task P.22` (switch on, quick): rc 0, all 31 err_* within limits (worst err_layer_L01 0.0338 / 0.0720),
min layer PCC 0.99676 (L03), pcc_chunk_out 0.99961, host_transfers_per_layer 0, chunk_wall_ms 99.2 (spread 14.2);
the same accuracy digits as attempt 1 (deterministic). Before / after numbers: attempt 1's section above.
Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.22`.

## O.1 optests attempt 1 (2026-10-23): a random-input case for every ttnn.bringup call

What: 56 distinct fork calls (ladder rung last, 2x2) now each have a case: fabric_all_gather 15, fabric_reduce_scatter
6, rms_norm 8, mhc_post 3, mhc_pre_xing 16 + mhc_pre_xing_pack 2, ring_indexer_score_dsa 3, moe_grouped_topk 1
(new suite: cases.py / reference.py / test_moe_grouped_topk.py: the previous gate failed "forks without
tests/test_*.py: ['moe_grouped_topk']" before running any test), flat_routed_expert 1 + flat_routed_expert_with_program
1 (new suite incl. tests/weighted_case.py). No op changed; CHANGELOG / README of each fork updated.
Decisions:
- K-45 / DS41-109: the 8 per-layer flat_routed_expert_with_program sigs were keyed on the `extra` descriptor's
  address; `ttnn/ttnn/bringup/__init__.py:_stable_arg_reprs` gives MeshProgramDescriptor a value repr (kernel counts per
  program) -> one stable sig 284a4a9ead. Re-captured once (63 -> 56 calls); every other sig unchanged.
- Limits: new math cases declare `noise` and check rel L2 <= fork_cases.noise_limit (F139); PCC printed only (asserting
  noise_limit's pcc failed 7 mhc_pre cases at 1 - 1e-15 with rel inside its limit). Where the op's real formats are
  coarser than its inputs, the formats run models them, never a measured device number: the FPU's operand mantissa
  bits per fidelity (DS41-108: in0 6 bits at LoFi / HiFi2, in1 4 at LoFi / 9 at HiFi2-4; fitted on probes, the device
  3e-4 from the model), bfp8 x / h tiles (round_to), fp32 running sums (mhc_pre pack / engram).
- Measured vs limits: reduce-scatter bf16 1.8e-3 / 3.6e-3 (exact case bit-identical); rms_norm 1.7e-3 / 3.3e-3,
  stats_out 2.8e-8 / 1.1e-7; mhc_post X' 1.66e-3 / 3.3e-3, HiFi2 projection 4.06e-3 / 8.2e-3, sum x'^2 5.9e-8 / 1.0e-7;
  mhc_pre worst ratio 0.73; router weights 1.7e-3 / 3.4e-3 (4096 / 4096 rows clear, all match); indexer (HiFi2) 7.7e-3
  / 1.55e-2; shared expert (LoFi bfp8) 4.0e-2 / 9.9e-2; routed + weighted combine (LoFi bfp4, bf16 sums) 2.0e-2 /
  2.95e-2. Negative checks by hand: a corrupted all-gather element and a halved block of routed rows both fail.
- The routed case ports the model's WeightedCombine (tt/weighted_combine.py) into the fork's tests (copied, not
  imported) on combine_rf's builders; experts at exact bfp4 values (check_exact), built in ~100 s.
Gotchas: fork_tests runs every top-level tests/test_*.py of the 8 forks (other models' cases too; 4x2 xing cases of
mhc_post / mhc_pre run on the 2x2 box per chip; flat's test_flat_bench passes, 13 s).
Checked (not the gate's own --check: its one unit, capture ~8 min + the 8 forks' suites ~12.5 min, exceeds a 10 min
foreground call): the gate's capture command by hand (56 calls), `fork_cases --capture ... --spec ... --lint`
(uncovered 0, lint 0), and `fork_tests --fork <f>` for all 8 forks on the final files, every model's cases: all pass
(rms_norm 18 + 2 skipped-size, mhc_pre 121, indexer 5, flat 2 + bench, all_gather 19, reduce_scatter 8, mhc_post 5,
moe_grouped_topk 1).
Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate O.1 --spec
models/demos/deepseek_v41_flash_d_p/bringup/spec.yaml --check`; one fork: `python -m
models.demos.common.bringup.testing.fork_tests --fork <fork>`.

## O.1 optests attempt 2 (2026-10-23): blocked, the remaining uncovered sig cannot be fixed in this step's paths

What: the gate (attempt 1's code) failed only on fork_calls_uncovered 1: flat_routed_expert_with_program sig
b155b17c4d (x8), while the case carries attempt 1's 284a4a9ead. Every fork test passed (fork_tests_failed 0).
Cause (DS41-110): the call's `extra_key` = `tt/weighted_combine.py:346` key0 = 9_100_000 + (id(self) % 100_000) * 8, a
Python object address that differs run to run. Checked: hashing the gate's entry with extra_key 9507296 instead of
9582688 gives exactly 284a4a9ead, so nothing else differs. DS41-109's descriptor repr does work (one sig for all 8
layers), but its "stable across captures" claim was wrong (attempt 1 captured only once).
Not done: no case for b155b17c4d. It would only match until the next capture, so it would be a guess, not a fix.
Fixing this needs a path this step may not change: a deterministic key0 in tt/weighted_combine.py (e.g. a class-level
itertools.count; the key only needs to be unique per WeightedCombine instance and pass), or a capture that normalizes
cache keys (testing/fork_capture.py). After that: capture twice, set the routed case's sig in
flat_routed_expert_ttnn/tests/cases.py to the stable value, and re-run the gate.
Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate O.1 --spec
models/demos/deepseek_v41_flash_d_p/bringup/spec.yaml --check`.

## O.1 optests attempt 4 (2026-10-23): stable routed sig, indexer test race fixed

What: (1) flat_routed_expert_ttnn/tests/cases.py routed case sig 284a4a9ead -> 3fc158f72b, the F179 stable sig
(capture leaves out extra_key, DS41-110 / OV-35). The captured entry re-hashed with attempt 1's extra_key gives
284a4a9ead, so it is the same call; no second case (it would rebuild the ~100 s routed case). (2) The orchestrator's
gate after attempt 3 also failed fork_tests_failed 1: test_ring_indexer_score_dsa_ds41 ratio-2 case "not
deterministic", A run 4, chip 3. Probe (indexer_score_ds41_det, deleted): 4 of 299 calls wrong on chip 3 with two
cycled semaphore sets (so not semaphore reuse), wrong on exactly the 13824 remote-shard columns. Cause (DS41-111): the
test wrote each call's fresh k buffer from the host just before the call; the peer done with call N starts N+1 and
writes its shard there while chip 3 (most causal work, slowest) still runs N, then chip 3's queued host write overwrites
it. Fix (test only): every call's buffer written before the first call (`kbufs`); 0 of 299 and 0 of 599. The model is
not affected: its buffer is written once at load (tt/indexer.py _kbuf). The hy4 / glm53 ring tests keep the per-call
write (other models' tests; latent, they pass). No op change.
Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate O.1 --spec
models/demos/deepseek_v41_flash_d_p/bringup/spec.yaml --check`.

## Z.1 settings audit attempt 1 (2026-10-23)

Audit result: tt/settings.py (Settings("DS41_"), 25 switches, env names unchanged) already held every environment read
of tt/ and bringup/hooks.py; settings_lint: 0 violations. Added `hooks.settings()` returning `settings.all()`.
Switches (default; before: earlier tasks moved them into the table, e.g. P.13/P.18/P.20/P.22): DEVICE_MODEL device,
MATMUL_FIDELITY HiFi2, FP32_ACC needed, DENSE_WEIGHTS, ATTN_FIDELITY, FP32_WEIGHT_ROUND, ATTN_O_SUM, ATTN_GLUE,
ROPE_IMPL, NORM_IMPL, GLUE_FOLD, STATIC_TABLES, BF16_ACTIVATIONS, ROUTER_IMPL, ROUTER_SOFTPLUS, INDEXER_GLUE,
EXPERTS_IMPL, EXPERTS_PRECISION, COMBINE_ACC, COMBINE_PLANNER, FFN_NORM_DTYPE, EXPERTS_SUM, RESIDUAL_MIX, MHC_FUSED,
HC_SINKHORN, CCL_IMPL, HC_TP_SUM (values: see tt/settings.py).
Left as literals (not switches): HiFi4 + fp32 DEST in tt/norm.py (the rms_norm fork's own exact setting) and tt/rope.py
(the only exact bf16 data move through the FPU); LoFi for EXPERTS_PRECISION=glm is derived from that setting.
Forks: mhc_pre_ttnn / rms_norm_ttnn env reads are marked `diagnostic`; flat_routed_expert_ttnn's ~60 MIMO_FL_* reads
are pre-existing fork knobs (F106, not converted here; the model's tuned values come from the fork defaults). Listed for the
owner: converting them to op arguments is a separate task.

## P.23 perf attempt 1 (2026-10-23): matmul program configs from a rule (MATMUL_CONFIG tuned / auto)

What: every dense ttnn.linear / ttnn.matmul of the model goes through `tt/mm_configs.py` (`mm.linear` / `mm.matmul`;
sites attn.wq_b / wo_a / wo_b, q_a, engram_kv, router, indexer.wq_b / weights_proj, compressor, hc.fn). Setting
MATMUL_CONFIG: auto = no program config (the old path), tuned (default) = `mm_configs.rule(M, K, N, dtypes, fp32 DEST,
grid, batch)`. Nothing else changed: dtypes, compute kernel configs (HiFi2, fp32 DEST, packer_l1_acc off), DRAM
interleaved memory configs, op sequence. The rule (K-51: auto is slow at fp32 DEST):
- batch > 1 (wo_a, one weight per group): auto. DS41-113: MatmulMultiCoreReuseProgramConfig needs per_core_N == N and
  rounds its K-block partials to bf16 (rel vs fp64 9.5e-3..2.8e-2 vs auto 4.4e-3); no other config takes per-batch
  weights.
- bf16 output and N > 4 tiles per grid column: ttnn.experimental.minimal_matmul, M4 K4 N8, subblock 2x2 (DS41-112).
- else 2D multicast (GLM-5.3's mm2d): M over grid rows, N over grid columns, in0_block_w 16/8/4/2/1 within
  L1_BUDGET 1.1 MB, the largest-area out subblock in DEST.
Sweep (tests/perf/test_matmul_sweep.py; one call per site, model's exact shapes / dtypes / ckc from an in-model capture,
L1 as the model leaves it: 1382 KiB free for CBs at every matmul call, DS41-114; stale-L1 check = a same-shape auto run
on other inputs right before each candidate; determinism over 2 runs and 4 chips; device time from the real-time
profiler, 20 calls; error vs float64 on the device-rounded inputs). 2x2, grid 11x10, ms per call:

| site (calls / chunk) | shape | auto | tuned | config | vs auto (rel) | vs fp64 (both) | ms / chunk |
|---|---|---|---|---|---|---|---|
| attn.wq_b (8) | 1024x1280 bf16 . 1280x8192 bfp8 -> bf16 | 0.6254 | 0.1533 | minimal M4 K4 N8 | 0 | 4.378e-3 | 5.00 -> 1.23 |
| attn.wo_b (8) | 1024x2048 bf16 . 2048x5120 bfp8 -> bf16 | 0.5928 | 0.1376 | minimal M4 K4 N8 | 3.4e-6 | 4.382e-3 | 4.74 -> 1.10 |
| engram_kv (2) | 1024x6144 bf16 . 6144x12800 bfp8 -> bf16 | 2.3117 | 0.7956 | minimal M4 K4 N8 | 1.1e-5 | 4.380e-3 | 4.62 -> 1.59 |
| attn.wo_a (8) | 2x1024x4096 bf16 . 2x4096x1024 bfp8 -> bf16 | 0.4761 | = | auto (DS41-113) | 0 | 4.382e-3 | 3.81 |
| q_a (8) | 1024x2560 bf16 . 2560x1792 bfp8 -> bf16 | 0.3072 | 0.0684 | minimal M4 K4 N8 | 0 | 4.381e-3 | 2.46 -> 0.55 |
| router (8) | 1024x5120 bf16 . 5120x96 bf16 -> fp32 | 0.0862 | 0.0692 | 2d 3x8 pm4 pn1 bw16 sb4x1 | 0 | 4.072e-3 | 0.69 -> 0.55 |
| indexer.wq_b (4) | 512x1280 bf16 . 1280x4096 bfp8 -> bf16 | 0.1573 | 0.0492 | minimal M4 K4 N8 | 0 | 4.386e-3 | 0.63 -> 0.20 |
| hc.fn (3) | 1024x10240 bf16 . 10240x32 fp32 -> fp32 | 0.1540 | 0.0993 | 2d 1x8 pm4 pn1 bw16 sb4x1 | 0 | 4.587e-3 | 0.46 -> 0.30 |
| compressor (2) | 1024x2560 bf16 . 2560x1024 bf16 -> bf16 | 0.1511 | 0.0452 | 2d 11x8 pm4 pn3 bw16 sb4x1 | 0 | 4.375e-3 | 0.30 -> 0.09 |

Best seen per site: within 2 % of the rule's pick everywhere except indexer.wq_b (minimal N16 0.0466) and wo_b (minimal
M16 0.1365); router 1D in1-mcast bw8 0.0632. Rejected: minimal for fp32 outputs (router vs fp64 4.33e-3 vs 4.07e-3);
every batched config (above). Explicit 2D on wo_b reached 0.1456 (bit-identical) vs minimal 0.137.
Other boxes (test_rule_boxes): every call at 640 / 512 rows (galaxy_8x4) with TP-4 widths (2560 -> 1280, 12800 ->
6400), on the 11x10 grid and an 8x8 sub-grid: the rule's config ran, deterministic, error vs fp64 = auto's, 1.6-3.3x
faster (e.g. wq_b 640 rows 0.264 -> 0.087; engram 0.733 -> 0.293); wo_a -> auto.
In the model: capture with tuned: chunk out vs golden rel 0.02803 (auto 0.02803). Quick A/B (measure, on): every layer
and state within its limit, worst L03 rel 0.08050 (P.22 0.08050), wall 97.4 ms. gate P.23 --check: PASS (8 swap units,
ladder last, profile), device_ms_total 69.66 (X.3 83.08): attention 30.27 -> 22.83, engram_kv 4.62 -> 1.59, q_a 3.09
-> 1.19, indexer 11.69 -> 11.24, compressor 1.33 -> 1.03, attn_hc 1.74 -> 1.58, router 1.04 -> 0.90; clean wall 98.6 ms
(X.3 120.3). Out of scope: the experts' fused matmuls; packer_l1_acc (GLM's other change) untouched.
Gotchas: `-k sweep` also selects test_capture (module name); MemoryView block addresses are base-relative (DS41-114);
minimal_matmul M8 K8 N16 (and on engram M8 K4 N16 / M16 K4 N8) refused: CBs beyond L1 (dataflow_buffer.cpp:2695) or clashing with the filler that stands for the model's L1 buffers (:2704); the rule's M4 K4 N8 needs ~330 KB.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
models/demos/deepseek_v41_flash_d_p/tests/perf/test_matmul_sweep.py -k "test_capture"` (DS41_MATMUL_CONFIG=auto writes
calls_auto.json), then `-k "test_sweep or test_rule_boxes"`; gate: `PYTHONPATH=$PWD python -m
models.demos.common.bringup gate P.23 --spec models/demos/deepseek_v41_flash_d_p/bringup/spec.yaml --check`.

## P.24 perf attempt 1 (2026-10-23): batched attention wo_a on a tuned 2D config (WO_A_CONFIG tuned / auto)

What: `tt/mm_configs.py:rule` gives the batched matmul with one weight per group (attn.wo_a, [1, 2, 1024, 4096] bf16 .
[1, 2, 4096, 1024] bfp8 -> bf16, HiFi2, fp32 DEST, 8 calls a chunk) a MatmulMultiCoreReuseMultiCastProgramConfig with
fuse_batch=False (auto's own family: each core loops the batches, fp32 partials between K blocks) at the widest in0
block whose CBs fit L1_BUDGET, 32 tiles of K included (batched calls only; auto picks 1, K-51). On 2x2: grid 11x8,
pm4 pn3, bw32, subblock 4x1, ~790 KB of CBs. The minimal_matmul branch is now batch == 1 only; a broadcast over
per-batch weights (x batch != w batch) stays auto. Started from the comparison agent's patch (reports/pr54943/
p24_mm_configs.patch, applied as is). New setting WO_A_CONFIG {tuned (default), auto}: auto = no config for the
batched call (the path before P.24), checked in `program_config` before its cache; tuned only acts with
MATMUL_CONFIG=tuned. Nothing else changed (dtypes, fidelity, ckc, memory configs, other sites).
DS41-113 (never MatmulMultiCoreReuseProgramConfig: bf16 K-block partials) held and is updated: the safe explicit
config exists, it is auto's family.
Sweep (tests/perf/test_matmul_sweep.py, extended: batched calls also get the 2D fuse_batch=False family, in0 blocks
1..32, both mcast orientations; minimal_matmul now tries every out subblock of each blocking), P23_SITES=attn.wo_a,
2x2, 20 calls, real-time profiler, L1 as in the model (1381 KiB for CBs):
| config | ms / call | vs auto (rel) | vs fp64 |
|---|---|---|---|
| auto | 0.4769 | - | 4.381e-3 |
| rule (2d nofuse 11x8 pm4 pn3 bw32 sb4x1) | 0.1160 | 0 (max abs 0) | 4.381e-3 |
| 2d nofuse bw16 / bw8 | 0.1229 / 0.1465 | 0 | 4.381e-3 |
| bmm (MatmulMultiCoreReuseProgramConfig) bw1 / 2 / 4 | rejected | - | 2.81e-2 / 1.56e-2 / 9.47e-3 |
wo_a per chunk 3.816 -> 0.928 ms. Other boxes (test_rule_boxes, same run): wo_a at 640 rows 0.2564 -> 0.0674 ms (11x10),
0.2569 -> 0.1078 (8x8 grid); 512 rows 0.2564 -> 0.0669 / 0.2557 -> 0.0783; every one bit-identical to auto, det 1;
every other site's rule unchanged and ok. In-model capture: chunk out vs golden rel 0.02803 (= P.23).
Quick A/B (measure --task P.24, on): 31 err_* within limits, min layer PCC 0.99676 (L03), host_transfers_per_layer 0,
wall 102.9 ms. gate P.24 --check: all 10 units PASS; the 8 swap units' metrics equal P.23's to the last digit (e.g.
swa_15 err_swap_out 0.026917). Profile (plain, on): device_ms_total 69.69 (P.23) -> 66.77 ms (limit < 69.7),
attention 22.85 -> 20.00, every other section within 0.03; chips 67.04-68.05 -> 64.07-65.22. Clean wall 111.1 ms
(P.23 run 98.6; this run's spread 27.1 ms vs 8.5: wall noise, the device time is the measure).
Gotchas: the background-moved gate --check was cut at the tool's 20-min limit after 7 units; per-unit
`--check --unit 'P.24#<name>'` (the full unit id) runs in the foreground. `-k sweep` still selects test_capture and
test_rule_boxes (module name).
Re-run: `P23_SITES=attn.wo_a PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
models/demos/deepseek_v41_flash_d_p/tests/perf/test_matmul_sweep.py -k "test_sweep"` (needs generated/ds41_p23/
calls_auto.json from `DS41_MATMUL_CONFIG=auto ... -k test_capture`); A/B: `python -m models.demos.common.bringup.
orchestrator measure --task P.24`; gate: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate P.24 --spec
models/demos/deepseek_v41_flash_d_p/bringup/spec.yaml --check`.

## W.1 port (galaxy_8x4-full, run1): the seed's layers on 8x4 at chunk 5120

What: the seed's 8 layers [0, 1, 2, 3, 14, 20, 21, 24] on mesh 8x4 (SP 8, TP 4: 640 rows a mesh row, 160 query rows a
chip, hidden share 1280), each run from its golden input. Four 8x4-only failures, each fixed in place for every box:
1. Shared expert (DS41-115): TP 4's slice 576 -> 768 (24 tile columns) lands in the flat op's 2 / 3-subgrid plans,
   which have no gate/up split at H 5120 (TT_FATAL at load). tt/shared_expert.py pads each part to >= 33 tile columns
   (ONE_SUBGRID_MIN), the one-subgrid plan the seed runs: 576 -> 1280 (2.2x shared-expert work on TP 4; 2x2 unchanged
   at 1152 -> 1280). Better: a one-subgrid / HG option in the flat fork for It 24 (48 units on 48 cores), a perf pick.
2. Served compressed caches (DS41-116): TP-dedup ratio-2 stripes are 640 / 2 / 4 = 80 rows a chip and the cache at
   56320 is 880 rows; stock update_padded_kv_cache checks both in 32-row units. New fork ttnn.bringup.update_padded_kv_cache
   (validation only, ROW_MAJOR copy path in rows; kernels untouched); tt/compressor.py and tt/attention.py call it.
   Fork test tests/unit/test_row_stripes.py (8x4, exact vs torch, scalar + metadata + clamp; mutation of the reference
   row fails it). The source op's tests do not collect on this tree (deepseek_v3_d_p tests conftest imports xing40's
   settings, which pass the removed max_precision): baseline recorded as that error, see the fork's CHANGELOG.
3. Indexer score (DS41-117): Q_CHUNK 64 does not divide 160 rows; tt/indexer.py takes the largest multiple of 32 <= 64
   dividing the chip's rows (32 on 8x4, 64 on 2x2).
4. Combine column barrier (DS41-118): tt/weighted_combine.py _column_barrier padded the column-end chips (1 fabric
   direction on > 2 rows) with a duplicate connection to the same router channel; intermittent hang on rows 0 / 7.
   Each chip now opens one connection per direction it has. Probe (8x4, add + barrier loop): old hangs < 50 calls,
   fixed 2000 / 2000 (probe script not kept: tt-probe saves under tests/, outside this step's paths).
Not changed (owner decisions / perf picks per the brief): settings defaults; mm_configs, link counts (indexer
num_links 2), core grids; the shared expert's 2.2x padding and the score's q chunk 32 are Galaxy perf-pick inputs.
Result: gate W.1 PASS. err_layer (limit): L00 0.0249 (0.054), L01 0.0368 (0.067), L02 0.0516 (0.090), L03 0.0711
(0.107), L14 0.0320 (0.055), L20 0.0300 (0.037), L21 0.0436 (0.074), L24 0.0312 (0.0336, the tightest); err_state_max
0.039; host_transfers_per_layer 0; load 1057 s, 2 chunks 91.6 s.
OV-1: sparse_sdpa (160 query rows a chip after the reshard), the ring indexer (W 640, 160 rows; fix 3) and the served
records (160-token sub-blocks, 80 / 40 compressed rows; fix 2) all run at 8x4's sizes in this gate, as do the
4-column TP paths. Every row count comes from mesh.shape and the chunk.
Gotchas: the gate --check takes ~20 min (8 layers load ~2.2 min each); a background tool call is cut at 10 min unless
given a longer timeout. The units log (runs/.../logs/W.1.units.log) appends across runs.
Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate W.1 --spec
models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`; fork test:
`scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/update_padded_kv_cache/tests/unit/test_row_stripes.py`.

## W.1 port, attempt 2 (2026-10-26)
Attempt 1 passed the gate but its diff included ledgers/galaxy_8x4-full/supervision.md. Those lines are the overseer's
(the W.1 pre-check review and the session hand-off notes), not the step's, and the orchestrator already committed them in
9582c9da53d. No code change in this attempt, and supervision.md was left as it is in HEAD: reverting it would delete the
overseer's notes and add a new out-of-scope change. Gotcha: `timeout 600` around the ~20-min gate kills it; run the gate
in the background with a timeout over 25 min. A SIGINT to the ladder's pytest during weight load takes effect only once
the load returns to Python.
The overseer appended another supervision.md line at 13:24:03 (after this brief). I restored the file to HEAD so the
step's scope check passes; the line is kept word for word in results/knowledge_use/W.1.port.2.json for the overseer.

## FIX.1 overseer, attempt 2 (2026-10-26): no padding around the flat op's limits (owner order, OV-3)
What: tt/shared_expert.py runs the chip's real intermediate slice (576 on TP 4: 8x4 / 4x4 / 2x4; 1152 on TP 2: 2x2)
as the one always-on expert: DOWN_K, ONE_SUBGRID_MIN and the zero-padding (576 -> 1280 = 2.2x, 1152 -> 1280) are
gone. The fork was extended instead (ttnn/ttnn/bringup/flat_routed_expert_ttnn, CHANGELOG "Intermediate widths with no
subgrid split"):
- flat_routed_expert_plan.cpp: when the 2 / 3 subgrids picked for It <= 32 have no gate/up split (H 5120: DS41-115),
  plan one subgrid; the uneven pair split there takes It 18 (np 1, 18 gate/up cores, 30 down, reader tails) and It 36
  (np 1, 36 cores). Plans that worked before are unchanged (the check only replaces the old TT_FATAL).
- se6_dcompute.cpp: the static_assert's `kt_d % 8` term removed. DS41-76 was wrong: hk = 8 is only matmul_block's
  kt_dim with rt_dim 1, unused by the LLK (llk_unpack_AB_matmul.h, reuse_a), so a down K of 18 / 36 tiles is fine.
- No option: both changes only reach inputs that failed before.
Measured:
- Fork tests/unit/test_flat_widths.py (new; one Galaxy card, H 5120, vs float64 within noise_limit, deterministic):
  I 576 glm rel 0.0400 (limit 0.099), full 0.0104 (0.021), 3 experts 0.0398 / 0.0401; I 1152 glm 0.0400, full
  0.0105. Corrupted h (MIMO_FL_NO_ACT=1) fails it: rel 1.475.
- Shared expert on 8x4 after the change (tt/perf_shared_expert.py::test_shared_flat, layer 0, chunk 5120, 640 rows a
  chip, EXPERTS_PRECISION=glm; one run, no device profiler, per OV-4): 0.153 ms a layer on the model's path (rows:
  0.484 minus the test's 0.331 ms rows clone), 0.513 ms with its own x conversion; MoE block 4.662 ms (3.922 without
  the shared expert and moe_add). vs float64: rel 0.0213, bias +1.43 % (P.12: 0.020-0.022); twice bit-identical. No
  before number on 8x4: OV-4 allowed one timing run, after the change. The padded op ran 2.2x the weight columns.
- Gate FIX.1 --check (W.1 unit, 17.5 min): passed, every layer's shared expert "parts=1 x 576 rows=640", np 1, 18
  gate/up cores. err_layer_L00..L24 and every err_state equal to W.1's to the last digit (L24 0.031249 <= 0.033573;
  host_transfers_per_layer 0): the padded lanes only added exact zeros at the end of the down's sequential fp32 K sum.
Gotchas:
- On a single Galaxy card (the `device` fixture) the flat op hangs on the first call for most shapes (the I 2304 / glm /
  k3 / mimo unit cases); glm_e12 hangs on the code before FIX.1 too; 2x2-mesh tests on the same box pass (DS41-119).
  test_flat_widths skips I 2304 on a 12-column grid.
- The fork's model case tests/cases.py "deepseek_v41_flash_d_p-2x2-shared-expert-s1024-i1280" is the old padded call
  (I 1280); the model now calls I 1152 on 2x2 and I 576 on 8x4. A later O.1 should add those cases (append only).
- The fork regression on one Galaxy card is not comparable to the 2x2 QuietBox baseline (CHANGELOG lists each failure).
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --device auto
ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/test_flat_widths.py`; timing `scripts/run_safe_pytest.sh --run-all
--no-precompile models/demos/deepseek_v41_flash_d_p/tt/perf_shared_expert.py -k "test_shared_flat and time and L00"`;
gate `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.1 --spec
models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.

## FIX.2 overseer, attempt 1 (2026-10-26): the flat op's Galaxy hangs root-caused (DS41-119)
What: the fork ttnn/ttnn/bringup/flat_routed_expert_ttnn only (no model code changed; CHANGELOG "The schedule leader
found by its logical coordinate").
- Cause (tt-triage, test_flat_uneven_split[glm_e12] on one Galaxy chip, 5 s dispatch timeout): with SE_DYN_PULL (plans
  with no reader tails and no combine attached) reader 0's BRISC reads the counts and sets semaphore SE_DYN_LSEM; every
  other core polls it. The leader recognised itself by `my_x[0] == SE_DYN_LPX` with LPX = reader 0's physical NoC0 x
  (flat_worker_noc0), but BH firmware fills my_x from NOC_ID_LOGICAL (translated). On this card physical column 1 is
  harvested (triage: logical (0,5) at physical 2-7), so reader 0 (0,9) is physical (2,11) vs translated (1,11): no
  leader, every core polling (36.9 M completed reads, one in flight). The QuietBox p150s have column 1, so it never hung there.
- Bisection probes (single chip, glm shape, 64 tokens an expert): MIMO_FL_DRISC=0 hung, MIMO_FL_COLS=11 hung, bfp8
  weights / bf16 x and h hung, MIMO_FL_DYN_PULL=0 passed. The same shape hung on the full 8x4 FABRIC_2D mesh as well
  (DS41-119's "mesh runs pass" was wrong; the mesh cases that passed had no SE_DYN_PULL). After the fix: single chip
  and 8x4 mesh pass.
- Fix: se_dyn.hpp compares get_absolute_logical_x / _y with SE_DYN_LLX / LLY (reader 0's logical core), set in the
  program factory. The model's calls never take SE_DYN_PULL (routed: flat_routed_expert_with_program + combine;
  shared expert I 576 / 1152: reader tails), so W.1's numbers should not move.
- Builder asserts: flat_expert.py `_layout` (Python parity builder) asserted rcols [0, 1, 6, 7] and an 11 x 10 grid
  (the 4 test_flat_routed_expert_op failures). Now the C++ plan's rule (readers 0, 1, E, E + 1; east part moved by
  E - 6; extra columns are down cores). test_flat_widths' I 2304 skip on a 12-column grid removed.
- Fork tests/unit/ on one Galaxy chip: 24 passed / 14 failed / 3 skipped (2x2-mesh tests skip on one card), no hangs
  (FIX.1: 11 hangs). 12 x TT_FATAL p.nsg == 1 = the QuietBox baseline's 12 (DS41-63). Not identical to the baseline:
  the mimo and k2 indexed_perf bands (QuietBox p150 timings): mimo 1843456 / 1838520 ns vs upper 1826488, k2
  1573274 / 1575501 vs 1572413 (k2 has reader tails, so the change does not reach it; FIX.1 had 1573083). Not
  adjusted: a Galaxy band would special-case the box; the owner's 2026-10-23 rule puts Galaxy re-tuning in a perf pick.
- Not run on the 2x2 QuietBox (not on this box). Where the old check found the leader, the new one selects the same core.
Gotchas: tt-probe.sh saves probes under tests/ttnn/unit_tests/operations/<name>/probes/ (outside the step's paths:
deleted). A 2x2 sub-mesh of the Galaxy fails the fabric handshake; use the 8x4 mesh and touch every
/tmp/tt-device-N.dirty after single-card runs (K-57).
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --device auto --run-all
ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/`; gate `PYTHONPATH=$PWD python -m models.demos.common.bringup
gate FIX.2 --spec models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.

## FIX.3 (overseer, attempt 2): real-time profiler off on chips 7 and 10 (DS41-120)

- Attempt 1 left nothing in the tree (its commit 5b1a3ec2106 passed on reused units with no change).
- Cause, measured: 30 TT cards are in a DMA-FQ IOMMU domain, `/dev/tenstorrent/7` (08:00.0) and `/10` (43:00.0) in
  `identity` (iommu=pt host; tt-iommu-domains.service converts the cards once at boot; these two were re-enumerated
  after it ran, 2026-10-22 10:34 and 2026-10-25 16:56). Metal caches chip 0's IOMMU state -> pinned-memory D2H for all;
  UMD pins 7 / 10 with CONTIGUOUS (0x4) -> EINVAL on the 2 MB + 4 KB anonymous mmap. Hugepages are not involved.
- Probe (8x4 FABRIC_2D, one add per chip, RT callback): records from 30 / 32 chips, missing [7, 10];
  IsProgramRealtimeProfilerActive() True. Saved outside the repo: runs/galaxy_8x4-full/run1/probes/FIX.3_rt_profiler_check.py.
- Not fixed at the cause: it needs root (`sudo systemctl restart tt-iommu-domains`; sudo asks a password here) and
  tt_metal (per-chip IOMMU state) is outside this step's paths. Owner / host action.
- (Reverted by the overseer, 17:35: the check moved to the framework, F194; nothing below is in the model.) Added: tt/rt_profiler_check.py; hooks.device_params logs the host mismatch (sysfs) before every mesh open;
  hooks.device_model runs the device check at build (one 32x32 add, <= 3 s settle when chips are missing) and warns
  (Setting RT_PROFILER_CHECK warn / raise / off; DS41_RT_PROFILER_CHECK=raise for a perf run that must have all chips).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.3 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.

## FIX.2 (overseer, reopened, attempt 2): flat fork tests/unit/ all pass on a Galaxy chip

- (a) 12 x TT_FATAL `p.nsg == 1` (TP4 / TP2 subgrid cases, since the import; DS41-63): the op was wrong, not the
  tests. Upstream c485d15bbae moved the plain program's schedule defines (SE_RPS / SE_MAX_E / SE_GU_NREG / pin) into
  `flat_schedule_defines` so flat_combine_overlap's combine could rebuild the schedule, and put the overlap's
  one-subgrid assert inside it; `FlatRoutedExpertProgramFactory::create` calls it for every plan, so every subgrid plan
  the planner builds died at program build. Fix: the assert now sits at flat_combine_overlap's call site
  (device/flat_combine_overlap_device_operation.cpp); flat_schedule_defines returns what create set inline before
  c485d15bbae, for any nsg. The model's calls are one-subgrid (DS41-115): same defines, same programs.
- (b) Perf bands: `_PERF_EXPECTED_NS` (tests/unit/test_flat_expert_indexed_shapes.py) keyed by chip class
  `(arch, grid x, grid y)`: the QuietBox p150 table as BLACKHOLE 11 x 10; a BLACKHOLE 12 x 10 table measured on this
  Galaxy chip (RT-CAL, median of 3, 2026-10-26): mimo flat 1834370 (11 x 10: 1773289, +3.4 %), k2 indexed_rm 1578919
  (+3.4 %), tp4 flat 928769 (+0.6 %), tp2 flat 1452958 (+4.3 %), tp2 indexed_rm 1706179 (+17 % over its flat; equal on
  11 x 10, worth a look in a perf pick). Margin stays 3 %. The 11 x 10 mimo flat band failed fast on the seed QuietBox
  (DS41-63, 1710341): recalibrate that class there.
- Result (one Galaxy chip, tests/unit/, --run-all): before the 12 x 10 table 34 passed / 0 failed / 7 skipped (4 perf
  "no baseline", 3 need a 2x2 mesh: not applicable on one card); with it 38 passed / 0 failed / 3 skipped (the 2x2-mesh
  tests only); a second run sits within -0.9 .. +0.6 % of every 12 x 10 baseline (tp2 indexed_rm again 1705894).
  Full gate: W.1 PASS (all err_layer / err_state within limits, host_transfers_per_layer 0), FIX.2 PASS.
- Fork docs: CHANGELOG.md entry, README.md Limits, INDEX.md entry; DS41-63 rewritten.
Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.2 --spec
models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.

## FIX.4 (overseer): routed-expert weights through the flat fork's laid-out cache, faster miss path

- What: `tt/experts.py` passes FlatRoutedExpert `cache_prefix=` and a weights callable (`_host_weights`, called by the
  fork only on a miss). Prefix `expert_cache_prefix`: `<art>/tt_cache/<mesh>[-scaled]/<cache_version>/experts-<hf
  revision[:12]>/<rows>x<cols>/L<layer>.<EXPERTS_IMPL>.<sha1 of the layer's stored expert tensors (file, dtype, shape,
  offset, bytes): real vs scaled differ>`; the fork's names add `.flat.bf4.lay<plan layout hash>.{w_gu,w_d}.tensorbin`.
  `_CacheMarker` (`<prefix>.done.json`): written after the fork dumped, lists the files with their sizes and holds
  `bfp4_stats` (so a hit logs the same EXPERTS_BFP4 line, `(cached)`). The fork dumps straight to the final names (no
  tmp + rename), so every `<prefix>.flat.*` file not in the marker with its size is deleted before the fork looks
  (a run killed mid-dump repacks instead of loading a truncated file; DS41-121). No fork change.
- Wiring: Setting `EXPERTS_WEIGHT_CACHE` on / off (default on). `expert_cache_dir(spec)` from hooks (`device_model`
  -> `TtDS41Model(weight_cache=)`, `_device_step_fn` experts for component / hybrid), the adapter's
  `weight_cache_path()` (was None) -> `DS41PrefillRuntime` -> `TtDS41Model(weight_cache=params.weight_cache_path)`.
  perf_experts.py's tests build without a cache (unchanged).
- Miss path (`_host_weights`): each slot's 32-chip stack read with 8 threads, dequantized in one fp4_dequant call
  (the same per-element arithmetic), one bf16-exactness check per stacked tensor (low 16 bits of the fp32 words zero
  + aminmax finite; was torch.equal per matrix), bfp8 `_bfp8_exact` / bfp4 `_bfp4_change` on the fp32 stacks (no
  .float() / slice copies); `_bfp4_round` and `_bfp8_exact` take max |v| as max(max, -min) (no |g| copy). Values,
  order and layout unchanged: `tt/perf_experts_load.py::test_host_weights_identical` (8x4, layer 0, all 384 experts):
  every (W1, W3, W2) bf16 tensor torch.equal to the old per-expert loop's, same chip / slot order, same changed count
  2262393486 / 13589544960, same worst matrix (122, w3) 0.11965. Batched vs per-matrix dequant measured equal
  (synthetic 32 x 2304 x 5120: 0.67 vs 0.70 s a stack); kept batched.
- Numbers (8x4, layer 0, bfp4, nice 19): old miss 118.9 s (cProfile: from_torch bfp4 packing 54.8 s,
  flat_weight_regions 22.4 s, _bfp4_change 20.2 s, _load_block loop 7.7 s, fp4_dequant 2.9 s); host part old 49-55 s
  -> new 31.7 s (same process, same page cache); new off (no cache) 127.1 s under cProfile, miss (packs + dumps)
  119.9 s (host_weights 24.4 s), **hit 1.0 s** (warm page cache; device tensors of devices 0 / 31 torch.equal to the
  miss's, bfp4_stats equal). 7.12 GiB of disk per layer (bfp4; 40 layers ~285 GiB; /scratch had 2.2 TB free). W.1
  gate run 1 (seed layers, L00 cached by the perf test): L01..L24 misses 116-130 s each, L00 hit 1.2 s; PASS, every
  err metric equal to the previous W.1's to the last digit; model_load_s 1057.3 s (old code, 8 misses) -> 887.7 s.
  `gate --check --rerun W.1` did not re-run it ("reused: same code, not run", F150), so the hit side was measured by
  `perf_experts_load.py::test_cache_hits` (the seed rung's 8 layers, as the ladder builds them): all hits, 1.0-1.23 s
  each, 8.3 s total (vs ~950 s of expert load on a miss): a W.1 with every layer hit loads in ~70 s (887.7 - 7 x
  ~117 + 7 x 1). The final gate check (after that test was added under tt/) re-ran W.1: all 8 layers hit (1.0-1.2
  s), PASS, metrics identical, **model_load_s 65.6 s** (was 1057.3 s), W.1 command 119 s (run 1: 956 s).
- Left: the miss is now ~75 % the fork's host packing (ttnn.from_torch bfp4 55 s, flat_weight_regions' 18880
  torch.cat 17 s); a fork-side change (parallel per-device packing, vectorised region building) is the next cut, not
  done here (needs the fork's regression). Dense weights (DENSE_BFP8, ~5 s / layer): not cached (not cheap: no
  cache path in the dense modules; the expert cache removes ~95 % of the load).
- Profile / check: `PYTHONPATH=$PWD nice -n 19 scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_experts_load.py -s`.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.4 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check` (`--rerun W.1` for the hit run).

## FIX.5 (overseer, attempt 1, 2026-10-27): flat_routed_expert_with_program keys its program without addresses

- Cause (confirmed): the op hashed every baked buffer address (`FlatWithProgramParams.addrs`, `counts_addr`) and its
  `override_runtime_arguments` was empty; of those addresses only the gathered x (`xg`) moves call to call (overseer
  keylog), so each new xg address rebuilt a 32-chip program (~0.55 s host). DS41-122.
- Fix IN THE OP, behind a new option (rule 6; default = the op before): `patch_addresses=True` + `extra_addrs`.
  Hash: plan key, extra_key, writers, collector, l1_args sizes, every tensor's spec (optional ones by presence), no
  address. A hit re-applies what moved since the cached program last ran: per buffer (x, y, counts, regions, ids,
  gate/up, down, reader-down, done, arena, words, token_index) the flat program's address args (its existing AddrPatch
  list, grouped by buffer, RuntimeArgsData pointers cached at the miss), the flat + appended arena CBs when the arena
  moved, the y writers' counts_addr / L1 hand-off args when they changed, and the combine's runtime args copied from
  `extra` when `extra_addrs` (arena, words, y, out, the planner's plan / table / dord addresses) change. On a miss the
  combine's arena CBs are bound to the call's arena by the op (the cached descriptor may name a freed arena tensor at
  the same address; the old `seen` / `rebind` dance is not needed). `extra` is now a non-owning view (no per-call deep
  copy). Weights: their addresses are patched like everything else (nothing is baked), so the 40 layers share one
  program per pass (the old key also split them by weight address).
- Model: Setting `EXPERTS_PROGRAM_KEY` patch (default) / addresses (the op before, for comparison);
  `tt/weighted_combine.py` passes the cached descriptor on every call with `extra_addrs`; the addresses mode keeps
  the old `seen` logic. EXPERTS_XG_PINNED (the overseer's hack, default off) is moot with patch; left in place.
- Numbers, 8x4 (`tt/perf_flat_program_key.py`, layer 0, chunk 5120, real weights, 6 calls a phase): op host per call,
  hit: addresses 0.41 ms; patch 0.44 ms (same buffers) / 0.46 ms (x moved every call); miss ~0.55 s. Moving x:
  addresses +1 program every call, patch +0. All 24 outputs bit-identical to the address-keyed op's first call. A
  first version that re-applied every patch on every hit cost 2.0 ms a hit (override 1.51 ms for 26496 arg / CB writes
  over 32 chips): now only what moved is written.
- F196 warm wall (`BRINGUP_PROFILE_WALL=1 test_profile.py`; the profile's rung is `last`, i.e. 40 layers, one 5120
  chunk at position 0, whatever BRINGUP_RUNG says): patch: warm after 1 warm-up run (237 -> 237 entries), 10 timed
  runs with misses refused all completed, chunk_wall_ms median 999.8 (755-1175). addresses (`DS41_EXPERTS_PROGRAM_KEY=
  addresses`): warm-up +40 +36 +37 +38 +23 +34 +18 +2 +0 (316 -> 504 entries, 17-30 s a run), then the first timed
  run missed in FlatWithProgramDeviceOperation and was refused: never warm. Logs /scratch/dnijemcevic/fix5/.
- Fork tests: new `tests/unit/test_with_program_patch.py` (the model's wiring via tests/weighted_case.py, model shape:
  384 experts / 12 a chip / 640 tokens a chip / top 6 on 8x4): BRINGUP_TEST_MESH=8x4 PASS (bit-identical vs the
  address-keyed op; moved x +0 programs; moved arena, i.e. extra_addrs changed and the combine's args copied, +0
  programs, bit-identical; arena back, bit-identical). It first hung on 8x4 in the unchanged address-keyed call: the
  fork's weighted_case.py never got the model's DS41-118 barrier fix (ported now, DS41-123). `tests/unit/` on one
  Galaxy chip (`--device auto`, as FIX.2): 38 passed, 4 skipped (the 2x2-mesh tests; the new one defaults to 2x2).
  A 2x2 mesh cannot open on this Galaxy (fabric router sync timeout).
- Gotcha: run_safe_pytest without `--no-precompile` runs the precompile collect pass; use the flag for mesh tests.
- Left: the fork-call capture (O.1) keys sigs on argument values; `extra_addrs` holds addresses (allocator-
  deterministic in practice) and the routed call now always carries the full descriptor, so its sig changes once;
  `fork_capture.VOLATILE_KWARGS` (not this step's path) could list `extra_addrs` like `extra_key` (K-45).
  The 2x2 model case in tests/cases.py still runs the address-keyed default.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.5 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`; measurement:
  `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_flat_program_key.py -s`; fork: `BRINGUP_TEST_MESH=8x4
  scripts/run_safe_pytest.sh --run-all --no-precompile
  ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/test_with_program_patch.py`.

## K.1 contract attempt 3 (galaxy_8x4-full)
- OV-12: `tt/runners/kv_contract.golden_rows` returns (rows, first row, mode). The window KV of a layer with per-chunk
  golden dumps (all chunks present: step_dump_layers 0,1,2,3,14,20,21,24) is compared over [0, length) ("full"); any
  other layer over the last min(128, length) positions from `golden.state(layer, at=length)["window_kv"]`, unrolled by
  position (ring row p % 128) ("last-window"). Ring layout checked on CPU: layer 0's ring at 10208 and at 10240 equal
  the per-chunk 'kv' dumps exactly (max diff 0). Compressed KV / index keys unchanged (final state, full). No layer is
  skipped; the per-layer mode is logged and summarized (`KV comparison per config and kind`). Test-side read-back
  only; the model path reads no golden.
- Gate units: test_contract PASS (window_kv 0.9762 / limit 0.9434, compress_kv 0.9839 / 0.9370, index_k 0.9914 / 0.9465;
  acks_early 0; contract_checks_failed 0); test_kv_layout PASS.
- Blocker outside K.1's paths (DS41-124): the frozen tests test_adapter_acks / test_runner_contract build
  `caches.GoldenKV()`, which needs ladder rung `s4096` (server_rules.GOLDEN_RUNG, CHUNK 2048 / MAX_SEQ 4096). The 8x4
  spec has no such rung and golden_full/galaxy_8x4 no s4096 golden: CPU probe `C.GoldenKV()` -> SpecError. With the
  rung, GoldenKV._boundary would still need per-chunk 'kv' / 'in' for all 40 layers. Needs a spec rung + golden, or a
  test change (overseer / owner). test_runner_smoke (no GoldenKV) not run by me: left to the orchestrator's gate.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate K.1 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check --unit 'K.1#test_contract'`.

## K.1 contract attempt 5 (galaxy_8x4-full)
- OV-15: `tests/bringup/contract/server_rules.golden_geometry()` derives (GOLDEN_RUNG, CHUNK, MAX_SEQ) from the running
  spec: the shortest ladder rung (by seq) with its own golden and `full_dumps`. Seed spec.yaml -> s4096 / 2048 / 4096,
  galaxy_8x4 -> s10240 / 5120 / 10240 (CPU check, Spec.load, no device). Scenario / runner / ack lengths are written
  as the seed's offsets from H = CHUNK / 2, CHUNK and MAX_SEQ: the seed gets exactly its old plans (1000 / 3000 /
  3017 / 1330 / 3001 / 4000 / 1500 / 3500); 8x4 gets 2536 / 6072 / 6089 / 2866 / 6073 / 10144 / 3036 / 6572.
  self_check asserts the same shapes symbolically (plus the ACK_TURNS interleave). test_runner_contract T1 / B come
  from RUNNER_TURNS.
- Golden gap (OV-12 rule, test side): `caches.GoldenKV.select` / `caches.golden_failures`. Window KV of a dumped layer
  (all chunks: 0,1,2,3,14,20,21,24 on 8x4; every served layer on the seed) and every compressed / index cache: every
  row ("full", unchanged). Other layers: the positions the golden's ring snapshots hold ("last-window": [4992, 5120)
  and [10080, 10240) on 8x4), the final ring first: the 10208 snapshot is a separate reference run, fp32 noise apart
  (max 7.7e-4 of 2.7 at layer 39, not bit-equal). `coverage_failures` fails any served (config, layer) never compared;
  each test prints the per-config, per-mode layer list. 8x4 coverage per test: adapter acks / runner contract / cache
  starts: window_kv layers 0-3,14,20,21,24 full, the other 32 last-window (128 rows in [0, 6072), 288 in [0, 10240));
  compress / index (2, 8, 14, 20) full. The ack-time checks of [0, 2536) compare no rows of a non-dumped layer's
  window KV (none in the golden there); the slot-final checks do.
- `parts.run_chunk`: a layer with no golden block input (neither its "in" nor the previous layer's "out", F164) runs
  from the previous layer's device output of the same chunk (test_cache_starts on a full 8x4 run).
- Paths: OV-15 orders `tests/bringup/contract` changed; the brief's contract-role allowed list does not include it.
- test_adapter_acks on 8x4 (unit run): all 4 chunks give 40 acks in order, coverage as above (dumped layers 24328 rows,
  others 512 last-window rows), 6 failures, all window_kv "nope" PCC of the dumped deep layers vs the frozen
  per-layer state limit: L20 0.981 / 0.9896 ([0, 2536)), 0.9873 ([0, 5120)), 0.9888 ([0, 6072)); L21 0.985 / 0.9886;
  L24 0.980 / 0.9837. DS41-125: the limit comes from the ring drift (last 128 positions); L20's KV is L19's output,
  which the ladder measures at err 0.159 over all positions. Limit unchanged; left to the owner.
- test_runner_smoke on 8x4 (unit run, feeder tt-d-gen engine /scratch/dnijemcevic/tt-d-gen @ a9701ce6675): the runner
  came up (40 layers, table and device map exported, request loop entered); the engine driver died before any chunk:
  ENGINE_FATAL chunk_size=5120 not divisible by sp_factor=1800000 (DS41-126: testing/dgen_prefill_driver.py:79 passes
  layers_per_chunk positionally; tt-d-gen 1cf343887 removed it). Framework file, outside K.1's paths: not changed.
  test_runner_contract uses the same driver; not run.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate K.1 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.

## K.1 decode hang: the stand-in decode's mesh had the default fabric payload (fix role, overseer-directed, 2026-10-27, bh-glx-110)
- Symptom (K.1 run 2 on b06u08, the 1-layer repro here): the first MoE call (flat_routed_expert_with_program + the
  weighted combine) hangs on all 32 chips: se3 / se6 compute in matmul, readers in cb_reserve_back, combine
  accumulators polling their rings, ports in cb / credit waits, routers on credits.
- Not the routing. The call's PAD rows put ~4900 of a column's 5120 tokens on each of 6 experts (54 / 99 / 153 / 318 /
  355 / 379). A standalone replay of the op (fork tests/unit/test_hot_experts.py, the model's wiring through
  weighted_case.py) with the captured routing (dump_cap128/call0000_L00.pt) PASSES (min PCC 0.99989 on every chip, 15 s).
  It also passes with poisoned L1, the model's own fabric all-gather of x, uint16 ids, the model's real x, and every
  synthetic extreme (one chip at 30720 rows, every column's tokens on one expert). In the model the call still hangs
  with random weights (diag plugin swapping the layer's fe), so weights and data were ruled out too.
- Cause: tests/bringup/contract/test_runner_smoke.py open_mesh (used by the K.1 stand-in decode and the diag
  repro) called ttnn.set_fabric_config(fabric_config) without the spec's fabric_router_config. The router kept its
  4352 B default payload, but the combine sends one 10272 B packet a row (10240 B + 32 B trailer). The replay on a mesh
  opened that way hangs with the model's exact triage signature (same kernel counts, same 15-3 / 15-2 timeout cores).
  tt-d-gen's runner opens its mesh with the payload (K-79), so the prefill ran and only the decode hung.
- Fix: open_mesh passes router_config (one call). Fork combine_rf: sparse_combine_rf_prepare refuses a combine group
  of > 1 chip whose ttnn.get_tt_fabric_max_payload_size_bytes() < piece + 32 B, naming the payload it needs
  (tests/unit/test_fabric_payload.py). No kernel or program change.
- Results: 1-layer repro REPRO_CAP=128 STEPS=0 PASS (prefill [0, 128) 13.5 s); default STEPS=1 tail PASS; all 40
  layers, REPRO_STEPS=4: PASS, model up 386 s, prefill 57.6 s, tail steps 1.1-2.7 s, greedy 'We need answer one'
  (coherent; the smoke's Paris check is K.1's runner smoke). Fork
  regression on 8x4 (BRINGUP_TEST_MESH=8x4, flat tests/unit + combine_rf tests/unit + test_flat_routed_expert_ttnn.py):
  every test that builds sparse_combine_rf here passes (test_with_program_patch, test_hot_experts 3/3,
  test_fabric_payload 2/2, test_max_y_writers 2/2). Not related to the change: test_acc_dtype (2x2 only, fabric
  init error on Galaxy), test_column_planner_mc forced to 8x4 (planner L1, never calls the combine),
  test_flat_routed_expert_op[tp4] (AssertionError (16, 1, 6, 2), the plain op), and the k2 perf band (+4.7%, E=7 node).
- Gotcha: a mesh opened by hand (ttnn.set_fabric_config + open_mesh_device) must carry every device_params entry the
  harness gives, fabric_router_config included; only the conftest / harness path applied it.
- Re-run: `cd /scratch/dnijemcevic/diag && REPRO_LAYERS=1 REPRO_CAP=128 REPRO_STEPS=0 ./run.sh <name> --no-precompile
  /scratch/dnijemcevic/diag/test_repro_tail.py -s`; op replay: `BRINGUP_TEST_MESH=8x4
  HOT_CAPTURE=/scratch/dnijemcevic/diag/dump_cap128/call0000_L00.pt scripts/run_safe_pytest.sh --run-all
  ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/test_hot_experts.py`.

## K.1 contract, attempt 6 (OV-16, owner decision B)
- test_adapter_acks failed only on dumped deep layers' window KV over every position (L20 0.981 < 0.9896, L21, L24):
  the limit (server_rules.state_threshold, drift state_window_kv_L*) is measured on the ring = the last 128 positions
  (DS41-125). Owner decision B: change the compared positions, not the limit.
- caches.GoldenKV(window="last"): a dumped layer's window KV is compared on the last min(ring, hi - lo) positions of
  each compared range [lo, hi) (ring = rows of the golden's window_kv state, 128), mode "last-window"; other layers keep
  the ring-snapshot positions; compressed / index caches every row; coverage_failures still fails any layer never
  compared. Default window="all" keeps test_cache_starts / parts unchanged.
- Used by test_adapter_acks (every ack: [max(lo, hi - 128), hi) of every layer that has dumps) and test_runner_contract's
  kv_checks (same check, same limit: its slot 1 final [0, 6572) would hit the same L20 gap).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate K.1 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.
- Then test_runner_contract's other check failed: the pulled-back chunk (0, 5120, 10240) rewrote layer 0's window KV
  [5120, 7616) with other bytes (704 rows, 1 bf16 ulp; contract: layer 0's KV depends on the token and position
  only). The differing tiles were exactly those whose tp_sum row block (640 / 4 = 160 rows) changes between starts
  2496 and 5120: the bf16 fabric reduce-scatter by row blocks adds the 4 TP partials in a block-dependent ring order
  (DS41-128). New Setting TP_SUM_BLOCKS (default cols): tt/ccl.py:tp_sum splits by column blocks (dim 3) + column
  gather, or the fixed-order gather_add where the width does not split (mHC [S/SP, 32] row). Same volume, same bf16
  precision; rows keeps the old path. After it: test_runner_contract PASS (0 failures, tt-d-gen engine).

## Merge of origin/dnijemcevic/model_bringup (owner's MoE work, d762b21db30) into the galaxy branch (2026-10-28)
- Kept both sides: theirs (Marko sync of flat_routed_expert_ttnn / combine_rf, EXPERTS_LAYOUT=marko, MOE_IMPL=block,
  MOE_HEAD / MOE_TAIL fused, matmul_reduce_scatter, D14 torus for the block, framework F200 / F201, O.1 cases); ours
  (FIX.5 patch_addresses, max_y_writers, combine_rf payload check, test_hot_experts, TP_SUM_BLOCKS, contract work,
  framework F202 open_mesh). Our framework F200 (engine wait) -> F203, F201 (hangdump) -> F204; F202 kept (free).
- tt/experts.py: their flat_op() split (shared with the block) carries max_y_writers=L1_MAX_WRITERS; the block's
  exchange (deepseek_v3_d_p tt_k3_token_split_moe.py) passes the op's max_y_writers to flat_routed_expert_with_program.
  tt/weighted_combine.py passes layout + patch_addresses + extra_addrs + max_y_writers.
- Weight cache: on this box the marko plan packs the routed weights exactly as the capped legacy plan (32 down x 5
  columns, 16 readers, no tails; key lay9c3da689dd35): layer 0 loads cache=hit under marko, flat and block. No rebuild.
- 1-layer repro (smoke prompt, prefill [0, 128) + tail [128, 160)): flat (defaults) PASS; block + fused head FAIL at
  build (DS41-129, assertion, no hang); block + MOE_HEAD=ops PASS, same tokens as flat.
- Re-run: `cd /scratch/dnijemcevic/diag && [DS41_MOE_IMPL=block [DS41_MOE_HEAD=ops]] REPRO_LAYERS=1 ./run.sh <name>
  --no-precompile /scratch/dnijemcevic/diag/test_repro_tail.py -s`; fork: `BRINGUP_TEST_MESH=8x4 scripts/run_safe_pytest.sh
  --run-all ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/test_with_program_patch.py` (and test_hot_experts.py,
  test_max_y_writers.py, combine_rf/tests/unit/test_fabric_payload.py).

## FIX.6 (overseer fix): MOE_IMPL default block, the one configuration on the Galaxy torus (owner 2026-10-28)
- tt/settings.py MOE_IMPL default flat -> block (MOE_HEAD / MOE_TAIL stay fused; no precision or limit changed); the
  setting's text carries the owner decision and the 06:20 torus evidence (block on 8x4 FABRIC_2D_TORUS_XY device
  219.5 / 233.9 / 237.5 / 249.1 / 284.4 ms at 0 / 50k / 100k / 200k / 500k vs flat FABRIC_2D 308.1 / 314.6 / 323.3 /
  338.0 / 375.7; warm wall 615-646 vs 672-1005 ms). Flat stays selectable (DS41_MOE_IMPL=flat) for the FABRIC_2D boxes.
  New DS41-130: flat hangs on the torus at the first MoE call. tt/MOE_BLOCK.md and the adapter docstring updated.
- Fabric of every gate path, checked in this gate's logs (no code change needed): the ladder / smoke / contract meshes
  open the spec's device_params (FABRIC_2D_TORUS_XY, hooks.device_params adds the 10880 B payload); the K.1 runner
  process gets PREFILL_FABRIC_MODE=2d_torus_xy from caches.fabric_mode(spec) and the payload from
  DS41Config.FABRIC_PAYLOAD_SIZE (10880, l1_small 24576): runner.log "Fabric config: FABRIC_2D_TORUS_XY (sp=8,
  PREFILL_FABRIC_MODE=2d_torus_xy)", then MOE_BLOCK layer=0 head=fused; the stand-in decode opens harness.open_mesh(S)
  (F202: every device_params entry, router_config included).
- Gate numbers, block (FIX.6) vs flat (the earlier passes): L.s10240 err_final_hidden 0.1920 vs 0.1891, top5 1.0 / 1.0
  (top1 0.9971); L.last 0.1878 vs 0.1852; L.s56320 0.1904 vs 0.1888 (limit 0.2586), top5 1.0; host_transfers 0;
  L.smoke Paris; K.1 contract 0 failures, acks_early 0, producer KV PCC window 0.9771 vs 0.9774, compress 0.9846 vs
  0.9845, index_k 0.9918 vs 0.9919; runner smoke (tt-d-gen a9701ce6675 prefill 333 s + stand-in decode) Paris.
  Unit times: W.1 2.4 min, M.1 / L.s10240 7.2-7.4, L.last 7.6, L.s56320 8.1, smoke 6.2, contract units ~6 each.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.6 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.

## P.1 (perf, galaxy 8x4): GENERIC_OP_PREPARED, the fabric reduce-scatters through prepared programs
- Fork option `ttnn.bringup.fabric_reduce_scatter(prepared=True)` (needs program_hash): `_fast` keeps a
  `ttnn.bringup.prepare_program(desc, [input, scratch, output])` next to each cached descriptor (its own LRU entry,
  dkey + "prepared") and dispatches with `ttnn.bringup.run_prepared`. The descriptor LRU is keyed by the buffer
  addresses, so a call patches nothing unless another descriptor of the same structure ran in between (then the 3 io
  slots). Not through `prepared_program.generic_op`: its 512-entry cache is shared with the MoE block.
- tt/settings.py `GENERIC_OP_PREPARED` (default on; off = generic_op on the cached descriptor, the old path);
  tt/ccl.py `prepared()` (on only with GLUE_FOLD=on) passed by `_rs_bf16`, `tp_sum`, `sum_rows`, `sum_cols`.
- Quick A/B change-on side (`measure --task P.1`): warm chunk wall median 594.3 ms (runs 570.6-639.3, spread 68.7) vs
  692.7 at X.1 (generic_op): ~98 ms a chunk host saved (estimate was ~50); every err_layer within its limit,
  host_transfers_per_layer 0, pcc_chunk_out 0.6206461212358627 = X.1's to the last digit (same programs, same args).
  Device time not measured here (no device change by construction); the orchestrator's A/B profile gives it.
- Fork tests: `BRINGUP_TEST_MESH=8x4 scripts/run_safe_pytest.sh --run-all --no-precompile
  ttnn/ttnn/bringup/fabric_reduce_scatter_ttnn/tests/unit/test_fabric_rs_prepared.py`: 9 passed (bit-identical to
  generic_op). Whole tests/unit/ on 8x4: 21 passed; 3 failures reproduce on the unchanged file (2x2-sized full-mesh
  tests), 13 errors are fabric router sync timeouts opening 2x4 sub-meshes on the Galaxy (not this change).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.orchestrator measure --task P.1`; off side:
  `DS41_GENERIC_OP_PREPARED=off`.

## P.2 (perf, galaxy 8x4): TAIL_A2A_PREPARED, the block MoE's fused tail through prepared programs
- Block option `MoEOptions.tail_prepared` (default False = the old path) -> `FusedTail(prepared=)` ->
  `TailA2A(prepared=True)` (models/demos/moe_block/tt/tail_a2a.py `run`): each cache entry, keyed by the y / se / out
  addresses as before, now holds `(descriptor, ttnn.bringup.prepare_program(descriptor, io))`; a hit calls
  `ttnn.bringup.run_prepared` (no per-call hashing of the inline kernel sources, no runtime args re-applied over 32
  chips). Built for exactly those addresses, so a hit patches nothing (same pattern as P.1, DS41-66). io order
  unchanged: [y, out, z, zrw] (+ se). `TailA2A.misses` counts descriptor builds (diagnostic).
- tt/settings.py `TAIL_A2A_PREPARED` (default on; off = ttnn.generic_op on the cached descriptor);
  tt/moe_block_adapter.py `block_options()` passes it.
- Not done: the persistent `out` of the proposal (optional): the model frees the tail's output downstream, a kept
  buffer needs lifetime changes in model code; the timed runs show no new program-cache entries (280 -> 280), so the
  per-call `out` already lands at repeating addresses and the descriptor cache hits.
- Quick A/B change-on side (`measure --task P.2`): warm chunk wall median 544.4 ms (runs 534.7-574.6, spread 39.9)
  vs 594.3 at P.1 (the same tree with the tail on generic_op): ~50 ms a chunk host saved (estimate ~36). Every
  err_layer / err_state within its limit and equal to P.1's to the last digit (L00 0.0247993108407984,
  pcc_chunk_out 0.6206461212358627): same programs, same args. host_transfers_per_layer 0. Device time not measured
  here (no device change by construction; the tail's program is 0.079 ms/layer); the orchestrator's A/B profile gives it.
- No new fork test: prepared_program's model cases (tests/cases.py) build the block with tail_prepared off; a tail case
  needs its signatures captured (optests).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.orchestrator measure --task P.2`; off side:
  `DS41_TAIL_A2A_PREPARED=off`.

## P.3 (perf, galaxy 8x4): FUSED_HEAD_PREPARED_PER_LAYER, one prepared fused-head program per layer
- Diagnostic first (brief: stop if ~0): the fused head's shared prepared program (fused_head.py `_run`, layers mode)
  records 40096 patch slots over 32 chips; per io: x 10848, w 3520, y 7040, arena 8352, rows 480, xbuf 3520, route
  160, router weights (io 7-9) 3680 + 160 + 160, slots 576, planner outs 5 x 320. Patched every call (weights move):
  w + router = 7520 slots (io 1, 7, 8, 9). Not ~0, so the change went ahead. Old path warm head call: ~710-745 us host
  (`run_prepared` timed in-process, DIAGTMP instrumentation, removed).
- Why the proposal's "40 PreparedPrograms keyed by skey" alone would not work: the prepared hash excludes patched
  slots, so 40 programs with w in io share ONE cached device program and still patch. The weights must be baked in.
- Fork option (ttnn/ttnn/bringup/prepared_program): `ttnn.bringup.freeze_prepared(prepared, io, frozen)` derives a
  PreparedProgram with the io tensors at `frozen` written into their slots / CBs, no longer patched, part of the key
  (salted). prepare_program / run_prepared unchanged (the hash walk moved to a helper, same result).
  CHANGELOG, README, INDEX updated; test `_check_frozen` added to the fused-head case (frozen A / B bit-identical to
  generic_op / shared, fewer patches, distinct keys, deterministic). Fork tests on this box: 2 skipped (2x2 cases need
  exactly 4 devices; the galaxy has 32): the new check has NOT run on a device yet.
- Model: `MoEOptions.head_per_layer` (moe_block.py) -> `HeadSE3.per_layer`; `HeadSE3._layer_pp` keeps `state["pl"]`
  {weight addresses: frozen program}, cleared when the shared program is re-prepared. tt/settings.py
  `FUSED_HEAD_PREPARED_PER_LAYER` (default on per the brief; off = the shared patched program, the old path);
  tt/moe_block_adapter.py passes it.
- Result (change on, `measure --task P.3`): accuracy identical to X.1 / P.2 to the last digit (pcc_chunk_out
  0.6206461212358627, every err_layer within limit, host_transfers_per_layer 0); program cache 320 (+40, one per
  layer). Warm chunk wall median 536.8 ms (spread 70.3) and 552.6 ms (spread 49.2) vs P.2 544.4 and an old-path run
  in this step 524.4 (with timing instrumentation). Warm head call host ~970 us (instrumented run; per-layer lookup
  ~9 us) vs ~710-745 us old: the change is SLOWER by ~250 us a call, ~10 ms a chunk. Patching 7520 slots was cheap;
  switching among 40 cached 32-chip programs costs more (likely host cache misses on their dispatch data; not
  isolated). DS41-131. Recommendation: off. Device time not measured (no device change by construction: the same
  kernels and arg values).
- Gotchas: one instrumented measure died with a bus error in to_torch after warm-up (F207-like), its automatic
  re-run found "No such device"; the next measure ran clean. tt-probe.sh saves scripts under tests/ (outside this
  step's paths): diagnostics went through measure with temporary logging instead.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.orchestrator measure --task P.3`; off side:
  `DS41_FUSED_HEAD_PREPARED_PER_LAYER=off`.

## FIX.7 (overseer fix): GEMM_INPUTS, the fp8-quantized GEMM inputs fed as bfloat8_b (owner 2026-10-28, F208)
- What: tt/settings.py `GEMM_INPUTS` (default bfp8; bf16 = the path before FIX.7), tt/common.py `gemm_dtype()` /
  `gemm_input(x)`. The reference (ds41_ref.py `_q8` / `_fp8_linear`, ~620-630; the attention output's q8 before wo_a,
  ~889) rounds these inputs to fp8; the device now stores and multiplies them in its own 8-bit format. Weights,
  fidelities (ATTN_FIDELITY HiFi2, MATMUL_FIDELITY HiFi2) and fp32 accumulation unchanged.
- Per matmul (8x4, per chip, input dtype before -> after; how):
  q_a [wq_a | wkv] 640x1280: bf16 -> bfp8, one typecast of attn_norm (attn_norm stays bf16: the indexer's
  weights_proj and the compressor read it, both bf16 in the reference); tt/kv.py's standalone path the same.
  attn wq_b 640x1280: bf16 -> bfp8, the softmax-scale multiply writes bfp8 (ROPE_IMPL=on; other paths gemm_input).
  attn wo_a 2x640x4096: bf16 -> bfp8, one typecast after nlp_concat_heads (rotary_embedding_indexed has no output
  dtype; a stock op). attn wo_b 640x2048: bf16 -> bfp8, wo_a writes bfp8 (nlp_concat_heads takes bfp8).
  indexer wq_b 160x1280: bf16 -> bfp8, one typecast after mesh_partition. engram wkv 640x6144: bf16 -> bfp8, one
  device typecast of the uploaded rows (host upload stays bf16: no host bfp8 packing per chunk).
  Routed experts x / h: bfp8 already (flat op, EXPERTS_PRECISION=glm). Shared expert x / h on the default block
  fused head: still bf16, hard-coded in models/demos/moe_block/tt/se_engine3.py, outside this step (DS41-132).
- Device time, plain profile, rung last chunk [51200, 56320], 40 layers (X.1 at 10:01 today, same programs bar this
  change, vs FIX.7): total 223.85 -> 222.75 ms a chunk (attention 78.02 -> 78.71, q_a 9.34 -> 9.54, indexer 9.73 ->
  9.80, engram_kv 0.58 -> 0.62; the others noise). Per chunk, matmuls bf16 -> bfp8 input: wq_b 3.554 -> 3.065, wo_b
  3.357 -> 3.203, wo_a 2.777 -> 2.391, q_a 1.116 -> 1.024, indexer wq_b 0.319 -> 0.313, engram 0.584 -> 0.553;
  added typecasts: wo_a input 1.543 (the one real cost), q_a 0.287, engram 0.073, indexer 0.023. Profile wall 692.7
  -> 674.8 ms (not a clean wall run). Profile kept in generated/bringup_adhoc/FIX.7_profile.json (+ _ops.json.gz).
- Accuracy, W.1 (seed rung, per-step; before = W.1's 10-26 record): err_layer L00 0.0249 -> 0.0241, L01 0.0368 ->
  0.0368, L02 0.0516 -> 0.0516, L03 0.0711 -> 0.0713, L14 0.0320 -> 0.0297, L20 0.0300 -> 0.0299, L21 0.0436 ->
  0.0429, L24 0.0313 -> 0.0310 (limit 0.0336); window_kv L14 0.0054 -> 0.0112 (limit 0.065); host_transfers 0.
- Gate (FIX.6 block numbers before -> FIX.7): final hidden rel L.s10240 0.1920 -> 0.1919 (top1 0.9971 -> 1.0, top5
  1.0), L.last 0.1878 -> 0.1843 (top1 0.9971, top5 1.0), L.s56320 0.1904 -> 0.1877 (limit 0.2586; top5 1.0); s56320
  err_layer L00 0.0248 -> 0.0248, L14 0.0941 -> 0.0933, L20 0.1639 -> 0.1606, L24 0.1382 -> 0.1362, L39 0.715 ->
  0.867 (limit 1.49); host_transfers 0; L.smoke Paris; K.1 contract 0 failures, acks_early 0, producer KV PCC window
  0.9771 -> 0.9783, compress 0.9846 -> 0.9843, index_k 0.9919 -> 0.9916. Every unit PASS. test_runner_smoke failed
  once with a SIGBUS in the tt-d-gen runner after warm-up ("Non-existent physical address", no IOMMU mismatch; the
  same intermittent host fault as P.3's measure); test_runner_contract (same engine, same code) passed, and the
  smoke's re-run passed (unit re-run, unchanged code).
- Follow-up: a bfp8 output on the inverse rope (a fork of rotary_embedding_indexed) would drop the wo_a typecast
  (~1.5 ms a chunk); a bfp8 x / h option in se_engine3 for the shared expert (DS41-132).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.7 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`; old path: DS41_GEMM_INPUTS=bf16.

## FIX.10 (overseer fix): ENGRAM_TABLE=resident coexists with the full model on the Galaxy host (DS41-133, DS41-134)
- Cause, measured (host-only probes, /tmp/fix10_scripts, no device; then the gate's runners):
  1. Threads (the slowdown). numba's omp threading layer sets the OpenMP max-threads ICV to NUMBA_NUM_THREADS (64 =
     every logical CPU) at its first launch, and torch takes its intra-op pool from the same ICV: 32 -> 64. A
     head-layout-like torch workload (32 chips x 32-row block permutes + cats) went 2.0 s -> 23 s (user 1295 s, sys 0,
     no reclaim / compaction during it) after one pack_rows(threads=8); torch.set_num_threads(32) -> 2.0 s again. The
     resident build runs numba at model init, so the whole load and the fused head's lazy per-layer layout ran at 64
     threads: the 18:39 runner's head first calls took ~112 s a layer (random: ~6 s; the ~59 busy cores of the py-spy
     note were OMP threads spinning). Random mode was hit too, from its first begin_chunk's pack_rows on; the same
     chunk's hash + gather + TILE bytes took 110 ms at 64 threads vs 8 ms at 32.
  2. THP compaction (the build). The host's THP defrag is 'madvise': each 2 MB fault of the MADV_HUGEPAGE rows
     compacts in the faulting thread while the page cache fills memory. Both tables: 236 s (with the 64-thread
     problem; sys 860 + 1867 s, compact_stall 214k) / 171 s with the thread fix (layer 14 alone 130 s, compact_stall
     125k, 22k fallbacks to 4 KB) and direct reclaim dropped 95 GB of page cache; 4 KB pages: 74 s, compact_stall 0,
     page cache kept. Huge pages gather faster: 6.3 vs 8.5 ms a layer-chunk (8 threads).
  3. Page cache: the build's pread left the ~200 GB of table bytes cached, evicting the weight cache the load reads
     next. Swap (8 GB, swappiness 60) fills from cold anonymous pages either way; it was not the slowdown.
- Changes: tt/engram_bfp8.py `_run` restores torch.get_num_threads() around every numba launch (set_num_threads
  included; holds from any thread), kernels nogil=True. tt/engram.py: rows in 4 KB pages unless ENGRAM_HUGEPAGES=on
  (default off), every read block's file pages dropped (POSIX_FADV_DONTNEED, `_drop_cache`; resident_copy too),
  ENGRAM_BUILD=background (default; load = the old blocking build): one builder thread a process started when the
  engram layers are created (before the decoder layers load), `TtEngramRows.packed` waits for it, the gather kernel
  warmed on that thread (~0.5 s off the first chunk). Probe: both tables built in 70.8 s behind the main thread, whose
  torch work ran at baseline meanwhile (1.8 -> 1.9-2.3 s). tt/perf_engram_host.py: CPU check of the thread guard.
- Rows bit-identical: tests/test_engram_table_modes.py with DS41_TEST_ENGRAM_RESIDENT=1 9 passed (resident_bfp8
  rows == pack_rows, tiles == the per-chunk pack, resident_copy bytes); tests/test_engram_bfp8.py 1 passed (device).
- Numbers (runner start -> first forward; random = the 19:12 run, resident before = 18:40):
  load: resident before 7m43s (4m23s blocking table build, then 40 layers in 3m16s at 64 threads), random 2m38s,
  resident now 3m21s (smoke) / 2m56s (runner contract). The rest vs random: 210 GB of tables leave ~220 GB of page
  cache for the 285 GB weight cache, so part of it comes from disk (expert loads summed 72.7 s in the smoke's runner,
  32.3 s in the contract's). Fused head's first call a layer: 112 s -> 3.5 s (random 6 s, also at 64 threads then).
  K.1 runner smoke (Paris, tt-d-gen): 848.6 s test call, runner prefill 427 s, runner wall 421 s (random 19:12:
  897.5 s / runner 455 s; resident 18:39: stopped at layer ~15 after 30 min). Runner peak RSS 244 GB (smoke) / 249 GB
  (contract), disk read 444 / 244 GB (200 GB of it the tables), system swap max 7 / 3 of 8 GB, MemAvailable min 176 /
  140 GB; compact_stall +2 over the whole gate. The 17:00 SIGBUS did not recur; FIX.7 saw the same tt-d-gen SIGBUS
  ("Non-existent physical address") with random tables, so it is likely not from the tables.
- Gate FIX.10: all 6 units PASS (L.s56320 591 s, err_final_hidden 0.1889 <= 0.2586, top5 1.0; contract 0 failed,
  acks_early 0, producer KV PCC window 0.9777 / compress 0.9847 / index_k 0.9919; smoke_runner_ok 1).
- Open: the spec's owner rules ("never hold the 203 GB of tables in host RAM", "Host RAM is shared (503 GB)") still
  disagree with the default (the overseer raises it). Pytest processes keep ENGRAM_TABLE=random (F222).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.10 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`; old behaviour:
  DS41_ENGRAM_HUGEPAGES=on DS41_ENGRAM_BUILD=load (the thread guard has no switch: it only restores torch's count).

## FIX.8 (overseer fix, 2026-10-28): CKV_FORMAT=bfp4_rows integrated and gated at full size on the Galaxy 8x4
- What: no code change was needed. The 2x2's work (merge bf6c8b497a6 = dnijemcevic/model_bringup_bfp4kv @ 9697878dfb0;
  ttnn rebuilt with it at 22:10, `_ttnn.so` has ROW_BFP4) runs as merged on the 8x4. CKV_FORMAT defaults to bfp4_rows
  (tt/settings.py; bf16 = the old stock path). Checked for box literals: ckv_bfp4 takes its core grid from the device
  (`_cores`: bands of 32 rows over compute_with_storage_grid_size, 80 bands at r 2 / 160 at r 1 for chunk 5120); the
  sparse_sdpa ROW_BFP4 program has no grid or chip constants; AttnTable sizes come from TtServedRows (`max_seq / r`
  rows, `WINDOW + geo.chunk` window rows); no CCL or fabric payload is involved (the table is replicated, every chip
  writes its own copy from the replicated gathered tiles).
- Gate FIX.8 (`--check`, all on the merged code, 4917 s + W.1 204 s): every unit PASS. Accuracy vs the bf16 path
  (FIX.10's records, same rungs; value / limit):
  - L.s56320: final hidden rel 0.1889 -> 0.1961 / 0.2586, top5 1.0 -> 1.0; err_layer L02 0.0526 -> 0.0573, L08 0.1125 ->
    0.1221, L14 0.0930 -> 0.0981, L20 0.1636 -> 0.1678, L24 0.1386 -> 0.1431, L39 0.905 -> 0.613 / 1.49; compress_kv
    state L02 0.0330 -> 0.1161 / 0.2011, L08 0.0732 -> 0.1445 / 0.2259, L14 0.1547 -> 0.1973 / 0.3294, L20 0.1288 ->
    0.1747 / 0.2912; topk_idx L20 0.1163 -> 0.1216, L24 0.1299 -> 0.1364; candidates L20 0.0364 -> 0.0382.
  - M.1 / L.s10240: final hidden pcc 0.9794 rel 0.2021 (FIX.7: 0.1919), top1 / top5 1.0. L.last: rel 0.1914 (FIX.7:
    0.1843), top1 0.9971, top5 1.0. W.1 (seed): err_layer L14 0.0331 / 0.0550, L20 0.0309 / 0.0370, L24 0.0318 / 0.0336
    (closest to its limit), compress_kv L02 0.116 / 0.201, L14 0.112 / 0.191, L20 0.113 / 0.191.
  - K.1: contract 0 failed, acks_early 0, producer KV PCC compress 0.9847 -> 0.9767 / 0.9370, window 0.9777 -> 0.9732,
    index_k 0.9919 -> 0.9909; L.smoke and the tt-d-gen runner smoke both answer Paris. host_transfers_per_layer 0.
  - The served compressed records stay bf16 (state_bits compress_kv 16): they hold the bfp4-rounded values, so the
    serving contract and tt-d-gen are unchanged.
- DRAM a chip of the replicated attention tables at the deploy context (1,044,480 tokens, KV sources 2 / 8 / 14 at
  ratio 2 and 20 at ratio 1; from the allocated shapes in tt/compressor.AttnTable):
  bf16 rows: 3 x [527488, 512] bf16 (515.1 MiB) + [1049728, 512] bf16 (1025.1 MiB) = 2570.5 MiB (2.51 GiB).
  bfp4_rows: 3 x ([522240, 320] u8 159.4 MiB + window [5248, 512] bf16 5.1 MiB) + ([1044480, 320] u8 318.8 MiB + 5.1
  MiB) = 817.4 MiB (0.80 GiB). Saves 1.71 GiB a chip (3.1x); the plan's deployment margin was 0.3 GiB.
- Time, chunk 5120, 40 layers, resident engram (DS41_ENGRAM_TABLE=resident), same session, bf16 -> bfp4_rows
  (scripts /scratch/dnijemcevic/fix8/, results /scratch/dnijemcevic/posprof/results/CKV16* / CKV4*):
  - sparse_sdpa kernel (64 heads x 160 queries a chip, top-k 512 after a 128 window): position 0 13.60 -> 13.94 ms a
    chunk (+2.5 %; per compressed layer 0.351 -> 0.360 ms, 38 calls; layers 0 / 1 stay stock, 0.136 ms each);
    position 512k 16.70 -> 17.39 ms (+4.2 %; 0.432 -> 0.451 ms a call). The 2x2 single-card figure was +3.5 % at 160 q.
  - Table write: ckv_bfp4 0.209 ms a chunk (4 calls) vs slice_write 0.066 ms; the start < 128 compaction (ttnn.gather
    0.18 ms at position 0) is gone in bfp4_rows.
  - Device time a chunk: position 0 216.44 -> 217.66 ms (+0.56 %), 512k 283.77 -> 284.25 ms (+0.17 %).
  - Warm wall (F196, 10 timed runs, misses refused), medians: position 0 419.7 -> 441.6 ms and in a second pair 412.5
    -> 426.3 ms; 512k 451.3 -> 472.0 ms. Run spreads 30-130 ms. cProfile of 5 warm chunks (position 0) puts the
    attributable cost at +5.3 ms a chunk, all of it ckv_bfp4's host time (1.33 ms a call, Python descriptor 1.4 ms of
    the 5.3; DS41-135); both cProfiled runs took 443 ms a chunk, and the other differences (engram tile packing,
    experts) moved in both directions run to run. So the real wall cost is about +5 ms a chunk (+1.2 %), not the +14 to
    +22 ms the medians show.
- Follow-up: cache ckv_bfp4's ProgramDescriptor (patch row0 / addresses; ttnn.bringup.prepared_program) or make it a
  C++ op, which removes the ~5 ms of host time a chunk (DS41-135). The sdpa fork README had no ROW_BFP4 entry: added
  (options, Galaxy numbers, used by); ckv_bfp4 README: Measured + the host limit.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.8 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`; old path DS41_CKV_FORMAT=bf16;
  profile series `/scratch/dnijemcevic/fix8/series_ckv.sh`, host profile `/scratch/dnijemcevic/fix8/hostprof_ckv.sh`.

## P.6 (perf, galaxy 8x4): ENGRAM_ASYNC, the engram host work overlapped with the layer dispatch
- What: setting ENGRAM_ASYNC (tt/settings.py, default on; off = the old path, unchanged code). On: begin_chunk
  (tt/model.py:_begin_engram_async) does only the new tails (TtEngramRows.next_tail: the 3 compressed ids, no hash) and
  allocates each engram layer's device rows (TtEngramRows.allocate, the same order as off) on the main thread, then
  submits one job a layer, layer 1 first, to a single worker thread (ThreadPoolExecutor(1)). The job
  (TtEngramRows.fill) hashes, looks up + packs, and uploads with copy_host_to_device_tensor into the allocated tensor;
  it also checks its hashed tail equals the one begin_chunk stored. take_engram_rows(i) waits for layer i's job only;
  begin_chunk waits for any job left from the last chunk (a layer that did not run) before freeing its tensor.
- Why no second CQ: the harness (spec device_params) and tt-d-gen (prefill/runners/runner_utils.open_mesh_device) open
  one CQ, and MeshDevice has no Python CQ count. The worker's write goes into CQ 0 (the API is locked:
  mesh_command_queue_base lock_api_function_); CQ 0 is in order, and layer i's ops are enqueued after the wait, so the
  write lands first. Allocation stays on the main thread so device addresses do not depend on thread timing.
- DS41-136: the off path's wrap (`.numpy().tobytes()` + tensor_from_bfp8_bytes, ~35 ms a layer at 5120) and
  ttnn.to_device hold the GIL, which would stall the dispatching main thread. On: one staging host tensor a layer
  (TtEngramRows._staging: 8 mesh-row bfp8 tensors, the mesh tensor's 32 shards share them), tile_rows(out=) writes the
  bytes in place through torch.from_dlpack(host_buffer().get_shard(c)); the copy (GIL released) reads an unpinned
  buffer synchronously (fd_mesh_command_queue.cpp:write_shard_to_device, pinned_memory null), so the reuse is safe.
- DS41-133: the worker's numba launches go through engram_bfp8._run (torch's thread count kept); perf_engram_host passes.
- host_transfers_per_layer: testing/host_transfers.py wraps the ttnn attributes for the whole process, so a
  begin_chunk upload issued by the worker while the main thread is inside a layer call would count against that layer.
  The worker calls copy_host_to_device_tensor / from_torch through references bound at import (tt/engram.py
  _COPY_TO_DEVICE / _FROM_TORCH, commented). The layers' own transfers are counted as before.
- Rows: tt/perf_engram_async.py::test_async_rows_bit_identical: fill() on a thread == the off path's upload on every
  chip, layers 1 and 14, bfp8 and bf16, full / padded / 2 / 0 real ids, next_tail == the hashed tail (staging reused
  across the 4 chunks). tests/test_engram_table_modes.py, test_engram_bfp8.py, tt/perf_engram_host.py: 8 passed, 1 skipped.
- Measured (tt/perf_engram_async.py::test_engram_async_chunk, DS41_ENGRAM_TABLE=resident, chunk
  51200-56320 (rung last), 40 layers, one process, on / off alternating, 8 warm-up + 6 timed runs each):
  - off: begin_chunk 48.9 ms (median; both layers' hash, gather + tile, wrap, to_device, all on the main thread).
  - on: begin_chunk 0.60 ms. Worker, median ms: layer 1 hash 6.8, lookup + pack (one numba pass, resident bfp8 rows)
    16.8, upload 2.75 (32 chips, 4.2 MB a chip), total 26.3, main thread waited 6.1 (embed + layer 0 dispatch hides
    the other ~20); layer 14 hash 3.0, lookup + pack 12.8, upload 2.7, total 18.5, waited 0.01 (all hidden).
  - Hidden: ~42 of ~49 ms a chunk (layer 1's 6.1 ms wait is what is left).
  - Chunk wall (begin_chunk to sync) median off 490.0 ms [443-552], on 449.1 ms [422-517]; run-to-run spread is
    larger than the gain, the medians differ by -41 ms (= -48.9 + 6.1 + 0.6). Last-layer output on == off bit for bit.
  - Device time is unchanged by construction (the same programs on the same bytes; the write is the same size).
- Gate P.6 `--check` (code final, 954 s): every accuracy metric ok (err_final_hidden 0.1914 / 0.2422, top5 1.0, err_layer
  L01 0.0367 / 0.0667, L14 and every state within limits), chunk_wall_ms 424.8 < 434 (measure --task P.6: 421.0,
  spread 26), host_transfers_per_layer 0. It still prints FAIL: `MISSING device_ms_total`. The task's gate command
  carries BRINGUP_PROFILE_WALL=1 on test_profile.py, so core/gate.py:with_clean_wall adds no profiled run and no unit
  can record device_ms_total (a tasks.yaml / framework issue; not editable in this step). The plain profiled run
  (`DS41_ENGRAM_TABLE=resident scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/common/bringup/tests/test_profile.py`, metrics in generated/bringup_adhoc/P.6.json): device_ms_total
  220.83 ms (<= 230; chunk 51200-56320), profiled_new_programs 0, host_transfers_per_layer 0.
- Next-chunk prefetch: the serving contract does not give the next chunk's ids: prefill_chunk(input_tensor, kv,
  slot_id, actual_start, actual_end) carries only this chunk (tt/runners/adapter.py:_run reads them from the engine's
  device input), so the first wait (layer 1, ~6 ms) cannot be hidden by prefetching without an engine change.
- Re-run: `DS41_ENGRAM_TABLE=resident PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  "models/demos/deepseek_v41_flash_d_p/tt/perf_engram_async.py"`; gate `PYTHONPATH=$PWD python -m
  models.demos.common.bringup gate P.6 --spec models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml
  --check`; old path DS41_ENGRAM_ASYNC=off.

## FIX.9 (overseer fix, attempt 3): the shared expert's x / h as bfp8 in the block MoE's fused head (DS41-132)
- Why: OWNER DECISION 2026-10-28 (F208, FIX.7 GEMM_INPUTS=bfp8): the reference fp8-quantizes the shared expert's x and
  h (reference/ds41_ref.py `_swiglu`: `_q8(x)`, `_q8(h)`); on MOE_HEAD=fused they stayed bf16 (se_engine3 hard-coded
  2048 B tiles), above the reference. Attempt 2 left no changes in the tree (only ledger files).
- Block (models/demos/moe_block, the option defaults to bf16 for every other user):
  - tt/se_engine3.py `SeEngine3(x_dtype=, h_dtype=)` (bf16 / bfloat8_b; `TILE_BYTES`): x own, the x rounds (XIN), the
    own-shard DRAM load, h own / h all and their CBs take the dtype's tile bytes (`SE_XTB` / `SE_HTB` defines in the
    BRISC); the dispatch prep's source view (PU) reads x own in its format (`SE_PREP_TB`; units' offsets x tile bytes),
    so the routed experts' dispatched rows are bfp8-rounded x (their own bfp8 x rounding is then exact). The non-RS
    down path now reconfigures the packer for OUT (it relied on h being bf16). No down (`mrs`): h must stay bf16.
  - tt/x_gather3.py: the x packets' addresses / payloads use `XTB` (the engine's x tile bytes); the port CB keeps its
    2048 B slots (RS, latent, logits unchanged), so a bfp8 packet carries the same tiles in 1088 B each.
  - tt/fused_head.py `HeadSE3(activations=)`, `run(x, w, xq=)`: the router keeps x (bf16), the engine and the gather
    take xq; xq is appended last to the prepared program's io (the weights' positions unchanged) and in its key.
  - moe_block.py: `MoEDtypes.shared_activations` = x and h inside the shared expert. FusedHead allocates a persistent
    bfp8 `xq` and typecasts x into it each call (`ttnn.typecast(output_tensor=)`; one op, no host). Ops head
    (SharedExpert): x cast for gate / up, gate / up write bf16 (they wrote shared_activations before: identical at
    the bf16 default), the activation's output cast for down.
- Model: tt/moe_block_adapter.py `block_dtypes()` sets `shared_activations=gemm_dtype()` (GEMM_INPUTS; bf16 = the old
  path); tt/settings.py GEMM_INPUTS text, tt/MOE_BLOCK.md D5, known_issues DS41-132 updated.
- Why a typecast copy, not rounding where the SE reads x: the router reads x from DRAM in bf16 (the reference's
  router is bf16), and the engine's own-shard load and the gather move tiles as stored, so rounding in the engine
  would need an extra unpack / pack pass of every x tile on every engine core; the typecast is 7 us a layer.
- Measured, 8x4, chunk 5120, real weights (tt/perf_se_activations.py, random x, every 7th token vs float64 with the
  checkpoint's exact fp8 weights), DS41_GEMM_INPUTS=bf16 -> bfp8:
  - engine x / h BFLOAT16 -> BFLOAT8_B, arena 1142 -> 985 KB; head twice: 0 values differ (both).
  - shared expert vs native (x, h unrounded): L00 rel 0.01158 -> 0.01704 (bias -0.0085 -> +0.0074, worst row 0.0145
    -> 0.0206); L24 0.01232 -> 0.01812 (bias -0.0085 -> +0.0077). vs q8 (fp8 act quant of x and h, per 32): L00
    0.04388 -> 0.04688, L24 0.04637 -> 0.04940 (bfp8 rounds differently from e4m3; the positive bias is the bfp8
    packer's ties-away rounding, DS41-77).
  - head alone (rt profiler, 10 calls): section 454.0 -> 462.2 us a call (L00; L24 454.7 -> 464.1): the program
    ~446 -> ~451 us, + typecast ~7 us, + the x copy 10 us both; synced wall 0.958 -> 0.973 ms.
- In the model (test_profile.py, DS41_ENGRAM_TABLE=resident, rung last, 40 layers, same session; bf16 side =
  DS41_GEMM_INPUTS=bf16, which also turns FIX.7's attention-side casts off): fused head (the "(other)" program in
  L*.experts) 38.35 -> 38.53 ms a chunk (median 0.951 -> 0.966 ms a layer), typecast 0.29 ms a chunk, experts section
  94.90 -> 95.32 ms; device total 224.30 -> 225.95 ms (P.6's run of the same code before this change: 220.83, run to
  run noise ~2 % in untouched sections, e.g. attention 78.35 / 79.74 / 80.00). host_transfers_per_layer 0,
  profiled_new_programs 0. Warm wall (BRINGUP_PROFILE_WALL=1): 429.4 ms (spread 48.2, loadavg 15.9) -> 435.3 ms
  (spread 47.1, loadavg 21.0): within the spread. Files generated/bringup_adhoc/FIX.9_{wall_bf16,wall_bfp8,profile,
  profile_bf16}*.json.
- Accuracy, W.1 seed rung (unit passed; before = FIX.8's record, the last on this code minus the change): err_layer
  L00 0.0241 -> 0.0243, L01 0.0369 -> 0.0359, L02 0.0568 -> 0.0557, L03 0.0777 -> 0.0762, L14 0.0331 -> 0.0329, L20
  0.0309 -> 0.0306, L21 0.0462 -> 0.0457, L24 0.0318 -> 0.0314 (limit 0.0336).
- Re-run: `DS41_GEMM_INPUTS=bf16 PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_se_activations.py` (and without the env for bfp8); gate
  `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.9 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`; old path
  DS41_GEMM_INPUTS=bf16.
- First full gate (code above, no precise packing): every unit PASS but K.1 test_adapter_acks: chunk 0 ack slot 0
  [0, 2536), last-window 128 rows, window_kv layer 20 nope PCC 0.98862 < limit 0.98957 (DS41-125's deep-layer check);
  the chain stopped there (runner_contract / runner_smoke not run). Diagnosis (temporary prints in the contract's
  caches.pcc_failures and an env switch for the SE dtype, both removed; caches.py restored byte for byte), the same
  test, every window_kv check of layers 14 / 20 / 21 / 24, SE bf16 (FIX.7 kept) / SE bfp8 / SE bfp8 + precise:
  mean 1 - PCC 0.00772 / 0.00751 / 0.00779, min margin to the limit +0.00323 / -0.00095 / +0.00174; L20 chunk-0
  window 0.99280 / 0.98862 / 0.99131, L24 0.99051 / 0.98717 / 0.99039; bfp8 is better on 5 of the 7 windows a layer.
  So the 128-row windows move by +-0.003 with any rounding change (routing flips at deep layers, DS41-90): the
  failing window is noise at the limit, not a loss of accuracy, and no variant is better on average.
- Then shipped: SE_PACK_PRECISE (tt/settings.py, default on; MoEDtypes.shared_pack_precise -> FusedHead / HeadSE3 ->
  SeEngine3(pack_precise=) -> ComputeConfigDescriptor.bfp8_pack_precise, plus a `#define SE_PACK_PRECISE` because the
  compute binary's cache key leaves the flag out, DS41-137). Why, on its own merit: one rounding of h from fp32 (as
  ttnn.typecast packs x, as the reference casts h once) instead of two (ties away twice); same format, same speed.
  Measured on the same inputs: SE vs native L00 rel 0.01704 -> 0.01647, bias +0.0074 -> +0.0063 (vs q8 0.04688 ->
  0.04662); head 462.2 -> 463.9-464.7 us a call (noise). The contract window passing with it (0.99131) is partly luck,
  per the means above: an owner / overseer reading this should treat that check as noise-limited at layer 20.
- Re-run of the contract diagnosis: not kept (it edits the frozen helper); use the numbers above.

## P.7 (perf, galaxy 8x4): CKV_BFP4_PREPARED, ttnn.bringup.ckv_bfp4 as a prepared program (DS41-135)
- What: setting CKV_BFP4_PREPARED (tt/settings.py, default on; off = the old path, unchanged code). On:
  tt/compressor.AttnTable.append_tiles calls ttnn.bringup.ckv_bfp4(..., prepared=True): the program is built once per
  (mesh.id(), n, rounded, memory configs) (ckv_bfp4.py _prepared, LRU 64), prepared with ttnn.bringup.prepare_program
  and dispatched with run_prepared each call: ckv / rows / out addresses patched, row0 patched as a value slot.
- Fork change (prepared_program, CHANGELOG 2026-10-29): option `value_slots=[(kernel, arg)]` on prepare_program and
  `values=[...]` on run_prepared. Before it only io-tensor addresses were patched, and row0 (the writer's runtime arg 3)
  changes every call, so without it every call would need a new prepare. A value slot is never matched to an address
  and is left out of the cache key. With no slots the hash walk and key are as before (fabric_reduce_scatter / TailA2A
  prepared paths unchanged). ckv_bfp4 checks patch_count == cores x (3 + rounded) and falls back to generic_op if a
  frozen arg collided with an io address.
- Bytes: ckv_bfp4 tests/unit 8 passed (new test_ckv_bfp4_prepared: 5 calls at other row0 / inputs / tables, byte-equal
  to the generic_op path, prepared once); sdpa tests/unit/test_sparse_sdpa_row_bfp4.py passed; prepared_program
  tests/test_prepared_program.py: its cases are 2x2-only, 2 skipped on the 8x4 (as before).
- Host (8x4, chunk 5120, position 0, DS41_ENGRAM_TABLE=resident, cProfile of 5 warm chunks after warm-up, 4 calls a
  chunk; /scratch/dnijemcevic/p7/hostprof_p7.sh and _rev.sh, results /scratch/dnijemcevic/p7/): ckv_bfp4 1.427 / 1.307
  ms a call off -> 0.503 / 0.508 on: 3.4-3.7 ms host saved a chunk. Isolated (tt/perf_ckv_bfp4_host.py, 20 warm calls):
  n=1280 0.514 -> 0.133 ms, n=64 0.168 -> 0.094 ms; tables equal. In the model the remaining ~0.5 ms is
  allocate_tensor_on_device for the rounded output + run_prepared (C++ calls, not split by cProfile).
- Warm wall (BRINGUP_PROFILE_WALL, 10 timed runs, medians): pair 1 (on first) on 415.7 / off 393.3; pair 2 (off first)
  off 391.8 / on 389.6. In pair 1 every unrelated host function was slower in the on run too (decorators __call__ +66
  ms over 5 chunks, fused_head +14, tail_a2a +7) while ckv_bfp4 saved 18.5 ms: the first run of a session is slower.
  So the wall gain is within the 30-50 ms run spread; expect ~3.5 ms. measure --task P.7 (quick, on): 413.3 ms, every
  err limit held (L39 0.907 / 1.533), host_transfers_per_layer 0.
- Device time: the same kernels, compile args and runtime args (bytes equal), so the device program is unchanged
  (FIX.8: ckv_bfp4 0.209 ms a chunk). Not re-profiled; the orchestrator's A/B profile has device_ms_total on and off.
- For optests: the model now calls ttnn.bringup.run_prepared with a value slot (ckv_bfp4 prepared); a prepared_program
  case for it (tests/cases.py) is still to write.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  ttnn/ttnn/bringup/ckv_bfp4_ttnn/tests/unit/test_ckv_bfp4.py`; `... models/demos/deepseek_v41_flash_d_p/tt/perf_ckv_bfp4_host.py -s`;
  `python -m models.demos.common.bringup.orchestrator measure --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --task P.7`; old path
  DS41_CKV_BFP4_PREPARED=off.
- Attempt 3 (P.7): only docs fixed after attempt 2 failed the lint: ckv_bfp4_ttnn/README.md now has the `## Options`,
  `## Limits` and `## How to call` sections (ops_lint: 0 problems); DS41-135's fix line labelled `Fix:`. Code unchanged.

## P.8 (galaxy 8x4, perf): INDEXER_TOPK_LOCAL, the indexer top-k stays on the chip that scored it (option C)

- What: setting INDEXER_TOPK_LOCAL (tt/settings.py, default on; off = the old path). On, the all-device model's
  indexers (`TtDS41Block` sets `indexer.local = topk_local()`, tt/model.py) return topk_large_indices' own
  [1, 1, Sq, 512] rows (no `ccl.all_gather` over TP in `TtIndexer.topk_of`), and the sparse attention's index list
  (`TtSparseAttention._mine`, used by `_index` / `_index_static`) skips the `mesh_partition` back to the chip's rows
  when the top-k already has q_rows rows. That works because the indexer's chip rows (natural blocks r W + c Sq + i)
  are the attention's query rows after its head->seq reshard (`WindowTables.geo.query_offsets`).
- Sizes come from the mesh (`geo.q_rows`, `geo.w`, `mesh.shape`); TP == 1 boxes take neither op, as before.
- The state read-back (`DS41DeviceState._index_rows` -> `_ids_to_host(per_chip=True)`) joins the chips' rows in device
  order (chip r TP + c), ids as given (the ladder compares sets, testing/ladder.py:index_overlap).
- Component / swap harnesses (bringup/hooks.py, outside this step's paths) build their indexers with `local=False`, so
  they keep the gathered layout their readback (ConcatMesh2dToTensor, column 0) and the uploaded reader top-k expect.
  `_mine` takes both layouts.
- Calls removed per chunk (40 layers, 8 index sources 2/8/14/20/24/28/32/36): 8 all_gathers (one a source layer)
  and 8 mesh_partitions. That is 16 calls, not the ~46 the study estimated, because ChunkMemo (INDEXER_GLUE, P.16)
  already builds one index list per index source, not one per compressed layer (38). Collectives per layer: an
  index-source layer has 1 TP all_gather fewer; other layers are unchanged. DRAM: a kept top-k is [Sq, 512] uint32 a
  chip instead of [W, 512] (0.33 instead of 1.31 MB per index-source layer per kept chunk on 8x4).
- Accuracy (measure --task P.8, quick, switch on): every limit holds, and every metric equals P.7's (same ids, same
  bytes): L39 err 0.9072 / limit 1.533, err_state_max 0.1998, topk_idx overlaps L02 0.980 ... L36 0.921 (P.7 has the
  same values), host_transfers_per_layer 0. Warm wall 417.5 ms (spread 52.7; P.7 415.1, spread 25.6): any gain is
  inside the run-to-run spread. No profile was run in this step. The orchestrator's A/B profile gives device ms per
  section and op calls on / off.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.orchestrator measure --task P.8`; old path
  `DS41_INDEXER_TOPK_LOCAL=off`.

## P.9 (galaxy 8x4, perf): ATTN_SP32_PROJ, the attention projections on the chip's own 160 rows (option B)

- What: setting ATTN_SP32_PROJ (tt/settings.py, default on; off = the old path; takes effect with ROPE_IMPL=on). On,
  `TtWindowAttention` (and `TtSparseAttention`, which inherits `_q` / `_out`) uploads wq_b, wo_a, wo_b replicated
  (bfp8, `wdim`), sets h_l / g_l to all 64 heads / 8 groups, and per chunk: `mesh_partition(q_resid, dim 2, TP)` to the
  chip's S/(SP TP) rows, then `_q_stock` (scale x bfp8 cast, wq_b, nlp_create_q_heads_split, rope_stock) gives
  [1, 64, 160, 512] directly, so there is no head -> seq all_to_all. After sparse_sdpa: inverse rope_stock on the chip's rows, wo_a (8
  groups), wo_b full writing bf16 (act_dtype) [1, 1, 160, 5120], then ONE `all_to_all_async_generic(in_dim=2,
  out_dim=3)` over TP -> [1, 1, 640, 1280] (the residual layout). No seq -> head all_to_all, no wo_b reduce-scatter.
  cos / sin: `_rope_rows` -> `RopeTable.rows(tp_split=True)` (memoized, the indexer's rows). The chip's rows are
  `WindowGeometry.query_offsets` (chip c: rows [c q_rows, (c+1) q_rows) of its SP block), the rows sparse_sdpa's index
  lists already use. Sizes come from mesh.shape; with TP == 1 neither the partition nor the all_to_all runs.
- DS41-138: all_to_all_async_generic concatenates on in_dim and splits on out_dim (my first try, in 3 / out 2,
  failed validation).
- DS41-139: replicated weights make the 160-row matmuls DRAM-bound (4x the weight bytes a chip). tt/mm_configs.py
  rule: new entry before minimal_matmul for bf16 output, wide N, Mt <= SMALL_M_TILES (8): 1D mcast of in0, in0 block 8
  (`_mcast_1d`). Probe (one chip, 12x10 grid, HiFi2 fp32 DEST): wq_b 160x1280x32768 0.273 (minimal) -> 0.199
  ms, wo_b 160x8192x5120 0.285 -> 0.158, bit-identical to auto (rel 0). The indexer wq_b 160x1280x4096 also takes it:
  0.056 -> 0.058 ms (a wash; same bytes as auto). wo_a batched 8x160x4096x1024 stays on the rule's 2D fuse_batch=False
  bw32: 0.207 ms (2D bw16 0.223, auto 0.696). Other boxes: 4x4 has 320 rows (Mt 10) and keeps minimal_matmul.
- Accuracy (measure --task P.9, quick, on, final code): all 102 limits hold. L39 err 0.834 / 1.533 (P.8: 0.907),
  min layer PCC 0.659 (P.8: 0.545), err_state max 0.206, tightest err_state_window_kv_L28 0.148 / 0.175 (0.85),
  host_transfers_per_layer 0. The bf16 TP partial sum of wo_b is gone: wo_b sums all of K in fp32 DEST.
  Component tests test_c_swa_attention / test_c_enc_source_attention fail on the 8x4 with both settings (on 0.0171 /
  0.0604, off 0.0173 / 0.0603 vs limits 0.0033 / 0.0499): this change did not cause it; not in the P.9 gate.
- Device (plain profile, chunk [51200, 56320), same code, DS41_ENGRAM_TABLE=resident): total 221.8 off -> 209.9 on ms,
  attention 78.9 -> 67.7; other sections within 0.6 ms. Per layer (op profile L0 window / L2 compressed): 1.91 ->
  1.60 / 2.05 -> 1.81 ms. Removed: 2 all_to_all at 0.24-0.31 ms + reduce-scatter 0.16. Added: one all_to_all 0.14,
  matmuls 0.217 -> 0.555 ms a layer.
- Op calls a chunk (op profile, all 40 layers): 2373 -> 2333 (attention 1126 -> 1086): -40 fabric_reduce_scatter,
  -40 all_to_all, +40 mesh_partition; 80 minimal_matmul -> 80 linear. Programs 77472 -> 76192. The brief estimated ~85
  fewer; the real count is 40 because the q_resid mesh_partition is a new op.
- Collectives per layer (attention): off 3 fabric_all_gather + 2 all_to_all (10.5 MB each) + 1 fabric_reduce_scatter;
  on 3 fabric_all_gather + 1 all_to_all (1.6 MB a chip). Per chunk 240 -> 160.
- Host: the op profile's host_ms a chunk is 482.5 off -> 426.2 on (attention 219.8 -> 166.5). The fabric_reduce_scatter
  cost 1.25 ms host a call. Warm wall (quick measure, on): 387.5 (spread 24.7), then 400.5 (spread 54.4) on the final
  code; P.8's quick runs were 417.5 / 415.8. Gate unit test_profile (on): clean wall median 420.4 with host loadavg
  23.7 during the run. The orchestrator's A/B gives the paired wall.
- DRAM per chip (computed, bfp8 1088 B a 1024-value tile): full wq_b + wo_a + wo_b = 119.0 MiB a layer, before 29.75
  (1/4) -> +89.25 MiB a layer, +3.49 GiB for 40 layers. It loaded and ran on the 8x4 (32 GB a chip). 4x4 / 2x4 add
  the same; 2x2 adds half. Not checked against plan_memory.
- Gotcha (K-57): `tt-probe.sh --device auto` probes reset their card on exit, and the next 8x4 open then failed with
  "SAT placement: mesh 'M0' has no PGD grouping". A `--mesh` probe (whole-box reset) cleared it.
- Next (not done): wo_a and wq_b are still at ~170-220 GB/s. A DRAM-sharded matmul would need sharded inputs or
  outputs, which means more ops on the host-bound path.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.orchestrator measure --task P.9`; old path
  `DS41_ATTN_SP32_PROJ=off`. The matmul probe (one chip, `scripts/tt-probe.sh --device auto`, then a --mesh run per
  K-57) was not kept, because it is outside this step's paths. It timed mm._run with mm.rule's config against auto over 40 calls
  after a synchronize, on random bfp8 inputs.

## FIX.11 (overseer fix, attempt 1): attention precision vs the reference. Disagree, no code change

- Request: L00 attention 0.0171 vs limit 0.0033 ("~10x the reference's arithmetic") and L02 0.0604 vs 0.0499 with
  4-bit KV. STEP 1 attribute on the CPU, STEP 2 raise the dominant items to the reference's precision.
- Finding (DS41-140): the 0.0033 is 2 x one bf16 rounding of the output (0.00166). The galaxy goldens have no
  official_floor.json, so noise_floor.arith returns None and the reference's arithmetic is missing from the limit.
  Measured on the CPU on the galaxy golden input: the official reference at bf16 is 0.0327 from native on L00 (fp32:
  0.0323) and 0.0371 on L02 (without its fp4 KV, which the input nudge covers: 0.0249). The 2x2 taps give 0.045 / 0.074.
  So the device (0.0171 / 0.0604) is closer to native than the reference is.
- Attribution, L00 (table in known_issues_detail.md "DS41-140"): bfp8 GEMM inputs 0.0117 alone (0.0139 with the
  packer's double rounding, DS41-137); sparse_sdpa HiFi2 operand truncation 0.0074 alone, but nothing when the other
  items are on; bf16 activations 0.0037; bfp8 weights 0.0030; RoPE bf16 tables 0.0012; bf16 P 0.0006; bf16 sink
  0.00005. All items together 0.0129-0.0149, device 0.0171.
- Device diagnostic: ATTN_FIDELITY=HiFi4 gives L00 0.0162 (from 0.0171), bias +0.48 %. Not adopted: -5 % of the error
  for ~1 % of the chunk (DS41-68), and the result is still 5x the nudge-only limit.
- Nothing changed in tt/ or the forks. The items narrower than the reference (HiFi2 sdpa, bf16 RoPE tables, bf16 sink)
  are small. The dominant one is the reference's own fp8 in the device's bfp8 (F208), which can't be raised without
  going above the reference. bfp8_pack_precise on the wq_b / wo_a-output packers would need forks of the stock
  multiply / matmul (no compute-config field), and would remove ~0.002 and the +0.65 % bias.
- Next (orchestrator / owner): generate official_floor.json for the galaxy goldens (`python -m
  models.demos.common.bringup.testing.official_floor --spec <galaxy spec>`), so the component limits include the
  reference's arithmetic as rule 14 says.
- Re-run: the CPU attribution script `/scratch/dnijemcevic/fix11/attrib.py [layer]` (native step + official + item
  emulations on the component golden, CPU only). Device: `DS41_ATTN_FIDELITY=HiFi4 scripts/run_safe_pytest.sh
  --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_attention.py`. Log:
  /scratch/dnijemcevic/fix11/swa_hifi4.log.

## P.10 (galaxy 8x4, perf): ATTN_SP32, the rest of attention on the chip's own 160 rows (option A)

- What: setting ATTN_SP32 (tt/settings.py, default on; off = the P.9 path; takes effect only with ATTN_SP32_PROJ,
  ROPE_IMPL and NORM_IMPL on, `tt/common.py:attn_sp32`). On, `TtDS41Block` (tt/model.py) runs:
  - attn_norm: ONE `all_to_all_async_generic(in_dim=3, out_dim=2)` over TP (`tt/common.py:cols_to_rows`, the inverse
    of P.9's exit reshard, DS41-138) [S/SP, H/TP] -> [S/(SP TP), H] bf16, then a local whole-row norm
    (`build_norm(rows=True)`: split=False, whole weight replicated; no stats_out / gather / apply split).
  - q_a: `TtAttnInProj(full_rows=True)`: [wq_a | wkv] replicated (bfp8), ONE matmul over the whole K (fp32 DEST)
    writing bf16 (act_dtype); no tp_sum (RS + AG). q_norm / kv_norm read their windows as before, on 160 rows.
  - kv: rope with the chip's cos / sin rows (`RopeTable.rows(tp_split=True)`, the memo P.9's attention already makes).
  - attention: q_resid is already the chip's rows (no mesh_partition); the window KV is gathered over TP, then SP
    (`_window_rows`), the rest unchanged (`chunk_of(rows)` derives the chunk from either layout).
  - compressor (KV sources 2 / 8 / 14 / 20): `TtCompressor(full_rows=True)`: [wkv | wgate] replicated, the chip's 160
    rows' whole sum (fp32 out), then ONE all_gather over TP -> [S/SP, 1024 | 512] and the old path from there (pool,
    norm, rope, wk, k_norm, rope, SP gathers). Departure from the brief ("compressor on 160 rows with 32-way gathers"):
    ratio 2 on 8x4 pools 160 rows to 80 (not whole tiles, K-5; padding is OV-3), and the 160-row path needs two 32-way
    gathers (ckv, k = 4 collectives) where this takes 1 TP gather + the 2 SP gathers it had; the redundant pooling /
    norm / rope there is 4 layers of small ops (compressor section 2.46 -> 2.09 ms anyway).
  - indexer: `TtIndexer(full_rows=True)`: weights_proj replicated, the gates ONE local matmul writing bf16 (no RS);
    `_sub(q_resid)` returns q_resid itself (K-1: query() frees sub / qq only when they are not q_resid).
  Component / swap harnesses (bringup/hooks.py) build the modules with full_rows=False: their layout is unchanged.
  Every size comes from mesh.shape (TP == 1: cols_to_rows returns its input; 4x4: 320 rows a chip, 2x4 640).
- Precision: the q_a, compressor and gates TP sums were bf16 reduce-scatters of bf16 partials (CCL_IMPL=fabric_bf16);
  now each is one fp32-DEST matmul over the whole K (q_a / gates written bf16, compressor fp32). Not a raise above the
  reference (its fp8 GEMM sums in fp32 and writes bf16).
- Accuracy (quick measure, on): all limits hold; pcc_chunk_out 0.707 (P.9 0.659), L39 err 0.723 / 1.533 (P.9 0.834),
  err_state_max 0.2014 (P.9 0.2064), window_kv L28 0.142 / 0.175 (P.9 0.148), host_transfers_per_layer 0. Gate units
  passed on this code: test_contract, test_profile.
- Device (plain profile, gate unit test_profile, chunk [51200, 56320), DS41_ENGRAM_TABLE=resident; before = P.9's
  test_profile stamp on the committed code, i.e. the off path): total 209.94 -> 206.04 ms. q_a 9.41 -> 3.42,
  attn_norm 3.10 -> 4.43 (the all_to_all in), attention 67.69 -> 69.31 (+1 TP all_gather of kv), indexer 9.29 ->
  8.65, compressor 2.46 -> 2.09, kv 0.86 -> 0.75; other sections within 0.3 ms. The study's 6-8 ms was not reached:
  the new all_to_all (~0.03 ms a layer) and the kv TP gather (~0.04) eat part of the q_a RS + AG saving.
- Op calls a chunk (the profile's op table, 40 layers): 2333 -> 2189 (-144): q_a 200 -> 120 (-RS, -AG; minimal_matmul
  -> linear), attn_norm 120 -> 80 (2 rms_norm + AG -> a2a + 1 rms_norm), indexer 90 -> 74 (-8 RS, -8 mesh_partition),
  compressor 133 -> 125 (-4 RS, -4 typecast), attention 1086 -> 1086 (-40 mesh_partition, +40 all_gather).
  Programs 76192 -> 71584. Host ms of those ops 426.2 -> 354.7 (q_a 62.4 -> 17.8, compressor 42.2 -> 22.2).
- Collectives per layer (attention side): off 1 AG (norm stats) + q_a RS + AG + 3 AG + 1 a2a (attention) = 7; on
  1 a2a (in) + 4 AG + 1 a2a (attention) = 6. KV sources: RS + AG -> AG; index sources: -1 RS.
- Wall: quick measure 367.5 ms, then 384.7 on the final code (spread 33.9 / 67.8; same metrics to the bit; P.9's quick table: 393.1 on / 397.7 off); test_profile clean wall
  370.7 (loadavg 16.4) vs P.9's gate 420.4 (loadavg 23.7). host_overhead_ms 210.5 -> 164.7.
- DRAM per chip (computed): [wq_a | wkv] bfp8 9.30 MiB a layer whole vs 2.33 (+279 MiB / 40 layers); compressor
  [wkv | wgate] bf16 +7.5 MiB x 3 (ratio 2) + 3.75 (ratio 1); attn_norm fp32 weight tile rows +0.47 MiB x 40;
  weights_proj +0.23 MiB x 8: ~+0.32 GiB a chip (loaded and ran on the 8x4). Not checked against plan_memory.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.orchestrator measure --task P.10`; old path
  `DS41_ATTN_SP32=off`.

## FIX.12 (overseer fix, attempt 1): the 2x2's lowbit phase 1 at full size. No code change

- Request (OWNER 2026-10-29, was P.12): confirm on the 8x4 that the merged defaults (7f55821f579: sparse_sdpa
  ROW_BFP4 one DRAM read a 320 B record + two local copies; SFPU round |x| + 2^23 LSB - 2^23 LSB) leave outputs
  unchanged, and measure them against P.10. Opt-in options (ckv_bfp4 bfloat8_b / blocks, ROW_BFP8, bfp_round,
  indexer_score bfp4 q / k) left off. Nothing in tt/ or the forks changed: nothing broke at 8x4 sizes.
- Outputs unchanged: test_ladder (last) passed, and all 207 accuracy metrics of FIX.12's ladder units equal P.10's
  to the bit (err_layer_*, err_state_*, overlaps, err_final_hidden 0.1923, top5 1.0). test_ladder s56320,
  test_contract, test_kv_layout passed.
- Measured (gate unit P.10#test_profile, chunk [51200, 56320), DS41_ENGRAM_TABLE=resident, same 71584 programs;
  before = P.10's stamp df1aab02c2dab74b, after c0c693795d1c5b12):
  - sparse_sdpa, compressed layers (38 calls): 368.0 -> 308.9 us a call (-16 %; 13.98 -> 11.74 ms a chunk). The 2x2
    single card predicted 352 -> 293 us at 160 q (-17 %). Window-only layers (2 calls): 136.7 -> 135.7 us (no change, as
    expected: they read no 4-bit rows).
  - ckv_bfp4 (the 4-bit compressed-KV write, 4 calls): 52.3 -> 40.3 us a call (-23 %; the 2x2 said ~-30 %).
  - attention section 69.31 -> 67.35 ms; compressor 2.09 -> 2.04; device total 206.04 -> 203.81 ms (-1.1 %); other
    sections within 0.25 ms.
  - warm wall (clean, median of 10): 370.7 -> 381.1 ms (spread 161.3 / 57.3, loadavg 16.4 / 15.1); host_overhead_ms
    164.7 -> 177.3. The 2.2 ms device gain is under the wall's run-to-run noise; the host path is unchanged (same ops,
    same programs), so the wall change is noise, not the merge.
- Gate: FAIL on one unit, test_adapter_acks: `chunk 0: ack slot 0 [0, 2536) (last-window 128 rows): 0_window_kv layer
  20 nope pcc 0.98855 < 0.98957`. Not the merge (bit-identical above). DS41-141: the unit last passed at FIX.9; P.7-P.10
  ran quick A/Bs only. Bisection: DS41_ATTN_SP32=off 0.98828 FAIL; DS41_ATTN_SP32=off DS41_ATTN_SP32_PROJ=off PASS.
  So P.9 (ATTN_SP32_PROJ) tipped it. P.9 is not narrower anywhere (wo_b: one fp32-DEST sum then bf16, the reference's
  output dtype, vs 4 bf16 partials in a bf16 reduce-scatter). The ladder's window_kv L20 went 0.090 -> 0.106 (limit
  0.151) while err_layer_L19 stayed 0.167. That is DS41-125's gap (the limit is the ring drift at s10240 positions
  10112-10239, applied to positions 2408-2535) tipped by a rounding reshuffle, not a bug. Per rule 14 I stopped:
  no test, limit or precision changed. Owner: a waiver for this check, or a per-position drift for the contract's
  window-KV limit (golden generation). Then run one unit at a time: test_runner_contract FAIL on the same check
  through the tt-d-gen engine (`slot 0 turn 1 (last-window 128 rows): 0_window_kv layer 20 nope pcc 0.98855 <
  0.98957`, the same bytes as acks); test_runner_smoke PASS; test_profile PASS (device 203.81 <= 230, wall 381.1 < 434).
  Units status on this code: PASS ladder last, ladder s56320, contract, kv_layout, runner_smoke, profile; FAIL
  adapter_acks, runner_contract (both DS41-141).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.12 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`; bisection:
  `DS41_ATTN_SP32=off DS41_ATTN_SP32_PROJ=off scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/bringup/contract/test_adapter_acks.py`. sparse_sdpa per call:
  /scratch/dnijemcevic/fix12/sdpa_sum.py <stamp dir>/_ops.json.gz.

## FIX.13 (overseer fix, attempt 1): DS41-141, is P.9's L20 window-KV rise rounding or a bug? Rounding. No code change

- Request (OWNER, device allowed): find the first layer / op where ATTN_SP32_PROJ on vs off differs beyond rounding
  on the 2536-token prompt, padded partial chunk first.
- Probe: scripts/tt-probe.sh --mesh with layers 0-20 of the full model, the contract's call sequence (note_tokens,
  begin_chunk, embed, layer(i, h, 0, state, end)), and golden s10240 chunk 0 tokens. It dumps every layer's output
  rows [0, 2560) and its window KV. Runs: defaults with end=2536 (PAD_ID tail) and end=5120, and
  DS41_ATTN_SP32=off DS41_ATTN_SP32_PROJ=off with end=2536. About 6-7 min each. Kept in /scratch/dnijemcevic/fix13/
  (probe_dump.py, ana1-5.py); the tt-probe copy under tests/ttnn/ was removed (outside this step's paths).
- Reproduced: L20 nope PCC on the last 128 rows of [0, 2536) is 0.98855 on, 0.99131 off.
- Padding ruled out: defaults with a PAD tail (end 2536) vs real tokens (end 5120) give bit-identical valid rows and
  window KV in every layer 0-20.
- No localized excess. Per 160-row chip group and per stream x TP column quarter, on and off are equally far from the
  golden from L0 on (L0 out 0.0409 / 0.0411, L0 window KV 0.0048 / 0.0051; chip 15 = rows 2400-2535 is no outlier).
  The on-off divergence grows smoothly (0.011 at L0, 0.040 at L19) and evenly over the chips.
- The check samples a spread. Over 286 sliding 128-row windows at L20: on-off PCC mean -0.00005, std 0.0032. The
  failing window is the 15th percentile. Off itself is below 0.98957 in 191 of 286 windows.
- Verdict: honest rounding, DS41-125's limit-vs-position gap (DS41-141 updated, detail in known_issues_detail.md). The
  tests.unit_waivers entries stay for the owner. Nothing in tt/ or the forks changed. I also renamed DS41-125's detail
  heading to its bullet title (the knowledge check flagged it).
- Re-run: `FIX13_OUT=<dir> [FIX13_ENDS=2536] [DS41_ATTN_SP32=off DS41_ATTN_SP32_PROJ=off] scripts/tt-probe.sh --mesh
  fix13_dump < /scratch/dnijemcevic/fix13/probe_dump.py`, then `python3 /scratch/dnijemcevic/fix13/ana3.py` and
  `ana4.py`.

## FIX.14 (overseer fix, attempt 1): the 2x2's lowbit phase 2 at full size. Gate PASS, no code change

- Request (OWNER 2026-10-29 13:00): the merged defaults (df92663a019 via 1f45dcccf1c) are WINDOW_KV_FORMAT=bfp8_blocks,
  INDEX_K_FORMAT=bfp4 and INDEX_Q_FORMAT=bfp4. Run P.10's units on the 8x4, fix anything that breaks at 8x4 sizes,
  and measure on vs off.
- Nothing broke. The full gate passed on the merged code, with nothing changed in tt/ or the forks: ladder last,
  ladder s56320, contract, kv_layout, adapter_acks, runner_contract, runner_smoke, profile. SP 32 / 160 rows a chip,
  40 layers, the window carry (the contract's partial chunks 2536 / 6072 and follow-up starts) and the per-chip index
  slabs (3520 / 7040 rows) all run on the merged code.
- Accuracy (s56320, on vs off = FIX.12's metrics, the same path with the formats off): every metric is within its
  limit, worst ratio 0.76 (err_layer_L30 0.154 / 0.202, off 0.152). Top-k state per layer, on / off / limit:
  L02 0.137 / 0.025 / 0.284, L08 0.130 / 0.055 / 0.237, L14 0.130 / 0.086 / 0.240, L20 0.193 / 0.118 / 0.354,
  L24 0.281 / 0.131 / 0.548, L28 0.199 / 0.121 / 0.364, L32 0.188 / 0.119 / 0.346, L36 0.177 / 0.115 / 0.319.
  Candidates L20: 0.078 / 0.038 / 0.151. The 8x4 limits are wider than the 2x2's (their official drift is larger),
  so none of the waivers the 2x2 needed (indexer, state.topk_idx) is needed here. Rung last: the worst was topk L08
  0.155 / 0.172. Final hidden 0.196 / 0.259 (off 0.192). top5 1.0.
- DS41-141: test_adapter_acks and test_runner_contract now pass with 0 failures, so the L20 last-window check is
  >= 0.98957 and the unit waiver no longer matches. The exact PCC is not printed on a pass. The waiver was left as
  it is.
- Device per call (plain profile, chunk [51200, 56320); off = the same code with DS41_WINDOW_KV_FORMAT=bf16
  DS41_INDEX_K_FORMAT=bf16 DS41_INDEX_Q_FORMAT=bf16, the profile run by hand into /scratch/dnijemcevic/fix14/off):
  - sparse_sdpa: compressed layers 308.9 -> 306.5 us (-0.8 %; 2x2 -4 %); window-only (ROW_BFP8 vs stock bf16)
    136.1 -> 134.1 us.
  - ring_indexer_score_dsa: r2 (28160 keys) 272.5 -> 249.4 us (-8.5 %); r1 (56320) 494.9 -> 457.8 us (-7.5 %).
  - New writes: ckv_bfp4 window blocks 5120 rows 47.5 us + carry 128 rows 23.5 us a layer (80 calls, 2.84 ms a
    chunk); bfp_round query 10.8 us x 8, keys 8.8 / 15.3 us x 4 (0.13 ms); +40 carry to_layout (0.19 ms). Gone: the
    bf16 table's slice_writes (25.4 + 1.8 us x 38, 1.04 ms).
  - Sections: attention 67.22 -> 68.68 ms, indexer 8.66 -> 8.52, device total 203.80 -> 205.36 ms (+1.56; 2x2
    +0.09). Programs 71584 -> 73312.
  - Wall (clean, median of 10): 367.4 off (spread 274) vs 386.7 on (spread 59), loadavg about 26 for both;
    host_overhead_ms 163.6 -> 181.3. The new ops' host time is ~0.21 ms a ckv_bfp4 call (prepared) and 0.5 ms a
    bfp_round call; that adds ~14 ms a chunk, less the slice_writes' 10.8. DS41-142.
- Per-chip DRAM at 1,044,480 tokens (computed from the shapes the code allocates; the window does not depend on seq):
  - Window: off = AttnTable.win per KV source with a table (2, 8, 14, 20), 4 x [5248, 512] bf16 = 20.5 MiB (plus a
    transient 5.1 MiB table per window-only layer call); on = one shared block buffer 164 x 18496 B = 2.9 MiB.
  - Index ring key buffers (TtIndexer._kbuf, one per indexer: 3 at r2 = 522240 rows, 5 at r1 = 1044480 rows, x 128):
    bf16 127.5 / 255 MiB each, 1657.5 MiB in total; bfp4 35.9 / 71.7 MiB, 466.2 MiB. That saves 1.16 GiB a chip.
    The served index-key slabs stay bf16 (contract).
  - Ring traffic received per chip per call (7/8 of kv_len x 128): at 1M, r2 111.6 -> 31.4 MiB and r1 223.1 -> 62.8
    MiB; at 56320, r2 6.0 -> 1.7 and r1 12.0 -> 3.4 MiB (3.56x). Window gathers are unchanged: the window rows are
    still gathered in bf16 over TP and SP (4 all_gathers a layer); only the sdpa read is narrower (578 vs 1024 B a
    row).
- Served caches: the brief says "they must hold the rounded values attention reads". That holds for compressed KV
  only (ckv_bfp4 rounded=True, FIX.8). The window-KV and index-key served caches hold the unrounded bf16 rows. Within
  prefill this changes nothing: the bfp groups are 16 values of one row, so the carry and the key slab re-round to
  the same values. A decode reading the served bf16 would see values above the reference. DS41-143. I did not change
  it: that is a change to the contract's numbers, outside "anything that breaks".
- Perf ideas (not done): (1) a per-chip window region. Each chip's 160 queries read ~288 window rows, so this would
  remove the replicated 5248-row rounding and the window all_gathers. (2) One _kbuf per capacity shared by the
  indexers (they rewrite it every call). That would save 4 x 71.7 + 2 x 35.9 = 358 MiB a chip at bfp4.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.14 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`; off profile:
  `BRINGUP_RESULTS_DIR=<dir> BRINGUP_RECORD_TASK=FIX14off DS41_WINDOW_KV_FORMAT=bf16 DS41_INDEX_K_FORMAT=bf16
  DS41_INDEX_Q_FORMAT=bf16 DS41_ENGRAM_TABLE=resident BRINGUP_PROFILE_WALL=1 scripts/run_safe_pytest.sh --run-all
  --no-precompile models/demos/common/bringup/tests/test_profile.py`, then the same with BRINGUP_WALL_CLEAN=1;
  per-op compare: `python3 /scratch/dnijemcevic/fix14/ops_sum.py <on _ops.json.gz> <off _ops.json.gz>`.

## P.12 (perf, galaxy 8x4): LOWBIT_HOST, the host time of FIX.14's low-bit writes
- What: setting LOWBIT_HOST (tt/settings.py, default on; off = FIX.14's path, code unchanged). On: (a) the window
  blocks' carry and chunk in ONE ttnn.bringup.ckv_bfp4 call a layer (new fork option `prefix`: [carry | chunk] at row0
  0; tt/attention._window_rows / _put_blocks; `masked` now freed after that call); (b) that call and bfp_round as the
  fork's `compact` programs; (c) bfp_round into each indexer layer's persistent outputs (tt/indexer.TtIndexer._round,
  one buffer per (query | keys, shape), allocated at the first call; memoized keys held as `Kept`, which ChunkMemo's
  drop leaves alone).
- (1) bfp_round did take the prepared path (patch count right, no generic_op fallback). Its extra cost over ckv_bfp4 was
  the per-call output allocation (34 us isolated) and 120 cores of per-core args (the window carry call has 4 cores).
- Root cause (DS41-145): a dispatched program's host cost on 32 chips is set by its per-core runtime args. Isolated
  (tt/perf_lowbit_host.py, 8x4): run_prepared alone ~150 us at 120 cores; the same round program with every arg common
  46 us flat at 1-120 cores; with one per-core value slot left 105 us; merged core ranges alone -40 us. So `compact` =
  merged ranges + common args only (the kernels split bands / tiles by their logical core index, kernels/ckv_share.hpp,
  the same split) + row0 / page0 frozen in the program (in the prepared key; every LOWBIT call site has one row0). I did
  not extend prepared_program with common-arg value slots (C++ rebuild, its tests are 2x2-only) because no call site needs it.
- Isolated host a call: window blocks 160 + 60 -> 58 us a layer; bfp_round query 207 -> 61 us, keys 213 -> 60 us
  (compact into a reused output).
- In the model (cProfile of 5 warm chunks at position 0, overseer hostprof plugin, /scratch/dnijemcevic/p12/hostprof.sh):
  _put_blocks 0.207 ms x 80 -> 0.123 ms x 40 a chunk (16.6 -> 4.9 ms), bfp_round 0.419 -> 0.182 ms x 12 (5.0 -> 2.2 ms),
  _window_rows 54.7 -> 37.5 ms a chunk; plain wall at position 0 (median of 10): off 373.6 (spread 74) -> on 328.5
  (spread 46).
- Profile rung (s56320 last chunk, DS41_ENGRAM_TABLE=resident, change on): device_ms_total 204.63 / 204.75 (two
  runs; FIX.14 on = this task's off: 205.36), attention 68.09 / 67.96 (68.68), programs 73312 -> 72032 (the 40 carry
  calls x 32 chips gone); clean wall median 377.4 ms (spread 68.8, loadavg 22.6; FIX.14 386.7), so host_overhead
  ~172.7 vs 181.3 ms. The wall change is within the run spread; the orchestrator's A/B has both sides.
- Bit-identical: tt/perf_lowbit_identity.py (one process, profile rung's last chunk, golden prefix reloaded each run):
  final hidden on == off == on (max |diff| 0, PCC vs golden 0.608661 all three). Fork tests/unit 51 passed (35 before,
  unchanged; 16 new: test_ckv_bfp8_prefix_compact, test_bfp_round_compact). Block pages' gap bytes are never written
  (stale L1), so whole pages differ between programs; compare masked (DS41-144).
- measure --task P.12 (the quick on side) did not finish: it needs ~10 min of device time (P.10's 600 s) and a
  foreground call is capped at 10 min. My timeout killed it at 587 s and the safe runner marked the cards dirty. The
  orchestrator's A/B runs it.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_lowbit_host.py -s` (host a call + byte checks, 1 min);
  `... tt/perf_lowbit_identity.py -s` (on / off / on, 6 min); `... ttnn/ttnn/bringup/ckv_bfp4_ttnn/tests/unit`;
  old path DS41_LOWBIT_HOST=off.

## P.4 (perf, galaxy 8x4, attempt 3): CCL_PERSISTENT_OUT, persistent outputs for the in-step fabric all_gathers

- Re-check (owner): after P.8-P.10 (SP=32 attention) and the block MoE's fused head / tail, the Python CCL path has
  NO reduce-scatter left in a chunk (sum_cols / sum_rows / q_a tp_sum gone; the MoE's collectives are inside the fused
  head / tail programs). All_gathers: 294 a chunk on the profile rung's last chunk, 212 of them in-step sites: mHC row
  exchange (`tt/mhc.py fused`, 80), window KV over TP and SP (`tt/attention.py _window_rows`, 80), split norm stats
  (`tt/norm.py _fork`, 40), compressor TP rows (4) and SP gathers (8). The other 82 (kept out: tp_sum's _gather_add,
  results that leave the step) allocate as before.
- What: setting CCL_PERSISTENT_OUT (tt/settings.py, default on; off = the path before). `ccl.all_gather(site=...)`
  writes `_site_out(mesh, site, shape, dtype, layout)` (allocated at the site's first call, kept) and returns that very
  object; `ccl.dealloc` returns on `id(t) in _KEEP` (one set lookup), so the step code is unchanged. Only for a gather
  over > 1 chip (K-1: a 1-chip axis would let full-range views share the buffer). Reuse is safe: each site's value is
  used and freed inside its step, each chip runs its programs in order, and fabric_all_gather's ready fence keeps a
  chip from writing a neighbour's output before that neighbour started the same call. 10 buffers a mesh (~7 MiB a chip).
- Gotcha (DS41-146): allocated lazily mid-forward, the persistent buffers left holes in the transient region; the block
  MoE's fused tail (moe_block/tt/tail_a2a.py TailA2A, ONE instance for all 40 layers) keys its prepared programs by the
  (y, se, out) addresses in a 16-entry FIFO, so the shifted layout made it rebuild: in-process A/B 612 ms on vs 426 off,
  tail _program 188 vs 22 ms a chunk. Fix here: `ccl.settle(mesh)` in `TtDS41Model.begin_chunk` (also the serving
  adapter's path) frees and re-allocates a mesh's site outputs once after the chunk that created them (steady-state
  tail misses 0-2 a chunk, as off). Still fragile: in tt/perf_ccl_persistent.py, which reloads the golden prefix every
  run, "on" falls into 19-23 tail misses every other run (672 / 676 ms runs), "off" does not. The real fix is outside
  this step's paths: TailA2A keyed on shape with its addresses patched (OV-23's prepared rule).
- Measured (8x4, profile rung's last chunk [51200, 56320)):
  - Bit-identical and deterministic: final hidden on == off (max |diff| 0), every run of a side equal, PCC vs golden
    0.608661 both (tt/perf_ccl_persistent.py).
  - Host (cProfile, one run a side after a warm block): ccl.all_gather 58.4 -> 48.1 ms (294 calls), ccl.dealloc 41.1 ->
    38.1 ms (2338 calls): ~13 ms a chunk less under cProfile (the proposal's estimate for 474 sites was 22; 212 remain).
    model.layer cum 493.8 -> 467.3 ms (that run of "on" had 2 tail misses, off 1).
  - Wall: quick measure (on, orchestrator, 10 clean runs) 414.8 ms, spread 44.7, loadavg 15.8; P.12's quick on (= this
    task's off code) 379.6 ms at loadavg 26.2, its off 392.8. In-process blocks: off median 437.5, on 471.9 (the two
    tail-miss runs), on without them ~420-443. The saving (~10 ms) is under the wall's run-to-run noise; the
    orchestrator's A/B decides. Device time: same programs and op count (no device change expected; not profiled here).
  - Accuracy (measure --task P.4, on): every one of 102 limited metrics within its limit, host_transfers_per_layer 0.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.orchestrator measure --task P.4 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml`; A/B + identity:
  `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_ccl_persistent.py -s` (7 min); old path `DS41_CCL_PERSISTENT_OUT=off`.

## P.5 (perf, galaxy 8x4): ROPE_UNIFORM_PROGRAM, one RoPE program for the mesh instead of 32

- Re-check (owner): after P.8-P.10 the target still exists unchanged in count. P.12's op table (the profile rung's last
  chunk, 40 layers): 136 rotary_embedding_indexed a chunk, every one 32 programs: attention q with nope prefix
  [64, 160, 64] (40), attention o inverse [64, 160, 512] (40), kv [160, 512] (40), compressor ckv / k on 320 / 640 rows
  (8), indexer q [32, 160, 64] (8). The shapes are now the chip's 160 rows (SP=32), and every call still passes
  per-chip pre-sliced cos / sin with kv_actual_global 0. The reader's update_idxt
  (`reader_rotary_embedding_indexed_interleaved_start_id.cpp`) is then 0 for any my_sp_coord, so the 32 programs are
  identical. That is verified both in the code and on the device (below).
- What: forked the op (fork_op.py) to `ttnn.bringup.rotary_embedding_indexed`, adding the option `uniform` (scalar form,
  default False = the stock per-coordinate programs). True builds ONE program per mesh coordinate range in
  create_mesh_workload. It is hashed, and refused (miss and hit) with metadata, kv_actual_global != 0, or
  seq_subshard_axis. Setting ROPE_UNIFORM_PROGRAM (tt/settings.py, default on; off = the stock op): `tt/rope.rope_stock`
  calls the fork with uniform=True, so all 136 sites change. Kernels unchanged; no device change expected.
- OV-23: this is a C++ device op (program cache, addresses patched on a hit), not a Python-built program; nothing
  non-prepared was added. OV-3: no padding.
- Fork tests: tests/unit/test_uniform.py on 8x4: 8 passed. Uniform output is byte-identical to the stock op on every
  chip for the 6 model shapes, on a miss and on a hit, and uniform=False matches the stock op too. A torch Meta-rotation
  check and the refusals pass. A corrupted chip-5 output fails the test, checked once by hand. Source tests
  (tests/source.yaml): the 8x4 cases (multi-iteration prefill 8, metadata-vs-scalar torus-xy 8x4) plus 9
  single-device cases, 18/18 on original and fork. The 2x2 / 2x4 upstream cases were not carried from the Galaxy.
- Host a call (test_uniform_host_time, warm, 8x4): stock 101.0 us -> uniform 52.0 us (DS41-147). That is ~6.7 ms a
  chunk for 136 calls in isolation.
- In the model (tt/perf_rope_uniform.py: one process, the profile rung's last chunk [51200, 56320), golden prefix
  reloaded each run, 6 alternating on / off pairs after a warm pair, wall = begin_chunk .. sync): **on median 399.7 ms
  (392.4-417.1), off 438.6 ms (409.8-514.6)**. On was faster in every pair, by 10.7-97.5 ms. The final hidden state
  was bit-identical across all 14 runs, on and off (determinism and on == off), PCC vs golden 0.608661 both (= P.4 /
  P.12). The saving is larger than the 6.7 ms isolated estimate. A likely cause is that 32 distinct programs a call
  also cost dispatch-side work in a full queue (not measured separately).
- measure --task P.5 (quick, on, final code, stamp f846e68ba484dac3). The first try was killed by the 10-minute
  foreground cap at 595 s (cold kernel cache). The second try finished: all 102 limited metrics within their limits;
  the tightest are err_state_window_kv_L28 0.907 of its limit, topk_idx_L08 0.902 and window_kv_L27 0.889 (the same
  as P.12 / P.4, since the bytes are identical). pcc_chunk_out 0.608661, host_transfers_per_layer 0, chunk_wall_ms
  408.6 (spread 24.5, loadavg 17.3; P.4's quick on 414.8 at 15.8, P.12's 379.6 at 26.2). Quick walls across tasks
  are not paired; the in-process A/B above is.
- Not measured here: device time (the gate's plain profile is two ~550 s processes). It should be unchanged: the same
  kernels and the same per-chip work. The orchestrator's A/B and gate measure it.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_rope_uniform.py -s` (9 min); fork: `... --mesh
  ttnn/ttnn/bringup/rotary_embedding_indexed/tests/unit/test_uniform.py -k 8x4 -s`; `python -m
  models.demos.common.bringup.testing.fork_source --fork rotary_embedding_indexed`; old path
  `DS41_ROPE_UNIFORM_PROGRAM=off`.

## P.11 (perf, galaxy 8x4): CKV_PACK_BEFORE_GATHER, each chip packs its own KV rows before the gather

- What: setting CKV_PACK_BEFORE_GATHER (tt/settings.py, default on; off = the path before, code unchanged). On, at
  chunk-aligned starts (start % S == 0):
  - Compressed KV (CKV_FORMAT=bfp4_rows; tt/compressor.TtCompressor._write_packed). Each SP row's own compressed rows
    [W / r, 512] go through the pad mask (per-SP-row load-time iota, `WindowTables.mask_local`). ttnn.bringup.ckv_bfp4
    (compact, prepared, rounded=True) packs them into a load-time local record buffer. The rounded bf16 rows go to the
    served cache locally (`TtServedRows.write_local`: no gathered rows, no server-order embedding). The 320 B records
    are all_gathered over SP into a load-time output and slice_written into the attention table
    (`AttnTable.put_records`).
  - Window KV (WINDOW_KV_FORMAT=bfp8_blocks; tt/attention.TtWindowAttention._window_packed, also from
    sparse_attention._static). The kv rows are gathered over TP in bf16, as before: the served write needs the SP row's
    [W, 512] slab. Then the pad mask, the served bf16 write done locally (`TtWindowCache.write_local`), ckv_bfp4 of the
    SP row's 20 blocks, the blocks' all_gather over SP, and a slice_write into the window blocks at block 4.
  - Window carry. A whole previous chunk leaves its last 4 blocks, copied (slice with output_tensor) into a per-cache
    load-time buffer (`keep_blocks`), and the next chunk slice_writes them. Otherwise, when the carry is one owner stripe
    (`carry_owner`), each chip packs the same 128 local served rows into 4 blocks, the blocks are gathered over TP and SP
    (`gather_carry`) and the owner's 4 are kept. Else the bf16 `carry` rows are packed.
- Why only aligned starts (DS41-148): the served order of an SP row is its own compute block only when start % S == 0.
  Other starts (prefix-hit follow-ups) run the off path, so serving gets nothing from this at those starts.
- Bytes: same table bytes. tt/perf_pack_gather.py, 8x4, every chip: window blocks (defined bytes, DS41-144) and
  compressed records equal the old path and the host rule, rounded rows equal, partial chunks too. In the whole model
  (tt/perf_pack_identity.py: the profile rung's prefix, two chunks, the second with the kept-blocks carry, on / off
  alternating): hidden state after each chunk bit-identical on == off, and deterministic.
- Gather bytes received a chip a chunk (8x4, S 5120):
  - window SP gather: 4.59 -> 2.59 MB a layer (183.5 -> 103.6 MB a chunk, 40 layers);
  - compressed KV: r2 2.29 -> 0.72 MB (3 sources), r1 4.59 -> 1.43 MB (1), 11.5 -> 3.6 MB a chunk;
  - carry from the served cache: 4.06 -> 2.30 MB a layer.
- Profile rung, change on, plain profile (DS41_ENGRAM_TABLE=resident; scratch results
  /scratch/dnijemcevic/p11/results/P11on*):
  - device 202 ms a chunk (P.12's table: 204.6), attention section 66.0 (68.0), compressor 1.8;
  - op sums vs P.12: all gathers in attention + compressor 14.46 -> 12.23 ms. Window SP gather 4.42 -> 3.40 ms. Carry
    gathers 6.41 -> 5.37 ms. ckv_bfp4 1.91 -> 1.91 ms: own rows 24 us + carry 23 us a layer vs one 47.5 us call.
    to_layout + embedding 1.82 -> 0.39 ms. New: slice_writes + tail slice 1.12 ms. Compressor ops 2.17 -> 1.94 ms;
  - op calls a chunk 2203 -> 2237 (attention +40: the carry's separate pack), programs 72032 -> 74400;
  - profiled wall 407.6 ms (not the clean wall).
- The profile chunk always takes the carry-from-cache path: its prefix is loaded from the host, so there is no kept
  tail. Consecutive chunks (serving, a long prompt) take the kept blocks: one slice_write a layer, no carry gathers.
- Window layer write sequence (tt/perf_pack_gather.py::test_window_layer, median of 20, ms a layer, old -> new):
  - consecutive chunks: synced 1.007 -> 0.914, host 0.842 -> 0.698;
  - carry from the cache: synced 1.530 -> 1.460, host 1.356 -> 1.252.
  - Compressed rows (pack + gather + table write, synced): n 320 0.777 -> 0.468, n 640 0.809 -> 0.679.
- Gotchas:
  - DS41-149: the fabric gather returns a new wrapper of a caller's output, so an `is` check freed it. Compare
    buffer addresses.
  - DS41-146: a kept tail allocated mid-forward (a plain ttnn.slice) made "on" second chunks 399-852 ms vs 339-371
    off. A load-time buffer written with slice(output_tensor=) fixed it (315-332 ms).
  - In-process on/off alternation leaves outliers right after each switch on both sides (address-keyed program
    caches), so its walls (on 324 / off 436 median, n 2) decide nothing. The orchestrator's separate-process A/B does.
- OV-23: ckv_bfp4 (Python ProgramDescriptor) is called prepared (CKV_BFP4_PREPARED=on) and compact at row0 0, so one
  prepared program per shape. Fork test added: test_ckv_bfp4_compact_rounded (compact + rounded on bfloat4_b rows, the
  new call); fork tests/unit 55 passed (51 before). No non-prepared Python op added. OV-3: no padding. Every box: the
  buffers and masks are sized from mesh.shape and the chunk (W, W / 2), the carry owner path needs a 128-row stripe
  (else the bf16 carry), and SP = 1 keeps the off path.
- Not finished here: `measure --task P.11` (quick, on) hit the 10-minute foreground cap at 597 s. It was in the clean
  wall's settle runs: 330-365 ms, run 1 396 ms. The orchestrator runs it. Accuracy: the bytes and the hidden state are
  identical to off (= P.5's code, whose measure passed all 102 limited metrics).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_pack_gather.py -s` (bytes + timings, 1 min); `...
  tt/perf_pack_identity.py -s` (on / off identity, 8 min, DS41_ENGRAM_TABLE=resident); fork: `scripts/run_safe_pytest.sh
  --device auto ttnn/ttnn/bringup/ckv_bfp4_ttnn/tests/unit`; old path DS41_CKV_PACK_BEFORE_GATHER=off.

## P.13 (perf, galaxy 8x4): INDEX_TOPK_OP, the indexers' top-k on the 2x2's ttnn.bringup.topk_large_indices fork

- What: no model code change. The merge 0b9fd49e26b (2x2 @ 9e5e0556b80) already has the setting INDEX_TOPK_OP
  (tt/settings.py, default fork; stock = ttnn.experimental.topk_large_indices, the old path) and every indexer top-k
  goes through `tt/indexer.py:topk_large_indices`. That covers the index top-k 512 (TtIndexer.topk_of: r2 layers
  2 / 8 / 14, r1 layers 20 / 24-36) and the candidate blocks' k 2048 (TtCandidateIndexer._candidates_glue). The
  fork needed no change at 8x4 sizes (ttnn rebuilt 01:56 with the merge).
- stock ttnn.experimental.topk_large_indices instead of ttnn.bringup.topk_large_indices: only with
  INDEX_TOPK_OP=stock (the A/B's old path), and in the fork's comparison test.
- 8x4 per-chip calls (P.12's op table, ladder rung last, chunk [51200, 56320), 160 query rows a chip):
  - r2: 160 x 28160, k 512, 3 calls a chunk;
  - candidates: 160 x 7040, k 2048, 1 call;
  - r1: 160 x 56320, k 512, 5 calls.
- Device time a call, one Galaxy p150b (tt/perf_topk_op.py under --profile, median of 5 after a warm call), stock ->
  fork:
  - r2 244 -> 110 us, candidates 76 -> 51 us, r1 486 -> 219 us (stock matches P.12's in-model 244 / 76 / 486).
  - The same calls at 1M: r2 4491 -> 1901 us (522240 keys), candidates 1098 -> 581 us (130560), r1 9240 -> 3909 us
    (1044480).
- Expected a chunk (40 layers): the top-k ops 3.24 -> 1.48 ms (-1.76 ms). The indexer section was 8.51 ms in P.12, so
  it should come out ~6.7 ms. At 1M: ~34.9 ms less device a chunk. Host: a C++ op either way. In P.12 its host time
  was 60-107 us a call (stock); fork host time was not measured separately. Not profiled in-model here (a plain
  profile is ~600 s, the shell cap); the orchestrator's A/B profile gives the section and device totals on / off.
- Selection vs stock (fork tests/unit/test_stock_identity.py, 31 cases: 8x4 shapes to 1M keys plus the 2x2's; inputs
  randn, 16-level tied, causal -inf tails, re-index -inf blocks):
  - The values are equal position by position, sentinels included, and both ops match torch's top-k values.
  - Ids differ only between tied bf16 keys (DS41-170): every k 512 row and the 40 pair-split rows of a k 2048 call.
    Neither op has a tie rule, so "ties as the stock op orders them" would mean reproducing the stock op's
    sorting-network order. That means the stock op's row split, the thing the speedup removes. Not done; explained
    instead. Downstream reads are sets (sparse attention, scatter of candidates; K-42).
- Fork tests on one Galaxy card: tests/unit 217 / 218. The one other is upstream's production perf band (prefill
  0.848 ms, band 0.887-0.905: faster). test_stock_identity 31 / 31. The model cases (tests/test_topk_large_indices.py)
  need a 2x2 mesh with FABRIC_2D: not run here. No new 8x4 cases.py entries: those need a fork_calls capture (optests).
- Accuracy, measure --task P.13 (quick, change on). Attempt 1 was cut by the 10-minute shell cap at 596 s, after its
  layer metrics:
  - 62 limited metrics recorded, every one within its limit;
  - the tightest are err_state_topk_idx_L08 0.1546 / 0.1716 (P.11 0.1548) and err_layer_L07 0.1186 / 0.1466;
  - err_layer_L39 0.875 / 1.533 (P.11 0.867); pcc_chunk_out 0.578 (P.11 0.609: tie flips, DS41-170).
  - Not reached: the state / final metrics and host_transfers_per_layer.
- Wall (warm, DS41_ENGRAM_TABLE=resident, change on): attempt 1 367.2 ms (spread 86.5, loadavg 29.3); attempt 2
  351.3 ms (spread 34.2, loadavg 19.1). Attempt 2 was also cut at 596 s, before accuracy, and its P.13.json
  overwrote attempt 1's metrics. P.11's quick on was 347.6 ms (spread 242). The orchestrator's measure gives the full
  on / off table.
- OV-23: topk_large_indices is a C++ device op (program cache), not a Python-built program: nothing non-prepared was
  added or touched. OV-3: nothing padded. The candidate path's -inf pad to k when fewer blocks than k are visible is
  the old path, unchanged.
- Re-run:
  - accuracy / wall: `PYTHONPATH=$PWD python -m models.demos.common.bringup.orchestrator measure --task P.13`;
  - per-call timing: `scripts/run_safe_pytest.sh --device auto --profile --no-precompile
    models/demos/deepseek_v41_flash_d_p/tt/perf_topk_op.py`;
  - stock comparison: `scripts/run_safe_pytest.sh --device auto --no-precompile
    ttnn/ttnn/bringup/topk_large_indices/tests/unit/test_stock_identity.py -s`;
  - old path: `DS41_INDEX_TOPK_OP=stock`.

## P.14 (perf, galaxy 8x4): PREPARED_ALL, every Python-built op as a prepared program (OV-23)

- Census (8x4, the profile rung's last chunk [51200, 56320), DS41_ENGRAM_TABLE=resident, 3 warm chunks after
  profile.warm_up; wrappers on ttnn.generic_op / ttnn.bringup.prepare_program / run_prepared and FastOperation by name,
  plugin /scratch/dnijemcevic/p14/plug/p14census_plugin.py, logs census_base.log (before) / census_on.log). DS41-171.
  Python-built (ProgramDescriptor) programs, all already prepared; host a call = run_prepared (C++) alone:

  | op (call site) | calls / chunk | host us / call | ms / chunk | prepared? | keyed on |
  |---|---|---|---|---|---|
  | fused head (moe_block fused_head._run) | 40 | 729-741 | 29.2-29.7 | yes (P.2-era, shared program, weights patched) | arena base + io addresses (skey) |
  | ckv_bfp4 window blocks (attention._window_packed) | 80 | 97-101 | 7.8-8.1 | yes (P.7 / P.12 compact) | shapes |
  | fused tail (TailA2A.run) | 40 | 157 off / 146 on | 6.3 / 5.8 | yes (P.2) | off: y / se / out ADDRESSES; on: shapes |
  | sparse_all_gather_2d (K3 exchange dispatch) | 40 | 142-151 | 5.7-6.0 | yes (prepared_program.generic_op, 512 by descriptor id) | addresses (256 LRU) |
  | bfp_round (indexer._round) | 12 | 160-163 | 1.9 | yes (P.12 compact) | shapes |
  | ckv_bfp4 compressor (append_tiles / _write_packed) | 4 + 4 | 487 / 164 | 1.95 / 0.65 | yes (P.7 / P.11) | shapes |
  | ttnn.generic_op (any site) | 0 | - | - | - | - |
  | fabric_reduce_scatter | 0 (none left after P.8-P.10) | - | - | yes (P.1) | addresses |

  The brief's other candidates are C++ device ops (nanobind, TTNN program cache: addresses patched on a hit, nothing
  to prepare): fabric_all_gather 294 x 120 us (35.2 ms), flat_routed_expert_with_program 40 x 487 (19.5),
  update_padded_kv_cache 168 x 114 (19.1), rms_norm 208 x 63 (13.2), mhc_post 82 x 136 (11.2), rotary_embedding_indexed
  136 x 70 (9.5), mhc_pre_xing 80 x 113 (9.0), topk_large_indices 9 x 405, ring_indexer_score_dsa 8 x 291,
  mhc_pre_xing_pack 3 x 106, sparse_sdpa 40 x 56. moe_grouped_topk and the combine_rf ops (except the exchange's
  dispatch and the fused head / tail) are not called on the block MoE path. Also stock: allocate_tensor_on_device 372 x
  83 us (30.7 ms), slice 219 x 91 (19.9), all_to_all_async_generic 80 x 170 (13.6), to_layout 176 x 64 (11.3), linear
  144 x 77 (11.1). ~2950 op calls a chunk in all.
- What changed: setting PREPARED_ALL (tt/settings.py, default on; off = the path before). On, the block MoE's fused
  tail (TailA2A, one instance for 40 layers) runs tt/prepared_tail.ShapeKeyedTailA2A: one descriptor built and
  prepared per (y, se, out) shape / dtype / layout / memory config, run_prepared patches y / se / out per call (z / zrw
  are the instance's). Guard: patch_count must equal the address slots the descriptor holds (workers 4 + se, ports 3
  a chip), else that shape takes TailA2A's own path (fallbacks: 0 seen). moe_block/ is outside this step's paths, so
  tt/moe_block_adapter.block() swaps the class of the built block's TailA2A instance (`prepared_tail.install`), after
  the block's own warm-up; tail_a2a.py is unchanged and is the off path. Device program and arguments are the same
  (same hash: patched slots are out of the key), so no device change.
- Not done: persistent CCL outputs (P.4, reverted by the owner; its code is not in the tree). P.4 measured ~10-13 ms
  host a chunk under cProfile and a wall change within noise; with the tail shape-keyed its DS41-146 regression should
  be gone, so it can be retried as its own pick. Not done either (outside the paths, behaviour change of other models'
  code): the fused head's skey and sparse_all_gather_2d's descriptor cache still key on addresses (both hit in steady
  state: 0 prepares a warm chunk).
- Evidence (tt/perf_prepared_all.py, one process, golden prefix reloaded each run, 16 runs): final hidden state
  bit-identical on == off on every run (PCC vs golden 0.578422 = P.13). A DRAM buffer allocated after the middle layer
  and kept (1.25-3.75 MB a chip): on 0 tail builds every run; off 1-2 builds a shifted run (+21-105 ms in those runs).
  Unshifted warm walls: on 337-351 ms, off 336-395 ms (medians of the last 3: 349.6 / 346.2) - the same within noise.
- measure --task P.14 (quick, on): PASSED inside pytest, "every layer and state within its limit"; every err / pcc
  metric equal to P.13's quick on to the last digit (205 err metrics, pcc_chunk_out 0.578422, err_layer_L39 0.875);
  host_transfers_per_layer 0; chunk_wall_ms 350.4 (spread 72.3, loadavg 19.6; P.13's A/B: on 343.5, off 346.3). The
  process was killed by the 10-minute shell cap in its teardown (596 s; the safe runner marked the cards dirty, they
  are reset at the next run). Device time / host_overhead not profiled here (no device change by construction); the
  orchestrator's A/B profile gives both sides.
- OV-3: nothing padded. OV-23: no Python-built op is dispatched outside the prepared path (census above).
- Re-run: census `/scratch/dnijemcevic/p14/census.sh <tag> DS41_PREPARED_ALL=on|off` (8 min); identity
  `PYTHONPATH=$PWD DS41_ENGRAM_TABLE=resident scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_prepared_all.py -s` (8 min); `python -m
  models.demos.common.bringup.orchestrator measure --task P.14`; old path DS41_PREPARED_ALL=off.

## P.15 (perf, galaxy 8x4): HOST_ALLOC, fewer per-chunk allocations (load-time outputs), the same bytes

- What: setting HOST_ALLOC (tt/settings.py, default on; off = the path before). New tt/host_alloc.py. Every buffer is
  allocated AT LOAD TIME (`reserve_sites` at the end of TtDS41Model's constructor, `reserve_tail_out` in
  TtMoEBlockLayer's): never mid-forward (DS41-146), no between-chunk settle. A call whose spec has no load-time buffer
  allocates as before (counted in ccl.SITE_MISSES: 0 on 8x4). The buffers are pinned by id (`ccl.pin`; ccl.dealloc
  returns on `id(t) in _KEEP`, one set lookup). Call sites hand back the kept object, never the op's new wrapper
  (DS41-149).
  1. Gathers (P.4 retried): `ccl.all_gather(site=)` for the mHC row exchange (`mhc.row` [1, 4, 640, 32] fp32, 80 a
     chunk), the split-norm stats (`norm.stats` [1, 1, 640, 256] fp32, 40; width from the fork's split_plan rule,
     host_alloc.norm_stats_width) and the attention window kv over TP (`attn.kv_tp` [1, 1, 640, 512] bf16, 40). The
     window blocks' SP gathers were already persistent (P.11). Not done: the compressor's 8 gathers (554 / 634) and
     the engram's 2 (other shapes per ratio; 0.25 ms a chunk).
  2. Block MoE (moe_block is outside the paths; installed on the built block's instances, `host_alloc.install_moe`):
     the exchange's `arena_fits` probe (allocate + free of the L1 arena, 181 + 18 us a layer) is kept as the arena its
     call allocates next (`_alloc_arena_kept`: TtK3TokenSplitMoE.alloc_arena's body with the probe); the fused tail
     (tt/prepared_tail.ShapeKeyedTailA2A) writes the load-time [1, 1, 640, 1024] bf16 output.
  3. Attention intermediates (ATTN_SP32 on, 160 query rows): `q_a.xq` (bfp8 typecast), `q_a.p` (projection; the kv
     step frees it), `attn.xq` (multiply), `attn.qb` (wq_b linear), `attn.oa` (bfp8 typecast), `attn.wo_a` (matmul):
     output_tensor / optional_output_tensor (mm.linear / mm.matmul / common.cast take `out`; minimal_matmul has none:
     allocates). One buffer per site shared by all 40 layers: each is read and freed inside its step (queue order on
     each chip makes the reuse safe). Shapes from a block's weights.
  4. Slices: the window carry from the served cache (`attn.carry_rows`, `attn.carry_mine`) slice into load-time
     outputs. No slice count change (153 a chunk) and no per-call host change measured in the model (slice_rows
     113.6 -> 122.9 us, the carry's owner slice 79.8 -> 79.2: noise; isolated interleaved TILE slice 171 -> 132 us).
     Not done: folding a slice into its consumer (slice_write takes a whole input; sparse_sdpa reads one blocks
     tensor), keep_blocks (already output_tensor), engram / compressor slices (28 a chunk, 2-6 layers).
- Not done, item (2) of the brief (fused head / sparse_all_gather_2d keyed on shape): both hit their address-keyed
  prepared programs every warm call (P.14 census: 0 builds a warm chunk), and the head's key must keep the arena
  base (its args hold arena base + offsets, which address patching cannot follow). The 0.74 ms of fused_head._run is
  run_prepared's own dispatch (40096 patch slots, per-core args over 32 chips, DS41-145), which a shape key does not
  change. Measured: fused_head._run 28.5 -> 27.5 ms a chunk (cProfile), unchanged.
- Census (8x4, profile rung's last chunk [51200, 56320), DS41_ENGRAM_TABLE=resident, 3 warm chunks after
  profile.warm_up, counted from begin_chunk to sync only; plugin /scratch/dnijemcevic/p15/plug/p15census_plugin.py,
  logs /scratch/dnijemcevic/p15/census_base.log (off) / census_on3.log (on)):
  - op launches a chunk 2820 -> 2540; allocate_tensor_on_device 372 -> 132 (31.1 -> 22.2 ms; the 132 are L1 now, ~168 us
    each); deallocate (ttnn op) 160 -> 120; cProfile a chunk: total 374.7 -> 347.6 ms, op dispatch 242.8 -> 221.7 ms,
    ccl.dealloc 29.4 -> 22.3 ms; linear 11.49 -> 9.46 ms, typecast 6.81 -> 5.86, multiply 6.07 -> 4.77, matmul 3.20 ->
    2.08, fabric_all_gather 35.8 -> 34.6.
  - Allocation sites a chunk, off -> on: mhc.py:125 gather 80 -> 0; norm.py:138 40 -> 0; attention.py:745 40 + 2 -> 0;
    prepared_tail run 40 -> 0; k3 arena_fits probe + alloc_arena 40 + 40 -> 40; se_engine3 realloc_arena (head arena,
    L1) 40 -> 40; k3 dispatch manifest (L1) 40 -> 40; compressor 554 / 634 4 + 4 -> same; engram 2 + 2 -> same.
  - Slice sites a chunk (unchanged counts): served.slice_rows (window carry) 40, keep_blocks 40, carry owner slice 40,
    compressor col 12, engram 16, indexer keys 4, DeviceScalars.at 1.
  - Plain walls in the census process: off 343.9-359.7 (median 349.0), on 322.5-336.3 (median 329.2) ms.
- Bit-identical (tt/perf_host_alloc_identity.py, one process, 10 runs alternating on / off, the golden prefix
  reloaded each run; "off" hides the load-time buffers): final hidden equal on every run, PCC vs golden 0.578422
  (= P.14). Walls: on 317.6-330.4 (median 326.8), off 341.6-375.1 (median 367.1); on faster in every pair.
- Host a call (tt/perf_host_alloc.py, 8x4, DS41-172): linear 67.1 -> 30.5 us with optional_output_tensor, typecast
  52.2 -> 20.5, multiply 89.1 -> 55.6, slice 171.0 -> 131.6; allocate + deallocate [640, 1024] bf16 DRAM 55.7 us.
- measure --task P.15 (quick, on): the clean wall finished, chunk_wall_ms 330.3 (spread 34.5, loadavg 18.2; P.14's
  A/B: on 350.9, off 353.8), program_cache_entries 276; the 10-minute shell cap killed it at 596 s before the
  per-layer accuracy (the safe runner marked the cards dirty). The orchestrator's A/B runs both sides; the bytes are
  those of off (identity above). Device time and host_overhead_ms not profiled here (same programs, outputs written in
  place: no device change by construction).
- Remaining top host items (census on, ms a chunk): fabric_all_gather 294 x 118 us 34.6; fused_head._run 40 x 0.69
  27.5; allocate (L1 MoE arenas + manifest) 132 x 168 us 22.2; ccl.dealloc 22.3; flat_routed_expert 40 x 474 us 19.0;
  slice 153 x 96 us 14.6; rms_norm 208 x 66 us 13.8; all_to_all_async_generic 80 x 168 us 13.4; mhc_post 82 x 148 us
  12.1; to_layout 176 x 67 us 11.8; rotary_embedding_indexed 136 x 83 us 11.3; ckv_bfp4 84 x 134 us 11.3. Next: an
  output_tensor option on the forks rms_norm / mhc_post / mhc_pre_xing / rotary_embedding_indexed (~500 calls, ~35
  us each), checking per site that no output is memoized across layers (INDEXER_GLUE's ChunkMemo).
- OV-3: nothing padded. OV-23: no Python-built op added; the ones touched stay prepared (tail run_prepared, the
  exchange's zero program through prepared_program.generic_op).
- Re-run: census `/scratch/dnijemcevic/p15/census.sh <tag> DS41_HOST_ALLOC=on|off` (8 min); identity
  `PYTHONPATH=$PWD DS41_ENGRAM_TABLE=resident DS41_HOST_ALLOC=on scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_host_alloc_identity.py -s` (8 min); per call `...
  tt/perf_host_alloc.py -s` (1 min); `python -m models.demos.common.bringup.orchestrator measure --task P.15`; old path
  DS41_HOST_ALLOC=off.

## P.16 perf (galaxy 8x4, attempts 1-2): FORK_OUT_BUFFERS (forks write load-time output buffers)

- Setting FORK_OUT_BUFFERS (tt/settings.py, default on; off = the ops allocate, the old path).
- Forks (C++ op + nanobind, default None = allocate; spec checked every call, miss and hit; the result is a new
  Python wrapper of the given buffer, DS41-149): ttnn.bringup.rms_norm `output_tensor` (default and split programs),
  mhc_post `output_tensor` / `stats_output` / `stats_scratch` (helper device/mhc_post_given_output.hpp), mhc_pre_xing /
  mhc_pre_xing_pack `hc_output` / `y_output`, rotary_embedding_indexed `output_tensor` (scalar form). Kernels and
  programs unchanged. CHANGELOG / README / INDEX updated in each fork.
- Model (tt/host_alloc.reserve_forks at load, one buffer per site shared by all layers; fork_out returns None when
  off): attn_norm, q_norm, kv_norm, ffn_norm (stats + output), attn_hc (hc, y), ffn_hc (y), attn_residual's stats
  row, both residuals' partials scratch, kv / q / output RoPE. Lifetime check: kept across layers or chunks, so still
  allocating: the mHC streams, ffn_hc's hc (the next layer's pending pre), ffn_residual's stats row (read by the next
  layer), the final norm, compressor / indexer sites (ChunkMemo).
- Census (8x4, profile rung's last chunk [51200, 56320), DS41_ENGRAM_TABLE=resident, 3 warm chunks; logs
  /scratch/dnijemcevic/p16/census_{off,on}.{log,json}), calls a chunk and host us a call off -> on:
  rms_norm 208, 65.9 -> 43.6; mhc_post 82, 137.4 -> 101.4; mhc_pre_xing 80, 126.9 -> 84.4; mhc_pre_xing_pack 3, 110
  -> 105 (not given buffers); rotary_embedding_indexed 136, 69.1 -> 50.8. Fork host total 44.9 -> 31.4 ms a chunk
  (-13.5). Op outputs allocated inside the forks a chunk: ~500 -> the kept sites only; ttnn.allocate_tensor_on_device
  132 -> 132 (MoE arenas, not this change). Plain warm walls in the census process: off median 332.6 ms
  (328.9-344.4), on 306.2 (304.1-314.5).
- Bit-identical: tt/perf_fork_out_identity.py (one process, alternating on / off): final hidden equal every run,
  PCC vs golden 0.578422 both; walls on 297.2 / 317.6, off 328.4 / 367.0.
- Device time / host_overhead_ms: not profiled here (same programs, outputs at other addresses: no device change by
  construction); the orchestrator's A/B profiles both sides.
- Tests: the four new unit tests passed on the 8x4 (rms_norm 4, mhc_post 3, mhc_pre_xing 3, rope with uniform 14).
  Fork regressions partly run in attempt 1, then stopped by the owner (OV-24 / OV-25; F.1 runs them): mhc_post model
  cases 206 passed, mhc_pre unit 144 + cases 71 passed, rms_norm source 26 passed, rope unit 22 + source 9 passed.
  DS41-173: rms_norm tests/unit (6 files) and mhc_post / mhc_pre tests/golden import a missing `eval` package.
  test_mhc_post_ttnn.py's mesh fixture opens the whole box when no case was captured at its size (the Galaxy).
- OV-3: nothing padded. OV-23: no Python-built op added or touched (the four forks are C++ device ops).
- Re-run: identity `PYTHONPATH=$PWD DS41_ENGRAM_TABLE=resident DS41_FORK_OUT_BUFFERS=on scripts/run_safe_pytest.sh
  --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tt/perf_fork_out_identity.py -s`; census
  `/scratch/dnijemcevic/p16/census.sh <tag> DS41_FORK_OUT_BUFFERS=on|off`; A/B `python -m
  models.demos.common.bringup.orchestrator measure --task P.16`; old path DS41_FORK_OUT_BUFFERS=off.

## P.17 perf (galaxy 8x4, attempt 1): MOE_DROP, the MoE block drop of model_bringup_lowbit @ 5e86f048ad4

- Applied the 5 commits after 8deddc500ea (6f7fc85f27b, 847723943b1, 26f61977390, a99234c84e0, 5e86f048ad4) with
  `git merge --squash --no-commit origin/dnijemcevic/model_bringup_lowbit; git reset -q`: no conflicts, 49 files,
  every file byte-equal to the branch tip (checked with cmp). No DS41 code changed; tt/moe_block_adapter.py needed
  nothing. `./build_metal.sh` clean; `import ttnn, ttnn.bringup, moe_block, tt.moe_block_adapter` OK.
- Only deviation from the drop: pre-commit's black rewrapped 3 long lines (formatting only) in moe_block/moe_block.py,
  combine_rf/sparse_combine_rf.py, deepseek_v3_d_p/tt/moe/tt_k3_token_split_moe.py; kept so the gate's commit is not
  refused (K-55).
- No new Setting: the drop is code, not a switch; both A/B sides run DS41_ENGRAM_TABLE=resident as the task names.
- Shipped configuration runs (MOE_IMPL=block, MOE_HEAD=fused, torus 8x4): `measure --task P.17` quick side
  (ab/P.17/new/quick, code before black) PASSED every limit; min layer PCC L39 0.578 -> 0.513, err_layer_L39 0.875 ->
  0.914 (limit 1.533), L00-L38 within +-0.0025 of 'before' (most slightly lower), states unchanged. Not bit-identical
  to before (the combine's sums / per-chip accumulators change the summation order). Quick wall 320.4 -> 358.7 ms
  (one sample each, spread 36 / 45 ms; the orchestrator's p51200 / p512000 profiles decide).
- The measure's p51200 profile was stopped by me during weight load (my shell slot would have cut it mid-run, and the
  black rewrap changed the code key anyway); the orchestrator measures 'after' itself.
- New switches the drop adds on the DS41 path (env, read in the drop's code; DS41 sets none, so the defaults run):
  SPC_ACC_PERCHIP=1 (combine per-chip accumulator roles, on), SPC_PLACE=noc1, SPC_YTIE off, MOE_EXPERTS_DRISC=1,
  MOE_ONE_ARENA=1, MOE_TAIL_GENERIC off, SE_PORT_NOC0=1, FAG_PORT_NOC0 off, PERSIST_L1 off (not "small"),
  PERSIST_L1_SEMS=48, se_engine3 tuning SE_NWS=3 / SE_WLA=6 / SE_RWL=1 / SE_COMMON_SPAN=1 / SE_KBD=KBD, SE_WPD=4,
  SE_XL=3, SE_DN1_AT=14, SE_WDD=24, SE_BRW_BAR=1 (used only with SE_BRW=1), A/B-only off: SE_BRW, SE_BRW_DN1,
  SE_HALFSYNC, SE_HXINC, SE_PASSREV, SE_XFIRST, SE_XN1, SE_XPIPE, SE_LOG, SE_PAD=0. Removed: TT_CMBF2D_IDLE (C++
  combine factory).
- OV-23: no Python-built op added or touched by DS41 code; the drop's own Python programs (se_engine3, combine_rf,
  flat_gather) were not audited for the prepared path here. OV-25: no fork suites run. OV-3: nothing padded.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.orchestrator measure --task P.17 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml`; old path = git stash of the drop.

## P.17 perf (galaxy 8x4, attempt 3): MOE_DROP sanity check only (OV-26)

- The drop from attempt 1 is still in the tree: the 49 files of 8deddc500ea..5e86f048ad4 are byte-equal to
  origin/dnijemcevic/model_bringup_lowbit, except black's rewrapping of 3 long lines (formatting only, not env knobs;
  K-55). Following OV-26, nothing was converted or marked, and nothing outside the drop changed (no DS41 code, no
  Setting, no adapter change).
- `./build_metal.sh` rc 0 (already up to date). `import ttnn, ttnn.bringup, moe_block, tt.moe_block_adapter` OK.
- Shipped configuration (MOE_IMPL=block, MOE_HEAD=fused, torus 8x4): `measure --task P.17 --part quick` rc 0, every
  limit passed, min layer PCC L39 0.513, chunk_wall_ms 359.0 (spread 219.7), host_transfers_per_layer 0. No profile
  run (the orchestrator measures p51200 / p512000 itself).
- The drop's switches and their defaults are listed in the attempt 1 section above (SPC_ACC_PERCHIP=1 on, etc.).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.orchestrator measure --task P.17 --part quick --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml`.

## P.19 perf (galaxy 8x4, attempt 2): MM_SWEEP_GALAXY, the matmul configs swept for the 8x4 (MATMUL_CONFIG=galaxy)

- (1) Routing: every dense matmul goes through tt/mm_configs. rope.py PairPerm (the one direct ttnn.matmul) now calls
  `mm.matmul(..., site="rope.pairs", rule=False)`: no config under tuned (as before), table-or-rule under galaxy. It does
  not run at the shipped settings (ROPE_IMPL=on): neither capture contains it. Other ttnn.linear calls are in perf_*.py
  probes only, outside the model.
- (2) Capture on the 8x4 at the shipped settings (SP32 on, ENGRAM_TABLE=resident, DS41_MATMUL_CONFIG=tuned):
  51200 (attempt 1, generated/ds41_p23/calls_tuned.json, golden prefix) and BRINGUP_POS=512000 (new: test_capture
  takes BRINGUP_POS like the F230 profile; calls_tuned_p512000.json). Diff: identical: 187 calls, 12 distinct, the same
  shapes, counts, configs and free L1 (1358 KiB for CBs at every call). No dense matmul shape depends on the context.
- Sweep on single cards (OV-27): new `test_sweep_card` (1x1 mesh, P23_SHARD=i/12, one call per process, 12 processes
  in parallel with `--device auto`), `merge_shards(12)` -> sweep.json + galaxy_table.py. The card is opened without
  fabric: FABRIC_2D on one chip TT_FATALs (DS41-175); no 1D fabric anywhere. A card alone shows the same 12x10 grid;
  candidates are held to the capture's grid and the filler gives the capture's L1. 12 shards took 29 s-171 s each.
- (3) tt/mm_configs.GALAXY: 12 entries keyed by table_key (shape, weight shape, dtypes, fp32 DEST, grid 12x10).
  Acceptance as P.23 (deterministic, L1 fit, within one bf16 rounding of auto: all 12 are bit-identical to auto,
  error vs float64 not worse than current; replace only when > 1 % faster). MATMUL_CONFIG: galaxy (default) / tuned
  (the rule, the old path) / auto. Anything not in the table (other chunk lengths, other boxes) takes the rule.
- (4) Per site and shape (device kernel ms on one card; calls a chunk; current = tuned's rule):
  | site | shape (per chip) | calls | current ms | galaxy ms | chunk ms current -> galaxy |
  |---|---|---|---|---|---|
  | q_a | 160x5120 bfp8 x 5120x1792 bfp8 -> bf16 | 40 | 0.0549 (1D in0) | 0.0450 (2D 12x5 pn5 bw16) | 2.196 -> 1.801 |
  | engram_kv | 640x6144 bfp8 x 6144x6400 bfp8 -> bf16 | 2 | 0.2728 (minimal M4 K4 N8) | 0.2237 (minimal M4 K16 N8 sb1x4) | 0.546 -> 0.447 |
  | attn.wo_a | 8x160x4096 x 8x4096x1024 bfp8 -> bfp8 | 40 | 0.2060 | same | 8.240 |
  | attn.wq_b | 160x1280 x 1280x32768 bfp8 -> bf16 | 40 | 0.1897 | same | 7.587 |
  | attn.wo_b | 160x8192 x 8192x5120 bfp8 -> bf16 | 40 | 0.1506 | same | 6.024 |
  | indexer.wq_b | 160x1280 x 1280x4096 bfp8 -> bf16 | 8 | 0.0269 | same | 0.215 |
  | indexer.weights_proj | 160x5120 x 5120x32 bf16 -> bf16 | 8 | 0.0168 | same | 0.135 |
  | compressor | 160x5120 x 5120x1024 bf16 -> fp32 | 3 | 0.0431 | same | 0.129 |
  | compressor | 160x5120 x 5120x512 bf16 -> fp32 | 1 | 0.0293 | same | 0.029 |
  | compressor | 320x512 x 512x128 bf16 -> bf16 | 3 | 0.0042 | same | 0.013 |
  | compressor | 640x512 x 512x128 bf16 -> bf16 | 1 | 0.0067 | same | 0.007 |
  | hc.fn | 640x5120 bf16 x 5120x32 fp32 -> fp32 | 1 | 0.0298 | same | 0.030 |
  Total 25.15 -> 24.66 ms a chunk (-0.49 ms). The 2x2 rule is within 1 % of the best for 10 of the 12 (wq_b / wo_b
  / wo_a best candidates 0.1-0.5 % faster: noise, kept). Auto would be 3.2x slower on engram_kv, 3.35x on wo_a.
- Out of scope (no program config): the flat routed experts (flat_routed_expert), the shared expert engine and the
  router in the fused MoE head (MOE_HEAD=fused, se_engine3 / moe_grouped_topk), sparse_sdpa, the indexer score.
- Quick A/B (change on, `measure --task P.19 --part quick`): rc 0, all 102 limits pass, L39 err 0.91435 (limit
  1.533; the same value as the tuned capture: bit-identical configs), chunk_wall_ms 301.9 (spread 26.5),
  host_transfers_per_layer 0. Position profiles p51200 / p512000 left to the orchestrator (brief).
- OV-3: nothing padded. OV-23: no Python-built op added or touched. OV-25: no fork suites run.
- Re-run: see tests/perf/test_matmul_sweep.py docstring (capture, test_sweep_card shards, merge_shards).

## P.20 perf (galaxy 8x4, attempt 1): SP8_RESWEEP, the SP=8 x TP=4 attention's matmuls swept on the Galaxy

- Change-on side: DS41_ATTN_SP32=off, ATTN_SP32_PROJ=off, INDEXER_TOPK_LOCAL=off, MATMUL_CONFIG=galaxy,
  ENGRAM_TABLE=resident. Off side: the three on (P.19's SP=32). No switch default changed (F104).
- (1) Off paths: the model ran unchanged with the three off, so nothing was fixed. That covers three warm runs in
  each capture process, at 51200 and 512000, and the quick measure. The off paths are P.15's load-time buffers, P.16's
  fork outputs and P.11's packing. No code change in tt/ besides the table.
- (2) Capture (test_capture, new P23_TAG=sp8 so P.19's calls_tuned*.json stay as they are): calls_sp8.json (51200)
  and calls_sp8_p512000.json are identical: 187 calls, 12 distinct, same configs, 1357 KiB free for CBs at every
  call. No dense matmul shape depends on the context. Sweep: test_sweep_card on 12 single cards in parallel, one
  distinct call each (P23_SHARD=i/12, P23_CALLS=calls_sp8.json, P23_NAME=_sp8, P23_MIN_MS=0). Merged with
  `P23_NAME=_sp8 merge_shards(12)`, giving sweep_sp8.json and galaxy_table_sp8.py. All shards rc 0 in about 3 min.
- tt/mm_configs.GALAXY: 7 SP=8 keys added (12 -> 19 entries). 5 SP=8 calls share a key with P.19's entries
  (engram_kv, hc.fn, indexer.wq_b, compressor 320 / 640 x 512 x 128). Their sweep winners equal P.19's entries, and
  those entries are unchanged. Acceptance is the same as P.19. All 7 are bit-identical to auto. One new winner:
  attn.wo_b 640x2048x5120, minimal M4 K8 N16 sb2x2, 0.0775 -> 0.0739 ms (-4.6 %). The rule is kept for the other 6.
  Cores (OV-31, no 110 cap in this pick): minimal 12x10 = 120 (wq_b, wo_b, q_a); 2D 11x10 = 110 (wo_a, compressor
  1024); 8x10 = 80 (compressor 512); 1x10 (weights_proj).
- (3) Per site, galaxy-tuned device kernel ms on one card. These are calls a chunk and ms a call / a chunk, at the
  5120-token chunk:
  | site | SP=8 shape (per chip) | calls | ms a call | ms a chunk | SP=32 shape (P.19) | calls | ms a call | ms a chunk |
  |---|---|---|---|---|---|---|---|---|
  | q_a | 640x1280 x 1280x1792 bfp8 -> bf16 (K split, + TP sum) | 40 | 0.0251 | 1.005 | 160x5120 x 5120x1792 | 40 | 0.0450 | 1.801 |
  | attn.wq_b | 640x1280 x 1280x8192 bfp8 -> bf16 | 40 | 0.0751 | 3.006 | 160x1280 x 1280x32768 | 40 | 0.1897 | 7.587 |
  | attn.wo_a | 2x640x4096 x 2x4096x1024 bfp8 -> bfp8 | 40 | 0.0593 | 2.371 | 8x160x4096 x 8x4096x1024 | 40 | 0.2060 | 8.240 |
  | attn.wo_b | 640x2048 x 2048x5120 bfp8 -> bf16 (+ RS) | 40 | 0.0739 | 2.957 | 160x8192 x 8192x5120 | 40 | 0.1506 | 6.024 |
  | indexer.wq_b | 160x1280 x 1280x4096 bfp8 -> bf16 | 8 | 0.0269 | 0.215 | same | 8 | 0.0269 | 0.215 |
  | indexer.weights_proj | 640x1280 x 1280x32 bf16 -> bf16 | 8 | 0.0093 | 0.075 | 160x5120 x 5120x32 | 8 | 0.0168 | 0.135 |
  | compressor | 640x1280 x 1280x1024 bf16 -> bf16 | 3 | 0.0199 | 0.060 | 160x5120 x 5120x1024 -> fp32 | 3 | 0.0431 | 0.129 |
  | compressor | 640x1280 x 1280x512 bf16 -> bf16 | 1 | 0.0151 | 0.015 | 160x5120 x 5120x512 -> fp32 | 1 | 0.0293 | 0.029 |
  | compressor | 320x512 x 512x128 bf16 | 3 | 0.0042 | 0.013 | same | 3 | 0.0042 | 0.013 |
  | compressor | 640x512 x 512x128 bf16 | 1 | 0.0067 | 0.007 | same | 1 | 0.0067 | 0.007 |
  | engram_kv | 640x6144 x 6144x6400 bfp8 -> bf16 | 2 | 0.2241 | 0.448 | same | 2 | 0.2237 | 0.447 |
  | hc.fn | 640x5120 bf16 x 5120x32 fp32 -> fp32 | 1 | 0.0298 | 0.030 | same | 1 | 0.0298 | 0.030 |
  Matmul total a chunk: SP=8 10.20 ms (tuned rule 10.44) vs SP=32 24.66 ms. SP=8's matmuls are TP partials (q_a,
  compressor, weights_proj: K split; wo_b) or column splits, so the TP reduce-scatters / gathers and the head <-> seq
  all_to_alls that SP=32 removed are not in these numbers. The orchestrator's profiles at 51200 / 512000 compare the
  whole chunk.
- Quick A/B (change on, `measure --task P.20 --part quick`): rc 0, all 102 limits pass. L39 err 0.8918 / 1.533
  (P.19 SP=32: 0.9144), min layer PCC 0.532 (L39). The tightest are err_state_topk_idx_L08 at 0.90 of its limit and
  window_kv L28 at 0.88. host_transfers_per_layer 0, chunk_wall_ms 374.2 (spread 70.3; P.19's quick on SP=32:
  301.9). No profile was run (brief).
- OV-3: nothing padded. OV-23: no Python-built op touched. OV-25: no fork suites run.
- Re-run: captures `DS41_ATTN_SP32=off DS41_ATTN_SP32_PROJ=off DS41_INDEXER_TOPK_LOCAL=off DS41_MATMUL_CONFIG=galaxy
  DS41_ENGRAM_TABLE=resident P23_TAG=sp8 [BRINGUP_POS=512000] scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/perf/test_matmul_sweep.py -k capture`. Sweep:
  `/scratch/dnijemcevic/p20/sweep.sh` (12 shards; logs in /scratch/dnijemcevic/p20/sweep). A/B: `python -m
  models.demos.common.bringup.orchestrator measure --task P.20`.

## FIX.15 overseer (galaxy 8x4, attempt 3): GALAXY_CORE_CAP, no op on more than 110 Tensix cores of a chip

Owner rule 18 (2026-10-30): a Galaxy's PSU trips on a chip's di/dt; at most 110 of the 120 cores an op. Attempt 2's
edits were in the tree (OV-34): kept, rebuilt (`./build_metal.sh` rc 0), finished. Attempt 1 / 2 stopped by the
overseer (P.20 revert, then the card-0 reset). The P.20 section above is the rejected pick's record, left as found.
- Setting `MAX_WORKER_CORES` (tt/settings.py, default 110): the cap on a Galaxy (cluster type GALAXY / TG /
  BLACKHOLE_GALAXY), the whole grid elsewhere; 0 = no cap. One module reads it: tt/cores.py (`worker_grid` 11 x 10,
  `max_grid()` for fork options, `sub_cores()` for stock `sub_core_grids`; None off a Galaxy, so other boxes run the
  calls as before).
- Forks, option `max_grid` (default None = today's grid; attempt 2): sparse_sdpa, rms_norm (both programs),
  mhc_pre_xing / mhc_pre_xing_pack, mhc_post (+ stats), rotary_embedding_indexed, update_padded_kv_cache (both
  forms), ckv_bfp4 / bfp_round (prepared path kept, OV-23), IndexerScoreProgramConfig `max_grid_x / max_grid_y`.
  New forks (stock op without a grid option): embedding, nlp_concat_heads, nlp_create_qkv_heads
  (nlp_create_q_heads_split), slice_write; README / CHANGELOG / INDEX written, tests/source.yaml skeletons (selection
  and baseline: F.1; no fork suite run here, OV-25).
- This attempt: (1) MoE block: `MoEOptions.max_grid` (moe_block.py; DS41 passes cores.max_grid): the fused head's
  engine + mover column inside 11 columns (HeadSE3 `max_grid`; planner: engine 10 x 10, MT 2, KB 10, predicted 231
  us), its x copy as a capped same-dtype typecast (ttnn.copy takes no grid), the bfp8 cast capped, the warm-up / router
  uploads host-built. (2) cores.maximum = where(ge(a, b), a, b) on the capped grid (bit-exact select; ttnn.maximum's
  binding takes no sub_core_grids; no fork of a binary op for 4 calls a chunk). (3) cores.typecast for ROW_MAJOR
  (tilize -> typecast -> untilize, capped; stock refuses a sub grid on RM): the indexer's block maxes.
  (4) served.slice_rows: no output_tensor under the cap (slice lands it with an uncapped ttnn.copy) and a whole-slab
  read as 2 slices + concat (a no-op slice of the ND-sharded slab is a whole-grid to_memory_config) (DS41-176).
  (5) cores.from_torch: host build + to_device under the cap (from_torch(device=) tilizes / typecasts on the whole
  grid), every tt/ upload; served._zero_fill_capped: DRAMZeroFill's kernel on 11 x 10, prepared (DS41-177).
  (6) compressor divide got its sub_core_grids.
- Audit (tests/perf/test_core_audit.py, real-time profiler core_count per program; DS41-179): before (attempt 2,
  01:09, today's code) 77 (section, op, shape) rows over 110 (120 cores; ring_indexer_score_dsa 114); after:
  0 over in the warm chunk AND in the load + compile chunk (`CORES_LOAD=1 CORES_AUDIT_STRICT=1`: PASS, max 110).
- Matmuls: capped program_config keys on grid 11 x 10, so the P.19 12 x 10 table entries miss on the Galaxy (kept for
  an uncapped 12 x 10 grid). Capture on the capped code (calls_cap110.json: 187 calls, 12 distinct, 1358 KiB free for
  CBs), swept on the 8x4 MESH (OV-34; `test_sweep` now holds candidates to cores.worker_grid): 12 calls in 5:44, all
  bit-identical to auto. ms a call, P.19 12 x 10 table -> 11 x 10 rule -> 11 x 10 best (now in GALAXY):
  q_a x40 0.0450 -> 0.0551 -> 0.0474 (1d bw16); attn.wq_b x40 0.1897 -> 0.1972 (rule); attn.wo_a x40 0.2060 ->
  0.2062; attn.wo_b x40 0.1506 -> 0.1364; engram_kv x2 0.2237 -> 0.2806 -> 0.2284 (minimal M4 K16 N8); compressor /
  indexer / hc.fn within 0.001. Swept total 24.66 (P.19) -> 24.52 ms a chunk.
- Per op (audit, one warm 5120-token chunk, kernel ms a chunk, all chips max; cores before -> after): attention
  linear 120 -> 110, 13.87 -> 13.52; sparse_sdpa 120 -> 110, 11.90 -> 11.61; rotary (attention) 120 -> 110, 4.90 ->
  4.96; mhc_pre_xing x2 120 -> 110, 8.06 -> 8.00; mhc_post x2 120 -> 100 / 110, 7.65 -> 7.73; nlp_concat_heads 120 ->
  110, 2.47 -> 2.58; nlp_create_q_heads_split 120 -> 110, 1.98 -> 2.04; slice_write 120 -> 110, 0.85 -> 0.90;
  ring_indexer_score_dsa 114 -> 104, 3.02 -> 3.31; topk_large_indices 120 -> 110, 1.48 -> 1.49; rms_norm (q_a, ffn,
  kv) 120 -> 110, unchanged; engram_kv minimal_matmul 120 -> 110, 0.455 -> 0.462; attention RM untilize
  (to_layout -> untilize_with_unpadding) 120 -> 107, 4.03 -> ~5.1; attention typecast 120 -> 80, 1.52 -> 1.81; q_a
  typecast 120 -> 100, 0.29 -> 0.55; experts x copy + bfp8 cast 120 -> 100, 0.67 -> 1.21 (DS41-180); MoE fused head
  120 -> 110 (experts section unchanged); flat_routed_expert 109 (unchanged). Full table:
  /scratch/dnijemcevic/fix15/per_op.md; JSON generated/ds41_cores/cores_before.json, cores_after_ms.json.
- Chunk (DS41_ENGRAM_TABLE=resident; before = P.19's shipped side): rung (gate profile) device 191.63 -> 195.03 ms,
  warm wall 302.5 -> 303.7; p51200 device 201.47 -> 205.54, wall 306.4 -> 302.6; p512000 device 239.13 -> 241.32,
  wall 308.6 -> 306.7 (indexer 43.74 -> 46.89 there: the ring indexer score on 104 cores). Gate FIX.15 --check PASS
  (112 checks; err_final_hidden 0.18973 (P.19 0.18954), top5 1.0, contract 0 failed, acks_early 0, smoke 1).
- Not done / open: a re-tune of the capped stand-ins (sub-grid typecast, untilize_with_unpadding, ~2 ms a chunk) and
  of the ring indexer score's work split at 110 (it lands on 104); galaxy_4x4 not run (the fused head at 1280 rows a
  row on 10 engine columns is unverified there; the planner raises if no layout fits). tt-probe.sh mesh mode reset
  the whole box after its (passing) probe (DS41-178): one probe this attempt, then only run_safe_pytest.
- OV-3: nothing padded. OV-23: the one Python-built program added (capped zero fill) runs prepared; ckv_bfp4 stays
  prepared. OV-25: no fork suite run. OV-34: no single-card job; the audit, the probe and the sweep on the 8x4 mesh.
- Re-run: audit `CORES_LOAD=1 PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/perf/test_core_audit.py`; sweep: `-k capture` on test_matmul_sweep.py,
  `cp generated/ds41_p23/calls_galaxy.json generated/ds41_p23/calls_cap110.json`, then `P23_CALLS=calls_cap110.json
  P23_MIN_MS=0 ... test_matmul_sweep.py -k "test_sweep and not card"`; positions: `DS41_ENGRAM_TABLE=resident
  BRINGUP_PROFILE_WALL=1 BRINGUP_POS=<p> BRINGUP_PROFILE_HOST=1 ... test_profile.py`, then `BRINGUP_WALL_CLEAN=1` the
  same; gate: `python -m models.demos.common.bringup gate FIX.15 --spec <spec> --check`.

## P.18 WINDOW_TILES (perf, galaxy 8x4, 2026-10-31): a per-chip window region, one neighbour send

- Setting WINDOW_KV_FORMAT gains `bfp8_tiles` (the default; `bfp8_blocks` = the path before, unchanged). Per layer, every
  chip keeps its OWN window region `TtWindowAttention.region` [1, 1, (128 + q_rows) / 32, 18496] uint8 (8x4: 288 rows,
  9 pages): [carry 128 | its own 160 query rows] as bfloat8_b pages of one tile row (32 rows), the same 16-value
  exponent groups and nearest-even rounding as bfp8_blocks (ckv_bfp4 blocks=True), so the same values.
  Per chunk (`tt/attention._window_region`): (1) the served 0_window_kv write unchanged (TP gather + local write at
  chunk-aligned starts; the full mesh gather + server-order write at other starts); (2) the chip's own rows (pad rows
  zeroed by a per-chip iota) rounded into region rows 128.. (one ckv_bfp4, 5 cores); (3) chip 0's carry: the slot's
  kept tail `TtWindowCache.rtail` when this layer's previous chunk ended at `start`, else the served rows packed into it
  - on the last chip alone (its local slice, no gather) whenever the carry's owner is the last chip (every
  chunk-aligned start), else `cache.carry` (gathers; non-aligned prefix hits); start 0 reads none; (4) ONE neighbour
  send (`tt/window_send.py:WindowSend`, a Python fabric program, one core a chip, prepared): chip q's last 4 pages ->
  chip q + 1's first 4, the last chip's kept tail -> chip 0, then the last chip keeps its own last 4 pages.
  sparse_sdpa (unchanged fork, ROW_BFP8 / ROW_BFP4 kv_window) reads the region with load-time per-chip index lists
  (`WindowTables.region_lists`: query d reads region rows d - d0 + 1 ..; positions < 0 dropped for chip 0 at starts
  < 128). Removed a layer: the SP gather of the blocks, its slice_write, the carry gathers (TP + SP), their slices.
- Deviation from the brief (owner please read): the region is stored as 32-row block pages, not TILE-layout tiles,
  and the sparse_sdpa fork is unchanged (no band-read option): a query's 128 keys start mid tile, so a band read from
  tiles needs a 5-tile window chunk (the kernel's Skt = 4 is compile-time across its softmax code) or a tile -> row
  re-tile in compute; the block page is a tile row's bytes row-major, read with 2 NoC reads a block (~70 KB a query
  vs 87 KB of tiles + a convert) and the existing reader already reads the window as a consecutive band from its first
  id (DS41-182). Same values either way.
- No ready fence in the send: the region is per layer and every layer's MoE exchange orders a chip's next write to a
  receiver after the receiver's read (tt/window_send.py docstring). Routes from tt-metal's dimension-order tables with
  the open fabric's torus wraps (combine_rf route / mesh_plan); 8x4 torus: (r,3) -> (r+1,0) is Y+ then X+ (wrap).
- Byte test `tt/perf_window_tiles.py` (8x4 mesh, regions of 9 and 24 pages, 3 prepared calls, io patched): PASS;
  host enqueue a warm call 220-300 us (32 per-chip programs). Rule 18: send 1 core, own-row ckv_bfp4 5 cores; core
  audit (`CORES_TAG=_p18`, generated/ds41_cores/cores_p18.json) PASS, max 110.
- Accuracy: measure quick (last chunk after the golden prefix, every layer vs golden): every layer and state within
  its limit; L00 0.025307 (= FIX.15), L03 0.07673, L20 0.16442, L39 0.85131 (P.19 0.91435); test_runner_contract
  (prefix hit at 2496, non-aligned start) PASS, 0 failures (gate unit P.18#test_runner_contract).
- Speed (this run, change on; before = FIX.15's shipped side, its BREADCRUMBS numbers): quick wall 280.5 ms (rung,
  FIX.15 303.7); p51200 device 205.54 -> 193.32 ms, wall 302.6 -> 285.2; p512000 device 241.32 -> 234.07, wall
  306.7 -> 289.1. Section attention 66.6 -> 59.24 ms (p51200; P.19-side profile, the same window path as FIX.15).
  Per op (core audit, calls a chunk / cores; the audit recorded no device ms for fork ops this run): attention
  fabric_all_gather 160 -> 40 (the served write's TP gather stays), slice_write 80 -> 0, slice 120 -> 40,
  ckv_bfp4 80 -> 80 on 20 -> 5 cores (own 160 rows + the carry instead of the SP row's 640 + the carry), window send
  +40 (1 core). Host (cProfile, p51200, a chunk): window path 60.6 -> 36.4 ms, all_gather 294 -> 174 calls (39.3 ->
  24.7 ms), the send 3.8 ms / 40 calls; ttnn op launches 2543 -> 2282.
- Window memory a chip (8x4, chunk 5120): before (bfp8_blocks + P.11) window blocks 164 pages 3.03 MB + packed
  blocks 20 + 160 pages 3.33 MB + carry buffers 148 pages 2.74 MB = 9.10 MB shared, + 4 pages (74 KB) a layer a slot;
  after: regions 9 pages a layer = 6.66 MB (40 layers) + index lists 0.41 MB, + the kept tail 74 KB a layer a slot.
- OV-3: nothing padded. OV-23: the one Python-built program added (the send) runs prepared (patch count checked,
  2 slots a chip). OV-25: no fork suite run (no fork changed). OV-35: DS41-175 updated.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_window_tiles.py -s`; `python -m models.demos.common.bringup.orchestrator
  measure --task P.18 --spec <spec>`; old path: DS41_WINDOW_KV_FORMAT=bfp8_blocks.
- Final check (after the hooks reformatted window_send.py): measure quick on the final code: wall 278.2 ms (spread 7.5),
  every layer and state within its limit, layer errors identical to the first run. Its p51200 / p512000 position runs
  were cut: my shell moved the job to the background and killed it at its 10 min limit; SAFE_PYTEST caught the SIGTERM
  and marked cards 0-31 dirty (09:05). No device job run after it (agent rule 19): the next job resets the box
  (DS41-183). The position numbers above are from the first measure (same code before black's reformat).

## FIX.16 (overseer, galaxy_8x4): steady_state(start) on the device model (framework F236)

- What: `TtDS41Model.steady_state(start)` (tt/model.py) -> `TtWindowCache.steady(start)` (tt/attention.py) on every
  layer of the active state. With a start (before each TIMED profile / wall / cProfile run) it marks the kept window
  carry as the tail of a whole previous chunk ending at `start`, in the load-time buffers that chunk would leave it in:
  bfp8_tiles (default) `rtail_end = start` (the last chip's neighbour send rewrites `rtail` every chunk); bfp8_blocks +
  P.11 `_tail_blocks = (start, _tail_buf)`; GLUE_FOLD bf16 rows re-keyed only when kept (nothing allocated). With None
  (before an accuracy run) every kept tail is dropped: the carry is rebuilt from the served cache, as before. Host
  only, no device op, nothing allocated. The carry's values are the last run's tail, not [start - 128, start): timing
  only. The ladder and the contract never call it.
- Why: every timed run repeats the same chunk, so step 3 of `_window_region` (slice 1760x512 + to_layout 128x512 +
  ckv_bfp4 into rtail) ran in every layer although consecutive serving chunks skip it (owner 2026-10-31).
- Switch: `STEADY_STATE` on (default) / off (no-op; the old timed path, comparison only), tt/settings.py.
- Measured (p51200 profile rung, chunk [51200, 56320), 40 layers; off = `DS41_STEADY_STATE=off`, same code, ad-hoc
  results /tmp/fix16_off; on = this gate): clean wall median 279.1 -> 262.5 ms (-16.6; owner estimate 16.3 host);
  device sections attention 59.10 -> 57.49 ms (-1.62), others within +-0.2 (experts 87.04 / 87.13, indexer 7.23 / 7.24);
  gate device_ms_total 184.64.
  Op counts (per layer section, ops file): attention 22 -> 19 rows a layer: exactly `bringup.ckv_bfp4` (the carry's),
  `slice`, `to_layout` gone in every layer; 407 of 416 layer sections op-for-op identical, the other 9 are layer 1's
  first sections that the off run's real-time profiler attributed to one "(other)" row (attribution artifact of that
  run). ckv_bfp4 a chunk 80 -> 40 (on 5 cores each, unchanged; no new op, rule 18 n/a).
- Gate: PASS FIX.16 (P.17#test_ladder, P.17#test_profile; ladder errors unchanged, e.g. err_final_hidden 0.18973,
  top5 1.0; chunk_wall_ms 262.5 < 434, device_ms_total 184.64 <= 230), 1333 s.
- OV-23: no Python-built op added or touched. OV-25: no fork suite run. OV-3: nothing padded.
- Note: profile units take ~12 min, over the 10-min foreground tool limit: device jobs ran as background tasks with a
  2 h limit, waited on (never two at once).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.16 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`; baseline:
  `DS41_STEADY_STATE=off BRINGUP_RESULTS_DIR=/tmp/x BRINGUP_RECORD_TASK=off BRINGUP_WALL_CLEAN=1 DS41_ENGRAM_TABLE=resident
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_profile.py` (after the
  BRINGUP_PROFILE_WALL=1 run).

## P.21 perf (galaxy 8x4, attempt 2): CHEAP_BUNDLE (#9 HOST_GLUE, #7 NORM_INTO_MOE, #8 NORM_GEMM_DTYPE; #10 deferred)
Attempt 1 left no changes in the tree. Owner pick: opportunities #9, #7, #8, #10 (report /scratch/dnijemcevic/opportunities/report.md).
Each item has its own setting (tt/settings.py, default on, off = the path before P.21).
- **HOST_GLUE (#9)**: `core/model_settings.Settings` gained `snapshot()` / `thaw()` / `refresh()`; with no snapshot, get() reads
  the environment as before, so every other model is unchanged. TtDS41Model.__init__ thaws, then snapshots when HOST_GLUE=on, so
  settings.get serves the load-time values with no os.environ read (~1500 calls a chunk). tt/mm_configs.py `_site_config`
  memoizes each call site's program config while the settings are frozen. The key is (id(w), id(ckc), dtypes, batched, rule),
  checked against the identity of w and ckc and against x.shape (DS41-88: ids alone can be reused). It is cleared when the
  snapshot changes. tt/perf_flat_program_key.py calls settings.refresh() after its monkeypatch.setenv (DS41-185). 0 ops removed.
  Host estimate from the report: ~4.7 ms (get) + ~3.9 ms (program_config) cum.
- **NORM_INTO_MOE (#7)**: ffn_norm (column split, bf16, the fork) writes `MoEBlock.input_buffers()`: the fused head's
  persistent x, plus its bfloat8_b xq through the new rms_norm option `bfp8_twin`. The block reads them in place:
  `forward(x, xq_ready=True)` skips the head's bf16->bf16 copy and its bfp8 typecast, and asserts that x is its own buffer
  when xq_ready is set. Both buffers are pinned (ccl.pin), so the block's `_free` leaves them alone. The bytes are the same
  (bfp8_twin = typecast(bf16 out, bfp8) bit for bit, DS41-184). 2 ops a layer, 80 a chunk; DS41-180 measured these two
  capped ops at 1.21 ms of device a chunk. moe_block.py changes: `input_buffers()`, `forward(xq_ready=)`,
  `FusedHead.__call__(xq_ready=)`; the defaults are the old behaviour.
- **NORM_GEMM_DTYPE (#8)**, ATTN_SP32 path only. A step's device readers come from block_steps (kv reads q_a's projection).
  - attn_norm: tried bfloat8_b where q_a alone reads it (32 layers), then dropped. With dtype=bfp8 the norm leaves the fork's
    default program for the split program's LOCAL mode, one core a row tile (5 cores for 160 rows): device_ms_attn_norm
    went 2.6-3.9 -> 5.58 ms a chunk (p51200), against ~0.45 ms for the q_a typecasts it removed (DS41-186). attn_norm stays
    bf16 + typecast.
  - q_norm, in those 32 layers: gamma x 512^-0.5 folded in fp32 at load, writes bfloat8_b. attention._q_stock takes it as
    wq_b's input (`q_prescaled`), so the q scale multiply goes.
  - q_norm, in the 8 indexer layers: writes bf16 (the attention scales it as before) plus a bfp8_twin into one pinned
    buffer (q_a._twin_buffer). The indexer reads that as wq_b's input (`qr_twin`), so its typecast goes.
  - Both bfp8 norm outputs use the new fork option `pack_precise`: one rounding fp32 -> bfp8 instead of fp32 -> bf16 -> bfp8
    (DS41-137). The twin is bit-identical to the typecast it replaces. 40 ops a chunk (32 q multiplies + 8 indexer typecasts).
- **rms_norm fork** (split program; CHANGELOG "pack_precise and bfp8_twin"):
  - pack_precise sets the compute config's bfp8_pack_precise, plus the define RMS_SPLIT_PACK_PRECISE so the binary gets its
    own cache key.
  - bfp8_twin: in pass B, an SFPU rounds y to bf16 ties-away into the free DEST half (gamma's slots; DEST block halved
    without gamma). It is packed to CB c_8 (bfp8) and written at the same pages.
  - tests/unit/test_rms_norm_ttnn_twin.py: 7 passed on the 8x4 (FABRIC_2D, 11 x 10 grid), bit-exact against the typecasts,
    with negative controls and refusals. No fork regression suites were run (OV-25). Cores: max_grid, 110.
- **TINY_KERNELS (#10) not done**: the compressor _pool and the engram _gate_math each need a new Python-built op on the
  prepared path (OV-23).
  - _pool: a reader that re-lays row pairs (2r, 2r+1) of the TILE input into even / odd tiles (64-byte face-row segments
    via L1), then the SFPU max / exp / div / mul chain.
  - _gate_math: a reader that moves columns 4-7 / 8-11 of the [R, 32] partials to columns 0-3 (word copies in L1), then
    rsqrt / sqrt / sigmoid on the SFPU.
  - It is not a fit for mhc_pre_xing_pack's engram mode: the gates need the TP-summed partials.
  - Deferred: a kernel hang on the Galaxy costs a whole-box reset, and it is the smallest item in the bundle (~0.4 ms
    device / ~8.5 ms host, report). Its setting is not added (no no-op switch).
- measure --task P.21 with attn_norm bfp8 still in (code c5e0d552): every part rc 0. quick: all limits pass, wall 258.7 ms.
  p51200: wall 253.7 ms, device 194.48 ms, host cProfile 270.3 ms, 2010 launches; P.18 new was 2282, though the runs
  between differ. p512000: wall 286.5 ms, device 231.71 ms. Then attn_norm went back to bf16 (above); the final measure
  is the next bullet.
- **Final measure** (`measure --task P.21`, change on, code 8ff681ee, attn_norm bf16): rc 0 on every part.
  - quick: every err_* within its limit (worst err_state_topk_idx_L08 at 0.903), min layer PCC 0.725 (L39),
    host_transfers_per_layer 0; wall 321.9 ms with spread 487 (one outlier, the quick run's own).
  - p51200: wall 254.1 ms, device total 190.79 ms (attn_norm 3.20, q_a 3.44, ffn_norm 3.43, experts 93.3), host cProfile
    269.8 ms, 2042 launches.
  - p512000: wall 270.7 ms, device 231.35 ms.
  - Gate limits: wall < 434, device <= 260. The off side (all three switches off) is the orchestrator's A/B.
- Earlier accuracy run, quick (change on, measure --part quick, code d7193162, before the clang-format / TINY_KERNELS-removal edits):
  rc 0, every err_* within its limit (worst err_state_topk_idx_L08 at 0.902 of its limit; P.20's was 0.901). min layer
  PCC 0.599 (L39). chunk_wall_ms 259.5 (spread 22.1). host_transfers_per_layer 0. Not profiled (brief: the orchestrator
  measures off vs on at 51200 / 512000).
- Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.21` (quick);
  `scripts/run_safe_pytest.sh --mesh --run-all ttnn/ttnn/bringup/rms_norm_ttnn/tests/unit/test_rms_norm_ttnn_twin.py`.

## P.22 perf: ROPE_CHAINS (opportunities #2 and #4 as one pick), 2026-10-31
- **What**: the three RoPE chains as one call each of `ttnn.bringup.rotary_embedding_indexed` in a new chain mode
  (fork options `input_heads`, ROW_MAJOR input, `output_layout`, `output_dtype`, `group_heads`; a second device
  operation `device/rope_chain_device_operation.cpp` with its own kernels `device/kernels/chain/`). Setting ROPE_CHAINS
  (default on; off = the path before P.22). Model: `tt/rope.py:rope_chain` / `rope_chains`, `tt/attention.py`
  (`_q` sp32 branch, `_out` sp32 branch; `_q_stock` split into `_q_proj` + split/rope, `_out_proj` into concat/cast +
  `_out_wo`), `tt/indexer.py:_query_chain` (scores skips `_round` when the query is already bfloat4_b),
  `tt/host_alloc.py` (no q / o rope output buffers when on).
  - #4 attention q: wq_b [1, 1, 160, 64 x 512] bf16 TILE -> (heads read in place, tail roped, untilized) -> sparse_sdpa's
    [1, 64, 160, 512] ROW_MAJOR q. Removes nlp_create_q_heads_split + to_rm (untilize): 2 ops a layer, 80 a chunk.
  - #2 attention output: sparse_sdpa's ROW_MAJOR output -> (tilized, inverse rope, packed bfloat8_b, 8-head groups) ->
    wo_a's [1, 8, 160, 4096] input in the HOST_ALLOC "attn.oa" buffer. Removes to_layout + nlp_concat_heads + typecast
    (+ 2 reshape views): 3 ops a layer, 120 a chunk.
  - #4 indexer q: wq_b [1, 1, 160, 32 x 128] -> bfloat4_b [1, 32, 160, 128] (RNE, bfp_round's SFPU rule, into the
    LOWBIT_HOST query buffer). Removes split + bfp_round: 2 ops a layer, 16 a chunk (8 indexer layers).
  - Total: 216 launches a chunk fewer. No sparse_sdpa option was needed: the heads <-> tokens transposition between
    sparse_sdpa (32 heads of one token a tile) and RoPE / wo_a (32 tokens of one head a tile) stays in the ROW_MAJOR
    tensor; the rope op tilizes / untilizes it in its compute kernel.
- **Precision**: unchanged (bit-identical). Fork tests `tests/unit/test_chain.py`: 8 passed on the 8x4, each fused call
  equal to the unfused chain on every chip, miss and hit. Model: `tt/perf_rope_chains.py` (on / off alternating, one
  process, profile rung's last chunk): every run's final hidden state bit-identical, pcc vs golden 0.725092 both sides.
- **Gotcha (DS41-137 used)**: the first bfp8 output differed from ttnn.typecast by one bfp8 step (max |d| 0.0625): the
  packer without bfp8_pack_precise rounds twice; typecast packs precise. The chain compute sets bfp8_pack_precise for
  a bfloat8_b output plus a define (CHAIN_BFP8_PACK_PRECISE: the kernel cache key leaves the flag out).
- **Host design**: a ProgramDescriptor factory with plain uint32 addresses in common args and a factory-level
  override_runtime_arguments (DS41-187: an output_tensor in tensor_args would push the binding fast path to a full
  rebuild per hit); block split from [per, extra] common args and the core's logical (x, y) (as ckv_share, DS41-145:
  no per-core runtime args). One program per mesh range (DS41-147).
- **Cores**: every chain call on max_grid 11 x 10 = 110 cores (attention: 320 blocks a chip; indexer: 160).
- **Measured** (not the gate's profile): q chain host a call 259.9 us (3 ops) -> 61.4 us (fused). Model warm wall,
  same process, median of 3: off 273.8 ms, on 250.1 ms (-23.7 ms a chunk). Device ms a chunk: the orchestrator's A/B.
- stock `ttnn.to_layout` (to_rm) instead of the chain is not used on the default path any more; the off path keeps it.
- Re-run: `scripts/run_safe_pytest.sh --run-all --no-precompile ttnn/ttnn/bringup/rotary_embedding_indexed/tests/unit/test_chain.py -k 8x4`;
  `DS41_ENGRAM_TABLE=resident DS41_ROPE_CHAINS=on scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tt/perf_rope_chains.py -s`;
  `python -m models.demos.common.bringup.orchestrator measure --task P.22`.
- **Final measure** (`measure --task P.22`, change on: ROPE_CHAINS=on ENGRAM_TABLE=resident): rc 0 on every part.
  (A first attempt was killed by the agent tool's 10-minute background limit after its quick part; rerun in full.)
  - quick: every err_* within its limit (99 checked; worst err_state_topk_idx_L08 at 0.903, as P.21), min layer PCC
    0.725 (L39, as P.21), host_transfers_per_layer 0, chunk_wall_ms 232.3.
  - p51200: wall 235.1 ms, device total 179.55 ms (attention 47.16, indexer 7.06), host cProfile 246.2 ms, 1746
    launches. P.21's final measure (before this change): wall 254.1, device 190.79, cProfile 269.8, 2042 launches.
  - p512000: wall 253.3 ms, device 219.0 ms (P.21: 270.7 / 231.35).
  - The off side of the same run is the orchestrator's A/B; the P.21 numbers are another run.

## P.23 perf (galaxy 8x4, attempt 1, 2026-10-31): OWN_ROWS, the residual / mHC streams on each chip's own 160 rows (opportunity #1)
- **What**: setting OWN_ROWS (tt/settings.py, default on; off = the column split, the path before P.23;
  tt/common.own_rows: needs ATTN_SP32, MHC_FUSED, GLUE_FOLD, MOE_IMPL=block, MOE_HEAD=fused). The block boundary,
  h_mid and the pending pre are [1, 1, S/(SP TP), 4 x H] / [.., 24] on chip (r, c) = row block r TP + c (the rows
  sparse_sdpa reads and the MoE block's owner rows; tt/layout.own_rows_to_device / _to_host for the harness).
  - mHC (tt/mhc.py `rows=`): hc_fn^T whole on every chip (stream-major), the partial row is the whole row: no
    fabric_all_gather of the [S/SP, 32] rows, mhc_pre_xing without `partials`; mhc_post(stats_weight=) emits the whole
    row. Removes 2 gathers a layer.
  - attn_norm: no cols_to_rows all_to_all (the collapse already writes the chip's rows); the attention keeps its rows
    (TtWindowAttention.keep_rows: no exit all_to_all). Removes 2 a2a a layer.
  - ffn_norm (tt/norm.py build_norm rows=True for ffn_norm): one local whole-row rms_norm in FFN_NORM_DTYPE; removes the
    stats-only rms_norm and the stats all_gather (2 a layer). NORM_INTO_MOE does not apply (the norm writes its
    FORK_OUT_BUFFERS site; the head's a2a writes the head's x).
  - MoE (models/demos/moe_block, behind options, defaults unchanged): the block's existing `MoEOptions(residual="owners")`
    (Marko's: the fused head fed through one TP all_to_all [S/N, H] -> [S/SP, H/TP]) plus two new options:
    `owners_a2a_into_head` (the a2a writes the head's persistent x buffer, persistent_output_buffer, BEFORE the arena
    is taken: no copy) and tail "fused" with owners = `TailA2A(local=True)` (tt/tail_a2a.py: the workers' tilize + the
    shared expert's add straight into [S/N, H], no ports / all-to-all, one program for the mesh, workers inside
    max_grid: 20 cores). The ops tail's add got sub_core_grids from max_grid (rule 18). tt/prepared_tail.py follows
    TailA2A's `io` / `descriptor` / `out_shape`; tt/host_alloc.reserve_tail_out reserves [S/N, H].
    Added a layer: the head's a2a + the head's xq typecast (P.21's bfp8 twin had folded it into ffn_norm).
  - Engram (layers with engram): tt/engram_kv.py own_rows = chip column c holds the BLOCK-major columns
    [c 5H/TP, (c+1) 5H/TP) of [key 0..3 | value], so ONE a2a (concat columns, split rows) of the unchanged [640, 6400]
    product lands the chip's own [160, 5 x 5120] in block order (same weight bytes, no replicated wkv); tt/engram_add.py
    own_rows: whole weight, gates from the whole rows (no exact tp_sum).
  - Embedding: one cols_to_rows a2a a chunk; final norm local (split=False), to_host concatenates the chips' rows.
- **Not done (owner's sketch)**: the x_gather3 row-shard variant in the fused head (the head gathers [160, 5120] row
  shards instead of an a2a to the width split). x_gather3 / se_engine3 / router_stream consume the gather by K shard
  (XREADY per K step, the engine loads its own K columns from DRAM, the router computes partial logits from the chip's
  H/TP columns): a row-shard gather changes the engine's pipeline, the watcher protocol and the router's partial-logit
  exchange together (~5000 lines of fused kernels). Used instead: the owners path's one TP a2a into the head's x (the
  same bytes as the attention's old exit a2a) + the xq typecast: 2 launches a layer the variant would save.
- **Precision**: no lowering. Only summation orders change: the mHC mixes / sums of squares and the ffn / attn norm
  statistics are full-K fp32 sums on one chip instead of TP partials added in mesh order; the engram gate sums likewise.
  The MoE (head, router, experts, combine) sees the same x bytes as before.
- **Measured** (`measure --task P.23 --part quick`, rc 0, change on): every err_* within its limit; L00-L32 equal to
  P.22's on-side to 3 digits, L33-L38 +0.8..1.2 %, L39 0.700 -> 0.758 (limit 1.533; pcc 0.725 -> 0.684): rounding,
  DS41-188 (L39 moved 0.83 -> 0.72 at P.10, 0.87 -> 0.91 at P.17, 0.85 -> 0.70 at P.21 on rounding-only changes);
  host_transfers_per_layer 0; chunk_wall_ms 212.9 (P.22 on-side quick: 232.3; spread 94.9).
  Launches a chunk (count, not profiled): -240 (80 mHC row gathers, 80 attention a2a, 40 stats rms_norm, 40 stats
  gathers) +80 (40 head a2a, 40 xq typecasts) -2 engram exact TP sums +2 engram a2a +1 embedding a2a: ~-160; the fused
  tail's a2a work is gone from its program. Device / host ms a chunk at 51200 / 512000: the orchestrator's A/B (the
  brief: no profiling by the agent).
- **Tests**: models/demos/moe_block/tests/test_tail_a2a_local.py (local mode vs torch to a bf16 ulp, eager and
  prepared, with / without se, H 5120 / 4096): 8 passed on the 8x4 (MOE_MESH=8x4 MOE_FABRIC=torus_xy).
- **Cores**: TailA2A local 20 workers (11 x 10 cap); every other op unchanged or already capped (mhc / norm forks
  max_grid; a2a CCL workers as before).
- Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.23`;
  `MOE_MESH=8x4 MOE_FABRIC=torus_xy scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/moe_block/tests/test_tail_a2a_local.py`.
- Stock calls touched (F113): stock ttnn.add instead of ttnn.bringup.combine_rf: the MoE block's owners ops tail add
  (an existing call, now given max_grid's sub_core_grids; DS41 runs the fused local tail instead). The new a2a calls
  (engram_kv, embedding) are stock ttnn.experimental.all_to_all_async_generic, as tt/common.cols_to_rows (no fork lists it).
- Gate unit `P.23#test_ladder` (rung last, `gate P.23 --check --unit P.23#test_ladder`): PASS, worst layer pcc 0.6836
  (L39), final hidden pcc 0.98197 rel 0.1902, top1 0.9971, top5 1.0000.

## P.24 perf (galaxy 8x4, attempt 1, 2026-10-31): TINY_KERNELS, the compressor pooling and the engram gate math as one fork program each (opportunity #10)
- **What**: setting TINY_KERNELS (tt/settings.py, default on; off = the op chains, the path before P.24).
  - Compressor (tt/compressor.py `_pool`): the ratio-2 pair pooling (reshape, 4 slices, maximum, 2 subtract, 2 exp,
    add, 2 divide, 2 multiply, add: 16 launches a layer on [S/SP, 1024] fp32) is ONE call of the new fork
    `ttnn.bringup.pair_pool` (ttnn/ttnn/bringup/pair_pool_ttnn, Python ProgramDescriptor op, registered in PYTHON_OPS).
    Prepared always (prepare_program / run_prepared, the 2 address slots patched; OV-23), common runtime args only (no
    per-core args, DS41-145), one program for the mesh range (DS41-147); output = a load-time buffer
    (host_alloc.reserve_forks, site `fork.compressor.pool`, DS41-146) when FORK_OUT_BUFFERS=on. Taken when p is fp32 and
    rows % 64 == 0 (every box: S/SP = 640 / 1280 / 2560), else the chain.
  - Engram (tt/engram_add.py): with MHC_FUSED=on and OWN_ROWS=on (the pass's sums are whole rows) the engram pass
    `mhc_pre_xing_pack` takes the new option `engram_gate_eps` and writes the gates into columns 12..15 of its row; the
    add's `mhc_post` reads them with `post_col=12` (comb_col 0 on the identity [R, 16]). Removes _gate_math (3 slices,
    2 x (multiply, add, rsqrt), 3 multiply, abs, maximum, sqrt, neg, ltz, where, sigmoid: ~19 launches a layer). With
    OWN_ROWS=off (TP partial sums) the gates stay the chain (the epilogue would see partial sums).
- **Fork changes**: mhc_pre_ttnn (C++, rebuilt): option `engram_gate_eps` (default None: the same program, args, CBs,
  hash). On: the writer lays the partial rows coefficient-major (column k = SFPU vector k, lane = token row), the root
  sums them as before (lane-wise, so columns 0..11 are the option-off bytes), the compute runs the gate steps on the
  summed tile, the writer re-lays the row row-major in its free stage page. New fork pair_pool_ttnn.
- **Precision**: none changed. Both kernels use the stock ops' per-vector SFPU bodies (`_calculate_sqrt_body_` rsqrt /
  sqrt, `_sfpu_sigmoid_<true>`, `_sfpu_exp_accurate_<true>`, the fp32 SFPU divide body) in the chains' order, each
  step one fp32 rounding (results stored to DEST between steps: no fused multiply-add). Result: bit-identical to the
  device op chains (unit tests: 0 ulp on every value) and the quick A/B's 103 err metrics equal P.23's change-on side
  to the last digit (L39 0.7581, L38 0.1650; worst err_state_window_kv_L28 at 0.911 of its limit; 102 limits pass;
  host_transfers_per_layer 0).
- **Gotchas**: DS41-189 (SFPI internal compiler error on a runtime-indexed DEST loop around the sqrt body: one template
  instance per stream); DS41-190 (sqrt / rsqrt and the reciprocal share a programmable constant register: two phases,
  re-init between); the gate scratch vectors (columns 16..18) must be zeroed before the pack; ttnn.Shape has no slice
  `__getitem__` (index it).
- **Cores**: pair_pool min(units, 110) = 110 a chip on the 8x4 (160 units: 10 tile rows x 16 column tiles, max_grid
  11 x 10); the engram pass unchanged (its max_grid).
- **Host**: pair_pool 75.8 us a warm call on 32 chips (prepared, output buffer; test_pair_pool_prepared_and_output).
- **Measured** (`measure --task P.24`, change on: TINY_KERNELS=on ENGRAM_TABLE=resident; rc 0 on every part):
  quick wall 194.0 ms (spread 111.6). p51200: wall 203.3 ms, device total 169.67 ms (compressor 1.78, engram_add 0.56),
  1482 launches (host_launch_ms 123.8, cProfile 208.5). p512000: wall 226.4, device 208.34 (compressor 1.78,
  engram_add 0.56), 1482 launches. P.23's change-on side (another run): device 172.01 / 207.73, compressor 2.05 / 1.95,
  engram_add 0.65 / 0.65, 1571 launches, wall 217.7 / 237.1. The off side of this change is the orchestrator's A/B.
- **Tests** (8x4, whole-box mesh, FABRIC_2D; no fork regression suites, OV-25):
  `scripts/run_safe_pytest.sh --run-all --no-precompile ttnn/ttnn/bringup/pair_pool_ttnn/tests/unit/test_pair_pool.py`
  (6 passed) and `.../mhc_pre_ttnn/tests/unit/test_engram_pack.py` (11 passed: the new test_engram_gate x3 +
  refusals, and the file's option-off cases).
- No stock op added on the model path (F113): the model calls only the two forks.
- Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.24`.

## P.25 perf (galaxy 8x4, attempt 1, 2026-10-31): MM_RETUNE_CAP, every dense matmul re-swept within the 110-core cap (MATMUL_CONFIG=galaxy110)
- **Switch**: MATMUL_CONFIG gains `galaxy110` (new default; tt/settings.py). galaxy110 = tt/mm_configs.GALAXY110 (12
  entries, keyed by table_key at grid 11 x 10), the rule for any call not in it. `galaxy` (GALAXY, P.19 + FIX.15) is
  unchanged. mm_configs: `table()` gives the mode's table; the program_config cache key carries the mode name.
- **Capture** (8x4 mesh, `DS41_MATMUL_CONFIG=galaxy DS41_ENGRAM_TABLE=resident P23_TAG=p25 ... -k capture`,
  7m38s): generated/ds41_p23/calls_p25.json, 187 calls, 12 distinct, 1357 KiB free L1 for CBs at every call (budget
  1342 KiB), chunk out vs golden rel 0.75809. Against FIX.15's capture: 11 calls the same. One changed: hc.fn is now
  [160, 20480] x [20480, 32] fp32 (P.23 OWN_ROWS: whole rows, stream-major; was [640, 5120]). The changed dtypes and
  layouts of P.21-P.24 sit in the norms and RoPE chains feeding these matmuls. The matmul inputs' shapes and dtypes are
  unchanged.
- **Sweep** (8x4 MESH, agent rule 19; one process; `P23_CALLS=calls_p25.json P23_CURRENT=galaxy P23_MIN_MS=0
  P23_FULL=1 ... -k "test_sweep and not card"`). Each candidate is held to tt/cores.worker_grid (11 x 10). New in
  test_matmul_sweep.py:
  - P23_CURRENT=galaxy: "current" (the before) = galaxy's config.
  - A pick must name its grid with at most 110 cores, so auto and minimal's default grid are never picked.
  - Per-config `cores` reported.
  - P23_TAG / P23_NAME name the files.
  - Wider candidates: minimal M / N blocks that divide the call's tiles (M5 for 160 rows, N10), and 1D in0 at
    per_core_N up to twice the narrowest (other core counts).
  - Run 1 (P.19 candidate set, _p25): 2m45s, no winner. Run 2 (wider set, _p25w): 12m56s, one winner.
  - Acceptance as P.19: deterministic, within one bf16 rounding of auto, error vs float64 <= galaxy's x 1.01, > 1 %
    faster. Every entry is bit-identical to auto (max |d| 0).
- **Per call** (ms a call, device kernel time on the mesh; cores a chip; calls a chunk):
  | site | shape (per chip) | calls | before (galaxy) | after (galaxy110) |
  |---|---|---|---|---|
  | attn.wo_a | 8x160x4096 x 8x4096x1024 bfp8 -> bfp8 | 40 | 0.2062, 55 (2D nofuse 11x5 bw32) | same |
  | attn.wq_b | 160x1280 x 1280x32768 bfp8 -> bf16 | 40 | 0.1978, 103 (1D in0 pn10) | same |
  | attn.wo_b | 160x8192 x 8192x5120 bfp8 -> bf16 | 40 | 0.1361, 80 (1D in0 pn2) | same |
  | q_a | 160x5120 x 5120x1792 bfp8 -> bf16 | 40 | 0.0473, 56 (1D in0 pn1 bw16) | same |
  | indexer.wq_b | 160x1280 x 1280x4096 bfp8 -> bf16 | 8 | 0.0285, 64 | same |
  | indexer.weights_proj | 160x5120 x 5120x32 bf16 | 8 | 0.0169, 5 | same |
  | engram_kv | 640x6144 x 6144x6400 bfp8 -> bf16 | 2 | 0.2295, 110 (minimal M4 K16 N8 sb1x4) | **0.2246, 110 (minimal M4 K8 N10 sb2x1)** |
  | compressor | 160x5120 x 5120x1024 bf16 -> fp32 | 3 | 0.0430, 55 | same |
  | compressor | 160x5120 x 5120x512 bf16 -> fp32 | 1 | 0.0297, 40 | same |
  | compressor | 320x512 x 512x128 bf16 | 3 | 0.0042, 40 | same |
  | compressor | 640x512 x 512x128 bf16 | 1 | 0.0067, 40 | same |
  | hc.fn | 160x20480 bf16 x 20480x32 fp32 -> fp32 | 1 | 0.0888, 5 (rule: 2D 1x5) | same (now a table entry) |
  Swept total 24.584 -> 24.574 ms a chunk (-0.010). No call runs over 110 cores. The 160-row projections are
  weight-DRAM bound. minimal with M_block 5 is slower than 1D in0 (wq_b 0.215 vs 0.198, q_a 0.067 vs 0.047). More
  cores at other per_core_N do not help (wq_b pn16 on 64 cores 0.216). DS41-191.
- **Measured** (`measure --task P.25`, change on: MATMUL_CONFIG=galaxy110 ENGRAM_TABLE=resident, code 96d85e228ed6efec;
  rc 0 on every part). quick: 102 limits pass (worst err_state_window_kv_L28 0.911, err_state_topk_idx_L08 0.901 of
  the limit), L39 0.7581 (= P.24, bit-identical configs), host_transfers_per_layer 0, wall 193.5 ms (spread 11.7).
  p51200: wall 202.5, device 169.55 ms (engram_kv 0.96). p512000: wall 227.5, device 208.14 (engram_kv 0.93).
  P.24's change-on side (another run): device 169.67 / 208.34. The galaxy-vs-galaxy110 A/B is the orchestrator's;
  expected difference ~0.01 ms a chunk, below run-to-run noise.
- **Not done**: no memory-config change (DRAM-sharded weights would be the next matmul lever; out of a program-config
  pick). No model profile (brief). No fork suites (OV-25).
- OV-3: nothing padded. OV-23: no Python-built op added or touched. OV-25: no fork suite run.
- Re-run: see the test_matmul_sweep.py docstring (P.25 lines). Logs: /scratch/dnijemcevic/p25/{capture,sweep,sweep_w}.log;
  generated/ds41_p23/sweep_p25w.json, galaxy_table_p25w.py. A/B: `python -m models.demos.common.bringup.orchestrator
  measure --task P.25`.

## FIX.17 overseer fix (galaxy 8x4, attempt 1, 2026-10-31): DEEP_LAYERS=all|tail (default all), layers 21-39 only on the prompt's tail
- **Switch**: DEEP_LAYERS (tt/settings.py), default `all` = today's model, unchanged (the only code on the all path is
  two set lookups in `TtDS41Model.layer`). `tail`:
  - `TtDS41Model.__init__`: deep layers = the loaded layers above the last KV / candidate source (max(kv_sources,
    cand_source) + 1 = 21, so 21-39); they must be consecutive with layer 20 loaded; tail_rows = (n_deep + 1) x
    (window - 1) + 1 = 2541 (the owner's 20 x 127 + 1).
  - Normal chunk: `layer(20)` runs as before, its output DevH (streams, pre and the mHC stats row for layer 21) goes to
    `DS41DeviceState.keep_deep_input` (`deep_in` {start: (DevH, end)}, this chunk's and the previous one's; no copy) and
    returns the `Skipped` marker; `layer(21..39)` return it unchanged (free of it is a no-op). Nothing of 21-39 runs.
  - `tail_pass(state, end=None, on_layer=None)` (the caller signals the prompt's end; no length needed before): the
    deep layers over the kept chunks holding [end - 2541, end): the last chunk, plus the previous one when the last holds
    fewer than 2541 rows; each at its own chunk start through the normal block (every layer's window carry as its cache
    holds it: zeros, or the previous tail chunk's kept tail; the top-k / candidate pools of layer 20 at that start;
    ChunkMemo cleared before each). Returns layer 39's output of the last chunk; then `final_norm` / logits as usual.
  - Serving keep_history=False: `put` keeps the previous chunk's top-k / pool too (the two-chunk tail reads them).
  - tt/runners/adapter.py refuses DEEP_LAYERS=tail (the engine reads every layer's window KV).
- **Not the owner's 3072-row chunk**: the tail runs as whole 5120-row chunks at their own starts. The model is built
  for one chunk size (WindowTables, the load-time host_alloc / fork-output buffers, program configs, the top-k and
  candidate stores keyed by chunk start, the window carry keyed by chunk end); a 3072-row pass would need a second
  geometry of all of them and layer 20's pools re-split over 96 rows a chip. Cost: 1 chunk (75.7 ms at 51200) or 2 when
  the last chunk holds < 2541 rows, about 0.1 s a prompt.
- **Accuracy** (tests/perf/test_deep_tail.py, rung `last`, golden prefix of layers 0-20 only, layers 21-39 caches zero,
  the ladder's drift limits): pass. Layers 0-20 over the chunk: rel 0.025-0.188, each within its limit (e.g. L20 0.165 /
  0.252). Layers 21-39 over the tail rows [end - 2541, end): 0.138-0.165, L39 0.613 / 1.533 (ladder L39 0.758: DS41-188
  rounding swing). Final hidden (tail rows) 0.191 / 0.242; top-1 1.000, top-5 1.000 (limit 0.994) over 188 sampled tail
  rows, last token top-1 match; layers 0-20 states within limits (worst rel 0.202, worst index overlap 0.836).
  Exactness: with a zero carry at chunk start s, layer 20 + k is exact for rows >= s + 127 k; the tail starts 2579 rows
  into the chunk here, past 19 x 127 = 2413.
- **Measured** (8x4, one job at a time, `DS41_ENGRAM_TABLE=resident`, test_profile.py with BRINGUP_POS, the gate's two
  commands: clean wall median, then device from the rt profiler; results in /scratch/dnijemcevic/fix17):
  | chunk at | all: wall / device ms | tail: wall / device ms |
  |---|---|---|
  | 51200 | 200.7 / 171.07 | 122.6 / 92.77 |
  | 512000 | 228.3 / 207.56 | 128.0 / 105.02 |
  Tail pass (layers 21-39, one 5120-row chunk, synced wall / rt device): 75.7 / 77.18 ms at 51200, 100.3 / 103.90 ms at
  512000 (test_deep_tail.py, DEEP_TAIL_POS=512000 for the second). Programs warm in every run (cache 140 entries tail,
  162 all; no misses in the timed runs).
- **Prompt tok/s** (chunk walls linear in position between the two measured starts, extrapolated past 512000, plus the
  tail pass at the end; as /scratch/dnijemcevic/opportunities/prefill_tps.py's arithmetic):
  | prompt | all: wall s, tok/s | tail: wall s (tail pass), tok/s | x | device-only tok/s all -> tail |
  |---|---|---|---|---|
  | 128k (131072, 26 chunks, 1 tail chunk) | 5.24, 25023 | 3.27 (80 ms), 40067 | 1.60 | 29295 -> 52376 |
  | 512k (524288, 103 chunks, 2 tail chunks) | 21.97, 23867 | 13.08 (201 ms), 40075 | 1.68 | 27120 -> 50711 |
  | 1M (1044480, 204 chunks, 1 tail chunk) | 46.67, 22382 | 26.26 (128 ms), 39776 | 1.78 | 24595 -> 48355 |
- Gotchas: keeping layer 20's DevH across chunks (no copy) did not add programs (F196 warm in 10 runs; DS41-146
  addresses: no miss). to_host of the Skipped marker fails: a harness that reads layer 20's output in tail mode reads
  `state.deep_in[start][0]`.
- OV-3: nothing padded (whole chunks as the model runs them). OV-23: no Python-built op added or touched. OV-25: no fork
  suite run.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/perf/test_deep_tail.py` (DEEP_TAIL_POS=512000: timing only);
  `DS41_DEEP_LAYERS=tail BRINGUP_POS=51200 BRINGUP_PROFILE_WALL=1 DS41_ENGRAM_TABLE=resident scripts/run_safe_pytest.sh
  --run-all --no-precompile models/demos/common/bringup/tests/test_profile.py`.

## FIX.18 overseer fix (galaxy 8x4, attempt 1, 2026-11-01): OWN_ROWS_KV_MARGIN, answered `disagree`, no code change
- **Request**: find what in P.23 (OWN_ROWS) moves the window KV of layers 20 / 35 / 38 (acks and runner-contract checks),
  and fix it in the OWN_ROWS path.
- **Found (device probes, on vs off, DS41_OWN_ROWS env; /scratch/dnijemcevic/fix18/)**: no defect. The values
  reproduce the gate exactly (L20 0.98947, L35 0.98910, L38 0.98994; off 0.99195 / 0.99857 / 0.99651).
  - Golden input per step (L0-3, 14, 20): every step equally far from the golden; attention / attn_norm / window KV
    bit-identical on vs off.
  - The first difference is attn_hc's fp32 mixes (summation order, <= 2.3e-4 rel), which flips 0.13 % of the bf16 stream elements.
  - MoE block on the same x (owners a2a into the head, the local tail vs the column split): bit-identical.
  - Over all positions L20 wkv on 0.1836 / off 0.1842, the same per TP column. Over 286 sliding windows, on - off has mean
    +0.0005 and std 0.004; the failing window is the 16th percentile.
  - Follow-up rows [2496, 2536) as accurate vs the golden as the first ones in both modes; the "last block" check
    compares two device runs.
  - DS41-192 (detail in known_issues_detail.md), the same mechanism as DS41-141 / DS41-125.
- **Decision**: per the brief ("a genuine one-rounding difference with no lower-precision choice behind it: stop and
  report"), nothing changed; the evidence goes to the owner for a waiver (or a per-position drift for the contract limit).
  Speed unchanged (no code change, no device / wall numbers taken).
- Gate not re-run: the code is K.1R's (3a1a390acd8 = FIX.17 + ledger commits); K.1R's failing values are reproduced to the digit
  by the probes. adapter_acks / runner_contract will fail as in K.1R; contract / kv_layout / smoke passed at K.1R.
- Re-run: `DS41_OWN_ROWS=on|off FIX18_OUT=<dir> scripts/tt-probe.sh --mesh fix18_dump < /scratch/dnijemcevic/fix18/probe_fu.py`
  (and probe.py with FIX18_LAST=20, probe_moe.py); analysis `python3 /scratch/dnijemcevic/fix18/ana.py gin|chain 0,1,2,3,14,20`,
  `ana_fu.py`.

## FIX.19 overseer fix (galaxy 8x4, attempt 1, 2026-11-01): CONTRACT_WINDOW_DRIFT, window-KV check like with like
- **Request (owner, 2026-11-01)**: judge the dumped layers' window KV in test_adapter_acks / test_runner_contract over
  many samples, against calib x the reference's drift over the SAME positions. Until now it was one 128-row window
  against the final ring's drift (DS41-125 decision B).
- **Changed** (tests/bringup/contract only, no model code):
  - `official_kv.py` (new) runs the official reference (numerics official, bf16 = reference.shipped_dtype, layers 0-24
    resident) over both chunks of the s10240 golden's tokens. It keeps "L<i>.kv" for the dumped layers
    0, 1, 2, 3, 14, 20, 21, 24 in `<golden dir>-official/window_kv.safetensors` (+ .json carrying the golden's
    content_hash; a file from another golden is refused). It took 530.5 + 557.8 s = **1096 s** (CPU, no device).
    The golden dumps had no per-position official KV: drift-s10240.json keeps only the ring.
    The new file's last 128 rows reproduce that ring entry exactly (L20 pcc 0.99739, rel 0.0722).
  - `caches.GoldenKV(window="drift")`, mode "drift": a dumped layer's window KV over EVERY position of the checked range.
    Each kv_dump_compare channel (nope / pe) is checked against `GoldenKV.limits`: drift._float_entry(official,
    golden) on the same rows and channel -> e2e_limits.pcc_limit (1 - calib^2 x max(1 - pcc, 1 - one bf16
    rounding)), floor R.PREFILL_GOLDEN_PCC. No fixed number (F138). `drift_summary()` prints each layer's worst range.
  - test_adapter_acks / test_runner_contract use window="drift" (docstrings updated). Unchanged: compressed caches,
    non-dumped layers (ring snapshots vs R.state_threshold), the 0.99 last-block rule (tt-d-gen's own; L35 / L38 stay
    waived), the byte checks.
- **Numbers** (DS41-193, detail in known_issues_detail.md). Official vs golden L20 nope: [0, 2536) 0.98311, last 128 of
  it 0.99191, ring 0.99660. Old limit 0.98957 = the ring's margin. FIX.18's device dumps (OWN_ROWS on) against the new limits:
  - [0, 2536): nope 0.97809 vs 0.93244, pe 0.99310 vs 0.97721
  - last 128 rows: nope 0.98947 vs 0.96765
  - 286 windows: device median 0.98761, official median 0.98926, correlation 0.95
  The device is as far from the golden as the reference at its own precision.
- **Waiver**: the layer-20 lines of tests.unit_waivers (both tests) can go; the L35 / L38 last-block lines stay. For the
  overseer (spec is not mine to edit).
- **Gate: not completed.** I started the gate in the background with a 600 s job limit. Before the limit could cut it
  mid-device-run, I stopped it (SIGTERM) 10 s into test_contract's UMD device open. run_safe_pytest's pessimistic
  marker left /tmp/tt-device-{0..31}.dirty, so the next mesh run will `tt-smi -r` the whole box. Rule 19 forbids
  triggering a reset, so I ran nothing more on the device and did not touch the markers. The CPU check of the new path
  on the FIX.18 device rows passes, and a noise input fails it. All 5 units are left to the orchestrator's gate.
- OV-3: nothing padded. OV-23: no Python-built op added or touched. OV-25: no fork suite run.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.deepseek_v41_flash_d_p.tests.bringup.contract.official_kv` (only for
  a new golden); `PYTHONPATH=$PWD python /scratch/dnijemcevic/fix19/predict.py`; gate `gate FIX.19 --check`.

## P.26 perf (8x4, 2026-11-01): WINDOW_KV_FORMAT=bf16_tiles; contract err_*_wkv_* diagnostics
- Detail: known_issues_detail.md "P.26 WINDOW_KV_BF16" (file at the 500 KB limit, DS41-194).
## FIX.20-22/P.27-33: known_issues_detail.md
## P.34 perf (8x4, 2026-11-06): POST_COLLAPSE (mhc_post collapse_pre) - known_issues_detail.md "P.34 POST_COLLAPSE"
## Weight cache: one layout, shared master (owner 2026-11-06; fork CHANGELOG "One weight cache layout", framework F261)
- Routed-expert cache = the fork's bank-major `.bm.bin` only (hit = the layer's .bm.bin; a miss writes only those);
  `.tensorbin` is never read (old caches: tools/convert_expert_cache.py). FLAT_CACHE_BANK_MAJOR is gone.
- TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES: tt/settings.py export_pinned_limit sets 16 GiB (32 with bfp8 weights)
  when unset, from bringup/hooks.py device_params (tests, before the mesh opens) and tt/runners/adapter.py (runner, at
  import). An explicit value is kept (test_runner_contract.py passes 0 by default: the KMD 2.9 workaround): the
  stager then logs one FLAT_CACHE line and takes the slower copy.
- galaxy spec paths.tt_cache_shared = /data/dnijemcevic/bringup/deepseek_v41_flash_d_p/tt_cache: the master; the
  orchestrator pulls it into art/tt_cache at run start and pushes new layers in the background
  (`python -m models.demos.common.bringup.core.weight_cache status|pull|push --spec <galaxy spec>`).

## P.37 perf (8x4, 2026-11-06): the carried block's features on MOE_BLOCK_SRC=main
- **What each feature needed in the main block** (C++ fork ttnn/ttnn/bringup/moe_block_main_ttnn, Python
  models/demos/moe_block_main, adapter tt/moe_block_main_adapter.py; fork CHANGELOG "MoEDtypes.shared_activations bf8
  and shared_pack_precise"):
  - GEMM_INPUTS=bfp8 in the head (FIX.9, DS41-132; prerequisite of SE_PACK_PRECISE): main's C++ head never read
    MoEDtypes.shared_activations, so main ran the shared expert's x / h in bf16 (DS41-211). Ported se_engine3's
    x_dtype / h_dtype: HeadShape / SeEngineArgs x_dtype, 1088 B tiles in the engine's regions and CBs, SE_XTB /
    SE_HTB / SE_PREP_TB / XTB defines in se_brisc, se_compute, the x gather; a new head input xq (bf8 x; the engine's
    own-shard load and the gather read it, the router keeps bf16 x from DRAM: SE_RTRX_L1 off with bf8 x own).
  - SE_PACK_PRECISE (DS41-137): MoEDtypes.shared_pack_precise -> HeadProgramConfig.pack_precise -> the SE compute's
    bfp8_pack_precise + define SE_PACK_PRECISE; in the program hash and the plan / setup keys.
  - NORM_INTO_MOE (P.21): input_buffers() returns a persistent bf16 x and the block's persistent bf8 xq (pinned);
    ffn_norm writes both (rms_norm bfp8_twin), __call__(xq_ready=True) hands xq to MoEBlock(xq=). Off: MoEBlock.to_xq
    (one typecast on the capped grid). As on carried, model.py engages it only with OWN_ROWS=off (the default OWN_ROWS=on
    feeds the head from a TP all-to-all: one typecast a layer, both blocks).
  - DECODER_SKIP=replay128 (P.31): at_chunk builds the (mesh, 1024)'s own MoEBlock; the routed weights are shared when
    the tail block's flat plan has the same weight-layout facts (_WEIGHT_KEYS; 8x4: shared for all 20 tail layers),
    else laid out for it; the head weights per block at first call. model.build_tail reserves the 48 KB L1 only for
    carried (main keeps its words in L1_SMALL and allocates its arena a call; the reservation was never given back).
  - EXPERTS_PRECISION: main now asserts glm like carried.
- **Audit** (every setting the carried block / adapter / its _RoutedWeights reads; value: works / ported / out of scope):
  | setting | values | main |
  |---|---|---|
  | MOE_IMPL | block / flat | works (block selects the adapter) |
  | MOE_BLOCK_SRC | carried / main | works |
  | MOE_HEAD | fused | works; ops: NOT ported (main has no ops head: the fused head alone writes the dispatch's route / plan tensors; asserts) |
  | MOE_TAIL | ops / fused | works: main has one tail (C++ fused_tail; OWN_ROWS: the local tail), both values run it |
  | GEMM_INPUTS | bfp8 / bf16 | ported (bf8 x / h in the head; bf16 = the source's program) |
  | SE_PACK_PRECISE | on / off | ported |
  | NORM_INTO_MOE | on / off | ported (bit-identical, test_moe_block_main_parity.py) |
  | DECODER_SKIP | off / exact / replay128 | off, exact work; replay128 ported (at_chunk) |
  | OWN_ROWS | on / off | works (pre-existing own-rows path; off: MoEBlock's forward) |
  | GLUE_FOLD | on / off | works |
  | MAX_WORKER_CORES | Galaxy 110 | works (set_max_grid; the new typecast takes the same cap) |
  | EXPERTS_IMPL | bfp4 / bfp4_mse / bfp8 | works; bfp8_down4: out of scope (owner) |
  | EXPERTS_PRECISION | glm | ported (assert, as carried); full: refused on both |
  | EXPERTS_WEIGHT_CACHE | on / off | works (expert_cache_prefix) |
  | EXPERTS_LAYOUT | marko / legacy | works: placement of carried's flat op only; main's op plans its own layout (same math) |
  | EXPERTS_PREFETCH | on / off | works: carried's bank-major read-ahead; main reads its own cache files (no result change) |
  | EXPERTS_SUM, COMBINE_ACC, COMBINE_PLANNER, MATMUL_FIDELITY | all | n/a: read by TtExperts for its weighted combine, which the carried block never calls |
  | TAIL_A2A_PREPARED, PREPARED_ALL, HOST_ALLOC (+ FORK_OUT_BUFFERS / ROPE_CHAINS / TINY_KERNELS via host_alloc) | all | out of scope (owner): the C++ ops patch addresses on cache hits |
- **Tests added**: fork `ttnn/ttnn/bringup/moe_block_main_ttnn/tests/test_fused_head_bf8.py` (bf8 accuracy +
  determinism; the packed h bit for bit, both options: crafted x / selection weights so h = silu(10) u exactly and the
  output = h; 1,777,311 decided values, 232,605 telling one rounding from two: 0 mismatches each way). DS41
  `tests/test_moe_block_main_parity.py`: SE_PACK_PRECISE on / off through the adapter's block_dtypes / block_dims (same
  bit check), NORM_INTO_MOE on vs off on layer 0 of s10240 chunk 0 (OWN_ROWS=off): max |diff| 0.0, xq_ready taken.
- Re-run: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/moe_block_main_ttnn/tests/test_fused_head_bf8.py`;
  `scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_moe_block_main_parity.py`.
- Gotchas: the fork builds into build_Release; python loads the installed copies (cmake --build build --target
  install). make_head_model builds its own SeEngineArgs beside head_plan's: a new engine arg goes in both (DS41-211).
  The fork test needs mesh_device parametrized (8, 4): the default fixture opens 1x32 and the torus tie check refuses.
- Rule 18: no new op on more than the cap (the head's programs are the fork's, inside set_max_grid; the xq typecast on
  set_max_grid's 11 x 10). Pre-existing on main's OWN_ROWS path: ttnn.to_layout / ttnn.add without sub_core_grids
  (not changed here).
- Note for the A/B: GEMM_INPUTS=bfp8 now reaches main's head on both sides (on and off differ in SE_PACK_PRECISE /
  NORM_INTO_MOE only); main's "before" numbers (2.91 / 1.24 ms) were with bf16 x / h.
- **Measured** (`orchestrator measure --task P.37`, change on: MOE_BLOCK_SRC=main SE_PACK_PRECISE=on NORM_INTO_MOE=on,
  OWN_ROWS=on default; ab/P.37/new): quick: every layer and state within its limit, host_transfers_per_layer 0,
  chunk wall 185.0 ms. p51200: chunk wall 194.0 ms, device 177.27 ms, experts section 106.8 ms = 2.67 ms a layer
  (main before 2.91, carried 2.39); host: the MoE layer's _own_rows 49.19 ms cum / 40 = 1.23 ms a layer (main before
  1.24, carried 1.70). The device gain is the head's bf8 x / h (GEMM_INPUTS reaching it), not the switch: the off side
  (SE_PACK_PRECISE=off NORM_INTO_MOE=off) has bf8 x / h too; under OWN_ROWS=on NORM_INTO_MOE does not engage (as on
  carried). Short turn (gate unit, replay128 on main): final hidden rel 0.14151 (limit 0.24219), top-1 / top-5 hit.
- P.37 attempt 2 (F113 stock-call check):
- stock ttnn.all_gather instead of ttnn.bringup.combine_rf.<module>: main's fused head contract takes x_full (the mesh row's x, bf8 xq when GEMM_INPUTS=bfp8) gathered over the TP axis in DRAM only when its planner picks the x-in-DRAM fallback (blk.x_dram); this is the source block's own call (models/demos/moe_block_main/tt/moe_block.py MoEBlock forward, unchanged since 494262dfccb apart from gathering xq), and combine_rf's fabric all-gather is a Python ProgramDescriptor builder sized for the carried block's token-split path, not a drop-in for a [1, 1, rows, H] DRAM gather. No code change in attempt 2; the measure results of attempt 1 stand.

## P.38 perf (8x4, 2026-11-06): both KV caches bf16 for now (CKV_FORMAT=bf16, WINDOW_KV_FORMAT=bf16_tiles defaults)
- Owner 2026-11-06 (PD migration first cut). Defaults changed in tt/settings.py; old path selectable:
  `DS41_CKV_FORMAT=bfp4_rows DS41_WINDOW_KV_FORMAT=bfp8_tiles`. INDEX_K_FORMAT stays bfp4. Contract unchanged (served
  caches were already bf16, DS41-143); `state.formats()` unchanged (all bf16).
- Chosen: bf16_tiles kept (per-chip region + neighbour send, P.18 / P.26), compressed layers read the bf16 history table
  plus the bf16 region through a new sdpa fork option, `ttnn.bringup.sparse_sdpa(kv_format=BF16, kv_window=region)`
  (KV_WINDOW: chunk 0 from kv_window, top-k chunks from kv, [window | top-k] lists with two sentinel tails, as ROW_BFP4;
  DS41-212). The table is history only (tt/compressor.history_only: base 0, [1, 1, hist, 512] bf16; no window region),
  so no region -> table copy per layer and no list compaction. Stock sparse_sdpa cannot take two tensors.
- Decoder tail (replay128, 32 rows a chip < 128): bf16_tiles now falls back to `bf16` (WindowTables, P.31 style):
  window-only layers the replicated bf16 row table through the fork (BF16, max_grid; the stock op takes no core cap),
  compressed layers `WindowTables.window_rows` (load-time [128 + S, 512] bf16) as kv_window (with a bf16 or a bfp4_rows
  history). Before, bf16_tiles asserted there. model.build_tail asserts history_only() instead of CKV_FORMAT=bfp4_rows.
- Rule 18: the fork call takes max_grid (11 x 10 = 110 cores on the Galaxy); slice_write of window_rows max_grid.
  OV-23: no Python-built op added (the sdpa fork is a C++ device op; WindowSend stays prepared). OV-3: nothing padded.
  OV-25: only the new fork tests run (test_sparse_sdpa_kv_window_bf16.py 10 passed; row_bfp4 test_refusals 1 passed).
- Memory: the bf16 table row is 1024 B vs 320 B (bfp4_rows), 3.2x; the plan budgets the table in bf16 (plan.md
  "sparse-attention table" 1.00 GiB a chip at 1M on 4x4), so it fits as planned. Region per layer 294912 B a chip
  (P.26) vs 166464 B (bfp8_tiles).
- Measured (`orchestrator measure --task P.38`, change on, ab/P.38/new; vs P.37's run of the old path, ab/P.37/new):
  quick: every layer / state within its limit, host_transfers_per_layer 0. compress_kv err L02 0.027 (old 0.115),
  L08 0.056 (0.136), L14 0.127 (0.176), L20 0.098 (0.156); layers L00-L38 5-15 % lower (L20 0.124 vs 0.135, L38 0.117
  vs 0.124); L39 0.825 vs 0.741 (limit 1.533; the end-of-model outlier, chaotic); window_kv L01-L19 lower, L21-L29
  slightly higher (0.100 vs 0.094 at L21, downstream of the changed history; limits 0.155+). p51200: chunk wall
  203.7 vs 194.0 ms, device 183.57 vs 177.27 ms, attention section 50.01 vs 44.12 ms (bf16 rows: 3.2x the history
  bytes, 2x the window bytes), compressor 1.90 vs 1.77; profiled host dispatch 186 vs 156 ms (1300 vs 1254 calls:
  bf16_tiles' to_rm + slice_write a layer instead of one ckv_bfp4). Gate units passed here: test_short_turn (wall
  158.2 ms, device 146.16), test_contract, test_kv_layout, test_adapter_acks; test_ladder / test_profile left to the
  orchestrator.
- Not changed: tests/bringup/test_logit_kl.py asserts the shipped WINDOW_KV_FORMAT is bfp8_tiles (owner 2026-11-01);
  it now fails that assert (diagnostic test, not in this gate; the owner's 2026-11-06 default supersedes it).
- Re-run: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/sdpa/tests/unit/test_sparse_sdpa_kv_window_bf16.py`;
  `python -m models.demos.common.bringup.orchestrator measure --task P.38 --spec <spec>`.

## P.39 perf (8x4, 2026-11-06): served index keys as Blaze decode's 192-byte fp8 rows (INDEX_K_SERVED=fp8_row192)
- Owner-authorized contract change (PD migration first cut). Setting `INDEX_K_SERVED` (tt/settings.py): `fp8_row192`
  (default) / `bf16` (the old contract). A/B: on `DS41_INDEX_K_SERVED=fp8_row192`, off `DS41_INDEX_K_SERVED=bf16`
  (both with `DS41_ENGRAM_TABLE=resident`).
- Row: uint8 [.., 192] a compressed position: e4m3 of key / scale (128 B) | scale fp32 LE (4 B) | 60 zero B; pad rows
  (groups past actual_end) all zero. Records 16 x 192 = 3072 B (2_index_k_r2), 32 x 192 = 6144 B (4_index_k_r1); same
  ND shard / bank / SP placement / TP replication as before.
- Write path (tt/compressor.py `fp8_rows`, `TtServedRows._put`): the chunk's server-order bf16 key slab (as before) ->
  to_layout TILE (capped) -> stock `ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(round_scale_to_power_of_two
  =False)` (TILE path: one core per 32 x 128 block, no grid option: 10 / 20 cores on 8x4, asserted <= MAX_WORKER_CORES)
  -> new Python op `ttnn.bringup.pack_fp8_rows` (fp8_rows_ttnn: data movement only, prepared, max_grid; 20 / 40 cores)
  -> `ttnn.bringup.update_padded_kv_cache` into the served uint8 slab (the copy path takes uint8 as is) and into each
  re-index mirror. Same slab feeds the bf16 compute copy as before.
- Re-index layers: tt-blaze's REINDEX stage (csa_attention/op.py:1126-1161) quantizes the stage payload's
  `index_cache_row` = the producer's post-norm post-RoPE bf16 key (stage_payload.py:29-32); the only producer before
  24-36 is the candidate-source stage, layer 20 (csa_candidate_pool_stage/op.py:271-281). So 4_index_k_r1 covers
  [20, 24, 28, 32, 36] under fp8_row192 (`compressor.reindex_layers`, kv_contract / caches `of` map); layer 20's
  TtServedRows writes the same rows into the mirrors (`mirrors`: the engine slabs, or own tensors in the ladder) in
  the same chunk, before layer 20's ack.
- Indexer choice: a persistent bf16 compute copy (the old served slab, now private, `TtServedRows.slab`), so the
  indexer's numerics are those of the old path (INDEX_K_FORMAT=bfp4 rounds it as before). Cost: memory (8x4, 1M, a
  slot, a chip: compute copy 80 MiB; served 60 MiB vs 80 MiB bf16; mirrors 96 MiB; total 236 vs 80 MiB) and per chunk
  a source: tilize + cast + pack + 1 (r2) / 5 (r1) extra update_padded_kv_cache on W / r rows. Reading the fp8 rows
  instead would dequantize the whole capacity each chunk (the stock per_token_cast_back takes a core a row: > 110).
- Byte checks (OV: the brief's "exact torch recipe" does not hold on this device, DS41-213): served bytes == the stock
  cast packed (test_kv_layout: every index layer incl. mirrors, 0 rows differ; RM-input cast = TILE-input cast); tt-
  blaze's `_quantize_scaled_keys`: scales byte-equal, ~1.1 % of values one e4m3 step away (the cast lands one step
  higher at truncation edges and keeps subnormals), never more. tt-blaze's own test compares its writer with the cast
  (`_glm_quantized_key`), not the recipe. Raw fp8 bytes must be read from the host buffer (DS41-214).
- Test readers: caches.decode / fp8_decode (independent of tt/), fp8_row_failures (zero padding, positive scale, the
  row's largest |e4m3| 416 or 448, all-zero pad rows), bytes_equal on raw rows; acks: a re-index layer's records
  byte-equal to layer 20's; golden comparison of decoded rows at the unchanged limits (re-index layers against their
  source's golden and limit, server_rules.state_threshold). kv_contract.read_rows decodes for the producer read-back.
  `state.served` / `to_torch` decode the served rows (load_served of decoded rows re-encodes: d2p is off).
- Rules: 18 (cast asserted <= cap, pack / tilize / update capped); OV-23 pack_fp8_rows prepared (patch_count 3);
  OV-3 nothing padded to fit an op (the 60 zero bytes are the contract's row); OV-25 only the new op's unit test run.
- Re-run: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/fp8_rows_ttnn/tests/unit/test_pack_fp8_rows.py`;
  `python -m models.demos.common.bringup gate P.39 --spec <spec> --check --unit P.39#test_kv_layout`;
  `python -m models.demos.common.bringup.orchestrator measure --task P.39 --spec <spec>`.
- Measured (`orchestrator measure --task P.39`, change on, ab/P.39/new; vs P.38's change-on run = this task's off path
  before the code changed, ab/P.38/new; the orchestrator measures the real off side): quick: no metric over its limit,
  host_transfers_per_layer 0; every err_layer_L* identical to P.38 to the last digit (the indexer scores the same bf16
  compute copy); err_state_index_k (the served rows decoded) L02 0.0525 (bf16 0.0156, limit 0.236), L08 0.0612 (0.0340,
  0.260), L14 0.0997 (0.0876, 0.292), L20 0.0856 (0.0707, 0.276): the fp8 rounding's cost, far inside the limits
  (calib x the reference's own fp4 drift). p51200: chunk wall 200.8 vs 203.7 ms (noise), device 182.97 vs 183.57 ms,
  compressor section 1.98 vs 1.90 ms (+0.08: tilize + cast + pack + the extra writes), indexer 6.98 vs 6.99,
  profiled host dispatch 175.1 vs 186.4 ms (run noise). pack_fp8_rows host 74 us a call (32 chips, prepared).
- Gate units passed here on the final code: test_kv_layout (byte-exact), test_adapter_acks (0 failures; 4 chunks x 40
  acks), test_contract (producer index_k PCC 0.9927, limit 0.9465; contract_checks_failed 0, acks_early 0). Left to
  the orchestrator: test_ladder, test_profile, test_short_turn.

## P.40 perf (8x4, 2026-11-07): index keys bf16 vs bfp8 vs bfp4 (INDEX_K_FORMAT gains bfp8)
- Change: `INDEX_K_FORMAT` (tt/settings.py) choices bf16 / bfp4 / bfp8; tt/indexer.py `INDEX_K_DTYPES` maps them to
  the score's key dtype: bfp8 = the tiled key slab rounded to bfloat8_b by ttnn.bringup.bfp_round (prepared, compact,
  into the layer's persistent buffer, as bfp4) and the ring's gathered-key buffer bfloat8_b. bf16 / bfp4 unchanged.
  Default stays bfp4 (owner decides). Query keeps INDEX_Q_FORMAT=bfp4; served keys stay INDEX_K_SERVED=fp8_row192.
  No op change: ring_indexer_score_dsa already takes q and k each in its own format; bfp_round already does bfloat8_b.
  Prepared ops touched (OV-23): bfp_round (prepared + compact). Cores: bfp_round on max_grid() (110), unchanged.
- Fork test: indexer_score tests gain a q bfp4 / k bfp8 ratio-2 case (q_dtype / k_dtype). Not run on the 8x4: a
  2x2 FABRIC_2D sub-mesh fails the router handshake (K-68; I ran it once, 24 s, no hang, no reset). For F.1 / a 2x2 box.
- Measured, same code (f5072f7f1d2aa663), env DS41_ENGRAM_TABLE=resident + the arm: bfp8 = `orchestrator measure
  --task P.40` (ab/P.40/new: quick + p51200); bf16 / bfp4 = the same batch plans and env (core/batch, BRINGUP_AB=1,
  BRINGUP_BATCH_MAX_SEQ=56320) by /scratch/dnijemcevic/p40ab/drive.py -> /scratch/dnijemcevic/p40ab/<arm>/; ladder
  rung last per arm (err_final_hidden, top5, every state metric) in p40ab/<arm>/ladder. All runs rc 0, nothing over a limit.

  | metric (limit) | bf16 | bfp4 (default) | bfp8 (new) |
  |---|---|---|---|
  | err_final_hidden (<= 0.242) | 0.1319 | 0.1334 | 0.1313 |
  | top5_overlap (>= 0.994) | 1.000 | 1.000 | 1.000 |
  | err_state_topk_idx L02 / L08 / L14 / L20 | .079 / .069 / .068 / .087 | .164 / .151 / .119 / .151 | .080 / .070 / .067 / .087 |
  | err_state_topk_idx L24 / L28 / L32 / L36 | .134 / .083 / .094 / .086 | .253 / .156 / .154 / .146 | .134 / .083 / .093 / .086 |
  | (limits L02..L36: .217 .172 .166 .222 .318 .216 .222 .212) | | | |
  | overlap_state_topk_idx L02 / L08 / L14 / L20 | .921 / .931 / .933 / .913 | .836 / .849 / .881 / .849 | .920 / .930 / .933 / .913 |
  | overlap_state_topk_idx L24 / L28 / L32 / L36 | .867 / .917 / .907 / .914 | .747 / .844 / .846 / .854 | .867 / .918 / .907 / .914 |
  | err_state_index_k L02 / L08 / L14 / L20 | .0525 / .0607 / .0985 / .0870 | .0525 / .0612 / .0997 / .0856 | .0525 / .0607 / .0981 / .0853 |
  | device_ms_indexer (p51200) | 7.18 | 6.97 | 7.04 |
  | device_ms_total (p51200) | 183.27 | 185.11 | 183.56 |
  | chunk_wall_ms (p51200; spread) | 204.3 (6.8) | 202.9 (12.0) | 206.5 (10.2) |
  | chunk_wall_ms (quick, last chunk) | 179.8 | 180.8 | 181.4 |
  | ring key buffers a chip, 56320 / 1,044,480 tokens | 93.7 / 1738 MB | 26.4 / 489 MB | 49.8 / 923 MB |

  Reading: bfp8 scores like bf16 (top-k overlap within 0.001 of bf16 at every layer); bfp4 loses ~6-12 points of
  overlap (L24: 0.747) but stays inside every limit. Time: indexer 7.0-7.2 ms a chunk in all three (diffs ~0.2 ms,
  run noise level); wall within spread. err_state_index_k is the served fp8 rows (INDEX_K_SERVED), so the same in every
  arm by design. Key-buffer bytes: 8 indexer layers each keep a replicated [T, 128] buffer (2 / 8 / 14 at T = seq / 2,
  20 / 24 / 28 / 32 / 36 at T = seq); bytes a 128-row value: bf16 256, bfp8 136, bfp4 72.
- Caveat (not in my paths): the F137 CPU diagnosis (reference/ds41_ref.py "index_k" site, bringup/hooks.py) models
  only the bfp4 grid for index keys; under bfp8 it would model bfp4. Diagnosis only, never gated.
- Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.40 --spec <spec>`; other arms:
  `python /scratch/dnijemcevic/p40ab/drive.py bf16:ql bfp4:ql bfp8:l` then devjob `bash /scratch/dnijemcevic/p40ab/run.sh`.

## P.41 perf (8x4, 2026-11-07): one bfp8 index-key copy, served 8-bit rows built from it (INDEX_K_COPY=bfp8)
- Setting `INDEX_K_COPY` (tt/settings.py): `bfp8` (default) / `bf16` (the path before, unchanged). A/B: on
  `DS41_INDEX_K_COPY=bfp8`, off `DS41_INDEX_K_COPY=bf16` (both `DS41_ENGRAM_TABLE=resident`).
- bfp8 needs INDEX_K_FORMAT=bfp8 and INDEX_K_SERVED=fp8_row192. Any other value of either: FALLS BACK to the bf16
  copy with one log line (`compressor.index_k_copy`), so INDEX_K_FORMAT bf16 / bfp4 and INDEX_K_SERVED bf16 keep working.
- Change (tt/compressor.py `TtServedRows`): no bf16 slab; `k8` = [1, 1, rows / r, 128] bfloat8_b TILE, DRAM
  interleaved, zeroed at load (served.zero_fill, capped), same local rows (block-cyclic order) as the slab was.
  Per chunk (`_put8`): the server-order rows tiled -> start % S == 0: ttnn.bringup.bfp_round in place at tile row
  (start / S) W / r (same on every chip), then that slice of k8 -> typecast bf16 (exact) -> stock per_token_cast_to_fp8
  -> pack_fp8_rows -> served + mirrors (`fp8_rows_tile`). Other starts (follow-ups, pulled-back last chunk; the c_start
  row's rows sit elsewhere, 16 rows off a tile for ratio 2 at 32 mod 64): `_region` (DS41-215): W blocks c, c + 1 of
  k8 read back (exact), update_padded_kv_cache into that ROW_MAJOR region with start / end shifted by c S, bfp_round
  back in place (old rows re-round to themselves); served rows from bfp_round of the chunk's rows.
  tt/indexer.py `_keys`: returns cache.k8 as it is (kept, never freed); no per-chunk slice / tilize / round of the
  history. Rounding is the same rule on the same bf16 values as P.40's bfp8 arm, so the scored keys are the same.
- Difference to the bf16 path: an aligned write with a partial end zeroes the copy's rows past ceil32(end) of that
  chunk (bfp_round writes the whole W / r rows; the bf16 slab's update left them). Never read: the causal mask hides
  them and the next chunk rewrites from its start.
- Test change (owner-authorized): test_kv_layout's recipe and stock-cast reference input = the golden keys rounded to
  bfloat8_b (ckv_bfp4_ttnn/tests/reference.round_values, mbits 7) when k8 is in use. Nothing else changed.
- New test `tests/test_index_k_copy.py` (8x4, 22 s): ratio 2 and 1, writes at 0 (partial end), 0, S, S + 1312 (32 mod
  64, partial end), 2S + 2944, 3S + 1312 (last block): k8 == the bf16 path's copy of host-rounded keys and served /
  mirror rows byte-equal, every chip, every write. Plus the fallback rule.
- Cores (rule 18): bfp_round max_grid (110); slice / typecast / tilize / untilize (cores.to_rm) on the capped
  sub_core_grids; update_padded_kv_cache max_grid; cast 10 / 20 cores (asserted); pack unchanged.
- OV-23: bfp_round prepared (non-compact: row0 changes per chunk); pack_fp8_rows prepared. OV-25: no fork suite run.
- Memory, index-key copies a chip a slot (3 ratio-2 sources + 1 ratio-1; bf16 256 B, bfp8 136 B a 128-value row;
  before = the bf16 copy + the indexers' persistent bfp8 rounded buffers of layers 2 / 8 / 14 / 20):
  56,320 tokens 6.90 -> 2.39 MB; 1,044,480 tokens 127.95 -> 44.39 MB. Also gone: the per-chunk transient tilized copy
  of the whole history (1M, layer 20: 33 MB RM + 33 MB TILE). Unchanged: served fp8 rows + mirrors, ring key buffers.
- Re-run: `scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/test_index_k_copy.py`;
  `python -m models.demos.common.bringup gate P.41 --spec <spec> --check --unit P.41#test_kv_layout`;
  `python -m models.demos.common.bringup.orchestrator measure --task P.41 --spec <spec>`.
- Gotcha (fixed in this step): test_adapter_acks first failed "Input Tensor is not allocated" at a follow-up (chunk
  [2496, 6072), max_seq two chunks): the region read was a full-range ttnn.slice of k8, a wrapper over k8's buffer, and
  was freed (DS41-52). `_k8_rows` / `_k8_free` compare buffer addresses; the new test covers capacities of 1 and 2 chunks.
- Measured, change on (`orchestrator measure --task P.41`, ab/P.41/new) vs the off path's numbers = P.40's change-on
  run (ab/P.40/new: INDEX_K_FORMAT=bfp8 with the bf16 copy, the same code the off side runs; the orchestrator measures
  the real off side). Ladder rung last: gate unit P.41#test_ladder (P.40 bfp8 arm: p40ab/bfp8/ladder).

  | metric (limit) | off: bf16 copy | on: bfp8 copy |
  |---|---|---|
  | index-key copies a chip, 56,320 / 1,044,480 tokens | 6.90 / 127.95 MB | 2.39 / 44.39 MB |
  | err_state_index_k L02 / L08 / L14 / L20 (.236 .260 .292 .276) | .0525 / .0607 / .0981 / .0853 | .0525 / .0609 / .0981 / .0854 |
  | err_state_topk_idx L02 / L08 / L14 / L20 | .0798 / .0698 / .0674 / .0871 | identical |
  | err_state_topk_idx L24 / L28 / L32 / L36 | .1335 / .0825 / .0929 / .0859 | identical |
  | overlap_state_topk_idx L02..L36 | .920 .930 .933 .913 .867 .917 .907 .914 | identical |
  | err_final_hidden (<= 0.242) | 0.1313 | 0.1313 |
  | top5_overlap (>= 0.994) | 1.000 | 1.000 |
  | device_ms_indexer (p51200) | 7.04 | 6.83 |
  | device_ms_compressor (p51200) | 1.95 | 1.98 |
  | device_ms_total (p51200) | 183.56 | 183.80 |
  | chunk_wall_ms (p51200; spread) | 206.5 (10.2) | 204.7 (30.3) |
  | chunk_wall_ms (quick, last chunk) | 181.4 | 180.7 |

  Reading: the scored keys are the same values (top-k identical to the last digit); the served rows now come from
  the bfp8 keys, a rounding before the e4m3 one (index_k error +0.0002 at most, limits ~0.25). Time at 51,200 (cache
  capacity 56,320): indexer -0.21 ms (no whole-capacity tilize + round per chunk), compressor +0.03 (the slice and
  typecast before the cast); totals and wall within run noise. The removed per-chunk work scales with the capacity,
  so the saving should grow toward 1M tokens (not measured: the orchestrator profiles at 51,200).
- Gate units passed here on the final code: test_kv_layout (0 failures; encoding byte-exact against the stock cast of
  the bfp8 keys), test_adapter_acks (0 failures), test_contract, test_ladder. Left to the orchestrator: test_profile,
  test_short_turn.

## P.42 perf (8x4, 2026-11-07): re-index aliases + 128-row window snapshot at cap (owner-authorized contract change)
- Settings (tt/settings.py): `REINDEX_SERVE` alias (default) / copy (P.39); `WINDOW_KV_KEEP` gains snapshot128 (default;
  full / ring / none kept); `KV_BLOCK_SIZE` 64 (the server's kv_block_size; cap = (end - 1) // it x it). A/B: on
  `REINDEX_SERVE=alias WINDOW_KV_KEEP=snapshot128`, off `copy full` (both ENGRAM_TABLE=resident).
- Aliases: tt-d-gen 48b79127fb1 has no address-uniqueness check (make_kv_chunk_table.cpp:1060 refuses duplicate
  coordinates only; kv_chunk_table.cpp validConfigs, migration_strategy_builder.cpp, kv_chunk_index_builder.cpp:191,
  tableWithinDramGeometry: none), nor has tt-metal's KvChunkAddressTable. kv_contract: config `alloc` = layer 20 only,
  alias entries = layer 20's (noc_addr, size, device group) at the same (slot, position); model.mirrors() -> [].
- Snapshot: TtWindowCache(keep snapshot128) = P.32's 192-row ring for attention + `snap` (CacheSlab [.., 1, 128, 512],
  ND-sharded 32-row records, replicated) + `pin` (the 128 rows before max_seq - chunk: the server's pulled-back last
  chunk needs that carry, outside the ring; found by test_runner_contract's engine case). TtDS41Model.layer ->
  `_snapshot` -> `write_snapshot(end)` after the layer's window write, before its ack, on every chunk not marked
  last=False (tt-d-gen's chunk header has no last flag). Rows: `_snapshot_rows` (ring slices; owned blocks: stock
  all_gather TP then SP of the ring itself into load-time buffers, WindowTables.snapshot_gather_buf; pieces sliced
  straight into p % 128 order), written by update_padded_kv_cache(kv_actual_global=0) (DS41-217). Not written for
  DECODER_SKIP=replay128's tail layers.
- Table: 0_window_kv tensor_type Snapshot, records at 0/32/64/96 only, device group = whole mesh. Declared
  max_sequence_length = max_seq, NOT 128: tt-metal's table TT_FATALs a lookup at >= it (kv_chunk_address_table.cpp:138)
  and the framework's test_contract reads every config at every chunk's first / last block; tt-d-gen's conformSnapshot
  reads it as a 128 extent (SnapshotOfExtent), what decode declares. One-line flip: kv_contract address_table
  `_config(L, seq_len, ..)` -> `c["seq"] or seq_len` once testing/contract.py skips Snapshot configs.
- FRAMEWORK CONFLICT (not fixable in my paths): test_contract FAIL `acks_early` 40 (of 800 blocks; limit 0): it
  re-reads every config's first block after the request; the snapshot's record 0 is rewritten by the next chunk by
  design (one per layer). Its other checks pass (producer snapshot PCC 0.987-0.999 over 96 golden rows a layer).
- DS41-216: snapshot replicas differed on whole TP columns / mesh rows at the gather ring's antipode with the model's
  high_bw gathers (2 of 2 acks runs), a stock whole-mesh gather (3 of 3, also with topology Linear); stock TP-then-SP
  gathers 3 of 3 clean (the shipped path). Golden checks (chip (0, 0)'s copy) passed in every run.
- Measured, change on (`orchestrator measure --task P.42`, ab/P.42/new) vs P.41's change-on run (ab/P.41/new; the
  orchestrator measures the real off side, full + copy): every err_layer_L00..L39 identical to the last digit (L39
  0.7067, limit 1.533), host_transfers_per_layer 0. p51200: chunk_wall_ms 210.0 (spread 27.1) vs 204.7 (30.3);
  device_ms_total 192.17 vs 183.80 (+8.4: the snapshot writes, in the ffn_residual section 13.60 vs 5.07, ~0.21 ms a
  layer, mostly the two gathers); write_snapshot host 22.7 ms a chunk (40 layers). Quick last-chunk wall 186.8 vs 180.7.
- Memory a chip a slot (8x4): window KV full 68.75 MiB at 56,320 / 1281.25 MiB at 1,048,576 -> 20 MiB at any length
  (40 x (192 ring + 192 pin + 128 snapshot) KiB), + one 6.75 MiB gather buffer pair a model; re-index copies 5.16 /
  96.09 MiB -> 0. Total before 73.91 / 1377.34 MiB, after 20 MiB.
- Gate units on the final code: test_kv_layout PASS, test_adapter_acks PASS, test_runner_contract PASS (tt-d-gen
  engine 48b79127fb1, pulled-back chunk), test_contract FAIL acks_early 40 (the framework conflict above). Left to the
  orchestrator: test_ladder, test_profile, test_short_turn.
- Tests: new tests/test_window_snapshot.py (8x4, exact on 32 chips: ends 100, 5120, 6120, 10310 (start 32 mod 64),
  15337, the pulled-back chunk via the pin, 40); caches.snapshot_failures / table_rules (Snapshot type, no record
  past 128, whole-mesh group, alias == source entry, alias records out of the overlap check); acks: snapshot vs golden
  [cap - 128, cap) at every ack and at the end + a 100-token request; runner: final and turn-1 snapshots.
  Per-position-only rules (reused-prefix bytes, last block, bytecmp, pulled-back rewrite of layer 0) skip the snapshot.
- Re-run: `scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_window_snapshot.py`;
  `python -m models.demos.common.bringup gate P.42 --spec <spec> --check --unit P.42#test_adapter_acks`;
  `python -m models.demos.common.bringup.orchestrator measure --task P.42 --spec <spec>`.

## P.42 perf attempt 2 (8x4, 2026-11-07)
- Attempt 1 was stopped because approvals.yaml / tasks.yaml changed. Those edits are the owner's P.43 queue entry
  (10:30, during attempt 1), not the agent's. Left as they are, not reverted.
- No code change in attempt 2. The gate status on the same code: test_kv_layout, test_adapter_acks and
  test_runner_contract kept PASS; test_contract FAIL. `orchestrator measure --task P.42` reused quick + p51200
  (rc 0, every err_layer within its limit, L39 0.7067 vs 1.533; p51200 chunk_wall_ms 210.0, device_ms_total 192.17).
- FAIL contract (framework, DS41-218): `acks_early` 40 (limit 0) of 800 blocks; contract_checks_failed 1. tt-d-gen
  ships Snapshot configs only with the last chunk (prefill_reader.cpp:235, ALL tensors only when is_last_chunk), so
  the bytes it reads are final. The fix belongs in testing/contract.py, which is outside this step's paths.
- Open serving question (DS41-219): at 48b79127fb1 a PD prefill's last actual_end is prompt_len - 1. Our cap and
  decode's differ when prompt_len % 64 == 1. Decode has to confirm which length its cap uses.

## P.43 perf (8x4, 2026-11-07): SKIP_TAIL_START=cap (last chunk / replay128 tail from decode's cap) + Snapshot configs in the framework ack check
- Setting `SKIP_TAIL_START` cap (default) | end (P.42). A/B: on `DECODER_SKIP=replay128 WINDOW_KV_KEEP=snapshot128
  SKIP_TAIL_START=cap ENGRAM_TABLE=resident`, off the same with `end`. DECODER_SKIP=off and chunks not marked last: no change.
- Model (tt/model.py): `last_start(end, request_start)` = the latest 32-aligned start of a request's last chunk
  (cap: exact min(end - 2688, cap - 2668), replay128 min(end - 128, cap - 128); end: exact end - 2688, replay128 no
  limit; clamped at the request start / 0). `tail_start` (replay128 tail: cap: min(ceil32(end - 128), cap - 128)).
  `_check_last_start` raises ValueError at the first decoder layer of a last chunk past last_start (exact only when a
  decoder carry is masked; replay128 unless the chunk is the request's own start). Under cap every layer's snapshot is
  written on the last chunk; replay128's decoder layers from the tail state's window ring
  (`TtWindowCache.write_snapshot(end, src=)`, the main cache's gather buffers and slab; the high_bw gathers kept).
- Caller rule (tests): the chunk before the last ends at last_start(end) (no overlap, so the encoder's carry is in
  the 192-row ring; a pulled-back overlapping start would truncate every layer's window under snapshot128).
- Framework (testing/contract.py): `AckReads`: history configs unchanged (each chunk's first / last block); Snapshot
  configs (`is_snapshot_config`) only at the last chunk's acks, every record of [0, max_sequence_length). Unit test
  tests/test_contract_acks.py (fake table, host only). kv_contract declares 0_window_kv max_sequence_length 128
  (`c["seq"] or seq_len`); caches.table_rules expects 128 for the snapshot.
- Test (tests/test_window_snapshot.py::test_skip_snapshot): every layer, s10240 tokens, ends 5184 / 5150 / 100. The
  limit for the step-dump layers comes from the official per-position window KV on the same rows (official_kv,
  DS41-193): with the ring drift, L20 rows [0, 64) (a plain no-skip chunk) failed 0.182 vs 0.144, the official itself
  is 0.180 there. Results: exact every layer <= 0.76 of limit; replay128 L00-L20 <= 0.78, L21-L39 diag 0.97-2.2x
  (DS41-221); refusal of last_start + 32 raises in both modes.
- Last-chunk start (8x4, chunk 5120), before (end) -> after (cap): replay128 3000: 0 -> 0 (tail 2880 -> 2816); 5150:
  5120 -> 4992 (tail 5024 -> 4992); 5184: 5120 -> 4992 (tail 5056 -> 4992). exact 3000: 288 -> 256; 5150: 2432 ->
  2432; 5184: 2496 -> 2432.
- Gate units passed on this code: test_kv_layout, test_window_snapshot (replay128, exact), test_contract (acks_early
  0, contract_checks_failed 0, 800 blocks), test_adapter_acks, test_runner_contract.
- Re-run: `DS41_DECODER_SKIP=exact DS41_WINDOW_KV_KEEP=snapshot128 DS41_ENGRAM_TABLE=resident scripts/run_safe_pytest.sh
  --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_window_snapshot.py` (~5 min incl. load);
  `scripts/run_safe_pytest.sh models/demos/common/bringup/tests/test_contract_acks.py` (host only).
- test_short_turn (replay128, cap): tail [51520, 51712) (192 rows; was [51584, 51712)), final hidden rel at the last
  token 0.108 (limit 0.242; DS41-205 had 0.199 with the 128-row tail), top-5 / top-1 hit; short_turn_ms 170.9,
  short_turn_device_ms 147.29.
- `orchestrator measure --task P.43` (change on; the harness never marks a chunk last, so the skip does not act):
  quick L39 0.7067 (limit 1.533, identical to P.42), host_transfers_per_layer 0, wall 186.9; p51200 chunk_wall_ms
  204.4 (spread 39.2), device_ms_total 189.02 (P.42's on run: 206.5 / 188.74).

## P.35 perf (8x4, 2026-11-07): NORM_IN_MHC, attn_norm / ffn_norm inside the collapse pass
- What: setting NORM_IN_MHC (on / off; on by default). Fork option `ttnn.bringup.mhc_post(..., collapse_pre=,
  norm_weight=w, norm_eps=)`: the y that POST_COLLAPSE writes for the next hc step comes out normalized,
  bf16((y * rsqrt(sum y^2 / C + eps)) * w), y = the bf16 collapse.
  - tt/mhc.py: `TtHcMixes.norm` (set by the block) and `norm_kwargs()` add norm_weight / norm_eps in collapse_kwargs.
    The weight is the norm's own replicated [1, 1, 1, H] fp32 tile.
  - tt/model.py: the block sets `attn_hc.norm = attn_norm` and `ffn_hc.norm = ffn_norm` when both of these hold:
    `model.norm_in_mhc` (POST_COLLAPSE's conditions + the setting), and a whole-row norm (not split) with a bf16
    output.
  - attn_hc / ffn_hc mark `ctx.extra["attn_normed"/"ffn_normed"]` when the y came from the residual's pass. The norm
    step then hands its input on. Layer 0's attn_hc (mhc_pre_xing) keeps its rms_norm: 1 program a chunk.
- Kernel design and the measured cost are in the fork CHANGELOG ("norm_weight"). In short:
  - y^2 is summed exactly into a spare column (25) of the partial rows.
  - The rstd is computed in the root's sum window.
  - The root's reader sends the 32 values per row straight into the owners' L1. With a spare core per row, the roots
    sit on cores with no units.
  - The norm runs on the SFPU after the hc.
- Traced, 8x4, [160, 5120] a chip: mhc_post 134.4 -> 139.4 us a call. The rms_norm it replaces is 18.7 us, so about
  -13.7 us device a call: -1.1 ms device a chunk (80 calls) and 80 programs fewer. Each removed rms_norm also had
  ~45-60 us of host a call (P.43 profile: attn_norm 2.4 + ffn_norm 4.7 ms host a chunk).
- First design: +27 us a call. A device-zone timeline (zones added only while probing, then removed) traced the
  cost to the BRISC word-copy transposes (DS41-222) and to the rstd waiting behind the root's Sinkhorn and
  transposes.
- Precision: not bit-identical to off. The rms_norm fork's statistic is FPU-grade (README Limits); this one is exact
  fp32. Fork test: y_n is within one bf16 step of float64 everywhere, and 1.0-1.3e-5 of the elements are one step off.
  X' and hc are bit-identical to POST_COLLAPSE without the norm.
- Gotcha: a spare-core root (no units) kept hw_startup's bf16 formats; it gave NaN hc until the reconfig (DS41-223).
  The hc rows after the norm phase need srcA / the packer back on fp32.
- Not done: the bfp8 twin for q_a (+0.55 ms). The rms_norm fork's twin is a separate pack path; the y_n tile is in
  DEST, so a precise bfp8 pack of the same DEST value could follow in a later pick. The collapse program's own
  transposes (3 x 5.6-8.2 us on its critical path) are another candidate pick: about -1 ms device a chunk, and they
  are left unchanged so off stays today's program.
- Boxes: [1280, 5120] (2x2 own rows, 8 units a core, no spare cores: roots stay on unit cores) and [640, 1280] pass
  in the fork test. The scratch shape is unchanged.
- `orchestrator measure --task P.35` (switch on; final code): rc 0.
  - quick: every one of the 205 err limits held; worst ratio 0.746 (err_state_window_kv_L25 0.147 / 0.197).
    L00-L38 within ~1 % of P.43's, L39 0.911 (limit 1.533; P.43 0.707, the chaotic last layer).
    host_transfers_per_layer 0. chunk_wall 186.7 ms.
  - p51200: wall 203.2 ms. Sections: attn_norm 0.70 -> 0.017 ms (layer 0 only), ffn_norm 0.70 -> 0, attn_residual
    5.31 -> 5.52 (P.43 profile at rung last; mhc_post is position-independent). 53280 programs (-2528: 79 launches a
    chip).
  - p512000: wall 247.0 ms (spread 52.6). The off side and the gate are the orchestrator's.
- Re-run: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/mhc_post_ttnn/tests/unit/test_mhc_post_norm.py`;
  `python -m models.demos.common.bringup.orchestrator measure --spec <spec> --task P.35`.
- Attempt 2 (2026-11-07): no code change. Attempt 1 failed only on the command-log check: a python3 heredoc doing
  host arithmetic counts as device code. The code is attempt 1's. pre-commit passed on every changed file and
  rewrote nothing. `orchestrator measure --task P.35` reused quick / p51200 / p512000 (code 99663929581d1fa6,
  measured 22:55-23:02 after the last kernel edit at 22:54): rc 0, with the numbers above.

## P.36 perf (8x4, 2026-11-08): FFN_FEED, the MoE head's x written by attn_residual's mhc_post over the fabric
- What: setting FFN_FEED (on / off, default on) + FFN_FEED_LINKS (2). Fork option `ttnn.bringup.mhc_post(...,
  norm_weight=, feed_output=X, feed_semaphores=[ready, arrival], feed_num_links=, feed_cluster_axis=1)`: the pass that
  writes ffn_norm's output (POST_COLLAPSE + NORM_IN_MHC) also writes the MoE head's column-split input X
  [S/SP, H/TP] on every TP chip, as all_to_all_async_generic(in_dim 2, out_dim 3) did (DS41-138). Fork CHANGELOG
  "feed_output" has the kernel design.
  - tt/ffn_feed.py: the two global semaphores, made once a mesh at model load (after ccl.setup, before the MoE blocks
    place their persistent L1), over the capped worker grid.
  - tt/moe_block_main_adapter.py: `can_feed()` (OWN_ROWS path, no token loops / splits / padding; reset in at_chunk),
    `feed_buffer()` (persistent bf16 [S/SP, H/TP] per (mesh, chunk), pinned, allocated at load: DS41-146),
    `__call__(fed=True)` -> `_own_rows` skips the all-to-all and reads the buffer (not freed).
  - tt/mhc.py: `TtHcMixes.feed` (callable) / `feed_kwargs()` / `fed`; tt/model.py: the block sets `ffn_hc.feed` when
    model.ffn_feed (norm_in_mhc, MOE_IMPL=block, TP >= 2), ffn_norm's hc has the norm and the MoE layer can_feed;
    ffn_hc marks ctx.extra["ffn_fed"], experts passes fed=True.
  - Fallback (no refusal): the carried block (MOE_BLOCK_SRC=carried, no can_feed) and the main block's own forward
    run the TP all-to-all as before.
- Bit-identical: quick (rung last, every layer vs golden): all 205 err metrics equal to P.35's to the last digit,
  pcc_chunk_out 0.6054857891895873 both, host_transfers_per_layer 0. Fork test: X bit-identical to the stock
  all-to-all of y.
- Measured (`orchestrator measure --task P.36`, change on, final code; vs P.35's change-on run = this task's off path,
  unpaired): p51200 wall 203.2 -> 203.1 ms, device 187.16 -> 186.92 (attn_residual 5.52 -> 7.06, experts 107.13 ->
  105.28); p512000 wall 247.0 -> 246.1, device 225.30 -> 223.95 (attn_residual 5.51 -> 7.64, experts 106.67 ->
  103.36). Launches 1744 -> 1704 a chunk (-40 all_to_all_async_generic), programs 53280 -> 52000 (-40 a chip).
  The skew wait moved into attn_residual (the brief's risk): the op ends when every TP peer's block has landed.
  The paired off / on table is the orchestrator's.
- Re-measured on main (op alone, 8x4 torus, [160, 5120], 2 links, 50-100 calls; probe `/tmp/p36_timing_probe.py`,
  not kept): host enqueue mhc_post 101.8 us, + feed 109.6 us, the all-to-all it replaces 98.4 us; synced loop 146.7 /
  192.1 / 98.0 us. So about -90 us host and -50 us device a call when the chips are in step.
- Gotchas:
  - The model's mesh is FABRIC_2D_TORUS_XY (TP axis a 4-ring): ports grouped by position hung at the ready fence;
    group targets by get_eth_forwarding_direction (DS41-224). The fork test runs both fabrics.
  - One program per chip (create_descriptor with the coordinate) cost 336 us host a call; the uniform program reads
    its chip from the routing table and the rest from a common-arg table: 110 us (DS41-225).
  - tt-probe saves probes under tests/ttnn/unit_tests/operations/<name>/ (outside this task's paths): deleted.
- Rule 18: the op's cores = units + spare-core roots + 2 L ports inside the 11 x 10 cap (8x4 [160, 5120]: 100 + 5 + 4
  = 109). Rule 4: FABRIC_2D / FABRIC_2D_TORUS_XY only. OV-23: a C++ descriptor op (addresses patched on cache hits),
  110 us host a call.
- Re-run: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/mhc_post_ttnn/tests/unit/test_mhc_post_feed.py`
  (24 passed); `python -m models.demos.common.bringup.orchestrator measure --spec <spec> --task P.36`.
- Attempt 2 (2026-11-08): no code change. Attempt 1 failed on two checks only: a python3 heredoc that edited
  settings.py was read as device code, and the DS41-225 bullet was 602 chars (now cut to fit by rewording one
  phrase). pre-commit on every changed file: passes (the only failure is check-large-files on the orchestrator's
  dashboard/*.html, which are not this step's files). `orchestrator measure --task P.36` reused quick / p51200 /
  p512000 (same code and switch): rc 0, with attempt 1's numbers above (quick chunk_wall 184.7 ms, p51200 203.1,
  p512000 246.1).

## P.44 (perf, attempt 1, 2026-11-10): served index rows encoded from the chunk's bf16 keys (INDEX_K_SERVED_FROM)

- Expected saving (written before measuring): per index-key write with start % chunk == 0, the k8 slice and the
  bfp8 -> bf16 typecast go (2 launches); at other starts the separate bfp_round to a bfp8 temp and the typecast go
  (2 launches). 4 index-key sources a chunk (layers 2, 8, 14, 20) -> 8 launches fewer: device ~-0.05 to -0.2 ms
  (640 / 320 rows x 128 a chip each, tiny ops), host ~-0.5 to -1 ms a chunk (~60-100 us a launch on 32 chips). The
  main point is correctness of the served bytes (fp8(key) as Blaze's writer), not speed. No persistent memory
  change; the per-chunk transient peak is lower (the tiled bf16 rows live through the encode instead of a k8 slice +
  a bf16 typecast copy).
- Change: new Setting INDEX_K_SERVED_FROM (bf16 default | bfp8 = P.41). `_put8` with bf16: start % chunk == 0 rounds
  the tiled rows into k8 in place as before and encodes the served / mirror rows from the same tiled bf16 rows
  (fp8_rows_tile), then frees the tile; any other start runs `_region` (unchanged) and encodes from the tile; the
  separate bfp_round to a bfp8 temp, the k8 slice and the bfp8 -> bf16 typecast are skipped. bfp8 keeps P.41's
  path byte for byte. INDEX_K_COPY=bf16 is untouched (its served rows were already from bf16). No new op, no new
  core grid (fp8_rows_tile / bfp_round as before: rule 18 unchanged; bfp_round prepared=True as before, OV-23).
- Tests (owner-authorized): test_kv_layout's served-row reference input is the unrounded golden keys under bf16
  (rounded under bfp8); test_index_k_copy adds a third cache (INDEX_K_COPY=bf16, unrounded keys) whose served /
  mirror bytes B's must equal under bf16; the k8 == rounded-copy assertions unchanged.
- Results (change on): gate units test_index_k_copy PASS (served + mirror byte-exact with the bf16-key encode on 32
  chips, all 6 write patterns incl. 1312 / 2944 follow-ups), test_kv_layout PASS (served bytes == stock
  per_token_cast_to_fp8 of the golden keys, every byte, all 4_index_k_r1 / 2_index_k_r2 layers).
  `orchestrator measure --task P.44` rc 0: quick chunk_wall 181.1 ms (spread 4.9); p51200 wall 205.9 (spread 22.1),
  device_ms_total 185.77, device_ms_indexer 6.88. err_state_index_k L02/08/14/20 0.0526 / 0.0607 / 0.1015 / 0.0868
  (limits 0.236 / 0.260 / 0.292 / 0.276); topk overlap 0.866-0.933. The paired off/on table is the orchestrator's.
- Host-only: golden index keys of chunk 2 (s10240 rung) of layers 2/8/14/20, CPU writer recipe (caches.blaze_rows):
  fp8(key) vs fp8(bfp8(key)) differ in 18.9 % of e4m3 values, 49.6 % of scales, 100 % of rows (DS41-226).
- Memory (per chip, k8 = the only index-key copy, 136 B a row): 56,320 tok 2.28 MiB, 1,044,480 tok 42.33 MiB, before
  = after (no persistent buffer added or removed; the per-chunk transient is smaller).
- Gotcha: I edited settings.py / compressor.py / tests with `python - <<EOF` text-replace heredocs and ran a host-only
  CPU script (/tmp/p44/share.py: golden keys + blaze_rows, imports ttnn, opens no device). Neither touches a device;
  noted because P.36 attempt 1 was flagged for a python heredoc editing settings.py.
- Re-run: `gate P.44 --spec <spec> --check --unit 'P.44#test_index_k_copy'` / `'P.44#test_kv_layout'`;
  `python -m models.demos.common.bringup.orchestrator measure --spec <spec> --task P.44`.

## P.50 (perf, attempt 1): COMBINE_SPLIT ported from FIX.23 onto the pre-replication base

- Expected per chunk (written before the change; perf_guide: this is a correctness change that adds work, no saving):
  device +~1 ms (routed_experts_combine +0.95 ms a chunk on the replica branch: one extra SFPU pass and a second fp32
  sum a token, ~24 us a layer x 40), host 0 (same launches, same program count; the switch is a compile arg). Wall:
  within noise (FIX.23: 183.2 -> 183.5 ms clean wall, spreads 15.6 / 6.5).
- Ported (DS41-235, owner: the fix only, no redesign): `git diff 4c79ce65ece 4e90f611e86` hunks for the combine fork
  (common/moe_block_config.{hpp,cpp}, common_nanobind.cpp, routed_experts_combine/device/{combine_host,combine_program}
  .{hpp,cpp}, kernels/compute/accumulator_compute.cpp, tests/routed_experts_combine/test_experts_combine_checks.py:
  applied cleanly, offsets only), README / CHANGELOG / INDEX lines, setting COMBINE_SPLIT (default off, owner) and
  its plumbing in tt/moe_block_main_adapter.py, tests/test_determinism.py, testing/b2b.py (ids() only),
  DS41-233 / 234 / 235 (+ detail). Not ported: replicas, slot maps, expert_slot_copy, tile_gather, probe_determinism.
- One addition of mine: the adapter's block cache key (`_BLOCKS`) now includes the split flag (DS41-185: settings are
  frozen at load; a refresh to the other value in one process would have reused the other side's program).
- The test also records `determinism_max_abs_diff_<COMBINE_SPLIT>` (off / on). The gate's command pins
  DS41_COMBINE_SPLIT=on, so the orchestrator's A/B cannot run the off side of this test; I ran it by hand.
- Determinism (40 layers, 4 chunks of 5120 from 0, 3 sequences a prompt, novel / code interleaved, run 0 = cold):
  off (DS41_COMBINE_SPLIT=off, results /scratch/dnijemcevic/bringup/p50_det_off/P.50.json): FAILS,
  determinism_max_abs_diff_off = 0.2542; novel cold vs warm chunk 0: 0.2417 (255 rows), chunks 1-3 0;
  code run 1 [0.0078, 0.154, 0.254, 0.219] (31 / 390 / 1362 / 271 rows), run 2 [0.225, 0.0195, 0.227, 0] rows
  [392, 132, 1050, 0]. on (gate unit P.50#test_determinism): PASS, determinism_max_abs_diff_on = 0, every chunk.
- Re-run: `PYTHONPATH=$PWD DS41_COMBINE_SPLIT=off|on scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_determinism.py` (~5 min); A/B: `python -m
  models.demos.common.bringup.orchestrator measure --task P.50`.
- `orchestrator measure --task P.50` (quick A/B, on side only so far; ab/P.50/new/side.json): quick rung rc 0, every
  one of 102 limits held (largest share err_state_window_kv_L25 0.746), host_transfers_per_layer 0, clean chunk wall
  184.3 ms (spread 23.0); at 51200: chunk wall 206.2 ms (spread 46.8), device total 185.85 ms, experts 104.35 ms.
  The off side and the combine op's own time: the orchestrator's A/B (I did not profile, per the brief).

## P.49 (perf, attempt 1): PREFILL_END=cap, prefill stops at decode's cap; the cap rule equals decode's (DS41-219)

- Expected per chunk (written before measuring; perf_guide section 1: a contract change, not a speed pick): a
  non-last chunk ends on a 64 block (cold starts, chunk 5120), so its cap is its end: device 0, host 0 (no op added,
  removed or resized; the same programs). The A/B (quick rung, 51,200) times full non-last chunks: expected flat, within
  the noise. The request's last chunk: the model still runs the chunk at its fixed padded size (rows past the end are
  pad, masked), so device about 0 there too; only a last chunk wholly past cap (prompt ends < 64 tokens into a chunk)
  skips every layer's compute (saves one whole chunk, ~185 ms device) and only rewrites the 40 snapshots and acks.
- Cap rule (DS41-219, tt-d-gen 48b79127fb1): PD prefill gets actual_end = prompt_compute_end = max(prompt_len - 1, 1)
  (engine/include/engine/runtime/types.hpp:101-110; the last chunk ends there, engine/src/runtime/prefill_writer.cpp:
  125-131); decode asks for [0, prompt_len - 1) (backend_runtime.cpp:817) and its cap is reusable_prefix_cap(64,
  prompt_len) = (prompt_len - 1) // 64 x 64 (engine/include/engine/control/prefix_indexer.hpp:78-81). So cap =
  actual_end // 64 x 64 (tt/attention.py snapshot_cap; was (actual_end - 1) // 64 x 64). Same for every length:
  server_rules.self_check checks lengths 1..258 at blocks 32 and 64. Non-last chunks end on a block because a
  recurrent slot always starts cold (backend_runtime.cpp:1063-1064), so their cap is their end.
- Change: setting PREFILL_END (cap default | end). tt/runners/adapter.py _run runs the chunk with the model's end =
  attention.prefill_end(actual_end); positions >= cap become PAD in the host ids (engram hash, id history) and pad rows
  in the model (no window ring, snapshot, compressed KV, index key writes). Acks still name the engine's range. A chunk
  with cap <= start computes nothing: _acks_only rewrites each layer's snapshot at cap, syncs, acks (K-71). Only the
  serving runtime reads the switch; the ladder / profile call model.layer directly, so the A/B rungs are unchanged.
- Tests updated to the new rule (caches.served_end): test_adapter_acks, test_runner_contract (ranges [lo, cap),
  [0, cap)), test_window_snapshot (test_snapshot_cap values), server_rules (prompt_compute_end, decode_cap). New
  tests/test_prefill_end.py (served_bytes_differ, past_cap_written, end_past_written).
- Bit-identity (test_prefill_end, 40 layers, end -> slot 0, cap -> slot 1, read through the table over UMD):
  prompt_len (mod 64, actual_end, cap, chunks): 191 (63, 190, 128, 1), 4097 (1, 4096, 4096, 1), 4098 (2, 4097, 4096,
  1), 5122 (2, 5121, 5120, 2; last chunk computes nothing), 7231 (63, 7230, 7168, 2), 10176 (0, 10175, 10112, 2);
  both prompts: 0 of 24,960 served records differ; the cap run wrote 0 records in [cap, actual_end); the end run wrote
  some there in every case that has such records (proves the two sides differ).
- Gate units: test_prefill_end PASS, test_adapter_acks PASS (the slot-0 last-block check [2496, 2536) is empty
  under cap: printed, not run), test_determinism PASS (max |diff| 0, both prompts, cold and warm).
- test_contract FAILS under PREFILL_END=cap (DS41-236), checks that measure positions the cap path no longer computes
  (request actual_end 10208 = seq 10240 - contract_tail_pad 32; cap 10176):
    FAIL pcc_contract_state_window_kv: 0.8525 (layer 38; every layer 0.853-0.868) < limit 0.9434 (P.43's limit; this
      run's hook raised before recording it): the window ring state at 10208 vs the device ring, which stops at 10176
    FAIL contract_state_exact_engram_tail: layers 1, 14 [5306, 14, 237] != golden [16, 43, 783] (ids before 10208 vs
      before 10176)
  Not failing: producer KV over [0, 10208) compress_kv 0.98722 (limit 0.9370; P.43 end 0.98714), index_k 0.99175
  (0.9465; P.43 0.99230), window_kv snapshot at cap 10176 min 0.98705 (P.43 0.98686). Per the brief I did not change
  the check, did not run measure / the profile, and stopped. The owner decides; proposals:
    (a) run test_contract with DS41_PREFILL_END=end in the gate (the full-prompt path is kept), and let
        test_prefill_end carry its verdict to cap: every served byte below cap is identical between the two;
    (b) or judge the cap path at the handoff point: KV over [0, cap) and the fixed state at cap, which needs a golden
        state snapshot at 10176 (or contract_tail_pad 64 with that snapshot): a golden / spec change.
- Perf: not measured (stopped before measure, as the brief says). Expected: 0 on the A/B chunks (full, non-last).
- Re-run: `gate P.49 --spec <spec> --check --unit 'P.49#test_prefill_end'` (~8 min);
  `DS41_PREFILL_END=end|cap` selects the side.

## FIX.25 (overseer fix, 2026-11-12): determinism in every gate

What changed (framework; the model's code untouched):
- `testing/model_determinism.py` (new, the one mechanism): the loaded device model runs the novel prompt (the spec's
  prompt file) and the code prompt (DS41-234's, `testing/b2b.py`) interleaved, 4 chunks of 5120 each from position 0,
  3 runs each, every sequence on a fresh state. Run 0 of novel is the cold one (the first sequence after the load,
  DS41-233); every later run follows the other prompt (determinism.py's A, B, A). Each later run must equal its
  prompt's run 0 bit for bit: every chunk's final hidden and, after the last chunk, every layer's served state
  (`state.served`: window KV, compressed KV, index keys; a model without it: `state.to_torch`). Compared with
  testing/determinism.py's snapshot / diff (bits: -0 vs +0 and NaN payloads count). On failure the message names the
  prompt, the run, the first differing chunk and the first differing layer state. Metrics: determinism_max_abs_diff
  (gated == 0; a bits-only difference records 2^-149 so it still fails), _novel / _code, _state_, elements_differ,
  per-chunk lists, determinism_s (cost). Coverage in defaults.yaml `tests.model_determinism` (chunks 4, runs 3).
- `tests/test_determinism.py` (framework, new): that check on the spec's model. The model's own
  `tests/test_determinism.py` now calls the same function (keeps P.49 / FIX.25 / FIX.26 gates that name it and P.50's
  `determinism_max_abs_diff_<COMBINE_SPLIT>` metric); no second mechanism.
- `core/gate.py` `with_determinism` / `needs_determinism` / `gate_limits`: every device gate (`device: true`) that runs
  a framework model-level test (ladder, profile, positions, contract, smoke) or is a full-model or fix step, and names
  no determinism test itself, runs the check without its tasks.yaml entry naming it. It is put FIRST into the gate's
  first unit whose runs share one process at the shipped configuration (no switch variable), so with
  tests.one_process it costs no model load and its first sequence is the cold one; without such a unit it is the
  gate's own first unit `<task>#test_determinism`. `gate_limits` adds `determinism_max_abs_diff: == 0` to the gate's
  limits (run_units, run_gate, delta_status, the brief's thresholds, the A/B gate_on); a tasks.yaml limit is never
  replaced. No switch turns it off (owner: no exemption for the model; tests.nondeterministic stays component-only).
- `orchestrator.py` A/B: a `determinism` item on both sides (quick: added when the pick's gate runs the check; full:
  the gate's injected segment), first in the side's process (`core/batch.phase` -1). The accept (gate_on, the reuse of
  the 'on' side) fails without it passing on the 'on' side; the table shows both sides. A 'before' side (ab.before)
  measured earlier stays valid without it.
- `core/batch.py`: test_determinism.py is a shared test; `hints` reads target.chunk only when a run needs a position
  (it raised on a spec without target: 10 test_perf_ab selftests failed at HEAD for that, now pass).
- `testing/b2b.py` `code_text`: the code prompt falls back to the same kind of text built from the repo's TTNN op
  sources when /scratch/.../code.txt is missing (another box).
- Selftest `selftest/test_model_determinism.py` (10 tests): a deterministic fake passes with 0; a deliberately
  non-deterministic fake (`fixture_model.NondeterministicFakeModel`, a term that depends on the call count, like an
  arrival-order sum) fails naming chunk 0 and layer state L02; a -0 vs +0 difference fails; injection rules; and end
  to end, a ladder gate (tasks.yaml names no determinism test) through run_gate passes with the deterministic fake and
  FAILS with the non-deterministic one (FIXTURE_NONDETERMINISTIC=1), the ladder never running after it.
  test_overseer's fix-job test now expects the check once, first. Full selftest suite vs HEAD: no new failure.

Numbers (Galaxy 8x4, 40 layers, COMBINE_SPLIT=on, the shipped defaults):
- Batched probe, exactly as an injected gate unit runs (determinism, then the `last` ladder rung, one process):
  bit-identical (hidden every chunk, every layer's served state, both prompts, cold and warm); process 347 s = load
  191 s + determinism 93.5 s + ladder 52 s (warm). A ladder unit alone took 266-285 s (FIX.23 / P.48), so the check
  adds about 65-75 s per gate; its cold sequence pays the program compiles the ladder then skips. The warm sequences
  are ~1.5 s each: the 4 x 3 x 2 coverage costs ~10 s over the one cold sequence it cannot avoid.
- Coverage choice: 4 chunks reach every chunk DS41-233 / 234 moved with COMBINE_SPLIT=off (P.50: 0.2542, rows
  [392, 132, 1050, 0]); 3 runs give cold vs warm and warm vs warm after the other prompt. Fewer would save seconds
  only; more chunks need a longer state (max chunks = target.seq / chunk).
- A gate without a shared unit (e.g. only contract tests) pays its own load: ~260-300 s (the model test's 260 s).

Gotcha: I ran `python -m pytest` once for the CPU selftests (an argument error, nothing ran); the selftests then ran
through scripts/run_safe_pytest.sh (CPU only, the runner takes the box lock). And one devjob was started with the bare
safe runner (usage error, rc 3, no device access).

Re-run: `gate FIX.25 --spec <spec> --check`; the batched probe: `BRINGUP_BATCH_LM_HEAD=1 BRINGUP_BATCH_CMD='<determinism
cmd> && BRINGUP_RUNG=last <ladder cmd>' scripts/run_safe_pytest.sh --no-precompile --run-all
models/demos/common/bringup/tests/test_batch.py`; selftest: `scripts/run_safe_pytest.sh --run-all --no-precompile
models/demos/common/bringup/selftest/test_model_determinism.py`.
