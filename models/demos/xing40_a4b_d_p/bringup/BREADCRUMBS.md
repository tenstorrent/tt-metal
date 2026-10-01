# XingChen-AGI/Xing4.0-29B-A4B bring-up: breadcrumbs

Append-only log, one section per task attempt: what was done, decisions and why, gotchas, the re-run command, the verdict.

## R.2 reference (run1, attempt 1)
- Wrote `reference/weights.py` (index-based loader, refuses `model.layers.40.*`) and `reference/xing_ref.py`
  (`XingReference`), hook `reference` in `bringup/hooks.py`. Stock HF loader works as the oracle (fp32, eager, ~50 s
  for 2048 tokens), so no `hf_model` / `hf_layers` hook: HF's [B, S, 4, H] layer output flattens to the reference's
  [S * 4, H] token-major block boundary.
- Block graph (all layers): attn_hc -> attn_collapse -> attn_norm -> q_a -> attention (stateful) -> attn_residual ->
  ffn_hc -> ffn_collapse -> ffn_norm -> mlp (dense 0-1) | router, experts, shared_expert, moe_add (MoE 2-39) ->
  ffn_residual. attn_hc / ffn_hc output [S, 24] fp32 = pre 4 | post 4 | comb 16 row-major.
- mHC differences vs glm53: no +eps on pre, no initial softmax (clamp [-30, 30], minus row max, exp, then 20 x
  row/col normalize), residual is `comb @ streams` (HF matmul(comb, hidden)), not comb^T; mHC RMS eps = rms_norm_eps
  1e-6. Final hidden = mean of 4 streams -> norm (recorded as `hc_mean`, `final_norm`).
- MLA: dense causal, scale 192^-0.5 * mscale^2 (0.14468), YaRN inv_freq checked bit-equal to transformers'
  `_compute_yarn_parameters` (cos/sin scale 1.0). RoPE kept in interleaved (checkpoint) order; HF writes the same
  values de-interleaved (known-issues proposal). State kv_latent [seq, 576] = kv_a_layernorm(latent) | RoPE(k_rope).
- Speed / determinism: the latent prefix is re-expanded through kv_b each chunk in fixed 512-row zero-padded GEMMs,
  query rows in 256-row blocks aligned to absolute positions, expert groups padded to 32 rows. Local check on layers
  0-3, 4096 in 2048: chunked == one-shot maxabs 0 (hidden and kv_latent), layer-2 graph replay maxabs 0.
- Gate result: every pcc_hidden_L* = 1.0000000 (max abs 9.6e-3 at L39), pcc_logits 1.0000000, top1_match 1.0000,
  text next-token acc 0.746. Full run 1 m 50 s.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.reference.check_hf --seq 2048`

## PL.1 plan (run1, attempt 1 on the 4x2 mesh)
- Wrote `plan.yaml`, `plan.md`, `components.yaml` for SP=4 (rows, axis 0) x TP=2 (columns, axis 1), the Kimi K2.7
  layout of deepseek_v3_d_p. Chip (r, c): chunk rows [r S/4, (r+1) S/4) and hidden columns [1792c, 1792(c+1)); the 4 mHC
  streams stay [S/4, 4 x 1792] fp32 per chip (split by row and column, as Kimi's TP-sharded hidden and hy4).
- Attention: ttMLA's dense chunked path (ring_mla over axis 0, 16 heads per chip = Kimi's count, absorbed 576 / 512),
  latent cache block-cyclic over rows with period = chunk, replicated over columns (dense ring_mla is not TP-dedup
  wired). q_a / kv_a K-split + all_reduce axis 1, o_proj row-parallel + reduce_scatter axis 1.
- Causal balance with a contiguous quarter per row: last chunk slowest / mean 1.036, whole s56320 run 1.068. Zigzag
  not planned (chunked ring_mla asserts is_balanced False).
- MoE: EP=8, one dispatch group of 4 chips per column (experts 32c + 8r .. +7), forks dispatch / combine /
  offset_cumsum / unified_routed_expert_moe, reduce_scatter over axis 1 after combine. Experts bfp8 (bf16 fits too).
- mHC: partial projection + one [S/4, 32] fp32 all_reduce axis 1; Sinkhorn via a new fork
  `ttnn.bringup.mhc_split_sinkhorn` (Xing mode) or composed, because the stock kernel differs (known issue).
- Gate: per-chip 9.63 of 27.20 GiB, 0 unplaced, 0 plan / component / ledger errors; plan_approved 0 until the owner
  approves. tasks.yaml unchanged (F48 lets the attn_hc task write ttnn/ttnn/bringup).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.plan.check_plan`

## C.dense.attn_hc test (run1, attempt 1)
- The unreviewed freeze failed: the auto sweep let noise1e-2 through (limit 0.0148 set by the bf16-everywhere model,
  whose bf16 comb logits give comb rel 0.0077). Reviewed the test: kept checks="auto" (PCC gate, second inputs
  layer39 / mixed / small / big) and added extra checks on the golden input: rel L2 per part vs the CPU step
  (pre 0.006, post 0.03, comb 0.012) and vs the golden (+ the fp32 step's own error), comb column sums within 0.01 of 1,
  ranges pre (0, 1], post [0, 2], comb [0, 1] (slack 1e-3).
- Measured on the golden (2048 rows): tf32 / bf16 projection operands give part rel <= 0.0009 (device on the fp32 plan);
  bf16 everywhere 0.0024 / 0.0118 / 0.0077; noise1e-2 0.0114 / 0.092 / 0.007 with colsum 0.022 and post min -0.011.
  20 Sinkhorn iterations do not converge: comb rows sum to 1 +- 0.11, columns to 1 exactly. 19 iterations
  (comb rel 0.0064), the clamp, the row-max subtraction and hc_eps are not visible on this golden.
- BRINGUP_IMPL=mutations now runs the auto sweep (printed, it still shows noise SLIPPED) and then the test's own sweep:
  all 7 float mistakes caught by auto or extra checks, reference / bf16 controls pass. reference PASS, stub FAIL.
  Device mode fails with NotImplementedError until the implement step.
- Re-run: `BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/xing40_a4b_d_p/tests/bringup/test_c_dense_attn_hc.py`
  (also `stub`, `mutations`); the gate is the same command without BRINGUP_IMPL.

## C.dense.attn_hc implement (run1, attempt 1)
- Wrote `tt/layout.py` (SP=4 x TP=2 stream layout: chip (r, c) holds rows [r S/4, ..) and hidden columns
  [1792 c, ..) of each stream, stream-major [1, 1, S/4, 4 x 1792] fp32; host helpers at the harness boundary only) and
  `tt/mhc.py:TtHcWeights` / `build_hc` (attn_hc and ffn_hc, any layer). Hooks: `device_component` (attn_hc, ffn_hc),
  `DEVICE_STEPS = {"dense": {"attn_hc"}}`, `HybridDeviceModel` (CPU reference + the device steps) as `device_model`.
- Projection as in hy4 TtHcGates: fn^T permuted to chip-major and split over axis 1, partial mixes (cols 0-23) plus
  the partial sum of squares (col 24) in one [S/4, 32] fp32 tile row, one `ttnn.all_reduce(cluster_axis=1)`, then
  rsqrt(ss / 14336 + 1e-6), x scale + base on [1, 32] row constants, sigmoid x (1, 2).
- Sinkhorn composed, not forked (the brief allows either): comb logits sliced to [S/4, 16] (a slice, since a
  selector matmul would round logits to tf32, 0.015 abs at 30), `ttnn.clamp(-30, 30)`, row max = `ttnn.maximum` of
  the logits and 3 within-row rotation matmuls (tf32-rounded max; it only shifts exp and cancels in the first row
  normalisation), `ttnn.exp`, then 20 x (row, column) as `ttnn.divide(m, ttnn.linear(m, RB|CB, bias=eps))`.
  All matmuls HiFi4 + fp32 dest. No host work in the forward; all constants built at load.
- Ops per call: ~110 (80 in the Sinkhorn loop). Folding the eps into the linear's bias saved 40 ops at a small cost
  (comb rel vs cpu 0.00066 -> 0.00087, worst column sum 0.00095 -> 0.00164). The fused fork
  (`ttnn.bringup.mhc_split_sinkhorn` with a Xing mode) is still the way to cut this, left for a perf step. Note that the
  stock kernel's elementwise ops run on the FPU from fp32 CBs (no UnpackToDestFp32), so a fork should check its own
  precision against these numbers.
- Gate: pcc_attn_hc_L00 0.999998; part rel vs cpu pre 0.00015 / post 0.00065 / comb 0.00087 (limits 0.006 / 0.03 /
  0.012); comb column sums 0.00164 (<= 0.01); auto second inputs (layer39, mixed, small, big) rel <= 0.00087.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/xing40_a4b_d_p/tests/bringup/test_c_dense_attn_hc.py`

## C.dense.attn_collapse implement (run1, attempt 1)
- Wrote `tt/collapse.py:TtHcCollapse` / `build_collapse` (attn_collapse and ffn_collapse, weightless): per chip, 4
  `ttnn.slice` of the stream-major [S/4, 4 x 1792] fp32 streams, 4 single-column slices of the replicated [S/4, 24]
  hc, then `ttnn.multiply` + 3 x `ttnn.addcmul` in fp32 (glm53 TtHcCollapse / deepseek `_mix`). Output
  [1, 1, S/4, 1792] fp32, rows split over axis 0, columns over axis 1; no collective, no host work, no constants.
- Added `tt/layout.py:col_split_to_host` (harness boundary only). Hooks: `_COLLAPSE_STEPS` + `_collapse_host_fn`
  (streams via `streams_to_device`, hc via `row_split_to_device`), `DEVICE_STEPS["dense"] = {attn_hc, attn_collapse}`.
- Output kept fp32 (the plan's attn_in, input to the distributed attn_norm); `out_dtype` option exists for bf16.
- Gate: pcc_attn_collapse_L00 0.999998; vs CPU rel 0 (bit-exact fp32 elementwise) on golden and all 4 second inputs;
  vs golden rel 0.0018. The precompile collect pass prints a `FAIL pcc=0` line first (stubbed device results, known
  issue); the real pass line counts.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/xing40_a4b_d_p/tests/bringup/test_c_dense_attn_collapse.py`

## C.dense.attn_norm implement (run1, attempt 1)
- Wrote `tt/norm.py:TtDistributedRmsNorm` / `build_norm` (adapted from hy4 tt/norm.py, itself from deepseek_v3_d_p):
  `rms_norm_pre_all_gather` (fp32 stats) -> multiply by a [1, 32] one-hot column-0 mask (fp32-input stats junk,
  known issue) -> `all_gather(dim=3, cluster_axis=1, Linear)` -> `rms_norm_post_all_gather` with the weight split by
  column (fp32 row-major [1, 1, H/32, 32], dim 2 over axis 1), eps 1e-6, HiFi4 + fp32 dest. In: the collapse output
  [1, 1, S/4, 1792] fp32; out: [1, 1, S/4, 1792] bf16, column-split for the K-split q_a / kv_a projections.
  Weight and mask built at load; no host work in the forward.
- Added `tt/layout.py:col_split_to_device` (harness boundary). Hooks: `_NORM_STEPS` + `_norm_host_fn`,
  `DEVICE_STEPS["dense"]` now {attn_hc, attn_collapse, attn_norm}.
- Gate: pcc_attn_norm_L00 0.999996; vs CPU rel 0.0018 (limit 0.0062), ratio [0.9987, 1.0003]; vs golden rel 0.0029;
  second inputs (layer39, mixed, small, big) rel <= 0.0018. The small negative bias (-0.0005) is the bf16 output.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/xing40_a4b_d_p/tests/bringup/test_c_dense_attn_norm.py`

## C.dense.q_a implement (attempt 1)
- Added `tt/q_a.py:TtQa` / `build_q_a`, adapted from `hy4_preview_d_p/tt/q_a.py:TtQa`: q_a_proj^T K-split over axis 1
  ([1792, 768] bf16 per chip) -> `ttnn.linear` HiFi4 + fp32 dest, fp32 partial -> `ttnn.all_reduce(cluster_axis=1)`
  (fp32, [S/4, 768]) -> `ttnn.bringup.rms_norm` (q_a_layernorm, eps = cfg.rms_norm_eps 1e-6, fp32 row-major weight)
  -> typecast bf16. Output row-split over axis 0, replicated over axis 1. Weights built at load; no host work in forward.
- Chose the fork norm over native ttnn.rms_norm (native scales rows ~0.1% low on fp32 input, known issues);
  `norm_impl="native"` kept for comparison. all_reduce instead of reduce_scatter + all_gather: no persistent buffers.
- Hooks: `_QA_STEPS` + `_qa_host_fn` (bf16 `col_split_to_device` in, `row_split_to_host` out),
  `DEVICE_STEPS["dense"]` now {attn_hc, attn_collapse, attn_norm, q_a}.
- Gate: pcc_q_a_L00 0.999998; vs CPU rel 0.00175 (limit 0.0066), ratio [0.99895, 1.00058]; vs golden rel 0.00215;
  second inputs (layer39, mixed, small, big) rel <= 0.00178.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/xing40_a4b_d_p/tests/bringup/test_c_dense_q_a.py`

## C.dense.attention implement (run1, attempt 2 of the step; brief attempt 1)
- Took attempt 1's wip (`runs/run1/wip_attention_attempt1`: attention.py, hooks.diff) into `tt/attention.py`
  (`TtMlaAttention` / `build_attention`) and `bringup/hooks.py` (`_AttentionHostFn`, stateful: reloads the golden
  prefix per call in component / swap tests, persists across chunks in the hybrid; `_HybridState` reads kv_latent
  back from the device; `DEVICE_STEPS["dense"]` += attention). Dropped the all_gather + chunked SDPA fallback (owner:
  ring_mla over axis 0 only).
- Pipeline per chip (r, c): kv_a K-split linear fp32 -> all_reduce axis 1 -> ttnn.bringup.rms_norm (latent) + typecast,
  rotary_embedding_indexed (k_rope, block-cyclic YaRN tables) -> concat -> update_padded_kv_cache (bf16 TILE cache
  [max_seq/4, 576], block-cyclic, TP-replicated); q_b linear -> nlp_create_q_heads_split -> W_uk batched linear |
  RoPE -> ring_mla (cluster_axis 0, Linear, head_dim_v 512, kv_actual_isl = start, logical_n = start + chunk) -> W_uv
  -> nlp_concat_heads -> o_proj fp32 -> reduce_scatter axis 1. No host work in the forward.
- L1: ring_mla at q32 / k512 overflows (2.15 MB of CBs; the cache is bf16, ttMLA's Kimi k640 is on bfp8). k256 fits.
- Owner setting first (ttnn.transformer.ring_mla, HiFi4, fp32 dest off): pcc 0.999988, vs cpu rel 0.0048 (limit
  0.0056), but the auto `big` check (inputs x2) failed: worst row 0.0537 > 0.045, at k128 / k256 / q64 alike. A device
  probe (runs/run1/attention_probes/probe_001.py; CPU kernel model sim_dest_accum.py) split the error: q_abs / cache at bf16
  rounding (rel 0.002 / 0.003), post-SDPA 0.0006, and ring_mla vs float32 on its own inputs rel 0.024, worst row 0.25
  (latent space). Scores reach 93 there. A CPU model shows the excess is the 16-bit DEST accumulation of QK^T (bf16
  running state adds ~0). So only fp32 DEST fixes it, and the source op refuses fp32 DEST for latent V.
- Extended the sdpa fork (host only, no new argument): ring_mla (latent V) at fp32 DEST, which the source refuses,
  now takes the streaming path, with its bf16 intermediate CBs (fork CHANGELOG). Rebuilt. New
  `ttnn/ttnn/bringup/sdpa/tests/unit/test_ring_mla_fp32_dest.py` (7 passed; bf16 dest bit-identical to the source;
  1.02-scaled output fails). Regression, option off: unit suite 35 passed, fork_source 210 tests / 0 regressions. The
  model cases in tests/test_sdpa.py (1x4 / 2x2 meshes) cannot open on this 4x2 box (fabric router sync timeout).
- Default is `XING_MLA_SDPA=fork` (ttnn.bringup.ring_mla, HiFi4 + fp32 dest). `XING_MLA_SDPA=source` is the owner's
  06:35 bf16-dest setting, which fails the frozen test's big check. This departs from the owner's "fp32_dest_acc_en=False"
  instruction: the owner should confirm, or relax the check.
- The ring_mla gather scratch [max_seq, 576] bf16 (65 MB at 56k) is shared by every layer (per mesh and max_seq).
- Gate: pcc_attention_L00 0.999993; vs cpu rel 0.0031 (limit 0.0056), worst row 0.0077; vs golden rel 0.0036;
  chunk0 0.0030, layer39 0.0027, mixed 0.0018, small 0.0046, big 0.0081 / worst row 0.035 (limit 0.045).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/xing40_a4b_d_p/tests/bringup/test_c_dense_attention.py`
  (`XING_MLA_SDPA=source` for the bf16-dest comparison); fork test: `scripts/run_safe_pytest.sh --run-all
  ttnn/ttnn/bringup/sdpa/tests/unit/test_ring_mla_fp32_dest.py`

## C.dense.attn_residual implement (run1, attempt 1)
- New `tt/residual.py:TtHcResidual`: per chip, streams [1, 1, S/4, 4 x 1792] fp32 + hc [1, 1, S/4, 24] fp32 (replicated
  over axis 1) + y [1, 1, S/4, 1792] (column split) -> streams fp32. out_i = post_i y + sum_j comb[i, j] x_j, where
  post_i is hc column 4 + i and comb[i, j] is column 8 + 4 i + j (Xing comb, not glm53's comb^T). No collective, no weights.
- Default `XING_RESIDUAL_MIX=addcmul`: 4 x (multiply + 4 addcmul), all on the SFPU in fp32, so the fp32 streams stay exact.
  `XING_RESIDUAL_MIX=matmul` is glm53's block-diagonal Mix with a transposed selector (one batched HiFi4 fp32-DEST
  matmul). It also passes: pcc 0.999997, rel vs cpu 4.1e-8. It is kept for comparison and for the perf step.
- hooks: `_RESIDUAL_STEPS = {attn_residual, ffn_residual}` -> `_residual_host_fn` (harness boundary:
  streams_to_device / row_split_to_device / col_split_to_device in, streams_to_host out). `attn_residual` added to
  `DEVICE_STEPS["dense"]`, so the hybrid device_model runs it on the device.
- Gotcha: the precompile collect pass prints `FAIL pcc_... 0.000000` lines before the real pass (stubbed outputs,
  known issue "Precompile collect pass"). Only the second block of lines counts.
- Gate: pcc_attn_residual_L00 0.999997; vs cpu rel 3.7e-8; vs golden rel 0.0025 (limit 0.0069); layer39 / mixed / small / big
  rel <= 5e-8.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/xing40_a4b_d_p/tests/bringup/test_c_dense_attn_residual.py`

## C.dense.ffn_norm implement (attempt 1)
- `tt/norm.py:TtDistributedRmsNorm` got `gather=False` option; `build_norm(..., "ffn_norm")` uses
  `post_attention_layernorm.weight` and `gather=True`: same pre_all_gather -> col-0 mask -> stats all_gather (axis 1)
  -> post_all_gather (HiFi4 + fp32 dest, bf16 out), then `ttnn.all_gather(dim=3, cluster_axis=1, Linear)` ->
  [1, 1, S/4, 3584] bf16 per chip, replicated within a row. attn_norm path unchanged (gather off).
- hooks: `_GATHERED_NORM_STEPS = {"ffn_norm"}`, `_gathered_norm_host_fn` (column-split fp32 in, `row_split_to_host`
  out, column 0's copy); ffn_norm added to `DEVICE_STEPS["dense"]`.
- Gate: pcc_ffn_norm_L00 0.999996; rel vs cpu 0.0018 (limit 0.0062), norm ratio [0.9985, 1.0004].
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/xing40_a4b_d_p/tests/bringup/test_c_dense_ffn_norm.py`

## C.dense.mlp implement (attempt 1)
- Added `tt/mlp.py:TtDenseMLP` + `build_mlp(mesh, loader, cfg, layer, prefix="mlp.")`, adapted from
  hy4_preview_d_p/tt/mlp.py. Gate / up are column-parallel ([3584, 4608] per chip column, bf16), and down is row-parallel
  ([4608, 3584]). The step is `silu(g) * u` as one `ttnn.multiply` with a SILU input activation. gate / up / h are fp32, every
  matmul runs at HiFi4 with fp32 dest. After that comes `ttnn.reduce_scatter(dim 3, cluster_axis 1, Linear)`, giving [S/4, 1792] fp32 column split.
- Input: the ffn_norm output, row-split and replicated over axis 1 (bf16 at the harness boundary). Output: the column split
  that ffn_residual consumes.
- hooks.py: `_MLP_STEPS`, `_mlp_host_fn`. "mlp" is added to DEVICE_STEPS["dense"]. The `prefix` arg lets the MoE
  shared expert (`mlp.shared_experts.`) reuse the module.
- Gate: pcc_mlp_L00 = 0.999997, and all the auto checks pass (rel vs cpu 6.3e-4).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/xing40_a4b_d_p/tests/bringup/test_c_dense_mlp.py`

## C.moe.attention test (run1, attempt 1)
- Previous freeze sweep: noise1e-2 SLIPPED (rel 0.0100 < calibrated golden limit 0.0135 = 2 x bf16-model rel 0.0068).
- The device module already at hand (the dense attention module, ring_mla fork, fp32 dest; moe attention is not in
  DEVICE_STEPS yet, but device_component builds it) measured rel 0.0108 vs cpu at layer 2 (pcc 0.999942). That is
  above the noise, so tightening the rel limit would fail a correct device.
- Added an extra check: the error vs the CPU step outside the top K=1024 right singular vectors of the CPU output,
  rel to the output norm, <= 0.006 on golden / chunk0 / layer39. Measured: device 0.0027 / 0.0027 / 0.0024, bf16
  model 0.0021 / 0.0023 / 0.0020, noise 0.0085. Metrics `off_subspace_rel_attention_<case>_L02`. Under
  BRINGUP_IMPL=mutations the test runs its own sweep (auto + extra): every mistake is caught, the controls pass.
- Gotcha: cache the SVD basis on the Expect object; the precompile collect pass builds a different Expect in the same process.
- The moe-attention implement step must stay close to the dense module's precision (off-subspace 0.0027, whole rel
  0.0108 against a limit of 0.0135).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/xing40_a4b_d_p/tests/bringup/test_c_moe_attention.py`
  (and `BRINGUP_IMPL=mutations|reference|stub` for the freeze checks).

## C.moe.router implement (run1, attempt 1)
- New `tt/router.py:TtRouter` + `build_router(mesh, loader, cfg, layer, max_rows)`, from glm53_flash_d_p's TtRouter:
  fp32 `ttnn.linear` (gate.weight^T [3584, 64] fp32 replicated, HiFi4 + fp32 dest) -> sigmoid -> add (bias recentred
  by its mean at load) -> `ttnn.topk` k=4 (fp32 keys, sorted) -> gather unbiased scores -> sum, (+1e-20) x (1/2.0) ->
  div. Returns (dense [1, 1, S/4, 64] fp32 ROW_MAJOR, idx uint16 [S/4, 4], weights fp32 [S/4, 4]) per chip, row split,
  replicated over axis 1. No CCL.
- Gotcha: `ttnn.scatter` rejects fp32 TILE, so the dense matrix is scattered in ROW_MAJOR (fp32 zeros built RM at
  load for max_rows, idx / weights to_layout RM per chunk). `dense_dtype=ttnn.bfloat16` gives the GLM-style TILE path.
- max_rows = max chunk over ladder + target / mesh rows (8192 / 4 = 2048), `hooks._max_rows`.
- hooks: `_ROUTER_STEPS = {"router"}`, `_router_host_fn` (row-split bf16 ffn_norm in, as ffn_norm's device output;
  dense read back from column 0). `DEVICE_STEPS["moe"] = {"router"}`, so the hybrid device_model runs it.
- Gate: pcc_router_L02 0.999496; vs cpu overlap 0.99939, matched 0.99756, rel 1.2e-4; vs golden overlap 0.99841,
  rel 0.0018; layer39 / mixed overlap 0.99963 / 0.99976.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/xing40_a4b_d_p/tests/bringup/test_c_moe_router.py`

## C.moe.experts implement (run1, attempt 1)
- New `tt/experts.py:TtExperts` + `build_experts(mesh, loader, cfg, layer, max_chunk)`, adapted from
  `hy4_preview_d_p/tt/experts.py:TtHy4Experts`: masked_bincount -> `ttnn.bringup.offset_cumsum` (axis 0) ->
  `ttnn.bringup.dispatch` (dispatch_group_size 4, cluster_axis 0, Topology.Linear on the FABRIC_2D mesh) ->
  `ttnn.bringup.unified_routed_expert_moe` (8 local experts, `RoutedExpertActivation.Silu`, high_precision, HiFi4 +
  fp32 dest, bfp8 weights, ROW_MAJOR bf16 x) -> `ttnn.bringup.combine` (axis 0, init_zeros) -> post_combine_reduce
  (dispatch table masks the other column's experts) -> typecast fp32 -> `reduce_scatter(dim 3, cluster_axis 1)` ->
  [1, 1, S/4, 1792] fp32 column split. Column c = group c = experts 32c..32c+31, chip (r, c) holds 32c + 8r .. +7.
- The forks needed no change for a dispatch group of 4 chips (only groups of 1 and 2 were used before).
- Capacity factor 4 (= top-k); per-expert cap and dispatch/combine sizes from `compute_constants`; max_seq_len =
  `hooks._max_chunk` (8192, the s16384 rung). Dispatch / combine modules are cached per (rows per chip, mesh).
- Weights: `LazyExpertWeights` reads gate / up / down one expert at a time (bf16 in checkpoint), bfp8 tensorbin cache
  under `generated/xing40_a4b_d_p/tt_cache/experts` (first run builds it).
- Gotcha: post_combine_reduce already returns TILE, so `ttnn.to_layout(red, TILE)` shares its buffer; freeing `red`
  broke the next typecast (known issue proposed).
- hooks: `_EXPERTS_STEPS`, `_experts_host_fn` (dense routing -> host topk -> idx uint16 / wts fp32 row-split, like the
  router's outputs; column-split read-back), `DEVICE_STEPS["moe"] = {router, experts}`.
- Gate: pcc_experts_L02 0.999960; vs cpu rel 0.0086 (limit 0.0137), worst row 0.0155, ratio [0.9954, 1.0052]; vs golden
  0.0090; layer39 0.0132 / mixed 0.0128 / small 0.0082 / big 0.0087 (limit 0.03). `XING_EXPERTS_MODE=loop`: pcc
  0.999962, vs cpu rel 0.0084, same accuracy.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/xing40_a4b_d_p/tests/bringup/test_c_moe_experts.py`
  (`XING_EXPERTS_MODE=loop` for the per-expert path).

## C.moe.shared_expert implement (run1, attempt 1)
- No new module: the shared expert is `tt/mlp.py:TtDenseMLP` built by `build_mlp(..., prefix="mlp.shared_experts.")`
  (intermediate 1024 -> 512 per column; gate / up column-parallel, down row-parallel, bf16 weights, fp32 mid, HiFi4 +
  fp32 dest, fp32 `reduce_scatter(dim 3, cluster_axis 1)` -> [S/4, 1792] column split). Same boundary as the dense mlp.
- hooks: `_SHARED_EXPERT_STEPS = {"shared_expert"}` -> `_mlp_host_fn`; `DEVICE_STEPS["moe"]` now has shared_expert.
- The reduce_scatter stays separate from the experts' one (swap-test boundary); fusing them (add partials first) is a
  perf item.
- Gate: pcc_shared_expert_L02 0.999999; vs cpu rel 6.4e-4 (limit 0.0046), ratio [0.99924, 0.99963]; vs golden 0.0018;
  layer39 / mixed / small / big rel 7.7e-4 / 7.8e-4 / 6.8e-4 / 6.4e-4. (The precompile collect pass prints FAIL lines
  with pcc 0; only the real pass counts.)
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/xing40_a4b_d_p/tests/bringup/test_c_moe_shared_expert.py`

## C.moe.moe_add implement (run1, attempt 1)
- New `tt/moe_add.py:TtMoeAdd` + `build_moe_add(cfg)`, from glm53_flash_d_p's TtMoeAdd, but fp32 output (default
  `out_dtype=ttnn.float32`): `ttnn.add(experts_out, shared_out)` on column-split [1, 1, S/4, 1792] fp32, DRAM. No CCL,
  no weights.
- hooks: `_MOE_ADD_STEPS = {"moe_add"}` -> `_moe_add_host_fn` (both inputs column-split fp32 in, column-split out);
  `DEVICE_STEPS["moe"]` now has moe_add, so the hybrid device_model runs it.
- Gate: pcc_moe_add_L02 0.999997; vs cpu rel 0 (exact fp32 add); vs golden rel 0.0022 (limit 0.0057); layer39 /
  mixed / small / big rel 0. (Precompile collect pass prints FAIL lines with rel 1; only the real pass counts.)
- Next (perf): the experts' and shared expert's reduce_scatters could be fused by adding partials before one RS.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/xing40_a4b_d_p/tests/bringup/test_c_moe_moe_add.py`

## M.1 assemble (run1, attempt 1)
- New `tt/model.py`: `TtXingModel` (TtEmbedding -> TtXingBlock x N -> TtFinalNorm) and `TtXingBlock` (glm53's
  TtGlmBlock pattern: step fns keyed by the reference block graph `xing_ref.HC_ATTN + HC_FFN + DENSE/MOE_FFN +
  FFN_RESIDUAL`, run through `run_block`, every boundary freed after its last reader). The step modules are the
  validated ones, built by the same `build_*` functions the component hooks use; no module was changed.
- Residual on the device: [1, 1, S/4, 4 x 1792] fp32 per chip (tt/layout.py streams layout), from the embedding to
  the final norm. Boundaries are the hybrid harness's minus the host; the one difference: the router step hands
  `(idx uint16, wts fp32)` straight to the experts and frees its dense [S/4, 64] matrix (the hybrid rebuilt top-k
  on the host from the dense matrix).
- TtEmbedding: bf16 table split by hidden columns over axis 1 ([131072, 1792] per chip, ROW_MAJOR, tensorbin
  `generated/xing40_a4b_d_p/tt_cache/embed_bf16_colsplit`), ids [1, 1, 1, S/4] uint32 row split, typecast fp32,
  concat x4 for the streams. TtFinalNorm: fp32 sum of the 4 stream slices x 1/4 -> TtDistributedRmsNorm
  (model.norm.weight, fp32 out) -> [S/4, 1792] column split. LM head on the host (ladder sampled rows).
- hooks: `XingDeviceModel` is the `device_model` default; `BRINGUP_HYBRID=1` keeps `HybridDeviceModel`.
  `_DeviceState(model, max_seq)` (new_state, outside the forward) calls `setup(chunk, max_seq)` on every block for
  the chunk(s) of the rungs / target with that seq (attention RoPE tables, latent cache, gather scratch, SDPA config;
  experts' dispatch / combine) and zeroes the caches, so prefix loads (rung `last`) have a geometry and warm chunks
  do no host work. load_prefix / to_torch go to `TtMlaAttention.load_state` / `read_state`.
- Gotcha: if two rungs with the same seq used different chunks, the first listed is the current geometry; a block
  asserts the geometry matches the chunk it is called with.
- Re-run: `PYTHONPATH=$PWD BRINGUP_RUNG=s4096 scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_ladder.py`
  (`BRINGUP_HYBRID=1` for the hybrid harness).

## L.s16384 fix.1 (ladder s16384, chunk 8192)
- Failure was environmental: `$HOME` (9.4G quota) was 100 % full, so JIT builds of the dispatch fork kernels
  failed (`No space left on device` under `~/.cache/tt-metal-cache`). No model code changed.
- Freed space: `uv cache clean` (3.2G uv download cache); removed the partly written
  `kernels/{reader_worker_dispatch,writer_worker_dispatch,writer_sender_dispatch}` under
  `~/.cache/tt-metal-cache/1149597549032367152` (truncated ELFs). Home at 64 % after.
- Gate result (real pass; the zeros printed first come from the precompile collect pass): worst layer pcc 0.9934,
  worst state pcc 0.9973, final hidden pcc 0.9985, top1 0.967, top5 1.0, host transfers per layer 0.
  Chunk 1 [8192,16384) took 89 s warm.
- Watch: the JIT cache (~3 GB, one 2.8G hash dir) keeps growing on the home quota; check `df -h $HOME` before long rungs.
- Re-run: `PYTHONPATH=$PWD BRINGUP_RUNG=s16384 scripts/run_safe_pytest.sh --run-all models/demos/common/bringup/tests/test_ladder.py`

## K.1 contract (run1, attempt 1)
- Previous attempt failed only because `xing40_a4b_d_p` was not in `ADAPTER_PATHS`; registered it
  (`models/demos/common/prefill/adapter.py`) -> `models.demos.xing40_a4b_d_p.tt.runners.adapter:XingPrefillAdapter`.
- New `tt/runners/adapter.py` (XingPrefillAdapter + XingPrefillRuntime, pattern of hy4_preview_d_p's runners) and
  `tt/runners/kv_contract.py` (XingContractKV). The engine cache IS the model state: one
  `init_kvpe_cache(576, sp_axis 0, tp_axis None)` bf16 TILE cache, [num_users * L, 1, max_seq/4, 576] per chip,
  batch = slot * L + layer row, block-cyclic over the 4 rows with period = the served chunk, replicated over the columns
  (same layout as the attention's geometry cache).
- `tt/attention.py`: new serving option `TtMlaAttention.bind_cache(cache, slot, row, rows)` / `unbind_cache`; `_cache()`
  feeds `update_padded_kv_cache(slot_idx, layer_idx, num_layers)` and `ring_mla(kv_cache_batch_idx = slot*rows+row)`.
  Unbound (default, ladder / component tests) = the geometry cache at batch 0, unchanged behaviour.
- Table: config "0", one 32-token entry = 18 bf16 tiles (36864 B); ROUND_ROBIN_1D walk (bank j % B, offset j // B);
  one device group per mesh row with both column replicas (as DeepSeek's TP-replicated table).
- Engine input [4, 1, chunk/4] sharded over axis 0 is already the embedding's contiguous row split: on the device
  `ttnn.minimum(ids, V-1)` then reshape to [1, 1, 1, chunk/4]. Ack after `event_synchronize` per layer, global layer id.
- Runtime builds TtXingModel with max_rows = chunk/4, max_chunk = chunk and `setup(chunk, max_seq)` once; compile()
  warms every chunk offset in slot 0.
- Gate: PASS in 69 s; producer kv_latent min PCC 0.99761 (layer 31) over [0, 4064), slot 1; no contract failures.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_contract.py`

## P.2 perf (run1, attempt 1): fused mHC residual (ttnn.bringup.mhc_post)
- Part (1) done. Fork `mhc_post_ttnn` got an opt-in `comb_transposed=False` (default True, the same program as before):
  X'_j = post_j F + sum_i comb[j*n + i] X_i. The kernel change is in `mhc_post_coef_expand.hpp` (raw column
  j*n + (t-1)), behind the define `MHC_POST_COMB_DIRECT`. Only the DM kernels get that define, and only when the
  option is off. The option is part of the program hash. Rebuilt with `./build_metal.sh`.
- `tt/residual.py`: new mode `fused`, now the default of XING_RESIDUAL_MIX. It slices post (hc columns 4..7) and comb
  (8..23) and calls `ttnn.bringup.mhc_post(y, x, post, comb, comb_transposed=False)` on the fp32 column-split
  streams as they are. `XING_RESIDUAL_MIX=addcmul` gives the old path back. The profile records it as
  `settings.residual_mix`.
- Measured (BRINGUP_PROFILE_OPS profile, warm chunk [51200, 56320)): attn_residual 90.0 -> 10.9 ms, ffn_residual
  90.5 -> 10.9 ms. Device time per chunk 1413 -> 1255 ms. attn_hc 39.2, ffn_hc 39.1 and the collapses 22.1 each are
  unchanged. Ladder `last`: worst layer pcc 0.9924, final hidden 0.9985, top1 0.974.
- Fork tests: new case `xing40_a4b_d_p-4x2-s1280-c1792-n4-fp32-comb` (fp32, pcc 1.0, rel ~0). It also asserts
  that the comb and comb^T references differ. The fixture now opens the case mesh that matches the box (2x2 on the
  4x2 box fails the FABRIC_2D handshake). Regression with the option off: unit 164 passed, golden 208 passed.
- Part (2) (mhc_pre for hc + collapse) NOT done. It needs a new program entry: the reader and compute take the
  reduced [S/4, 32] row instead of running the op's multi-core projection -> gather -> mcast. It also needs a
  different Sinkhorn: clamp, exp(L - rowmax) with no initial softmax + eps, 20 x (row, col) instead of
  softmax + col + 19 pairs, and no +eps on pre. That Sinkhorn is hand-scheduled SFPLOADMACRO code in
  `mhc_pre_compute.cpp` (1617 lines). The op also does not output pre, and the frozen attn_hc tests check the
  [S, 24] hc boundary, so the hc step would still have to produce pre. This is left as a separate step. It is worth
  ~120 ms per chunk (hc 78 + collapse 44).
- Re-run: `PYTHONPATH=$PWD TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000 scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_profile.py`
  (add `XING_RESIDUAL_MIX=addcmul` for the baseline). Fork: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/mhc_post_ttnn/tests/test_mhc_post_ttnn.py`.

## P.2b perf (run1, attempt 1): mHC coefficients + collapse on the mhc_pre fork (ttnn.bringup.mhc_pre_xing)
- The previous "attempt" was only the orchestrator's baseline gate run (nothing implemented; numbers = P.2's).
- Fork `mhc_pre_ttnn` got two new entries with their own program (CHANGELOG 4); `ttnn.bringup.mhc_pre` itself
  (kernels, host code, hash) is unchanged, so every existing call keeps its program:
  - `ttnn.bringup.mhc_pre_xing_pack(mix, streams)`: partial mix row [S/4, 32] (the matmul's) -> column 24 = sum x^2
    of the chip's streams, one exact fp32 SFPU pass. Replaces multiply + sum + one-hot multiply + add.
  - `ttnn.bringup.mhc_pre_xing(red, None, scale, base, norm_width=4H, ...)`: after the axis-1 all_reduce, RMS
    scale, sigmoid pre (no eps), 2 sigmoid post, clamp [-30, 30], exp(L - rowmax), 20 x (rows, columns) with
    /(sum + eps) -> hc [S/4, 24] (same boundary / layout as before; residual still slices post / comb).
    `coefficients_given=True` with streams = the collapse (y = sum pre_i x_i). A one-call hc + y mode exists but the
    model keeps two steps so each frozen test tests exactly what the model runs.
  - Scalars (scale, base, eps, clamp, iters) are runtime args patched per call: one program for all 80 hc calls.
- Model: `tt/mhc.py` and `tt/collapse.py` switch on `XING_HC_IMPL` = fused (default) | composed (the P.2 op chains).
  Recorded in the profile as `settings.hc_impl`. hc step = matmul (HiFi4 fp32, unchanged) -> pack -> all_reduce ->
  mhc_pre_xing: 5 programs per chip per call instead of 112.
- Measured, device time per call (probe, 4x2, 1280 rows): hc 977 -> 353 us (matmul 211, pack 106, all_reduce 18,
  mhc_pre_xing 21); collapse 548 -> 134 us. Profile warm chunk [51200, 56320): attn_hc 39.2 -> 14.3 ms, ffn_hc
  39.2 -> 14.3 ms, attn_collapse 22.1 -> 5.3, ffn_collapse 22.1 -> 5.3; device 1255 -> 1172 ms, wall 1349 -> 1250.
- Accuracy: mhc_pre_xing vs float64 reference rel ~1e-7. Frozen attn_hc L0: pre / post / comb rel vs cpu 0.00015 /
  0.00065 / 0.00034 (composed: comb 0.00087); collapse rel 0 vs cpu. Ladder `last`: worst layer pcc 0.9928, final
  hidden 0.9985, top1 0.977, host transfers 0.
- Next perf on hc: the HiFi4 fp32 matmul [1280, 7168] x [7168, 32] is 211 us of the 353 (auto and 1D-mcast configs
  both ~200 us; 40 output tiles -> 40 cores). A split-K projection inside the fork (as mhc_pre's group combine)
  would be the next step.
- Gotchas: tt-probe.sh writes probes into tests/ttnn/unit_tests/operations/<op>/ (deleted before finishing);
  rms_norm_pre_all_gather overflows L1 on the 7168-wide fp32 row; new compute kernels need api/compute/common.h,
  cb_api.h, reg_api.h + `using namespace ckernel` (known issues proposed). The fork's glm53 model-case fixture
  now opens the whole box when its 2x2 case mesh does not match (2x2 on 4x2 fails the fabric handshake).
- Re-run: gate command of the brief; fork: `scripts/run_safe_pytest.sh --run-all
  ttnn/ttnn/bringup/mhc_pre_ttnn/tests/test_mhc_pre_xing.py`; baseline: `XING_HC_IMPL=composed` with the profile.

## P.3 perf (run1, attempt 1): routed experts at HiFi2 (owner pick) — FAILS accuracy, stopped
- The fork `ttnn.bringup.unified_routed_expert_moe` already honours compute_kernel_config.math_fidelity with
  `high_precision=True`, so there is no fork change. The switch is in `tt/experts.py`: `XING_EXPERTS_FIDELITY` =
  hifi4 (default) | hifi2. The profile records it as `settings.experts_fidelity` (hooks.py `settings`).
- Frozen `test_c_moe_experts.py` at HiFi2 (layer 2) fails:
  - `FAIL auto_experts_L02_vs_cpu: rel=0.0167674 row=0.0253368 ratio_min=0.977155 ratio_max=0.994691 bias=-0.0136182 rel_limit=0.0136675 (rel 0.01677 > 0.01367; row norm ratio [0.97716, 0.99469] outside 1 +- 0.0150; median row norm ratio off by -0.01362 (limit 0.0040))`
  - `FAIL auto_experts_L02_vs_golden: rel=0.0169449 row=0.0267309 ratio_min=0.975406 ratio_max=0.994536 bias=-0.0134388 rel_limit=0.0161675 (rel 0.01694 > 0.01617; row norm ratio [0.97541, 0.99454] outside 1 +- 0.0150; median row norm ratio off by -0.01344 (limit 0.0040))`
  - `FAIL auto_experts_L02_small: rel=0.0162137 row=0.0285502 ratio_min=0.973308 ratio_max=0.993443 bias=-0.0134467 rel_limit=0.03 (row norm ratio [0.97331, 0.99344] outside 1 +- 0.0225)`
  - pcc_experts_L02 0.99995 passes. HiFi4 baseline (C.moe.experts): bias ~0, ratios [0.995, 1.008].
- Cause: HiFi2 truncates operand mantissas, which gives a systematic ~-1.4% norm bias through the chained
  gate/up -> down matmuls (known issues 26; new proposed bullet).
- Not profiled (accuracy failed). The default is back on HiFi4, so the tree runs the previous path; HiFi2 stays
  opt-in for the owner.
- Re-run: `PYTHONPATH=$PWD XING_EXPERTS_FIDELITY=hifi2 scripts/run_safe_pytest.sh --run-all models/demos/xing40_a4b_d_p/tests/bringup/test_c_moe_experts.py`

## X.3 fix (run1, attempt 1): test_positions
- The gate's ladder s56320 and profile passed earlier; test_positions failed at `new_state(5120)`:
  `hooks.py:_chunks_for` only knows ladder/target seqs, and positions.py asks for start + 5120 at every position.
- With a temporary hooks.py fallback (`_chunks_for` -> [target.chunk] when no rung matches; reverted, hooks.py is
  outside this step's paths), the first position hit a TT_FATAL in the sdpa fork: kv_actual_isl=0 with cache length
  == chunk (1280 == 1280 per SP row). Fixed in the fork (`ring_joint_sdpa_device_operation.cpp` invoke: drop host
  kv_actual_isl=0 when Q.seq == K.seq; CHANGELOG entry, `tests/unit/test_ring_mla_single_chunk.py`). Rebuilt.
- With both, the collect pass ran all five positions (0, 51200, ..., 204800). The real pass then ran 0, 204800,
  409600, 614400 (positions.py raises spec target.seq in memory and the module-level spec is shared by the
  precompile collect pass and the real pass) and timed out building the geometry at 819200: every
  (chunk, max_seq) `_Geometry` stays in `TtMlaAttention._geoms` (no free), so the caches accumulate.
  Timings seen in the real pass: 0->5120 586 ms, 204800->209920 3132 ms, 409600 5677 ms, 614400 8256 ms.
- Still needed (outside this step's paths): hooks.py `_chunks_for` fallback to target.chunk, a `_DeviceState.free()`
  that drops the attention geometries (needs a free in tt/attention.py), and either a harness fix in positions.py
  or `perf.positions: [0, 51200, 102400, 153600, 204800]` in spec.yaml.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/common/bringup/tests/test_positions.py`;
  fork: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/sdpa/tests/unit/test_ring_mla_single_chunk.py`.

## X.3 fix 1 (test_positions)
- Failure: test_positions calls `new_state(start + chunk)` for starts 0, 51200, ..., 204800 (max_seq 5120 .. 209920);
  `_chunks_for` only knew ladder/target seqs, so `_DeviceState` asserted "no ladder rung / target runs seq 5120".
- Fix (hooks.py): `_chunks_for` falls back to `target.chunk` when no rung matches and max_seq is a multiple of it.
  `_DeviceState.free()` calls new `TtMlaAttention.release(max_seq)` (tt/attention.py), which pops and deallocates
  every geometry for that max_seq (latent cache, cos/sin/trans) and the shared ring_mla kv scratch, so the five
  position geometries do not pile up in DRAM. Forward path untouched; ladder/profile do not call free().
- Positions alone: passed, 204800->209920 one chunk 3138 ms.
- Re-run: the X.3 gate command (ladder s56320 && profile && positions).

## O.1 optests (run1, attempt 1)
- Capture (ladder rung last): 90 distinct ttnn.bringup calls to 8 forks. The 80 mhc_pre_xing coefficient calls
  (attn_hc / ffn_hc x 40 layers) differ only in their per-layer hc_scale / hc_base float-list arguments. Then
  1 collapse, 1 pack, 1 each of mhc_post, dispatch, combine, offset_cumsum, unified_routed_expert_moe and
  ring_mla, and 2 rms_norm (kv_a 512, q_a 768).
- The previous attempt's crash (`KeyError: 'sig'` in fork_cases.py): the xing mhc_post case (task P.2) had no
  sig. Added `"sig": "f7215974bc"` (no other change). fork_cases reads only `<fork>/tests/cases.py`, so
  mhc_pre_ttnn's cases.py now also adds the sigged entries of `xing_cases.py` (with an `op` key);
  test_mhc_pre_ttnn.py runs only entries without `op`. Sigs added to the existing pack (8d5301954a) and collapse
  (de9b297083) cases. 80 new coef cases (`_O1_COEF`): each carries its captured scale / base, and the input
  row is random. test_mhc_pre_xing.py uses `case["base"]` when present.
- New cases (all 4x2, FABRIC_2D, l1_small 24576, random inputs):
  - dispatch / combine: dgs 4, s1280, h3584, e64, k4, buffer 20704, exact.
  - offset_cumsum: e64, epc 8, max count 1280.
  - unified_routed_expert_moe: Silu, high_precision, HiFi4, bfp8 weights, i1024. Group tokens 5120.
    Measured pcc 0.999996, rel 0.0030 (limit 0.008).
  - rms_norm x2: fp32 with row-major fp32 weight. Max rel 0.0030 (rtol 0.006).
  - sdpa ring_mla: new `_ring_mla` path in test_sdpa.py. q [1, 16, 1280, 576] per chip, heads split over the
    columns. Block-cyclic ND-sharded cache [1, 1, 14080, 576]. isl 51200, logical_n 56320. HiFi2 + fp32 dest,
    q64 / k256. Checked against the float32 causal reference over every row of both columns. Seeds 0-2
    measured pcc 0.99943, rel 0.034, worst row <= 0.198. Limits: pcc 0.999, rel 0.042, row 0.28. Zeroing one
    output row fails the case (checked by hand, reverted).
- On this 8-chip box the other models' 1x4 / 2x2 cases fail the FABRIC_2D handshake. The six conftest-based
  fork tests now set `require_exact_physical_num_devices` in `_device_params`, so a case runs only on a box
  of its mesh size; the 38 4-chip cases skip here. No case or limit of another model changed. The mhc tests
  keep their own fixtures, which open the whole box, and all their cases ran.
- Gate: ladder PASSED (final hidden pcc 0.9985). Fork tests: 106 passed, 38 skipped.
  `{"forks_used": 8, "fork_calls": 90, "fork_calls_uncovered": 0, "fork_tests_failed": 0}`.
- Re-run: the brief's gate command. One fork: `scripts/run_safe_pytest.sh --run-all --no-precompile
  ttnn/ttnn/bringup/sdpa/tests/test_sdpa.py -k xing40`.
