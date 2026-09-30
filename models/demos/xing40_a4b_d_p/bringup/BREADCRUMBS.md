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
