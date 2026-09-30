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
