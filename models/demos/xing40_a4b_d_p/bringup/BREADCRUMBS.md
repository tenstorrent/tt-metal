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
