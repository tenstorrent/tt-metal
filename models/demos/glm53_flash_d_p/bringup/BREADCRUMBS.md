# zai-org/GLM-5.3-Flash bring-up: breadcrumbs

Append-only log, one section per task attempt: what was done, decisions and why, gotchas, the re-run command, the verdict.

## R.2 reference (attempt 1 of the reference role; the previous run failed at the stock HF loader)

What was done:
- `reference/hf/`: the transformers glm5_next modeling + configuration files (commit 0291458166a6, 5.17 line),
  vendored with relative imports made absolute; five helpers 5.12.1 lacks come from `reference/hf/_compat.py`
  (kernel-hub / accelerate decorators as no-ops, `create_recurrent_attention_mask` = local 2D mask, vision stubs).
  The model code itself is unchanged.
- `reference/hf_oracle.py` (hook `hf_model`): text model only (`Glm5NextTextModel` + lm_head; no vision tower, no
  MTP), built on meta, weights assigned. Routed experts are `PackedExperts`: FP8 read from the safetensors (via the
  index) and dequantized per use, same forward as HF's experts. `num_layers=None` gives bf16 (intake sanity),
  otherwise fp32. fp32 kept for conv1d / dt_bias / A_log / e_score_correction_bias as `_keep_in_fp32_modules_strict`.
  Name mapping: `hc_{attn,ffn}_{fn,base,scale}` -> `{attn,ffn}_hc.*`, `{q,k,v}_conv1d` concatenated into `conv1d`,
  `f_a/f_b/dt_bias/A_log` -> `forget_gate.*`. `generate` is a greedy loop over `GlmCache` (see gotchas).
- `reference/glm_ref.py` (hook `reference`): standalone CPU reference, every block through `run_block`. Graph:
  attn_hc, attn_collapse, attn_norm, [KDA: attention] or [DSA: q_a, indexer, attention], attn_residual, ffn_hc,
  ffn_collapse, ffn_norm, [dense: mlp] or [MoE: router, experts, shared_expert, moe_add], ffn_residual (`out`).
  KDA runs HF's chunk-64 delta rule from the carried state (intra-chunk terms built 8 sub-chunks at a time). The
  indexer caches pooled keys and scores 128-query blocks against them. MLA uses absorbed kv_b and gathers the selected
  latent rows (topk [S, 2051] int32: 512 pools x 4 tokens + 3 tail, -1 = none). Experts are grouped per expert and
  padded to 32 rows (known issue on M-dependent sgemm).
- `reference/weights.py`: index-based loader, FP8 128x128 block dequant (fp32 product, exact).

Decisions:
- Residual-stream boundaries (`in`, `h_mid`, `out`) are `[S * 4, H]`, token-major (row 4t + n = stream n of token t):
  check_hf compares against HF's layer output flattened to `[-1, H]`, and `[S, 4, H]` fails its maxabs print.
- mHC coefficients are one boundary `[S, 24]` fp32 = pre 4 | post 4 | comb 16 (row-major, after Sinkhorn), shared
  by the collapse and residual steps.
- State: DSA `kv_latent [len, 512]`, `index_key [len // 4, 128]` (complete pools only); KDA `kda_recurrent
  [64, 128, 128]` fp32, `kda_conv [3, 24576]` (last 3 pre-conv q|k|v rows); a chunk at start 0 resets the KDA state.
  The indexer asserts chunk starts are multiples of 4. One-shot == chunked also needs starts that are multiples of 64
  (KDA sub-chunk grid); all ladder chunks are.
- Experts dequantized to fp32 at load when a subset has <= 2 MoE layers (58 GB for layers 3-4), else per use.

Gotchas:
- transformers 5.12.1 has linear-attention and indexed cache layers, but with other semantics than the 5.17 code
  expects (`update_conv_state` returns the stored state instead of [state | input]; `conv_states` is a tensor, the
  code indexes `[0]`). A 5.12 DynamicCache would silently corrupt KDA; `forward` therefore defaults to no cache and
  `generate` uses `GlmCache`.
- config.json names DSA layers `deepseek_sparse_attention`; the modeling code's mask table only has
  `indexed_attention` / `linear_attention` (KeyError). The oracle renames layer_types in the config object.
- The whole-model sanity streams ~300 GB of FP8 experts from disk; it took 10 min on this host (page cache warm).

Results (local runs): check_hf --seq 4096: L00-L04 pcc 1.0000000 (maxabs <= 5.7e-6), logits pcc 1.0000000, top-1
match 1.0000. check_hf_sanity: revision ok, smoke '</think>Paris<|user|>' ok, text_top1_acc 0.964.
Chunked (64) vs one-shot at 128 tokens: max abs 0.

Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.intake.check_hf_sanity && PYTHONPATH=$PWD python -m
models.demos.common.bringup.reference.check_hf --seq 4096` (about 14 min).
Also run locally (R.3's check, not this gate): check_reference --seq 4096 --chunk 2048: hidden pcc 1.000000000,
state min pcc 1.000000000, graph replay max abs 0 for kda_dense (L0, 10 steps), dsa_moe (L3, 15), kda_moe (L4, 13);
4 min. Gate run: 13 min 23 s, all R.2 metrics pass.
