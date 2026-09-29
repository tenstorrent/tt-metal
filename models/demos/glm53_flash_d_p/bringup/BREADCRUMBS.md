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

## PL.1 plan (attempt 1)

What was done:
- `plan.yaml`: placements for every checkpoint tensor (76108, whole-model map from `bringup_trim.json`): vision, MTP
  layer 45 and layers 5-44 skipped; fp8 block scales folded at load. State: MLA latent [56320, 512] and pooled keys
  (seq_stride 4) replicated for layer 3; KDA recurrent (32 heads x 128 x 128 fp32) and conv tail per chip for 0-2, 4.
  Activations 7.0 GiB (8192-token chunk, 4-stream residual replicated). Gate: 13.59 of 27.20 GiB per chip.
- `components.yaml`: all 41 steps (kda_dense 10, dsa_moe 15, kda_moe 13, model 3), NATIVE or COMPOSED, none CPU or
  OPGEN.
- `plan.md`: per-layer tables, per-chip total, collectives, departures. `tasks.yaml` unchanged.

Decisions:
- KDA is TP=2 (axis 1) x SP=2 (axis 0), because ttKDA requires distinct SP / TP axes. Fallback: SP=1 on 1x2 row
  submeshes.
- DSA attention: replicated weights, query rows split 4 ways, all 64 heads per chip (sparse_sdpa needs H % 32 == 0).
  Caches replicated, every chip computes latent and pooled keys for all S rows, one all_gather after o_proj.
- Indexer is composed, not OPGEN: key pooling is reshape + 4 slices + an elementwise softmax. indexer_score_dsa runs
  unmasked (chunk_start_idx = kv_len - 1) plus a constant pool-causal mask on the chunk's own pools. Sentinel
  compaction: for q >= 2047 the pool part has no sentinels, so [pools | tail | pad] ends in sentinels; for q < 2047
  the row is the constant "all tokens 0..q".
- Experts EP=4 via the MiMo 2x2 dispatch (axis 0, groups = columns, 72 per chip), high_precision unified kernel with
  ClampedSiluGlu. Dense MLP and shared expert TP=4 + all_reduce(cluster_axis=None). Router fp32 (MiMo TtRouter).
- mHC: DeepSeek mhc_split_sinkhorn matches GLM's Sinkhorn order and pre/post forms (checked against
  reference/mhc/mhc_reference.py); RMS eps 1e-5 is a config field.

Gotchas / to verify at implement: ttnn.topk at width 288; unified_routed_expert_moe and the dispatch tables with 72
local / 288 global experts; ttKDA's SP segment order under a chunk-aligned actual_start; sparse_sdpa scale must be
passed explicitly as 1/16 (its default is 512^-0.5); KDA recurrence and o_proj default to HiFi2 in the Kimi config
(set HiFi4); core grids from the device (p300c 11x10). The gate builds the CPU reference for layers 0, 3 and 4 (14 s
here, page cache warm).

Result: plan_fits 1, unplaced_tensors 0, plan_errors 0, component_errors 0, ledger_errors 0; plan_approved 0 (waits
for the person's approval).

Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.plan.check_plan`.
