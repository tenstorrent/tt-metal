# tencent/Hy4-preview bring-up: breadcrumbs

Append-only log, one section per task attempt: what was done, decisions and why, gotchas, the re-run command, the verdict.

## R.2 reference (attempt 1)

What was done
- `reference/hf_hy_v4/`: transformers 5.17.0's `modeling_hy_v4.py` / `configuration_hy_v4.py`, vendored because
  python_env has 5.12.1. The imports were rewritten from `from ...x` to `from transformers.x`; 5.12.1 has all of them,
  `DynamicIndexedLayer` included, so `generate` works with the indexer cache. One semantic fix, marked "bring-up fix": RoPE (below).
- `reference/hf_oracle.py` (`hooks.hf_model`): builds HYV4ForCausalLM on meta with eager attention and eager experts, applies the
  conversion_mapping renames (rename-only), keeps HF's `_keep_in_fp32_modules_strict` modules in fp32 and never reads MTP.
  Routed experts run HF's own `HYV4Experts.forward`, but `gate_up_proj` / `down_proj` are `ExpertSlab` views that read
  expert e's byte range. A layer's hit experts are prefetched with 16 parallel 8 MB preads. Mask shim: 5.12 has no
  `allow_is_causal_skip`. Precision: bf16 for the full model (sanity), fp32 for a layer prefix (check_hf).
- `reference/weights.py`: `WeightLoader` (every tensor through model.safetensors.index.json, never a shard name, so R.4
  trimming is safe) and `ExpertSlab`.
- `reference/hy4_ref.py` (`hooks.reference`): standalone chunked sparse CPU reference. It owns explicit state: kv_latent
  [max_seq, 576] (kv_a_layernorm(latent) | RoPE'd k_rope) and index_key [max_seq, 128] (full layers; [0, 128] on shared
  layers). Attention runs absorbed (q_nope through W_uk to the 512-wide latent, scores over the 576-wide key) over the
  2048 gathered keys of each query, with the sink as an extra logit, the elementwise sigmoid output gate and o_proj.
  Indexer: ReLU scores weighted per head, top-2048 per row. A row that sees at most 2048 keys keeps all of them. Indices
  are sorted ascending, -1 padded.
  MoE: grouped per expert, padded to 32 rows. iHC and the router run in fp32, the LM head in fp32.
- `hooks.hf_layers`: identity taps, so check_hf sees HF's layer output as [S, 4H] like the reference.

Block graph (every block goes through run_block; boundaries are recorded as `L{i}.<output>`)
- attn_hc -> attn_hc_pre -> attn_norm -> q_a -> indexer (full) | topk_shared (shared) -> attention -> attn_residual ->
  ffn_hc -> ffn_hc_pre -> ffn_norm -> mlp (dense), or router -> experts -> shared_expert -> moe_combine (MoE) -> ffn_residual.
- Block in/out: the 4 streams flat, [S, 4H] (stream j = columns [jH, (j+1)H), HF's flatten(2) order). attn_hc / ffn_hc
  = [S, 8] fp32 (pre 4 | post 4). topk: int64 [S, 2048]. router: dense [S, 256] fp32. Extra model-level record:
  `hc_head` [S, H] before `final_norm`.
- Shared layers (2-4) get the top-k of the latest full layer's current chunk from the reference (`Hy4Reference._topk`,
  keyed (layer, start, length)). A component or swap test that runs a shared block without its source layer must set
  `ctx.extra["shared_topk"]` (the golden's `L{src}.topk`, src = cfg.topk_source(layer); for layers 2-4 that is L1).

Decisions and why
- RoPE is interleaved (GPT-J pairs) in the MLA and in the indexer's last 64 dims, not rotate-half as in the 5.17 port.
  Evidence: with the port as released, the full-model sanity scored accuracy 0.237 (smoke still "Paris") while
  reference == HF at PCC 1.0. SGLang's Hy4 config sets rope_interleave = indexer_rope_interleave = True
  (is_neox_style=False). With interleaved RoPE: accuracy 0.965, parity unchanged. See findings.yaml R2-hf-rope-interleaved and known issues.
  For the device: no host permutation of the rope columns for Meta-order kernels (this reverses the note in
  hy4_preview_findings.md).
- q_a_layernorm / kv_a_layernorm eps 1e-6 as in HF (HYV4RMSNorm default); SGLang uses 1e-5; negligible.
- The routed experts of layers 0-5 are held in fp32 in memory (5 x 38.6 GB, about 200 GB). More than 8 MoE layers switch
  to ExpertSlab reads.
- Query-row blocks of 64 are aligned to absolute positions, so chunked prefill reproduces one-shot exactly.

Results (hand runs, generated/bringup_adhoc)
- check_hf_sanity: revision ok, smoke "Paris" ok, text_top1_acc 0.965 (17.7 min, disk-bound expert reads).
- check_hf --seq 4096 (6 layers, fp32): pcc_hidden_L00..L05 1.0000000 (maxabs <= 6.2e-5), pcc_logits 1.0000000,
  top1_match 1.0. The whole gate command took 23.7 min.
- check_reference --seq 4096 --chunk 2048 (R.3's gate, run early): hidden and state PCC 1.000000000; graph replay max
  abs diff 0 for all 3 block types (6.5 min).

Gotchas
- check_hf's maxabs needs the recorded `L{i}.out` in HF's reshape(-1, H) layout, hence the flat [S, 4H] block
  boundary plus the taps.
- The full-model sanity reads about 1.5 TB of experts per 2048-token pass; do not run it alongside another heavy host job.

Re-run
    PYTHONPATH=$PWD python -m models.demos.common.bringup.intake.check_hf_sanity && \
    PYTHONPATH=$PWD python -m models.demos.common.bringup.reference.check_hf --seq 4096
