# Mistral-Small-4-119B (mistral4) on Tenstorrent Blackhole — bring-up status

`mistralai/Mistral-Small-4-119B-2603` — a **Mistral3 VLM**: a Pixtral vision tower + a
**DeepSeek-V3-class text core** (Multi-head Latent Attention + 128-expert ungrouped top-4 MoE +
1 shared expert), fp8 weights, YaRN RoPE, 36 layers, hidden 4096, vocab 131072. Target:
**Blackhole Loudbox 1×8 (P150x8)**.

> Status: **CORRECTNESS COMPLETE** (every block PCC-verified end-to-end on P150x8, generic ttnn
> ops, no new kernels). **Performance + serving packaging in progress.** Draft README — moves into
> `models/demos/mistral4/` on relocation.

## Architecture / where the code lives
- **Vision tower + projector** — reuse `models/tt_transformers/` Pixtral path (config-driven).
- **Text core (MLA + MoE)** — model-local modules built from generic ttnn ops (MLA = linear +
  RMSNorm + interleaved-RoPE-as-permutation-matmul + SDPA; MoE = softmax→top-4/128→SwiGLU experts +
  shared expert). A DeepSeek-class core does **not** fit `tt_transformers`' GQA Generator (its paged
  attention + KV cache are GQA-hard-coded), so — matching the `models/demos/deepseek_v3/` precedent —
  the text core is a **dedicated `models/demos/mistral4/`** that reuses deepseek's MLA serving
  (`paged_flash_multi_latent_attention`, MLA KV cache, generator) for paged decode / trace / 2CQ / sampling.
- **fp8** — vanilla per-tensor + per-expert `scale_inv` dequant (`dequantize_fp8_state_dict`), distinct
  from transformers' block-wise FineGrainedFP8. Experts run on-device as `bfloat8_b` (≈ the native fp8).
- **MoE memory** — 128 experts sharded across the 8 devices (16/device, `ttnn.all_reduce` combine);
  batched-matmul expert compute.

## Measured PCC (TT vs HF reference, P150x8)
| Component | PCC |
|---|---|
| Vision tower (Pixtral, 24 layers, bf16) | 0.9987 |
| MLA attention block (full) | 0.99976 |
| MoE block (router+experts+shared) | 0.99980 |
| MoE block, expert-sharded 16/device | 0.99980 |
| Decoder layer (norms+MLA+MoE+residuals) | 0.99928 |
| 2-layer logits | 0.998 |
| 8-layer logits | 0.990 |
| **Full-depth 36-layer logits** | **0.98164** |
| fp8 dequant loader | 3 CPU unit tests pass |

Full-depth 0.9816 reflects bf16 + bfp8-expert + on-device-routing error accumulating over 36 layers;
the routing softmax/top-k in **fp32** (planned) is the lever to lift it. All per-block PCCs ~0.999.

## Performance
_In progress._ On-device forward is fast (the standalone 36-layer logit forward runs in ~seconds once
weights are resident; the test wall-clock is dominated by the one-time weight load + sharded upload).
ISL sweep (TTFT / prefill tok/s / decode tok/s-user, device + E2E) pending the paged-MLA serving wrapper.

## Reproduce (correctness)
```sh
export HF_MODEL=<Mistral-Small-4-119B snapshot dir>   MESH_DEVICE=P150x8
# vision tower
pytest models/tt_transformers/tests/multimodal/mistral_24b/test_m4_vision_tower.py -s
# text blocks
pytest models/tt_transformers/tests/multimodal/mistral_24b/test_m4_mla.py -s
pytest models/tt_transformers/tests/multimodal/mistral_24b/test_m4_moe.py -s
pytest models/tt_transformers/tests/multimodal/mistral_24b/test_m4_moe_sharded.py -s
pytest models/tt_transformers/tests/multimodal/mistral_24b/test_m4_decoder_layer.py -s
# full-depth logit gate (one-time ~40-min golden build, cached after)
M4_N_LAYERS=36 M4_SHARD=1 M4_EXPERT_DTYPE=bfp8 \
  pytest models/tt_transformers/tests/multimodal/mistral_24b/test_m4_text_model.py -s --timeout=0
# fp8 loader (CPU)
pytest models/tt_transformers/tests/multimodal/mistral_24b/test_m4_fp8_dequant.py
```

## Remaining to mergeable
- Relocate text modules → `models/demos/mistral4/`.
- Paged-MLA decode + deepseek-style generator (trace / 2CQ / on-device sampling); on-device routing
  (drop the per-layer host W round-trip); fp32-softmax routing refine.
- VLM glue (vision → projector → masked_scatter → text → lm_head) vs `Mistral3ForConditionalGeneration`.
- ISL perf sweep + measured perf table; CI registration (Blackhole demo pipeline, Tier 3); pinned deps
  (transformers ≥ 5.10 — a repo-wide bump, documented merge prerequisite); self-review vs `review-ign-model`.

## Deps note
transformers 5.10 supports `mistral4`/`mistral3`/`pixtral` natively (the repo pins 4.53 — bumping it is a
repo-wide, all-models-regression change, flagged as a merge prerequisite, not done unilaterally here).
