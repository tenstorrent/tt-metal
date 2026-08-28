# Command-R (`CohereLabs/c4ai-command-r-v01`) — Validation Evidence

Bring-up of Cohere's Command-R 35B on the TT `tt_transformers` stack, for
bounty [tenstorrent/tt-metal#49307](https://github.com/tenstorrent/tt-metal/issues/49307).

**Hardware class (stated honestly):** all validation below was executed on a
**Blackhole QuietBox 2** (4× Blackhole, mesh (1,4), TP=4) — *not* the Wormhole
T3K (8-chip, TP=8) target named in the bounty. No T3K hardware was available to
us. Performance numbers are Blackhole-class references, not T3K bounty numbers.

Implementation: 12 files touched vs our fork base — `tt/cohere/` (new:
`cohere_norm.py`, `cohere_decoder.py`, `cohere_lm_head.py`, `README.md`,
`__init__.py`), plus `tt/model.py`, `tt/model_config.py`, `tt/generator_vllm.py`,
`tt/generator_sglang.py`, and `tests/test_cohere_pcc.py`,
`tests/test_cohere_fullmodel_pcc.py`, `tests/test_cohere_vllm_e2e.py`.
Fork branch: `canada-quant/tt-metal@command-r-bringup`.

## Correctness (PCC vs HF PyTorch fp32 CPU reference)

| Gate | Result | Verdict |
|---|---|---|
| Single-layer layernorm PCC (gate 0.9999) | 0.9999898 | PASS |
| Single decoder-layer prefill PCC (gate 0.99) | 0.9998952 | PASS |
| All-40-layer chained sweep (gate 0.99) | min 0.995873 / mean 0.999017 | PASS |
| Final-norm PCC | 0.992257 | PASS |
| Logits PCC (tied embedding, logit_scale 0.0625) | 0.997250 | PASS |
| Post-fix full chain at a failing 36-token prompt | min 0.999789; logits 0.999901 (head) / 0.999164 (tail window incl. last position) | PASS |
| e2e prefill+decode through the vLLM-plugin call path | `1 passed in 230.18s` — 16-token coherent decode | PASS |

## Served validation (vLLM OpenAI endpoint on QB2, ctx 131,072)

- Live chat serving verified; chat template applied server-side.
- Grounded-generation (RAG citation) template validated: model emits `<co: 0>`
  citation spans over supplied documents (3 citation spans, clean stop).
- Correctness battery post-fix: 12.4k-token needle-in-haystack PASS;
  served math probes PASS (one remaining near-tie math case reproduced on the
  CPU reference itself — first-token top-5 logit gaps ~0.12 — i.e. sampling
  numerics, not a stack defect).
- Benchmarks of record (model-card sampling: temp 1.0, top_p 0.95, top_k 20):
  GPQA-Diamond pass@1 0.267 (264/990, 0 errors, 0 truncated);
  AIME 2026 avg@4 0.008 (1/120) — honest capability numbers for a 2024
  non-reasoning 35B at temperature 1.0.

## Serving performance (Blackhole QB2, mesh (1,4))

- TTFT (first content chunk, short prompt): 0.078–0.095 s.
- Decode throughput: ~21.8 tok/s single-stream; ~42.9 tok/s aggregate at
  concurrency 2 (linear). Serial per-item rate in benchmarks: ~21.4 tok/s.
- **Known defect:** a 4-way concurrent prefill burst reproducibly crashes the
  engine (`IndexError: index 3 is out of bounds for dimension 0 with size 3`
  in `generator.py:prefill_forward_text`, page-table path) — observed twice;
  recovery requires `tt-smi -r` before container restart. Sequential and 2-way
  load are stable.

## Root-cause findings that may be generally useful

1. **Q/K `reverse_permute` vs interleaved-native RoPE (the served-quality bug).**
   The stock `use_hf_rope=False` loader path applies the Llama NeoX→Meta
   permutation to Q/K weights. HF `modeling_cohere` (transformers 4.57.6) uses
   *adjacent-pair* (interleaved-native) rotation, so permuting scrambles the
   weights: prefill activations still PCC-pass on short probes, but served
   decode flips content-token argmax on longer/chat-templated prompts
   (length-correlated: ≤17 tokens OK, ≥24 fail). Fix: gate
   `model_type=="cohere"` → no-permute (`d30148f7`); layer-0 PCC went
   0.9324 → 0.999789.
2. Parallel-block tensor lifetime: `Attention.forward` and `MLP.forward` both
   `ttnn.deallocate` their input — the shared normed tensor must be copied for
   the MLP branch *before* attention consumes it (`ttnn.add(h, 0.0)`;
   `ttnn.clone` segfaults on the post-all-gather mesh tensor in this build).
3. Decode-shape layernorm: distributed pre/post-all-gather variants fatal at
   decode shapes in this build; decode uses full-width layernorm
   (all_gather_async + plain LN) with TILE-shaped gamma; sharded output layout
   required (interleaved norm output → DRAM-sharded matmul = `bad_optional_access`).
4. Plain interleaved `ttnn.layer_norm` over-allocates L1 at hidden 8192 — do
   not use it on the mesh.

## Status vs the bounty's three stages

- **Stage 1 (bring-up):** done on Blackhole QB2 — TTNN-stack implementation,
  all 40 layers PCC-validated, e2e decode, logits verified. The T3K/TP=8 mesh
  itself is untested (no such hardware on site).
- **Stage 2 (basic optimizations):** TP sharding + CCL collectives in use
  (all-gather/reduce-scatter via the tt_transformers stack); sharded memory
  configs throughout; KV cache paged + distributed. T3K-specific TP=8 tuning
  untested.
- **Stage 3 (deeper optimization):** not done — batch-32 T/S/U targets,
  chunked prefill to 128K, BFP8 compression, and the T3K perf targets
  (~60 T/S/U, ~70 ms TTFT) are all open; the conc-4 prefill crash above is a
  known blocker for high-concurrency serving.
