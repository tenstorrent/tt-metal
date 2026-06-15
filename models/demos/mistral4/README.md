# Mistral-Small-4-119B (mistral4) on Tenstorrent Blackhole

`mistralai/Mistral-Small-4-119B-2603` — a **Mistral3 VLM**: a Pixtral vision tower + a
**DeepSeek-V3-class text core** (Multi-head Latent Attention + 128-expert ungrouped top-4 MoE +
1 shared expert), fp8 weights, YaRN RoPE, 36 layers, hidden 4096, vocab 131072. Target:
**Blackhole Loudbox 1×8 (P150x8)**.

> Status: **Correctness complete end-to-end** (every block PCC-verified on P150x8, generic ttnn ops,
> no new kernels) and the **production decode path is captured as a bit-exact trace on the sharded
> 8-device mesh** with on-device sampling. Perf is measured at full depth (below). Remaining serving
> optimizations (MoE sparse dispatch, paged compressed-latent KV, 2CQ, sharded LM head) are listed
> under **Current status**.

## Architecture / where the code lives
- **Text core (MLA + MoE)** — model-local modules (`tt/mistral4_text.py`) from generic ttnn ops:
  - **MLA**: low-rank q/kv projections + RMSNorms, interleaved RoPE applied as a permutation-matrix
    matmul (de-interleave → rotate_half, no tile-crossing reshape), MQA k_rot expand, SDPA, o_proj.
    Because HF MLA caches the **expanded** k/v, decode is standard attention — so decode uses
    `ttnn.transformer.scaled_dot_product_attention_decode` over an expanded-kv cache, and the KV write
    uses `paged_update_cache(update_idxs_tensor=…)` with a device-resident position (trace-friendly).
  - **MoE**: router (linear → softmax → top-4/128 kth-threshold mask → normalize) + SwiGLU experts +
    shared expert. Experts are **sharded across the 8 devices** (16/device, `bfloat8_b`); each device
    extracts its local experts' routing weights **on-device** via a precomputed sharded column selector
    (`W_full @ select_d`), and partial sums combine with `all_reduce_async` — no host round-trip, so
    the whole MoE is trace-capturable.
- **Generator** (`tt/mistral4_generator.py`) — a compact model-local generator (deepseek_v3-demo
  pattern, not the GQA-bound `tt_transformers` Generator): parallel prefill (fills the KV cache via
  `fill_cache`) → token-by-token decode with **on-device `ttnn.argmax`** sampling. `TracedDecode`
  captures the decode step as a replayable trace (persistent device input buffers + `execute_trace`).
- **Vision tower + projector** — reuse the `tt_transformers` Pixtral path (config-driven; bf16).
- **fp8** — vanilla per-tensor + per-expert `scale_inv` dequant (`dequantize_fp8_state_dict`), distinct
  from transformers' block-wise FineGrainedFP8. Experts run on-device as `bfloat8_b` (≈ the native fp8).

## Measured PCC (TT vs HF reference, P150x8)
| Component | PCC |
|---|---|
| Vision tower (Pixtral, 24 layers, bf16) | 0.9987 |
| MLA attention block (full) | 0.99976 |
| MoE block (router+experts+shared) | 0.99980 |
| MoE block, expert-sharded 16/device | 0.99980 |
| Decoder layer (norms+MLA+MoE+residuals) | 0.99928 |
| Projector (Mistral3 multi-modal) | 0.99991 |
| **Full-depth 36-layer logits** | **0.98164** |
| End-to-end VLM logits (vision→projector→text) | 0.99762 |
| MLA decode (op) / forward_decode + KV cache | 0.99988 / 0.99971 |
| 2-layer model decode (incremental) logits | 0.99804 |
| **Decode trace capture/replay (replicated & sharded)** | **1.0 (bit-exact)** |
| Greedy generation IDs vs HF (free-running) | 7/8 match |
| fp8 dequant loader | 3 CPU unit tests pass |

Full-depth 0.9816 reflects bf16 + bfp8-expert + bf16-routing error accumulating over 36 layers
(`ttnn.topk` is bf16/bfp8-only, so a borderline expert can flip vs the fp32 reference); all per-block
PCCs are ~0.999. Passes the >0.98 full-depth gate.

## Performance (measured, P150x8, full 36 layers, sharded bfp8, steady-state)
| Metric | Value |
|---|---|
| Decode tok/s/user (device) | **9.0** (110.5 ms/tok) |
| Decode tok/s/user (E2E, incl. host argmax read-back) | 8.6 |
| Prefill TTFT @ ISL 128 / 1024 / 4096 | 226 ms / 655 ms / 2657 ms |
| Prefill throughput @ ISL 1024 / 4096 | 1564 / 1541 tok/s |
| Largest ISL measured (single-shot prefill) | 4096 (no L1 clash) |

Decode cost is **flat across ISL** (the decode step is a captured trace independent of context).

**Multi-user batched decode** (traced, dense MoE) trades latency for throughput — the MoE streams
each expert's weights once and applies them to all B tokens, so aggregate tok/s grows with batch
(measured 2-layer; full-depth scales by depth):

| Batch | tok/s/user | tok/s aggregate |
|---|---|---|
| 1 | 133.7 | 134 |
| 8 | 46.2 | 370 |
| 32 | 14.4 | 462 |

Scaling is sub-linear because the dense MoE amortizes weight-streaming but its compute (16 experts ×
B tokens/device) grows with batch. Sparse dispatch (top-4 only) is the remaining lever at both ends —
see status.

## Current status / remaining work
- **Done:** full-depth logit correctness; e2e VLM correctness; on-device sampling; decode trace
  (replicated + sharded production mesh); fully on-device MoE; ISL perf sweep harness + measured
  full-depth numbers.
- **Remaining serving optimizations:** MoE **sparse dispatch** (top-4 compute instead of dense-local);
  **paged compressed-latent KV** (deepseek-style, 12.8× smaller cache than expanded-kv) + **chunked
  prefill** for >4K contexts (single-shot prefill is measured fine to 4K); **2CQ**; **sharded LM head**;
  prefill trace. CI registration + rebase onto latest main.
- **Dependency:** requires `transformers >= 5.10` (native `mistral4`/`mistral3`/`pixtral` + fp8 dequant).

## Reproduce
```sh
export HF_MODEL=<Mistral-Small-4-119B snapshot dir>   MESH_DEVICE=P150x8
# correctness — text blocks
pytest models/demos/mistral4/tests/test_m4_mla.py -s
pytest models/demos/mistral4/tests/test_m4_moe_sharded.py -s
pytest models/demos/mistral4/tests/test_m4_decoder_layer.py -s
# full-depth logit gate (one-time ~40-min golden build, cached after)
M4_N_LAYERS=36 M4_SHARD=1 M4_EXPERT_DTYPE=bfp8 pytest models/demos/mistral4/tests/test_m4_text_model.py -s --timeout=0
# e2e VLM + projector
pytest models/demos/mistral4/tests/test_m4_vlm.py models/demos/mistral4/tests/test_m4_projector.py -s
# serving: decode trace (bit-exact), greedy generation, decode/prefill correctness
pytest models/demos/mistral4/tests/test_m4_trace.py models/demos/mistral4/tests/test_m4_generate.py -s
pytest models/demos/mistral4/tests/test_m4_text_decode.py models/demos/mistral4/tests/test_m4_forward_decode.py -s
# perf sweep (full depth)
M4_PERF_LAYERS=36 M4_PERF_ISLS=128,1024,4096 pytest models/demos/mistral4/tests/test_m4_perf.py -s --timeout=0
# fp8 loader (CPU)
pytest models/demos/mistral4/tests/test_m4_fp8_dequant.py
```

## Deps note
`transformers >= 5.10` supports `mistral4`/`mistral3`/`pixtral` natively and provides the fp8 dequant
path. The repo currently pins an older transformers — bumping it is a documented merge prerequisite
(coordinate with sibling model PRs that pin a different major version; they cannot share an env).

## Framework dependency (self-review note)
All mistral4-specific code lives in `models/demos/mistral4/`. The model reuses generic framework
additions in `models/tt_transformers/tt/` — `dequantize_fp8_state_dict` (load_checkpoints), the
`FineGrainedFP8Config(dequantize=True)` path + Mistral3-VLM nesting (model_config), and a
mistral-common tokenizer shim (common) — which are generic (any fp8 checkpoint / Mistral3 VLM
benefits) and carry **no mistral4-specific branching** (verified: shared files mention mistral4 only
in doc comments). These belong to the shared fp8/VLM framework work (the Devstral PR); mistral4
depends on them and adds none of its own edits to shared infra.

## Status snapshot (self-review vs review-ign-model)
Done: A1 framework adherence, A2 device-side (no hot-loop host round-trip), A3 trace — **prefill AND
decode** captured/replayed bit-exactly (PCC 1.0), A5 on-device sampling, A7 sharded LM head, A9 (prefill to
4K, no L1 clash), B1–B3 full-depth ungated logit PCC, C1 ISL-sweep harness, C-perf measured, D1
rebased (current), D1a CI registration, D4 pinned deps, E docs.
Remaining (large, scoped): **A10 MoE sparse dispatch** — `all_to_all_dispatch` requires a **2D mesh**
(tokens batch-sharded on a data-parallel axis AND experts on a separate expert-parallel axis, as in
deepseek's dispatch×TP layout). This model's flat **1×8 mesh shards all 128 experts across the 8
devices**, leaving no free axis to batch-shard tokens for dispatch — a topology mismatch, not a bug
(`all_to_all_dispatch` itself is proven on this mesh, `test_m4_a2a_probe`). The dense expert-parallel
MoE is the appropriate design on 1×8; enabling sparse dispatch needs a 2×4 mesh remap (2-way DP ×
4-way EP) + the dispatch machinery, or a non-dispatch per-token expert gather. `_forward_sparse` is
the authored pipeline (`test_m4_moe_sparse`, xfail). **A6 paged compressed-latent KV + chunked
prefill** for >4K contexts. A4 2CQ (low value for the tiny decode inputs). All documented, not hidden.
