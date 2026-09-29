<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GPT-OSS-120B Prefill (`gpt_oss_d_p`)

Long-context **prefill** for GPT-OSS-120B on **4×8 Blackhole Galaxy** (TP=8, SP=4, EP=32),
built "the MiniMax-M3 way": reuse the shared DeepSeek EP-MoE dispatch/combine substrate and the
fused `unified_routed_expert_ffn` (SwiGLU-OAI + biases), a block-cyclic KV cache, and a runtime
that plugs into the model-agnostic `models/demos/common/prefill` engine.

> **Multi-chunk prefill runs through one path.** Every sequence-parallel chunk, including chunk 0,
> uses the cache-backed native ring SDPA over the block-cyclic SP KV cache (Pavle Josipović's op,
> #51438).

## Roadmap

Stacked PRs, bottom-up. **This PR is P6 (chunked prefill via the ring SDPA).**

- [x] **P1 — package scaffold + attention** (merged): GQA, RoPE (YaRN), attention sinks, and
      per-layer sliding/full alternation; single-Blackhole-card PCC vs a torch reference.
- [x] **P2 — KV cache + indexed RoPE** (merged): chunked, block-cyclic, SP-sharded KV cache.
- [x] **P3 — MoE** (merged): `TtGptOssMoE` over the DeepSeek EP submodules (SwiGLU-OAI + biases, no shared expert).
- [x] **P4 — model + runtime** (merged): full model, chunked-prefill runtime, and the `common/prefill` adapter.
- [x] **P5 — galaxy bring-up** (this PR): TP=8 / SP=4 / EP=32 on the 4×8 Blackhole Galaxy — full 36L model, real weights, per-layer KV-cache PCC vs golden.
- [x] **P6 — ring SDPA** *(this PR)*: the sinks + sliding + halo-CCL ring SDPA (Pavle Josipović's op, #51438) — routes every sequence-parallel chunk through the cache-backed ring path.
- [ ] **P7 — unification**: hoist the shared prefill scaffolding (attention output-proj/CCL tail, config, `utils/`) into `common/prefill`.

## Beyond the bring-up stack

P1–P7 above is the **package bring-up**. Functional / perf / scaling work continues beyond it and is
tracked separately (not part of the P-stack):

- **Perf** — profile the full prefill; chunked (ring cache-read) is ~16× slower than one-shot. (#52000)
- **Functional sign-off** — top-1 / logits agreement vs HF (KV-cache PCC is only a proxy). (#52002)
- **CI** — register the galaxy chunked KV-PCC test in the tiered pipeline. (#52003)
- **Compute config** — a Blackhole `ComputeKernelConfig` (the `WormholeComputeKernelConfig` name is misleading on BH). (#51998)
- **Long context** — validate at 55k (one-shot OOMs; chunked is required).
- **Disaggregation** — KV-cache migration prefill→decode (Joe Malone).
- **Bounded sliding-window KV cache** — window-sized circular buffer for the sliding layers (Pavlo Hilei).
- **Serving** — multi-user + prefix + padding.

## Correctness reference

Bottom-up **PCC against a self-contained torch/HF reference** (HF `modeling_gpt_oss` conventions),
plus CPU-generated golden KV caches for the full-model check. This package is a fresh build; the
Wormhole/Galaxy `models/demos/gpt_oss` demo was a code-lineage source only and is **not** imported
or used as a runtime or correctness oracle here.

## What we reuse vs. write fresh

**Reuse (import, don't reimplement)**, from `models/demos/deepseek_v3_d_p/tt/moe/`:
`TtDispatchModule`, `TtCombineModule`, `TtReduceModule`, `TtMoERoutingSetup`, `TtRoutedExpert`
(fused `unified_routed_expert_ffn`, `RoutedExpertActivation.SwiGluOai`), `init_helpers`
(`ExpertMapping`, mesh mappers); the chunked-KV op
(`ttnn.experimental.deepseek_prefill.update_padded_kv_cache`) and indexed RoPE
(`rotary_embedding_indexed`); and `GptOss120BConfig`
(`deepseek_v3_d_p/reference/gpt_oss_120b_config.py`), already the single source of truth for dims
and already exercised at gpt-oss shapes by the DeepSeek MoE op tests.

**Write fresh (gpt-oss-specific):**
- `tt/attention/` — GQA, RoPE (YaRN), **attention sinks**, **sliding/full alternation**.
- `tt/moe/` — thin `TtGptOssMoE` composing the DeepSeek EP submodules; SwiGLU-OAI **+ biases**, **no shared expert**.
- `tt/router.py` — linear+bias, topk(4), softmax-over-4.
- weight prep — MXFP4 to bf16 (host) to bfloat4_b (device); gate/up de-interleave; expert permutation.
- `tt/tt_prefill_runtime.py` + `tt/runners/adapters/gpt_oss.py` — runtime + `common/prefill` adapter.

## Attention (this PR): shapes & correctness notes

Per chip, per layer, one prefill chunk (`S_loc = S/SP`):

| tensor | shape | dtype | layout |
|---|---|---|---|
| Q | `[1, 8, S_loc, 64]` (8 of 64 Q-heads) | bf16 | TILE |
| K | `[1, 1, S_loc, 64]` (1 of 8 KV-heads) | bf16 (cache bf8_b) | TILE |
| V | `[1, 1, S_loc, 64]` | bf16 (cache bf8_b) | TILE |
| sinks | `[8]` (per local Q-head), **pre-divided by `config.scaling`** | bf16 | — |
| out | `[1, 8, S_loc, 64]` | bf16 | TILE |

- GQA group = 8 Q-heads share the 1 local KV-head (no on-chip KV repeat).
- Sinks are stored **pre-divided by `config.scaling`** (the `1/√head_dim` softmax scale, i.e. ×√64),
  so the SDPA kernel's own `×scale` of the sink logit recovers the raw HF value (HF does not scale the sink).
- Layers alternate `sliding_attention` (window 128) and `full_attention` off `hf_config.layer_types`.
- **Every chunked SP chunk uses the native ring cache-read** (`tt/attention/dense_sp.py`,
  RingJointSDPA) over the block-cyclic SP cache. Chunk 0 writes its K/V into the cache and uses the
  same complete-first-group ring path as chunks 1+. Equal-sized one-shot prefill retains the exact
  replicated Q/K/V bootstrap because sliding RingJointSDPA requires short Q against a longer K/V cache.

## Testing

Bottom-up PCC vs the HF reference (`modeling_gpt_oss.py`); thresholds per the design note. A single
Blackhole card covers per-op and per-layer PCC (norm/rope ≥0.999, attn/router ≥0.99, expert bf4 ≥0.98).
The full-model EP=32 and perf run happens on Galaxy **via CI** (workflow_dispatch), like the
DeepSeek-prefill galaxy job.

See the project plan: `GPT_OSS_PREFILL_PLAN.md`.
