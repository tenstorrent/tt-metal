<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GPT-OSS-120B Prefill (`gpt_oss_d_p`)

Long-context **prefill** for GPT-OSS-120B on **4×8 Blackhole Galaxy** (TP=8, SP=4, EP=32),
built "the MiniMax-M3 way": reuse the shared DeepSeek EP-MoE dispatch/combine substrate and the
fused `unified_routed_expert_ffn` (SwiGLU-OAI + biases), a block-cyclic KV cache, and a runtime
that plugs into the model-agnostic `models/demos/common/prefill` engine.

> **"Chunked" is the target architecture, not yet the running behavior.** The KV cache and runtime
> are chunk-oriented (block-cyclic, chunk-at-a-time), but the multi-chunk *cache-read* path (chunk
> N attending chunks 0..N-1) lands at P6. P1 through P4 run **single-chunk / one-shot** prefill.

## Roadmap

Stacked PRs, bottom-up. **This PR is P3.**

- [x] **P1 — package scaffold + attention** (merged): GQA, RoPE (YaRN), attention sinks, and
      per-layer sliding/full alternation; single-Blackhole-card PCC vs a torch reference.
- [x] **P2 — KV cache + indexed RoPE** (merged): chunked, block-cyclic, SP-sharded KV cache.
- [ ] **P3 — MoE** *(this PR)*: `TtGptOssMoE` over the DeepSeek EP submodules (SwiGLU-OAI + biases, no shared expert).
- [ ] **P4 — model + runtime**: full model, chunked-prefill runtime, and the `common/prefill` adapter.
- [ ] **P5 — galaxy bring-up**: TP=8 / SP=4 / EP=32 on the 4×8 Blackhole Galaxy (gated on galaxy access).
- [ ] **P6 — ring SDPA**: the sinks + sliding + halo-CCL ring SDPA (Pavle Josipović's op), for scalable SP and multi-chunk long context.
- [ ] **P7 — unification**: hoist the shared prefill scaffolding (attention output-proj/CCL tail, config, `utils/`) into `common/prefill`.

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
- **Bring-up (P1 through P4) uses AllGather + normal SDPA** (`ttnn.transformer.scaled_dot_product_attention`
  with `is_causal`, `sliding_window_size`, `attention_sink`, all supported today). Correct, but it
  replicates the full K/V per chip so it does not scale to 128k; the scalable ring SDPA swaps in at P6.

## Testing

Bottom-up PCC vs the HF reference (`modeling_gpt_oss.py`); thresholds per the design note. A single
Blackhole card covers per-op and per-layer PCC (norm/rope ≥0.999, attn/router ≥0.99, expert bf4 ≥0.98).
The full-model EP=32 and perf run happens on Galaxy **via CI** (workflow_dispatch), like the
DeepSeek-prefill galaxy job.

See the project plan: `GPT_OSS_PREFILL_PLAN.md`.
