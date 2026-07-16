<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GPT-OSS-120B Prefill (`gpt_oss_d_p`)

Chunked, long-context **prefill** for GPT-OSS-120B on **4×8 Blackhole Galaxy**
(TP=8, SP=4, EP=32), built "the MiniMax-M3 way": reuse the shared DeepSeek EP-MoE
dispatch/combine substrate + the fused `unified_routed_expert_ffn` (SwiGLU-OAI +
biases), a chunked block-cyclic KV cache, and a runtime that plugs into the
model-agnostic `models/demos/common/prefill` engine.

> Status: **bring-up (P1).** Built fresh, mirroring `models/demos/minimax_m3/`.
> Not a continuation of `models/demos/gpt_oss` (Joe's Wormhole decode+prefill),
> which is used only as a correctness reference.

## What we reuse vs. write fresh

**Reuse (import, don't reimplement)** — from `models/demos/deepseek_v3_d_p/tt/moe/`:
`TtDispatchModule`, `TtCombineModule`, `TtReduceModule`, `TtMoERoutingSetup`,
`TtRoutedExpert` (fused `unified_routed_expert_ffn`, `RoutedExpertActivation.SwiGluOai`),
`init_helpers` (`ExpertMapping`, mesh mappers); the chunked-KV op
(`ttnn.experimental.deepseek_prefill.update_padded_kv_cache`) and indexed RoPE
(`rotary_embedding_indexed`); and `GptOss120BConfig`
(`deepseek_v3_d_p/reference/gpt_oss_120b_config.py`) — already the single source of
truth for dims and already exercised at gpt-oss shapes by the DeepSeek MoE op tests.

**Write fresh (gpt-oss-specific):**
- `tt/attention/` — GQA + RoPE(YaRN) + **attention sinks** + **sliding/full alternation**.
- `tt/moe/` — thin `TtGptOssMoE` composing the DeepSeek EP submodules; SwiGLU-OAI **+ biases**,
  **no shared expert**.
- `tt/router.py` — linear+bias → topk(4) → softmax-over-4.
- weight prep — MXFP4 → bf16 (host) → bfloat4_b (device); gate/up de-interleave; expert permutation.
- `tt/tt_prefill_runtime.py` + `tt/runners/adapters/gpt_oss.py` — runtime + `common/prefill` adapter.

## Directory layout

```
gpt_oss_d_p/
  reference/    torch/HF reference glue (dims via deepseek_v3_d_p GptOss120BConfig)
  tt/
    attention/  GQA + sinks + sliding/full + RoPE(YaRN)   [fresh]
    moe/        TtGptOssMoE over deepseek EP submodules    [thin]
    runners/adapters/gpt_oss.py   GptOssPrefillAdapter     [common/prefill contract]
    tt_prefill_runtime.py, model.py, layer.py, router.py, rms_norm.py, weights.py, model_config.py
  tests/unit/   per-op / per-layer PCC vs HF reference (single Blackhole card)
  scripts/      generate_golden_kv_cache.py (adapted from minimax_m3)
```

## Attention (net-new): shapes & correctness notes

Per chip, per layer, one prefill chunk (`S_loc = S/SP`):

| tensor | shape | dtype | layout |
|---|---|---|---|
| Q | `[1, 8, S_loc, 64]` (8 of 64 Q-heads) | bf16 | TILE |
| K | `[1, 1, S_loc, 64]` (1 of 8 KV-heads) | bf16 (cache bf8_b) | TILE |
| V | `[1, 1, S_loc, 64]` | bf16 (cache bf8_b) | TILE |
| sinks | `[8]` (per local Q-head), **pre-scaled by 1/√64** | bf16 | — |
| out | `[1, 8, S_loc, 64]` | bf16 | TILE |

- GQA group = 8 Q-heads share the 1 local KV-head (no on-chip KV repeat).
- Sinks stored **pre-divided by 1/√head_dim** so the SDPA kernel's internal scale reproduces HF.
- Layers alternate `sliding_attention` (window 128) / `full_attention` off `hf_config.layer_types`.
- **Bring-up (P1–P4) uses AllGather + normal SDPA** (`ttnn.transformer.scaled_dot_product_attention`
  with `is_causal`, `sliding_window_size`, `attention_sink` — all supported today). Correct but
  replicates full K/V per chip → does not scale to 128k. The scalable **ring SDPA (sinks + sliding +
  halo CCL)** is Pavle Josipović's op; swapped in behind a flag at P6.

## Testing

Bottom-up PCC vs HF (`modeling_gpt_oss.py`); thresholds per the design note.
Single Blackhole card covers per-op + per-layer PCC (norm/rope ≥0.999, attn/router ≥0.99,
expert bf4 ≥0.98). Full-model EP=32 + perf run on Galaxy **via CI** (workflow_dispatch),
like the DeepSeek-prefill galaxy job.

See the project plan: `GPT_OSS_PREFILL_PLAN.md`.
