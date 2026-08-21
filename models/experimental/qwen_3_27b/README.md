<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.6-27B

TTNN implementation of the dense [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) model. Work in progress — this commit is the initial skeleton.

## Model specs

| | |
|---|---|
| Parameters | 27B (dense) |
| Layers | 64 |
| Hidden size | 5120 |
| FFN intermediate size | 17408 |
| Activation | SiLU (gated MLP) |
| Norm | RMSNorm, eps 1e-6 |
| Vocab size | 248320 (untied embeddings) |
| Context length | 262144 native (up to ~1M with YaRN) |
| dtype | bfloat16 |

### Layer pattern

Layers alternate `3 × linear attention` + `1 × full attention` (`full_attention_interval = 4`), giving 48 Gated DeltaNet layers and 16 Gated Attention layers. Each attention block is followed by an MLP.

| | Gated Attention (full) | Gated DeltaNet (linear) |
|---|---|---|
| Count | 16 | 48 |
| Q heads | 24 | 16 (key) |
| KV heads | 4 (GQA) | 48 (value) |
| Head dim | 256 | 128 |
| Positional | RoPE, θ=1e7, partial factor 0.25 (rotary dim 64), interleaved mRoPE, sections [11, 11, 10] | — |
| Output gate | sigmoid (2× wide Q proj) | swish |
| Other | — | short conv, kernel 4; fp32 SSM state |

### Extras

- **Vision encoder:** 27 layers, hidden 1152, FFN 4304, 16 heads, patch 16, spatial merge 2, temporal patch 2, projects to hidden 5120.
- **MTP:** 1 multi-token-prediction layer (shared embeddings).
- **Modes:** thinking (default) and non-thinking.
