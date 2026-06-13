# BRINGUP LOG: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
**Slug:** `nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16`
**Target Device:** qb (blackhole)
**Started:** 2026-06-13T22:50:54Z
**Updated:** 2026-06-13T23:29:03Z

## Block Status

| Block | Phase | Status | PCC | Attempts | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Embedding | reference | done | 1.000000 | 0 |  |
| Embedding | ttnn | pending | — | 0 |  |
| Embedding | debug | n/a | — | 0 |  |
| Embedding | optimization | pending | — | 0 |  |
| Embedding | real_weights | pending | — | 0 |  |
| LayerNorm | reference | done | 1.000000 | 1 |  |
| LayerNorm | ttnn | pending | — | 0 |  |
| LayerNorm | debug | n/a | — | 0 |  |
| LayerNorm | optimization | pending | — | 0 |  |
| LayerNorm | real_weights | pending | — | 0 |  |
| RoPE | reference | done | 1.000000 | 0 |  |
| RoPE | ttnn | pending | — | 0 |  |
| RoPE | debug | n/a | — | 0 |  |
| RoPE | optimization | pending | — | 0 |  |
| RoPE | real_weights | pending | — | 0 |  |
| Mamba2Layer | reference | done | 0.999970 | 0 | sequential-step SSD loop; gated RMSNorm n_groups=8; conv1d prefill path; pre-norm + residual included |
| Mamba2Layer | ttnn | pending | — | 0 |  |
| Mamba2Layer | debug | n/a | — | 0 |  |
| Mamba2Layer | optimization | pending | — | 0 |  |
| Mamba2Layer | real_weights | pending | — | 0 |  |
| DenseAttention | reference | done | 0.999988 | 0 | GQA 32Q/2KV, no RoPE (HF has it commented out with TODO). Layer 5 weights. |
| DenseAttention | ttnn | pending | — | 0 |  |
| DenseAttention | debug | n/a | — | 0 |  |
| DenseAttention | optimization | pending | — | 0 |  |
| DenseAttention | real_weights | pending | — | 0 |  |
| DenseMLP | reference | done | 0.999999 | 0 | relu2 activation (no SwiGLU). Used shared_expert weights (no '-' layers in this model's pattern). |
| DenseMLP | ttnn | pending | — | 0 |  |
| DenseMLP | debug | n/a | — | 0 |  |
| DenseMLP | optimization | pending | — | 0 |  |
| DenseMLP | real_weights | pending | — | 0 |  |
| MoEAttention | reference | done | 1.000000 | 1 | GQA attention (32Q/2KV heads, head_dim=128, RoPE theta=10000). Architecturally identical to DenseAttention. PCC=1.0 vs inline HF-style reference (bit-exact, max_diff=0). |
| MoEAttention | ttnn | pending | — | 0 |  |
| MoEAttention | debug | n/a | — | 0 |  |
| MoEAttention | optimization | pending | — | 0 |  |
| MoEAttention | real_weights | pending | — | 0 |  |
| MoEGate | reference | done | 0.999999 | 1 | NemotronHTopkRouter: float32 proj, sigmoid, top-6 of 128, norm_topk_prob, scale=2.5. |
| MoEGate | ttnn | pending | — | 0 |  |
| MoEGate | debug | n/a | — | 0 |  |
| MoEGate | optimization | pending | — | 0 |  |
| MoEGate | real_weights | pending | — | 0 |  |
| MoEExperts | reference | done | 1.000000 | 0 | 128 experts, top-6 routing with NemotronHTopkRouter, relu2 activation, moe_intermediate_size=1856 |
| MoEExperts | ttnn | pending | — | 0 |  |
| MoEExperts | debug | n/a | — | 0 |  |
| MoEExperts | optimization | pending | — | 0 |  |
| MoEExperts | real_weights | pending | — | 0 |  |
| SharedExpert | reference | done | 0.999997 | 0 | moe_shared_expert_intermediate_size=3712, relu2, takes pre-normed input |
| SharedExpert | ttnn | pending | — | 0 |  |
| SharedExpert | debug | n/a | — | 0 |  |
| SharedExpert | optimization | pending | — | 0 |  |
| SharedExpert | real_weights | pending | — | 0 |  |
| LMHead | reference | done | 1.000000 | 0 | final RMSNorm + linear projection, tie_word_embeddings=False, vocab_size=131072 |
| LMHead | ttnn | pending | — | 0 |  |
| LMHead | debug | n/a | — | 0 |  |
| LMHead | optimization | pending | — | 0 |  |
| LMHead | real_weights | pending | — | 0 |  |

## Use cases

| Name | Input | Output | needs_ar | Generation | Perf |
| :--- | :--- | :--- | :--- | :--- | :--- |
| text_generation | text | text | yes | pending | pending |

## Recent Ticks

- tick 1 (2026-06-13T22:58:19Z): architecture[all] — ok
- tick 2 (2026-06-13T23:10:00Z): reference[LayerNorm] — ok
- tick 3 (2026-06-13T23:05:58Z): reference[Embedding/LayerNorm/RoPE/Mamba2Layer] — 3ok+1blocked
- tick 4 (2026-06-13T23:30:00Z): reference[MoEAttention] — ok
- tick 5 (2026-06-13T23:12:33Z): reference[DenseAttention/DenseMLP/MoEAttention/MoEGate] — 2ok+2blocked
- tick 6 (2026-06-13T23:45:00Z): reference[MoEGate] — ok
- tick 7 (2026-06-13T23:19:35Z): reference[DenseMLP/MoEGate/MoEExperts/SharedExpert] — 2ok+2blocked
- tick 8 (2026-06-13T23:22:05Z): unblock Mamba2Layer (LayerNorm done) — reset to pending
- tick 9 (2026-06-13T23:29:03Z): reference[Mamba2Layer/MoEExperts/SharedExpert/LMHead] — 4ok

## Host-Resident Exceptions

_None._
