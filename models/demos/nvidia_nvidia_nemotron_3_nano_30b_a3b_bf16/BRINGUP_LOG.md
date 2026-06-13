# BRINGUP LOG: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
**Slug:** `nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16`
**Target Device:** qb (blackhole)
**Started:** 2026-06-13T22:50:54Z
**Updated:** 2026-06-13T23:05:58Z

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
| Mamba2Layer | reference | blocked | — | 0 | dep LayerNorm was pending |
| Mamba2Layer | ttnn | pending | — | 0 |  |
| Mamba2Layer | debug | n/a | — | 0 |  |
| Mamba2Layer | optimization | pending | — | 0 |  |
| Mamba2Layer | real_weights | pending | — | 0 |  |
| DenseAttention | reference | pending | — | 0 |  |
| DenseAttention | ttnn | pending | — | 0 |  |
| DenseAttention | debug | n/a | — | 0 |  |
| DenseAttention | optimization | pending | — | 0 |  |
| DenseAttention | real_weights | pending | — | 0 |  |
| DenseMLP | reference | pending | — | 0 |  |
| DenseMLP | ttnn | pending | — | 0 |  |
| DenseMLP | debug | n/a | — | 0 |  |
| DenseMLP | optimization | pending | — | 0 |  |
| DenseMLP | real_weights | pending | — | 0 |  |
| MoEAttention | reference | pending | — | 0 |  |
| MoEAttention | ttnn | pending | — | 0 |  |
| MoEAttention | debug | n/a | — | 0 |  |
| MoEAttention | optimization | pending | — | 0 |  |
| MoEAttention | real_weights | pending | — | 0 |  |
| MoEGate | reference | pending | — | 0 |  |
| MoEGate | ttnn | pending | — | 0 |  |
| MoEGate | debug | n/a | — | 0 |  |
| MoEGate | optimization | pending | — | 0 |  |
| MoEGate | real_weights | pending | — | 0 |  |
| MoEExperts | reference | pending | — | 0 |  |
| MoEExperts | ttnn | pending | — | 0 |  |
| MoEExperts | debug | n/a | — | 0 |  |
| MoEExperts | optimization | pending | — | 0 |  |
| MoEExperts | real_weights | pending | — | 0 |  |
| SharedExpert | reference | pending | — | 0 |  |
| SharedExpert | ttnn | pending | — | 0 |  |
| SharedExpert | debug | n/a | — | 0 |  |
| SharedExpert | optimization | pending | — | 0 |  |
| SharedExpert | real_weights | pending | — | 0 |  |
| LMHead | reference | pending | — | 0 |  |
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

## Host-Resident Exceptions

_None._
