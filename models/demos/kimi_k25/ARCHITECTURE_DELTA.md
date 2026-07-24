# Kimi K2.5 vs DeepSeek V3 — Architecture Delta for Reviewers

This document summarizes the architectural differences between **Kimi K2.5** and
**DeepSeek V3** at the level of detail needed to review the `kimi_k25` demo.

The implementation reuses the entire `models/demos/deepseek_v3/` runtime. Every
difference listed here is either handled transparently by a config field, or
requires a specific code adaptation noted below.

---

## 1. Transformer Backbone

| Parameter | Kimi K2.5 | DeepSeek V3 | Impact |
|-----------|-----------|-------------|--------|
| `num_hidden_layers` | 61 | 61 | Same |
| `hidden_size` | 7168 | 7168 | Same |
| `num_attention_heads` | **64** | 128 | Config-only; MLA projection shapes differ |
| `rms_norm_eps` | **1e-5** | 1e-6 | Critical for accuracy; wrong value → subtle numerical drift |

**Action**: `KimiK25Config` validates `rms_norm_eps` at startup and raises
`ValueError` if it doesn't match. Caught by `_validate()` in `config_adapter.py`.

---

## 2. MoE Routing — Flat vs Grouped

This is the most significant routing difference.

### DeepSeek V3: grouped top-k

```
n_routed_experts = 256
n_group = 8             # experts divided into 8 groups of 32
topk_group = 4          # activate top-4 groups
num_experts_per_tok = 8 # activate 8 experts total
```

Routing proceeds in two stages:
1. Select `topk_group=4` groups (based on sum of top-2 scores per group)
2. Select `num_experts_per_tok=8` experts from those 4 groups

### Kimi K2.5: flat top-k (n_group=1)

```
n_routed_experts = 384
n_group = 1             # one "group" — all experts
topk_group = 1          # activate the one group
num_experts_per_tok = 8 # activate 8 experts globally
```

With `n_group=1` the group-selection stage degenerates to a no-op. The
`moe_gate.py` fused kernel is already fully parameterized for this case — no
code changes needed (verified in M3 analysis, see `NOTES.md`).

**Action**: `KimiK25Config` passes `n_group=1, topk_group=1` to DSV3 modules
unchanged. The `TOPK_MIN_WIDTH` padding in `moe_gate.py` handles the edge case
where group shape would otherwise be 1.

---

## 3. Expert Count and Sharding

| | Kimi K2.5 | DeepSeek V3 |
|--|-----------|-------------|
| `n_routed_experts` | **384** | 256 |
| `n_shared_experts` | 1 | 1 |
| Experts/device on TG (32 chips) | **12** | 8 |

384 is cleanly divisible by 32, 64, and 128 — all supported Galaxy topologies.

Expert weights are INT4 group-32 quantized (Kimi) vs FP8 block-128 (DSV3).
`KimiLazyStateDict` handles transparent dequantization before the weights
reach any DSV3 module.

---

## 4. MLA (Multi-head Latent Attention)

Both models use MLA with identical low-rank projection dimensions:

| Parameter | Kimi K2.5 | DeepSeek V3 |
|-----------|-----------|-------------|
| `q_lora_rank` | 1536 | 1536 |
| `kv_lora_rank` | 512 | 512 |
| `qk_nope_head_dim` | 128 | 128 |
| `qk_rope_head_dim` | 64 | 64 |
| `v_head_dim` | 128 | 128 |
| `num_attention_heads` | **64** | 128 |

The attention head count (64 vs 128) changes the shape of `q_proj` and the
final projection but not the MLA algorithm itself. DSV3 MLA is parameterized
on `num_attention_heads` — no code change needed.

---

## 5. Rotary Embeddings (RoPE / YaRN)

| Parameter | Kimi K2.5 | DeepSeek V3 |
|-----------|-----------|-------------|
| `rope_theta` | **50,000** | 10,000 |
| `rope_scaling.type` | yarn | yarn |
| `rope_scaling.factor` | **64.0** | (different) |
| `rope_scaling.beta_fast` | **32.0** | (different) |
| `max_position_embeddings` | 131,072 | 163,840 |

Both use YaRN. DSV3's RoPE setup is parameterized on these fields from
`hf_config` — no code change needed.

---

## 6. Vocabulary

| | Kimi K2.5 | DeepSeek V3 |
|--|-----------|-------------|
| `vocab_size` | **163,840** | ~129,280 |
| Padded vocab | 163,840 (already 64-aligned) | aligned |

Larger vocab increases `lm_head` weight size: `163840 × 7168 = 1.17B params`
vs DSV3's `~0.93B`. Tile layout in TTNN is unaffected (vocab dim is outer).

---

## 7. Dense Layer Position

| | Kimi K2.5 | DeepSeek V3 |
|--|-----------|-------------|
| `first_k_dense_replace` | **1** | 3 |

Only layer 0 uses a dense FFN (SwiGLU, no expert routing). Layers 1–60 are
MoE. DSV3 has layers 0–2 as dense. This is a config-only difference — DSV3
runtime checks `first_k_dense_replace` per-layer.

---

## 8. Weight Quantization

| | Kimi K2.5 | DeepSeek V3 |
|--|-----------|-------------|
| Format | **INT4 group-32 symmetric** | FP8 block-128 |
| Targets | Routed expert linears only | All linears |
| Checkpoint keys | `*.weight_packed` (I32) + `*.weight_scale` (BF16) | `*.weight` (FP8) |

**Implementation**: `KimiLazyStateDict` in `utils/weight_loader.py`:
1. Strips `language_model.` prefix (Kimi checkpoint structure)
2. Detects `*_packed` keys and dequantizes I32 → nibbles → BF16 on first access
3. Passes BF16 non-expert weights through unchanged

The random-weights smoke test path bypasses this entirely — it uses
`prepare_model_state_dict(random_weights=True)` from DSV3 utilities, which
generates BF16 random tensors directly.

---

## 9. Multi-Token Prediction (MTP)

| | Kimi K2.5 | DeepSeek V3 |
|--|-----------|-------------|
| `num_mtp_layers` | **0** | 1 |

DSV3 has one MTP head for speculative decoding training. Kimi K2.5 drops it.
DSV3 runtime skips MTP logic when `num_mtp_layers=0` — no code change needed.

---

## 10. Scoring Function

| | Kimi K2.5 | DeepSeek V3 |
|--|-----------|-------------|
| `scoring_func` | **sigmoid** | softmax |
| `routed_scaling_factor` | **2.827** | 2.5 |

Kimi uses sigmoid gating (with `routed_scaling_factor` to compensate for the
different normalization). `norm_topk_prob=True` normalizes selected expert
weights to sum to 1. DSV3 `moe_gate.py` handles both via config — no code
change needed.

---

## Summary: Code Changes vs Config-Only

```
Change                              Type          File
----------------------------------  ------------  ----------------------------
384 experts, n_group=1              Config-only   config_adapter.py
64 attention heads                  Config-only   config_adapter.py
rms_norm_eps=1e-5                   Config-only   config_adapter.py (+ validation)
rope_theta=50000, YaRN factor=64    Config-only   config_adapter.py
vocab_size=163840                   Config-only   config_adapter.py
first_k_dense_replace=1             Config-only   config_adapter.py
num_mtp_layers=0                    Config-only   config_adapter.py
INT4 weight dequantization          New code      utils/weight_loader.py
                                                  utils/int4_dequantize.py
language_model. prefix stripping    New code      utils/weight_loader.py
KimiGenerator factory               New code      tt/kimi_model.py
KimiK25Config validation            New code      utils/config_adapter.py
```

**Bottom line**: 3 new utility files + 1 thin model adapter. Zero changes to
any `deepseek_v3/` runtime file.
