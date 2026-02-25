# Olmo-3-1125-32B on TT Hardware: Effort Assessment

**Reference codebase:** `models/demos/llama3_70b_galaxy`
**Target model:** [allenai/Olmo-3-1125-32B](https://huggingface.co/allenai/Olmo-3-1125-32B)
**Area:** `/home/ttuser/ssinghal/PR-fix/main/debug/tt-metal`
**Settings:** `export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate`

---

## Olmo-3-32B Architecture (from config.json)

| Parameter | Value |
|-----------|-------|
| hidden_size | 5120 |
| num_hidden_layers | 64 |
| num_attention_heads | 40 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| intermediate_size | 27648 |
| vocab_size | 100278 |
| max_position_embeddings | 65536 |
| rope_theta | 500000 |
| rope_type | yarn |
| sliding_window | 4096 |
| layer_types | 3× sliding_attention, 1× full_attention (repeat 16×) |

**Key differences from Qwen3-32B (already in Galaxy):**

| Aspect | Qwen3-32B | Olmo-3-32B |
|--------|-----------|------------|
| dim / hidden_size | 5120 | 5120 |
| n_layers | 64 | 64 |
| n_heads / n_kv_heads | 40 / 8 | 40 / 8 |
| intermediate_size | 25600 | 27648 |
| RoPE | factor-based (linear) | YaRN |
| Attention | Full only | 48 sliding (4096) + 16 full |
| Q/K norm | Yes | No |

---

## What Exists in llama3_70b_galaxy

- **TtQwenModelArgs** for Qwen3-32B: same dims (5120, 64 layers, 40/8 heads)
- **Load checkpoints**: HF→Meta mapping for Llama-style keys; Olmo uses same structure (`model.embed_tokens`, `model.layers.*.self_attn`, `model.layers.*.mlp`) — no q_norm/k_norm
- **Galaxy stack**: CCL, prefetcher, mesh config, paged attention
- **ttnn SDPA**: `scaled_dot_product_attention`, `scaled_dot_product_attention_decode`, `ring_distributed_scaled_dot_product_attention` all support `sliding_window_size`
- **tt_transformers**: `layer_types`, `sliding_window`, `RopeScalingYarn` in common.py and rope.py
- **GPT-OSS demo**: Per-layer sliding window via `layer_types` and `sliding_window` in attention

---

## Gaps vs. Olmo-3-32B

### 1. Per-layer sliding-window attention

**Current:** `llama_attention.py` does not pass `sliding_window_size` to SDPA (prefill or decode).

**Required:** Per-layer `sliding_window_size`:
- `sliding_attention` → 4096
- `full_attention` → None

**Effort:** 1–2 days
- Add `layer_types` and `sliding_window` to model config
- Pass per-layer `sliding_window_size` into `scaled_dot_product_attention`, `scaled_dot_product_attention_decode`, and `ring_distributed_scaled_dot_product_attention`
- Mirror GPT-OSS pattern in [models/demos/gpt_oss/tt/attention/](models/demos/gpt_oss/tt/attention/)

### 2. YaRN RoPE

**Current:** `llama_rope.py` uses Llama-3–style `apply_scaling` (factor-based), not YaRN.

**Required:** Olmo YaRN params: `rope_type=yarn`, `factor=8`, `original_max_position_embeddings=8192`, `attention_factor=1.2079`, `beta_fast=32`, `beta_slow=1.0`.

**Effort:** 1–2 days
- Option A: Use tt_transformers `RopeScalingYarn` and rope factory in the Galaxy stack
- Option B: Implement YaRN in `llama_common.py` (similar to `apply_scaling` but with YaRN formula)
- tt_transformers already has `RopeScalingYarn` with `beta_fast`, `beta_slow`, `mscale`; verify compatibility with Olmo’s `attention_factor`

### 3. Model config (TtOlmoModelArgs)

**Required:** New config class with:
- dim=5120, n_layers=64, n_heads=40, n_kv_heads=8
- intermediate_size=27648
- vocab_size=100278
- layer_types, sliding_window=4096
- rope_scaling (YaRN)
- qk_norm=False

**Effort:** 0.5–1 day
- Copy `TtQwenModelArgs` and adjust dims, rope, and attention settings

### 4. Load checkpoints

**Required:** Olmo uses Llama-style keys; no q_norm/k_norm. Existing `map_hf_to_meta_keys` should work.

**Effort:** ~0.5 day
- Add Olmo-specific path in `load_checkpoints.py` if needed
- Confirm Olmo HF keys match Llama (e.g. `input_layernorm` vs `input_layernorm`)

### 5. Prefetcher / verified configs

**Required:** Add Olmo-3-32B to prefetcher verified configs if prefetcher is used.

**Effort:** ~0.5 day

### 6. Tokenizer

**Required:** Olmo tokenizer (vocab 100278). Use HuggingFace `AutoTokenizer.from_pretrained("allenai/Olmo-3-1125-32B")`.

**Effort:** Minimal (standard HF tokenizer)

---

## Effort Summary

| Task | Effort | Notes |
|------|--------|-------|
| TtOlmoModelArgs + config | 0.5–1 day | Reuse Qwen config |
| Load checkpoints | 0.5 day | Likely no changes |
| Per-layer sliding window in attention | 1–2 days | Main code change |
| YaRN RoPE integration | 1–2 days | Reuse or adapt tt_transformers |
| Prefetcher / verified configs | 0.5 day | If needed |
| Integration, tests, accuracy | 2–3 days | End-to-end validation |

**Total: ~6–10 person-days (about 1.5–2 weeks)** for a developer familiar with the Galaxy stack.

---

## Risk / Complexity

1. **Sliding window + ring SDPA:** `ring_distributed_scaled_dot_product_attention` may need validation with `sliding_window_size`; current code does not pass it.
2. **Sliding window + paged attention:** `attention_1d.py` rejects `sliding_window` with paged attention. Confirm whether Olmo on Galaxy uses paged attention and if that combination is supported.
3. **YaRN params:** Olmo’s `attention_factor` may not map 1:1 to tt_transformers `RopeScalingYarn`; may need small extensions.

---

## Suggested Implementation Order

1. Add `TtOlmoModelArgs` and load Olmo weights (no sliding window, linear RoPE) to confirm base run.
2. Integrate YaRN RoPE.
3. Add per-layer sliding window to attention.
4. Run full accuracy and performance tests.
