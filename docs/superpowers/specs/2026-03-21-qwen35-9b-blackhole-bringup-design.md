# Qwen3.5-9B Text-Only Bringup on Blackhole P150

**Date:** 2026-03-21
**Status:** Design Approved
**Device:** Blackhole P150 (single device)
**Model:** Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2
**Weights:** `/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2` (safetensors, ~18.1GB bfloat16)
**Scope:** Text-only language model (no vision encoder)

---

## 1. Architecture Overview

Qwen3.5-9B is a **hybrid linear-attention + softmax-attention** transformer, not a standard Llama-style model. Its defining feature is the **Gated DeltaNet** mechanism in 75% of layers.

### Core Dimensions

| Parameter | Value |
|---|---|
| hidden_size | 4096 |
| num_layers | 32 |
| vocab_size | 248,320 |
| MLP intermediate_size | 12,288 |
| MLP activation | SwiGLU (SiLU) |
| norm | RMSNorm (eps=1e-6) |
| max_position_embeddings | 262,144 (start with 8192) |

### Hybrid Attention Pattern

32 layers follow a repeating `[linear, linear, linear, full_attention]` pattern (×8):

**Gated DeltaNet (linear attention) — 24 layers** (0-2, 4-6, 8-10, 12-14, 16-18, 20-22, 24-26, 28-30):
- num_key_heads: 16, key_head_dim: 128
- num_value_heads: 32, value_head_dim: 128
- Causal Conv1D kernel_size: 4
- Maintains fixed-size recurrent state [B, H, K, V] — O(1) memory per token
- Delta rule with exponential gating

**Gated Full Attention (softmax GQA) — 8 layers** (3, 7, 11, 15, 19, 23, 27, 31):
- num_attention_heads (Q): 16, head_dim: 256
- num_kv_heads: 4 (GQA ratio 4:1)
- Attention output gate (sigmoid)
- Q/K RMSNorm before attention
- RoPE with partial_rotary_factor=0.25 (64 of 256 dims), rope_theta=10M

### Key Differences from Llama

| Feature | Llama | Qwen3.5-9B |
|---|---|---|
| Attention | All layers softmax GQA | Hybrid: 75% DeltaNet, 25% GQA |
| head_dim | 128 | 256 (full attn), 128 (linear) |
| partial_rotary_factor | 1.0 | 0.25 |
| rope_theta | 500K | 10,000,000 |
| Conv1D in attention | No | Yes (DeltaNet layers) |
| Attention output gate | No | Yes |
| vocab_size | 128,256 | 248,320 |

---

## 2. Approach: Wrap Experimental Branch TTNN Ops

The branch `sdawle/gated_attention_gated_deltanet` contains validated TTNN implementations of both Gated Attention and Gated DeltaNet under `models/experimental/gated_attention_gated_deltanet/`.

**What exists and will be reused:**
- `tt/ttnn_gated_attention.py` — SDPA + sigmoid gate, FlashAttention-2, PCC ≥ 0.999
- `tt/ttnn_gated_deltanet.py` — recurrent + chunked modes, PCC ≥ 0.999
- `tt/ttnn_delta_rule_ops.py` — core delta rule algorithms
- `torch_functional/` — PyTorch reference implementations for validation

**What we build on top:**
- Model config (Qwen35ModelArgs)
- Weight remapping (HF → experimental op format)
- Layer assembly (TransformerBlock with hybrid dispatch)
- Full model (embedding → layers → norm → LM head)
- State management (KV cache + recurrent state)
- Blackhole P150 device tuning
- Demo and tests

---

## 3. Directory Structure

```
models/demos/blackhole/qwen3_5_9b/
├── tt/
│   ├── model_config.py            # Qwen35ModelArgs(ModelArgs)
│   ├── qwen35_model.py            # Qwen35Transformer - full model
│   ├── qwen35_decoder.py          # Qwen35TransformerBlock - hybrid layer
│   ├── qwen35_gated_attention.py  # Wraps experimental gated attention
│   ├── qwen35_gated_deltanet.py   # Wraps experimental gated deltanet
│   ├── qwen35_mlp.py              # MLP (reuse base or thin wrapper)
│   ├── qwen35_rope.py             # RoPE with partial_rotary_factor=0.25
│   └── weight_mapping.py          # HF→internal weight remapping
├── demo/
│   └── demo.py                    # End-to-end text generation
└── tests/
    ├── test_model_config.py       # Config loading validation
    ├── test_single_layer.py       # Per-layer PCC validation
    └── test_model_e2e.py          # Full model inference test
```

---

## 4. Model Config (`model_config.py`)

`Qwen35ModelArgs` subclasses `ModelArgs` and overrides `_set_hf_params()`:

```python
# Key parameters from config.json
dim = 4096
n_layers = 32
n_heads = 16              # full attention Q heads
n_kv_heads = 4            # full attention KV heads
head_dim = 256            # full attention head dim
hidden_dim = 12288        # MLP intermediate
vocab_size = 248320
norm_eps = 1e-6
rope_theta = 10_000_000
partial_rotary_factor = 0.25

# DeltaNet-specific
linear_num_key_heads = 16
linear_num_value_heads = 32
linear_key_head_dim = 128
linear_value_head_dim = 128
linear_conv_kernel_dim = 4

# Layer type map
attention_type_list = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8
```

**Blackhole P150 device config:**
- max_batch_size: 1 (initial bringup, scale to 4 later)
- max_seq_len: 2048 (initial, scale to 8192 later)
- Weight dtype: `ttnn.bfloat8_b` (~9GB in DRAM)
- Activation dtype: `ttnn.bfloat16`

---

## 5. Weight Mapping (`weight_mapping.py`)

### Full Attention Layers (self_attn)

```
HF key                                    → Internal key
self_attn.q_proj.weight [8192, 4096]      → q_proj_weight (2x wide: query + gate)
self_attn.k_proj.weight [1024, 4096]      → k_proj_weight
self_attn.v_proj.weight [1024, 4096]      → v_proj_weight
self_attn.o_proj.weight [4096, 4096]      → o_proj_weight
self_attn.q_norm.weight [256]             → q_norm_weight
self_attn.k_norm.weight [256]             → k_norm_weight
```

Note: `q_proj` is 2× wide (8192 = 16 heads × 256 head_dim × 2). First half is query, second half is sigmoid gate. The experimental gated attention op handles this split internally.

### DeltaNet Layers (linear_attn)

```
HF key                                    → Internal key
linear_attn.in_proj_qkv.weight [8192, 4096] → split Q [2048], K [2048], V [4096]
linear_attn.in_proj_a.weight [32, 4096]   → a_proj_weight (decay)
linear_attn.in_proj_b.weight [32, 4096]   → b_proj_weight (beta/write strength)
linear_attn.in_proj_z.weight [4096, 4096] → g_proj_weight (output gate)
linear_attn.conv1d.weight [8192, 1, 4]    → split q_conv [2048,1,4], k_conv [2048,1,4], v_conv [4096,1,4]
linear_attn.A_log [32]                    → A_log
linear_attn.dt_bias [32]                  → dt_bias
linear_attn.norm.weight [128]             → o_norm_weight
linear_attn.out_proj.weight [4096, 4096]  → o_proj_weight
```

Split sizes for `in_proj_qkv` (dim 0): Q = `[0:2048]`, K = `[2048:4096]`, V = `[4096:8192]`
Split sizes for `conv1d.weight` (dim 0): same Q/K/V partitioning.

### Common Per-Layer

```
input_layernorm.weight      → attention_norm
post_attention_layernorm.weight → ff_norm
mlp.gate_proj.weight        → w1
mlp.up_proj.weight          → w3
mlp.down_proj.weight        → w2
```

### Top-Level

```
model.language_model.embed_tokens.weight → tok_embeddings.weight
lm_head.weight                           → output.weight
model.language_model.norm.weight         → norm.weight (final RMSNorm)
```

All keys strip the `model.language_model.` prefix. Vision encoder and MTP weights are ignored (text-only scope).

---

## 6. Component Design

### `qwen35_gated_attention.py`

LightweightModule wrapping `gated_attention_forward_ttnn()`:
- Stores q/k/v/o projection weights, q_norm, k_norm as ttnn tensors
- Receives RoPE cos/sin from rope setup, passes to the TTNN op
- Manages KV cache for the 8 full-attention layers
- Adapts SDPA program configs for Blackhole grid

### `qwen35_gated_deltanet.py`

LightweightModule wrapping `gated_deltanet_forward_ttnn()`:
- Stores q/k/v projections, a_proj, b_proj, g_proj, conv weights, A_log, dt_bias, o_norm, o_proj
- Maintains recurrent state `[B, 16, 128, 128]` per layer (fixed size, replaces KV cache)
- Maintains conv state `[B, 8192, 3]` per layer (causal conv1d history)
- Decode (T=1): `mode="recurrent"` — single-step state update
- Prefill (T>1): `mode="chunk"`, `chunk_size=64` — parallel chunked processing

### `qwen35_mlp.py`

Reuses base MLP class (same SwiGLU: gate_proj × silu, up_proj, down_proj). Thin wrapper if any weight name adjustments needed.

### `qwen35_rope.py`

Custom RoPE setup:
- Subclass `HfRotarySetup` with `head_dim = int(256 * 0.25) = 64` for rotary portion
- rope_theta = 10,000,000
- Only used by the 8 Gated Attention layers; DeltaNet layers don't use RoPE

### `qwen35_decoder.py`

Hybrid TransformerBlock:
```python
class Qwen35TransformerBlock(LightweightModule):
    def __init__(self, args, layer_num, ...):
        if args.attention_type_list[layer_num] == "full_attention":
            self.attention = Qwen35GatedAttention(...)
        else:
            self.attention = Qwen35GatedDeltaNet(...)
        self.feed_forward = MLP(...)  # same for both

    def forward(self, x, current_pos, rot_mats=None, ...):
        h = x + self.attention(self.attention_norm(x), rot_mats, ...)
        out = h + self.feed_forward(self.ff_norm(h))
        return out
```

### `qwen35_model.py`

Full model: `tok_embeddings → 32 × Qwen35TransformerBlock → RMSNorm → LMHead`

---

## 7. State Management & Inference Flow

### State Types

| Layer Type | State | Shape (B=1) | Count | Memory |
|---|---|---|---|---|
| Gated Attention | KV cache | [1, 4, seq_len, 256] × 2 | 8 layers | ~32MB at 2048 seq |
| Gated DeltaNet | Recurrent state | [1, 32, 128, 128] | 24 layers | ~24MB total |
| Gated DeltaNet | Conv state | [1, 8192, 3] | 24 layers | ~2MB total |

DeltaNet recurrent state is **fixed size** regardless of sequence length.

### Prefill Flow

1. Tokenize input → token_ids [B, T]
2. Embed tokens → x [B, T, 4096]
3. Compute RoPE cos/sin for positions [0..T-1], head_dim=64
4. For each layer 0..31:
   - If DeltaNet: forward mode="chunk", chunk_size=64; store recurrent_state + conv_state
   - If GatedAttention: forward with RoPE; write to KV cache [0..T-1]
   - MLP forward
5. Final RMSNorm → LM Head → logits [B, vocab_size]
6. Sample next token

### Decode Flow (T=1, autoregressive)

1. Embed single token → x [B, 1, 4096]
2. Compute RoPE for current_pos
3. For each layer 0..31:
   - If DeltaNet: forward mode="recurrent"; update recurrent_state + conv_state
   - If GatedAttention: forward with RoPE; update KV cache at current_pos
   - MLP forward
4. Final RMSNorm → LM Head → logits
5. Sample, increment current_pos

### State Initialization

- DeltaNet recurrent states: zeros [B, 16, 128, 128]
- DeltaNet conv states: zeros [B, 8192, 3]
- KV cache: pre-allocated to max_seq_len on device DRAM

---

## 8. Blackhole P150 Considerations

### Device Tuning

- Compute grid may differ from Wormhole's 8×8 — verify at init time
- L1 SRAM size per core affects memory config thresholds
- Program configs (matmul tiling, grid dims) need Blackhole-specific values
- SDPA kernel config for gated attention must be validated

### Memory Strategy

- Weights: `bfloat8_b` in DRAM (~9GB)
- Activations: `bfloat16`, L1 where possible, DRAM fallback
- DeltaNet recurrent state: L1 (small, hot path)
- KV cache: DRAM (grows with seq_len)
- Start conservative: batch=1, seq_len=2048

### Known Risks

1. **Blackhole compatibility** — experimental ops validated on Wormhole only. May need grid/tile adjustments.
2. **Conv1d on Blackhole** — DeltaNet conv1d may behave differently. FIR fallback exists.
3. **Large vocab LM Head** — 248,320 vocab needs split matmuls (follow base LMHead pattern).
4. **Weight remapping correctness** — in_proj_qkv and conv1d splits must match HF packing order exactly. Validate against reference model output.

---

## 9. Testing Strategy

### Phase 1: Config & Weight Loading (no device)
- Verify Qwen35ModelArgs loads config.json correctly
- Verify remap_qwen35_weights() produces correctly shaped tensors
- Compare against HuggingFace reference model load

### Phase 2: Single Layer Validation (on P150)
- Run one DeltaNet layer + one Gated Attention layer
- Compare TTNN output vs torch reference (PCC ≥ 0.99)
- Test both prefill (T=128) and decode (T=1)

### Phase 3: End-to-End Model (on P150)
- Load full 32-layer model, run short prompt
- Compare logits against HuggingFace transformers reference (PCC ≥ 0.98)
- Validate coherent text generation

### Phase 4: Demo
- Interactive text generation with reasoning prompts
- Exercise the model's distilled reasoning capabilities

---

## 10. Dependencies

- **Experimental branch:** `sdawle/gated_attention_gated_deltanet` — Gated Attention + DeltaNet TTNN ops
- **Base framework:** `models/tt_transformers/tt/` — ModelArgs, MLP, Embedding, LMHead, RoPE, load_checkpoints
- **HF weights:** `/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2`
- **HF transformers:** For tokenizer and reference model validation
