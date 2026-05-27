# Qwen3.5-27B Architecture Analysis

## Model Family
**Hybrid Decoder-only LLM** — Qwen3.5-27B is a 64-layer transformer with *two interleaved attention types*:
- **GDN (Gated DeltaNet)** — linear attention via causal conv1d + DeltaNet recurrence (48 of 64 layers)
- **Full GQA** — standard grouped-query attention with partial RoPE (16 of 64 layers)

Layer pattern: `[GDN, GDN, GDN, FullAttn] × 16`

---

## Architecture Parameters

| Parameter | Value |
|-----------|-------|
| Total layers | 64 |
| GDN (linear attn) layers | 48 |
| Full attention layers | 16 |
| Hidden size | 5120 |
| MLP intermediate size | 17408 |
| Attention heads (Q) | 24 |
| KV heads | 4 |
| Head dimension | 256 |
| RoPE dims (of 256) | 64 (25 %) |
| RoPE theta | 10,000,000 |
| GDN key heads / dim | 16 × 128 |
| GDN value heads / dim | 48 × 128 |
| Conv kernel size | 4 (causal) |
| Vocabulary size | 248,320 |
| Max context length | 262,144 |
| Weight format | FP8 block (128 × 128) |

---

## Complete Component Inventory

| Component | Location | Status |
|-----------|----------|--------|
| Embedding + LM head | framework TTTransformer | ✅ Complete |
| Full GQA attention (16 layers) | `tt/attention.py` | ✅ Complete |
| GDN linear attention (48 layers) | `tt/gdn.py` | ✅ Complete |
| Fused GDN kernel (decode) | `tt/gdn_kernel/gdn_fused.cpp` | ✅ Complete |
| Prefill GDN kernel | `tt/gdn_kernel/gdn_prefill.cpp` | ✅ Complete |
| Causal conv1d (4-tap, in GDN) | `tt/gdn.py` (inline) | ✅ Complete |
| Partial RoPE (64 of 256 dims) | `tt/rope.py` | ✅ Complete |
| SwiGLU MLP with fused W1+W3 | `tt/fused_mlp.py` | ✅ Complete |
| RMSNorm with unit offset | framework `models/common/rmsnorm.py` | ✅ Complete |
| QK L2 norm (full attention) | `tt/attention.py` (`_rms_norm_dev`) | ✅ Complete |
| Sigmoid output gate (both attn) | `tt/attention.py`, `tt/gdn.py` | ✅ Complete |
| KV cache (paged + standard) | framework + `tt/model.py` | ✅ Complete |
| L1 rolling window state | `tt/model.py` | ✅ Complete |
| FP8 weight dequantization | `tt/model_config.py` | ✅ Complete |
| vLLM adapter | `tt/generator_vllm.py` | ✅ Complete |
| Chunked prefill | `tt/model.py` | ✅ Complete |

---

## Similar Implementations

### Full Attention Layers (16/64 layers)

| Block | Reference Implementation | Similarity | Differences |
|-------|-------------------------|------------|-------------|
| GQA + KV cache | `models/demos/llama3_70b_galaxy/tt/llama_attention.py` | GQA structure, decode/prefill split | QK L2 norm, sigmoid output gate, partial RoPE |
| Partial RoPE | `models/demos/deepseek_v3/tt/rope.py` | Partial rotation concept | Only 64 of 256 dims; `mrope_section` interleaving |
| RMSNorm (+1 offset) | `models/common/rmsnorm.py` | Pre-norm placement | `rms_norm_add_unit_offset=True` (Gemma-style) |
| SwiGLU MLP | `models/demos/llama3_70b_galaxy/tt/llama_mlp.py` | W1×gate + W3×up → W2 | W1+W3 fused into single DRAM-sharded matmul |

### GDN Layers (48/64 layers) — Novel, No Direct Reference

| Block | Closest Reference | Similarity | Key Differences |
|-------|------------------|------------|-----------------|
| Causal conv1d (4-tap) | No direct reference | Conv kernel concept | 4-tap shift register state; not a standard attention op |
| DeltaNet recurrence | No direct reference | Linear attention family | `state = decay·state + k⊗(β·(v − k^T state))` |
| L2-normalized QK | `models/demos/qwen3_vl/tt/vision_attention.py` | QK norm | L2 norm (divide by magnitude) rather than RMSNorm |
| GDN fused kernel | No reference | Custom TT kernel | Fuses L2 norm + gates + recurrence + RMSNorm + SiLU |

---

## Key Architectural Differences from Standard Llama/Qwen

1. **Hybrid linear + full attention** — 75 % of layers are GDN (O(n) in sequence length), not O(n²)
2. **Partial RoPE** — Only the first 64 of 256 head dims are rotated (`partial_rotary_factor=0.25`)
3. **QK L2 norm** — Full attention Q and K are L2-normalized (with learned scale) before attention
4. **Sigmoid output gate** — Both GDN and full attention produce `output * sigmoid(gate_proj(x))`
5. **GDN causal conv1d** — 4-tap causal convolution applied to input before GDN projection
6. **RMSNorm unit offset** — All layer norms use `(weight + 1.0) * x` (Gemma format)
7. **FP8 block quantization** — Weights stored as `float8_e4m3fn` with 128×128 block scale factors

---

## Weight Mapping (HuggingFace → TTNN)

### Full Attention Layers (`full_attention_interval=4`)

| HuggingFace Key | TTNN Key | Notes |
|-----------------|----------|-------|
| `model.layers.{i}.self_attn.q_proj.weight` | `attention.wqg` (fused with gate) | Q + gate_proj fused, interleaved for TP |
| `model.layers.{i}.self_attn.gate_proj.weight` | `attention.wqg` (fused with Q) | See above |
| `model.layers.{i}.self_attn.k_proj.weight` | `attention.wk` | Separate; replicated if TP > n_kv_heads |
| `model.layers.{i}.self_attn.v_proj.weight` | `attention.wv` | Same as K |
| `model.layers.{i}.self_attn.o_proj.weight` | `attention.wo` | Row-parallel |
| `model.layers.{i}.self_attn.q_norm.weight` | `attention.q_norm` | L2 scale |
| `model.layers.{i}.self_attn.k_norm.weight` | `attention.k_norm` | L2 scale |
| `model.layers.{i}.mlp.gate_proj.weight` | `mlp.w1` (fused with w3) | W1+W3 interleaved for TP |
| `model.layers.{i}.mlp.up_proj.weight` | `mlp.w3` (fused with w1) | See above |
| `model.layers.{i}.mlp.down_proj.weight` | `mlp.w2` | Row-parallel |
| `model.layers.{i}.input_layernorm.weight` | `norm` | +1 unit offset applied |
| `model.layers.{i}.post_feedforward_layernorm.weight` | `ffn_norm` | +1 unit offset applied |

### GDN Layers (`linear_attention`)

| HuggingFace Key | TTNN Key | Notes |
|-----------------|----------|-------|
| `model.layers.{i}.linear_attn.qkv_proj.weight` | `gdn.qkvz` | Q+K+V+Z fused; TP-sharded |
| `model.layers.{i}.linear_attn.ab_proj.weight` | `gdn.ab` | A+B fused for DeltaNet decay |
| `model.layers.{i}.linear_attn.out_proj.weight` | `gdn.out` | Row-parallel |
| `model.layers.{i}.linear_attn.A_log` | `gdn.A_log` | Per-head log decay param |
| `model.layers.{i}.linear_attn.dt_bias` | `gdn.dt_bias` | DeltaNet time-step bias |
| `model.layers.{i}.linear_attn.inner_attn_ln.weight` | `gdn.norm_w` | RMSNorm before SiLU gate |
| `model.layers.{i}.linear_attn.conv1d.weight` | `gdn.conv_taps[0..3]` | 4-tap causal conv, interleaved TP |
| `model.layers.{i}.input_layernorm.weight` | `norm` | +1 unit offset applied |

---

## Tensor Parallelism (TP=4, P150×4)

| Dimension | Strategy | Detail |
|-----------|----------|--------|
| Q / gate projection | Column-parallel | `[hidden, heads*head_dim]` → each device gets `heads*head_dim/TP` |
| K / V projection | Column-parallel + replicate | 4 KV heads; each of 4 devices gets 1 KV head |
| Output projection | Row-parallel | All-reduce across TP |
| MLP W1+W3 | Column-parallel (fused) | Interleaved shards for single matmul |
| MLP W2 | Row-parallel | All-reduce across TP |
| GDN qkvz | Column-parallel | Full `[hidden, qkvz_dim]` → TP shards |
| GDN out | Row-parallel | All-reduce |
| Embedding | Vocab-parallel | — |

---

## Memory Architecture

| Buffer | Location | Dtype | Notes |
|--------|----------|-------|-------|
| Model weights | DRAM | BF16 (dequant from FP8) | DRAM-sharded for decode |
| KV caches (full attn) | DRAM | BF8 | Paged or contiguous |
| GDN rec_states (active window) | L1 | BF16 | 3-layer rolling window |
| GDN rec_states (other layers) | DRAM | BF16 | Swapped in/out per group |
| GDN conv_states | L1 | BF16 | 4-tap shift register |
| Activations | L1 | BF16 | Tiled for matmuls |

**L1 rolling window:** The 3 GDN + 1 attn pattern means GDN state for one group (3 layers) fits in L1 simultaneously. The `enable_l1_state()` / `_swap_l1_state()` mechanism swaps between active and backup DRAM copies at each group boundary.

---

## Kernel Architecture (GDN)

```
Input token x  [1, hidden]
    │
    ├─ conv1d (4-tap causal)      ← shift register state
    │       ↓
    │   conv_out  [1, hidden]
    │       │
    ├─ qkvz_proj               →  [Q, K, V, Z]  (fused matmul, DRAM-sharded)
    │       │
    │   L2-norm Q, L2-norm K
    │       │
    ├─ ab_proj                 →  [A, B]
    │       │
    │   decay = neg_exp(A * softplus(dt_bias + B))
    │       │
    │   DeltaNet recurrence:
    │   state_new = decay * state + K ⊗ (beta * (V - K^T state))
    │                                             ← gdn_fused_inplace kernel
    │       │
    │   out_raw = state_new @ K     [1, Dv]
    │       │
    │   RMSNorm(out_raw) * SiLU(Z)  (gated output)
    │       │
    └─ out_proj                →  [1, hidden]
```

**Fused kernel speedup:** `gdn_full_fused_inplace` combines L2 norm + DeltaNet step + RMSNorm + SiLU gate into one kernel dispatch (~12× vs pure TTNN ops). Each kernel call processes one (batch × value-head) pair. State tiles are 4×4 (128×128 per pair).

---

## GDN vs Full Attention — Comparison

| Property | GDN (48 layers) | Full GQA (16 layers) |
|----------|-----------------|---------------------|
| Complexity | O(n) per layer (recurrence) | O(n²) or O(n) with paged |
| State | 2 per layer: conv + rec | KV cache (grows with context) |
| Parallelism | Sequential recurrence in prefill | Parallel flash attention |
| Bottleneck | Prefill speed (serial), custom kernel needed | KV cache memory |
| Key challenge | State management across requests | Paged attention for long ctx |

---

## Implementation Status

The model is **production-ready** on 4×P150 Blackhole (TP=4):
- ✅ Decode path fully fused with custom GDN kernel
- ✅ Chunked prefill with batched projections
- ✅ L1 rolling window state management
- ✅ Paged KV cache for full attention layers
- ✅ vLLM integration
- ✅ FP8 weight loading with block dequantization

**Current baseline (decode, batch=32):** ~68.6 ms/step | ~14.6 tok/s/user

Open optimizations tracked in `OPTIMIZATIONS.md` (24 categories, 439 lines).
