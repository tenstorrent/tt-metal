# MiniMax-M2.5 Architecture Analysis

## Model Family
Decoder-only LLM, MoE (Mixture of Experts) — 229B parameters, 62 layers, all-MoE (no dense layers).

HuggingFace: https://huggingface.co/MiniMaxAI/MiniMax-M2.5

---

## Config Summary (`config.json`)

| Parameter | Value |
|-----------|-------|
| `num_hidden_layers` | 62 |
| `hidden_size` | 3072 |
| `head_dim` | 128 |
| `num_attention_heads` | 48 (Q) |
| `num_key_value_heads` | 8 (KV) — GQA 6:1 |
| `num_local_experts` | 256 |
| `num_experts_per_tok` | 8 |
| `intermediate_size` | 1536 (per-expert FFN hidden) |
| `shared_intermediate_size` | 0 (no shared experts) |
| `scoring_func` | sigmoid |
| `use_routing_bias` | true (`e_score_correction_bias`) |
| `rotary_dim` | 64 (partial RoPE, NoPE on remaining 64 dims) |
| `rope_theta` | 5,000,000 |
| `use_qk_norm` | true |
| `qk_norm_type` | per_layer |
| `use_mtp` | true |
| `num_mtp_modules` | 3 |
| `mtp_transformer_layers` | 1 |
| `vocab_size` | 200,064 |
| `max_position_embeddings` | 196,608 |
| Quantization | fp8 (float8_e4m3fn), block 128×128 |

---

## Complete Component Inventory

| Component | Weight Key Prefix | Required For | Implementation Status |
|-----------|-------------------|--------------|----------------------|
| Token Embedding | `model.embed_tokens` | Input projection | Not started |
| Decoder Layers (×62) | `model.layers.{i}` | Main inference | Not started |
| — Attention | `model.layers.{i}.self_attn` | Token mixing | Not started |
| — QK-norm | `model.layers.{i}.self_attn.q_norm / k_norm` | Attention stability | Not started |
| — MoE Block | `model.layers.{i}.block_sparse_moe` | FFN / experts | Not started |
| — Router gate | `model.layers.{i}.block_sparse_moe.gate` | Expert routing | Not started |
| — Routing bias | `model.layers.{i}.block_sparse_moe.e_score_correction_bias` | Expert routing | Not started |
| — Experts (×256) | `model.layers.{i}.block_sparse_moe.experts.{j}` | FFN compute | Not started |
| — Input LayerNorm | `model.layers.{i}.input_layernorm` | Pre-attn norm | Not started |
| — Post-attn LayerNorm | `model.layers.{i}.post_attention_layernorm` | Pre-MoE norm | Not started |
| Final Norm | `model.norm` | Output norm | Not started |
| LM Head | `lm_head` | Token sampling | Not started |
| MTP Modules (×3) | `model.mtp_modules.{k}` | Training only — **skip during inference** | N/A |

---

## Similar Implementations

| Component | Reference Implementation | Similarity |
|-----------|-------------------------|------------|
| GQA Attention (decode/prefill) | `models/demos/gpt_oss/tt/attention/` | Same Galaxy 4×8, GQA, paged attn, fused QKV |
| MoE Experts + EP/TP | `models/demos/gpt_oss/tt/experts/` | Same Galaxy target, same EP+TP parallelism |
| TopK Router | `models/demos/gpt_oss/tt/topk.py` | Same top-k selection pattern |
| MeshConfig (EP+TP) | `models/demos/gpt_oss/config.py` | Same 4×8 mesh with TP/EP modes |
| CCL (all-reduce/all-gather) | `models/demos/gpt_oss/tt/ccl.py` | Same Galaxy CCL patterns |
| RMSNorm | `models/demos/gpt_oss/tt/rms_norm.py` | Identical |
| KV-Cache | `models/demos/gpt_oss/tt/attention/kv_cache.py` | Same paged cache pattern |
| Partial RoPE (nope+rope split) | `models/demos/deepseek_v3/tt/rope.py` | `qk_rope_head_dim` concept maps to `rotary_dim=64` |
| Routing bias | `models/demos/deepseek_v3/tt/moe_gate.py` | `e_score_correction_bias` is identical concept |
| QK-norm (per-layer) | `models/demos/qwen3_vl/tt/vision_attention.py` | Per-layer RMSNorm on Q/K before reshape |

---

## Key Differences from References

### vs `gpt_oss`
1. **QK-norm** (`use_qk_norm=true`, `qk_norm_type="per_layer"`): Applied to full flattened Q/K *after* projection but *before* reshape to heads. Shape: `q_norm=(48*128,)`, `k_norm=(8*128,)`. Not present in gpt_oss.
2. **Partial RoPE** (`rotary_dim=64`): Only first 64 of 128 head dims get RoPE; remaining 64 are NoPE (passed through unchanged). gpt_oss applies RoPE to full head_dim.
3. **Routing bias** (`e_score_correction_bias`): Added to sigmoid scores before top-k selection. Not in gpt_oss.
4. **No shared experts**: `shared_intermediate_size=0`. gpt_oss may have shared experts depending on model.
5. **MTP modules** (3×): Multi-token prediction heads. New component not in gpt_oss.
6. **vocab_size=200,064**: Larger vocab — affects embedding and lm_head padding.

### vs `deepseek_v3`
1. **GQA not MLA**: Plain `q_proj/k_proj/v_proj` (no `kv_lora_rank`, `q_lora_rank`). Much simpler attention.
2. **1x Galaxy** (4×8) not 2x/4x: Single-host, no multi-host MPI.
3. **All layers are MoE**: No `first_k_dense_replace` — all 62 layers use MoE.
4. **3 MTP modules** vs DeepSeek's 1.

---

## Weight Mapping (HuggingFace → TTNN)

| HuggingFace Key | TTNN Key | Notes |
|-----------------|----------|-------|
| `model.embed_tokens.weight` | `embed_tokens.weight` | Vocab embedding |
| `model.layers.{i}.self_attn.q_proj.weight` | `layers.{i}.attention.wq.weight` | [48*128, 3072] |
| `model.layers.{i}.self_attn.k_proj.weight` | `layers.{i}.attention.wk.weight` | [8*128, 3072] |
| `model.layers.{i}.self_attn.v_proj.weight` | `layers.{i}.attention.wv.weight` | [8*128, 3072] |
| `model.layers.{i}.self_attn.o_proj.weight` | `layers.{i}.attention.wo.weight` | [3072, 48*128] |
| `model.layers.{i}.self_attn.q_norm.weight` | `layers.{i}.attention.q_norm.weight` | [48*128] — per-layer QK-norm |
| `model.layers.{i}.self_attn.k_norm.weight` | `layers.{i}.attention.k_norm.weight` | [8*128] — per-layer QK-norm |
| `model.layers.{i}.input_layernorm.weight` | `layers.{i}.input_layernorm.weight` | [3072] |
| `model.layers.{i}.post_attention_layernorm.weight` | `layers.{i}.post_attention_layernorm.weight` | [3072] |
| `model.layers.{i}.block_sparse_moe.gate.weight` | `layers.{i}.moe.gate.weight` | [256, 3072] |
| `model.layers.{i}.block_sparse_moe.e_score_correction_bias` | `layers.{i}.moe.routing_bias` | [256] |
| `model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight` | `layers.{i}.moe.experts.{j}.w1` | gate proj [1536, 3072] |
| `model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight` | `layers.{i}.moe.experts.{j}.w2` | down proj [3072, 1536] |
| `model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight` | `layers.{i}.moe.experts.{j}.w3` | up proj [1536, 3072] |
| `model.norm.weight` | `norm.weight` | [3072] |
| `lm_head.weight` | `lm_head.weight` | [200064, 3072] |

---

## Attention Block — Critical Details

### QK-norm (per_layer)
Applied to the **full flattened** Q and K tensors after projection, before reshape to per-head view:
```python
# HuggingFace forward order:
query_states = self.q_proj(hidden_states)  # [B, S, 48*128]
key_states   = self.k_proj(hidden_states)  # [B, S, 8*128]
value_states = self.v_proj(hidden_states)  # [B, S, 8*128]

query_states = self.q_norm(query_states)   # RMSNorm([48*128])
key_states   = self.k_norm(key_states)     # RMSNorm([8*128])

# Then reshape to [B, S, heads, head_dim] → transpose → RoPE
```
This is different from per-head QK-norm (Molmo) — the norm is applied across all heads jointly.

### Partial RoPE (`rotary_dim=64`)
```python
rotary_dim = 64  # cos/sin has shape [B, S, 64]
q_rot, q_pass = q[..., :64], q[..., 64:]   # split head_dim
k_rot, k_pass = k[..., :64], k[..., 64:]
q_embed = apply_rotary(q_rot, cos, sin)
k_embed = apply_rotary(k_rot, cos, sin)
q = torch.cat([q_embed, q_pass], dim=-1)    # NoPE passthrough
k = torch.cat([k_embed, k_pass], dim=-1)
```

---

## MoE Block — Critical Details

```python
# Routing:
router_logits = gate(hidden_states)               # [T, 256]
routing_weights = sigmoid(router_logits)          # NOT softmax
scores = routing_weights + e_score_correction_bias
_, top_k_index = topk(scores, k=8)               # select top-8 experts
top_k_weights = routing_weights.gather(top_k_index)
top_k_weights /= top_k_weights.sum(dim=-1)       # normalize selected weights

# Expert compute (SwiGLU):
# w1 = gate_proj, w3 = up_proj, w2 = down_proj
output = w2(silu(w1(x)) * w3(x))
```

---

## Galaxy Parallelism (1×Galaxy = 4×8 = 32 chips)

| Mode | TP | EP | Notes |
|------|----|----|-------|
| Decode | 8 | 4 | EP across rows, TP within row |
| Prefill | 32 | 1 | Full TP (sequence parallel) |

This matches `gpt_oss`'s `MeshConfig` pattern exactly.

---

## Implementation Order

Following the Relay Race flow (Architecture → Reference → TTNN → Debug → Opt):

1. **RMSNorm** — copy from `gpt_oss/tt/rms_norm.py` directly
2. **Embedding** — copy from `gpt_oss`
3. **Partial RoPE** (rotary_dim=64) — adapt `deepseek_v3/tt/rope.py` (strip nope_head_dim, use rotary_dim=64)
4. **GQA Attention** — fork `gpt_oss/tt/attention/` and add:
   - QK-norm (per_layer, before reshape) using `gpt_oss/tt/rms_norm.py`
   - Partial RoPE integration
5. **MoE Router** — fork `gpt_oss/tt/topk.py`, add routing bias (`e_score_correction_bias`)
6. **MoE Experts** — copy `gpt_oss/tt/experts/` (SwiGLU identical)
7. **Decoder Layer** — assemble attention + MoE + norms
8. **Full Model** — assemble 62 decoder layers + embedding + final norm + lm_head
9. **MTP Modules** — skip entirely; training-only weights, not called in `MiniMaxM2ForCausalLM.forward()`

---

## Verification Command

Run to confirm weight structure once weights are available:
```python
from safetensors.torch import load_file
state_dict = load_file("model.safetensors")
prefixes = {}
for k in state_dict.keys():
    prefix = ".".join(k.split(".")[:3])
    prefixes[prefix] = prefixes.get(prefix, 0) + 1
for k, v in sorted(prefixes.items())[:30]:
    print(f"  {k}: {v} tensors")
```
