# OLMo-3.1-32B Architecture Analysis

## Model Family
**Decoder-only LLM** - Grouped Query Attention (GQA) with RoPE

## Target Model
- **HuggingFace**: [allenai/OLMo-3.1-32B-Think](https://huggingface.co/allenai/OLMo-3.1-32B-Think)
- **Architecture Type**: `olmo3` / `Olmo3ForCausalLM`
- **Precision**: bfloat16

---

## Architecture Comparison: OLMo-3.1-32B vs Qwen3-32B vs Llama3.1-70B

| Parameter | OLMo-3.1-32B | Qwen3-32B | Llama3.1-70B |
|-----------|--------------|-----------|--------------|
| hidden_size | 5120 | 5120 | 8192 |
| num_hidden_layers | 64 | 64 | 80 |
| num_attention_heads (Q) | **40** | 64 | 64 |
| num_key_value_heads (KV) | 8 | 8 | 8 |
| GQA ratio (Q:KV) | **5:1** | 8:1 | 8:1 |
| head_dim | 128 | 128 | 128 |
| intermediate_size | **27648** | 25600 | 28672 |
| vocab_size | **100278** | 151936 | 128256 |
| max_position_embeddings | 65536 | 40960 | 131072 |
| rope_theta | 500000 | 1000000 | 500000 |
| RoPE type | **YaRN** | Linear | Linear (scaled) |
| Sliding window | **4096 (hybrid)** | None | None |
| QK-norm | **Yes** | Yes | No |
| hidden_act | silu | silu | silu |
| rms_norm_eps | 1e-6 | 1e-6 | 1e-5 |

---

## Key Architectural Differences from Qwen3-32B

### 1. Fewer Query Heads (Major)
- **OLMo**: 40 Q heads / 8 KV heads = 5 Q heads per KV group
- **Qwen3**: 64 Q heads / 8 KV heads = 8 Q heads per KV group
- **Impact**: Different QKV tensor sizes, affects `n_local_heads` calculations

```python
# Per-device on Galaxy TG (8 KV heads across 8 devices)
# OLMo:  n_local_heads = 40 // 8 = 5
# Qwen3: n_local_heads = 64 // 8 = 8
```

### 2. Larger MLP Intermediate Dimension
- **OLMo**: 27648 (per-device: 27648 // 8 = 3456)
- **Qwen3**: 25600 (per-device: 25600 // 8 = 3200)
- **Impact**: Larger weight matrices for w1/w2/w3

### 3. YaRN RoPE (vs Linear RoPE)
OLMo uses YaRN (Yet another RoPE extensioN) for position embeddings:
```python
# OLMo rope_scaling config
{
    "rope_type": "yarn",
    "factor": 8.0,
    "original_max_position_embeddings": 8192,
    "attention_factor": 1.2079441541679836,
    "beta_fast": 32.0,
    "beta_slow": 1.0
}
```

### 4. Hybrid Sliding Window Attention
OLMo uses alternating sliding/full attention pattern:
- **Pattern**: 3 sliding + 1 full, repeated 16 times across 64 layers
- **Sliding window size**: 4096 tokens
- **Layer types**: `["sliding", "sliding", "sliding", "full"] * 16`

```python
# Layer type assignment
layer_types = []
for i in range(64):
    if (i + 1) % 4 == 0:  # Every 4th layer is full attention
        layer_types.append("full_attention")
    else:
        layer_types.append("sliding_attention")
# Result: 48 sliding layers + 16 full layers
```

### 5. QK-Normalization
- **OLMo**: HAS q_norm/k_norm weights (q_norm: [5120], k_norm: [1024])
- **Qwen3**: Has q_norm/k_norm (RMSNorm on Q and K)
- **Impact**: Same as Qwen3, can reuse QK-norm implementation

### 6. Smaller Vocabulary
- **OLMo**: 100,278 tokens
- **Qwen3**: 151,936 tokens
- **Impact**: Smaller embedding/lm_head matrices

---

## Similar Implementations (Reference Code)

| Component | Reference Implementation | Similarity |
|-----------|-------------------------|------------|
| Attention (GQA) | `models/demos/llama3_70b_galaxy/tt/llama_attention.py` | GQA with RoPE, same structure |
| MLP (SwiGLU) | `models/demos/llama3_70b_galaxy/tt/llama_mlp.py` | Identical SwiGLU: gate_proj * silu(up_proj) |
| RMSNorm | `models/common/rmsnorm.py` | Same RMSNorm implementation |
| RoPE | `models/demos/llama3_70b_galaxy/tt/llama_rope.py` | Need YaRN extension |
| Decoder Block | `models/demos/llama3_70b_galaxy/tt/llama_decoder.py` | Pre-norm transformer block |
| Full Model | `models/demos/llama3_70b_galaxy/tt/llama_model.py` | Same stack architecture |
| Load Weights | `models/demos/llama3_70b_galaxy/tt/load_checkpoints.py` | HF key mapping works |
| Model Config | `models/demos/llama3_70b_galaxy/tt/qwen_model_config.py` | Base template |

---

## Weight Mapping (HuggingFace to TTNN)

OLMo3 uses different weight names than standard Llama:

| HuggingFace Key | TTNN Key | Notes |
|-----------------|----------|-------|
| `model.embed_tokens.weight` | `tok_embeddings.weight` | |
| `model.layers.{i}.post_attention_layernorm.weight` | `layers.{i}.attention_norm.weight` | Different name! |
| `model.layers.{i}.self_attn.q_proj.weight` | `layers.{i}.attention.wq.weight` | |
| `model.layers.{i}.self_attn.k_proj.weight` | `layers.{i}.attention.wk.weight` | |
| `model.layers.{i}.self_attn.v_proj.weight` | `layers.{i}.attention.wv.weight` | |
| `model.layers.{i}.self_attn.o_proj.weight` | `layers.{i}.attention.wo.weight` | |
| `model.layers.{i}.self_attn.q_norm.weight` | `layers.{i}.attention.q_norm.weight` | QK-norm! [5120] |
| `model.layers.{i}.self_attn.k_norm.weight` | `layers.{i}.attention.k_norm.weight` | QK-norm! [1024] |
| `model.layers.{i}.post_feedforward_layernorm.weight` | `layers.{i}.ffn_norm.weight` | Different name! |
| `model.layers.{i}.mlp.gate_proj.weight` | `layers.{i}.feed_forward.w1.weight` | |
| `model.layers.{i}.mlp.up_proj.weight` | `layers.{i}.feed_forward.w3.weight` | |
| `model.layers.{i}.mlp.down_proj.weight` | `layers.{i}.feed_forward.w2.weight` | |
| `model.norm.weight` | `norm.weight` | |
| `lm_head.weight` | `output.weight` | |

**Note**: OLMo3 HAS `q_norm`/`k_norm` weights (same as Qwen3).

---

## Galaxy TG Tensor Sharding (32 devices, 8x4 mesh)

### Per-Device Dimensions
```python
# Configuration
num_devices = 32
n_kv_heads = 8
n_q_heads = 40
hidden_size = 5120
intermediate_size = 27648
head_dim = 128

# Galaxy TG sharding
num_devices_per_group = n_kv_heads  # 8 devices per row
num_device_groups = num_devices // n_kv_heads  # 4 device groups (columns)

# Per-device calculations
n_local_heads = n_q_heads // num_devices_per_group  # 40 // 8 = 5
n_local_kv_heads = n_kv_heads // num_devices_per_group  # 8 // 8 = 1
dim_per_device = hidden_size // 4  # 5120 // 4 = 1280 (column sharding)
intermediate_per_device = intermediate_size // 8  # 27648 // 8 = 3456 (row sharding)

# QKV sizes per device
qkv_size = (n_local_heads + 2 * n_local_kv_heads) * head_dim  # (5 + 2) * 128 = 896
```

### Tensor Shapes
| Tensor | Shape (per device) | Notes |
|--------|-------------------|-------|
| WQ | [1280, 640] | 5 heads * 128 dim |
| WK | [1280, 128] | 1 KV head * 128 dim |
| WV | [1280, 128] | 1 KV head * 128 dim |
| WO | [640, 1280] | Concat heads to hidden |
| W1 (gate) | [1280, 3456] | SwiGLU gate |
| W3 (up) | [1280, 3456] | SwiGLU up |
| W2 (down) | [3456, 1280] | SwiGLU down |

---

## Implementation Order

### Phase 0: Ring SDPA Sliding Window Extension
Extend C++ kernel to accept `sliding_window_size` parameter.

### Phase 1: Model Config + Weight Loading
1. Create `tt/olmo_model_config.py` (copy from `qwen_model_config.py`)
2. Set correct dimensions: `n_q_heads=40`, `intermediate_size=27648`, etc.
3. Add `layer_types` and `sliding_window=4096`
4. Set `qk_norm=False`, `is_qwen=False`
5. Verify weight loading (no q_norm/k_norm)

### Phase 2: YaRN RoPE Integration
1. Modify `llama_rope.py` to support YaRN scaling
2. Use `RopeScalingYarn` from `tt_transformers`
3. Map `attention_factor` to `mscale`

### Phase 3: Per-Layer Sliding Window Attention
1. Modify `llama_decoder.py` to pass `sliding_window_size` per layer
2. Modify `llama_attention.py` to pass to all 4 SDPA variants:
   - `scaled_dot_product_attention` (prefill)
   - `ring_distributed_scaled_dot_product_attention` (prefill, long seq)
   - `scaled_dot_product_attention_decode` (decode)
   - `paged_scaled_dot_product_attention_decode` (decode, paged)

### Phase 4: End-to-End Demo + Accuracy
1. Create `demo/text_olmo_demo.py`
2. Create `tests/test_olmo_accuracy.py`
3. Target: top-1 >= 80%, top-5 >= 98%

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Ring SDPA + sliding_window | Medium | Phase 0 kernel extension |
| YaRN attention_factor mapping | Low | Validate vs HF reference |
| Paged attention + sliding_window | Low | Already supported in kernel |
| Different Q head count | Low | Config change only |

---

## Effort Estimate

| Task | Effort |
|------|--------|
| Model config (TtOlmoModelArgs) | 0.5-1 day |
| Weight loading verification | 0.5 day |
| YaRN RoPE integration | 1-2 days |
| Per-layer sliding window | 1-2 days |
| Ring SDPA kernel extension | 1 day |
| Integration & testing | 2-3 days |
| **Total** | **6-10 days** |

---

## Files to Create/Modify

### New Files
- `models/demos/llama3_70b_galaxy/tt/olmo_model_config.py`
- `models/demos/llama3_70b_galaxy/demo/text_olmo_demo.py`
- `models/demos/llama3_70b_galaxy/tests/test_olmo_model.py`
- `models/demos/llama3_70b_galaxy/tests/test_olmo_accuracy.py`

### Modified Files
- `models/demos/llama3_70b_galaxy/tt/llama_rope.py` (YaRN support)
- `models/demos/llama3_70b_galaxy/tt/llama_decoder.py` (per-layer sliding_window)
- `models/demos/llama3_70b_galaxy/tt/llama_attention.py` (pass sliding_window to SDPA)
- `ttnn/cpp/.../ring_distributed_sdpa_*.cpp` (sliding_window parameter)
