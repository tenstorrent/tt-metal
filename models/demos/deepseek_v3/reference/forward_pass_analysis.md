# DeepSeekV3 Forward Pass Analysis

This document traces through a complete forward pass of the DeepSeekV3 model, comparing the reference implementation in `modeling_deepseek.py` with the extracted submodules.

## Model Configuration
Key parameters from `config.json`:
- `hidden_size`: 7168
- `num_attention_heads`: 128
- `num_hidden_layers`: 61
- `vocab_size`: 129280
- `intermediate_size`: 18432 (dense MLP)
- `moe_intermediate_size`: 2048 (expert MLP)
- `n_routed_experts`: 256
- `n_shared_experts`: 1
- `num_experts_per_tok`: 8
- `q_lora_rank`: 1536
- `kv_lora_rank`: 512
- `qk_nope_head_dim`: 128
- `qk_rope_head_dim`: 64
- `v_head_dim`: 128

## Forward Pass Trace

### 1. Token Embeddings
**Module**: `DeepseekV3Embeddings`
```
Input:  token_ids [batch=1, seq_len=100]
Output: embeddings [1, 100, 7168]
```

### 2. Transformer Layers (61 layers)
For each layer:

#### 2.1 Input Layer Norm
**Module**: `DeepseekV3RMSNorm`
```
Input:  hidden_states [1, 100, 7168]
Output: normed [1, 100, 7168]
```

#### 2.2 Multi-Head Attention
**Module**: `DeepseekV3Attention` with LoRA compression and MQA-style KV compression

The attention mechanism uses several optimizations to reduce memory and computation:

**Query Projection (LoRA Compression):**
```
# LoRA: 7168 -> 1536 -> 24576 (8x parameter reduction)
hidden_states [1, 100, 7168]
-> q_a_proj -> [1, 100, 1536]           # Down-project to low rank
-> q_a_layernorm -> [1, 100, 1536]      # Normalize compressed representation
-> q_b_proj -> [1, 100, 24576]          # Up-project to full size (128 heads * 192 dim)
-> reshape -> [1, 128, 100, 192]        # [batch, heads, seq, q_head_dim]

# Split into non-rotary and rotary parts
q_nope = q[..., :128]  # [1, 128, 100, 128] (qk_nope_head_dim)
q_pe = q[..., 128:]    # [1, 128, 100, 64]  (qk_rope_head_dim)
```

**Key-Value Projection (MQA-style Compression):**
```
# KV Compression: 7168 -> 576 -> 32768+64
hidden_states [1, 100, 7168]
-> kv_a_proj_with_mqa -> [1, 100, 576]  # 512 (kv_lora_rank) + 64 (qk_rope_head_dim)

# Split into compressed KV and rope key
compressed_kv = [1, 100, 512]           # For non-rotary parts of K,V
k_pe = [1, 100, 64]                     # For rotary part of K (MQA: single head)
k_pe = k_pe.unsqueeze(2) -> [1, 1, 100, 64]  # Broadcast to single head

# Expand compressed KV
compressed_kv -> kv_a_layernorm -> [1, 100, 512]
-> kv_b_proj -> [1, 100, 32768]         # 128 heads * (128 + 128) dims
-> reshape -> [1, 128, 100, 256]

# Split into non-rotary key and value
k_nope = kv[..., :128]     # [1, 128, 100, 128] (qk_nope_head_dim)
value_states = kv[..., 128:]  # [1, 128, 100, 128] (v_head_dim)
```

**RoPE Application:**
```
# Get rotary embeddings (YARN scaled)
cos, sin = rotary_emb(seq_len=kv_seq_len)  # [seq_len, 64]

# Apply rotation only to rope parts
q_pe_rot, k_pe_rot = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
# q_pe_rot: [1, 128, 100, 64]
# k_pe_rot: [1, 1, 100, 64] -> expanded to [1, 128, 100, 64] for all heads
```

**Attention Computation:**
```
# Concatenate non-rotary and rotary parts
query_states = concat([q_nope, q_pe_rot], dim=-1)  # [1, 128, 100, 192]
key_states = concat([k_nope, k_pe_rot.expand(-1, 128, -1, -1)], dim=-1)  # [1, 128, 100, 192]
value_states = value_states  # [1, 128, 100, 128]

# Standard scaled dot-product attention
attn_weights = query_states @ key_states.transpose(-1, -2)  # [1, 128, 100, 100]
attn_weights = attn_weights * softmax_scale  # (192^-0.5) * mscale_factor
attn_weights = attn_weights + attention_mask
attn_weights = softmax(attn_weights, dim=-1)
attn_output = attn_weights @ value_states  # [1, 128, 100, 128]

# Reshape and project
attn_output = attn_output.transpose(1, 2)  # [1, 100, 128, 128]
attn_output = attn_output.reshape(1, 100, 16384)  # 128 heads * 128 v_head_dim
attn_output = o_proj(attn_output)  # [1, 100, 7168]
```

**Memory Efficiency Analysis:**
- **Standard Attention**: 3 × (7168 × 128 × 192) ≈ 525M parameters
- **LoRA + Compression**: (7168×1536 + 1536×24576) + (7168×576 + 512×32768) ≈ 67M parameters
- **~8x Parameter Reduction** in attention projections

**Key Features:**
- **LoRA Query**: Rank-1536 bottleneck for query projection
- **MQA RoPE**: Single-headed key for rotary part, expanded to all heads
- **Separate RoPE/NoRoPE**: 128d non-rotary + 64d rotary dimensions
- **YARN Scaling**: Enhanced RoPE with mscale factor for length extrapolation

#### 2.3 Post-Attention Layer Norm
**Module**: `DeepseekV3RMSNorm`
```
Input:  hidden_states [1, 100, 7168]
Output: normed [1, 100, 7168]
```

#### 2.4 Feed-Forward Network
Layers 0-2: Dense MLP (`DeepseekV3MLP`)
```
Input:  [1, 100, 7168]
-> gate_proj & up_proj -> [1, 100, 18432] each
-> silu(gate) * up -> [1, 100, 18432]
-> down_proj -> [1, 100, 7168]
```

Layers 3-60 (every layer): MoE (`DeepseekV3MoE`)
```
Input: [1, 100, 7168]

Gate computation:
-> MoEGate -> topk_idx [100, 8], topk_weight [100, 8]

Expert routing:
- Route to 8 selected experts out of 256
- Each expert is DeepseekV3MLP with intermediate_size=2048
- Weighted sum of expert outputs -> [1, 100, 7168]

Shared expert:
- Always active DeepseekV3MLP with intermediate_size=2048
- Output added to routed expert output
```

### 3. Final Layer Norm
**Module**: `DeepseekV3RMSNorm`
```
Input:  hidden_states [1, 100, 7168]
Output: normed [1, 100, 7168]
```

### 4. Language Model Head
**Module**: `DeepseekV3LMHead`
```
Input:  hidden_states [1, 100, 7168]
Output: logits [1, 100, 129280] (float32)
```

## Decode Pass (with KV Cache)

For autoregressive generation with cached keys/values:

### Input
```
token_ids [1, 1] (single new token)
position_ids [1, 1] (current position)
past_key_values: cached K,V from previous steps
```

### Attention Changes
```
New Q: [1, 128, 1, 192]
New K,V: [1, 128, 1, 192], [1, 128, 1, 128]
Cached K,V: [1, 128, prev_len, 192], [1, 128, prev_len, 128]
Combined K,V: [1, 128, prev_len+1, 192], [1, 128, prev_len+1, 128]

Attention: Q @ K.T -> [1, 128, 1, prev_len+1]
Output: [1, 1, 7168]
```

### Final Output
```
logits [1, 1, 129280]
```

## Verification Summary

All submodules correctly implement their corresponding parts in the reference:

1. ✓ **DeepseekV3RMSNorm**: Matches RMSNorm implementation
2. ✓ **DeepseekV3Embeddings**: Standard embedding lookup
3. ✓ **DeepseekV3LMHead**: Linear projection with float32 cast
4. ✓ **DeepseekV3MLP**: SwiGLU MLP matching the reference
5. ✓ **DeepseekV3YarnRotaryEmbedding**: YARN RoPE with correct scaling
6. ✓ **apply_rotary_pos_emb**: Correct rotation application
7. ✓ **MoEGate**: Grouped top-k selection with correction bias
8. ✓ **DeepseekV3MoE**: Expert routing with shared expert addition
9. ✓ **DeepseekV3Attention**: LoRA-compressed Q projection + MQA-style KV compression
10. ✓ **DeepseekV3FlashAttention2**: Memory-efficient attention with O(N) complexity

**Attention Module Innovations:**
- **LoRA Compression**: 8x parameter reduction in attention projections
- **MQA RoPE Keys**: Single-headed rotary keys expanded to all attention heads
- **Separated Dimensions**: 128d non-rotary + 64d rotary for better position encoding
- **YARN Scaling**: Enhanced extrapolation beyond training context length
- **Flash Attention**: Tiled computation for memory efficiency

The main complexity not captured in submodules:
- **KV Cache Management**: Incremental key-value caching for autoregressive generation
- **Flash Attention Optimizations**: Variable-length sequence handling and kernel fusion
- **Attention Mask Processing**: Causal masking and padding token handling
