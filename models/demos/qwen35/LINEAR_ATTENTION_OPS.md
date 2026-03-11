# Linear Attention Operations Analysis

## 1. Causal Conv1D

### What It Does
Applies 1D temporal convolution where each output position only depends on current and past inputs (causal masking).

### Requirements
- **Input**: `[batch, channels, seq_len]` or `[batch, seq_len, channels]` (depends on layout)
- **Kernel size**: 4 (from config)
- **Groups**: `linear_value_dim` (4096) - depthwise separable convolution
- **Padding**: `kernel_size - 1 = 3` (left-padding only for causality)
- **Stride**: 1
- **Output**: Same sequence length as input

### TTNN Implementation
✅ **YES - `ttnn.conv1d` supports this!**

```python
# Based on Mamba implementation (models/demos/wormhole/mamba/tt/mamba_conv.py)
conv_config = ttnn.Conv1dConfig(
    weights_dtype=ttnn.bfloat16,
    shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    deallocate_activation=True,
)

output, out_length = ttnn.conv1d(
    input_tensor=v,  # [B, L, C] in ROW_MAJOR
    weight_tensor=conv_weights,  # [C, 1, K] where K=4
    in_channels=linear_value_dim,  # 4096
    out_channels=linear_value_dim,  # 4096
    device=device,
    kernel_size=4,
    stride=1,
    padding=3,  # Left padding for causality
    batch_size=batch_size,
    input_length=seq_len,
    groups=linear_value_dim,  # Depthwise (each channel separate)
    conv_config=conv_config,
    dtype=ttnn.bfloat16,
    return_output_dim=True,
)
```

### State Management for Decode Mode
For single-token decode, we need to maintain a sliding window state:
- **State shape**: `[batch, linear_value_dim, kernel_size-1]` = `[B, 4096, 3]`
- **Update**: Shift state left, append new input, apply conv

---

## 2. Delta Rule (Gated DeltaNet)

### What It Does
Computes linear attention using a recurrent state update with gating mechanism:

```
For each timestep t:
1. Compute beta_t = sigmoid(cumsum(K_t))  # Gating coefficient
2. Update state: h_t = h_{t-1} * (1 - beta_t) + beta_t * outer(K_t, V_t)
3. Compute output: O_t = h_t @ Q_t
```

### Mathematical Formula
```
O[t] = sum_{i=1}^{t} (beta[t] - beta[i-1]) * V[i] * K[i]^T * Q[t]
```

Where:
- `beta[t] = sigmoid(sum_{i=1}^{t} K[i])` - cumulative gating
- This is the **delta** (difference) in beta values that gives the method its name

### Shapes
- **Q**: `[batch, seq_len, num_q_heads, head_dim]` = `[B, L, 16, 256]`
- **K**: `[batch, seq_len, linear_num_key_heads, linear_key_head_dim]` = `[B, L, 16, 128]`
- **V**: `[batch, seq_len, linear_num_value_heads, linear_value_head_dim]` = `[B, L, 32, 128]`
- **Recurrent State**: `[batch, linear_num_key_heads, linear_key_head_dim, linear_value_head_dim]` = `[B, 16, 128, 128]`
- **Output**: `[batch, seq_len, num_q_heads, head_dim]` = `[B, L, 16, 256]`

### Required TTNN Operations

| Operation | TTNN Support | Notes |
|-----------|--------------|-------|
| Matrix multiplication | ✅ `ttnn.matmul` | For Q@K^T, state@Q |
| Outer product | ✅ `ttnn.matmul` | K@V^T via matmul with transpose |
| Sigmoid | ✅ `ttnn.sigmoid` | For beta gating |
| Cumulative sum | ✅ `ttnn.cumsum` | For cumulative K |
| Element-wise ops | ✅ `ttnn.mul`, `ttnn.add` | For state updates |
| Reshape/transpose | ✅ `ttnn.reshape`, `ttnn.permute` | For dimension handling |

### Two Modes

#### Prefill Mode (Chunked Processing)
- Process 64-256 tokens per chunk
- Compute beta differences: `delta_beta[i] = beta[i] - beta[i-1]`
- Use parallel scan or chunked iteration
- Reference: `torch_chunk_gated_delta_rule` in transformers

**Pseudo-code:**
```python
# For chunk of length L
K_cumsum = ttnn.cumsum(K, dim=1)  # [B, L, H, D_k]
beta = ttnn.sigmoid(K_cumsum)     # [B, L, H, D_k]

# Compute deltas
beta_prev = ttnn.pad(beta[:, :-1], ...)  # Shift by 1
delta_beta = beta - beta_prev  # [B, L, H, D_k]

# For each position, accumulate weighted KV contributions
# This requires a custom kernel or chunked iteration
```

#### Decode Mode (Recurrent)
- Single token update (seq_len = 1)
- Maintain recurrent state from previous step
- Fast incremental computation

**Pseudo-code:**
```python
# Single token: K, V are [B, 1, H, D]
# Compute beta increment
K_new_sum = prev_K_sum + K  # [B, 1, H, D_k]
beta = ttnn.sigmoid(K_new_sum)  # [B, 1, H, D_k]
delta_beta = beta - prev_beta  # [B, 1, H, D_k]

# Update state
# state: [B, H, D_k, D_v]
KV_outer = ttnn.matmul(K^T, V)  # [B, H, D_k, D_v]
state = state + delta_beta * KV_outer  # Element-wise

# Compute output
output = ttnn.matmul(state, Q)  # [B, 1, H, D_v]
```

---

## 3. Query Gating

### What It Does
Gates the attention output based on query projection:
```
gate = sigmoid(linear(Q))
output = gate * attention_output
```

### Shapes
- **Q**: `[batch, seq_len, num_q_heads, head_dim]` = `[B, L, 16, 256]`
- **Gate weights**: `[dim, num_q_heads]` = `[4096, 16]`
- **Gate bias**: `[num_q_heads]` = `[16]`
- **Gate**: `[batch, seq_len, num_q_heads, 1]` = `[B, L, 16, 1]`

### TTNN Implementation
✅ **Simple - all ops available**

```python
# Q_flat: [batch, seq_len, num_q_heads * head_dim] = [B, L, 4096]
Q_flat = ttnn.reshape(Q, (batch, seq_len, -1))

# Project to gate logits
gate_logits = ttnn.linear(Q_flat, q_gate_weight) + q_gate_bias
# gate_logits: [B, L, num_q_heads]

# Apply sigmoid
gate = ttnn.sigmoid(gate_logits)  # [B, L, 16]

# Expand to match attention output
gate = ttnn.reshape(gate, (batch, seq_len, num_q_heads, 1))

# Gate the output
output = ttnn.mul(gate, attn_output)  # Element-wise
```

---

## 4. Output Gating (Optional)

### What It Does
Optional additional gating based on input:
```
gate = sigmoid(linear(x))
output = gate * output
```

### Shapes
- **x**: `[batch, seq_len, dim]` = `[B, L, 4096]`
- **Gate weights**: `[dim, num_q_heads * head_dim]` = `[4096, 4096]`
- **Output**: `[batch, seq_len, num_q_heads, head_dim]` = `[B, L, 16, 256]`

### TTNN Implementation
✅ **Simple - same as query gating**

```python
# Project input to gate
gate_logits = ttnn.linear(x, output_gate_weight)  # [B, L, 4096]
gate = ttnn.sigmoid(gate_logits)

# Reshape and apply
gate = ttnn.reshape(gate, (batch, seq_len, num_q_heads, head_dim))
output = ttnn.mul(gate, output)
```

---

## Implementation Priority

### High Priority (Core Functionality)
1. ✅ **Causal Conv1D** - Straightforward using `ttnn.conv1d`
2. ⚠️ **Delta Rule (Decode Mode)** - Needs custom implementation but uses standard ops
3. ✅ **Query Gating** - Simple, all ops available

### Medium Priority
4. ⚠️ **Delta Rule (Prefill Mode)** - More complex, may need chunked iteration or custom kernel

### Low Priority
5. ✅ **Output Gating** - Simple, same pattern as query gating

---

## Recommended Implementation Strategy

### Phase 1: Basic Operations (1-2 days)
1. Implement causal_conv1d using ttnn.conv1d
2. Implement query_gating and output_gating (simple ops)
3. Test these components independently

### Phase 2: Delta Rule - Decode Mode (2-3 days)
1. Start with decode mode (simpler, single token)
2. Implement recurrent state update
3. Test against PyTorch reference for single token

### Phase 3: Delta Rule - Prefill Mode (3-5 days)
1. Implement chunked processing
2. May need to iterate in smaller chunks if custom kernel not available
3. Optimize with parallel operations where possible

### Phase 4: Optimization (2-3 days)
1. Profile and optimize memory layout
2. Fuse operations where beneficial
3. Consider custom kernel for delta rule if needed

---

## References

- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) - Triton implementations
- [Gated DeltaNet](https://github.com/NVlabs/GatedDeltaNet) - Official implementation
- [DeltaNet Explained](https://sustcsonglin.github.io/blog/2024/deltanet-1/) - Technical blog
- [Qwen3.5 Paper](https://arxiv.org/abs/2412.06464) - Model architecture details

---

## Decision: Can We Implement This?

### ✅ YES - All core operations are available in TTNN!

**Causal Conv1D**: Direct support via `ttnn.conv1d`
**Delta Rule**: All required ops available (`matmul`, `sigmoid`, `cumsum`, element-wise)
**Gating**: Simple composition of existing ops

**Challenges**:
- Delta rule prefill mode may need iterative implementation (can't do full parallel scan easily)
- May need custom kernel for optimal performance later
- State management for decode mode needs careful handling

**Recommendation**: Start implementation now, beginning with decode mode and simple operations.
