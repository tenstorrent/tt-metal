# Llama3-70B Galaxy: Prefill Operations Detailed Report

## Overview

Prefill operations process the entire input prompt in one forward pass, unlike decode which processes one token at a time. This report details the prefill-specific optimizations and operations.

## Key Differences from Decode

1. **Variable Sequence Length**: Prefill handles sequences from 128 to 128k tokens
2. **Batch Processing**: Can process multiple users simultaneously (up to 32)
3. **Memory Layout**: Uses DRAM memory configs (not L1-optimized)
4. **Chunking**: Long sequences are chunked for memory efficiency
5. **Minimal MatMul**: For sequences >= 4096, uses minimal matmul for efficiency

---

## Attention Prefill Operations

### QKV Projection (Prefill)

**Location**: `llama_attention.py::forward_prefill()`

**Operation**: `ttnn.linear(x_11SH, self.wqkv_interleaved, ...)`

**Step-by-Step**:

```
Step 1: Input Preparation
├─ Input: x_11SH [1, 1, seq_len, 1280]
│   └─ Sequence length can vary (128 to 128k)
│   └─ Hidden dimension sharded
│
Step 2: Sequence Length Handling
├─ If seq_len > 2048:
│   ├─ Reshape: [1, 1, seq_len, 1280] → [1, seq_len//2048, 2048, 1280]
│   └─ Process in chunks of 2048
│
└─ If batch_size > 1:
    └─ Reshape for batch processing
│
Step 3: MatMul Execution
├─ Weight: wqkv_interleaved
│   └─ Interleaved layout (not sharded)
│   └─ Memory: DRAM_MEMORY_CONFIG
├─ Program Config: XQKV_PREFILL_PROGCFG(seq_len)
│   └─ Optimized for specific sequence length
├─ Compute Kernel: HIFI2
│   └─ High-fidelity compute
├─ Dtype: ccl_dtype (if TG) else bfloat16
│
Step 4: Output
└─ xqkv: [1, 1, seq_len, 12288]
    └─ QKV concatenated
    └─ Memory: DRAM_MEMORY_CONFIG
```

**Key Details**:
- **Interleaved Weights**: Uses interleaved layout for prefill (not sharded)
- **Chunking**: Long sequences reshaped into chunks
- **DRAM Memory**: Uses DRAM instead of L1 for large sequences

**Code Reference**:
```python
# Reshape for long sequences
if seq_len > 2048:
    x_11SH = ttnn.reshape(x_11SH, [1, seq_len // 2048, 2048, -1])

# QKV projection
xqkv = ttnn.linear(
    x_11SH,
    self.wqkv_interleaved,
    dtype=self.ccl_dtype if self.TG else ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    compute_kernel_config=self.compute_kernel_config_hifi2,
    program_config=self.model_config["XQKV_PREFILL_PROGCFG"](seq_len),
)
```

---

### All-Reduce for QKV (Prefill)

**Location**: `llama_attention.py::forward_prefill()`

**Operation**: `self.tt_ccl.line_all_reduce(xqkv, ...)`

**Purpose**: Reduce QKV results across devices for ring topology

**Step-by-Step**:

```
Step 1: Input
├─ xqkv: [1, 1, seq_len, 12288]
│   └─ Partial QKV per device
│
Step 2: All-Reduce Operation
├─ Cluster Axis: 1 (column dimension)
├─ Num Links: 3
├─ Buffer Key: "QKV"
│   └─ Uses persistent buffer
├─ Operation:
│   ├─ Reduces partial results across devices
│   ├─ Sums all partial results
│   └─ Uses ring topology
│
Step 3: Output
└─ xqkv_fused: [1, 1, seq_len, 12288]
    └─ Full QKV (reduced)
    └─ Memory: DRAM_MEMORY_CONFIG
```

**Key Details**:
- **Ring Topology**: Uses ring for efficient reduction
- **Persistent Buffer**: Reuses pre-allocated buffer
- **Sum Operation**: Reduces by summing (not concatenating)

**Code Reference**:
```python
xqkv_fused = self.tt_ccl.line_all_reduce(
    xqkv,
    cluster_axis=1,
    num_links=3,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    buffer_key="QKV",
)
```

---

### Create QKV Heads (Prefill)

**Location**: `llama_attention.py::forward_prefill()`

**Operation**: `ttnn.experimental.nlp_create_qkv_heads()`

**Purpose**: Split QKV into separate heads

**Step-by-Step**:

```
Step 1: Input
├─ xqkv_fused: [1, 1, seq_len, 12288]
│   └─ QKV concatenated
│
Step 2: Split Operation
├─ Operation: nlp_create_qkv_heads
│   ├─ num_heads: n_local_heads (8 per device)
│   ├─ num_kv_heads: n_local_kv_heads (1 per device)
│   └─ transpose_k_heads: False
│
Step 3: Output
├─ q_heads_1QSD_pre_rot: [1, 8, seq_len, 128]
│   └─ Query heads (before RoPE)
├─ k_heads_1KSD_pre_rot: [1, 1, seq_len, 128]
│   └─ Key heads (before RoPE)
└─ v_heads_1VSD: [1, 1, seq_len, 128]
    └─ Value heads
```

**Key Details**:
- **No Reduce-Scatter**: Prefill doesn't use reduce-scatter (different from decode)
- **Full Sequence**: Processes entire sequence at once
- **Memory**: DRAM_MEMORY_CONFIG

**Code Reference**:
```python
(
    q_heads_1QSD_pre_rot,
    k_heads_1KSD_pre_rot,
    v_heads_1VSD,
) = ttnn.experimental.nlp_create_qkv_heads(
    xqkv_fused,
    num_heads=self.n_local_heads,
    num_kv_heads=self.n_local_kv_heads,
    transpose_k_heads=False,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

---

### RoPE Application (Prefill)

**Location**: `llama_attention.py::forward_prefill()`

**Operation**: `ttnn.experimental.rotary_embedding_llama()`

**Purpose**: Apply rotary position embeddings to Q and K

**Step-by-Step**:

```
Step 1: Input Preparation
├─ q_heads_1QSD_pre_rot: [1, 8, seq_len, 128]
├─ k_heads_1KSD_pre_rot: [1, 1, seq_len, 128]
├─ rot_mats[0]: cos_matrix [1, 1, seq_len, 128]
├─ rot_mats[1]: sin_matrix [1, 1, seq_len, 128]
└─ transformation_mats["prefill"]: Prefill transformation matrices
│
Step 2: Type Conversion
├─ If dtype != bfloat16:
│   ├─ Convert to bfloat16 (RoPE requires bfloat16)
│   └─ Deallocate original
│
Step 3: QK Norm (if enabled)
├─ If qk_norm:
│   ├─ Apply q_norm to q_heads
│   └─ Apply k_norm to k_heads
│
Step 4: RoPE Application
├─ Operation: rotary_embedding_llama
│   ├─ is_decode_mode: False
│   ├─ Applies rotation to entire sequence
│   └─ Uses prefill transformation matrices
│
Step 5: Output
├─ q_heads_1QSD: [1, 8, seq_len, 128]
│   └─ Query heads with RoPE
└─ k_heads_1KSD: [1, 1, seq_len, 128]
    └─ Key heads with RoPE
```

**Key Details**:
- **Full Sequence**: Applies RoPE to entire sequence
- **Type Conversion**: Requires bfloat16 for RoPE
- **QK Norm**: Optional normalization for Qwen models

**Code Reference**:
```python
# Type conversion
if q_heads_1QSD_pre_rot.dtype != ttnn.bfloat16:
    q_heads_1QSD_pre_rot = ttnn.typecast(q_heads_1QSD_pre_rot, dtype=ttnn.bfloat16)

# QK norm (if enabled)
if self.qk_norm:
    q_heads_1QSD_pre_rot = self.q_norm(q_heads_1QSD_pre_rot, mode="prefill")

# RoPE application
q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
    q_heads_1QSD_pre_rot,
    rot_mats[0],
    rot_mats[1],
    self.transformation_mats["prefill"],
    is_decode_mode=False,
)
```

---

### KV Cache Fill (Prefill)

**Location**: `llama_attention.py::forward_prefill()`

**Operation**: KV cache update for entire sequence

**Step-by-Step**:

```
Step 1: Input Preparation
├─ k_heads_1KSD: [1, 1, seq_len, 128]
├─ v_heads_1VSD: [1, 1, seq_len, 128]
├─ kv_cache: [batch, n_kv_heads, max_seq_len, head_dim]
│   └─ Or self.layer_past if no external cache
│
Step 2: Type Conversion
├─ Convert to bfloat8_b for cache storage
│   ├─ k_heads_1KSD_8b = typecast(k_heads_1KSD, bfloat8_b)
│   └─ v_heads_1VSD_8b = typecast(v_heads_1VSD, bfloat8_b)
│
Step 3: Batch Handling
├─ If batch_size > 1:
│   ├─ Reshape: [1, 1, seq_len, 128] → [1, 1, seq_len, 128]
│   └─ Handle batch dimension
│
Step 4: Cache Update
├─ If paged attention:
│   ├─ Use page_table to map logical to physical blocks
│   └─ Fill cache blocks
│
└─ Else (standard cache):
    ├─ Fill entire sequence into cache
    └─ cache[:, :, :seq_len, :] = k_heads_1KSD_8b, v_heads_1VSD_8b
│
Step 5: Output
└─ KV cache updated with entire sequence
```

**Key Details**:
- **Full Sequence**: Fills entire sequence at once
- **Type Conversion**: Stores in bfloat8_b for memory efficiency
- **Paged Attention**: Supports paged attention for long sequences

**Code Reference**:
```python
# Type conversion
k_heads_1KSD_8b = ttnn.typecast(k_heads_1KSD, dtype=ttnn.bfloat8_b)
v_heads_1VSD_8b = ttnn.typecast(v_heads_1VSD, dtype=ttnn.bfloat8_b)

# Cache fill
if page_table:
    # Paged attention fill
    ttnn.experimental.paged_fill_cache(...)
else:
    # Standard cache fill
    ttnn.fill_cache(keys, k_heads_1KSD_8b, user_id)
    ttnn.fill_cache(values, v_heads_1VSD_8b, user_id)
```

---

### Scaled Dot-Product Attention (Prefill)

**Location**: `llama_attention.py::forward_prefill()`

**Operation**: `ttnn.transformer.scaled_dot_product_attention()`

**Purpose**: Compute attention over entire sequence

**Step-by-Step**:

```
Step 1: Input Preparation
├─ q_heads_1QSD: [1, 8, seq_len, 128]
├─ k_heads_1KSD: [1, 1, seq_len, 128]
├─ v_heads_1VSD: [1, 1, seq_len, 128]
│
Step 2: Attention Computation
├─ Q @ K^T: [1, 8, seq_len, 128] @ [1, 1, 128, seq_len]
│   └─ Result: [1, 8, seq_len, seq_len]
│   └─ Attention scores
├─ Scale: scores / sqrt(head_dim)
├─ Causal Mask: Lower triangular mask
│   └─ Prevents attending to future tokens
├─ Softmax: exp(scores) / sum(exp(scores))
└─ Attention @ V: [1, 8, seq_len, seq_len] @ [1, 1, seq_len, 128]
    └─ Result: [1, 8, seq_len, 128]
│
Step 3: Program Configuration
├─ Program Config: SDPA_PREFILL_PROGCFG(seq_len)
│   └─ Optimized for prefill sequence length
│
Step 4: Output
└─ attn_output: [1, 8, seq_len, 128]
    └─ Attention output per head
```

**Key Details**:
- **Full Sequence**: Computes attention over entire sequence
- **Causal Mask**: Lower triangular mask for causal attention
- **Memory Efficient**: Uses flash attention for long sequences

**Mathematical Formula**:
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + causal_mask) @ V
```

Where:
- Q: [batch, n_heads, seq_len, head_dim]
- K: [batch, n_kv_heads, seq_len, head_dim]
- V: [batch, n_kv_heads, seq_len, head_dim]
- causal_mask: Lower triangular matrix

**Code Reference**:
```python
attn_output = ttnn.transformer.scaled_dot_product_attention(
    q_heads_1QSD,
    k_heads_1KSD,
    v_heads_1VSD,
    is_causal=True,
    scale=self.scale,
    program_config=self.model_config["SDPA_PREFILL_PROGCFG"](seq_len),
    compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

---

## MLP Prefill Operations

### W1/W3 Projection (Prefill)

**Location**: `llama_mlp.py::forward_prefill()`

**Operation**: `ttnn.linear()` or `ttnn.experimental.minimal_matmul()`

**Purpose**: Compute gate and up projections

**Step-by-Step**:

```
Step 1: Input Preparation
├─ Input: x [1, 1, seq_len, 1280]
│   └─ Sequence length varies
│
Step 2: Sequence Length Handling
├─ If 1024 <= seq_len < 4096:
│   └─ Reshape: [1, 1, seq_len, 1280] → [1, seq_len//1024, 1024, 1280]
│
Step 3: MatMul Selection
├─ If seq_len < 4096:
│   ├─ Use: ttnn.linear()
│   ├─ Program Config: PREFILL_MLP_W1_W3_PRG_CONFIG(seq_len)
│   ├─ Compute Kernel: LOFI (if bfp4) else HIFI2_FP16
│   └─ Memory: DRAM_MEMORY_CONFIG
│
└─ Else (seq_len >= 4096):
    ├─ Use: ttnn.experimental.minimal_matmul()
    ├─ Config: PREFILL_FF1_FF3_MINIMAL_MATMUL_CONFIG(seq_len)
    ├─ Compute Kernel: LOFI
    └─ Memory: DRAM_MEMORY_CONFIG
│
Step 4: W1 Projection
├─ w1_out = x @ w1_interleaved (or w1)
└─ Output: [1, 1, seq_len, 14336]
│
Step 5: Reduce-Scatter
├─ Operation: line_reduce_scatter
├─ Cluster Axis: 1
├─ Num Links: 3
├─ Buffer Key: "FF1"
└─ Output: w1_out_reduced [1, 1, seq_len, 14336/8]
│
Step 6: W3 Projection
├─ w3_out = x @ w3_interleaved (or w3)
└─ Output: [1, 1, seq_len, 14336]
│
Step 7: Reduce-Scatter
├─ Operation: line_reduce_scatter
├─ Cluster Axis: 1
├─ Num Links: 3
├─ Buffer Key: "FF3"
└─ Output: w3_out_reduced [1, 1, seq_len, 14336/8]
```

**Key Details**:
- **Minimal MatMul**: For long sequences, uses minimal matmul for efficiency
- **Separate Operations**: W1 and W3 computed separately (not fused like decode)
- **Reduce-Scatter**: Reduces results across devices

**Why Minimal MatMul**:
- More efficient for very long sequences
- Reduces memory bandwidth
- Better utilization of compute resources

**Code Reference**:
```python
# W1 projection
if seq_len < 4096:
    w1_out = ttnn.linear(
        x,
        self.w1_interleaved if use_w1_w3_interleaved else self.w1,
        compute_kernel_config=(
            self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2_fp16
        ),
        dtype=ttnn.bfloat8_b,
        program_config=short_lens_pc_1_3,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
else:
    w1_out = ttnn.experimental.minimal_matmul(
        input_tensor=x,
        weight_tensor=self.w1_interleaved if use_w1_w3_interleaved else self.w1,
        config=minimal_pc_1_3,
        compute_kernel_config=self.args.compute_kernel_config_lofi,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

# Reduce-scatter
w1_out_reduced = self.tt_ccl.line_reduce_scatter(
    w1_out,
    cluster_axis=1,
    num_links=3,
    memory_config=w1_out.memory_config(),
    buffer_key="FF1",
    dim=3
)
```

---

### Element-wise Multiply with SiLU (Prefill)

**Location**: `llama_mlp.py::forward_prefill()`

**Operation**: `ttnn.mul(w1_out_reduced, w3_out_reduced, input_tensor_a_activations=[SILU])`

**Step-by-Step**:

```
Step 1: Input Preparation
├─ w1_out_reduced: [1, 1, seq_len, 1792]
├─ w3_out_reduced: [1, 1, seq_len, 1792]
│
Step 2: SiLU Activation
├─ Applied to w1_out_reduced
├─ SiLU(x) = x * sigmoid(x)
└─ Computed element-wise
│
Step 3: Element-wise Multiply
├─ ff1ff3 = SiLU(w1_out_reduced) * w3_out_reduced
└─ Output: [1, 1, seq_len, 1792]
│
Step 4: All-Gather
├─ Operation: line_all_gather
├─ Cluster Axis: 1
├─ Num Links: 3
├─ Buffer Key: "FF3"
└─ Output: w2_in [1, 1, seq_len, 14336]
```

**Code Reference**:
```python
# Multiply with SiLU
w2_in = ttnn.mul(
    w1_out_reduced,
    w3_out_reduced,
    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
    dtype=ttnn.bfloat8_b,
    memory_config=w1_out.memory_config(),
)

# All-gather
w2_in_gathered = self.tt_ccl.line_all_gather(
    w2_in,
    cluster_axis=1,
    num_links=3,
    memory_config=w3_out.memory_config(),
    buffer_key="FF3",
    dim=3
)
```

---

### W2 Projection (Prefill)

**Location**: `llama_mlp.py::forward_prefill()`

**Operation**: `ttnn.linear()` or `ttnn.experimental.minimal_matmul()`

**Step-by-Step**:

```
Step 1: Input Preparation
├─ w2_in_gathered: [1, 1, seq_len, 14336]
│   └─ Full hidden dimension
│
Step 2: MatMul Selection
├─ If seq_len < 4096:
│   ├─ Use: ttnn.linear()
│   ├─ Program Config: PREFILL_MLP_W2_PRG_CONFIG(seq_len)
│   ├─ Compute Kernel: HIFI2_FP16
│   └─ Memory: DRAM_MEMORY_CONFIG
│
└─ Else (seq_len >= 4096):
    ├─ Use: ttnn.experimental.minimal_matmul()
    ├─ Config: PREFILL_FF2_MINIMAL_MATMUL_CONFIG(seq_len)
    ├─ Compute Kernel: HIFI2_FP16
    └─ Memory: DRAM_MEMORY_CONFIG
│
Step 3: MatMul Execution
├─ w2_out = w2_in_gathered @ w2_interleaved
└─ Output: [1, 1, seq_len, 1280]
│
Step 4: All-Reduce
├─ Operation: line_all_reduce
├─ Cluster Axis: 0
├─ Num Links: 3
├─ Buffer Key: "FF2"
└─ Output: w2_out_reduced [1, 1, seq_len, 1280]
│
Step 5: Reshape (if needed)
├─ If 1024 <= seq_len < 4096:
│   └─ Reshape back to original shape
```

**Code Reference**:
```python
# W2 projection
if seq_len < 4096:
    w2_out = ttnn.linear(
        w2_in_gathered,
        self.w2_interleaved,
        compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
        dtype=ttnn.bfloat8_b,
        program_config=short_lens_pc_2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
else:
    w2_out = ttnn.experimental.minimal_matmul(
        input_tensor=w2_in_gathered,
        weight_tensor=self.w2_interleaved,
        config=minimal_pc_2,
        compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

# All-reduce
w2_out_reduced = self.tt_ccl.line_all_reduce(
    w2_out,
    cluster_axis=0,
    num_links=3,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    buffer_key="FF2"
)
```

---

## Summary

Prefill operations are optimized for:

1. **Variable Sequence Lengths**: Handles 128 to 128k tokens
2. **Memory Efficiency**: Uses DRAM and minimal matmul for long sequences
3. **Batch Processing**: Supports multiple users simultaneously
4. **Chunking**: Long sequences processed in chunks
5. **Ring Topology**: Uses ring-based CCL operations

Key optimizations:
- **Minimal MatMul**: For sequences >= 4096
- **Interleaved Weights**: For prefill (not sharded)
- **DRAM Memory**: For large sequences
- **Separate W1/W3**: Not fused (unlike decode)
