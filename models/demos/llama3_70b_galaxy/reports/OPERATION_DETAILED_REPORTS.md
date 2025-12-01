# Llama3-70B Galaxy: Detailed Operation Reports

## Table of Contents

1. [Overview](#overview)
2. [Attention Operations](#attention-operations)
   - [QKV Projection (Decode)](#qkv-projection-decode)
   - [Create QKV Heads with Reduce-Scatter](#create-qkv-heads-with-reduce-scatter)
   - [RoPE Application](#rope-application)
   - [KV Cache Update](#kv-cache-update)
   - [Scaled Dot-Product Attention (SDPA)](#scaled-dot-product-attention-sdpa)
   - [Output Projection with All-Gather](#output-projection-with-all-gather)
3. [MLP Operations](#mlp-operations)
   - [Double MatMul with Reduce-Scatter](#double-matmul-with-reduce-scatter)
   - [Element-wise Multiply with SiLU](#element-wise-multiply-with-silu)
   - [All-Gather for W2 Input](#all-gather-for-w2-input)
   - [W2 MatMul with All-Reduce](#w2-matmul-with-all-reduce)
4. [Distributed Norm Operations](#distributed-norm-operations)
5. [CCL Operations](#ccl-operations)
   - [Ring Topology Operations](#ring-topology-operations)
   - [Line Topology Operations](#line-topology-operations)
6. [Prefetcher System](#prefetcher-system)

---

## Overview

The Llama3-70B Galaxy implementation is optimized for **Galaxy hardware** (32-chip Wormhole system) with several key optimizations:

- **Ring Topology**: Uses ring-based collective communication for optimal bandwidth
- **Prefetcher**: Dedicated sub-device for weight prefetching
- **Fused Operations**: Combines multiple operations for efficiency
- **Memory Management**: Specialized memory configs for prefill vs decode
- **Batch Processing**: Optimized for batch size 32

### Key Differences from Generic TT-Transformers

1. **Galaxy-specific CCL**: Ring topology operations (`line_reduce_scatter`, `line_all_gather`, `line_all_reduce`)
2. **Double MatMul**: Fused w1/w3 projections with reduce-scatter
3. **Prefetcher**: Separate sub-device for weight management
4. **Memory Layouts**: Ring-specific memory configurations
5. **QK Norm Support**: For Qwen models

---

## Attention Operations

### QKV Projection (Decode)

**Location**: `llama_attention.py::forward_decode()`

**Operation**: `ttnn.matmul(x, self.wqkv)`

**Step-by-Step**:

```
Step 1: Input Preparation
├─ Input: x [1, 1, 32, 1280] (batch_size=32, dim=1280 per device)
├─ Weight: wqkv [sharded across devices]
│   └─ Shape per device: [1, 1, 1280, 12288/8] = [1, 1, 1280, 1536]
│   └─ Memory: SHARDED_QKV_RING_MEMCFG (ring-optimized sharding)
│   └─ Layout: TILE_LAYOUT
│   └─ Sharding: dims=(3, 2) for TG, dims=(2, 3) otherwise
│   └─ Mesh: ShardTensor2dMesh with cluster_shape
│
Step 2: MatMul Execution
├─ Program Config: XQKV_DECODE_RING_PROGCFG
│   └─ Optimized for ring topology decode
├─ Compute Kernel: HIFI2
│   └─ High-fidelity compute for DRAM-sharded matmuls
├─ Memory Config: SHARDED_QKV_OUT_RING_MEMCFG
│   └─ Ring-optimized output memory layout
├─ Prefetcher Support:
│   ├─ global_cb: Prefetcher circular buffer (if enabled)
│   └─ sub_device_id: Worker sub-device ID
│
Step 3: Output
└─ Output: xqkv_fused_sharded [1, 1, 32, 1536]
    └─ Contains Q, K, V concatenated
    └─ Sharded across devices (column-wise)
    └─ Memory: SHARDED_QKV_OUT_RING_MEMCFG
```

**Key Details**:
- **Why HIFI2**: DRAM-sharded matmuls are flop-bound, HIFI2 provides good balance
- **Ring Optimization**: Memory config optimized for ring topology communication
- **Prefetcher**: Weights are prefetched into L1 circular buffer for faster access
- **Sharding**: Column-wise sharding splits QKV across devices

**Code Reference**:
```python
xqkv_fused_sharded = ttnn.matmul(
    x,  # [1, 1, 32, 1280]
    self.wqkv,
    program_config=self.model_config["XQKV_DECODE_RING_PROGCFG"],
    memory_config=self.model_config["SHARDED_QKV_OUT_RING_MEMCFG"],
    compute_kernel_config=self.compute_kernel_config_hifi2,
    global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
    dtype=ttnn.bfloat16,
    sub_device_id=self.prefetcher_setup.worker_sub_device_id,
)
```

---

### Create QKV Heads with Reduce-Scatter

**Location**: `llama_ccl.py::llama_rs_create_heads()`

**Operation**: `self.tt_ccl.llama_rs_create_heads(xqkv_fused_sharded, ...)`

**Purpose**: Split QKV into separate heads while performing reduce-scatter for ring topology

**Step-by-Step**:

```
Step 1: Input
├─ xqkv_fused_sharded: [1, 1, 32, 1536]
│   └─ Contains Q, K, V concatenated
│   └─ Sharded across devices (column-wise)
│
Step 2: Reshape and Split
├─ Reshape to separate Q, K, V
│   ├─ Q portion: [1, 1, 32, 1024] → [1, 8, 8, 128]
│   │   └─ n_local_heads = 8 per device
│   │   └─ head_dim = 128
│   ├─ K portion: [1, 1, 32, 256] → [1, 8, 1, 128]
│   │   └─ n_local_kv_heads = 1 per device (GQA)
│   │   └─ head_dim = 128
│   └─ V portion: [1, 1, 32, 256] → [1, 8, 1, 128]
│       └─ Same as K
│
Step 3: Reduce-Scatter Operation
├─ Cluster Axis: 1 (column dimension)
├─ Num Links: GALAXY_NUM_LINKS (typically 3)
├─ Dimension: 3 (feature dimension)
├─ Operation:
│   ├─ Each device has partial Q, K, V
│   ├─ Reduce-scatter along cluster_axis=1
│   ├─ Reduces partial results across devices
│   └─ Scatters to appropriate devices
│
Step 4: Memory Configuration
├─ QKV Memory Config: CREATE_HEAD_OUTPUT_MEMCFG
│   └─ Optimized for ring topology
│   └─ Height-sharded layout
│
Step 5: Output
├─ q_heads_pre_rot_1BQD: [1, 8, 8, 128]
│   └─ Query heads (before RoPE)
│   └─ Format: [batch, n_heads, seq_len, head_dim]
├─ k_heads_pre_rot_1BKD: [1, 8, 1, 128]
│   └─ Key heads (before RoPE)
│   └─ Format: [batch, n_kv_heads, seq_len, head_dim]
└─ v_heads_1BKD: [1, 8, 1, 128]
    └─ Value heads
    └─ Format: [batch, n_kv_heads, seq_len, head_dim]
```

**Key Details**:
- **Reduce-Scatter**: Combines reduction and scattering in one operation
- **Ring Topology**: Optimized for ring communication pattern
- **GQA Support**: Handles grouped query attention (fewer KV heads)
- **Memory Layout**: Height-sharded for efficient ring communication

**Why Reduce-Scatter**:
- In ring topology, we need to reduce partial results and scatter to correct devices
- More efficient than separate reduce + scatter operations
- Reduces memory bandwidth requirements

**Code Reference**:
```python
(
    q_heads_pre_rot_1BQD,
    k_heads_pre_rot_1BKD,
    v_heads_1BKD,
) = self.tt_ccl.llama_rs_create_heads(
    xqkv_fused_sharded,
    cluster_axis=1,
    num_links=self.model_config["GALAXY_NUM_LINKS"],
    dim=3,
    qkv_memory_config=self.model_config["CREATE_HEAD_OUTPUT_MEMCFG"],
    use_optimal_ccl_for_llama=True,
)
```

---

### RoPE Application

**Location**: `llama_attention.py::forward_decode()`

**Operation**: `ttnn.experimental.rotary_embedding_llama_fused_qk()`

**Purpose**: Apply rotary position embeddings to Q and K tensors

**Step-by-Step**:

```
Step 1: Input Preparation
├─ q_heads_pre_rot_1BQD: [1, 8, 8, 128]
│   └─ Query heads before RoPE
├─ k_heads_pre_rot_1BKD: [1, 8, 1, 128]
│   └─ Key heads before RoPE
├─ rot_mats[0]: cos_matrix
│   └─ Cosine rotation matrices
│   └─ Shape: [1, 1, max_seq_len, head_dim]
├─ rot_mats[1]: sin_matrix
│   └─ Sine rotation matrices
│   └─ Shape: [1, 1, max_seq_len, head_dim]
└─ transformation_mats["decode"]: Transformation matrices
    └─ For efficient RoPE application
│
Step 2: Position Indexing
├─ current_pos: [32] (one position per user in batch)
├─ Extract rotation matrices for current positions
│   └─ For each user, get cos/sin at their current position
│
Step 3: Fused RoPE Application
├─ Operation: rotary_embedding_llama_fused_qk
│   ├─ Fused operation for Q and K
│   ├─ Applies rotation in complex plane
│   └─ More efficient than separate Q and K operations
│
Step 4: Rotation Formula
├─ For each head dimension pair (i, i+1):
│   ├─ q_rotated[i] = q[i] * cos[pos] - q[i+1] * sin[pos]
│   └─ q_rotated[i+1] = q[i] * sin[pos] + q[i+1] * cos[pos]
│   └─ Same for K
│
Step 5: Output
├─ q_heads_1BQD: [1, 8, 8, 128]
│   └─ Query heads with RoPE applied
└─ k_heads_1BKD: [1, 8, 1, 128]
    └─ Key heads with RoPE applied
```

**Key Details**:
- **Fused Operation**: Processes Q and K together for efficiency
- **Complex Rotation**: Rotates in complex plane (treats pairs of dimensions as complex numbers)
- **Position-Dependent**: Different rotation for each position in sequence
- **Memory Efficient**: In-place operation where possible

**Why RoPE**:
- Provides relative positional information
- Better than absolute positional embeddings for long sequences
- Allows extrapolation beyond training length

**Code Reference**:
```python
q_heads_1BQD, k_heads_1BKD = ttnn.experimental.rotary_embedding_llama_fused_qk(
    q_heads_pre_rot_1BQD,
    k_heads_pre_rot_1BKD,
    rot_mats[0],  # cos_matrix
    rot_mats[1],  # sin_matrix
    self.transformation_mats["decode"]
)
```

---

### KV Cache Update

**Location**: `llama_attention.py::forward_decode()`

**Operation**: `ttnn.experimental.paged_fused_update_cache()`

**Purpose**: Update KV cache with new K and V values at current positions

**Step-by-Step**:

```
Step 1: Input Preparation
├─ keys: KV cache for K [batch, n_kv_heads, max_seq_len, head_dim]
│   └─ Shape: [32, 1, max_seq_len, 128]
├─ k_heads_1BKD: New K values [1, 8, 1, 128]
│   └─ Format: [batch, n_kv_heads, 1, head_dim]
├─ values: KV cache for V [batch, n_kv_heads, max_seq_len, head_dim]
│   └─ Shape: [32, 1, max_seq_len, 128]
├─ v_heads_1BKD: New V values [1, 8, 1, 128]
│   └─ Format: [batch, n_kv_heads, 1, head_dim]
├─ update_idxs_tensor: current_pos [32]
│   └─ Current position for each user
└─ page_table: [batch, max_num_blocks] (if paged attention)
    └─ Maps logical positions to physical cache blocks
│
Step 2: Paged Attention Logic
├─ If paged_attention:
│   ├─ For each user in batch:
│   │   ├─ Get logical position: current_pos[user_id]
│   │   ├─ Calculate logical block: pos // block_size
│   │   ├─ Get physical block: page_table[user_id, logical_block]
│   │   ├─ Calculate offset in block: pos % block_size
│   │   └─ Update cache[physical_block, :, offset, :] = new_kv
│   └─ Else (standard cache):
│       └─ cache[user_id, :, current_pos[user_id], :] = new_kv
│
Step 3: Fused Update
├─ Operation: paged_fused_update_cache
│   ├─ Updates both K and V in one operation
│   ├─ Handles paging logic internally
│   └─ Optimized for batch updates
│
Step 4: Memory Management
├─ Cache stored in DRAM
├─ Updates are atomic per user
└─ No need to read entire cache, only update specific positions
│
Step 5: Output
└─ KV cache updated in-place
    └─ Ready for attention computation
```

**Key Details**:
- **Paged Attention**: Efficient memory management for long sequences
- **Batch Updates**: Updates all users in batch simultaneously
- **Fused Operation**: Updates K and V together
- **Memory Efficient**: Only updates specific positions, doesn't read entire cache

**Why Paged Attention**:
- Allows handling sequences longer than max_seq_len
- Reduces memory fragmentation
- Enables dynamic memory allocation
- Compatible with vLLM

**Code Reference**:
```python
ttnn.experimental.paged_fused_update_cache(
    keys,           # K cache
    k_heads_1BKD,   # New K values
    values,         # V cache
    v_heads_1BKD,   # New V values
    update_idxs_tensor=current_pos,
    page_table=page_table
)
```

---

### Scaled Dot-Product Attention (SDPA)

**Location**: `llama_attention.py::forward_decode()`

**Operation**: `ttnn.transformer.scaled_dot_product_attention_decode()`

**Purpose**: Compute attention scores and apply to values

**Step-by-Step**:

```
Step 1: Input Preparation
├─ query_layer: q_heads_1BQD [1, 8, 8, 128]
│   └─ Query heads with RoPE
│   └─ Format: [batch, n_heads, seq_len, head_dim]
├─ keys: KV cache [32, 1, current_pos+1, 128]
│   └─ All cached K values up to current position
│   └─ Format: [batch, n_kv_heads, seq_len, head_dim]
├─ values: KV cache [32, 1, current_pos+1, 128]
│   └─ All cached V values up to current position
│   └─ Format: [batch, n_kv_heads, seq_len, head_dim]
└─ cur_pos: current_pos [32]
    └─ Current position for each user
│
Step 2: Attention Score Computation
├─ For each query head:
│   ├─ Q @ K^T: [1, 8, 8, 128] @ [32, 1, 128, current_pos+1]
│   │   └─ Result: [1, 8, 8, current_pos+1]
│   │   └─ Attention scores (logits)
│   ├─ Scale: scores / sqrt(head_dim)
│   │   └─ Prevents large dot products
│   ├─ Causal Mask: Set future positions to -inf
│   │   └─ Only attend to past and current tokens
│   └─ Softmax: exp(scores) / sum(exp(scores))
│       └─ Attention weights [0, 1]
│
Step 3: Apply Attention to Values
├─ Attention @ V: [1, 8, 8, current_pos+1] @ [32, 1, current_pos+1, 128]
│   └─ Result: [1, 8, 8, 128]
│   └─ Weighted sum of values
│
Step 4: Program Configuration
├─ Program Config: SDPA_DECODE_PROGCFG
│   └─ Optimized for decode attention
│   └─ Handles variable sequence lengths
│
Step 5: Output
└─ attn_output: [1, 8, 8, 128]
    └─ Attention output per head
    └─ Format: [batch, n_heads, seq_len, head_dim]
```

**Key Details**:
- **Causal Masking**: Prevents attending to future tokens
- **GQA Support**: Handles grouped query attention (Q has more heads than K/V)
- **Variable Length**: Handles different sequence lengths per user
- **Memory Efficient**: Uses KV cache instead of recomputing

**Mathematical Formula**:
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Where:
- Q: Query [batch, n_heads, 1, head_dim]
- K: Keys [batch, n_kv_heads, seq_len, head_dim]
- V: Values [batch, n_kv_heads, seq_len, head_dim]
- d_k: head_dim (128)

**Code Reference**:
```python
attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
    query_layer,      # Q
    keys,            # K cache
    values,          # V cache
    cur_pos=current_pos,
    attn_mask=None,
    compute_kernel_config=self.sdpa_decode_compute_kernel_cfg,
    program_config=self.model_config["SDPA_DECODE_PROGCFG"],
)
```

---

### Output Projection with All-Gather

**Location**: `llama_attention.py::forward_decode()`

**Operation**: `ttnn.matmul(attn_output, self.wo)` + `line_all_gather` or fused all-gather matmul

**Purpose**: Project attention output back to hidden dimension

**Step-by-Step**:

```
Step 1: Concat Heads
├─ Input: attn_output [1, 8, 8, 128]
│   └─ Attention output per head
├─ Operation: nlp_concat_heads_decode
│   └─ Concatenates heads along feature dimension
└─ Output: attn_output_concat [1, 1, 32, 1024]
    └─ Format: [batch, 1, seq_len, n_heads * head_dim]
│
Step 2: Output Projection
├─ Input: attn_output_concat [1, 1, 32, 1024]
├─ Weight: wo [sharded]
│   └─ Shape per device: [1, 1, 1024, 1280]
│   └─ Row-wise sharded (dims=(2, 3))
│   └─ Memory: SHARDED_WO_RING_MEMCFG
│
Step 3: MatMul Options
├─ Option A: Fused All-Gather MatMul (if enabled)
│   ├─ Operation: Fused matmul + all-gather
│   ├─ More efficient for ring topology
│   └─ Reduces memory traffic
│
└─ Option B: Separate MatMul + All-Gather
    ├─ Step 3a: MatMul
    │   ├─ attn_output @ wo
    │   └─ Output: [1, 1, 32, 1280/8] = [1, 1, 32, 160]
    │       └─ Sharded across devices
    │
    └─ Step 3b: All-Gather
        ├─ Operation: line_all_gather
        ├─ Cluster Axis: 1 (column dimension)
        ├─ Num Links: GALAXY_NUM_LINKS
        ├─ Dimension: 3 (feature dimension)
        └─ Output: [1, 1, 32, 1280]
            └─ Full hidden dimension
│
Step 4: Memory Configuration
├─ Output Memory: DECODE_RESIDUAL_MEMCFG
│   └─ For residual connection
│   └─ Ring-optimized layout
│
Step 5: Output
└─ output: [1, 1, 32, 1280]
    └─ Attention output projected to hidden dim
    └─ Ready for residual connection
```

**Key Details**:
- **Fused Operation**: If `use_fused_all_gather_matmul`, combines matmul and all-gather
- **Ring Optimization**: Uses ring topology for all-gather
- **Row-wise Sharding**: Weight is row-wise sharded, requires all-gather
- **Memory Efficient**: Output in residual memory config

**Why All-Gather**:
- Weight is row-wise sharded (split across devices)
- Each device has partial result
- Need to gather all partial results to get full output

**Code Reference**:
```python
# Concat heads
attn_output = ttnn.experimental.nlp_concat_heads_decode(
    attn_output,
    memory_config=self.model_config["CONCAT_HEADS_DECODE_MEMCFG"],
)

# Output projection
if self.use_fused_all_gather_matmul:
    output = ttnn.matmul(...)  # Fused with all-gather
else:
    output = ttnn.matmul(attn_output, self.wo, ...)
    output = self.tt_ccl.line_all_gather(
        output,
        dim=3,
        cluster_axis=1,
        num_links=self.model_config["GALAXY_NUM_LINKS"],
        ...
    )
```

---

## MLP Operations

### Double MatMul with Reduce-Scatter

**Location**: `llama_mlp.py::forward()` (decode mode)

**Operation**: `self.tt_ccl.double_matmul_line_reduce_scatter()`

**Purpose**: Compute w1 and w3 projections simultaneously with reduce-scatter

**Step-by-Step**:

```
Step 1: Input Preparation
├─ Input: x [1, 1, 32, 1280]
│   └─ Hidden dimension (sharded)
│   └─ Memory: SHARDED_FF12_RING_MEMCFG
├─ Weight w1: [sharded]
│   └─ Shape per device: [1, 1, 1280, 14336/8] = [1, 1, 1280, 1792]
│   └─ Column-wise sharded (dims=(-1, -2))
│   └─ Memory: W1W3_RING_MEMCFG
├─ Weight w3: [sharded]
│   └─ Shape per device: [1, 1, 1280, 14336/8] = [1, 1, 1280, 1792]
│   └─ Column-wise sharded (dims=(-1, -2))
│   └─ Memory: W1W3_RING_MEMCFG
│
Step 2: Double MatMul Execution
├─ Operation: double_matmul_line_reduce_scatter
│   ├─ Computes x @ w1 and x @ w3 simultaneously
│   ├─ More efficient than separate matmuls
│   └─ Reduces memory bandwidth
│
Step 3: MatMul Computation
├─ w1_out = x @ w1
│   └─ Shape: [1, 1, 32, 1792]
│   └─ Partial result per device
├─ w3_out = x @ w3
│   └─ Shape: [1, 1, 32, 1792]
│   └─ Partial result per device
│
Step 4: Reduce-Scatter Operation
├─ Cluster Axis: 1 (column dimension)
├─ Num Links: GALAXY_NUM_LINKS (typically 3)
├─ Operation:
│   ├─ Reduces partial results across devices
│   ├─ Scatters to appropriate devices
│   └─ Uses ring topology
│
Step 5: Output
├─ w1_out_reduced: [1, 1, 32, 1792]
│   └─ Gate projection output (reduced)
│   └─ Memory: REDUCE_SCATTER_OUT_MEMCFG
└─ w3_out: [1, 1, 32, 1792]
    └─ Up projection output (not yet reduced)
    └─ Memory: SHARDED_FF12_OUT_RING_MEMCFG
```

**Key Details**:
- **Double MatMul**: Computes w1 and w3 in one fused operation
- **Reduce-Scatter**: Only w1 is reduced, w3 reduced separately
- **Ring Topology**: Optimized for ring communication
- **Memory Efficient**: Reduces memory traffic

**Why Double MatMul**:
- Both w1 and w3 use same input
- Can share input data loading
- Reduces memory bandwidth
- More efficient than separate matmuls

**Code Reference**:
```python
w1_out_reduced, w3_out = self.tt_ccl.double_matmul_line_reduce_scatter(
    x,
    self.w1,
    self.w3,
    cluster_axis=1,
    num_links=self.model_config["GALAXY_NUM_LINKS"],
    RS_memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
    compute_kernel_config=self.args.compute_kernel_config_lofi if self.four_bit_mlp else self.args.compute_kernel_config_hifi2,
    dtype=ttnn.bfloat8_b,
    program_config=pc_1_3,
    memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
    global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
    sub_device_id=self.prefetcher_setup.worker_sub_device_id if mode == "decode" else None,
    use_noc1_only=False,
)
```

---

### Element-wise Multiply with SiLU

**Location**: `llama_mlp.py::forward()` (decode mode)

**Operation**: `ttnn.mul(w1_out_reduced, w3_out_reduced, input_tensor_a_activations=[SILU])`

**Purpose**: Apply SiLU activation and multiply w1 and w3 outputs

**Step-by-Step**:

```
Step 1: Reduce w3 Output
├─ Input: w3_out [1, 1, 32, 1792]
│   └─ Up projection output (not yet reduced)
├─ Operation: line_reduce_scatter
│   ├─ Cluster Axis: 1
│   ├─ Num Links: GALAXY_NUM_LINKS
│   └─ Reduces partial results
└─ Output: w3_out_reduced [1, 1, 32, 1792]
    └─ Memory: REDUCE_SCATTER_OUT_MEMCFG
│
Step 2: SiLU Activation
├─ Input: w1_out_reduced [1, 1, 32, 1792]
├─ Operation: SiLU(x) = x * sigmoid(x)
│   ├─ Computed element-wise
│   └─ Applied in-place during multiply
└─ Output: w1_silu [1, 1, 32, 1792]
    └─ Gate projection with SiLU
│
Step 3: Element-wise Multiply
├─ Input A: w1_silu [1, 1, 32, 1792]
├─ Input B: w3_out_reduced [1, 1, 32, 1792]
├─ Operation: element-wise multiply
│   └─ ff1ff3[i] = w1_silu[i] * w3_out_reduced[i]
└─ Output: ff1ff3 [1, 1, 32, 1792]
    └─ Memory: REDUCE_SCATTER_OUT_MEMCFG
│
Step 4: Memory Management
├─ Deallocate w3_out_reduced
└─ Deallocate w1_out_reduced
```

**Key Details**:
- **Fused Operation**: SiLU applied during multiply
- **In-place**: Where possible for memory efficiency
- **Element-wise**: Each element multiplied independently

**Mathematical Formula**:
```
ff1ff3 = SiLU(w1_out) * w3_out
       = (w1_out * sigmoid(w1_out)) * w3_out
```

**Code Reference**:
```python
# Reduce w3
w3_out_reduced = self.tt_ccl.line_reduce_scatter(
    w3_out,
    cluster_axis=1,
    num_links=self.model_config["GALAXY_NUM_LINKS"],
    memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
    use_noc1_only=False,
)

# SiLU + Multiply
ff1ff3 = ttnn.mul(
    w1_out_reduced,
    w3_out_reduced,
    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
    dtype=ttnn.bfloat8_b,
    memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
)
```

---

### All-Gather for W2 Input

**Location**: `llama_mlp.py::forward()` (decode mode)

**Operation**: `self.tt_ccl.line_all_gather(ff1ff3, ...)`

**Purpose**: Gather distributed ff1ff3 results for w2 projection

**Step-by-Step**:

```
Step 1: Input Preparation
├─ Input: ff1ff3 [1, 1, 32, 1792]
│   └─ Element-wise product of w1_silu and w3_out
│   └─ Sharded across devices (column-wise)
│   └─ Memory: REDUCE_SCATTER_OUT_MEMCFG
│
Step 2: All-Gather Operation
├─ Cluster Axis: 1 (column dimension)
├─ Num Links: GALAXY_NUM_LINKS (typically 3)
├─ Dimension: 3 (feature dimension)
├─ Buffer Key: "BINARY_MUL"
│   └─ Uses persistent buffer for efficiency
├─ Operation:
│   ├─ Each device has partial ff1ff3
│   ├─ Gathers all partial results
│   └─ Uses ring topology
│
Step 3: Ring Topology Communication
├─ Ring Pattern:
│   ├─ Device 0 → Device 1 → Device 2 → ... → Device 31 → Device 0
│   ├─ Each device passes its partial result
│   └─ Accumulates all partial results
│
Step 4: Output
└─ w2_in: [1, 1, 32, 14336]
    └─ Full hidden dimension
    └─ Memory: FF2_IN_RING_MEMCFG
    └─ Ready for w2 projection
```

**Key Details**:
- **Persistent Buffer**: Uses pre-allocated buffer for efficiency
- **Ring Topology**: Optimized communication pattern
- **Memory Efficient**: Reuses buffers where possible

**Why All-Gather**:
- w2 needs full hidden dimension input
- ff1ff3 is column-wise sharded
- Need to gather all shards before w2 projection

**Code Reference**:
```python
w2_in = self.tt_ccl.line_all_gather(
    ff1ff3,
    dim=3,
    cluster_axis=1,
    num_links=self.model_config["GALAXY_NUM_LINKS"],
    memory_config=self.model_config["FF2_IN_RING_MEMCFG"],
    buffer_key="BINARY_MUL",
    use_optimal_ccl_for_llama=False if mode == "prefill" else True,
)
```

---

### W2 MatMul with All-Reduce

**Location**: `llama_mlp.py::forward()` (decode mode)

**Operation**: `ttnn.linear(w2_in, self.w2)` + `line_all_reduce()`

**Purpose**: Down projection with all-reduce for row-wise sharding

**Step-by-Step**:

```
Step 1: Input Preparation
├─ Input: w2_in [1, 1, 32, 14336]
│   └─ Full hidden dimension
│   └─ Memory: FF2_IN_RING_MEMCFG
├─ Weight w2: [sharded]
│   └─ Shape per device: [1, 1, 14336/8, 1280] = [1, 1, 1792, 1280]
│   └─ Row-wise sharded (dims=(-2, -1))
│   └─ Memory: W2_RING_MEMCFG
│
Step 2: MatMul Execution
├─ Operation: ttnn.linear
├─ Program Config: FF2_TG_RING_PROGCFG
│   └─ Optimized for ring topology
├─ Compute Kernel: HIFI2
│   └─ High-fidelity compute
├─ Memory Config: FF2_OUT_RING_MEMCFG
│   └─ Ring-optimized output layout
├─ Prefetcher Support:
│   ├─ global_cb: Prefetcher circular buffer
│   └─ sub_device_id: Worker sub-device ID
│
Step 3: MatMul Result
├─ w2_out = w2_in @ w2
└─ Output: [1, 1, 32, 1280/8] = [1, 1, 32, 160]
    └─ Partial result per device
    └─ Memory: FF2_OUT_RING_MEMCFG
│
Step 4: All-Reduce Operation
├─ Cluster Axis: 0 (row dimension)
├─ Num Links: GALAXY_NUM_LINKS
├─ Operation: line_all_reduce
│   ├─ Reduces partial results across devices
│   ├─ Sums all partial results
│   └─ Uses ring topology
│
Step 5: Output
└─ w2_out_reduced: [1, 1, 32, 1280]
    └─ Full hidden dimension
    └─ Memory: DECODE_RESIDUAL_MEMCFG
    └─ Ready for residual connection
```

**Key Details**:
- **Row-wise Sharding**: Weight is row-wise sharded
- **All-Reduce**: Sums partial results (not gather, since we sum)
- **Ring Topology**: Optimized communication
- **Residual Memory**: Output in residual memory config

**Why All-Reduce**:
- Weight is row-wise sharded
- Each device computes partial result
- Need to sum all partial results (not concatenate)

**Code Reference**:
```python
# MatMul
w2_out = ttnn.linear(
    w2_in,
    self.w2,
    compute_kernel_config=self.args.compute_kernel_config_hifi2,
    dtype=ttnn.bfloat8_b,
    program_config=pc_2,
    memory_config=self.model_config["FF2_OUT_RING_MEMCFG"],
    core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
    global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
    sub_device_id=self.prefetcher_setup.worker_sub_device_id if mode == "decode" else None,
)

# All-Reduce
w2_out_reduced = self.tt_ccl.line_all_reduce(
    w2_out,
    cluster_axis=0,
    num_links=self.model_config["GALAXY_NUM_LINKS"],
    memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
    use_optimal_ccl_for_llama=True,
)
```

---

## Distributed Norm Operations

**Location**: `distributed_norm.py`

**Purpose**: RMSNorm with distributed all-reduce for ring topology

**Step-by-Step**:

```
Step 1: Input Preparation
├─ Input: x [sharded]
│   └─ Hidden dimension sharded across devices
│   └─ Memory: Input memory config
├─ Residual: h [optional]
│   └─ Previous layer output for fused residual
│
Step 2: RMSNorm Computation
├─ Operation: RMSNorm
│   ├─ Mean: mean(x^2)
│   ├─ Normalize: x / sqrt(mean + eps)
│   └─ Scale: output * weight
│
Step 3: Distributed Reduction
├─ If is_distributed:
│   ├─ Compute partial statistics per device
│   ├─ All-reduce statistics across devices
│   └─ Apply normalization
│
Step 4: Output
└─ Output: [sharded or replicated]
    └─ Normalized output
    └─ Memory: Output memory config
```

**Key Details**:
- **Distributed Statistics**: Computes mean across all devices
- **Ring All-Reduce**: Uses ring topology for reduction
- **Fused Residual**: Can fuse with residual addition

---

## CCL Operations

### Ring Topology Operations

**Location**: `llama_ccl.py`

**Ring Topology**: Devices arranged in a ring (0 → 1 → 2 → ... → 31 → 0)

**Operations**:
1. **line_reduce_scatter**: Reduce and scatter along ring
2. **line_all_gather**: Gather along ring
3. **line_all_reduce**: Reduce along ring

**Key Features**:
- **Bandwidth Efficient**: Uses all links simultaneously
- **Latency**: O(num_devices) steps
- **Memory**: Minimal intermediate buffers

### Line Topology Operations

**Alternative topology** (for debugging):
- **Line Pattern**: Devices in a line
- **Higher Latency**: O(num_devices) but less efficient
- **Debug Only**: Not recommended for production

---

## Prefetcher System

**Location**: `prefetcher_common.py`

**Purpose**: Dedicated sub-device for weight prefetching

**Step-by-Step**:

```
Step 1: Prefetcher Setup
├─ Sub-device: Dedicated cores for prefetching
├─ Circular Buffer: Pre-allocated L1 buffer
└─ Weight Management: Tracks which weights to prefetch
│
Step 2: Weight Prefetching
├─ Before computation:
│   ├─ Prefetcher loads weights from DRAM to L1
│   ├─ Stores in circular buffer
│   └─ Ready for computation
│
Step 3: Computation
├─ Worker cores:
│   ├─ Read weights from circular buffer
│   ├─ Perform computation
│   └─ Prefetcher loads next weights
│
Step 4: Benefits
└─ Overlaps weight loading with computation
    └─ Reduces memory stalls
    └─ Improves throughput
```

**Key Details**:
- **Dedicated Cores**: Separate sub-device for prefetching
- **Circular Buffer**: Reusable L1 buffer
- **Overlap**: Computation and prefetching happen simultaneously
- **Performance**: Significant speedup for decode

---

## Summary

The Llama3-70B Galaxy implementation uses several key optimizations:

1. **Ring Topology CCL**: Efficient collective communication
2. **Fused Operations**: Double matmul, fused RoPE, fused cache update
3. **Prefetcher**: Dedicated weight prefetching
4. **Memory Optimization**: Ring-specific memory layouts
5. **Batch Processing**: Optimized for batch size 32

Each operation is carefully designed to maximize throughput on Galaxy hardware while maintaining accuracy.
