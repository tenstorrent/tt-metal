# Llama3-70B Galaxy: Complete Code Walkthrough

This document provides a detailed, line-by-line walkthrough of the Llama3-70B Galaxy implementation with actual code references, tensor shapes, and configuration details.

## Table of Contents

1. [Decode Attention Forward Pass](#decode-attention-forward-pass)
2. [Decode MLP Forward Pass](#decode-mlp-forward-pass)
3. [Prefill Attention Forward Pass](#prefill-attention-forward-pass)
4. [Prefill MLP Forward Pass](#prefill-mlp-forward-pass)
5. [CCL Operations Deep Dive](#ccl-operations-deep-dive)
6. [Prefetcher System Deep Dive](#prefetcher-system-deep-dive)
7. [Memory Management Deep Dive](#memory-management-deep-dive)

---

## Decode Attention Forward Pass

### Overview

**File**: `llama_attention.py`
**Method**: `forward_decode()`
**Lines**: 373-567

### Input Parameters

```python
def forward_decode(
    self,
    x: ttnn.Tensor,              # Shape: [1, 1, 32, 1280]
    current_pos,                 # Shape: [32] - current position per user
    rot_mats=None,              # Tuple of (cos_matrix, sin_matrix)
    page_table=None,            # Optional: for paged attention
    kv_cache=None,              # Optional: external KV cache
) -> ttnn.Tensor:
```

### Step 1: QKV Projection (Lines 389-398)

**Operation**: Matmul of input with QKV weight

```python
xqkv_fused_sharded = ttnn.matmul(
    x,                          # Input: [1, 1, 32, 1280]
    self.wqkv,                  # Weight: [1, 1, 1280, 1536] per device
    program_config=self.model_config["XQKV_DECODE_RING_PROGCFG"],
    memory_config=self.model_config["SHARDED_QKV_OUT_RING_MEMCFG"],
    compute_kernel_config=self.compute_kernel_config_hifi2,
    global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
    dtype=ttnn.bfloat16,
    sub_device_id=self.prefetcher_setup.worker_sub_device_id,
)
```

**Detailed Breakdown**:

1. **Input Shape**: `[1, 1, 32, 1280]`
   - Batch dim: 1 (fixed for decode)
   - Sequence dim: 1 (always 1 for decode)
   - User batch: 32 users
   - Hidden dim: 1280 (per device, full is 8192 / 8 devices = 1024, but with padding = 1280)

2. **Weight Shape**: `[1, 1, 1280, 1536]` per device
   - 1536 = (1024 + 256 + 256) / 8 devices
   - 1024 = Q projection (8 heads * 128 dim) per device
   - 256 = K projection (1 head * 128 dim) per device (GQA)
   - 256 = V projection (1 head * 128 dim) per device (GQA)
   - Full QKV: 12288 = (8192 Q + 2048 K + 2048 V)
   - Per device: 12288 / 8 = 1536

3. **Weight Sharding**: `dims=(3, 2)` for TG
   - Sharded along dimension 3 (output features)
   - Dimension 2 specifies the mesh dimension to shard along

4. **Program Config**: `XQKV_DECODE_RING_PROGCFG`
   - Ring-optimized matmul program configuration
   - Configured for batch=32, seq_len=1

5. **Memory Config**: `SHARDED_QKV_OUT_RING_MEMCFG`
   - Ring-optimized output memory layout
   - Width-sharded for efficient ring communication

6. **Compute Kernel**: `compute_kernel_config_hifi2`
   - High-fidelity compute (HIFI2)
   - Used because DRAM-sharded matmuls are flop-bound
   - Provides good balance of speed and precision

7. **Prefetcher**: `global_circular_buffer`
   - Weight is prefetched from DRAM to L1 circular buffer
   - Prefetcher runs on separate sub-device
   - Overlaps weight loading with computation

8. **Sub-device**: `worker_sub_device_id`
   - Computation runs on worker sub-device
   - Separate from prefetcher sub-device

**Output Shape**: `[1, 1, 32, 1536]`
- Contains Q, K, V concatenated
- Sharded across 8 devices (column-wise)

**Memory Deallocation**:

```python
ttnn.deallocate(x)  # Line 399
```

Immediately deallocate input to free memory.

### Step 2: Create QKV Heads with Reduce-Scatter (Lines 405-416)

**Operation**: Split QKV and perform reduce-scatter

```python
(
    q_heads_pre_rot_1BQD,       # [1, 8, 8, 128]
    k_heads_pre_rot_1BKD,       # [1, 8, 1, 128]
    v_heads_1BKD,               # [1, 8, 1, 128]
) = self.tt_ccl.llama_rs_create_heads(
    xqkv_fused_sharded,        # [1, 1, 32, 1536]
    cluster_axis=1,            # Column axis for reduce-scatter
    num_links=self.model_config["GALAXY_NUM_LINKS"],  # 3 links
    dim=3,                     # Dimension to reduce-scatter on
    qkv_memory_config=self.model_config["CREATE_HEAD_OUTPUT_MEMCFG"],
    use_optimal_ccl_for_llama=True,
)
```

**Detailed Breakdown**:

1. **Input**: `xqkv_fused_sharded [1, 1, 32, 1536]`
   - Contains concatenated Q, K, V projections
   - Sharded across devices

2. **Operation**: `llama_rs_create_heads`
   - Location: `llama_ccl.py`, lines 866-899
   - Calls: `ttnn.experimental.llama_rs_create_heads`

3. **Internal Operations**:

   a. **Split QKV** (inside ttnn op):
      - Q: `[1, 1, 32, 1024]` → `[1, 8, 8, 128]`
        - 8 heads per device
        - 8 users per head (32 users / 4 device groups)
        - 128 head dimension
      - K: `[1, 1, 32, 256]` → `[1, 8, 1, 128]`
        - 8 device groups, 1 KV head per group
        - 1 user per head (after reduce-scatter)
        - 128 head dimension
      - V: Same as K

   b. **Reduce-Scatter**:
      - Operation: Ring-based reduce-scatter
      - Cluster axis: 1 (column dimension)
      - Num links: 3 (GALAXY_NUM_LINKS)
      - Reduces partial results across 4 devices in a row
      - Scatters to appropriate devices

4. **Persistent Buffer**: `rs_create_heads_buffers[cluster_axis]`
   - Pre-allocated intermediate buffer
   - Location: `llama_ccl.py`, lines 414-443
   - Size: `[8, 4, 32, 640]` (8x4 mesh, 32 batch, 640 = 4 devices * 4 pages * 32 tile_width * 5)
   - Memory: `RS_CREATE_HEADS_INTERIM_MEMCFG`

5. **Semaphore**: `gather_semaphore_handles[cluster_axis][gather_idx[cluster_axis]]`
   - For synchronization across devices
   - Double-buffered (2 semaphores)

**Output Shapes**:
- `q_heads_pre_rot_1BQD`: `[1, 8, 8, 128]`
  - 1: batch dimension
  - 8: device groups (for GQA)
  - 8: users per group
  - 128: head dimension
- `k_heads_pre_rot_1BKD`: `[1, 8, 1, 128]`
  - 1: batch dimension
  - 8: device groups
  - 1: single KV head per group (GQA)
  - 128: head dimension
- `v_heads_1BKD`: Same as K

**Memory Deallocation**:

```python
ttnn.deallocate(xqkv_fused_sharded)  # Line 471
```

### Step 3: QK Normalization (Optional, Lines 418-469)

**Condition**: `if self.qk_norm` (for Qwen models)

This section is skipped for standard Llama3-70B but included for Qwen3-32B support.

**Operations**:
1. Reshape Q and K to `[1, 1, 64, 128]`
2. Apply RMSNorm to Q and K separately
3. Reshape back to `[1, 8, 8, 128]` and `[1, 8, 1, 128]`

### Step 4: RoPE Application (Lines 474-479)

**Operation**: Apply rotary position embeddings

```python
q_heads_1BQD, k_heads_1BKD = ttnn.experimental.rotary_embedding_llama_fused_qk(
    q_heads_pre_rot_1BQD,      # [1, 8, 8, 128]
    k_heads_pre_rot_1BKD,      # [1, 8, 1, 128]
    rot_mats[0],               # cos_matrix: [1, 1, max_seq_len, 128]
    rot_mats[1],               # sin_matrix: [1, 1, max_seq_len, 128]
    self.transformation_mats["decode"]  # Transformation matrices
)
```

**Detailed Breakdown**:

1. **Rotation Matrices**:
   - `rot_mats[0]`: Cosine matrix `[1, 1, max_seq_len, 128]`
   - `rot_mats[1]`: Sine matrix `[1, 1, max_seq_len, 128]`
   - Precomputed for all possible positions

2. **Transformation Matrices**:
   - `transformation_mats["decode"]`: Decode-specific transformation
   - Used for efficient RoPE computation

3. **Operation**:
   - Fused Q and K rotation (more efficient than separate)
   - For each position `pos` in `current_pos` (32 positions):
     - Extract cos[pos] and sin[pos]
     - For each head dimension pair (i, i+1):
       - `q_rot[i] = q[i] * cos[pos] - q[i+1] * sin[pos]`
       - `q_rot[i+1] = q[i] * sin[pos] + q[i+1] * cos[pos]`
       - Same for K

4. **Why Fused**:
   - Shares cos/sin extraction
   - Single kernel launch
   - Better memory locality

**Output Shapes**:
- `q_heads_1BQD`: `[1, 8, 8, 128]` (with RoPE applied)
- `k_heads_1BKD`: `[1, 8, 1, 128]` (with RoPE applied)

**Memory Deallocation**:

```python
ttnn.deallocate(q_heads_pre_rot_1BQD)  # Line 477
ttnn.deallocate(k_heads_pre_rot_1BKD)  # Line 478
```

### Step 5: KV Cache Update (Lines 484-500)

**Operation**: Update KV cache with new K and V values

```python
if kv_cache:
    keys = kv_cache[0]
    values = kv_cache[1]
else:
    keys = self.layer_past[0]      # [32, 1, max_seq_len, 128]
    values = self.layer_past[1]    # [32, 1, max_seq_len, 128]

ttnn.experimental.paged_fused_update_cache(
    keys,                          # K cache
    k_heads_1BKD,                  # New K values [1, 8, 1, 128]
    values,                        # V cache
    v_heads_1BKD,                  # New V values [1, 8, 1, 128]
    update_idxs_tensor=current_pos,  # [32] current positions
    page_table=page_table          # Optional: for paged attention
)
```

**Detailed Breakdown**:

1. **KV Cache Shape**: `[batch_size_per_device_group, n_local_kv_heads, max_seq_len, head_dim]`
   - Batch: 32 users (or 8 per device group for TG)
   - KV heads: 1 per device (GQA)
   - Sequence: max_seq_len (e.g., 128k tokens)
   - Head dim: 128

2. **Cache Initialization** (lines 318-371):
   ```python
   if self.paged_attention_config:
       cache_k = torch.zeros((
           self.paged_attention_config.max_num_blocks,
           self.n_local_kv_heads,
           self.paged_attention_config.block_size,
           self.head_dim,
       ))
   else:
       cache_k = torch.zeros((
           self.batch_size_per_device_group,
           self.n_local_kv_heads,
           self.max_seq_len,
           self.head_dim,
       ))
   ```

3. **Paged Attention** (if enabled):
   - `page_table`: Maps logical blocks to physical blocks
   - Allows efficient memory management for long sequences
   - Compatible with vLLM

4. **Update Operation**:
   - For each user `i` in batch:
     - Get position: `pos = current_pos[i]`
     - If paged:
       - `logical_block = pos // block_size`
       - `physical_block = page_table[i, logical_block]`
       - `offset = pos % block_size`
       - `cache[physical_block, :, offset, :] = new_kv[i, :, :, :]`
     - Else:
       - `cache[i, :, pos, :] = new_kv[i, :, :, :]`

5. **Fused Update**:
   - Updates both K and V in single operation
   - More efficient than separate updates

**Memory Deallocation**:

```python
ttnn.deallocate(k_heads_1BKD)  # Line 498
ttnn.deallocate(v_heads_1BKD)  # Line 499
```

### Step 6: Scaled Dot-Product Attention (Lines 506-530)

**Operation**: Compute attention and apply to values

```python
sdpa_out_mem_cfg = self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"](self.batch_size_per_device_group)

if page_table:
    attn_output_1G4D_sharded = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_heads_1BQD,              # [1, 8, 8, 128]
        keys,                      # [32, 1, max_seq_len, 128]
        values,                    # [32, 1, max_seq_len, 128]
        cur_pos_tensor=current_pos,  # [32]
        page_table_tensor=page_table,
        scale=self.scale,          # 1/sqrt(128)
        program_config=self.model_config["PAGED_SDPA_DECODE_PROGCFG"],
        compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
        memory_config=sdpa_out_mem_cfg,
    )
else:
    attn_output_1G4D_sharded = ttnn.transformer.scaled_dot_product_attention_decode(
        q_heads_1BQD,              # [1, 8, 8, 128]
        keys,                      # [32, 1, max_seq_len, 128]
        values,                    # [32, 1, max_seq_len, 128]
        cur_pos_tensor=current_pos,  # [32]
        scale=self.scale,          # 1/sqrt(128)
        program_config=self.model_config["SDPA_DECODE_PROGCFG"],
        compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
        memory_config=sdpa_out_mem_cfg,
    )
```

**Detailed Breakdown**:

1. **Scale Factor**: `self.scale = self.head_dim**-0.5 = 1/sqrt(128) ≈ 0.0884`

2. **Attention Computation**:

   a. **Q @ K^T**:
      - Q: `[1, 8, 8, 128]` (8 groups, 8 users, 128 dim)
      - K^T: `[32, 1, 128, current_pos+1]` (32 users, 1 head, 128 dim, sequence)
      - Result: `[1, 8, 8, current_pos+1]` (attention scores)

   b. **Scale**:
      - `scores = scores * scale`
      - Prevents large dot products

   c. **Causal Mask**:
      - Set future positions to -inf
      - Each user only attends to their own history
      - `mask[i, j] = -inf if j > current_pos[i]`

   d. **Softmax**:
      - `attn_weights = softmax(scores, dim=-1)`
      - Attention weights sum to 1 along sequence dimension

   e. **Apply to V**:
      - `attn_output = attn_weights @ V`
      - `[1, 8, 8, current_pos+1] @ [32, 1, current_pos+1, 128]`
      - Result: `[1, 8, 8, 128]`

3. **Program Config**: `SDPA_DECODE_PROGCFG`
   - Decode-optimized SDPA configuration
   - Handles variable sequence lengths per user

4. **Compute Config**: `SDPA_DECODE_COMPUTE_PROGCFG`
   - Compute kernel configuration for SDPA
   - Optimized for decode throughput

5. **Memory Config**: `SCORES_BATCHED_MM_OUTPUT_MEMCFG(batch_size)`
   - Dynamic memory config based on batch size
   - Sharded output for efficient next operation

6. **GQA (Grouped Query Attention)**:
   - Q has 8 heads per device
   - K/V have 1 head per device
   - K/V broadcasted to match Q heads
   - Reduces KV cache size by 8x

**Output Shape**: `[1, 8, 8, 128]`
- 1: batch dimension
- 8: device groups
- 8: users per group
- 128: head dimension

**Memory Deallocation**:

```python
ttnn.deallocate(q_heads_1BQD)  # Line 531
```

### Step 7: All-Gather Concat (Lines 533-542)

**Operation**: Gather attention outputs and concatenate heads

```python
attn_output_cat = self.tt_ccl.all_gather_concat(  # [1, 1, 32, 1024]
    attn_output_1G4D_sharded,      # [1, 8, 8, 128]
    dim=1,                         # Gather on head dimension
    cluster_axis=1,                # Column axis
    num_links=self.model_config["GALAXY_NUM_LINKS"],  # 3 links
    memory_config=self.model_config["SHARDED_ATTN_WO_INPUT_RING_MEMCFG"],
    num_heads=self.n_local_heads,  # 8 heads per device
)
```

**Detailed Breakdown**:

1. **Operation**: `all_gather_concat`
   - Location: `llama_ccl.py`, lines 1129-1147
   - Calls: `ttnn.experimental.all_gather_concat`

2. **Internal Operations**:

   a. **All-Gather**:
      - Gathers from 4 devices in a row
      - Dim 1: head dimension
      - Ring topology with 3 links

   b. **Concatenate Heads**:
      - Input: `[1, 8, 8, 128]` per device
      - After gather: `[1, 8, 32, 128]` (32 users across all devices)
      - After concat: `[1, 1, 32, 1024]` (8 heads * 128 dim = 1024)

3. **Intermediate Tensor**: `all_gather_concat_inter_tensor`
   - Pre-allocated intermediate buffer
   - Location: `llama_ccl.py`, lines 131-174
   - Shape: `[8, 128, 32, 128]`
   - Used for efficient all-gather-concat operation

4. **Memory Config**: `SHARDED_ATTN_WO_INPUT_RING_MEMCFG`
   - Ring-optimized memory layout
   - Prepared for output projection

5. **Semaphore**: Uses gather semaphore for synchronization

**Output Shape**: `[1, 1, 32, 1024]`
- 1: batch dimension
- 1: sequence dimension (always 1 for decode)
- 32: users
- 1024: concatenated heads (8 heads * 128 dim)

**Memory Deallocation**:

```python
ttnn.deallocate(attn_output_1G4D_sharded)  # Line 541
```

### Step 8: Output Projection (Lines 545-554)

**Operation**: Project concatenated attention output back to hidden dimension

```python
dense_out_ttnn = ttnn.matmul(  # [1, 1, 32, 1280]
    attn_output_cat,               # [1, 1, 32, 1024]
    self.wo,                       # [1, 1, 1024, 1280]
    program_config=self.model_config["WO_DECODE_RING_PROGCFG"],
    memory_config=self.model_config["SHARDED_WO_OUT_RING_MEMCFG"],
    compute_kernel_config=self.compute_kernel_config_hifi2,
    global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
    dtype=ttnn.bfloat8_b,
    sub_device_id=self.prefetcher_setup.worker_sub_device_id,
)
```

**Detailed Breakdown**:

1. **Weight Shape**: `[1, 1, 1024, 1280]` per device
   - Input: 1024 (8 heads * 128 dim) per device
   - Output: 1280 per device (8192 / 8 devices = 1024, with padding = 1280)

2. **Weight Sharding**: `dims=(2, 3)` for TG
   - Row-wise sharding
   - Each device has different rows

3. **Program Config**: `WO_DECODE_RING_PROGCFG`
   - Ring-optimized output projection
   - Configured for decode mode

4. **Memory Config**: `SHARDED_WO_OUT_RING_MEMCFG`
   - Ring-optimized output memory layout

5. **Compute Kernel**: `compute_kernel_config_hifi2`
   - High-fidelity compute

6. **Prefetcher**: Weight prefetched to circular buffer

7. **Data Type**: `ttnn.bfloat8_b`
   - 8-bit bfloat for reduced memory bandwidth
   - Acceptable precision loss for this operation

**Intermediate Output Shape**: `[1, 1, 32, 1280]` per device (before all-reduce)

### Step 9: All-Reduce (Lines 556-563)

**Operation**: Reduce partial results across devices

```python
dense_out_reduced = self.tt_ccl.line_all_reduce(  # [1, 1, 32, 1280]
    dense_out_ttnn,                # [1, 1, 32, 1280] per device
    cluster_axis=0,                # Row axis
    num_links=self.model_config["GALAXY_NUM_LINKS"],  # 3 links
    memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
    use_optimal_ccl_for_llama=True,
)
```

**Detailed Breakdown**:

1. **Why All-Reduce**:
   - Weight is row-wise sharded
   - Each device computes partial result
   - Need to sum all partial results
   - All devices get same final result

2. **Operation**: `line_all_reduce`
   - Location: `llama_ccl.py`, lines 644-721
   - Cluster axis: 0 (row dimension)
   - Ring topology with 3 links

3. **Internal Operations** (decode mode):
   - Uses `ttnn.experimental.all_reduce_async`
   - Persistent buffer: `persistent_buffers[cluster_axis]`
   - Ring-based all-reduce
   - Optimized for Llama with `use_optimal_ccl_for_llama=True`

4. **Memory Config**: `DECODE_RESIDUAL_MEMCFG`
   - Memory layout for residual connection
   - Ready for addition with input

5. **Semaphore**: Uses gather semaphore for synchronization

**Output Shape**: `[1, 1, 32, 1280]`
- Full hidden dimension output
- Same on all devices (replicated)

**Memory Deallocation**:

```python
ttnn.deallocate(dense_out_ttnn)  # Line 563
return dense_out_reduced          # Line 567
```

---

## Decode MLP Forward Pass

### Overview

**File**: `llama_mlp.py`
**Method**: `forward()`  (decode mode)
**Lines**: 116-195

### Input Parameters

```python
def forward(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
    # x: [1, 1, 32, 1280]
    # mode: "decode"
```

### Step 1: Double MatMul with Reduce-Scatter (Lines 120-139)

**Operation**: Fused W1 and W3 matmuls with reduce-scatter

```python
pc_1_3 = self.model_config["FF1_3_TG_RING_PROGCFG"]
pc_2 = self.model_config["FF2_TG_RING_PROGCFG"]

w1_out_reduced, w3_out = self.tt_ccl.double_matmul_line_reduce_scatter(
    x,                             # [1, 1, 32, 1280]
    self.w1,                       # [1, 1, 1280, 1792]
    self.w3,                       # [1, 1, 1280, 1792]
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

**Detailed Breakdown**:

1. **Weight Shapes**:
   - W1 (gate_proj): `[1, 1, 1280, 1792]` per device
   - W3 (up_proj): `[1, 1, 1280, 1792]` per device
   - Full hidden_dim: 14336
   - Per device: 14336 / 8 = 1792

2. **Weight Sharding**: `dims=(-1, -2)` (column-wise)
   - Dimension -1 (output): Sharded across devices
   - Dimension -2 (input): Replicated

3. **Operation**: `double_matmul_line_reduce_scatter`
   - Location: `llama_ccl.py`, lines 763-812
   - Calls: `ttnn.experimental.llama_rs_matmul`

4. **Internal Operations**:

   a. **Double MatMul**:
      - Computes `w1_out = x @ w1` and `w3_out = x @ w3` simultaneously
      - Shares input data loading
      - More efficient than separate matmuls
      - Both outputs: `[1, 1, 32, 1792]` per device

   b. **Reduce-Scatter on W1**:
      - Only W1 is reduced immediately
      - Cluster axis: 1 (column dimension)
      - Reduces partial results across 4 devices in a row
      - Output: `[1, 1, 32, 1792]` (reduced)

   c. **W3 Output**:
      - W3 output is NOT reduced yet
      - Will be reduced separately (next step)
      - Output: `[1, 1, 32, 1792]` (not reduced)

5. **Persistent Buffer**: `reduce_scatter_buffers[cluster_axis][reduce_scatter_buffer_idx]`
   - Pre-allocated intermediate buffer
   - Location: `llama_ccl.py`, lines 383-412
   - Shape: `[8, 4, 32, 512 * num_cores]`
   - 512 = 4 devices * 4 pages per packet * 32 tile_width
   - Memory: `REDUCE_SCATTER_INTERIM_MEMCFG`

6. **Program Config**: `FF1_3_TG_RING_PROGCFG`
   - Ring-optimized program configuration
   - For double matmul + reduce-scatter

7. **Compute Kernel**:
   - `compute_kernel_config_lofi`: If 4-bit MLP weights
   - `compute_kernel_config_hifi2`: If 8-bit MLP weights

8. **Prefetcher**: Both W1 and W3 prefetched to circular buffer

9. **Why Double MatMul**:
   - Both use same input
   - Share input data loading
   - Reduces memory bandwidth by ~50%
   - Single kernel launch

**Output Shapes**:
- `w1_out_reduced`: `[1, 1, 32, 1792]` (reduced)
- `w3_out`: `[1, 1, 32, 1792]` (not reduced yet)

**Memory Deallocation**:

```python
ttnn.deallocate(x)  # Line 141
```

### Step 2: Reduce-Scatter W3 (Lines 143-150)

**Operation**: Reduce-scatter W3 output

```python
w3_out_reduced = self.tt_ccl.line_reduce_scatter(
    w3_out,                        # [1, 1, 32, 1792]
    cluster_axis=1,
    num_links=self.model_config["GALAXY_NUM_LINKS"],
    memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
    use_noc1_only=False,
)
```

**Detailed Breakdown**:

1. **Operation**: `line_reduce_scatter`
   - Location: `llama_ccl.py`, lines 901-968
   - Decode mode uses `ttnn.experimental.llama_reduce_scatter`

2. **Internal Operations**:
   - Persistent buffer: `reduce_scatter_buffers[cluster_axis][reduce_scatter_buffer_idx]`
   - Ring-based reduce-scatter
   - Cluster axis: 1 (column dimension)
   - Num links: 3

3. **Why Separate**:
   - `double_matmul_line_reduce_scatter` only reduces W1
   - W3 needs separate reduce-scatter
   - Allows pipeline optimization

**Output Shape**: `[1, 1, 32, 1792]` (reduced)

**Memory Deallocation**:

```python
ttnn.deallocate(w3_out)  # Line 150
```

### Step 3: Element-wise Multiply with SiLU (Lines 152-158)

**Operation**: Apply SiLU activation and multiply

```python
ff1ff3 = ttnn.mul(
    w1_out_reduced,                # [1, 1, 32, 1792]
    w3_out_reduced,                # [1, 1, 32, 1792]
    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
    dtype=ttnn.bfloat8_b,
    memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
)
```

**Detailed Breakdown**:

1. **SiLU Activation**:
   - Applied to `w1_out_reduced` during multiply
   - Formula: `SiLU(x) = x * sigmoid(x)`
   - Fused with multiply for efficiency

2. **Operation**:
   - `ff1ff3 = SiLU(w1_out_reduced) * w3_out_reduced`
   - Element-wise operation
   - Fused activation and multiply

3. **Mathematical Formula**:
   ```
   ff1ff3[i] = (w1_out_reduced[i] * sigmoid(w1_out_reduced[i])) * w3_out_reduced[i]
   ```

4. **Data Type**: `ttnn.bfloat8_b`
   - Reduced memory bandwidth

5. **Memory Config**: `REDUCE_SCATTER_OUT_MEMCFG`
   - Same as input memory config

**Output Shape**: `[1, 1, 32, 1792]`

**Memory Deallocation**:

```python
ttnn.deallocate(w3_out_reduced)  # Line 160
ttnn.deallocate(w1_out_reduced)  # Line 161
```

### Step 4: All-Gather (Lines 163-171)

**Operation**: Gather distributed results

```python
w2_in = self.tt_ccl.line_all_gather(
    ff1ff3,                        # [1, 1, 32, 1792]
    dim=3,
    cluster_axis=1,
    num_links=self.model_config["GALAXY_NUM_LINKS"],
    memory_config=self.model_config["FF2_IN_RING_MEMCFG"],
    buffer_key="BINARY_MUL",
    use_optimal_ccl_for_llama=False if mode == "prefill" else True,
)
```

**Detailed Breakdown**:

1. **Why All-Gather**:
   - ff1ff3 is column-wise sharded
   - W2 needs full hidden dimension
   - Gather all shards

2. **Operation**: `line_all_gather`
   - Location: `llama_ccl.py`, lines 1009-1085
   - Decode mode uses ring topology

3. **Persistent Buffer**: `buffer_key="BINARY_MUL"`
   - Location: `llama_ccl.py`, lines 271-290
   - Pre-allocated buffer: `[1, 1, 32, 3584]` (for Llama)
   - Or `[1, 1, 32, 3200]` (for Qwen)
   - Memory: `FF2_IN_RING_MEMCFG`

4. **Internal Operations**:
   - Uses `ttnn.experimental.all_gather_async`
   - Ring topology
   - Cluster axis: 1
   - Num links: 3
   - Semaphore: gather semaphore

5. **Optimization**: `use_optimal_ccl_for_llama=True` for decode
   - Llama-specific optimizations

**Output Shape**: `[1, 1, 32, 14336]` (full hidden dimension)
- For Llama: 14336 = 1792 * 8 devices
- For Qwen: 12800 = 1600 * 8 devices

**Memory Deallocation**:

```python
ttnn.deallocate(ff1ff3)  # Line 173
```

### Step 5: W2 MatMul (Lines 175-185)

**Operation**: Down projection

```python
w2_out = ttnn.linear(
    w2_in,                         # [1, 1, 32, 14336]
    self.w2,                       # [1, 1, 1792, 1280]
    compute_kernel_config=self.args.compute_kernel_config_hifi2,
    dtype=ttnn.bfloat8_b,
    program_config=pc_2,
    memory_config=self.model_config["FF2_OUT_RING_MEMCFG"],
    core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
    global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
    sub_device_id=self.prefetcher_setup.worker_sub_device_id if mode == "decode" else None,
)
```

**Detailed Breakdown**:

1. **Weight Shape**: `[1, 1, 1792, 1280]` per device
   - Input: 14336 / 8 = 1792 per device
   - Output: 1280 per device

2. **Weight Sharding**: `dims=(-2, -1)` (row-wise)
   - Each device has different rows
   - Requires all-reduce after matmul

3. **Program Config**: `FF2_TG_RING_PROGCFG`
   - Ring-optimized program configuration

4. **Compute Kernel**: `compute_kernel_config_hifi2`
   - High-fidelity compute

5. **Prefetcher**: W2 prefetched to circular buffer

6. **Data Type**: `ttnn.bfloat8_b`

**Output Shape**: `[1, 1, 32, 1280]` per device (before all-reduce)

### Step 6: All-Reduce (Lines 186-193)

**Operation**: Reduce partial results

```python
w2_out_reduced = self.tt_ccl.line_all_reduce(
    w2_out,                        # [1, 1, 32, 1280]
    cluster_axis=0,
    num_links=self.model_config["GALAXY_NUM_LINKS"],
    memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
    use_optimal_ccl_for_llama=True,
)
```

**Detailed Breakdown**:

1. **Why All-Reduce**:
   - W2 is row-wise sharded
   - Each device computes partial result
   - Need to sum all partial results

2. **Cluster Axis**: 0 (row dimension)
   - Reduces across 8 device rows

3. **Memory Config**: `DECODE_RESIDUAL_MEMCFG`
   - Ready for residual connection

**Output Shape**: `[1, 1, 32, 1280]` (reduced, replicated)

**Memory Deallocation**:

```python
ttnn.deallocate(w2_out)  # Line 193
return w2_out_reduced    # Line 195
```

---

## Prefill Attention Forward Pass

### Overview

**File**: `llama_attention.py`
**Method**: `forward_prefill()`
**Lines**: 569-824

### Key Differences from Decode

1. **Full Sequence**: Processes entire input sequence at once
2. **No Prefetcher**: Prefetcher not used for prefill
3. **DRAM Memory**: Uses DRAM for large sequences
4. **All-Reduce QKV**: Uses all-reduce instead of reduce-scatter
5. **Ring Distributed SDPA**: For sequences > 1k tokens
6. **Chunking**: Long sequences split into chunks

### Input Parameters

```python
def forward_prefill(
    self,
    x_11SH,                        # [1, 1, seq_len, hidden_dim]
    rot_mats,                      # Rotation matrices
    user_id: int = 0,              # User ID for KV cache indexing
    page_table=None,               # Optional: for paged attention
    chunk_page_table=None,         # Optional: for chunked paged attention
    chunk_start_idx=None,          # Optional: chunk start index
    kv_cache=None,                 # Optional: external KV cache
    batch_size=1,                  # Batch size (number of sequences)
):
```

### Step 1: Input Reshaping (Lines 580-592)

**Operation**: Reshape input for processing

```python
if batch_size > 1:
    x_11SH = ttnn.reshape(x_11SH, [1, 1, x_11SH.shape[-2] * x_11SH.shape[-3] * x_11SH.shape[-4], -1])

seq_len = x_11SH.shape[-2]
assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"

# Reshaping long sequence to matmul fit on device
if seq_len > 2048:
    x_11SH = ttnn.reshape(x_11SH, [1, seq_len // 2048, 2048, -1])
```

**Detailed Breakdown**:

1. **Batch Reshaping**:
   - If batch_size > 1, flatten batch into sequence dimension
   - Input: `[batch_size, 1, seq_len, hidden_dim]`
   - Output: `[1, 1, batch_size * seq_len, hidden_dim]`

2. **Sequence Length Check**:
   - Must be divisible by 128 (tile size * 4)
   - Ensures proper tiling

3. **Long Sequence Reshaping**:
   - If seq_len > 2048, split into chunks of 2048
   - Input: `[1, 1, seq_len, hidden_dim]`
   - Output: `[1, seq_len // 2048, 2048, hidden_dim]`
   - Required for matmul to fit on device

### Step 2: QKV Projection (Lines 593-611)

**Operation**: Project input to Q, K, V

```python
xqkv = ttnn.linear(
    x_11SH,                        # [1, 1, seq_len, hidden_dim] or chunked
    self.wqkv_interleaved,         # Interleaved (DRAM) weight
    dtype=self.ccl_dtype if self.TG else ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    compute_kernel_config=self.compute_kernel_config_hifi2,
    program_config=self.model_config["XQKV_PREFILL_PROGCFG"](seq_len),
)

ttnn.deallocate(x_11SH)

xqkv_fused = self.tt_ccl.line_all_reduce(
    xqkv,
    cluster_axis=1,
    num_links=3,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    buffer_key="QKV",
)
ttnn.deallocate(xqkv)
```

**Detailed Breakdown**:

1. **Weight**: `wqkv_interleaved`
   - Uses interleaved (DRAM) weights for prefill
   - Not sharded weight like decode

2. **Program Config**: `XQKV_PREFILL_PROGCFG(seq_len)`
   - Dynamic configuration based on sequence length
   - Different configs for different sequence lengths

3. **Output**: `xqkv` - `[1, 1, seq_len, qkv_size]` per device

4. **All-Reduce** (not reduce-scatter):
   - Why: Prefill doesn't use reduce-scatter for QKV
   - Uses `line_all_reduce` instead
   - Reduces across cluster_axis=1
   - Result: Full QKV on all devices

5. **Persistent Buffer**: `buffer_key="QKV"`
   - Location: `llama_ccl.py`, lines 463-477 (prefill buffers)
   - Different buffers for different sequence lengths

6. **Reshape Back** (if chunked):
   ```python
   if seq_len > 2048:
       xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])
   ```

7. **Batch Reshape** (if batch_size > 1):
   ```python
   if batch_size > 1:
       xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len // batch_size, -1])
   ```

**Output Shape**: `[1, 1, seq_len, qkv_size]` (or batched)

### Step 3: Create QKV Heads (Lines 620-630)

**Operation**: Split QKV into separate heads

```python
(
    q_heads_1QSD_pre_rot,          # [batch, n_heads, seq_len, head_dim]
    k_heads_1KSD_pre_rot,          # [batch, n_kv_heads, seq_len, head_dim]
    v_heads_1VSD,                  # [batch, n_kv_heads, seq_len, head_dim]
) = ttnn.experimental.nlp_create_qkv_heads(
    xqkv_fused,
    num_heads=self.n_local_heads,      # 8 heads per device
    num_kv_heads=self.n_local_kv_heads,  # 1 KV head per device
    transpose_k_heads=False,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

**Detailed Breakdown**:

1. **Operation**: Standard create heads (not fused with reduce-scatter)
   - Splits QKV into Q, K, V
   - Reshapes to head format

2. **Output Shapes**:
   - Q: `[batch, 8, seq_len, 128]`
   - K: `[batch, 1, seq_len, 128]`
   - V: `[batch, 1, seq_len, 128]`

3. **Memory**: DRAM (not sharded like decode)

### Step 4: RoPE Application (Lines 638-671)

**Operation**: Apply rotary embeddings to Q and K

```python
# Ensure bfloat16 for RoPE
if q_heads_1QSD_pre_rot.dtype != ttnn.bfloat16:
    q_heads_1QSD_pre_rot = ttnn.typecast(q_heads_1QSD_pre_rot, dtype=ttnn.bfloat16)

# QK norm if needed (Qwen)
if self.qk_norm:
    q_heads_1QSD_pre_rot = self.q_norm(q_heads_1QSD_pre_rot, mode="prefill")

# Apply RoPE to Q
q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
    q_heads_1QSD_pre_rot,
    rot_mats[0],                   # cos_matrix
    rot_mats[1],                   # sin_matrix
    self.transformation_mats["prefill"],
    is_decode_mode=False,
)
ttnn.deallocate(q_heads_1QSD_pre_rot)

# Apply RoPE to K (similar process)
...
```

**Detailed Breakdown**:

1. **Type Casting**: RoPE requires bfloat16 inputs

2. **QK Norm**: Optional for Qwen models

3. **RoPE Operation**: `rotary_embedding_llama`
   - Non-fused version (separate for Q and K)
   - `is_decode_mode=False` for prefill
   - Uses prefill transformation matrices

4. **Full Sequence**: Applies RoPE to entire sequence at once

### Step 5: Fill KV Cache (Lines 673-719)

**Operation**: Fill KV cache with entire sequence

```python
# Typecast to bfloat8_b
k_heads_1KSD_8b = ttnn.typecast(k_heads_1KSD, dtype=ttnn.bfloat8_b)
ttnn.deallocate(k_heads_1KSD)
k_fill = k_heads_1KSD_8b

v_heads_1VSD_8b = ttnn.typecast(v_heads_1VSD, dtype=ttnn.bfloat8_b)
ttnn.deallocate(v_heads_1VSD)
v_fill = v_heads_1VSD_8b

if batch_size > 1:
    k_fill = ttnn.reshape(k_fill, [1, 1, seq_len, -1])
    v_fill = ttnn.reshape(v_fill, [1, 1, seq_len, -1])

if self.TG and not page_table:
    k_fill = self.prefill_prepare_tensor_for_kv_cache(k_fill, user_id)
    v_fill = self.prefill_prepare_tensor_for_kv_cache(v_fill, user_id)

if page_table:
    # Paged fill
    if isinstance(user_id, int):
        user_id = ttnn.from_torch(
            torch.tensor([user_id], dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.int32,
            ...
        )
    ttnn.experimental.paged_fill_cache(keys_BKSD, k_fill, page_table, batch_idx_tensor=user_id)
    ttnn.experimental.paged_fill_cache(values_BKSD, v_fill, page_table, batch_idx_tensor=user_id)
else:
    # Standard fill
    ttnn.fill_cache(keys_BKSD, k_fill, user_id % self.batch_size_per_device_group)
    ttnn.fill_cache(values_BKSD, v_fill, user_id % self.batch_size_per_device_group)
```

**Detailed Breakdown**:

1. **Type Casting**: Convert to bfloat8_b for cache storage

2. **Batch Reshaping**: Flatten batch dimension if needed

3. **TG-specific Processing**: `prefill_prepare_tensor_for_kv_cache`
   - Location: lines 853-864
   - Selects appropriate device column
   - For Galaxy with device groups

4. **Paged Fill**:
   - Uses `paged_fill_cache` if page_table provided
   - Fills cache using page table mapping

5. **Standard Fill**:
   - Uses `fill_cache` if no page table
   - Fills cache at user_id position

### Step 6: Scaled Dot-Product Attention (Lines 722-750)

**Operation**: Compute attention

```python
q_heads_1QSD_8b = ttnn.typecast(q_heads_1QSD, dtype=ttnn.bfloat8_b)
ttnn.deallocate(q_heads_1QSD)

# Ring distributed SDPA for long sequences
ring_distributed_sdpa = seq_len > 1024 and batch_size == 1

if ring_distributed_sdpa:
    # Ring attention
    attn_output_1QSD = ttnn.transformer.ring_distributed_scaled_dot_product_attention(
        q_heads_1QSD_8b,
        k_heads_1KSD_8b,
        v_heads_1VSD_8b,
        ring_size=4,               # 4 devices per row
        scale=self.scale,
        compute_kernel_config=self.compute_kernel_config_hifi4,
        program_config=self.model_config["SDPA_PROGCFG"](seq_len),
    )
else:
    # Standard SDPA
    attn_output_1QSD = ttnn.transformer.scaled_dot_product_attention(
        q_heads_1QSD_8b,
        k_heads_1KSD_8b,
        v_heads_1VSD_8b,
        is_causal=True,
        scale=self.scale,
        compute_kernel_config=self.compute_kernel_config_hifi4,
        program_config=self.model_config["SDPA_PROGCFG"](seq_len),
    )
```

**Detailed Breakdown**:

1. **Type Casting**: Convert Q to bfloat8_b

2. **Ring Distributed SDPA** (for seq_len > 1024):
   - Splits sequence into 8 chunks (ring_size=4, 2 chunks per device)
   - Each device computes chunk i and chunk (ring_size - i - 1)
   - Uses ring communication to exchange partial results
   - More efficient for long sequences
   - Compute kernel: HIFI4 (highest fidelity)

3. **Standard SDPA** (for seq_len <= 1024):
   - Standard causal attention
   - `is_causal=True`: Causal masking
   - Compute kernel: HIFI4

4. **Program Config**: `SDPA_PROGCFG(seq_len)`
   - Dynamic configuration based on sequence length

**Output Shape**: `[batch, n_heads, seq_len, head_dim]`

This completes the first major sections. The document is getting very long. Would you like me to:

1. Continue with the remaining sections (Prefill MLP, CCL Deep Dive, Prefetcher Deep Dive, Memory Management)?
2. Create these as separate files?
3. Focus on specific sections you're most interested in?

Given the level of detail needed, I'll create multiple comprehensive documents to cover everything thoroughly.
