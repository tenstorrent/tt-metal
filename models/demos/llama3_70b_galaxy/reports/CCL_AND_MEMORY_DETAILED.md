# Llama3-70B Galaxy: CCL Operations and Memory Management

## Overview

This document details the Collective Communication Library (CCL) operations and memory management strategies used in the Llama3-70B Galaxy implementation.

## CCL Operations

### Ring Topology

**Architecture**: 32 devices arranged in a ring
- Device 0 → Device 1 → Device 2 → ... → Device 31 → Device 0

**Benefits**:
- **Bandwidth Efficient**: Uses all links simultaneously
- **Scalable**: Works well for large device counts
- **Low Latency**: O(num_devices) steps

**Cluster Shape**: (8, 4) = 8 rows × 4 columns = 32 devices

---

### Line Reduce-Scatter

**Operation**: `line_reduce_scatter()`

**Purpose**: Reduce partial results and scatter to appropriate devices

**Step-by-Step**:

```
Step 1: Input Preparation
├─ Input: tensor [sharded]
│   └─ Partial results per device
│   └─ Sharded along cluster_axis
│
Step 2: Ring Communication
├─ Pattern: Ring topology
│   ├─ Each device sends its partial result to next device
│   ├─ Each device accumulates received results
│   └─ After num_devices steps, all devices have reduced result
│
Step 3: Scatter Operation
├─ Scatters reduced result to appropriate devices
│   └─ Each device gets its portion of the result
│
Step 4: Output
└─ Output: tensor [sharded]
    └─ Reduced and scattered result
```

**Parameters**:
- `cluster_axis`: Which dimension to reduce/scatter along (0=row, 1=column)
- `num_links`: Number of links to use (typically 3 for Galaxy)
- `memory_config`: Output memory configuration
- `buffer_key`: Key for persistent buffer reuse

**Use Cases**:
- W1/W3 MLP outputs (reduce partial results)
- QKV creation (reduce-scatter heads)

**Code Example**:
```python
w1_out_reduced = self.tt_ccl.line_reduce_scatter(
    w1_out,
    cluster_axis=1,  # Column dimension
    num_links=3,
    memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
    buffer_key="FF1",
    dim=3,  # Feature dimension
)
```

---

### Line All-Gather

**Operation**: `line_all_gather()`

**Purpose**: Gather distributed results from all devices

**Step-by-Step**:

```
Step 1: Input Preparation
├─ Input: tensor [sharded]
│   └─ Partial results per device
│   └─ Sharded along cluster_axis
│
Step 2: Ring Communication
├─ Pattern: Ring topology
│   ├─ Each device sends its partial result to next device
│   ├─ Each device accumulates received results
│   └─ After num_devices steps, all devices have full result
│
Step 3: Output
└─ Output: tensor [replicated or sharded]
    └─ Full result gathered from all devices
```

**Parameters**:
- `dim`: Dimension to gather along
- `cluster_axis`: Which cluster axis to gather along
- `num_links`: Number of links (typically 3)
- `memory_config`: Output memory configuration
- `buffer_key`: Key for persistent buffer reuse

**Use Cases**:
- W2 MLP input (gather full hidden dimension)
- Attention output concat (gather heads)

**Code Example**:
```python
w2_in = self.tt_ccl.line_all_gather(
    ff1ff3,
    dim=3,  # Feature dimension
    cluster_axis=1,  # Column dimension
    num_links=3,
    memory_config=self.model_config["FF2_IN_RING_MEMCFG"],
    buffer_key="BINARY_MUL",
)
```

---

### Line All-Reduce

**Operation**: `line_all_reduce()`

**Purpose**: Reduce results across all devices (sum operation)

**Step-by-Step**:

```
Step 1: Input Preparation
├─ Input: tensor [sharded]
│   └─ Partial results per device
│
Step 2: Ring Communication
├─ Pattern: Ring topology
│   ├─ Each device sends its partial result to next device
│   ├─ Each device accumulates received results
│   └─ After num_devices steps, all devices have sum of all results
│
Step 3: Output
└─ Output: tensor [replicated]
    └─ Sum of all partial results (same on all devices)
```

**Parameters**:
- `cluster_axis`: Which cluster axis to reduce along
- `num_links`: Number of links (typically 3)
- `memory_config`: Output memory configuration
- `use_optimal_ccl_for_llama`: Use optimized CCL for Llama

**Use Cases**:
- W2 MLP output (sum partial results)
- Attention output projection (sum partial results)

**Code Example**:
```python
w2_out_reduced = self.tt_ccl.line_all_reduce(
    w2_out,
    cluster_axis=0,  # Row dimension
    num_links=3,
    memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
    use_optimal_ccl_for_llama=True,
)
```

---

### Double MatMul Line Reduce-Scatter

**Operation**: `double_matmul_line_reduce_scatter()`

**Purpose**: Compute two matmuls simultaneously and reduce-scatter one result

**Step-by-Step**:

```
Step 1: Input Preparation
├─ Input: x [sharded]
├─ Weight 1: w1 [sharded]
├─ Weight 2: w3 [sharded]
│
Step 2: Fused MatMul Execution
├─ Computes x @ w1 and x @ w3 simultaneously
│   ├─ Shares input data loading
│   └─ More efficient than separate matmuls
│
Step 3: Reduce-Scatter
├─ Reduces w1_out across devices
├─ Scatters to appropriate devices
└─ w3_out kept as-is (reduced separately)
│
Step 4: Output
├─ w1_out_reduced: [reduced and scattered]
└─ w3_out: [not yet reduced]
```

**Benefits**:
- **Efficiency**: Shares input data loading
- **Memory**: Reduces memory bandwidth
- **Performance**: Faster than separate matmuls

**Code Example**:
```python
w1_out_reduced, w3_out = self.tt_ccl.double_matmul_line_reduce_scatter(
    x,
    self.w1,
    self.w3,
    cluster_axis=1,
    num_links=3,
    RS_memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
    compute_kernel_config=self.args.compute_kernel_config_lofi,
    dtype=ttnn.bfloat8_b,
    program_config=pc_1_3,
    memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
)
```

---

### Llama RS Create Heads

**Operation**: `llama_rs_create_heads()`

**Purpose**: Create QKV heads with reduce-scatter for ring topology

**Step-by-Step**:

```
Step 1: Input Preparation
├─ Input: xqkv_fused_sharded [sharded]
│   └─ QKV concatenated
│
Step 2: Reshape and Split
├─ Reshape to separate Q, K, V
│   ├─ Q: [batch, n_heads, seq_len, head_dim]
│   ├─ K: [batch, n_kv_heads, seq_len, head_dim]
│   └─ V: [batch, n_kv_heads, seq_len, head_dim]
│
Step 3: Reduce-Scatter
├─ Reduces partial heads across devices
├─ Scatters to appropriate devices
└─ Uses ring topology
│
Step 4: Output
├─ q_heads: [reduced and scattered]
├─ k_heads: [reduced and scattered]
└─ v_heads: [reduced and scattered]
```

**Code Example**:
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

### All-Gather Concat

**Operation**: `all_gather_concat()`

**Purpose**: Gather attention heads and concatenate

**Step-by-Step**:

```
Step 1: Input Preparation
├─ Input: attn_output_1G4D_sharded [sharded]
│   └─ Attention output per head
│
Step 2: All-Gather
├─ Gathers heads from all devices
│   └─ Uses ring topology
│
Step 3: Concatenate
├─ Concatenates heads along feature dimension
│   └─ Creates full attention output
│
Step 4: Output
└─ attn_output_cat: [gathered and concatenated]
    └─ Full attention output
```

**Code Example**:
```python
attn_output_cat = self.tt_ccl.all_gather_concat(
    attn_output_1G4D_sharded,
    dim=1,  # Head dimension
    cluster_axis=1,
    num_links=self.model_config["GALAXY_NUM_LINKS"],
    memory_config=self.model_config["SHARDED_ATTN_WO_INPUT_RING_MEMCFG"],
    num_heads=self.n_local_heads,
)
```

---

## Memory Management

### Memory Configurations

**Types**:
1. **DRAM_MEMORY_CONFIG**: DRAM storage (for large tensors)
2. **L1 Memory Configs**: L1 cache (for frequently accessed data)
3. **Sharded Memory Configs**: Distributed across devices
4. **Ring Memory Configs**: Optimized for ring topology

**Key Memory Configs**:

```
SHARDED_QKV_RING_MEMCFG:
├─ Purpose: QKV weight storage
├─ Layout: Sharded across devices
└─ Optimized for ring topology

SHARDED_WO_RING_MEMCFG:
├─ Purpose: Output projection weight storage
├─ Layout: Row-wise sharded
└─ Optimized for ring topology

REDUCE_SCATTER_OUT_MEMCFG:
├─ Purpose: Reduce-scatter output
├─ Layout: Sharded
└─ Optimized for ring communication

FF2_IN_RING_MEMCFG:
├─ Purpose: W2 MLP input
├─ Layout: Gathered (full hidden dimension)
└─ Optimized for ring topology

DECODE_RESIDUAL_MEMCFG:
├─ Purpose: Residual connection storage
├─ Layout: Replicated or sharded
└─ Optimized for decode operations
```

---

### Persistent Buffers

**Purpose**: Pre-allocated buffers for CCL operations

**Benefits**:
- **Performance**: Avoids allocation overhead
- **Memory Efficiency**: Reuses buffers
- **Predictability**: Known memory usage

**Buffer Types**:

```
SDPA Buffer:
├─ Shape: (1, 32, 32, 128)
├─ Purpose: Attention scores
└─ Memory: DRAM_MEMORY_CONFIG

LAYERNORM Buffer:
├─ Shape: (1, 1, 32, 128)
├─ Purpose: LayerNorm statistics
└─ Memory: Width-sharded L1

SAMPLING_VALUES Buffer:
├─ Shape: (1, 1, 32, max_top_k * cluster_shape[0])
├─ Purpose: Sampling values
└─ Memory: DRAM_MEMORY_CONFIG

SAMPLING_INDICES Buffer:
├─ Shape: (1, 1, 32, max_top_k * cluster_shape[0])
├─ Purpose: Sampling indices
└─ Memory: DRAM_MEMORY_CONFIG

BINARY_MUL Buffer:
├─ Shape: (1, 1, 32, 3584) or (1, 1, 32, 3200) for Qwen
├─ Purpose: Element-wise multiply output
└─ Memory: FF2_IN_RING_MEMCFG
```

**Code Example**:
```python
# Get persistent buffers
persistent_buffers = self.get_all_gather_buffers = self.get_all_gather_buffers()

# Use buffer in operation
w2_in = self.tt_ccl.line_all_gather(
    ff1ff3,
    buffer_key="BINARY_MUL",  # Uses persistent buffer
    ...
)
```

---

### Buffer Management

**Double Buffering**:
- Two sets of buffers for each operation
- Allows overlap of computation and communication
- Reduces stalls

**Buffer Cycling**:
```python
# Get and cycle buffer index
buffer_idx = self.reduce_scatter_buffer_idx[cluster_axis]
self.reduce_scatter_buffer_idx[cluster_axis] = (buffer_idx + 1) % self.num_cbs

# Use buffer
buffer = self.persistent_buffers[cluster_axis][buffer_idx]
```

**Semaphore Management**:
- Semaphores for synchronization
- Double buffered semaphores
- Prevents race conditions

---

### Memory Layouts

**TILE_LAYOUT**:
- Tiled memory layout
- Optimized for compute operations
- Used for weights and activations

**ROW_MAJOR_LAYOUT**:
- Row-major memory layout
- Used for intermediate results
- Easier to manipulate

**Sharding Strategies**:

```
HEIGHT Sharding:
├─ Shards along height dimension
├─ Used for: Attention heads, sequence length
└─ Example: [1, 8, seq_len, head_dim] sharded along heads

WIDTH Sharding:
├─ Shards along width dimension
├─ Used for: Feature dimensions
└─ Example: [1, 1, seq_len, dim] sharded along dim

2D Sharding:
├─ Shards along both dimensions
├─ Used for: Large tensors
└─ Example: [batch, seq_len, dim] sharded along both seq_len and dim
```

---

## Prefetcher System

### Architecture

**Sub-Devices**:
- **Prefetcher Sub-Device**: Dedicated cores for weight prefetching
- **Worker Sub-Device**: Main computation cores

**Circular Buffer**:
- Pre-allocated L1 buffer
- Stores prefetched weights
- Circular access pattern

**Operation Flow**:
```
1. Prefetcher loads weights from DRAM to L1
2. Worker reads weights from L1 circular buffer
3. Prefetcher loads next weights while worker computes
4. Overlaps computation and memory access
```

### Benefits

1. **Performance**: Overlaps computation and memory access
2. **Memory Efficiency**: Reuses L1 buffer
3. **Predictability**: Known memory access patterns

### Code Example

```python
# Prefetcher setup
self.prefetcher_setup = TtLlamaPrefetcherSetup(
    self.mesh_device,
    n_tensors=5,
    n_layers=self.n_layers,
    mode="decode",
)

# Insert weights into prefetcher
self.prefetcher_setup.insert_tensor(self.w1)
self.prefetcher_setup.insert_tensor(self.w3)
self.prefetcher_setup.insert_tensor(self.w2)

# Use prefetcher in operations
w1_out = ttnn.linear(
    x,
    self.w1,
    global_cb=self.prefetcher_setup.global_circular_buffer,
    sub_device_id=self.prefetcher_setup.worker_sub_device_id,
    ...
)
```

---

## Summary

CCL operations and memory management are optimized for:

1. **Ring Topology**: Efficient communication pattern
2. **Persistent Buffers**: Pre-allocated for performance
3. **Double Buffering**: Overlaps computation and communication
4. **Prefetcher**: Dedicated weight prefetching
5. **Memory Layouts**: Optimized for compute operations

Key optimizations:
- **Fused Operations**: Double matmul, fused CCL ops
- **Buffer Reuse**: Persistent buffers for efficiency
- **Ring Communication**: Optimal for Galaxy hardware
- **Memory Efficiency**: Careful memory layout management
