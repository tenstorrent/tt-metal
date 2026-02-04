# SDPA vs Ring Attention SDPA: Complete Technical Comparison

## Overview

This document provides a comprehensive comparison between the standard **Scaled Dot Product Attention (SDPA)** implementation and the **Ring Attention SDPA** used in Wan2.2. Based on analysis of:

- **Standard SDPA**: `tests/tt_eager/python_api_testing/unit_testing/misc/test_scaled_dot_product_attention_sprint.py`
- **Ring Attention SDPA**: `models/experimental/tt_dit/models/transformers/wan2_2/attention_wan.py`

## Summary of Key Differences

| Aspect | Standard SDPA | Ring Attention SDPA |
|--------|---------------|---------------------|
| **Scope** | Single device | Multi-device distributed |
| **Sequence Length** | Limited by device memory (~10K tokens) | Virtually unlimited (75K+ tokens) |
| **Memory Complexity** | O(N²) | O(N²/P) where P = devices |
| **Communication** | None | AllGather + synchronization |
| **API Call** | `scaled_dot_product_attention()` | `ring_joint_scaled_dot_product_attention()` |
| **Core Usage** | Full device grid | Partitioned grid (SDPA + CCL cores) |
| **Architecture** | Monolithic | Distributed with joint attention |

---

## 1. API and Function Signatures

### Standard SDPA
```python
# From test_scaled_dot_product_attention_sprint.py:76-83
tt_back = ttnn.transformer.scaled_dot_product_attention(
    tt_Q,                          # Query tensor
    tt_K,                          # Key tensor
    tt_V,                          # Value tensor
    is_causal=False,               # Attention mask type
    program_config=program_config, # Compute configuration
    compute_kernel_config=compute_kernel_config,
)
```

### Ring Attention SDPA
```python
# From attention_wan.py:263-290
spatial_BHNE, prompt_BHLE, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
    q_BHNE,                        # Query tensor (local chunk)
    k_BHNE,                        # Key tensor (local chunk)
    v_BHNE,                        # Value tensor (local chunk)
    self.dummy_joint_input,        # Joint Q (text prompts)
    self.dummy_joint_input,        # Joint K (text prompts)
    self.dummy_joint_input,        # Joint V (text prompts)
    persistent_output_buffer_k=..., # Pre-allocated buffers
    persistent_output_buffer_v=..., # Pre-allocated buffers
    joint_strategy="rear",         # How to handle joint tokens
    logical_n=N,                   # Actual sequence length
    program_config=...,            # Distributed compute config
    compute_kernel_config=...,     # Kernel configuration
    dim=2,                         # Sequence dimension
    multi_device_global_semaphore=..., # Cross-device sync
    num_links=...,                 # Communication links
    cluster_axis=...,              # Ring topology axis
    mesh_device=self.mesh_device,  # Multi-device mesh
    topology=ttnn.Topology.Linear, # Ring topology
    subdevice_id=...,              # CCL subdevice
    ccl_core_grid_offset=...,      # Core grid partitioning
)
```

**Key Differences**:
- **Ring SDPA**: Returns 3 tensors (spatial output, joint output, LSE)
- **Standard SDPA**: Returns 1 tensor (attention output)
- **Ring SDPA**: Requires 12+ additional parameters for distributed operation
- **Ring SDPA**: Handles two attention types simultaneously (spatial + joint)

---

## 2. Device and Hardware Setup

### Standard SDPA
```python
# Single device operation
device = ttnn.GetDefaultDevice()
program_config = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=device.compute_with_storage_grid_size(), # Full grid
    q_chunk_size=q_chunk_size,
    k_chunk_size=k_chunk_size,
    exp_approx_mode=False,
)
```

### Ring Attention SDPA
```python
# Multi-device mesh operation
self.mesh_device = mesh_device                    # 2x4 or 8x4 mesh
full_grid = self.mesh_device.compute_with_storage_grid_size()
self.sdpa_worker_grid = (full_grid.x, full_grid.y - 1)  # SDPA cores
ccl_core_grid_offset = (0, full_grid.y - 1)             # AllGather cores

ring_sdpa_program_config = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=self.sdpa_worker_grid, # Partitioned grid
    q_chunk_size=ring_sdpa_chunk_size[0],
    k_chunk_size=ring_sdpa_chunk_size[1],
    exp_approx_mode=False,
)
```

**Key Differences**:
- **Standard SDPA**: Uses full core grid on single device
- **Ring SDPA**: Partitions core grid between SDPA and communication operations
- **Ring SDPA**: Operates on device mesh (2x4, 8x4 configurations)
- **Ring SDPA**: Adaptive chunk sizes based on hardware and parallelization

---

## 3. Memory Management

### Standard SDPA
```python
# Simple tensor creation - device handles memory
tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

# No persistent buffers needed
# Memory lifecycle managed automatically
```

### Ring Attention SDPA
```python
# Complex multi-device tensor sharding
tt_Q = ttnn.from_torch(
    padded_Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=submesh,
    mesh_mapper=ttnn.ShardTensor2dMesh(submesh, dims=sdpa_input_shard_dims)
)

# Pre-allocated persistent buffers for communication
persistent_output_buffer_k = self.ccl_manager.get_ag_ping_pong_buffer(
    k_BHNE.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
)
persistent_output_buffer_v = self.ccl_manager.get_ag_ping_pong_buffer(
    v_BHNE.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
)

# Global synchronization semaphores
multi_device_global_semaphore = self.ccl_manager.get_ag_ping_pong_semaphore(...)
```

**Key Differences**:
- **Standard SDPA**: Simple memory allocation, device-managed
- **Ring SDPA**: Complex sharding across devices with explicit mesh mapping
- **Ring SDPA**: Persistent ping-pong buffers to avoid allocation overhead
- **Ring SDPA**: Global semaphores for cross-device synchronization

---

## 4. Input Data Characteristics

### Standard SDPA Test Inputs
```python
# From test_scaled_dot_product_attention_sprint.py:171-179
INPUT_SHAPES = [
    [1, 10, 9472, 128],    # wan_1xGLX_analog
    [1, 10, 2368, 128],    # wan_4xGLX_analog
]

# Single device can handle ~10K tokens max
# Full sequence processed on one device
Q = fa_rand(b, nh, sq, d)    # Full sequence
K = fa_rand(b, nkv, sk, d)   # Full sequence
V = fa_rand(b, nkv, sk, d)   # Full sequence
```

### Ring Attention Inputs
```python
# From test_ring_joint_attention.py:380-385
benchmark_model_input_shapes = {
    "wan_14b_720p": (1, 40, 75600, 0, 128),   # 75K tokens!
    "wan_14b_480p": (1, 40, 32760, 0, 128),   # 32K tokens
    "mochi": (1, 24, 44520, 118, 128),        # 44K + 118 joint tokens
    "flux": (1, 24, 4096, 512, 128),          # 4K + 512 joint tokens
}

# Sequence distributed across devices
input_tensor_q: B × NH × local_padded_N × DH     # Local chunk
gathered_k:     B × NH × padded_N × DH           # Full sequence
joint_q:        B × NH × L × DH                  # Joint tokens
```

**Key Differences**:
- **Standard SDPA**: ~10K token limit due to O(N²) memory
- **Ring SDPA**: 75K+ tokens possible due to distributed memory
- **Ring SDPA**: Handles both spatial (distributed) and joint (replicated) tokens
- **Ring SDPA**: Sequences padded for even distribution across devices

---

## 5. Computational Algorithm

### Standard SDPA Algorithm
```cpp
// Simplified algorithm from sdpa.cpp
1. Load Q, K, V tensors into device memory
2. For each Q chunk:
   a. For each K,V chunk:
      - Compute attention weights: QK^T
      - Apply softmax normalization
      - Compute output: softmax(QK^T)V
   b. Accumulate results
3. Return single output tensor
```

### Ring Attention Algorithm
```cpp
// From ring_joint_sdpa_program_factory.cpp:71-86
for each ring iteration:
    - read a Q chunk from input_tensor_q
    - for each KV chunk in local_padded_N:
        - on the first ring iteration: read from local input_tensor_k, input_tensor_v
        - otherwise: read from gathered_input_tensor_k, gathered_input_tensor_v
        - on the last ring iteration: also read from joint_tensor_k, joint_tensor_v
        - if KV chunk contains global token index (logical_n - 1): generate mask
        - compute attention with LSE update
    - write output Q chunk
    - if not first iteration: do LSE update for numerical stability
```

**Key Differences**:
- **Standard SDPA**: Single-pass algorithm, full K,V always available
- **Ring SDPA**: Multi-pass algorithm, K,V chunks arrive sequentially via AllGather
- **Ring SDPA**: LSE (Log-Sum-Exp) updates for numerical stability across iterations
- **Ring SDPA**: Joint tokens processed only in final iteration

---

## 6. Performance Characteristics

### Standard SDPA Performance
```python
# From test_scaled_dot_product_attention_sprint.py:304-330
def compute_sdpa_utilization(seqlen, head_dim, num_heads, duration_ns, core_count):
    # MM FLOPs: 4 * seqlen^2 * head_dim * num_heads
    mm_flops = 4 * seqlen * seqlen * head_dim * num_heads

    # Theoretical throughput: core_count * cycles * 2048 flops/cycle
    theoretical_flops = core_count * cycles * 2048
    utilization = (mm_flops / theoretical_flops) * 100

# Performance bounded by O(N²) memory scaling
# Optimal chunk sizes: q=256-512, k=256-512
# Utilization: typically 60-85% on single device
```

### Ring Attention Performance
```python
# Distributed performance with communication overlap
# Total time ≈ max(T_computation, T_communication)
# Memory scaling: O(N²/P) where P = number of devices

# Example configurations:
# T3000 (8 devices): 75K tokens → ~9.4K tokens per device
# Galaxy (32 devices): 75K tokens → ~2.3K tokens per device

# Performance optimizations:
- AllGather overlapped with SDPA computation
- Persistent buffers eliminate allocation overhead
- Optimal chunk sizes vary by hardware: (128,512) vs (256,256)
```

**Key Differences**:
- **Standard SDPA**: Single-device utilization focus
- **Ring SDPA**: Multi-device efficiency with communication hiding
- **Standard SDPA**: Chunk size optimization for memory vs compute
- **Ring SDPA**: Chunk size optimization for communication vs compute balance

---

## 7. Error Handling and Validation

### Standard SDPA Validation
```python
# From test_scaled_dot_product_attention_sprint.py:97-106
gt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
out_pass, out_pcc = comp_pcc(gt, tt_back, pcc_threshold)

# Simple validation:
- PCC > 0.9997 vs PyTorch reference
- RMSE < 4e-2
- Determinism: exact equality across runs
```

### Ring Attention Validation
```python
# From test_ring_joint_attention.py:269-321
# Complex multi-output validation
pt_Q = torch.cat([Q, joint_Q], dim=2)  # Combine spatial + joint
gt = torch.nn.functional.scaled_dot_product_attention(pt_Q, pt_K, pt_V)
gt_out = gt[:, :, :base_seq_len, :]      # Spatial component
gt_joint_out = gt[:, :, base_seq_len:, :] # Joint component

# Validate both outputs:
- Spatial output vs reference
- Joint output vs reference (per replica)
- LSE numerical stability
- Cross-device synchronization
```

**Key Differences**:
- **Standard SDPA**: Single output validation against PyTorch
- **Ring SDPA**: Dual output validation (spatial + joint) with distributed semantics
- **Ring SDPA**: Additional validation for numerical stability and device synchronization

---

## 8. Use Case and Applications

### Standard SDPA
**Optimal for**:
- Single-device inference
- Short to medium sequences (< 10K tokens)
- Development and prototyping
- Models like BERT, smaller transformers
- Scenarios where simplicity is preferred

**Limitations**:
- Memory bound by O(N²) scaling
- Cannot handle very long sequences
- Single device performance ceiling

### Ring Attention SDPA
**Optimal for**:
- Large-scale video generation (Wan2.2)
- Extremely long sequences (75K+ tokens)
- Multi-device distributed inference
- Production deployments at scale
- Models requiring both spatial and prompt attention

**Limitations**:
- Complex setup and configuration
- Communication overhead
- Requires multi-device hardware
- More complex debugging

---

## 9. Code Architecture Comparison

### Standard SDPA Implementation Stack
```
User API: ttnn.transformer.scaled_dot_product_attention()
    ↓
Core Implementation: ttnn::prim::sdpa()
    ↓
Device Operation: SDPADeviceOperation
    ↓
Program Factory: SDPAProgramFactory
    ↓
Kernels: Single-device SDPA kernels
```

### Ring Attention SDPA Implementation Stack
```
User API: ttnn.transformer.ring_joint_scaled_dot_product_attention()
    ↓
Core Implementation: ttnn::prim::ring_joint_scaled_dot_product_attention()
    ↓
Device Operation: RingJointSDPADeviceOperation
    ↓
Program Factory: RingJointSDPAProgramFactory
    ↓
Kernels: Multi-device SDPA + CCL kernels
    ↓
CCL Operations: RingAttentionAllGatherAsync
    ↓
Synchronization: Global semaphores + fused signaling
```

**Key Differences**:
- **Ring SDPA**: Much deeper implementation stack
- **Ring SDPA**: Integrated with CCL (Collective Communication Library)
- **Ring SDPA**: Complex device orchestration and synchronization

---

## 10. Configuration Examples

### Standard SDPA Configuration
```python
# Simple configuration
device = ttnn.open_device(device_id=0)
program_config = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=(8, 8),  # Full grid
    q_chunk_size=256,
    k_chunk_size=256,
    exp_approx_mode=False
)

# Run attention
output = ttnn.transformer.scaled_dot_product_attention(
    Q, K, V, program_config=program_config
)
```

### Ring Attention Configuration
```python
# Complex mesh setup
mesh_device = ttnn.open_mesh_device(
    ttnn.MeshShape(2, 4),  # 2x4 mesh
    device_ids=list(range(8))
)

# Parallel configuration
parallel_config = ParallelConfig(
    tensor_parallel=TensorParallelConfig(factor=4, mesh_axis=1),
    sequence_parallel=SequenceParallelConfig(factor=2, mesh_axis=0)
)

# CCL manager setup
ccl_manager = CCLManager(mesh_device, parallel_config)

# Attention layer initialization
attention = WanAttention(
    dim=3200, num_heads=40,
    mesh_device=mesh_device,
    ccl_manager=ccl_manager,
    parallel_config=parallel_config
)

# Run distributed attention
output = attention(
    spatial_1BND,  # Distributed sequence
    N=75600,       # Logical sequence length
    prompt_1BLP,   # Joint/prompt tokens
    rope_cos, rope_sin, trans_mat  # Position encodings
)
```

## Conclusion

The **Ring Attention SDPA** in Wan2.2 represents a fundamental architectural evolution from standard SDPA, enabling:

1. **Massive Scale**: 75K+ token sequences vs ~10K token limit
2. **Distributed Computing**: Multi-device parallelization with communication overlap
3. **Dual Attention**: Simultaneous spatial and joint token processing
4. **Production Ready**: Optimized for real-world video generation workloads

However, this comes at the cost of significantly increased complexity in setup, configuration, and debugging. The choice between implementations depends on your specific requirements:

- **Use Standard SDPA** for simplicity, development, and moderate-scale applications
- **Use Ring Attention SDPA** for production-scale video generation and extremely long sequences

The ring attention implementation showcases how modern AI systems are evolving beyond single-device limitations to enable previously impossible applications like high-quality long-form video generation.
