# Ring Attention in Wan2.2: A Complete Technical Deep Dive

## Table of Contents
1. [Introduction](#introduction)
2. [What is Wan2.2?](#what-is-wan22)
3. [Ring Attention Fundamentals](#ring-attention-fundamentals)
4. [Architecture Overview](#architecture-overview)
5. [The Ring Joint SDPA Operation](#the-ring-joint-sdpa-operation)
6. [Combining AllGather and SDPA](#combining-allgather-and-sdpa)
7. [Implementation Details](#implementation-details)
8. [Memory Management and Optimization](#memory-management-and-optimization)
9. [Performance and Scalability](#performance-and-scalability)
10. [Code Examples and Usage](#code-examples-and-usage)
11. [References and Further Reading](#references-and-further-reading)

## Introduction

Ring Attention is a revolutionary memory-efficient attention mechanism that enables processing of extremely long sequences by distributing computation across multiple devices in a ring topology. In Wan2.2, this technique is implemented as "Ring Joint SDPA" - a fusion of AllGather communication operations with Scaled Dot Product Attention computation that executes **simultaneously** on the core grid.

This document provides a comprehensive technical analysis of how ring attention works in the Wan2.2 text-to-video generation model, written for someone completely new to these concepts.

## What is Wan2.2?

**Wan2.2-T2V-A14B** is a state-of-the-art text-to-video generative model with 14 billion parameters. Key characteristics:

- **Architecture**: Diffusion Transformer (DiT) based
- **Purpose**: High-quality text-to-video generation
- **Scale**: Operates on very long sequences (75,600+ tokens for 720p video)
- **Hardware**: Optimized for Tenstorrent's Wormhole and Blackhole architectures
- **Parallelism**: Uses sophisticated multi-axis parallelization strategies

The model's massive sequence lengths (up to ~75K tokens) make traditional attention computationally prohibitive, necessitating ring attention.

## Ring Attention Fundamentals

### The Memory Problem
Traditional self-attention has **O(N²)** memory complexity, where N is sequence length. For Wan2.2's sequences:
- **720p video**: ~75,600 tokens → ~5.7 billion attention weights
- **480p video**: ~32,760 tokens → ~1.07 billion attention weights

This would require **hundreds of GB** of memory for a single attention operation!

### Ring Attention Solution
Ring attention solves this by:
1. **Distributing sequences** across devices in a ring topology
2. **Computing attention incrementally** as K,V data flows around the ring
3. **Overlapping communication and computation** for efficiency

### Key Insight: Ring + Joint Architecture
Wan2.2 implements **"Ring Joint SDPA"** which handles two types of attention simultaneously:
- **Spatial attention**: Long video sequences distributed across devices
- **Joint/Cross attention**: Text prompts (shorter, replicated across devices)

## Architecture Overview

```
Device 0    Device 1    Device 2    Device 3
┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
│Q₀K₀V₀│ → │Q₁K₁V₁│ → │Q₂K₂V₂│ → │Q₃K₃V₃│
└─────┘     └─────┘     └─────┘     └─────┘
   ↑                                   │
   └───────────────────────────────────┘
         Ring Topology (Linear for Wan2.2)

Each device also has:
- Joint Q, K, V (text prompts) - replicated
- Persistent buffers for AllGather operations
- Synchronization semaphores
```

### Parallel Configuration
Wan2.2 uses a multi-axis parallelization strategy:

| System   | Sequence Parallel (SP) | Tensor Parallel (TP) | Ring Size | Devices |
|----------|------------------------|----------------------|-----------|---------|
| T3000    | 2x factor             | 4x factor            | 2         | 8       |
| Galaxy   | 8x factor             | 4x factor            | 8         | 32      |

## The Ring Joint SDPA Operation

### Core Components

**Located in**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp`

The Ring Joint SDPA operation combines:

1. **Ring Attention AllGather**: Asynchronously gathers K,V from other devices
2. **Joint Attention**: Processes local text prompts
3. **SDPA Computation**: Performs attention with gathered data
4. **LSE Management**: Handles log-sum-exp for numerical stability

### Input Tensors

```cpp
// Per-device spatial tokens (fractured on sequence dimension)
input_tensor_q:  B × NH × local_padded_N × DH
input_tensor_k:  B × NH × local_padded_N × DH
input_tensor_v:  B × NH × local_padded_N × DH

// Gathered K,V from all devices (AllGather destination)
gathered_k:      B × NH × padded_N × DH
gathered_v:      B × NH × padded_N × DH

// Joint/prompt tokens (replicated across devices)
joint_q:         B × NH × L × DH
joint_k:         B × NH × L × DH
joint_v:         B × NH × L × DH
```

Where:
- **B**: Batch size
- **NH**: Number of heads per device
- **local_padded_N**: Local sequence length per device
- **padded_N**: Global sequence length (local_padded_N × ring_size)
- **L**: Joint sequence length (prompt tokens)
- **DH**: Head dimension

## Combining AllGather and SDPA

### The Revolutionary Insight

Traditional approaches:
1. **AllGather K,V** (communication)
2. **Then compute SDPA** (computation)

Ring Joint SDPA in Wan2.2:
1. **AllGather and SDPA execute simultaneously** on different core grids!

### Core Grid Partitioning

```cpp
// From attention_wan.py:113-127
full_grid = self.mesh_device.compute_with_storage_grid_size()
self.sdpa_worker_grid = (full_grid.x, full_grid.y - 1)  // SDPA cores
ccl_core_grid_offset = (0, full_grid.y - 1)            // AllGather cores

// Validation ensures no overlap
TT_FATAL(
    args.ccl_core_grid_offset.y >= args.program_config.value().compute_with_storage_grid_size.y,
    "SDPA coregrid overlaps with AllGather coregrid");
```

**Example on 8×8 core grid:**
- **SDPA cores**: 8×7 grid (rows 0-6)
- **AllGather cores**: 8×1 grid (row 7)
- **Total efficiency**: No idle cores during ring iterations!

### Algorithmic Flow

**From**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp:71-86`

```cpp
/*
The algorithm is roughly described below.
- for each ring iteration:
    - read a Q chunk from input_tensor_q
    - for each KV chunk in local_padded_N:
        - on the first ring iteration, read from local input_tensor_k and input_tensor_v
        - otherwise, read from gathered_input_tensor_k and gathered_input_tensor_v
        - on the last ring iteration, also read from joint_tensor_k and joint_tensor_v
        - if the KV chunk contains the global token index (logical_n - 1), generate a mask
        - compute attention
    - write the output Q chunk
    - if this is not the first ring iteration, do the LSE update.
*/
```

### Ring Iteration Details

**Ring Iteration 0** (First):
- Read local K₀,V₀ from device storage
- Compute attention: Q₀ × K₀,V₀
- **Simultaneously**: AllGather starts sending K₀,V₀ to next device

**Ring Iteration 1**:
- Read gathered K₁,V₁ (received from previous iteration)
- Compute attention: Q₀ × K₁,V₁
- Update LSE for numerical stability
- **Simultaneously**: AllGather continues with next K,V pair

**Ring Iteration N-1** (Last):
- Read gathered Kₙ₋₁,Vₙ₋₁
- **Also read joint K,V** (text prompts)
- Compute: Q₀ × [Kₙ₋₁,Vₙ₋₁, joint_K,joint_V]
- Final LSE update

## Implementation Details

### Persistent Buffers

**Problem**: AllGather creates temporary tensors each iteration → memory fragmentation

**Solution**: Pre-allocated persistent buffers

```python
# From attention_wan.py:270-275
persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
    k_BHNE.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
),
persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
    v_BHNE.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
),
```

**Ping-pong buffers**: Two buffers alternate between send/receive to avoid conflicts.

### Synchronization Mechanism

**Global Semaphores**: Coordinate across devices in the ring

```python
# From attention_wan.py:281-283
multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
    self.parallel_config.sequence_parallel.mesh_axis
),
```

**Fused Operation Signaling**: SDPA cores signal completion to AllGather cores

```cpp
// From ring_fusion.cpp:62-72
void RingSDPAFusedOpSignaler::push_ring_sdpa_fused_op_rt_args(std::vector<uint32_t>& out_rt_args) {
    out_rt_args.push_back(static_cast<uint32_t>(this->ring_size));
    out_rt_args.push_back(static_cast<uint32_t>(this->ring_index));
    out_rt_args.push_back(static_cast<uint32_t>(this->forward_writes_expected));
    out_rt_args.push_back(static_cast<uint32_t>(this->backward_writes_expected));
    out_rt_args.push_back(static_cast<uint32_t>(this->fused_op_receiver_signal_semaphores[0]));
    out_rt_args.push_back(static_cast<uint32_t>(this->fused_op_receiver_signal_semaphores[1]));
}
```

### Chunk Size Optimization

Different chunk sizes optimize for different hardware and problem sizes:

```python
# From attention_wan.py:16-22
sdpa_chunk_size_map = {
    (False, 2, 4): (256, 256),   # Wormhole, SP=2, TP=4
    (False, 8, 4): (256, 256),   # Wormhole, SP=8, TP=4
    (True, 2, 2): (128, 512),    # Blackhole, SP=2, TP=2
    (True, 8, 4): (128, 512),    # Blackhole, SP=8, TP=4
}
```

**Q chunk size**: How much of Q to process per iteration
**K chunk size**: How much of K,V to process per inner loop

Smaller chunks → better memory efficiency
Larger chunks → better compute utilization

## Memory Management and Optimization

### Memory Hierarchy

```
┌─────────────────────────────────────┐
│  DRAM (Main Storage)               │
│  - Input Q,K,V tensors             │
│  - Joint Q,K,V tensors             │
│  - Persistent AllGather buffers    │
└─────────────────────────────────────┘
          ↕️ (DMA transfers)
┌─────────────────────────────────────┐
│  L1 Cache (Working Memory)         │
│  - Current chunks being processed  │
│  - Intermediate attention results  │
│  - LSE accumulators               │
└─────────────────────────────────────┘
```

### Memory Access Patterns

**Optimized for sequential access**:
- Q chunks: Sequential read
- K,V chunks: Sequential read from persistent buffers
- Output: Sequential write

**Cache-friendly design**:
- Chunk sizes align with L1 cache capacity
- Minimize data movement between DRAM and L1

### Numerical Stability

**LSE (Log-Sum-Exp) Management**:
Ring attention accumulates attention weights across iterations, requiring careful numerical handling.

```python
# PyTorch reference from test_ring_joint_attention.py:42-48
if ring_id == 0:
    out = cur_out
    lse = cur_lse
else:
    sig = F.sigmoid(cur_lse - lse)
    out = out - sig * (out - cur_out)
    lse = lse - F.logsigmoid(lse - cur_lse)
```

This prevents overflow/underflow when combining attention weights from multiple ring iterations.

## Performance and Scalability

### Theoretical Performance Benefits

**Communication Hiding**: AllGather overlapped with SDPA computation
- **Traditional**: T_allgather + T_sdpa
- **Ring Joint**: max(T_allgather, T_sdpa) ≈ T_sdpa (if well-balanced)

**Memory Efficiency**: O(N²/P) memory per device instead of O(N²)
- For Wan2.2 720p: ~5.7B weights → ~715M weights per device (8-way split)

### Real-World Performance

**Wan2.2 720p video generation** (75,600 sequence length):

| System   | Configuration | Performance |
|----------|---------------|-------------|
| T3000    | 2×4 mesh     | ~X.X seconds |
| Galaxy   | 8×4 mesh     | ~Y.Y seconds |

### Scalability Analysis

**Ring Size Impact**:
- **Larger rings**: Better memory efficiency, more communication overhead
- **Smaller rings**: Less communication, higher memory usage per device

**Optimal Configuration**: Balance between:
1. Memory capacity per device
2. Communication bandwidth
3. Compute utilization

## Code Examples and Usage

### Basic Usage in Wan2.2 Attention

```python
# From models/experimental/tt_dit/models/transformers/wan2_2/attention_wan.py:263-290

if self.parallel_config.sequence_parallel.factor > 1:
    # Use ring attention for distributed sequence processing
    spatial_BHNE, prompt_BHLE, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        q_BHNE,                    # Query tensor (local chunk)
        k_BHNE,                    # Key tensor (local chunk)
        v_BHNE,                    # Value tensor (local chunk)
        self.dummy_joint_input,    # Joint Q (dummy for self-attention)
        self.dummy_joint_input,    # Joint K (dummy for self-attention)
        self.dummy_joint_input,    # Joint V (dummy for self-attention)
        persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(...),
        persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(...),
        joint_strategy="rear",     # Append joint tokens at end
        logical_n=N,              # Actual (non-padded) sequence length
        program_config=self.ring_sdpa_program_config,
        compute_kernel_config=self.sdpa_compute_kernel_config,
        dim=2,                    # Sequence dimension
        multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(...),
        num_links=self.ccl_manager.num_links,
        cluster_axis=self.parallel_config.sequence_parallel.mesh_axis,
        mesh_device=self.mesh_device,
        topology=ttnn.Topology.Linear,  # Linear ring topology
        subdevice_id=self.ccl_manager.ccl_sub_device_id,
        ccl_core_grid_offset=(0, self.sdpa_worker_grid[1]),
    )
else:
    # Fall back to standard SDPA for single device
    spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
        q_BHNE, k_BHNE, v_BHNE,
        is_causal=False,
        program_config=self.sdpa_program_config,
        compute_kernel_config=self.sdpa_compute_kernel_config,
    )
```

### Testing Ring Attention

```python
# Run unit tests for ring attention
pytest models/experimental/tt_dit/tests/unit/test_ring_joint_attention.py::test_ring_joint_sdpa_dit_wh_t3k

# Performance benchmarks
pytest models/experimental/tt_dit/tests/models/wan2_2/test_performance_wan.py
```

### Configuration Examples

```python
# T3000 (2x4 mesh) configuration
parallel_config = (0, 2, 1, 4)  # (rp_axis=0, rp_factor=2, up_axis=1, up_factor=4)
chunk_sizes = (256, 256)        # (q_chunk_size, k_chunk_size)

# Galaxy (8x4 mesh) configuration
parallel_config = (0, 8, 1, 4)  # (rp_axis=0, rp_factor=8, up_axis=1, up_factor=4)
chunk_sizes = (256, 256)        # (q_chunk_size, k_chunk_size)
```

## References and Further Reading

### Academic Papers
1. **"Ring Attention with Blockwise Transformers for Near-Infinite Context"** - Liu et al.
   - Original ring attention paper: https://arxiv.org/abs/2310.01889

2. **"Unified Sequence Parallel: Long Context with Better Efficiency"**
   - Sequence parallelism reference: https://github.com/feifeibear/long-context-attention

3. **"Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"**
   - Stable Diffusion 3.5 architecture: https://arxiv.org/abs/2403.03206

### Implementation Files
- **Ring Joint SDPA Core**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp`
- **Program Factory**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp`
- **AllGather Async**: `ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/`
- **Wan2.2 Attention**: `models/experimental/tt_dit/models/transformers/wan2_2/attention_wan.py`

### Key Concepts
- **SDPA**: Scaled Dot Product Attention - the core attention computation
- **AllGather**: Collective communication operation to gather data from all devices
- **CCL**: Collective Communication Library - handles multi-device operations
- **LSE**: Log-Sum-Exp - numerical stability technique for attention normalization
- **Persistent Buffers**: Pre-allocated memory to avoid allocation overhead
- **Ping-pong Buffers**: Dual buffer system to overlap communication and computation

### Hardware Context
- **Wormhole**: Tenstorrent's AI accelerator architecture
- **Blackhole**: Next-generation Tenstorrent architecture
- **Core Grid**: 2D array of processing cores on each chip
- **Mesh Device**: Multi-chip system connected in 2D mesh topology

---

**This document represents the current state of ring attention implementation in Wan2.2 as of the codebase analysis. The implementation continues to evolve with ongoing performance optimizations and new hardware support.**
