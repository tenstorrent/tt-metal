# GPT-OSS MoE MLP: Complete Tensor Flow and Parallelism Deep Dive

## Executive Summary

The GPT-OSS MoE (Mixture of Experts) MLP implementation represents a sophisticated parallelization strategy for distributing expert computation across a multi-device mesh. The key innovation lies in the **ThroughputExperts** approach, which uses all-to-all communication to dynamically batch tokens to their assigned experts, maximizing hardware utilization while maintaining low latency during decode operations.

This documentation traces the complete tensor journey through the system for the specific test configuration:
- **Device Mesh**: 4×8 galaxy (32 devices total)
- **Parallelism**: EP=4 (Expert Parallel), TP=8 (Tensor Parallel), DP=1 (Data Parallel)
- **Experts**: 128 total experts, 4 experts per device
- **Operation Mode**: Decode with high throughput, L1 memory placement

## System Architecture

### 4×8 Galaxy Mesh Configuration

```
         Column Devices (TP=8)
         0   1   2   3   4   5   6   7
    +---+---+---+---+---+---+---+---+
R 0 | D0| D1| D2| D3| D4| D5| D6| D7|  Experts 0-31   (8 groups × 4 experts)
o   +---+---+---+---+---+---+---+---+
w 1 | D8| D9|D10|D11|D12|D13|D14|D15|  Experts 32-63  (8 groups × 4 experts)
s   +---+---+---+---+---+---+---+---+
  2 |D16|D17|D18|D19|D20|D21|D22|D23|  Experts 64-95  (8 groups × 4 experts)
(EP=4) +---+---+---+---+---+---+---+---+
  3 |D24|D25|D26|D27|D28|D29|D30|D31|  Experts 96-127 (8 groups × 4 experts)
    +---+---+---+---+---+---+---+---+
```

**Key Dimensions:**
- **Expert Parallelism (EP=4)**: Experts distributed across 4 device rows
- **Tensor Parallelism (TP=8)**: Weight matrices sharded across 8 column devices
- **Data Parallelism (DP=1)**: No explicit data replication in this configuration
- **Expert Distribution**: 128 experts ÷ 32 devices = 4 experts per device

### Expert Assignment Strategy

Each device hosts 4 consecutive experts based on its position in the mesh:
```python
device_id = row * 8 + col
expert_start = device_id * 4
experts_on_device = [expert_start, expert_start+1, expert_start+2, expert_start+3]
```

Example:
- Device D0 (row=0, col=0): Experts [0, 1, 2, 3]
- Device D8 (row=1, col=0): Experts [32, 33, 34, 35]
- Device D31 (row=3, col=7): Experts [124, 125, 126, 127]

## Complete Tensor Journey

### Stage 1: Input Creation and Sharding

```
Original Input Tensor
┌─────────────────────────┐
│ Shape: [128, 1, 2880]   │  ← Batch=128, Seq=1, Hidden=2880
└─────────────────────────┘
            ↓
      Row Sharding (EP=4)
┌─────────────────────────┐
│ Row 0: [32, 1, 2880]    │  ← Tokens 0-31
│ Row 1: [32, 1, 2880]    │  ← Tokens 32-63
│ Row 2: [32, 1, 2880]    │  ← Tokens 64-95
│ Row 3: [32, 1, 2880]    │  ← Tokens 96-127
└─────────────────────────┘
            ↓
     TTNN Format Transform
┌─────────────────────────┐
│ Per Row: [1, 1, 32, 2880]│  ← TTNN 4D format
└─────────────────────────┘
```

**Code Reference**: `models/demos/gpt_oss/tests/unit/test_modules.py:565-570`
```python
# Input creation in test
hidden_states = torch.randn(
    batch_size, seq_len, hidden_size, dtype=torch.bfloat16
).to("cuda")
```

### Stage 2: Router Computation

```
Router Linear Projection
┌──────────────────────────────┐
│ Input: [1, 1, 32, 2880]      │
│ Weight: [2880, 128]          │  ← Projects to expert space
└──────────────────────────────┘
            ↓
┌──────────────────────────────┐
│ Logits: [1, 1, 32, 128]      │  ← Score for each expert
└──────────────────────────────┘
            ↓
         Top-K Selection (K=8)
┌──────────────────────────────┐
│ Indices: [1, 1, 32, 8]       │  ← Expert IDs
│ Weights: [1, 1, 32, 8]       │  ← Routing probabilities
└──────────────────────────────┘
            ↓
      Softmax Normalization
┌──────────────────────────────┐
│ Normalized: [1, 1, 32, 8]    │  ← Sum to 1.0 per token
└──────────────────────────────┘
```

**Key Operations:**
1. Linear projection to expert dimension
2. Top-K selection (K=8 experts per token)
3. Softmax normalization of selected weights

**Code Reference**: `models/demos/gpt_oss/tt/topk.py`

### Stage 3: All-to-All Dispatch

```
Before All-to-All Dispatch
┌─────────────────────────────────────┐
│ Row 0: 32 tokens, each assigned    │
│        to 8 experts (possibly remote)│
│ Row 1: 32 tokens, each assigned    │
│        to 8 experts (possibly remote)│
│ Row 2: 32 tokens, each assigned    │
│        to 8 experts (possibly remote)│
│ Row 3: 32 tokens, each assigned    │
│        to 8 experts (possibly remote)│
└─────────────────────────────────────┘
            ↓
    All-to-All Communication (axis=0)
            ↓
After All-to-All Dispatch
┌─────────────────────────────────────┐
│ Row 0: Variable # tokens for        │
│        experts 0-31                 │
│ Row 1: Variable # tokens for        │
│        experts 32-63                │
│ Row 2: Variable # tokens for        │
│        experts 64-95                │
│ Row 3: Variable # tokens for        │
│        experts 96-127               │
└─────────────────────────────────────┘

Per-Device View (example for Device D0):
┌─────────────────────────────────────┐
│ Expert 0: N₀ tokens                │
│ Expert 1: N₁ tokens                │
│ Expert 2: N₂ tokens                │
│ Expert 3: N₃ tokens                │
│ Shape: [1, 4, Σ(Nᵢ), 2880]         │
└─────────────────────────────────────┘
```

**Communication Pattern:**
- **Direction**: Along row axis (axis=0)
- **Purpose**: Route tokens to devices hosting their assigned experts
- **Topology**: Ring with 4 links (EP=4)
- **Dynamic Batching**: Each device receives variable number of tokens

**Code Reference**: `models/demos/gpt_oss/tt/experts_throughput/__init__.py:270-280`

### Stage 4: Expert MLP Computation

```
Expert MLP Processing (per device)
┌──────────────────────────────────────┐
│ Input: [1, 4, total_tokens, 2880]    │
└──────────────────────────────────────┘
            ↓
    Fused Gate/Up Projection
┌──────────────────────────────────────┐
│ Gate W: [1, 4, 2880, 6144]           │
│ Up W:   [1, 4, 2880, 6144]           │
│ Output: [1, 4, tokens, 12288]        │  ← Concatenated
└──────────────────────────────────────┘
            ↓
         SwiGLU Activation
┌──────────────────────────────────────┐
│ gate = output[:, :, :, :6144]        │
│ up = output[:, :, :, 6144:]          │
│ act = silu(gate) * up                │
│ Shape: [1, 4, tokens, 6144]          │
└──────────────────────────────────────┘
            ↓
         Down Projection
┌──────────────────────────────────────┐
│ Weight: [1, 4, 6144, 2880]           │
│ Output: [1, 4, tokens, 2880]         │
└──────────────────────────────────────┘
```

**MLP Architecture:**
1. **Fused Gate/Up**: Single matmul producing both gate and up projections
2. **SwiGLU**: `silu(gate) * up` with clamping for stability
3. **Down Projection**: Maps back to hidden dimension

**Tensor Parallelism Within MLP:**
- Each column device holds 1/8 of the weight matrices
- Hidden dimension 2880 ÷ 8 = 360 per device
- Intermediate dimension 6144 ÷ 8 = 768 per device

**Code Reference**: `models/demos/gpt_oss/tt/experts_throughput/decode.py:30-120`

### Stage 5: All-to-All Combine

```
Before All-to-All Combine
┌─────────────────────────────────────┐
│ Row 0: Expert outputs for tokens    │
│        that selected experts 0-31   │
│ Row 1: Expert outputs for tokens    │
│        that selected experts 32-63  │
│ Row 2: Expert outputs for tokens    │
│        that selected experts 64-95  │
│ Row 3: Expert outputs for tokens    │
│        that selected experts 96-127 │
└─────────────────────────────────────┘
            ↓
    All-to-All Communication (axis=0)
            ↓
After All-to-All Combine
┌─────────────────────────────────────┐
│ Row 0: 8 expert outputs for each    │
│        of its 32 original tokens    │
│        Shape: [8, 1, 32, 2880]      │
│ Row 1: 8 expert outputs for each    │
│        of its 32 original tokens    │
│        Shape: [8, 1, 32, 2880]      │
│ ...                                  │
└─────────────────────────────────────┘
```

**Key Points:**
- Uses dispatch metadata to route outputs back
- Each token receives exactly K=8 expert outputs
- Preserves original token ordering

**Code Reference**: `models/demos/gpt_oss/tt/experts_throughput/__init__.py:300-310`

### Stage 6: Weighted Aggregation

```
Expert Output Weighting
┌──────────────────────────────────────┐
│ Outputs: [8, 1, 32, 2880]           │
│ Weights: [8, 1, 32, 1]              │  ← From router
└──────────────────────────────────────┘
            ↓
      Element-wise Multiplication
┌──────────────────────────────────────┐
│ Weighted: [8, 1, 32, 2880]          │
└──────────────────────────────────────┘
            ↓
        Sum Across Experts (dim=0)
┌──────────────────────────────────────┐
│ Aggregated: [1, 1, 32, 2880]        │
└──────────────────────────────────────┘
```

**Mathematics:**
```
output[token] = Σ(i=1 to K) weight[i] * expert_output[i]
```

### Stage 7: Cross-Column All-Reduce

```
Before All-Reduce (per row)
┌─────────────────────────────────────┐
│ Col 0: [1, 1, 32, 360]  (partial)  │
│ Col 1: [1, 1, 32, 360]  (partial)  │
│ ...                                 │
│ Col 7: [1, 1, 32, 360]  (partial)  │
└─────────────────────────────────────┘
            ↓
    All-Reduce Across Columns (axis=1)
            ↓
After All-Reduce
┌─────────────────────────────────────┐
│ All Cols: [1, 1, 32, 2880] (complete)│
└─────────────────────────────────────┘
```

**Purpose**: Aggregate partial results from tensor parallelism
**Communication**: Ring topology across 8 column devices

## Parallelism Deep Dive

### Expert Parallelism (EP=4)

```
Expert Distribution Across Rows
┌──────────────────────────────┐
│ Row 0: Experts 0-31          │
│ Row 1: Experts 32-63         │
│ Row 2: Experts 64-95         │
│ Row 3: Experts 96-127        │
└──────────────────────────────┘

Communication Pattern:
Row 0 ←→ Row 1 ←→ Row 2 ←→ Row 3
     All-to-All Ring (4 links)
```

**Benefits:**
- Distributes expert memory across devices
- Enables dynamic load balancing via all-to-all
- Scales to thousands of experts

### Tensor Parallelism (TP=8)

```
Weight Sharding Across Columns
┌─────────────────────────────────────────┐
│ Weight Matrix [2880, 6144]              │
└─────────────────────────────────────────┘
                ↓ Shard
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│Col0 │Col1 │Col2 │Col3 │Col4 │Col5 │Col6 │Col7 │
│[360,│[360,│[360,│[360,│[360,│[360,│[360,│[360,│
│768] │768] │768] │768] │768] │768] │768] │768] │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

Communication: All-Reduce after computation
```

**Benefits:**
- Reduces per-device memory requirements
- Enables larger model dimensions
- Efficient ring all-reduce communication

### Implicit Data Parallelism

While DP=1 in this configuration, there's implicit data parallelism:
- Batch dimension naturally sharded across EP dimension
- Each row processes 32 tokens independently
- No additional communication required

## Communication Patterns

### All-to-All Operations

```
All-to-All Dispatch Flow (EP=4)
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│ Row 0  │───▶│ Row 1  │───▶│ Row 2  │───▶│ Row 3  │
│ 32 tok │    │ 32 tok │    │ 32 tok │    │ 32 tok │
└────────┘    └────────┘    └────────┘    └────────┘
     ▲                                          │
     └──────────────────────────────────────────┘
                  Ring Topology

Data Exchange Pattern:
- Step 1: Row i sends tokens for Row (i+1)%4's experts
- Step 2: Row i sends tokens for Row (i+2)%4's experts
- Step 3: Row i sends tokens for Row (i+3)%4's experts
- Result: All tokens reach their assigned expert devices
```

**Performance Characteristics:**
- Latency: O(EP-1) steps
- Bandwidth: Fully utilized with ring algorithm
- Memory: Requires buffering for in-flight data

### All-Reduce Operations

```
All-Reduce Pattern (TP=8)
┌───┬───┬───┬───┬───┬───┬───┬───┐
│C0 │C1 │C2 │C3 │C4 │C5 │C6 │C7 │
└─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┘
  │   │   │   │   │   │   │   │
  └───┴───┴───┴───┴───┴───┴───┘
       Ring All-Reduce

Reduction Steps:
1. Reduce-Scatter: Each device gets 1/8 of final result
2. All-Gather: Broadcast reduced chunks to all devices
```

**Optimization:** Uses NCCL-style ring algorithm for efficiency

## Performance Optimizations

### Memory Placement Strategy

```python
# Decode mode: Use L1 for low latency
self.mlp = TtMLP(
    mesh_device=self.mesh_device,
    state_dict=state_dict,
    expert_grid_size=(EP, TP),
    decode_mode=True,  # ← Triggers L1 placement
    memory_config=ttnn.L1_MEMORY_CONFIG
)
```

### Dynamic Batching

```
Traditional Static Assignment:
┌──────────────────────────┐
│ Expert 0: Always N tokens│  ← Fixed, may waste compute
│ Expert 1: Always N tokens│
│ ...                      │
└──────────────────────────┘

Dynamic Batching via All-to-All:
┌──────────────────────────┐
│ Expert 0: N₀ tokens      │  ← Variable, based on routing
│ Expert 1: N₁ tokens      │  ← Adapts to actual usage
│ ...                      │
└──────────────────────────┘
```

**Benefits:**
- Better load balancing
- Higher hardware utilization
- Reduced bubbles in pipeline

### Fused Operations

1. **Fused Gate/Up Projection:**
   ```python
   # Instead of two separate matmuls:
   gate = input @ gate_weight
   up = input @ up_weight

   # Single fused matmul:
   concat = input @ concat_weight  # [gate_weight; up_weight]
   gate, up = split(concat)
   ```

2. **SwiGLU with Clamping:**
   ```python
   # Prevents numerical instability
   gate = torch.clamp(gate, min=-10, max=10)
   output = silu(gate) * up
   ```

### 1D Multicast Matmul Configuration

```python
# Optimized for decode workload
matmul_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=(8, 4),
    in0_block_w=4,  # Tuned for L1 memory
    per_core_M=1,   # Single token decode
    per_core_N=4,   # Optimized for hidden dimension
)
```

## Key Implementation Files

### Core MoE Components

| File | Purpose | Key Functions |
|------|---------|---------------|
| `mlp.py` | Top-level MLP orchestrator | `TtMLP.__init__()`, `forward()` |
| `topk.py` | Router implementation | `TtTop1Router.forward()` |
| `experts_throughput/__init__.py` | ThroughputExperts class | `forward()`, `all_to_all_dispatch/combine()` |
| `experts_throughput/decode.py` | Decode forward pass | `forward_decode()`, expert MLP logic |
| `experts_throughput/config.py` | All-to-all configurations | Communication configs |

### Test Infrastructure

| File | Purpose | Key Functions |
|------|---------|---------------|
| `test_modules.py` | Main test entry | `test_modules()` with parametrization |
| `test_factory.py` | Test setup utilities | Device mesh creation, model loading |

## Visual Flow Summary

```
┌─────────────────────────────────────────────────────────┐
│                    Input Tensor                         │
│                   [128, 1, 2880]                        │
└────────────────────────┬────────────────────────────────┘
                         ↓ Row Shard (EP=4)
┌─────────────────────────────────────────────────────────┐
│   Row 0: [32,2880]  Row 1: [32,2880]  Row 2: [32,2880] │
└────────────────────────┬────────────────────────────────┘
                         ↓ Router (Top-K=8)
┌─────────────────────────────────────────────────────────┐
│        Each token → 8 expert assignments + weights      │
└────────────────────────┬────────────────────────────────┘
                         ↓ All-to-All Dispatch
┌─────────────────────────────────────────────────────────┐
│     Tokens routed to devices hosting their experts      │
└────────────────────────┬────────────────────────────────┘
                         ↓ Expert MLP (TP=8)
┌─────────────────────────────────────────────────────────┐
│   Gate/Up → SwiGLU → Down (parallelized across cols)    │
└────────────────────────┬────────────────────────────────┘
                         ↓ All-to-All Combine
┌─────────────────────────────────────────────────────────┐
│    Expert outputs returned to original token positions  │
└────────────────────────┬────────────────────────────────┘
                         ↓ Weighted Sum
┌─────────────────────────────────────────────────────────┐
│           Σ(weight[i] * expert_output[i])               │
└────────────────────────┬────────────────────────────────┘
                         ↓ All-Reduce (TP)
┌─────────────────────────────────────────────────────────┐
│                  Final Output Tensor                     │
│                 [32, 1, 2880] per row                   │
└──────────────────────────────────────────────────────────┘
```

## Quick Reference Guide

### Key Concepts

| Term | Definition | Value in Test |
|------|------------|---------------|
| EP | Expert Parallelism - experts distributed across device rows | 4 |
| TP | Tensor Parallelism - weights sharded across device columns | 8 |
| DP | Data Parallelism - batch replication (implicit here) | 1 |
| K | Top-K experts selected per token | 8 |
| All-to-All | Communication primitive for token routing | Ring topology |
| ThroughputExperts | Dynamic batching approach using all-to-all | Core innovation |

### Common Configurations

```python
# Decode Configuration (L1 Memory)
config = {
    "expert_grid_size": (4, 8),  # (EP, TP)
    "decode_mode": True,
    "memory_config": ttnn.L1_MEMORY_CONFIG,
    "top_k": 8,
}

# Prefill Configuration (DRAM)
config = {
    "expert_grid_size": (4, 8),
    "decode_mode": False,
    "memory_config": ttnn.DRAM_MEMORY_CONFIG,
    "top_k": 8,
}
```

### Performance Tuning Parameters

1. **Memory Configuration:**
   - L1: Lower latency, limited capacity (decode)
   - DRAM: Higher capacity, higher latency (prefill)

2. **Matmul Configurations:**
   - Block sizes: Tune based on tensor dimensions
   - Grid sizes: Match hardware compute grid
   - Multicast: Enable for broadcast operations

3. **All-to-All Tuning:**
   - Ring links: Number of parallel communication channels
   - Buffer sizes: Balance memory vs. performance
   - Overlap: Enable computation/communication overlap

## Troubleshooting Guide

### Common Issues and Solutions

1. **Out of Memory (OOM) in L1:**
   - Reduce block sizes in matmul configs
   - Switch to DRAM for larger batches
   - Decrease top_k value

2. **All-to-All Timeout:**
   - Check device connectivity
   - Verify ring topology setup
   - Increase timeout values

3. **Numerical Instabilities:**
   - Enable SwiGLU clamping
   - Check for NaN/Inf in router weights
   - Verify softmax normalization

4. **Load Imbalance:**
   - Monitor expert usage statistics
   - Consider load balancing loss term
   - Adjust router temperature

### Debugging Commands

```bash
# Run specific test with verbose output
pytest models/demos/gpt_oss/tests/unit/test_modules.py \
  -k "4x8 and decode_high_throughput and layer_0" \
  --test-modules mlp -xvs

# Enable debug logging
export TTNN_DEBUG=1
export TT_METAL_LOGGER_LEVEL=DEBUG

# Profile performance
export TRACY_ENABLE=1
```

## Conclusion

The GPT-OSS MoE MLP implementation demonstrates sophisticated parallelization strategies that efficiently distribute computation across a multi-device mesh. The combination of expert parallelism, tensor parallelism, and dynamic batching via all-to-all communication enables scaling to thousands of experts while maintaining high hardware utilization.

Key takeaways:
- **ThroughputExperts** approach maximizes efficiency through dynamic batching
- **All-to-all communication** enables flexible token-to-expert routing
- **Multi-dimensional parallelism** (EP×TP) scales both model size and throughput
- **Memory hierarchy optimization** (L1 vs DRAM) balances latency and capacity
- **Fused operations** reduce memory bandwidth requirements

This architecture provides a foundation for scaling MoE models to trillion-parameter scales while maintaining practical inference latency.
