# DeepSeek V3 MoE Tensor Flow and Parallelism Architecture

## Executive Summary

DeepSeek V3 introduces a revolutionary approach to Mixture of Experts (MoE) architectures through its hierarchical GroupedTopK routing mechanism. Unlike traditional flat routing approaches, DeepSeek V3 organizes its 256 routed experts into 8 groups of 32 experts each, enabling a two-stage selection process that significantly improves both computational efficiency and model quality.

The architecture features 256 routed experts plus 1 shared expert, where the shared expert processes all tokens in parallel with the routed experts to retain common knowledge. This hybrid approach allows the model to maintain general capabilities while specializing through the routed experts. With support for both TG (4×8) and QUAD (16×8) device meshes, DeepSeek V3 can scale from 32 to 128 devices while maintaining efficient all-to-all communication patterns.

The implementation leverages a unified MoEBlock that supports multiple backends, enabling seamless integration with both DeepSeek and GPT-OSS architectures. Through careful orchestration of expert parallelism (EP) and tensor parallelism (TP), the system achieves over 99.99% accuracy while efficiently distributing computation across the device mesh.

## System Architecture

### Device Mesh Configurations

DeepSeek V3 supports two primary device mesh configurations:

#### TG (Torus Galaxy) Configuration: 4×8 Mesh
- **Total Devices**: 32 (4 rows × 8 columns)
- **Expert Parallelism (EP)**: 4 (distributed across rows, axis=0)
- **Tensor Parallelism (TP)**: 8 (distributed across columns, axis=1)
- **Experts per EP group**: 256 / 4 = 64 experts
- **Experts per device**: 256 / 32 = 8 experts

```
TG Device Mesh (4×8):
    TP0   TP1   TP2   TP3   TP4   TP5   TP6   TP7
EP0 [0-7] [0-7] [0-7] [0-7] [0-7] [0-7] [0-7] [0-7]   <- Experts 0-63
EP1 [64-71][64-71][64-71][64-71][64-71][64-71][64-71][64-71] <- Experts 64-127
EP2 [128-135][128-135][128-135][128-135][128-135][128-135][128-135][128-135] <- Experts 128-191
EP3 [192-199][192-199][192-199][192-199][192-199][192-199][192-199][192-199] <- Experts 192-255
```

#### QUAD Configuration: 16×8 Mesh
- **Total Devices**: 128 (16 rows × 8 columns)
- **Expert Parallelism (EP)**: 16 (distributed across rows, axis=0)
- **Tensor Parallelism (TP)**: 8 (distributed across columns, axis=1)
- **Experts per EP group**: 256 / 16 = 16 experts
- **Experts per device**: 256 / 128 = 2 experts

```
QUAD Device Mesh (16×8):
     TP0   TP1   TP2   TP3   TP4   TP5   TP6   TP7
EP0  [0-1] [0-1] [0-1] [0-1] [0-1] [0-1] [0-1] [0-1]   <- Experts 0-15
EP1  [16-17][16-17][16-17][16-17][16-17][16-17][16-17][16-17] <- Experts 16-31
...
EP15 [240-241][240-241][240-241][240-241][240-241][240-241][240-241][240-241] <- Experts 240-255
```

### MoE Components

The DeepSeek V3 MoE architecture consists of five key components:

1. **GroupedTopKRouter (MoEGate)**
   - Location: `models/tt_moe/components/routers/grouped_topk_router.py`
   - Function: Hierarchical two-stage expert selection
   - Innovation: Groups experts for efficient routing

2. **RoutedExperts**
   - Location: `models/tt_moe/components/experts/routed_experts.py`
   - Function: 256 specialized expert MLPs
   - Architecture: Gate + Up projections → SiLU → Down projection

3. **SharedExpert**
   - Location: `models/tt_moe/components/experts/shared_expert.py`
   - Function: Single expert processing all tokens
   - Intermediate size: 10752 (larger than routed experts' 2048)

4. **MoEPreamble**
   - Location: `models/tt_moe/components/moe_preamble.py`
   - Function: Preprocessing for all-to-all communication
   - Operations: Weight transformation and index preparation

5. **Unified MoEBlock**
   - Location: `models/tt_moe/moe_block.py`
   - Function: Orchestrates entire MoE computation
   - Feature: Configurable backend support (DeepSeek/GPT-OSS)

## Complete Tensor Journey

### Stage 1: Input and TP Gather

The journey begins with a TP-sharded input tensor:

```python
# Input shape (TP-sharded)
input: [batch, 1, seq_len, hidden_size // TP]
# For DeepSeek: [batch, 1, seq_len, 7168 // 8] = [batch, 1, seq_len, 896]

# All-gather across TP dimension to reconstruct full hidden size
gathered_input = ttnn.all_gather(input, dim=3, cluster_axis=1)  # TP axis
# Result: [batch, 1, seq_len, 7168]
```

### Stage 2: Router Computation (GroupedTopKRouter)

The GroupedTopKRouter performs hierarchical selection through multiple steps:

```python
# Step 1: Linear projection to expert scores
# Input: [batch, seq_len, 7168]
# Weight: [7168, 256]
scores = input @ router_weight  # [batch, seq_len, 256]

# Step 2: Apply score correction bias
scores = scores + score_bias  # Learned per-expert biases

# Step 3: Reshape into groups
# 256 experts = 8 groups × 32 experts/group
group_scores = scores.reshape(batch, seq_len, 8, 32)

# Step 4: Within-group TopK selection (K=2)
within_group_topk = topk(group_scores, k=2, dim=-1)
# Result: Top-2 experts from each of 8 groups = 16 candidates

# Step 5: Group selection (K=4)
group_scores_max = max(within_group_topk, dim=-1)
selected_groups = topk(group_scores_max, k=4, dim=-1)
# Result: Top-4 groups selected

# Step 6: Final expert selection
# From the 4 selected groups, take the top-2 experts from each
# Total: 4 groups × 2 experts = 8 experts selected

# Step 7: Normalize routing weights
routing_weights = softmax(selected_scores) * routed_scaling_factor
# routed_scaling_factor = 2.5 for DeepSeek V3
```

Key differences from flat routing:
- Two-stage selection reduces computational complexity
- Groups provide semantic clustering of experts
- Scaling factor ensures proper gradient flow

### Stage 3: MoE Preamble

The preamble prepares data for all-to-all communication:

```python
# Input transformations
# Original shape: [batch, seq_len, hidden_size]
# After preamble: [batch * seq_len * num_experts_per_tok, hidden_size]

# Weight transformations
# routing_weights: [batch, seq_len, num_experts_per_tok]
# After repeat and permute operations:
# weights_repeated: [batch * seq_len * num_experts_per_tok, 1]

# Index preparation for dispatch
# expert_indices: [batch, seq_len, num_experts_per_tok]
# dispatch_indices: flattened and prepared for all-to-all
```

### Stage 4: All-to-All Dispatch

The all-to-all operation routes tokens to their selected experts:

```python
# Communication along EP dimension (axis=0)
dispatched = ttnn.experimental.all_to_all(
    preprocessed_input,
    cluster_axis=0,  # EP axis
    concat_dimension=0,
    split_dimension=0,
    memory_config=mem_cfg
)

# Each device now has tokens assigned to its local experts
# Shape per device: [num_tokens_for_local_experts, hidden_size]
```

The all-to-all ensures:
- Tokens are sent to devices hosting their selected experts
- Load balancing through auxiliary loss during training
- Efficient batching of tokens per expert

### Stage 5: Expert MLP Computation (RoutedExperts)

Each expert processes its assigned tokens:

```python
# For each expert on the device:
def expert_mlp(x):
    # Input: [num_tokens, 7168]

    # Gate projection
    gate = x @ W_gate  # [num_tokens, 7168] → [num_tokens, 2048]

    # Up projection
    up = x @ W_up      # [num_tokens, 7168] → [num_tokens, 2048]

    # SiLU activation (no clamping unlike GPT-OSS)
    hidden = silu(gate) * up  # [num_tokens, 2048]

    # Down projection
    output = hidden @ W_down  # [num_tokens, 2048] → [num_tokens, 7168]

    return output
```

Key architectural choices:
- Intermediate size: 2048 (smaller than shared expert's 10752)
- Activation: Pure SiLU without clamping
- Weight distribution: Each expert's weights stored on specific device

### Stage 6: Shared Expert Computation

In parallel with routed experts, the shared expert processes all tokens:

```python
def shared_expert_mlp(x):
    # Input: [batch, seq_len, 7168]

    # Larger intermediate dimension
    gate = x @ W_gate_shared  # → [batch, seq_len, 10752]
    up = x @ W_up_shared      # → [batch, seq_len, 10752]

    # SiLU activation
    hidden = silu(gate) * up

    # Down projection
    output = hidden @ W_down_shared  # → [batch, seq_len, 7168]

    return output
```

The shared expert:
- Processes every token (not selective)
- Has 5.25× larger intermediate size
- Provides baseline knowledge for all inputs

### Stage 7: All-to-All Combine

Expert outputs are gathered back to original positions:

```python
# All-to-all to return outputs to original devices
combined = ttnn.experimental.all_to_all(
    expert_outputs,
    cluster_axis=0,  # EP axis
    concat_dimension=0,
    split_dimension=0
)

# Apply routing weights
weighted_output = combined * routing_weights

# Combine with shared expert output
final_output = weighted_output + shared_expert_output
```

### Stage 8: Reduce-Scatter

Finally, the output is distributed across TP dimension:

```python
# Reduce-scatter to shard output across TP
output = ttnn.reduce_scatter(
    final_output,
    cluster_axis=1,  # TP axis
    scatter_split_dim=3,  # Hidden dimension
    reduce_op=ReduceOpType.Sum
)

# Final shape: [batch, 1, seq_len, hidden_size // TP]
# For DeepSeek: [batch, 1, seq_len, 896]
```

## Parallelism Strategies

### Expert Parallelism (EP)

Expert parallelism distributes the 256 experts across device rows:

**TG Configuration (EP=4)**:
- Row 0: Experts 0-63 (64 experts)
- Row 1: Experts 64-127 (64 experts)
- Row 2: Experts 128-191 (64 experts)
- Row 3: Experts 192-255 (64 experts)

**QUAD Configuration (EP=16)**:
- Each row: 16 experts (256 / 16)
- More fine-grained distribution
- Better load balancing for large batches

**Expert Location Formula**:
```python
def get_expert_location(expert_id, ep_size):
    ep_rank = expert_id // (256 // ep_size)
    local_expert_id = expert_id % (256 // ep_size)
    return ep_rank, local_expert_id
```

### Tensor Parallelism (TP)

Tensor parallelism shards weights across device columns:

**Weight Sharding**:
- Hidden dimension: 7168 / 8 = 896 per device
- Intermediate dimension: 2048 / 8 = 256 per device
- Shared expert intermediate: 10752 / 8 = 1344 per device

**Communication Patterns**:
- All-gather: Reconstruct full tensors before router
- Reduce-scatter: Distribute results after MoE
- All-reduce: During expert computation if needed

## Communication Patterns

### All-to-All Operations

The all-to-all is the core communication primitive for MoE:

```python
# Configuration for all-to-all
all_to_all_config = {
    "cluster_axis": 0,  # EP axis
    "concat_dimension": 0,
    "split_dimension": 0,
    "num_links": 4,  # CCL links
    "memory_config": ttnn.L1_MEMORY_CONFIG  # For decode
}
```

**Dispatch Phase**:
- Each device sends tokens to devices hosting selected experts
- Batching ensures efficient expert utilization
- Non-blocking communication overlaps with computation

**Combine Phase**:
- Expert outputs return to originating devices
- Routing weights applied after gathering
- Synchronized before final combination

### TP Communication

Tensor parallel operations use different collectives:

```python
# All-gather for input reconstruction
ttnn.all_gather(input, dim=3, cluster_axis=1, num_links=1)

# Reduce-scatter for output distribution
ttnn.reduce_scatter(output, cluster_axis=1, scatter_split_dim=3, num_links=1)
```

## Memory Optimization

### Decode Mode (Single Token)

For decode mode with small batch sizes:
- **Memory**: L1 (fastest on-chip memory)
- **Strategy**: Keep all intermediate activations in L1
- **Configuration**: `ttnn.L1_MEMORY_CONFIG`

### Prefill Mode (Multiple Tokens)

For prefill with larger sequences:
- **Memory**: DRAM (larger capacity)
- **Strategy**: Stream through DRAM for large activations
- **Configuration**: `ttnn.DRAM_MEMORY_CONFIG`

### Expert Weight Distribution

Expert weights are distributed to minimize memory per device:

**TG (8 experts/device)**:
- Per-expert memory: ~60MB (FP16/BFP16)
- Total per device: ~480MB
- Fits comfortably in DRAM

**QUAD (2 experts/device)**:
- Per-expert memory: ~60MB
- Total per device: ~120MB
- More headroom for activations

## Key Implementation Files

The implementation is organized across several key modules:

**Core MoE Components**:
- `models/tt_moe/moe_block.py`: Unified MoE block supporting multiple backends
- `models/tt_moe/components/routers/grouped_topk_router.py`: Hierarchical routing
- `models/tt_moe/components/experts/routed_experts.py`: 256 expert MLPs
- `models/tt_moe/components/experts/shared_expert.py`: Shared expert MLP
- `models/tt_moe/components/moe_preamble.py`: Preprocessing layer

**DeepSeek Integration**:
- `models/demos/deepseek_v3/tt/decoder_block/moe_decoder_block_2d.py`: MoE layer wrapper
- `models/demos/deepseek_v3/tt/decoder_block/decoder_block_tt_2d.py`: Full decoder block
- `models/demos/deepseek_v3/reference/decoder_block.py`: Reference implementation

**Configuration and Testing**:
- `models/demos/deepseek_v3/reference/configuration_deepseek.py`: Model configuration
- `models/demos/deepseek_v3/tests/test_decoder_block.py`: Comprehensive tests
- `models/demos/deepseek_v3/tests/scripts/run_moe_decoder_block_test_2d.sh`: Test runner

## Performance Characteristics

### Accuracy Requirements

The implementation achieves:
- **PCC (Pearson Correlation Coefficient)**: > 0.9999 (99.99%)
- **Numerical precision**: BFP16 for efficiency
- **Validation**: Against HuggingFace reference implementation

### Scaling Efficiency

**TG to QUAD Scaling**:
- 4× increase in devices (32 → 128)
- 4× increase in EP (4 → 16)
- Constant TP (8 → 8)
- Near-linear scaling for large batches

### Latency Considerations

**Decode Mode**:
- Single token latency dominated by all-to-all
- L1 memory reduces activation movement
- Expert selection overhead minimal

**Prefill Mode**:
- Parallel processing of sequence
- DRAM bandwidth becomes bottleneck
- Batching improves throughput

## Differences from GPT-OSS Implementation

### Architectural Differences

| Feature | DeepSeek V3 | GPT-OSS |
|---------|-------------|---------|
| Router Type | GroupedTopK (hierarchical) | TopK (flat) |
| Number of Experts | 256 routed + 1 shared | 128 routed |
| Expert Selection | 2-stage (groups → experts) | Direct selection |
| Experts per Token | 8 | 2-8 (configurable) |
| Activation Function | SiLU | SwiGLU (clamped) |
| Scaling Factor | 2.5 | Variable |

### Implementation Differences

1. **Unified Backend**: MoEBlock supports both architectures through configuration
2. **Router Logic**: GroupedTopK adds group-level selection stage
3. **Shared Expert**: Additional parallel computation path
4. **Memory Layout**: Different expert distribution patterns

## Conclusion

The DeepSeek V3 MoE architecture represents a significant advancement in efficient large-scale neural network design. Through its innovative GroupedTopK routing mechanism and hybrid expert architecture, it achieves exceptional model quality while maintaining computational efficiency. The implementation's support for multiple device configurations (TG and QUAD) and unified backend demonstrates the flexibility needed for production deployments at various scales.

The careful orchestration of expert and tensor parallelism, combined with optimized communication patterns, enables the system to scale from 32 to 128 devices while maintaining over 99.99% accuracy. This documentation provides the technical foundation for understanding, deploying, and extending the DeepSeek V3 MoE implementation in the TT-Metal framework.
