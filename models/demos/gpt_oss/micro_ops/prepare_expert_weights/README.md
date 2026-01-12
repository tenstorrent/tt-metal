# PrepareGptOssExpertsTensor Micro-Op

## Overview

This fused micro-op transforms MoE (Mixture of Experts) routing weights from `[B, 1, S, K]` to `[K, 1, B*S, H]` format for efficient element-wise multiplication with expert outputs.

### What It Replaces

The original unfused implementation required 5 separate TTNN operations:

```python
def prepare_expert_weights_original(topk_expert_weights, num_experts_per_tok, hidden_size):
    topk_expert_weights = ttnn.reshape(topk_expert_weights, (-1, 1, 1, num_experts_per_tok))
    topk_weights_rm = ttnn.to_layout(topk_expert_weights, ttnn.ROW_MAJOR_LAYOUT)
    topk_weights_rm = ttnn.repeat(topk_weights_rm, ttnn.Shape((1, 1, hidden_size, 1)))
    topk_weights_rm = ttnn.permute(topk_weights_rm, (3, 1, 0, 2))
    topk_weights_reshaped = ttnn.to_layout(topk_weights_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(topk_weights_rm)
    return topk_weights_reshaped
```

This fused version performs all operations in a single kernel dispatch, avoiding:
- 4 intermediate tensor allocations
- Multiple memory round-trips through DRAM
- Layout conversion overhead

## How It Works

### Data Transformation

```
Input:  [B*S, K] routing weights (one weight per expert per token)
Output: [K, 1, B*S, H] broadcast weights (each weight repeated H times)

For each input weight at position [bs, k]:
    output[k, 0, bs, 0:H] = input[bs, k]  (broadcast across hidden dim)
```

### Implementation Strategy

1. **Reader Kernel**: Signals that the sharded input buffer is ready
2. **Compute Kernel**:
   - Reads each scalar weight from input tile
   - Creates a "scalar tile" filled with that weight value
   - Copies the scalar tile to output for each H-dimension tile
3. **Writer Kernel**: Waits for all output tiles to be written

### Memory Layout

- **Input**: `[B*S, K]` stored in L1 with 1x32 tiles (ROW_MAJOR layout)
  - Allows direct scalar access via L1 pointer
  - B*S and K must each fit within one tile dimension (≤32)

- **Output**: `[K, 1, B*S, H]` stored in L1 with 32x32 tiles (TILE layout)
  - Standard tile format for subsequent compute operations
  - Total tiles = K × B*S × ceil(H/32)

### Kernel Variants

| Variant | Description | Best For |
|---------|-------------|----------|
| `SingleCore` | Basic implementation | Small tensors, debugging |
| `Pipelined` | Block processing, 32-bit writes | Production use |
| `MultiCore` | Distributed across cores | Large K values |

## API Reference

### PrepareGptOssExpertsTensorSingleCore

```python
class PrepareGptOssExpertsTensorSingleCore:
    @staticmethod
    def golden(topk_expert_weights, num_experts_per_tok, hidden_size) -> torch.Tensor:
        """PyTorch reference implementation."""

    @staticmethod
    def op(input_tensor, output_tensor, num_experts_per_tok, hidden_size) -> ttnn.Tensor:
        """Execute fused operation on device."""
```

### PrepareGptOssExpertsTensorPipelined

```python
class PrepareGptOssExpertsTensorPipelined:
    @staticmethod
    def op(
        input_tensor,
        output_tensor,
        num_experts_per_tok,
        hidden_size,
        tiles_per_block=4,  # Tiles processed per dst register acquire
    ) -> ttnn.Tensor:
        """Execute optimized fused operation."""
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_tensor` | `ttnn.Tensor` | Routing weights `[B*S, K]`, sharded in L1 |
| `output_tensor` | `ttnn.Tensor` | Pre-allocated output `[K, 1, B*S, H]`, sharded in L1 |
| `num_experts_per_tok` | `int` | Number of experts selected per token (K) |
| `hidden_size` | `int` | Hidden dimension size (H) |
| `tiles_per_block` | `int` | (Pipelined only) Tiles per register acquire cycle |

## Usage Example

```python
import torch
import ttnn
from models.demos.gpt_oss.micro_ops.prepare_expert_weights import (
    PrepareGptOssExpertsTensorPipelined,
)

# Configuration
batch_seq = 32  # B * S
num_experts_per_tok = 8  # K
hidden_size = 7168  # H

# Create input weights [B*S, K]
torch_input = torch.randn(batch_seq, num_experts_per_tok, dtype=torch.bfloat16)

# Setup sharding for single core
input_shard_spec = ttnn.ShardSpec(
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
    (batch_seq, num_experts_per_tok),
    ttnn.ShardOrientation.ROW_MAJOR,
)
input_mem_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    input_shard_spec,
)

# Create TTNN input tensor
ttnn_input = ttnn.from_torch(
    torch_input,
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=input_mem_config,
    tile=ttnn.Tile((1, 32)),
)

# Create pre-allocated output [K, 1, B*S, H]
output_shape = (num_experts_per_tok, 1, batch_seq, hidden_size)
torch_output = torch.zeros(output_shape, dtype=torch.bfloat16)

output_shard_spec = ttnn.ShardSpec(
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
    (num_experts_per_tok * batch_seq, hidden_size),
    ttnn.ShardOrientation.ROW_MAJOR,
)
output_mem_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    output_shard_spec,
)

ttnn_output = ttnn.from_torch(
    torch_output,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=output_mem_config,
)

# Execute fused op
result = PrepareGptOssExpertsTensorPipelined.op(
    ttnn_input,
    ttnn_output,
    num_experts_per_tok=num_experts_per_tok,
    hidden_size=hidden_size,
    tiles_per_block=4,
)

# Verify against golden
expected = PrepareGptOssExpertsTensorPipelined.golden(
    torch_input, num_experts_per_tok, hidden_size
)
result_torch = ttnn.to_torch(result)
```

## Testing

### Running Tests

```bash
# Run all tests for this op
pytest models/demos/gpt_oss/micro_ops/prepare_expert_weights/tests/test_prepare_expert_weights.py -v

# Run specific test class
pytest models/demos/gpt_oss/micro_ops/prepare_expert_weights/tests/test_prepare_expert_weights.py::TestPrepareExpertWeightsSingleCore -v

# Run golden-only tests (no device required)
pytest models/demos/gpt_oss/micro_ops/prepare_expert_weights/tests/test_prepare_expert_weights.py::TestPrepareExpertWeightsSingleCore::test_golden_function -v
pytest models/demos/gpt_oss/micro_ops/prepare_expert_weights/tests/test_prepare_expert_weights.py::test_broadcast_correctness -v
```

### Test Coverage

| Test | Description | Requires Device |
|------|-------------|-----------------|
| `test_golden_function` | Validates PyTorch reference implementation | No |
| `test_broadcast_correctness` | Verifies broadcast values match across H dim | No |
| `test_fused_op` | Compares fused kernel output to golden | Yes |
| `test_matches_original` | Compares fused op to unfused TTNN ops | Yes |

### Test Parameters

The tests cover various configurations:
- `batch_seq`: 1, 4, 8, 16, 32
- `num_experts_per_tok`: 4, 8
- `hidden_size`: 256, 512, 1024, 2048, 7168

## Performance Considerations

### Optimizations in Pipelined Variant

1. **32-bit Packed Writes**: Fills scalar tiles 2x faster by writing two bf16 values per 32-bit write
2. **Block Processing**: Reduces `tile_regs_acquire/release` overhead by processing multiple output tiles per cycle
3. **Single Scalar Tile**: Creates one scalar tile per (k, bs) pair, then copies to all H positions

### Memory Requirements

```
Input L1:  B*S × K × 2 bytes (bfloat16)
Output L1: K × B*S × H × 2 bytes (bfloat16)
Scalar CB: 32 × 32 × 2 = 2KB (one tile buffer)
```

### Constraints

- `B*S ≤ 32` (must fit in one tile height for single-core variant)
- `K ≤ 32` (must fit in one tile width)
- Both tensors must be sharded on the same core(s)

## File Structure

```
models/demos/gpt_oss/micro_ops/prepare_expert_weights/
├── __init__.py                    # Exports op classes
├── op.py                          # Python op definitions
├── README.md                      # This file
├── kernels/
│   ├── reader.cpp                 # Reader kernel
│   ├── writer.cpp                 # Writer kernel
│   ├── compute.cpp                # Basic compute kernel
│   ├── compute_optimized.cpp      # Optimized with mul_tiles_bcast
│   └── compute_pipelined.cpp      # Block-processed compute kernel
└── tests/
    ├── __init__.py
    └── test_prepare_expert_weights.py  # Pytest tests
```

## Extending This Op

### Adding Multi-Core Support

To extend for larger `B*S` or `K` values:

1. Shard input by K dimension (each core handles subset of experts)
2. Shard output correspondingly
3. No inter-core communication needed (embarrassingly parallel)

### Fusing with Downstream Ops

This op's output is typically used in element-wise multiplication:
```python
weighted_output = expert_outputs * prepared_weights  # [K, 1, B*S, H] × [K, 1, B*S, H]
```

To fuse further, extend the compute kernel to:
1. Read both routing weights AND expert outputs
2. Perform broadcast multiply in-place
3. Write final weighted outputs

## Troubleshooting

### Common Issues

**"B*S must fit in input tile height"**
- Current single-core implementation requires B*S ≤ 32
- Use multi-core variant for larger batch sizes

**PCC mismatch in tests**
- bfloat16 precision limits accuracy to ~0.98-0.99 PCC
- Check for tile padding issues if PCC < 0.95

**Kernel hang**
- Verify input/output tensor sharding matches kernel expectations
- Check CB sizes are sufficient for tile counts
