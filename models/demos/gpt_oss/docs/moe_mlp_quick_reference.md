# GPT-OSS MoE MLP: Quick Reference Guide

## Test Command
```bash
# Setup environment
cd /home/ntarafdar/tt-moe/tt-metal && source python_env/bin/activate
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD MESH_DEVICE=TG

# Run the specific test
pytest models/demos/gpt_oss/tests/unit/test_modules.py \
  -k "4x8 and decode_high_throughput and layer_0" \
  --test-modules mlp -xvs
```

## Configuration at a Glance

| Parameter | Value | Description |
|-----------|-------|-------------|
| Device Mesh | 4×8 | 32 devices total |
| Expert Parallelism (EP) | 4 | Experts across rows |
| Tensor Parallelism (TP) | 8 | Weights across columns |
| Total Experts | 128 | 4 experts per device |
| Top-K | 8 | Experts per token |
| Batch Size | 128 | Total tokens |
| Hidden Size | 2880 | Model dimension |
| Intermediate Size | 6144 | FFN dimension |

## Tensor Shapes Through Pipeline

| Stage | Shape | Description |
|-------|-------|-------------|
| Input | `[128, 1, 2880]` | Original batch |
| After Row Shard | `[32, 1, 2880]` | Per row (EP=4) |
| TTNN Format | `[1, 1, 32, 2880]` | 4D tensor |
| Router Output | `[1, 1, 32, 8]` | Top-K weights |
| After Dispatch | `[1, 4, N, 2880]` | N tokens per expert |
| After MLP | `[1, 4, N, 2880]` | Expert outputs |
| After Combine | `[8, 1, 32, 2880]` | K outputs per token |
| Final Output | `[1, 1, 32, 2880]` | Aggregated result |

## Key Files Map

```
models/demos/gpt_oss/
├── tt/
│   ├── mlp.py                    # Main MLP class
│   ├── topk.py                   # Router logic
│   └── experts_throughput/
│       ├── __init__.py           # ThroughputExperts
│       ├── decode.py             # Decode forward pass
│       └── config.py             # All-to-all configs
└── tests/
    ├── unit/
    │   └── test_modules.py       # Test entry point
    └── test_factory.py           # Test utilities
```

## Communication Patterns

### All-to-All (Token Dispatch/Combine)
```
Direction: Along rows (axis=0, EP dimension)
Topology: Ring with 4 links
Purpose: Route tokens ↔ experts
```

### All-Reduce (Tensor Parallel)
```
Direction: Along columns (axis=1, TP dimension)
Topology: Ring with 8 links
Purpose: Sum partial results
```

## Device-Expert Mapping

```python
def get_experts_on_device(row, col):
    device_id = row * 8 + col
    start = device_id * 4
    return list(range(start, start + 4))

# Examples:
# Device (0,0): Experts [0,1,2,3]
# Device (1,0): Experts [32,33,34,35]
# Device (3,7): Experts [124,125,126,127]
```

## Memory Configurations

### Decode Mode (L1)
```python
memory_config = ttnn.L1_MEMORY_CONFIG
decode_mode = True
# Low latency, limited capacity
```

### Prefill Mode (DRAM)
```python
memory_config = ttnn.DRAM_MEMORY_CONFIG
decode_mode = False
# High capacity, higher latency
```

## MLP Operations Sequence

1. **Gate/Up Projection (Fused)**
   - Input: `[B, 1, T, 2880]`
   - Weights: `[2880, 12288]` (6144×2)
   - Output: `[B, 1, T, 12288]`

2. **SwiGLU Activation**
   - Split: gate=`[:6144]`, up=`[6144:]`
   - Apply: `silu(gate) * up`
   - Output: `[B, 1, T, 6144]`

3. **Down Projection**
   - Input: `[B, 1, T, 6144]`
   - Weights: `[6144, 2880]`
   - Output: `[B, 1, T, 2880]`

## Debugging Checklist

- [ ] Environment variables set (`PYTHONPATH`, `TT_METAL_HOME`, `MESH_DEVICE`)
- [ ] Python environment activated
- [ ] Device mesh initialized (4×8)
- [ ] Model weights loaded correctly
- [ ] All-to-all rings configured
- [ ] Memory config matches mode (L1/DRAM)
- [ ] Top-K value within expert range
- [ ] Batch size divisible by EP

## Performance Metrics

| Metric | Typical Value | Notes |
|--------|--------------|-------|
| All-to-All Latency | ~1-2ms | Ring communication |
| MLP Compute | ~0.5-1ms | Per expert batch |
| All-Reduce | ~0.5ms | TP aggregation |
| Total Latency | ~3-5ms | Full forward pass |
| Memory (L1) | ~1MB | Per device |
| Memory (DRAM) | ~100MB | Per device |

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| OOM in L1 | Batch too large | Reduce batch or use DRAM |
| All-to-all timeout | Ring broken | Check device connectivity |
| Shape mismatch | Wrong parallelism | Verify EP×TP = devices |
| NaN outputs | Numerical overflow | Enable clamping |
| Slow performance | Wrong memory | Use L1 for decode |

## Environment Setup Script

```bash
#!/bin/bash
# Save as setup_moe_env.sh

cd /home/ntarafdar/tt-moe/tt-metal
source python_env/bin/activate

export PYTHONPATH=$PWD
export TT_METAL_HOME=$PWD
export MESH_DEVICE=TG

# Optional: Enable debug
# export TTNN_DEBUG=1
# export TT_METAL_LOGGER_LEVEL=DEBUG

echo "MoE environment ready!"
echo "Mesh: 4x8 (32 devices)"
echo "Run test with: pytest models/demos/gpt_oss/tests/unit/test_modules.py -k '4x8 and decode_high_throughput and layer_0' --test-modules mlp"
```

## Tensor Parallelism Sharding

```
Original Weight: [2880, 6144]
                    ↓
Device Col 0: [360, 768]  (2880/8, 6144/8)
Device Col 1: [360, 768]
...
Device Col 7: [360, 768]
```

## Expert Assignment Formula

```python
# For 128 experts on 32 devices (4×8 mesh)
experts_per_device = 128 // 32 = 4

# Device at position (row, col)
device_id = row * 8 + col
expert_start = device_id * 4
experts = [expert_start + i for i in range(4)]
```

## All-to-All Data Flow

```
Step 1: Local tokens → Expert assignments
Step 2: Send tokens to remote expert devices
Step 3: Process tokens with local experts
Step 4: Return outputs to original devices
Step 5: Aggregate K expert outputs per token
```

## Useful Pytest Markers

```bash
# Run all MoE tests
pytest -k "moe"

# Run decode tests only
pytest -k "decode"

# Run 4x8 mesh tests
pytest -k "4x8"

# Run with specific layer
pytest -k "layer_0"

# Combine conditions
pytest -k "4x8 and decode and mlp"
```

## Code Navigation Tips

1. **Start with test**: `test_modules.py:565` - See full flow
2. **Trace router**: `topk.py` - Understand token assignment
3. **Follow dispatch**: `experts_throughput/__init__.py:270` - All-to-all logic
4. **Expert compute**: `experts_throughput/decode.py:30` - MLP implementation
5. **Configuration**: `experts_throughput/config.py` - Tuning parameters
