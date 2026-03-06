# DeepSeek V3 MoE Quick Reference Guide

## Environment Setup

### Quick Setup (One-liner)
```bash
cd /home/ntarafdar/tt-moe/tt-metal && source python_env/bin/activate && export PYTHONPATH=$PWD TT_METAL_HOME=$PWD MESH_DEVICE=TG DEEPSEEK_V3_HF_MODEL=/data/deepseek/DeepSeek-R1-0528 DEEPSEEK_V3_CACHE=/tmp/deepseek_cache_$(date +%Y%m%d_%H%M%S)_$$ && mkdir -p $DEEPSEEK_V3_CACHE
```

### Step-by-Step Setup
```bash
# 1. Navigate to tt-metal root
cd /home/ntarafdar/tt-moe/tt-metal

# 2. Activate Python environment
source python_env/bin/activate

# 3. Set environment variables
export PYTHONPATH=$PWD
export TT_METAL_HOME=$PWD
export MESH_DEVICE=TG  # or QUAD for 16×8 mesh

# 4. Set DeepSeek model paths
export DEEPSEEK_V3_HF_MODEL=/data/deepseek/DeepSeek-R1-0528

# 5. Create test-specific cache directory
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache_$(date +%Y%m%d_%H%M%S)_$$
mkdir -p $DEEPSEEK_V3_CACHE
```

## Test Commands

### Basic MoE Block Test
```bash
# Run all MoE decoder block tests
pytest models/demos/deepseek_v3/tests/test_decoder_block.py -xvs

# Run specific layer test
pytest models/demos/deepseek_v3/tests/test_decoder_block.py::test_moe_decoder_block_2d -xvs

# Run with specific parameters
pytest models/demos/deepseek_v3/tests/test_decoder_block.py::test_moe_decoder_block_2d -xvs \
    --k "layer_idx=3 and mode='prefill' and device_mesh='TG'"
```

### Test with Different Configurations

#### TG Mesh (4×8)
```bash
export MESH_DEVICE=TG
pytest models/demos/deepseek_v3/tests/test_decoder_block.py::test_moe_decoder_block_2d \
    -k "device_mesh='TG'" -xvs
```

#### QUAD Mesh (16×8)
```bash
export MESH_DEVICE=QUAD
pytest models/demos/deepseek_v3/tests/test_decoder_block.py::test_moe_decoder_block_2d \
    -k "device_mesh='QUAD'" -xvs
```

### Debug Mode
```bash
# Enable trace logging
export TT_METAL_TRACE=1
export TT_METAL_TRACE_LEVEL=TRACE

# Run with debug output
pytest models/demos/deepseek_v3/tests/test_decoder_block.py::test_moe_decoder_block_2d \
    -xvs --capture=no
```

### Clean Up After Tests
```bash
# Remove cache directory
rm -rf $DEEPSEEK_V3_CACHE

# Reset devices if needed
tt-smi -glx_reset
```

## Configuration Parameters

### Model Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_size` | 7168 | Model hidden dimension |
| `moe_intermediate_size` | 2048 | Routed expert intermediate dimension |
| `shared_expert_intermediate_size` | 10752 | Shared expert intermediate dimension |
| `n_routed_experts` | 256 | Total number of routed experts |
| `n_shared_experts` | 1 | Number of shared experts |
| `num_experts_per_tok` | 8 | Experts selected per token |
| `n_group` | 8 | Number of expert groups |
| `topk_group` | 4 | Number of groups selected |
| `routed_scaling_factor` | 2.5 | Scaling factor for routed expert outputs |

### Device Mesh Configurations
| Mesh Type | Shape | Total Devices | EP | TP | Experts/Device |
|-----------|-------|---------------|----|----|----------------|
| TG | 4×8 | 32 | 4 | 8 | 8 |
| QUAD | 16×8 | 128 | 16 | 8 | 2 |

### Test Modes
| Mode | Batch Size | Sequence Length | Memory Config |
|------|------------|-----------------|---------------|
| Prefill (TG) | 1 | 128 | DRAM |
| Decode (TG) | 128 | 1 | L1 |
| Prefill (QUAD) | 1 | 128 | DRAM |
| Decode (QUAD) | 512 | 1 | L1 |

## Tensor Shape Reference

### Input/Output Shapes
```python
# Initial input (TP-sharded)
[batch, 1, seq_len, hidden_size // TP]
# TG/QUAD: [batch, 1, seq_len, 896]

# After TP all-gather
[batch, 1, seq_len, hidden_size]
# [batch, 1, seq_len, 7168]

# Router scores
[batch, seq_len, n_routed_experts]
# [batch, seq_len, 256]

# Selected indices
[batch, seq_len, num_experts_per_tok]
# [batch, seq_len, 8]

# Routing weights
[batch, seq_len, num_experts_per_tok]
# [batch, seq_len, 8]

# After MoE preamble
[batch * seq_len * num_experts_per_tok, hidden_size]
# [batch * seq_len * 8, 7168]

# Expert computation (per expert)
Input: [num_tokens, hidden_size] → [num_tokens, 7168]
Gate/Up: [num_tokens, moe_intermediate_size] → [num_tokens, 2048]
Output: [num_tokens, hidden_size] → [num_tokens, 7168]

# Shared expert computation
Input: [batch, seq_len, hidden_size] → [batch, seq_len, 7168]
Intermediate: [batch, seq_len, 10752]
Output: [batch, seq_len, hidden_size] → [batch, seq_len, 7168]

# Final output (TP-sharded)
[batch, 1, seq_len, hidden_size // TP]
# [batch, 1, seq_len, 896]
```

## Device Mapping Formulas

### Expert to Device Mapping
```python
def get_expert_device(expert_id, mesh_shape):
    """Map expert ID to device coordinates"""
    ep_size, tp_size = mesh_shape
    num_experts_per_ep = 256 // ep_size

    # EP dimension (row)
    ep_rank = expert_id // num_experts_per_ep

    # All TP devices in the row have same experts
    return (ep_rank, tp_rank) for tp_rank in range(tp_size)

# TG Example: Expert 100
# 100 // 64 = 1 (EP rank 1)
# Devices: (1,0), (1,1), ..., (1,7)

# QUAD Example: Expert 100
# 100 // 16 = 6 (EP rank 6)
# Devices: (6,0), (6,1), ..., (6,7)
```

### Token to Expert Routing
```python
def route_token_to_device(token_idx, expert_id, mesh_shape):
    """Determine which device processes a token-expert pair"""
    ep_size, tp_size = mesh_shape
    expert_device_ep = expert_id // (256 // ep_size)

    # Token goes to EP rank hosting the expert
    # All TP devices in that rank participate
    return expert_device_ep
```

## Common Errors and Fixes

### Error: "Cache directory not found"
```bash
# Fix: Create cache directory
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache_test
mkdir -p $DEEPSEEK_V3_CACHE
```

### Error: "Active ethernet dispatch core detected"
```bash
# Fix: Reset devices
pkill -9 pytest
tt-smi -glx_reset
```

### Error: "Model weights not found"
```bash
# Fix: Verify model path
ls -la /data/deepseek/DeepSeek-R1-0528
export DEEPSEEK_V3_HF_MODEL=/data/deepseek/DeepSeek-R1-0528
```

### Error: "Out of memory"
```bash
# Fix: Use DRAM for large tensors
# In test, change memory_config from L1 to DRAM
memory_config = ttnn.DRAM_MEMORY_CONFIG
```

### Error: "PCC below threshold"
```python
# Check for numerical issues
# 1. Verify weight loading
# 2. Check activation ranges
# 3. Enable debug logging for intermediate values
```

## Debugging Checklist

### Before Running Tests
- [ ] Environment activated: `source python_env/bin/activate`
- [ ] PYTHONPATH set: `export PYTHONPATH=$PWD`
- [ ] TT_METAL_HOME set: `export TT_METAL_HOME=$PWD`
- [ ] MESH_DEVICE set: `export MESH_DEVICE=TG` or `QUAD`
- [ ] Model path exists: `ls $DEEPSEEK_V3_HF_MODEL`
- [ ] Cache directory created: `mkdir -p $DEEPSEEK_V3_CACHE`

### During Test Failures
- [ ] Check device status: `tt-smi`
- [ ] Review error logs: Look for tensor shape mismatches
- [ ] Verify memory config: L1 for decode, DRAM for prefill
- [ ] Check expert distribution: Ensure experts map correctly
- [ ] Validate routing: Confirm K=8 experts selected

### After Tests
- [ ] Clean cache: `rm -rf $DEEPSEEK_V3_CACHE`
- [ ] Reset devices if hung: `tt-smi -glx_reset`
- [ ] Check for zombie processes: `ps aux | grep pytest`

## Performance Profiling

### Enable Profiling
```python
# In test code
import ttnn
ttnn.enable_profiling()

# Run test
# ...

# Get profile
ttnn.print_profiling_report()
```

### Key Metrics to Monitor
- All-to-all latency (dispatch and combine)
- Expert compute time
- Memory bandwidth utilization
- PCC accuracy per layer

## File Structure Reference

```
models/
├── tt_moe/                          # Unified MoE implementation
│   ├── moe_block.py                 # Main MoE block
│   ├── components/
│   │   ├── routers/
│   │   │   └── grouped_topk_router.py
│   │   ├── experts/
│   │   │   ├── routed_experts.py
│   │   │   └── shared_expert.py
│   │   └── moe_preamble.py
│   └── tests/
│       └── test_moe_block.py
│
└── demos/deepseek_v3/               # DeepSeek V3 specific
    ├── tt/
    │   └── decoder_block/
    │       ├── moe_decoder_block_2d.py
    │       └── decoder_block_tt_2d.py
    ├── reference/
    │   ├── configuration_deepseek.py
    │   └── decoder_block.py
    ├── tests/
    │   └── test_decoder_block.py
    └── docs/
        ├── moe_tensor_flow_deepseek.md
        ├── moe_quick_reference_deepseek.md
        └── moe_parallelism_diagrams_deepseek.md
```

## Quick Tips

1. **Always use test-specific cache directories** to avoid conflicts
2. **Run device reset** if tests hang unexpectedly
3. **Start with smaller batch sizes** when debugging
4. **Use DRAM memory config** for prefill mode
5. **Monitor PCC values** - should be > 0.9999
6. **Check expert distribution** matches expected pattern
7. **Verify all environment variables** before running tests
8. **Clean up cache directories** after test runs

## Support Resources

- TT-Metal Documentation: Internal wiki
- MoE Architecture Guide: `moe_tensor_flow_deepseek.md`
- Visual Diagrams: `moe_parallelism_diagrams_deepseek.md`
- Test Scripts: `models/demos/deepseek_v3/tests/scripts/`
