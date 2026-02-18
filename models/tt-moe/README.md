# Unified MoE Block Implementation

This directory contains a unified, configurable Mixture of Experts (MoE) implementation that can support multiple architectures through JSON configuration files.

## Directory Structure

```
tt-moe/
├── moe_block.py              # Main unified MoE block
├── configs/
│   ├── deepseek_v3.json     # DeepSeek-V3 configuration
│   └── gpt_oss.json        # GPT-OSS configuration
├── components/
│   ├── routers/
│   │   ├── base_router.py   # Abstract router interface
│   │   ├── moe_gate.py      # DeepSeek MoEGate implementation
│   │   └── topk_router.py   # GPT-OSS TopK router
│   ├── experts/
│   │   ├── base_expert.py   # Abstract expert interface
│   │   ├── distributed_expert.py  # Unified expert with configurable activation
│   │   └── shared_expert.py      # DeepSeek shared expert
│   └── collective/
│       └── all_to_all_ops.py # All-to-all operations
├── utils/
│   ├── ccl.py               # CCL utilities
│   └── lazy_state_dict.py  # Lazy loading utilities
└── tests/
    ├── test_deepseek_moe_block.py  # DeepSeek tests (PCC: 0.989)
    ├── test_gpt_oss_moe_block.py   # GPT-OSS tests
    └── test_moe_components.py      # Component unit tests
```

## DeepSeek-V3 Implementation

The unified MoE block fully replicates the functionality of `models/demos/deepseek_v3/tt/decoder_block/moe_decoder_block_2d.py`.

### Original Implementation Flow

From `moe_decoder_block_2d.py:forward_mlp_prefill()`:

1. **Check if input is TP-sharded**: `x_dim == hidden_size // tp_size`
2. **All-gather if sharded**: `ttnn.experimental.all_gather_async()` on cluster_axis=1
3. **Run MoE**: `MoE.forward_prefill(x_gathered)`
4. **Run SharedExpert**: `SharedExpert.forward_prefill(x_gathered)`
5. **Combine outputs**: `ttnn.add(mlp_out, shared_expert_out)`
6. **Reduce-scatter if gathered**: `ttnn.experimental.reduce_scatter_minimal_async()` on cluster_axis=1

### Unified MoEBlock Flow

The `MoEBlock.forward()` method implements the exact same logic:

1. **TP sharding detection**: `_is_tp_sharded(x)` checks dimension
2. **All-gather if needed**: Based on `tensor_parallel.cluster_axis: 1`
3. **Router forward**: MoEGate router with score corrections
4. **Expert computation**:
   - Distributed experts with all-to-all on `dispatch_cluster_axis: 0`
   - Shared expert runs in parallel (`parallel_with_moe: true`)
5. **Combine outputs**: Automatic addition of expert outputs
6. **Reduce-scatter if gathered**: On same TP axis

### Configuration Mapping

The `configs/deepseek_v3.json` file maps all DeepSeek-V3 parameters:

```json
{
  "tensor_parallel": {
    "enabled": true,
    "cluster_axis": 1  // TP on mesh columns
  },
  "router": {
    "type": "moe_gate",  // DeepSeek's MoEGate
    "config": {
      "score_correction_bias": true,
      "routed_scaling_factor": 1.0
    }
  },
  "experts": {
    "distributed": {
      "dispatch_cluster_axis": 0  // EP on mesh rows
    },
    "shared": {
      "enabled": true,
      "parallel_with_moe": true  // Runs in parallel
    }
  }
}
```

### Key Design Points

1. **Tensor Parallelism (TP)**: Always follows the pattern all-gather → compute → reduce-scatter. Only the cluster axis needs to be specified.

2. **Expert Parallelism (EP)**: Uses all-to-all dispatch/combine on a different axis (typically perpendicular to TP axis).

3. **Direct TTNN Calls**: All collective operations directly call ttnn functions without abstraction layers.

4. **Parallel Execution**: Shared expert runs in parallel with MoE experts when `parallel_with_moe: true`.

5. **Unified DistributedExpert**: Single expert implementation with configurable activation modes:
   - **Simple SwiGLU**: DeepSeek-V3 style activation (default)
   - **Clamped SwiGLU**: GPT-OSS style with gate/up clamping

## Usage Example

### Using the Unified MoEBlock

```python
from tt_moe import MoEBlock

# Initialize with DeepSeek-V3 configuration
moe = MoEBlock("configs/deepseek_v3.json", mesh_device, ccl)

# Load weights
moe.load_weights(state_dict)

# Forward pass (handles all TP operations internally)
output = moe.forward(x, mode="prefill")
```

## Testing

Run the tests to verify the implementation:

```bash
# Test DeepSeek-V3 implementation (PCC: 0.989)
pytest tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference -xvs

# Test GPT-OSS configuration loading
pytest tests/test_gpt_oss_moe_block.py::test_gpt_oss_config_loading -xvs

# Test all components
pytest tests/test_moe_components.py -xvs
```

This verifies:
- DeepSeek-V3 PCC accuracy (>0.98 requirement)
- Configuration completeness
- Component functionality
- Parameter mappings

## Current Status

### DeepSeek-V3
- ✅ Fully working with PCC: 0.989 (exceeds 0.98 requirement)
- ✅ All tests passing with simplified implementation
- ✅ SharedExpert numerical explosion fixed

### GPT-OSS
- ✅ Basic infrastructure complete
- ✅ TopKRouter implemented and tested
- ✅ All-to-All working with Linear topology
- ✅ DistributedExpert with clamped SwiGLU activation
- ⚠️ Full model validation pending hardware testing

## Next Steps

To add support for other models:

1. **Mixtral**: Create `configs/mixtral.json` with 8 experts, top-2 routing
2. **Custom Models**: Create new JSON configs with desired parameters

The infrastructure is designed to be extensible - new router and expert types can be added by implementing the base interfaces.
