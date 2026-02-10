# Unified MoE Block Implementation

This directory contains a unified, configurable Mixture of Experts (MoE) implementation that can support multiple architectures through JSON configuration files.

## Directory Structure

```
tt-moe/
├── moe_block.py              # Main unified MoE block
├── deepseek_adapter.py       # Adapter showing DeepSeek equivalence
├── configs/
│   └── deepseek_v3.json     # DeepSeek-V3 configuration
├── components/
│   ├── routers/
│   │   ├── base_router.py   # Abstract router interface
│   │   └── moe_gate.py      # DeepSeek MoEGate implementation
│   └── experts/
│       ├── base_expert.py   # Abstract expert interface
│       ├── distributed_expert.py  # All-to-all expert
│       └── shared_expert.py      # DeepSeek shared expert
└── tests/
    └── test_deepseek_equivalence.py  # Equivalence tests
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

### Using the DeepSeek Adapter

The adapter shows how to use the unified block as a drop-in replacement:

```python
from tt_moe import DeepSeekMoEAdapter

# Create adapter
adapter = DeepSeekMoEAdapter(hf_config, mesh_device, ccl)

# Use exactly like original implementation
output = adapter.forward_mlp_prefill(x, cfg)
```

## Testing

Run the equivalence tests to verify the implementation:

```bash
python tests/test_deepseek_equivalence.py
```

This verifies:
- Configuration completeness
- TP sharding detection
- Logic flow equivalence
- Parameter mappings

## Next Steps

To add support for other models:

1. **GPT-OSS**: Create `configs/gpt_oss.json` and implement `TopKRouter`
2. **Mixtral**: Create `configs/mixtral.json` with 8 experts, top-2 routing
3. **Custom Models**: Create new JSON configs with desired parameters

The infrastructure is designed to be extensible - new router and expert types can be added by implementing the base interfaces.
