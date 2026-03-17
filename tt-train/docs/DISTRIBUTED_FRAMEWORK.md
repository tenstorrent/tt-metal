# TTML Distributed Framework

A Python-first, layout-aware dispatch layer for tensor-parallel and data-parallel training on Tenstorrent hardware.

## Overview

The `ttml.distributed` package provides automatic tensor redistribution and collective communication for distributed training. It replaces manual parallelism management with a declarative, rule-based system where you specify *what* layout you want, and the framework handles *how* to achieve it.

### Key Features

- **Layout-driven dispatch**: Tensors carry sharding metadata that flows through computation
- **Automatic redistribution**: Operations automatically insert collectives (all_gather, all_reduce, scatter) as needed
- **Rule-based sharding**: Extensible system for defining how each operation handles distributed inputs
- **Module-level distribution**: One-line API to distribute entire models with a policy dict
- **Backward-compatible**: Works alongside existing TTML autograd system

## Quick Start

```python
import ttml
from ttml.distributed import (
    Layout, Shard, Replicate,
    distribute_module, distribute_tensor, sync_gradients,
    init_ops, MeshRuntime, set_runtime
)

# 1. Enable distributed dispatch
init_ops()

# 2. Open mesh device
ttml.core.distributed.enable_fabric(32)
auto_ctx = ttml.autograd.AutoContext.get_instance()
auto_ctx.open_device([8, 4])  # 8 DP x 4 TP
mesh_device = auto_ctx.get_device()

# 3. Set up runtime
rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
set_runtime(rt)

# 4. Define sharding policy
policy = {
    # Column-parallel: shard output features
    r".*\.w1\.weight": Layout(placements=(Replicate(), Shard(-2))),
    r".*\.w3\.weight": Layout(placements=(Replicate(), Shard(-2))),
    # Row-parallel: shard input features
    r".*\.w2\.weight": Layout(placements=(Replicate(), Shard(-1))),
}

# 5. Distribute model
model = MyLlamaModel(config)
distribute_module(model, mesh_device, policy)

# 6. Training loop
for batch in dataloader:
    x = distribute_tensor(batch, mesh_device, Layout((Replicate(), Replicate())))
    loss = model(x)
    loss.backward(False)
    sync_gradients(model.parameters(), cluster_axes=[0])  # DP gradient sync
    optimizer.step()
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER CODE                                        │
│  model = MyModel()                                                       │
│  policy = {"layer.weight": Layout(placements=(Replicate(), Shard(-2)))} │
│  distribute_module(model, mesh_device, policy)                          │
│  y = model(x)  # ops go through dispatch                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      DISPATCH LAYER                                      │
│  1. Check if any input has distributed layout                           │
│  2. Look up sharding rule for op                                        │
│  3. Redistribute inputs if needed (all_gather/scatter)                  │
│  4. Call raw C++ op                                                     │
│  5. Apply post-collectives (all_reduce for row-parallel)                │
│  6. Stamp output with computed layout                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         C++ OPS (ttml.ops.*)                            │
│  Raw operations that are autograd-aware                                 │
│  Collectives (all_gather, all_reduce, scatter) build backward graph     │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Concepts

### Layout Primitives

```python
from ttml.distributed import Layout, Shard, Replicate

# Shard(dim) - tensor is split along dimension `dim`
# Replicate() - tensor is copied identically on all devices

# 2D mesh layout: (DP axis placement, TP axis placement)
col_parallel = Layout(placements=(Replicate(), Shard(-2)))  # Shard on out_features
row_parallel = Layout(placements=(Replicate(), Shard(-1)))  # Shard on in_features
replicated = Layout(placements=(Replicate(), Replicate()))  # Full copy everywhere
```

### Tensor Parallelism Pattern

The framework implements the standard Megatron-LM tensor parallelism pattern:

**Column-Parallel Linear** (weight sharded on output features):
```
Input:  [B, S, T, in_features]  - REPLICATED
Weight: [1, 1, out_features, in_features] - SHARDED on dim -2
Output: [B, S, T, out_features] - SHARDED on last dim

Forward: Each device computes partial output features
Backward: broadcast collective does all_reduce on input gradient
```

**Row-Parallel Linear** (weight sharded on input features):
```
Input:  [B, S, T, in_features]  - SHARDED (from column-parallel)
Weight: [1, 1, out_features, in_features] - SHARDED on dim -1
Output: [B, S, T, out_features] - REPLICATED (after all_reduce)

Forward: all_reduce sums partial results across TP devices
Backward: noop (column-parallel handles gradient sync)
```

### MLP Example

```python
# Typical MLP with TP=4:
#   w1: [hidden, input] column-parallel → output sharded
#   activation: preserves sharding
#   w2: [output, hidden] row-parallel → output replicated

policy = {
    "mlp.w1.weight": Layout((Replicate(), Shard(-2))),  # Column
    "mlp.w2.weight": Layout((Replicate(), Shard(-1))),  # Row
}
```

## API Reference

### distribute_module

```python
def distribute_module(
    module: Module,
    mesh_device: MeshDevice,
    policy: Dict[str, Layout]
) -> Module:
    """
    Distribute a module's parameters according to a sharding policy.

    Args:
        module: The model to distribute
        mesh_device: Target mesh device
        policy: Dict mapping parameter names (or regex patterns) to Layouts

    Returns:
        The same module with distributed parameters
    """
```

### distribute_tensor

```python
def distribute_tensor(
    tensor: Tensor,
    mesh_device: MeshDevice,
    layout: Layout,
    requires_grad: Optional[bool] = None
) -> Tensor:
    """
    Distribute a tensor to the mesh with specified layout.

    Args:
        tensor: Input tensor
        mesh_device: Target mesh device
        layout: Desired sharding layout
        requires_grad: Override gradient tracking (None preserves original)

    Returns:
        Distributed tensor with layout metadata
    """
```

### sync_gradients

```python
def sync_gradients(
    parameters: List[Parameter],
    cluster_axes: Optional[List[int]] = None
) -> None:
    """
    Synchronize gradients across specified mesh axes (typically DP axis).

    Args:
        parameters: List of parameters to sync
        cluster_axes: Mesh axes to sync across (default: infer from runtime)
    """
```

### redistribute

```python
def redistribute(
    tensor: Tensor,
    target_layout: Layout,
    grad_replicated: bool = False
) -> Tensor:
    """
    Convert tensor from current layout to target layout using collectives.

    Args:
        tensor: Input tensor with layout metadata
        target_layout: Desired output layout
        grad_replicated: If True, backward divides by device count

    Returns:
        Tensor with new layout
    """
```

## Sharding Rules

Rules define how each operation handles distributed inputs. Register custom rules with:

```python
from ttml.distributed import register_rule, ShardingPlan

@register_rule("my_custom_op")
def my_op_rule(*layouts, **kwargs):
    """
    Args:
        *layouts: Layout objects for each tensor input
        **kwargs: Non-tensor arguments passed to the op

    Returns:
        ShardingPlan with input_layouts, output_layout, and optional collectives
    """
    return ShardingPlan(
        input_layouts=[...],
        output_layout=Layout(...),
        post_collective="all_reduce",  # Optional
        reduce_mesh_axis=1,            # Which axis for collective
    )
```

### Built-in Rules

| Operation | Rule Behavior |
|-----------|---------------|
| `linear` | Column/row parallel based on weight sharding |
| `matmul` | Propagates sharding through matrix multiply |
| `add`, `mul` | Elementwise preserves layout |
| `silu`, `relu`, `gelu` | Unary preserves layout |
| `rmsnorm`, `layernorm` | Requires replicated on norm dim |
| `cross_entropy_loss` | Gathers sharded logits to replicated |

## Module Rules

For complex modules that need special handling:

```python
from ttml.distributed import register_module_rule
from ttml.modules import LinearLayer

@register_module_rule(LinearLayer)
def distribute_linear(module, mesh_device, weight_layout, bias_layout=None):
    """Transform a LinearLayer for distributed execution."""
    module.weight.tensor = distribute_tensor(
        module.weight.tensor, mesh_device, weight_layout
    )
    if module.has_bias and bias_layout:
        module.bias.tensor = distribute_tensor(
            module.bias.tensor, mesh_device, bias_layout
        )
```

## Debugging

Use `DispatchTracer` to inspect what the dispatch layer is doing:

```python
from ttml.distributed import DispatchTracer, dispatch_trace

with DispatchTracer():
    y = model(x)

for entry in dispatch_trace.entries:
    print(f"Op: {entry.op_name}")
    print(f"  Input layouts: {entry.input_layouts}")
    print(f"  Output layout: {entry.output_layout}")
    print(f"  Redistributions: {entry.redistributions}")
    print(f"  Post-collectives: {entry.post_collectives}")
```

### Trace entries

Each `TraceEntry` contains:
- `op_name`: The dispatched operation name
- `input_layouts`: Layouts of input tensors before dispatch
- `rule_name`: Name of the sharding rule applied (or `"fast_path"` if no distributed tensors)
- `redistributions`: Layout conversions applied to inputs
- `post_collectives`: Collectives applied after the op (e.g., all_reduce for row-parallel)
- `output_layout`: Layout of the output tensor

Use `entry.to_dict()` to get a JSON-serializable representation.

### Trainer integration

The `train_llama_tp.py` script supports dispatch tracing via CLI flags:

```bash
# Enable tracing for all steps (prints first 20 entries at end)
python train_llama_tp.py -c config.yaml --debug_dispatch

# Trace only the first step (reduces memory for long runs)
python train_llama_tp.py -c config.yaml --debug_dispatch --debug_dispatch_first_step_only

# Dump all entries to a JSON lines file
python train_llama_tp.py -c config.yaml --debug_dispatch --debug_dispatch_dump trace.jsonl
```

## File Structure

```
ttml/distributed/
├── __init__.py          # Public API exports
├── layout.py            # Layout, Shard, Replicate primitives
├── mesh_runtime.py      # MeshRuntime configuration
├── dispatch.py          # Central dispatch logic
├── redistribute.py      # Layout conversion via collectives
├── cache.py             # PlanCache for optimization
├── debug.py             # DispatchTracer for debugging
├── training.py          # High-level training utilities
├── module_rules.py      # Module-level distribution rules
├── _register_ops.py     # Op registration and monkey-patching
├── utils.py             # Helper functions
└── rules/
    ├── registry.py      # ShardingPlan, @register_rule
    ├── matmul.py        # linear, matmul rules
    ├── elementwise.py   # Binary and unary op rules
    ├── norm.py          # Normalization rules
    ├── attention.py     # Attention-related rules
    └── loss.py          # Loss function rules
```

## Performance Considerations

1. **Plan Caching**: Dispatch caches sharding plans by (op_name, input_layouts, kwargs) to avoid repeated rule lookups

2. **Minimal Redistribution**: Rules are designed to minimize collective communication by propagating sharding through compatible operations

3. **Fused Collectives**: Row-parallel uses `all_reduce` with `noop_backward=True` to avoid redundant backward collectives

4. **Fast Path**: Operations with no distributed inputs bypass the dispatch layer entirely

## Testing

```bash
# Unit tests (no device required)
pytest tests/python/test_distributed_dispatch.py -v -m "not requires_device"

# Full tests (requires 32-device mesh)
pytest tests/python/test_distributed_dispatch.py -v
```

## Requirements

- TTML with multi-device support
- 32-device mesh for full TP+DP (8x4 configuration)
- Fabric enabled: `ttml.core.distributed.enable_fabric(32)`
