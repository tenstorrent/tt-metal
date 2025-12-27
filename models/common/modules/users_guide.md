# TTTv2 Modules User's Guide

This document is for **users** of TTTv2 modules. We identify two categories of users:
- **Quick Start Users** (90% of users) who want to get started quickly with the simple 3-weight API.
- **Power Users** (10% of users) who need fine-grained control and customization of the TTNN ops within the modules.

---

## Zen of TTTv2

TTTv2 is built on a few core principles that guide its design and usage.

### 1. Library, not Framework
Users control the execution flow. TTTv2 provides high-performance building blocks, not an opinionated orchestration layer.

### 2. No if-else on static conditions in `forward()`
Avoid runtime branching on static configuration to keep execution paths predictable and fast.
- **Keep hot paths simple**: Implementation should be a straight line of compute.
- **Decouple config and implementation**: Strategy decisions happen during construction, not in the inner loop.

### 3. Lazy and Transparent is better than Proactive and Opaque in weight loading and API design
Efficiency and predictability through lazy initialization and explicit APIs.
- **Lazy weight loading**: Weights load on first use, not at construction, saving memory and time during model setup.
- **Transparent API**: TTNN interfaces are used directly where possible so the underlying operations are clear.
- **Sensible defaults**: Each module ships with a config known to work for a select set of ML models.
- **Full override capability**: Every default can be customized for your specific model architecture.

### 4. More unit tests than end-to-end tests
We prioritize fast, focused, and debuggable unit tests of modules. This ensures faster iteration, easier debugging, and left-shifted CI testing.
- Initially, we parameterize unit tests with real use cases by all the models TTTv1 supports.
- As we add more models to TTTv2, we will continue to add more parameterizations to the unit tests to cover those models.
- We also added code coverage tests to ensure that we are covering most of the code paths in the TTTv2 modules.

---

## Quick Start (90% of Users)

For most use cases, the simple 3-weight API is all you need. You just wrap your PyTorch tensors in `LazyWeight` and pass them to the constructor:

```python
import ttnn
from models.common.modules.mlp.mlp_1d import MLP1D
from models.common.modules.lazy_weight import LazyWeight

# 1. Prepare weights (from PyTorch)
# LazyWeight doesn't load to device until needed
w1 = LazyWeight(source=torch_w1, dtype=ttnn.bfloat4_b)
w2 = LazyWeight(source=torch_w2, dtype=ttnn.bfloat8_b)
w3 = LazyWeight(source=torch_w3, dtype=ttnn.bfloat4_b)

# 2. Construct the module
# Sensible defaults are automatically resolved (device, topology, etc.)
mlp = MLP1D(w1, w2, w3)

# 3. Forward pass
# First forward pass will trigger weight loading to device
x = ttnn.from_torch(torch_x, device=mesh_device, dtype=ttnn.bfloat16)
y = mlp.forward(x, mode="prefill") # or mode="decode"
```

See the test at `models/common/tests/modules/mlp/test_mlp_1d.py::test_mlp_1d_vs_reference` for a complete working example that shows:
- How to create `LazyWeight` instances with disk caching
- How to run prefill/decode forward passes
- How to validate outputs against a HuggingFace reference model

---

## Power Users (10% of Users)

When you need fine-grained control, use `MLP1DConfig`:

```python
from models.common.modules.mlp.mlp_1d import MLP1D, MLP1DConfig

# Create config with any overrides you need
config = MLP1DConfig(
    w1=w1, w2=w2, w3=w3,
    mesh_device=mesh_device,
    topology=ttnn.Topology.Ring,
    max_batch_size=64,
    # ... any other overrides
)

mlp = MLP1D.from_config(config)
y = mlp.forward(x, mode="prefill")
```

### Why Use MLP1DConfig?

The config dataclass is the single source of truth for the module.

| Benefit | Description |
|---------|-------------|
| **Composable** | Pass around, modify, inherit |
| **Self-documenting** | All options in one dataclass |
| **Gradual customization** | Override just what you need |


### Example: Overriding Program Configs

See `models/common/tests/modules/mlp/test_mlp_1d.py::test_mlp_1d_config_prefill_override` for a complete example that demonstrates:
- Creating a custom `prefill_w2_prg_config` function
- Overriding it on an existing `MLP1D` instance
- Validating the custom config works correctly

```python
# After constructing the model, override specific configs
tt_model = MLP1D.from_config(MLP1DConfig(w1=lazy_w1, w2=lazy_w2, w3=lazy_w3))

@lru_cache
def custom_prefill_w2_prg_config(seq_len: int):
    # Your custom config logic here
    return _matmul_config(...)

tt_model.config.prefill_w2_prg_config = custom_prefill_w2_prg_config
```

### The Continuous Customization Path

TTTv2 provides a **continuous experience** from simple to advanced usage:

```
Simple Usage                       Advanced Customization
      │                                     │
      │ (pass weights)                      │ (pass config)
      ▼                                     ▼
MLP1D(w1, w2, w3)                   MLP1D.from_config(cfg)
      │                                     │
      │  + sensible defaults                │  + your overrides
      └──────────────────┬──────────────────┘
                         │
                         ▼
                  Resolved Config
             (Fully defined at runtime)
```

---

## MLP2D Support
MLP2D is used for larger mesh shapes (e.g., Galaxy). Its tests are in `models/common/tests/modules/mlp/test_mlp_2d.py`, which contains basic test parameters. As TTTv1's 2D mesh support is not tested in CI, we do not have a comprehensive test suite for MLP2D yet. We will add more targeted unit tests like the ones in test_mlp_1d.py when relevant model are implemented.

---

## Running the Tests

```bash
# Run all MLP1D tests (fast subset)
pytest models/common/tests/modules/mlp/test_mlp_1d.py -v

# Include slow tests for full coverage
pytest models/common/tests/modules/mlp/test_mlp_1d.py -v --slow

# Run a specific test
pytest models/common/tests/modules/mlp/test_mlp_1d.py::test_mlp_1d_config_creation -v
```

### Device Topologies Tested

| Mesh Shape | Device Type | Module |
|------------|-------------|--------|
| `(1, 1)` | N150 (single device) | MLP1D |
| `(1, 2)` | N300 | MLP1D |
| `(1, 8)` | T3K | MLP1D |
| `(8, 4)` | Galaxy | MLP2D |
