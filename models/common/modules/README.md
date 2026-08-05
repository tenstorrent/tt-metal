# TTTv2 Modules User's Guide

This document is for **users** of TTTv2 modules (`models/common/modules`). We identify two categories of users:
- **Quick Start Users** (90% of users) who want to get started quickly with the simple positional-weight API.
- **Power Users** (10% of users) who need fine-grained control and customization of the TTNN ops within the modules.

Every module in TTTv2 follows the **same contract** (see [The Universal Module Contract](#the-universal-module-contract)), so once you learn one (e.g. `MLP1D` below) you know them all.

---

## Zen of TTTv2

TTTv2 is built on a few core principles that guide its design and usage.

### 1. Library, not Framework
Users control the execution flow. TTTv2 provides high-performance building blocks, not an opinionated orchestration layer.

### 2. No if-else on static conditions in `forward()`
Avoid runtime branching on static configuration to keep execution paths predictable and fast.
- **Keep hot paths simple**: Implementation should be a straight line of compute.
- **Decouple config and implementation**: Strategy decisions happen during construction, not in the inner loop. (See how `Sampling1D` binds a topology-specific strategy at construction time instead of branching in `forward()`.)

### 3. Lazy and Transparent is better than Proactive and Opaque in weight loading and API design
Efficiency and predictability through lazy initialization and explicit APIs.
- **Lazy weight loading**: Weights (`LazyWeight`) and mutable state buffers (`LazyBuffer`) load/allocate on first use, not at construction, saving memory and time during model setup.
- **Transparent API**: TTNN interfaces are used directly where possible so the underlying operations are clear.
- **Sensible defaults**: Each module ships with a config known to work for a select set of ML models.
- **Full override capability**: Every default can be customized for your specific model architecture.

### 4. More unit tests than end-to-end tests
We prioritize fast, focused, and debuggable unit tests of modules. This ensures faster iteration, easier debugging, and left-shifted CI testing.
- Initially, we parameterize unit tests with real use cases by all the models TTTv1 supports.
- As we add more models to TTTv2, we will continue to add more parameterizations to the unit tests to cover those models.
- We also added code coverage tests to ensure that we are covering most of the code paths in the TTTv2 modules.

---

## The Universal Module Contract

Every TTTv2 module is a `LightweightModule` subclass that exposes the same surface:

- a **`<Name>Config`** dataclass — the single source of truth; every field is optional except the weights, and unset fields are filled with sensible defaults at construction;
- a **simple constructor** (90% path) that takes only weights + essential dimensions and derives everything else;
- a **`from_config(cfg)`** classmethod (10% path) for full customization;
- a **`forward(...)`** that is a straight line of compute (no static if-else — see Zen #2);
- a **`from_model_args(...)`** bridge used by the retiring TTTv1 stack. It exists for backward compatibility and most users can ignore it.

Current module inventory:

| Module | Class (file) | Simple constructor | `forward` signature |
|--------|--------------|--------------------|---------------------|
| MLP (1D) | `MLP1D` (`mlp/mlp_1d.py`) | `MLP1D(w1, w2, w3)` | `forward(x, mode)` |
| MLP (2D) | `MLP2D` (`mlp/mlp_2d.py`) | `MLP2D(w1, w2, w3)` | `forward(x, mode)` |
| Attention | `Attention1D` (`attention/attention_1d.py`) | `Attention1D(wqkv, wo, n_heads, n_kv_heads, head_dim, max_batch_size, max_seq_len)` | `forward(..., mode)` |
| RMSNorm (1D) | `RMSNorm1D` (`rmsnorm/rmsnorm_1d.py`) | `RMSNorm1D(weight)` | `forward(x, mode)` |
| RMSNorm (2D) | `RMSNorm2D` (`rmsnorm/rmsnorm_2d.py`) | `RMSNorm2D(weight)` | `forward(x, mode)` |
| RoPE | `RotarySetup1D` (`rope/rope_1d.py`) | `RotarySetup1D(cos_matrix, sin_matrix, max_batch_size)` | `forward(mode, **kwargs)` |
| Embedding | `Embedding1D` (`embedding/embedding_1d.py`) | `Embedding1D(weights, embed_scale=1.0)` | `forward(x)` |
| LM Head | `LMHead1D` (`lm_head/lm_head_1d.py`) | `LMHead1D(output_weights)` | `forward(x)` |
| Sampling | `Sampling1D` (`sampling/sampling_1d.py`) | `Sampling1D(vocab_size, mesh_device)` | `forward(logits, **kwargs)` |
| Penalties | `Penalties1D` (`sampling/penalties_1d.py`) | `Penalties1D(vocab_size, mesh_device)` | `forward(logits, params, accum)` |

Notes:
- `forward(x, mode)` modules take `mode="prefill"` or `mode="decode"` (a `str` or the `Mode` enum). Modules without a `mode` argument (`Embedding1D`, `LMHead1D`) are called the same way regardless of phase.
- `Sampling1D`/`Penalties1D` are **stateful** ops driven by runtime token data, so they take `vocab_size` + `mesh_device` (not `LazyWeight`s) and manage device state through `LazyBuffer` (see [Supporting Infrastructure](#supporting-infrastructure)).

---

## Quick Start (90% of Users)

For most use cases, the simple positional-weight API is all you need. You wrap your PyTorch tensors in `LazyWeight` and pass them to the constructor:

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

The same three steps apply to every module — only the constructor args change (see the inventory table above).

See the test at `models/common/tests/modules/mlp/test_mlp_1d.py::test_mlp_1d_vs_reference` for a complete working example that shows:
- How to create `LazyWeight` instances with disk caching
- How to run prefill/decode forward passes
- How to validate outputs against a HuggingFace reference model

---

## Power Users (10% of Users)

When you need fine-grained control, use the module's config dataclass (here `MLP1DConfig`):

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

### Why Use the Config Dataclass?

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

## Supporting Infrastructure

These shared building blocks back every module. You rarely construct them directly (modules wire them up for you), but understanding them explains the design.

### Weights vs. Buffers: `LazyWeight` and `LazyBuffer`

Both defer device allocation until first use and accept any tensor `ttnn.from_torch()` can handle (no hard `torch` dependency).

| | `LazyWeight` (`lazy_weight.py`) | `LazyBuffer` (`lazy_buffer.py`) |
|---|---|---|
| For | Immutable model weights | Mutable state tensors (e.g. token counts, penalty masks) |
| Disk cache | Yes — fingerprinted for cache invalidation | No — caching a mutable buffer would corrupt state |
| Materialize | `get_device_weight()` | `get_device_buffer()` |
| Update | n/a (immutable) | `update(new_source)` writes device in-place, same handle |

`Sampling1D` and `Penalties1D` are the only modules that use `LazyBuffer` today, because they mutate device state across decode steps.

### Collectives: `TT_CCL`

Multi-device modules need collective ops (reduce-scatter, all-gather). `TT_CCL` (`tt_ccl.py`) owns the hardware semaphores those ops require. There is **one instance per `mesh_device`**, created and cached for you via `get_tt_ccl(mesh_device)` — so modules sharing a device share semaphores. Pass your own through the config's `tt_ccl` field only if you need to. CCL tuning constants (`CCL_CHUNKS_PER_SYNC`, `CCL_NUM_WORKERS_PER_LINK`, `CCL_NUM_BUFFERS_PER_CHANNEL`) live in the same file and are shared across all modules.

### `Mode`

`mode`-aware modules accept either the string `"prefill"`/`"decode"` or the `Mode` enum from `models.tt_transformers.tt.common`.

---

## 1D vs 2D Modules

`*1D` modules target 1D-topology devices: N150 (1×1), N300 (1×2), and T3K (1×8). `*2D` modules (`MLP2D`, `RMSNorm2D`) target larger 2D mesh shapes (e.g. Galaxy, 8×4). The 2D variants share the same contract as their 1D counterparts.

As TTTv1's 2D mesh support is not tested in CI, the 2D modules do not yet have a comprehensive test suite (`test_mlp_2d.py` / `test_rmsnorm_2d.py` contain basic parameters). We will add more targeted unit tests — like the ones in `test_mlp_1d.py` — as 2D models are implemented.

---

## Running the Tests

Each module has its own test directory under `models/common/tests/modules/`:

```bash
# Run all MLP1D tests (fast subset)
pytest models/common/tests/modules/mlp/test_mlp_1d.py -v

# Include slow tests for full coverage
pytest models/common/tests/modules/mlp/test_mlp_1d.py -v --slow

# Run a specific test
pytest models/common/tests/modules/mlp/test_mlp_1d.py::test_mlp_1d_config_creation -v

# Run the whole module test suite
pytest models/common/tests/modules/ -v
```

Test files by module: `mlp/test_mlp_1d.py`, `mlp/test_mlp_2d.py`, `attention/test_attention_1d.py`, `rmsnorm/test_rmsnorm_1d.py`, `rmsnorm/test_rmsnorm_2d.py`, `rope/test_rope_1d.py`, `embedding/test_embedding_1d.py`, `lm_head/test_lm_head_1d.py`, `sampling/test_sampling_1d.py`, `sampling/test_penalties_1d.py`. Shared infrastructure has its own tests too (`test_lazy_buffer.py`, `test_tensor_utils.py`).

### Device Topologies Tested

| Mesh Shape | Device Type | Module |
|------------|-------------|--------|
| `(1, 1)` | N150 (single device) | `*1D` |
| `(1, 2)` | N300 | `*1D` |
| `(1, 8)` | T3K | `*1D` |
| `(8, 4)` | Galaxy | `*2D` |
