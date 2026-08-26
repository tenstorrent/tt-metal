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
- selected **1D modules** retain a `from_model_args(...)` bridge for the retiring TTTv1 stack. New 2D modules intentionally do not expose this bridge; product models construct their configs explicitly.

Current module inventory:

| Module | Class (file) | Simple constructor | `forward` signature |
|--------|--------------|--------------------|---------------------|
| MLP (1D) | `MLP1D` (`mlp/mlp_1d.py`) | `MLP1D(w1, w2, w3)` | `forward(x, mode)` |
| MLP (2D) | `MLP2D` (`mlp/mlp_2d.py`) | `MLP2D(w1, w2, w3)` | `forward(x, mode)` |
| Attention | `Attention1D` (`attention/attention_1d.py`) | `Attention1D(wqkv, wo, n_heads, n_kv_heads, head_dim, max_batch_size, max_seq_len)` | `forward(..., mode)` |
| Attention (2D) | `Attention2D` (`attention/attention_2d.py`) | `Attention2D(wqkv, wo, ...)` | `forward(..., mode)` |
| RMSNorm (1D) | `RMSNorm1D` (`rmsnorm/rmsnorm_1d.py`) | `RMSNorm1D(weight)` | `forward(x, mode)` |
| RMSNorm (2D) | `RMSNorm2D` (`rmsnorm/rmsnorm_2d.py`) | `RMSNorm2D(weight)` | `forward(x, mode)` |
| RoPE | `RotarySetup1D` (`rope/rope_1d.py`) | `RotarySetup1D(cos_matrix, sin_matrix, max_batch_size)` | `forward(mode, **kwargs)` |
| RoPE (2D) | `RotarySetup2D` (`rope/rope_2d.py`) | `RotarySetup2D(cos_matrix, sin_matrix, max_batch_size)` | `forward(mode, **kwargs)` |
| Embedding | `Embedding1D` (`embedding/embedding_1d.py`) | `Embedding1D(weights, embed_scale=1.0)` | `forward(x)` |
| Embedding (2D) | `Embedding2D` (`embedding/embedding_2d.py`) | `Embedding2D(weights, embed_scale=1.0)` | `forward(x, mode)` |
| LM Head | `LMHead1D` (`lm_head/lm_head_1d.py`) | `LMHead1D(output_weights)` | `forward(x)` |
| LM Head (2D) | `LMHead2D` (`lm_head/lm_head_2d.py`) | `LMHead2D(output_weights, vocab_size)` | `forward(x, mode)` |
| Sampling | `Sampling1D` (`sampling/sampling_1d.py`) | `Sampling1D(vocab_size, mesh_device)` | `forward(logits, **kwargs)` |
| Sampling (2D) | `Sampling2D` (`sampling/sampling_2d.py`) | `Sampling2D(vocab_size, mesh_device)` | `forward(logits, **kwargs)` |
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

`*1D` modules target 1D-topology devices: N150 (1×1), N300 (1×2), and T3K (1×8). Production `*2D` configs target Wormhole Galaxy with the canonical logical mesh shape `(8, 4)` and exactly 32 devices. Resolution fails closed for another architecture, mesh orientation, device count, incompatible collaborator, or indivisible partition.

The 2D tensor-placement contract is explicit in each config: source and padded weight shapes, row/column shard dimensions, decode and prefill input/output placement, and transient ownership are resolved before execution. Static strategy decisions do not branch in a module hot path.

Galaxy collectives are injected from `models/common/models/galaxy`; 2D modules do not extend or specialize the 1D `TT_CCL` owner. The target ownership contract makes `Prefetcher2D` the model-owned resource root for subdevice managers, the global circular buffer, and sealed weight-address registration. Modules borrow immutable prefill/decode contexts, and the executor activates a context at operation boundaries and owns cleanup. Integrated Prefetcher2D/Galaxy-resource ownership is qualified on hardware for the MLP2D consumer shape: the prefill↔decode transition matrix, the failed-transition rollback, and cleanup from either active mode all run on a `(8, 4)` 6U Galaxy with PCC asserted at every step.

Two ownership limits are known and documented. **Cleanup of the global circular buffer is not deterministic from the owner's side:** ttnn exposes no free for one, so its L1 is reclaimed when the last handle dies, and every module holding a `Prefetcher2DContext` holds one — tear consumers down before, or together with, the owner. And **`Attention2D`'s current decode projection grid is incompatible with the prefetch subdevice partition**; its `(7,1)` QKV grid straddles the sender and worker subdevices, so choosing a compatible grid is Milestone B work.

The reusable 2D set consists of `Embedding2D`, `RotarySetup2D`, `RMSNorm2D`, `Attention2D`, `MLP2D`, `LMHead2D`, and `Sampling2D`. There is no `Penalties2D`.

Everything Galaxy-specific but model-neutral lives in `models/common/models/galaxy`: the `(8, 4)` geometry and placement recipes, the collective-resource plans, the `Attention2D`/`LMHead2D` collective adapters, the prefetch construction policy, the paged-KV metadata view, and a direct prefill/decode runner used by Milestone B tests and demos. No transformer graph lives there. Each product package (`models/common/models/llama33_70b_galaxy`, `models/common/models/qwen3_32b_galaxy`) owns its own graph — checkpoint contract, precision recipe, provider weight conversion, every 2D module config, its decoder layer, its tensor model, and its construction order — and borrows only that topology-neutral machinery. Neither package imports another model-named package.

`Attention2D` takes two different paged page-table layouts, because the two modes address the cache differently. Prefill fills a named user's blocks with `paged_fill_cache(..., batch_idx=u)`, so its device-local table needs one row for every user the request fills; concatenated physical-batch-32 prefill therefore carries all 32 rows. Decode attends to one mesh column's users, so `paged_update_cache` and the paged decode SDPA both require the device-local table to carry exactly `users_per_column` rows — or that batch repeated once per core when the table is L1-sharded. A table sized to the full physical batch is the prefill layout and decode rejects it.

All seven modules plus Galaxy CCL/resources and `Prefetcher2D` are qualified on real WH `(8, 4)` hardware: 37 device cases pass in one sweep with clean teardown and no reset. See [Milestone A 2D Module Status](MILESTONE_A_STATUS.md) for the evidence matrix, the four defects that qualification uncovered, the modularity scorecard, and the items deferred to later milestones.

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

Test files follow the module layout and use `test_<name>_1d.py` / `test_<name>_2d.py` where both variants exist. `prefetcher/test_prefetcher_2d.py` covers explicit registration, sealing, activation, and cleanup. Shared infrastructure has its own tests too (`test_lazy_buffer.py`, `test_tensor_utils.py`).

`test_<name>_2d.py` files are **host-only** and run against a mock mesh; the real `(8, 4)` Galaxy suites are `test_<name>_2d_wh_galaxy.py`, plus `sampling/test_sampling_2d_wh_galaxy_stochastic.py` and `prefetcher/test_prefetcher_2d_wh_galaxy.py`. Select against them deliberately — passing a module directory to pytest picks up its device suite as well, which needs 32 devices. Filter with `--ignore-glob="*_wh_galaxy*.py"` for a host-only run.

Both 1D and 2D numerical suites qualify against the same HuggingFace references, shared through `tests/modules/_hf_reference.py`; the Galaxy MLP geometry that several 2D device suites reuse lives in `tests/modules/_mlp_2d_galaxy.py`, and the hardware plumbing in `tests/modules/_wh_galaxy_hardware.py`.

### Device Topologies Tested

| Mesh Shape | Device Type | Module |
|------------|-------------|--------|
| `(1, 1)` | N150 (single device) | `*1D` |
| `(1, 2)` | N300 | `*1D` |
| `(1, 8)` | T3K | `*1D` |
| `(8, 4)` | Galaxy | `*2D` |
