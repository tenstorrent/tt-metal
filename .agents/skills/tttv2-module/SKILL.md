---
name: tttv2-module
description: The contract a TTTv2 module must satisfy, including structural requirements, clean forward path strategy, and the test suite that proves it. Use when designing, implementing, or reviewing a reusable module under models/common/modules (or a model-local candidate), or when checking an existing module against the Universal Module Contract before shipping it.
---

# TTTv2 Module Contract

A TTTv2 module is a reusable, topology-aware, ttnn-only building block for composing models, for example: MLP, attention, RMSNorm, RoPE, embedding, LM head, sampling. This skill is a exhaustive checklist for what a module must be: its structure, its strategy discipline, and its tests.

The user-facing version of this contract is [models/common/modules/README.md](../../../models/common/modules/README.md), it reflects the philosophy of TTTv2.

Canonical example to copy:
- [models/common/modules/mlp/mlp_1d.py](../../../models/common/modules/mlp/mlp_1d.py) + [models/common/tests/modules/mlp/test_mlp_1d.py](../../../models/common/tests/modules/mlp/test_mlp_1d.py) — full-featured: `<Name>Config` dataclass, simple + `from_config` constructors, `prefill_forward` / `decode_forward`, multi-device CCL, parametrized vs-reference test.

Where a new module lives: a generally reusable module goes under `models/common/modules/<area>/<name>.py`. A model-local candidate that isn't yet general enough to promote can be placed near the model code.

## When to use this skill

- You are extracting reusable logic from a model into a TTTv2-style module and need the contract it must hit.
- A module is functionally working but you need to confirm it satisfies the contract before shipping (run the grep checks in §2).
- You are reviewing a module PR, or deciding whether a piece of variation belongs in the forward signature or in construction-time config (§2).

## 1. Structural requirements (the Universal Module Contract)

Every module must:

- **Subclass [`LightweightModule`](../../../models/common/lightweightmodule.py).** Never `nn.Module` — the module is ttnn-only and must not drag in torch's module machinery.
- **Have exactly one `<Name>Config` dataclass** as the single source of truth. *Every field is optional except the weights.* A `_resolve_<name>_config()` helper fills defaults in phases: foundational (device, topology) → derived defaults (dims inferred from weights) → resolve `LazyWeight`s. See `MLP1DConfig` + its resolve function in [mlp_1d.py](../../../models/common/modules/mlp/mlp_1d.py).
- **Guarantee the `is_resolved()` invariant** holds before any device work — config resolution complete, no `None` left in a required field.
- **Offer two constructors:**
  - simple positional `__init__(weights, …essential dims)` — the 90% path; derives everything else from sensible defaults.
  - `from_config(cfg)` classmethod — the 10% power-user path for full customization.
- **Load device weights lazily on first forward** via an idempotent `load_device_weights()`. Weights are immutable `LazyWeight` (disk-cached, fingerprinted for invalidation); mutable device state is `LazyBuffer` (never disk-cached — caching mutable state corrupts it). Only stateful ops (`Sampling1D`, `Penalties1D`) need `LazyBuffer`.
- **Route every multi-device collective through [`tt_ccl.py`](../../../models/common/modules/tt_ccl.py)** — there is one `TT_CCL` per mesh device, shared via `get_tt_ccl(mesh_device)`, so modules sharing a device share semaphores. Don't allocate semaphores yourself.
- **Expose every internal `program_config` / `memory_config` / `dtype` / `compute_kernel_config` as an optional `<Name>Config` field.** A power user must be able to override any of them without forking the module. Each non-trivial default (why bf8 vs bf4, why L1 vs DRAM, why this core grid) is justified in the module docstring.

### Naming and scope

- **`<Name>1D`** when the module is topology-aware (sharding / multi-device collective code paths): targets N150 (1×1), N300 (1×2), T3K (1×8). **`<Name>2D`** for larger 2D meshes (e.g. Galaxy 8×4). Bare **`<Name>`** when topology-agnostic (identical code everywhere, no CCL).
- **Cover only what the references cover.** If every reference is decode-only on N300, ship `decode_forward` on N300 and defer prefill / other SKUs to follow-up work — premature coverage is untested code. But **shape the extension points** (strategy enums, optional config fields) so a future variant slots in as one new enum value + one branch, not a redesign.
- **Single-device first.** When a module shards or uses collectives, nail accuracy on a single device (no fabric, no CCL) before layering in topology — it isolates math errors from sharding/reduce errors and iterates faster.

## 2. Strategy discipline (the hot path)

The forward is where the contract is most often violated. Rules:

- **`forward(...)` is a straight line of compute.** Mode-aware modules dispatch *once* at the top: `forward(x, mode)` → `prefill_forward(x)` / `decode_forward(x)`, with `mode` a `str` (`"prefill"`/`"decode"`) or the `Mode` enum. Inside the per-mode methods, **no static `if mode ==` and no `if device_kind ==` branching.** Only seq-len / runtime-data conditionals are allowed in the hot path.
- **Strategy decisions happen at construction, not in `forward()`** (Zen rule #2). When behavior depends on topology or static config, bind the strategy when the config is resolved (the way `Sampling1D` binds a topology-specific strategy at construction time instead of branching per call), then run it unconditionally. This keeps execution paths predictable and fast.
- **Lazy and transparent** (Zen rule #3): `LazyWeight`/`LazyBuffer` defer device allocation to first use, and ttnn ops are called directly with no wrapper layer hiding which op runs.
- **Where does variation live?** Prefer a single forward signature shared across different models, absorbing per-reference peculiarities into *optional* construction-time config (default `None` → step skipped), **until** that absorption starts multiplying distinct compute paths, then a slightly non-standard forward argument is the simpler choice. Optimize for total simplicity, not interface purity. State explicitly what the **caller still owns** (residuals, heads, transforms outside the unit) so the boundary is unambiguous.

### Contract grep checks (run before shipping)

```bash
AREA=<area>; NAME=<name>
F=models/common/modules/$AREA/$NAME.py   # or new_modules/modules/...

# Hot forward must not touch torch or HF
! rg -n "import torch|from torch|transformers" $F

# No nn.Module — must subclass LightweightModule
! rg -n "nn\.Module" $F

# Static-mode branching only inside the top-level dispatcher, never in prefill_/decode_forward
rg -n "if mode" $F          # expected: matches only inside `def forward(`

# Every program_config / memory_config / dtype is a Config field or a docstring-justified constant
rg -n "program_config|memory_config|dtype" $F
```

## 3. Testing requirements

Zen rule #4: **more unit tests than end-to-end** — broad parametrization over shapes / dtypes / mesh shapes is the point, not an e2e model run. Tests are co-located under `models/common/tests/modules/<area>/test_<name>.py`, styled after [test_mlp_1d.py](../../../models/common/tests/modules/mlp/test_mlp_1d.py) / [test_rmsnorm_1d.py](../../../models/common/tests/modules/rmsnorm/test_rmsnorm_1d.py). The file must contain:

- **Config-construction unit tests (no device):** the simple constructor builds, defaults resolve, `from_config` round-trips, and power-user overrides take effect. `is_resolved()` holds.
- **vs-reference accuracy test:** **PCC ≥ 0.99** per (reference × mesh shape × mode) the bringup covers, parametrized over the mesh shapes the references actually exercise — `(1,1)` N150, `(1,2)` N300, `(1,8)` T3K for `*1D`; `(8,4)` Galaxy for `*2D`. Drive the module and the golden/reference from **identical inputs and weights**, cast to the same dtype the device path sees — a higher-precision golden makes quantization noise look like module error.
- **Parametrize over every model the module supports.** A mature module's test sweeps a real model roster — see the `hf_model_name` list crossed with mesh/mode/dtype in [test_attention_1d.py](../../../models/common/tests/modules/attention/test_attention_1d.py) (Llama / Mistral / Qwen across sizes). A *new* module is only used by the one or two models it was brought up from, so its sweep is naturally smaller — **don't copy a sibling's full roster**, those are models this module doesn't yet serve. The compromise: cover **at least the config (heads / dims / dtypes) of every model the bringup targets**, plus a few boundary cases (e.g. GQA vs MQA, smallest/largest head_dim, seq-len edges) where they widen coverage cheaply. Grow the roster as new models adopt the module.
- **At least one power-user override test:** override a `<Name>Config` field (e.g. a program config) on a constructed instance and validate it still produces correct output (see `test_mlp_1d_config_prefill_override`).
- **Optional `--slow` tests** for larger-seq-len / model-size sweeps.

Coverage target: at parity with sibling modules — compare against [test_mlp_1d.py](../../../models/common/tests/modules/mlp/test_mlp_1d.py). Run module tests one device-touching command at a time; importing `ttnn` opens the cluster, so don't run them concurrently with other device work.

### Debugging low PCC

Bisect at the **module boundary**: feed reference-correct inputs into intermediates and check the ttnn output at each op call.
- Broad per-op drift (every op at PCC 0.88–0.97 but layer output still ≥ 0.999) almost always means a math-fidelity / dtype default is too aggressive — promote the relevant op to higher fidelity (`fp32_dest_acc_en`, higher `MathFidelity`) before chasing individual ops.
- Discrete-selection ops (top-k, argmax, sort) fail differently — a tiny rounding difference flips *which* element is chosen, unmoved by fidelity bumps. Symptom: PCC stuck ~0.97–0.99, spiky per sample. Fix: compute pre-selection math in high precision, cast only for the selection, and match that same rounding in the golden so both make the same choice.
