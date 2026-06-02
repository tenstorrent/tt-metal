# Qwen3.5‑9B Building‑Block Mesh‑Awareness — Implementation Plan (Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the Qwen3.5‑9B building blocks (RoPE, RMSNorm, Embedding, MLP, LM‑head) to be **mesh‑aware / `num_devices`‑parameterized**, so the *same* code runs 9B on 1 device and (eventually) Qwen3.5‑27B on 4 devices (TP=4) — validated on the 9B's single device, where the multi‑device ops degrade to no‑ops.

**Architecture:** Both 9B and 27B `Qwen35ModelArgs` subclass the framework `tt_transformers.tt.model_config.ModelArgs`, which already provides `num_devices`, `cluster_shape`, `create_dram_sharded_mem_config`, `ccl_topology`, `ccl_dtype`. The 9B currently ignores them (every weight is a bare `from_torch(..., device=...)`, no mesh_mapper, no tt_ccl, no all‑reduce). This plan makes each block use the framework's mesh‑aware path, mirroring the **27B reference** (`qwen27b-metal/.../qwen35_27b/`, read‑only — never imported). At `num_devices=1`: `tt_all_reduce` returns its input, a `ShardTensor*Mesh` over a 1×1 mesh replicates, and `// num_devices` leaves dims full — so single‑device behavior is preserved.

**Tech Stack:** Python, ttnn, tt‑metal, `models/tt_transformers`, `models/common/rmsnorm.py`, pytest. Target HW: Blackhole P150 (this plan validates on 1 device; the same code targets 4×P150 for 27B).

**Spec:** `docs/superpowers/specs/2026-06-01-qwen9b-generator-conformance-design.md` §7 (reuse), reinterpreted for the shared‑codebase goal (memory `qwen9b-27b-shared-codebase-goal`).

> **EXECUTION STATUS (2026-06-01): T0–T4 DONE; T5–T6 DEFERRED (option B).** The replicated blocks (RoPE/RMSNorm/Embedding) + the `num_devices`/`tt_ccl` prerequisite are implemented, validated single‑device (15/15 unit tests green, baseline‑equivalent), and final‑reviewed as 27B‑extendable. The **sharded** blocks — Task 5 (MLP) and Task 6 (LM‑head) — were **deferred to a 4‑device‑validatable setting**: their multi‑device value can't be validated on the 9B's single device and they carry single‑device regression risk + a `ModelArgs` TP‑config expansion. In‑code handoff comments mark the multi‑device dependencies (`grep "multi-device/TP handoff"`). Attention+GDN TP remains Plan 3. All work local/uncommitted on `qwen9b-p150`.

---

## Conventions for this plan

- **NO GIT COMMITS.** Per the project's local‑only rule, every "commit" step is replaced by a **Validation checkpoint** (run the named test/command, confirm output, leave files uncommitted).
- **Device required.** `pytest` runs on a Blackhole P150 with `HF_MODEL=/local/ttuser/atupe/Qwen9b`; use `python_env/bin/pytest`. Component tests use `n_layers`‑free single‑layer construction where possible; full‑model tests use `n_layers=4` for speed.
- **Reference, don't import.** The 27B impl (`/local/ttuser/atupe/qwen9b-metal/qwen27b-metal/tt-metal/models/demos/qwen35_27b/tt/`) is read‑only **reference**. Never import from it (separate checkout). Replicate its patterns using the shared `tt_transformers` / `models.common` classes.
- **Acceptance bar (spec D2):** each swap is accepted if its component **PCC vs the HuggingFace reference** stays in the validated band (>0.98; the existing `tests/unit/test_component_pcc.py` uses >0.90 for some) AND the existing suites pass. Not required to be bit‑identical to the current TT op.
- **new‑before‑delete:** for each block, add the mesh‑aware version and prove it (PCC) BEFORE deleting the bespoke one.

## Strategic note (read before executing)

This plan does **in‑place block mesh‑awareness** on the existing bespoke `Qwen35Model` (the chosen Option (a) — parallel convergence, 27B untouched). Two honesty caveats the executor should keep in mind:
1. **RoPE/RMSNorm/Embedding are replicated** (not sharded) — making them mesh‑aware is essentially "add a `ReplicateTensorToMesh` mapper / use the framework class." Low risk, modest multi‑device value (they'd replicate anyway).
2. **MLP/LM‑head are where real multi‑device value is** (column/row‑parallel sharding + `tt_all_reduce`). These require the model to hold a `tt_ccl` (Task 1) and to use the inherited sharded‑memcfg helpers.
3. **Attention + GDN multi‑device** (the bulk of TP weight/compute) and the deepest convergence — making `Qwen35Model` a framework `Transformer` subclass like 27B (which would also *inherit* the Generator contract Plan 1 hand‑wrote) — are **out of scope here**, deferred to a Plan 3. If the team later prefers the Transformer‑subclass route, Tasks 5–6 of this plan transfer directly (same framework `MLP`/`LMHead`).

## File structure

| File | Responsibility | Change |
|---|---|---|
| `tt/model.py` | `Qwen35Model` — gains `num_devices`/`tt_ccl`; constructs framework Embedding/RMSNorm/LMHead; uses mesh‑aware MLP/RoPE | Modify |
| `tt/rope.py` | `Qwen35RoPESetup` — replicate cos/sin across mesh; keep `get_cos_sin_host` (trace) | Modify |
| `tt/rms_norm.py` | keep `rms_norm_ttnn` only if still used; otherwise retire after Task 3 | Modify |
| `tt/mlp.py` | `Qwen35MLP` — sharded gate/up/down + `tt_all_reduce` (mesh‑aware), or delegate to framework `MLP` | Modify |
| `tests/unit/test_component_pcc.py` | per‑block PCC vs HF, parameterized to run on a 1×1 mesh | Modify |
| `tests/unit/test_multidevice_blocks.py` | New: per‑block "mesh‑aware == bespoke" equivalence on single device | **Create** |

---

## Task 0: Component‑PCC baseline vs HuggingFace (P0)

Lock in the current per‑block numerics (vs HF) so each mesh‑aware swap is provably non‑regressing.

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tests/unit/test_component_pcc.py`

- [ ] **Step 1: Confirm the existing component‑PCC tests pass (capture the bar)**

Run: `export HF_MODEL=/local/ttuser/atupe/Qwen9b && python_env/bin/pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_component_pcc.py -v`
Expected: PASS (or note which components have tests). Record each component's PCC from `-s` output. These are the reference numbers each swap must hold.

- [ ] **Step 2: Validation checkpoint (no commit)**

Note the per‑component PCC values (MLP, etc.) in the run log. No code change. Do not commit.

---

## Task 1: Model gains `num_devices` + `tt_ccl` plumbing (P1, prerequisite for MLP/LM‑head)

The sharded blocks need a CCL object. The framework creates `self.tt_ccl = TT_CCL(mesh_device)` in `Transformer.__init__`; the bespoke `Qwen35Model` must create one too (None on single device, matching the 27B tests' `TT_CCL(mesh_device) if num_devices>1 else None`).

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/model.py` (`Qwen35Model.__init__`)

- [ ] **Step 1: Add num_devices + tt_ccl in `__init__`**

In `Qwen35Model.__init__`, right after `self.mesh_device = mesh_device` (added in Plan 1), add:

```python
        self.num_devices = mesh_device.get_num_devices()
        # CCL collective for multi-device all-reduce; None on single device (the
        # framework MLP/LMHead/all-reduce ops no-op when there is nothing to reduce).
        if self.num_devices > 1:
            from models.tt_transformers.tt.ccl import TT_CCL

            self.tt_ccl = TT_CCL(mesh_device)
        else:
            self.tt_ccl = None
```

- [ ] **Step 2: Validation checkpoint (no commit)**

Syntax + import check: `python_env/bin/python -c "from models.demos.blackhole.qwen3_5_9b.tt.model import Qwen35Model; print('ok')"` → `ok`. Run the Plan‑1 contract suite to confirm no regression: `export HF_MODEL=/local/ttuser/atupe/Qwen9b && python_env/bin/pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_generator_contract.py -v` → all PASS. Do not commit.

---

## Task 2: RoPE — replicate cos/sin across the mesh (P1)

`Qwen35RoPESetup` builds cos/sin device tensors with a bare `from_torch(..., device=device)` (no mesh_mapper). On a multi‑device mesh that errors/places on one device. Mirror the 27B's replicated tables. Keep `get_cos_sin_host` (the prefill trace depends on it) and `cos_cpu`/`sin_cpu` unchanged.

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/rope.py` (the two device `from_torch` calls, ~lines 56‑64)
- Test: `models/demos/blackhole/qwen3_5_9b/tests/unit/test_multidevice_blocks.py`

- [ ] **Step 1: Write the equivalence test (mesh‑aware rope == current rope, single device)**

```python
# models/demos/blackhole/qwen3_5_9b/tests/unit/test_multidevice_blocks.py
import os
import pytest
import torch
import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.demos.blackhole.qwen3_5_9b.tt.rope import Qwen35RoPESetup

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2}]

@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_rope_cos_sin_shapes(device):
    args = Qwen35ModelArgs(mesh_device=device, max_batch_size=1, max_seq_len=2048)
    rope = Qwen35RoPESetup(device, args)
    pos = torch.arange(8).unsqueeze(0)
    cos, sin = rope.get_rot_mats(pos)
    # Replicated across a 1-device mesh: shape unchanged, last dim == rope_head_dim.
    assert cos.shape[-1] == args.rope_head_dim
    assert sin.shape[-1] == args.rope_head_dim
    # Host path (used by the prefill trace) still works.
    ch, sh = rope.get_cos_sin_host(0)
    assert tuple(ch.shape) == (1, 1, args.rope_head_dim)
```

- [ ] **Step 2: Run; confirm it passes against the CURRENT rope (baseline green)**

Run: `export HF_MODEL=/local/ttuser/atupe/Qwen9b && python_env/bin/pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_multidevice_blocks.py::test_rope_cos_sin_shapes -v`
Expected: PASS (this test passes before AND after the change — it guards shape/host‑path invariants while we add the mesh mapper).

- [ ] **Step 3: Add the replicate mesh mapper to the device cos/sin tables**

In `tt/rope.py`, the two `ttnn.from_torch(self.cos_cpu.unsqueeze(0), ...)` / `self.sin_cpu.unsqueeze(0)` calls (the `cos_device`/`sin_device` construction) — add `mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)`:

```python
        self.cos_device = ttnn.from_torch(
            self.cos_cpu.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.sin_device = ttnn.from_torch(
            self.sin_cpu.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
```

Also add the same `mesh_mapper=ttnn.ReplicateTensorToMesh(self.device)` to the `from_torch` in `get_rot_mats`'s general (prefill) path (the `cos_ttnn`/`sin_ttnn` device uploads). Do NOT change `get_cos_sin_host` (it returns host tensors for `copy_host_to_device_tensor`, no mapper needed) or `cos_cpu`/`sin_cpu`.

- [ ] **Step 4: Run the test + the Plan‑1 contract suite**

Run: `export HF_MODEL=/local/ttuser/atupe/Qwen9b && python_env/bin/pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_multidevice_blocks.py::test_rope_cos_sin_shapes models/demos/blackhole/qwen3_5_9b/tests/unit/test_generator_contract.py -v`
Expected: all PASS (rope still produces identical single‑device cos/sin; decode/trace unaffected).

- [ ] **Step 5: Validation checkpoint (no commit).** Confirm Step 4 green. Do not commit.

---

## Task 3: RMSNorm — use framework `RMSNorm` with `add_unit_offset` (P1)

Replace the free function `rms_norm_ttnn` + the manual `+1.0` weight offset (done at load in `layer.py` / `model.py`) with the framework `models.common.rmsnorm.RMSNorm`, which has `add_unit_offset=True` (the exact Qwen3.5 zero‑centered offset) and degrades to a plain `ttnn.rms_norm` on single device. Construct one `RMSNorm` per norm site (per‑layer `attention_norm`/`ffn_norm` + the final `norm`).

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/layer.py` (per‑layer norms), `tt/model.py` (final norm), `tt/rms_norm.py` (retire the free fn if unused)
- Test: `tests/unit/test_multidevice_blocks.py`

- [ ] **Step 1: Write a PCC test — framework RMSNorm vs torch reference (with +1 offset)**

```python
# append to tests/unit/test_multidevice_blocks.py
def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a - a.mean(), b - b.mean()]))[0, 1].item()

@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_rmsnorm_pcc(device):
    from models.common.rmsnorm import RMSNorm
    from models.tt_transformers.tt.common import Mode
    args = Qwen35ModelArgs(mesh_device=device, max_batch_size=1, max_seq_len=2048)
    dim = args.dim
    w = torch.randn(dim, dtype=torch.float32)
    sd = {"norm.weight": w}
    norm = RMSNorm(device=device, dim=dim, state_dict=sd, weight_key="norm",
                   weight_cache_path=None, weight_dtype=ttnn.bfloat16,
                   add_unit_offset=True, eps=args.norm_eps)
    x = torch.randn(1, 1, 32, dim, dtype=torch.float32)
    # Torch reference: zero-centered RMSNorm => weight is (w + 1)
    var = x.pow(2).mean(-1, keepdim=True)
    ref = (x * torch.rsqrt(var + args.norm_eps)) * (w + 1.0)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(norm(x_tt, mode=Mode.DECODE))
    assert _pcc(ref, out) > 0.99, f"RMSNorm PCC too low: {_pcc(ref, out)}"
```

- [ ] **Step 2: Run; expect FAIL or PASS depending on op availability**

Run: `export HF_MODEL=/local/ttuser/atupe/Qwen9b && python_env/bin/pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_multidevice_blocks.py::test_rmsnorm_pcc -v`
Expected: PASS (this validates the framework RMSNorm reproduces Qwen's zero‑centered norm). If it FAILS on a sharding/`is_distributed` arg, pass `is_distributed=None` and re‑run; if it fails on the `mode` enum import, import `Mode` from `models.tt_transformers.tt.common`.

- [ ] **Step 3: Swap the per‑layer norms in `layer.py`**

In `Qwen35DecoderLayer.__init__` (`tt/layer.py`), replace the manual norm‑weight load + `+1.0` + `rms_norm_ttnn` calls with two framework `RMSNorm` instances. Construct:

```python
        from models.common.rmsnorm import RMSNorm
        self.attention_norm = RMSNorm(
            device=mesh_device, dim=args.dim, state_dict=state_dict,
            weight_key="input_layernorm",
            state_dict_prefix=f"layers.{layer_idx}.",
            weight_cache_path=tensor_cache_path, weight_dtype=ttnn.bfloat16,
            add_unit_offset=True, eps=args.norm_eps,
        )
        self.ffn_norm = RMSNorm(
            device=mesh_device, dim=args.dim, state_dict=state_dict,
            weight_key="post_attention_layernorm",
            state_dict_prefix=f"layers.{layer_idx}.",
            weight_cache_path=tensor_cache_path, weight_dtype=ttnn.bfloat16,
            add_unit_offset=True, eps=args.norm_eps,
        )
```

(Confirm the per‑layer norm weight keys via `weight_mapping.py`: they pass through as `layers.{i}.input_layernorm.weight` / `layers.{i}.post_attention_layernorm.weight`.) In `forward`, replace `rms_norm_ttnn(x, self.attn_norm_weight, ...)` with `self.attention_norm(x, mode=mode)` and the ffn one with `self.ffn_norm(x, mode=mode)`. Map the layer's `mode` string ("prefill"/"decode") to the framework `Mode` enum (import `from models.tt_transformers.tt.common import Mode`; `Mode.PREFILL`/`Mode.DECODE`).

- [ ] **Step 4: Swap the final norm in `model.py`**

In `Qwen35Model.__init__`, replace the `norm_weight` load (`state_dict["norm.weight"] + 1.0` → `as_tensor`) with:

```python
        from models.common.rmsnorm import RMSNorm
        self.norm = RMSNorm(
            device=mesh_device, dim=args.dim, state_dict=state_dict, weight_key="norm",
            weight_cache_path=tensor_cache_path, weight_dtype=ttnn.bfloat16,
            add_unit_offset=True, eps=args.norm_eps,
        )
```

Replace every `rms_norm_ttnn(x, self.norm_weight, eps=self.norm_eps)` in `model.py` (prefill/decode/`_forward_decode`/traced paths) with `self.norm(x, mode=Mode.DECODE)` for decode and `self.norm(x, mode=Mode.PREFILL)` for prefill (import `Mode`). Remove `self.norm_weight`/`self.norm_eps` if now unused.

- [ ] **Step 5: Run the full unit suite (PCC + contract + decode equivalence)**

Run: `export HF_MODEL=/local/ttuser/atupe/Qwen9b && python_env/bin/pytest models/demos/blackhole/qwen3_5_9b/tests/unit/ -v`
Expected: all PASS. The Plan‑1 decode/traced tests (which compare against the baseline fixture) must still pass — the framework RMSNorm must be numerically equivalent (PCC>0.99) to the old `rms_norm_ttnn`+offset. If decode drifts beyond 1e‑2, investigate the offset/eps wiring before proceeding.

- [ ] **Step 6: Retire `rms_norm_ttnn` if unused**

`grep -rn "rms_norm_ttnn" models/demos/blackhole/qwen3_5_9b/` — if no callers remain, delete the function from `tt/rms_norm.py` (and the file if empty). If still used by an un‑migrated path, keep it.

- [ ] **Step 7: Validation checkpoint (no commit).** Confirm Step 5 green + Step 6 grep. Do not commit.

---

## Task 4: Embedding — framework `Embedding` (P1)

Replace the inlined `ttnn.embedding(token_ids, self.tok_embeddings, ...)` (with `tok_embeddings` loaded bare) with the framework `Embedding`, which uploads the table via `ShardTensor2dMesh(dims=(None, 3))` (replicates on a 1×1 mesh) and exposes `forward(x, memory_config=None)`.

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/model.py`
- Test: `tests/unit/test_multidevice_blocks.py`

- [ ] **Step 1: Write a PCC test — framework Embedding vs torch index_select**

```python
# append to tests/unit/test_multidevice_blocks.py
@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_embedding_pcc(device):
    from models.tt_transformers.tt.embedding import Embedding
    args = Qwen35ModelArgs(mesh_device=device, max_batch_size=1, max_seq_len=2048)
    vocab, dim = args.vocab_size, args.dim
    table = torch.randn(vocab, dim, dtype=torch.bfloat16)
    sd = {"tok_embeddings.weight": table}
    emb = Embedding(mesh_device=device, args=args, weight_cache_path=None, state_dict=sd, dtype=ttnn.bfloat16)
    ids = torch.tensor([[1, 5, 9, 13]], dtype=torch.int32)
    ref = torch.nn.functional.embedding(ids.long(), table.float())
    ids_tt = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.to_torch(emb(ids_tt)).float()[..., :dim].reshape(ref.shape)
    assert _pcc(ref, out) > 0.99
```

- [ ] **Step 2: Run; expect PASS** (validates the framework Embedding reproduces the lookup).

Run: `export HF_MODEL=/local/ttuser/atupe/Qwen9b && python_env/bin/pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_multidevice_blocks.py::test_embedding_pcc -v`
If it fails on `args.get_model_config()["EMB_WEIGHTS_MEMCFG"]` being absent, report it (the 9B ModelArgs may need that config key — check the framework default).

- [ ] **Step 3: Swap the embedding in `model.py`**

In `Qwen35Model.__init__`, replace the `self.tok_embeddings = ttnn.as_tensor(...)` block with:

```python
        from models.tt_transformers.tt.embedding import Embedding
        self.embd = Embedding(
            mesh_device=mesh_device, args=args, weight_cache_path=tensor_cache_path,
            state_dict=state_dict, dtype=ttnn.bfloat16,
        )
```

Replace every `ttnn.embedding(<ids>, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)` (in `prefill`/`decode`/`prefill_paged`/`_forward_decode`/`_forward_prefill_chunk`/etc.) with `self.embd(<ids>)`. Remove `self.tok_embeddings` if now unused.

- [ ] **Step 4: Run the full unit suite**

Run: `export HF_MODEL=/local/ttuser/atupe/Qwen9b && python_env/bin/pytest models/demos/blackhole/qwen3_5_9b/tests/unit/ -v`
Expected: all PASS (decode/traced baseline equivalence holds — the embedding lookup is numerically identical).

- [ ] **Step 5: Validation checkpoint (no commit).** Confirm Step 4 green. Do not commit.

---

## Task 5: MLP — sharded gate/up/down + `tt_all_reduce` (mesh‑aware) (P1)

Make the MLP column‑parallel on gate/up, row‑parallel on down, with `tt_all_reduce` after down — mirroring the framework `MLP` (which the 27B's `Qwen35FusedMLP` subclasses). At `num_devices=1` the shards are full and the all‑reduce is a no‑op, so single‑device output is unchanged. Simplest faithful route: **delegate to the framework `MLP`** rather than hand‑rolling the sharding.

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/mlp.py` (or `tt/layer.py` where the MLP is constructed)
- Test: `tests/unit/test_component_pcc.py` (existing MLP PCC test) + `tests/unit/test_multidevice_blocks.py`

- [ ] **Step 1: Write/confirm the MLP PCC test (vs torch SwiGLU), single device**

Reuse the existing `test_mlp_pcc` pattern in `tests/unit/test_component_pcc.py` (torch ref: `down(silu(gate(x)) * up(x))`, PCC>0.90). It currently builds `Qwen35MLP(device, mlp_state)`. After the swap it must still pass at the same PCC. Add an assertion that captures the bar (e.g. `assert pcc > 0.90`).

Run (baseline): `export HF_MODEL=/local/ttuser/atupe/Qwen9b && python_env/bin/pytest "models/demos/blackhole/qwen3_5_9b/tests/unit/test_component_pcc.py" -v -k mlp` → PASS; record PCC.

- [ ] **Step 2: Build the mesh‑aware MLP via the framework `MLP`**

The framework `MLP` (`models/tt_transformers/tt/mlp.py`) constructor is `(mesh_device, tt_ccl, args, state_dict, weight_cache_path, layer_num, dtype, model_config, state_dict_prefix=None, prefetcher=None)` and expects weights under `state_dict_prefix` with keys it transposes; it derives sharded memcfgs from `args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)` (gate/up) and `(args.hidden_dim // args.num_devices, args.dim)` (down), shards via `ShardTensor2dMesh`, and calls `tt_all_reduce(..., tt_ccl)` after down.

Two integration facts to resolve in this step (the framework MLP assumes the standard meta key scheme):
- **Key mapping:** the framework MLP reads `{prefix}.w1/w2/w3` or `gate_proj/up_proj/down_proj` depending on `args.get_state_dict_prefix`. The 9B remap keeps `layers.{i}.mlp.gate_proj/up_proj/down_proj`. Set `state_dict_prefix="layers.{layer_num}.mlp"` and confirm the framework MLP's expected sub‑keys match (`gate_proj`/`up_proj`/`down_proj`); if it expects `w1/w3/w2`, add those aliases in `weight_mapping.py` OR pass a prefix the framework resolves. Verify by reading `models/tt_transformers/tt/mlp.py` weight‑load lines before wiring.
- **Dtypes:** the framework MLP gets ff1/ff3/ff2 dtypes from `args`'s `decoders_optimizations`. To preserve Qwen's bf4 gate/up + bf8 down, confirm the 9B `ModelArgs.optimizations` yields those, or set them.

Construct in `Qwen35DecoderLayer.__init__` (replacing `Qwen35MLP(...)`):

```python
        from models.tt_transformers.tt.mlp import MLP
        self.feed_forward = MLP(
            mesh_device=mesh_device, tt_ccl=getattr(model, "tt_ccl", None), args=args,
            state_dict=state_dict, weight_cache_path=tensor_cache_path, layer_num=layer_idx,
            dtype=args.weight_dtype, model_config=args.get_model_config(),
            state_dict_prefix=f"layers.{layer_idx}.mlp",
        )
```

(The `tt_ccl` must thread from `Qwen35Model` (Task 1) into the layer/MLP constructor — add a `tt_ccl` param to `Qwen35DecoderLayer.__init__` and pass `self.tt_ccl` from the model.) In `forward`, replace `self.mlp.forward(x)` with `self.feed_forward(x, mode)` (framework MLP takes `Mode`).

- [ ] **Step 3: Run the MLP PCC test + full‑model short e2e**

Run: `export HF_MODEL=/local/ttuser/atupe/Qwen9b && python_env/bin/pytest "models/demos/blackhole/qwen3_5_9b/tests/unit/test_component_pcc.py" -v -k mlp`
Expected: PASS at ≥ the recorded PCC. Then a 4‑layer smoke: `python_env/bin/pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_generator_contract.py -v` → all PASS (the model still decodes correctly with the framework MLP).
**If the framework MLP's key/dtype/`model_config` expectations don't fit the 9B cleanly** (likely — it's built for the full framework `ModelArgs` surface), STOP and report: the fallback is to keep `Qwen35MLP` but make ITS loads sharded (`args.create_dram_sharded_mem_config` + `ShardTensor2dMesh(dims=(-2,-1)/(-1,-2))`) and append `tt_all_reduce(out, mesh_device, tt_ccl, ...)` after down — mirroring the 27B `Qwen35FusedMLP` forward without the framework MLP base. This is the more surgical, lower‑risk path and may be preferable.

- [ ] **Step 4: Validation checkpoint (no commit).** Confirm Step 3 green. Do not commit.

---

## Task 6: LM head — framework `LMHead` (vocab‑sharded + all‑reduce) (P1)

Replace the inlined single‑weight `ttnn.linear(x, self.lm_head_weight)` with the framework `LMHead`, which splits the vocab across devices (`size_per_device = padded_vocab // num_devices`), runs per‑split linears, concats, and `tt_all_reduce`s. Single device → one split, all‑reduce no‑op.

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/model.py`
- Test: `tests/unit/test_multidevice_blocks.py`

- [ ] **Step 1: Write a PCC test — framework LMHead vs torch linear**

```python
# append to tests/unit/test_multidevice_blocks.py
@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_lmhead_pcc(device):
    from models.tt_transformers.tt.lm_head import LMHead
    args = Qwen35ModelArgs(mesh_device=device, max_batch_size=1, max_seq_len=2048)
    dim, vocab = args.dim, args.vocab_size
    w = torch.randn(vocab, dim, dtype=torch.bfloat16)
    sd = {"output.weight": w}
    head = LMHead(args=args, mesh_device=device, tt_ccl=None, dtype=ttnn.bfloat8_b,
                  state_dict=sd, state_dict_prefix="", weight_cache_path=None,
                  max_columns_per_device=getattr(args, "max_columns_per_device_lm_head", 128256))
    x = torch.randn(1, 1, 32, dim, dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x.float(), w.float())
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(head(x_tt)).float()[..., :vocab].reshape(ref.shape[0], ref.shape[1], 32, vocab)
    assert _pcc(ref, out) > 0.98
```

- [ ] **Step 2: Run; expect PASS** (validates framework LMHead reproduces the projection). If it fails on a missing `args.padded_vocab_size`/`max_columns_per_device_lm_head`, report — those framework `ModelArgs` attrs may need a 9B value.

Run: `export HF_MODEL=/local/ttuser/atupe/Qwen9b && python_env/bin/pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_multidevice_blocks.py::test_lmhead_pcc -v`

- [ ] **Step 3: Swap the LM head in `model.py`**

In `Qwen35Model.__init__`, replace the `lm_head_weight` load with:

```python
        from models.tt_transformers.tt.lm_head import LMHead
        self.lm_head = LMHead(
            args=args, mesh_device=mesh_device, tt_ccl=self.tt_ccl, dtype=ttnn.bfloat8_b,
            state_dict=state_dict, state_dict_prefix="", weight_cache_path=tensor_cache_path,
            max_columns_per_device=getattr(args, "max_columns_per_device_lm_head", 128256),
        )
```

Replace every `ttnn.linear(x_last, self.lm_head_weight)` (prefill/decode/traced paths) with `self.lm_head(x_last)`. Remove `self.lm_head_weight` if unused. Note the framework LMHead output is padded to `padded_vocab_size`; downstream slicing already takes `[..., :vocab_size]` in `process_output_*`, so confirm those still slice correctly.

- [ ] **Step 4: Run the full unit suite + short demo**

Run: `export HF_MODEL=/local/ttuser/atupe/Qwen9b && python_env/bin/pytest models/demos/blackhole/qwen3_5_9b/tests/unit/ -v` → all PASS. Then `python_env/bin/pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py -v -s -k "traced_4k"` (with `QWEN9B_GDN_CHUNK_SEQ=1`) → PASS, coherent (full‑model end‑to‑end with all swapped blocks).

- [ ] **Step 5: Validation checkpoint (no commit).** Confirm Step 4 green. Do not commit.

---

## Deferred to Plan 3 (noted, not in scope)

- **Attention + GDN multi‑device:** column/row‑parallel sharding of QKV/out projections + `tt_all_reduce`, `num_devices`‑derived GDN dims (`gdn_nv_tp = linear_num_value_heads // num_devices`, etc.), KV‑head replication when `num_kv_heads < tp`. This is the bulk of the TP work and mirrors the 27B `TtGatedDeltaNet` / `Qwen35Attention`.
- **Data‑driven GDN dims under TP:** keep the 9B's HF‑config reads (`linear_num_value_heads` = 32 for 9B, 48 for 27B) and divide by `num_devices` — never hardcode (the 27B hardcodes; the 9B approach is preferable).
- **Optional deeper convergence:** restructure `Qwen35Model` as a framework `Transformer` subclass (as 27B is), which would inherit the Generator contract Plan 1 hand‑wrote and the full mesh wiring. Larger; decide separately.
- **Multi‑device validation:** none of this plan is *exercised* on >1 device here (9B is single‑device). The mesh‑aware code is validated to be a no‑op‑equivalent on 1 device; real 4‑device validation happens when this lands in the shared 27B path.

## Self‑review notes

- **Spec/goal coverage:** §7 blocks RoPE/RMSNorm/Embedding/MLP/LMHead → Tasks 2–6; the data‑driven‑config + tt_ccl prerequisite → Task 1; attention/GDN TP + Transformer‑subclass → explicitly deferred to Plan 3.
- **Validation:** every swap has a per‑block PCC test (vs HF/torch ref) AND must keep the Plan‑1 decode/traced baseline equivalence green (full‑model regression). new‑before‑delete honored (add+prove, then delete the bespoke version).
- **Single‑device correctness:** relies on the verified fact that `tt_all_reduce`/`ShardTensor*Mesh`/`//num_devices` are no‑ops at `num_devices=1`; the 9B is the single‑device test bed, the same code targets 27B/4‑device.
- **Risk flags:** Tasks 5 (MLP) and 6 (LMHead) depend on the framework blocks' `ModelArgs` surface (key scheme, dtypes, `padded_vocab_size`, `max_columns_per_device_lm_head`, `EMB_WEIGHTS_MEMCFG`). Each task's run‑step says to STOP/report if the 9B `ModelArgs` lacks the needed attr, with the surgical fallback (shard the existing Qwen block + append `tt_all_reduce`, mirroring 27B's `Qwen35FusedMLP`) called out for MLP.
```
