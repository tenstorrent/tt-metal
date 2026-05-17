# Extending tt_hw_planner

Every checker in the planner is an append-only registry — adding a new entry
is a single dataclass / function literal, never a structural change. This
doc shows what to add and where, with a worked example for each case.

Pick the row that matches what you want to add:

| You want to add… | File to edit | Skim time |
|---|---|---|
| 1. A new HF building block (e.g. attention sinks) | `compatibility.py` | 2 min |
| 2. A new kernel-level constraint (new `TT_FATAL` predicate) | `kernel_constraints.py` | 3 min |
| 3. A model that is now supported | `compatibility.py` (`SUPPORTED_HF_MODELS`) | 30 sec |
| 4. A new ttnn op | `kernel_constraints.py` (new check fn) + `compatibility.py` | 5 min |
| 5. A new hardware box (e.g. p300c-Galaxy) | `hardware.py` | 2 min |
| 6. A new architecture family (e.g. linear attention) | `architecture.py` | 10 min |
| 7. A new HF pipeline tag (e.g. text-ranking) | `probe.py` | 1 min |
| 8. A new dtype (e.g. fp4_e2m1) | `architecture.py` (`DTYPE_BYTES`) | 30 sec |

---

## 1. Add a new HF building block

Use when an HF model class introduces a concept the planner doesn't recognize
yet — e.g. *attention sinks*, *MoE with shared experts*, *vision-encoder
RoPE*.

**File:** `compatibility.py`
**Where:** append to `BUILDING_BLOCKS`

### Example: detect attention sinks (GPT-OSS, recent Mistral)

```python
BuildingBlock(
    name="Attention sinks",
    description="Persistent leading tokens preserved across the KV window",
    needed_when=lambda c: bool(_text_config(c).get("attention_sink_size")
                               or _text_config(c).get("sink_token_size")),
    tt_path="models/demos/gpt_oss/tt/attention.py",
    status_when_needed=Status.PARTIAL,
    effort_when_needed=Effort.LIGHT,
    notes=("Used by GPT-OSS. SDPA accepts an `attention_sink` argument but "
           "streaming-compute fast path rejects it."),
),
```

### Anatomy

- **`name`** — short label shown in the report
- **`description`** — one-sentence purpose
- **`needed_when(cfg)`** — predicate on the raw HF config dict. Use the
  helpers `_text_config(c)`, `_is_moe(c)`, etc. for nested configs.
- **`tt_path`** — file or directory implementing it, or `None` if missing
- **`status_when_needed`** — `SUPPORTED` / `PARTIAL` / `MISSING`
- **`effort_when_needed`** — `DROP_IN` / `LIGHT` / `HEAVY` / `NEW`
- **`notes`** — gotchas / limitations / pointers

### Verify

```bash
python -m scripts.tt_hw_planner compat <hf-model-id-with-this-feature>
# The block should appear in Section 1.
```

---

## 2. Add a new kernel-level constraint

Use when you find a new `TT_FATAL(...)` predicate in a ttnn device op, or
when an existing op gets a new restriction.

**File:** `kernel_constraints.py`
**Where:** new `check_<thing>(cfg, tp)` function, then add it to
`CONSTRAINT_CHECKS`.

### Example: enforce that `qk_rope_head_dim` is exactly 64 for MLA decode

```python
def check_mla_rope_head_dim(cfg: dict, _tp: int) -> List[KernelFinding]:
    if not _is_mla(cfg):
        return []
    t = _text_cfg(cfg)
    rope_hd = t.get("qk_rope_head_dim")
    if rope_hd is None:
        return []
    return [KernelFinding(
        op="ttnn.transformer.flash_multi_latent_attention_decode",
        field="qk_rope_head_dim",
        value=rope_hd,
        constraint="qk_rope_head_dim must equal 64 for the current MLA flash kernel",
        passes=(int(rope_hd) == 64),
        severity=Severity.BLOCKER,
        fix="Custom MLA variants need their own kernel; raise an issue.",
        source="ttnn/cpp/.../sdpa_decode_device_operation.cpp:NNN",
    )]

CONSTRAINT_CHECKS.append(check_mla_rope_head_dim)
```

### Rules of thumb

- **One check function per logical predicate** — keep them focused so the
  report rows are readable.
- **Cite the source line** in the `source` field so reviewers can verify
  against the C++.
- **Skip irrelevant configs** by returning `[]` early. The example above
  returns `[]` for non-MLA models.
- **Severity selection:**
  - `BLOCKER` — kernel will refuse to launch
  - `WARN` — kernel falls back to a slower path or auto-pads
  - `INFO` — automatic adjustment with no perf cost
- **TP-dependent checks** must include the literal substring `TP(` in
  their `constraint` string — that's how the renderer routes them into the
  per-TP divisibility table.

### Verify

```bash
python -m scripts.tt_hw_planner compat <model-with-feature> --verbose
# Check the new row appears in Section 2 with the right severity.
```

---

## 3. Mark a model as already supported

When a new HF model id gets brought up in `tt_transformers/`, register it
so the planner stops saying "FEASIBLE WITH WORK" and starts saying
"ALREADY SUPPORTED".

**File:** `compatibility.py`
**Where:** add to `SUPPORTED_HF_MODELS`.

```python
SUPPORTED_HF_MODELS = {
    ...
    "Qwen/Qwen4-72B-Instruct",     # added after PR #12345
    "google/gemma-4-27b-it",
}
```

You should also add the model to `MAX_PREFILL_CHUNK_SIZES_DIV1024` in
`models/tt_transformers/tt/model_config.py` (that's the canonical truth
source for the tt-metal side). The planner's list is a mirror.

---

## 4. Add a new ttnn op

When a brand-new ttnn op is exposed in `ttnn.experimental.*` or
`ttnn.transformer.*` and a HF model needs it.

This is two steps: (1) add the building block, (2) add its kernel checks.

**Step 1 — `compatibility.py`** declares "do we have a module for this?":

```python
BuildingBlock(
    name="Ring all-to-all",
    description="MoE-style expert dispatch over the mesh fabric",
    needed_when=lambda c: _is_moe(c) and _g(c, "moe_layer_freq") > 1,
    tt_path="ttnn.experimental.all_to_all_dispatch",
    status_when_needed=Status.SUPPORTED,
    effort_when_needed=Effort.DROP_IN,
    notes="Requires fabric configured; see scripts/tt_hw_planner/smoke.py for setup.",
),
```

**Step 2 — `kernel_constraints.py`** declares "do its preconditions hold?":

```python
def check_all_to_all_dispatch(cfg: dict, tp: int) -> List[KernelFinding]:
    if not _text_cfg(cfg).get("num_local_experts"):
        return []
    e = int(_text_cfg(cfg)["num_local_experts"])
    return [KernelFinding(
        op="ttnn.experimental.all_to_all_dispatch",
        field="num_local_experts",
        value=e,
        constraint=f"num_local_experts must be divisible by TP({tp})",
        passes=(e % tp == 0),
        severity=Severity.BLOCKER,
        fix="Pad experts or pick a different mesh.",
        source="ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/...",
    )]

CONSTRAINT_CHECKS.append(check_all_to_all_dispatch)
```

---

## 5. Add a new hardware box

Use when a new SKU ships (e.g. p300c QuietBox 3, BH-Galaxy).

**File:** `hardware.py`
**Where:** append to `HARDWARE`.

```python
Box(
    name="QB3",
    arch="Blackhole",
    chips=8,
    hbm_per_chip_gb=32.0,
    mesh_shapes=[(1,1), (1,2), (2,1), (1,4), (2,2), (4,1), (1,8), (2,4), (4,2), (8,1)],
    eth_link_gbps=200.0,
    notes="8x Blackhole p300c, 256 GB total HBM.",
),
```

If the chip's overhead differs from existing arches, also add an entry to
`OVERHEAD_BY_ARCH` (e.g. `"Blackhole-X": Overhead(dispatch_gb=..., ...)`)
and update `Box.arch`.

### Calibration

After adding, run `python -m scripts.tt_hw_planner calibrate --box QB3 --mesh 1,4`
on real hardware. That writes the measured overhead to `data/calibration.yaml`,
which the planner picks up automatically.

---

## 6. Add a new architecture family

Use when a model has fundamentally different memory characteristics — e.g.
hybrid SSM-attention, linear attention, RetNet, MoR.

**File:** `architecture.py`
**Where:** subclass `MemoryModel`, then add detection in `select_model()`.

```python
class LinearAttentionModel(MemoryModel):
    """O(d^2) state instead of O(seq * d) KV cache (RetNet, Mamba-1)."""

    def weights_bytes(self, dtype: str) -> int:
        return super().weights_bytes(dtype)  # standard

    def kv_cache_bytes(self, batch: int, seq: int, kv_dtype_bytes: float) -> int:
        # State is fixed-size; ignores seq.
        return self.arch.num_hidden_layers * self.arch.hidden_size**2 * 2 * int(kv_dtype_bytes)

    def activations_bytes(self, batch: int, seq: int, dtype: str) -> int:
        return super().activations_bytes(batch, seq, dtype)
```

Then dispatch in `select_model`:

```python
def select_model(arch, total_params, weight_bytes_on_disk):
    family = arch.family
    if family == "linear":
        return LinearAttentionModel(arch, total_params, weight_bytes_on_disk)
    ...
```

And update `detect_architecture(cfg)` to return `"linear"` for the right
`model_type` strings.

---

## 7. Add a new HF pipeline tag

Use when HF introduces a new `pipeline_tag` (e.g. `"text-ranking"`,
`"feature-extraction"`).

**File:** `probe.py`
**Where:** the `PIPELINE_CATEGORY` dict near the top.

```python
PIPELINE_CATEGORY = {
    ...
    "text-ranking": "Embed",
    "feature-extraction": "Embed",
}
```

The category controls the recommended dtype list (`_dtypes_for` in
`cli.py`) and the report's confidence message.

---

## 8. Add a new dtype

Use when tt-metal adds support for a new precision (e.g. fp4, mxfp4).

**File:** `architecture.py`
**Where:** `DTYPE_BYTES` dict.

```python
DTYPE_BYTES = {
    "fp32": 4.0, "fp16": 2.0, "bf16": 2.0,
    "fp8": 1.0,
    "bfp8_b": 1.0625,   # 1 byte + 1 exponent / 16 mantissas
    "bfp4_b": 0.5625,
    "fp4": 0.5,         # <-- new
    "mxfp4_e2m1": 0.5625,
}
```

Then add it to the CLI's dtype `choices` in `cli.py` (`pp.add_argument
"--dtype"`).

---

## Style notes

- Every new entry is a literal — no factories, no metaclasses. The whole
  point is that future authors can read the registry top-to-bottom.
- Predicates take the raw HF config dict — never assume it has been
  flattened. Use the existing `_text_cfg(c)` / `_g(c, ...)` helpers.
- Always cite the *exact* C++ file the rule comes from in `source`. If the
  C++ relaxes the rule later, the citation makes the stale entry findable.
- One predicate per `KernelFinding`. Don't bundle "head_dim and GQA ratio
  and TP divisibility" into one row.
- If you're unsure whether a constraint is blocker / warn / info, look at
  whether the C++ raises `TT_FATAL` (blocker), falls back silently to a
  slower path (warn), or silently pads / clamps (info).
