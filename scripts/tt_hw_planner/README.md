# tt_hw_planner

Pre-flight memory planner for Tenstorrent hardware.

Given a HuggingFace model ID, this package answers: **which Tenstorrent box
(N150 / N300 / T3K / QB2 / Galaxy) can run this model, at what dtype, at
what mesh shape, and with how much headroom?** No weights downloaded; no
model code run; only HF Hub metadata + `config.json`.

The design follows the structure used by NVIDIA TRT-LLM's model deployment
planner: a static memory model that's calibrated against measured hardware,
plus optional smoke-tests of the hot ops at the model's exact shapes.

> **Adding support for something new?** See [`EXTENDING.md`](EXTENDING.md) for
> the append-only registries (building blocks, kernel constraints, hardware
> boxes, architecture families, dtypes, pipeline tags) and worked examples
> for each.

---

## Commands

### `plan` — memory-budget recommendation (default)

```bash
# implicit (auto-detects HF model IDs)
python scripts/tt_hw_recommender.py Qwen/Qwen3-32B

# explicit
python -m scripts.tt_hw_planner plan Qwen/Qwen3-32B

# show every TP×PP combination, JSON output
python -m scripts.tt_hw_planner plan deepseek-ai/DeepSeek-V4-Flash --explore-pp --format json
```

Tunable knobs:
| Flag | What |
|---|---|
| `--batch N` | batch size for KV/activation accounting (default 1) |
| `--seq N` | sequence length / generation horizon (default 8192) |
| `--dtype {bf16,bfp8_b,bfp4_b,fp8,fp4,fp16,fp32}` | repeat to evaluate multiple |
| `--kv-dtype {bf16,fp16,fp32}` | KV cache dtype (default bf16) |
| `--box {N150,N300,T3K,QB2,Galaxy}` | limit fit table to one or more boxes |
| `--all-meshes` | report every canonical mesh, not just largest TP |
| `--explore-pp` | enumerate TP×PP combinations (e.g. T3K TP=4 PP=2) |
| `--format {table,json,markdown}` | output backend |

### `compat` — bring-up compatibility checklist

Walks the HuggingFace config against a registry of TT building blocks *and* a
registry of kernel-level constraints (the `TT_FATAL` preconditions inside
ttnn device ops). No hardware required.

```bash
python -m scripts.tt_hw_planner compat Qwen/Qwen3-32B
python -m scripts.tt_hw_planner compat deepseek-ai/DeepSeek-V3 --format json
python -m scripts.tt_hw_planner compat google/gemma-3-27b-it --verbose
python -m scripts.tt_hw_planner compat <model> --skip-kernel-check     # modules only
python -m scripts.tt_hw_planner compat <model> --tp-grid 1 4           # custom TPs
```

The report has two sections:

**Section 1 — Building-block availability.** Per HF-equivalent module (decoder,
attention, MLP, norm, RoPE, vision tower, ...) reports drop-in / partial / missing
and the TT file or directory that implements it.

**Section 2 — Kernel-level constraints.** Per ttnn op (matmul, SDPA, RoPE,
RMSNorm, embedding, top-k, MoE) reports whether the model's shapes and dtypes
satisfy the kernel's preconditions. Severity tiers:
- `[FAIL]` blocker — kernel will refuse to launch (e.g. `head_dim % 32 != 0`)
- `[warn]` runtime warning — works but may fall back to a slow path or pad
- `[info]` informational — automatic padding overhead, etc.

A "Per-TP divisibility" sub-table shows which mesh shapes are feasible for the
model's hidden / heads / intermediate sizes (e.g. TP=2,4 pass; TP=8,32 fail).

Top-line verdicts:
| Verdict | Meaning |
|---|---|
| `ALREADY SUPPORTED` | Model id is in the `tt_transformers` prefill/perf tables. |
| `READY` | All required blocks exist in `models/tt_transformers/`; expect drop-in port. |
| `FEASIBLE WITH WORK` | All blocks exist *somewhere* but some are demo-only and need lifting. |
| `BLOCKED` | At least one required block has no TT implementation; needs new kernels. |

Exit code is `2` when there are blockers at TP=1 (i.e. unfixable by mesh choice),
`0` otherwise — usable in CI.

### `scaffold` — generate a first-draft port for a NEW model

Closes the loop between "compat says yes" and "demo can run it". Given a
HuggingFace id that *isn't* already in tt-metal's supported list, picks the
closest already-ported sibling and emits the minimum-viable set of edits to
make tt_transformers recognize it:

- Adds an entry to `MAX_PREFILL_CHUNK_SIZES_DIV1024` (in `tt/model_config.py`).
- Adds an entry to `trace_region_size_dict` (in `demo/trace_region_config.py`).
- Creates `model_params/<new-tail>/` with the sibling's JSONs copied over.

```bash
# dry-run (prints the plan + diff)
python -m scripts.tt_hw_planner scaffold Qwen/Qwen2.5-14B-Instruct

# apply directly to the working tree
python -m scripts.tt_hw_planner scaffold Qwen/Qwen2.5-14B-Instruct --apply

# emit a `git apply`-compatible patch
python -m scripts.tt_hw_planner scaffold Qwen/Qwen2.5-14B-Instruct --format patch > port.diff

# structured output for CI
python -m scripts.tt_hw_planner scaffold Qwen/Qwen2.5-14B-Instruct --format json
```

What `scaffold` does and doesn't do:

| Compat verdict | `scaffold` behaviour |
|---|---|
| `ALREADY SUPPORTED` | refuses cleanly — nothing to scaffold, just `prepare` |
| `READY` | emits the full first-draft patch |
| `FEASIBLE WITH WORK` (no MISSING blocks) | emits the patch + a `[warn]` per PARTIAL block |
| `FEASIBLE WITH WORK` (with MISSING blocks) | refuses; lists what's missing |
| `BLOCKED` | refuses; lists the missing TTNN kernel(s) |

Warnings to take seriously:

- **Sibling size mismatch** — the sibling's chunk-size row was tuned for the sibling's size. If the new model is much smaller, the `None` entries for small boxes may now be feasible (and the safer copied value pessimistic). Edit the table after applying.
- **PARTIAL blocks** — sliding-window attention works but breaks chunked prefill; MLA is partial; etc. Read each warning and decide whether your use case hits the limitation.

End-to-end new-model bring-up:

```bash
# 1. Drop a first-draft port into the tree
python -m scripts.tt_hw_planner scaffold Qwen/Qwen2.5-14B-Instruct --apply

# 2. Run it
python -m scripts.tt_hw_planner prepare  Qwen/Qwen2.5-14B-Instruct --execute

# 3. Undo if you want to discard
git restore models/tt_transformers/tt/model_config.py \
            models/tt_transformers/demo/trace_region_config.py
rm -rf models/tt_transformers/model_params/Qwen2.5-14B-Instruct/
```

### `prepare` — emit ready-to-run env + pytest invocation

Bridges `plan` + `compat` to `models/tt_transformers/demo/simple_text_demo.py`.
Given an HF model id, picks the recommended `(box, mesh, dtype)`, maps to the
demo's `MESH_DEVICE` parametrization, checks per-model tuning tables, and emits
the env vars + pytest command. Optionally runs it.

```bash
# print the env + command for the recommended box
python -m scripts.tt_hw_planner prepare Qwen/Qwen3-32B

# force a different box / mesh
python -m scripts.tt_hw_planner prepare Qwen/Qwen3-32B --box T3K --mesh 1,8

# write a self-contained bash script
python -m scripts.tt_hw_planner prepare Qwen/Qwen3-32B --write-script bringup.sh

# emit JSON for CI consumption
python -m scripts.tt_hw_planner prepare Qwen/Qwen3-32B --format json

# actually run it (only when compat is ALREADY SUPPORTED or READY)
python -m scripts.tt_hw_planner prepare Qwen/Qwen3-32B --execute
```

`prepare` checks two per-model tables and warns when an entry is missing:

| Table | File | Symptom if missing |
|---|---|---|
| `MAX_PREFILL_CHUNK_SIZES_DIV1024` | `tt/model_config.py` | falls back to `MAX_PREFILL_CHUNK_SIZE=4` (slow prefill) |
| `trace_region_size_dict` | `demo/trace_region_config.py` | uses the parametrize default; trace may OOM |

Pytest-shaping knobs (all optional):
| Flag | What |
|---|---|
| `--batch N` | passes through as `--batch_size N` |
| `--max-seq-len N` | demo `--max_seq_len` (default 1024) |
| `--max-generated-tokens N` | demo `--max_generated_tokens` (default 200) |
| `--accuracy` | use the accuracy parametrization instead of performance |
| `--no-trace` | disable `--enable_trace` (required for accuracy + teacher-forcing) |
| `--no-paged-attention` | disable paged KV |
| `--no-instruct` | raw completion path instead of chat template |
| `--allow-port` | permit `--execute` when compat says porting is required |

Exit codes: `0` when a runnable command is emitted, `2` when blockers prevent
one (missing config, NO_FIT, BLOCKED compat, or kernel-level blockers at TP=1).

### `calibrate` — measure actual usable HBM (Phase 2)

Opens a mesh on real hardware and binary-searches the largest single-tensor
allocation that succeeds.  Writes the measured per-chip overhead to
`scripts/tt_hw_planner/data/calibration.yaml`.  Future `plan` invocations
automatically use the measurement instead of the analytical estimate.

```bash
# requires the tt-metal python env to be active
source python_env/bin/activate

# single-chip on QB2
python -m scripts.tt_hw_planner calibrate --box QB2 --mesh 1,1

# the full 4-chip QB2 mesh
python -m scripts.tt_hw_planner calibrate --box QB2 --mesh 1,4

# T3K (if you have one)
python -m scripts.tt_hw_planner calibrate --box T3K --mesh 1,8
```

The output looks like:

```
  Measured per-chip usable:   31.75 GB
  Predicted per-chip usable:  29.20 GB
  Implied per-chip overhead:  0.25 GB
  Analytical estimate is PESSIMISTIC by 2.55 GB/chip
  Wrote calibration entry. Now active for box QB2.
```

To re-check what's active:
```bash
python -m scripts.tt_hw_planner show-overhead
```

### `smoke-test` — run hot ops at the model's shapes on hardware (Phase 2)

For a (model, box, mesh) you're considering, opens the mesh and runs:
matmul (Q-projection size + MLP up-proj), rms_norm, scaled_dot_product_attention,
and all_gather (when TP>1).  Returns pass/fail per op.

```bash
python -m scripts.tt_hw_planner smoke-test \
  --model Qwen/Qwen3-1.7B --box QB2 --mesh 1,1 --seq 2048
```

A passing smoke-test means **the ops your model needs work on this hardware
at these shapes**.  A failing op is a deal-breaker that needs to be fixed
in TTNN before the port can land.

### `list-meshes` / `show-overhead`

```bash
python -m scripts.tt_hw_planner list-meshes        # canonical mesh topology per box
python -m scripts.tt_hw_planner show-overhead      # active per-chip overhead (cal or analytical)
```

---

## Architecture

```
scripts/tt_hw_planner/
├── probe.py            HF metadata fetch + architecture detection
├── architecture.py     MemoryModel hierarchy:
│                         Dense / MLA / SlidingWindow / SSM / MoE
├── compatibility.py    HF-block ↔ tt-metal building-block registry
├── kernel_constraints.py  TT_FATAL precondition mirror
├── hardware.py         Box specs + per-arch overhead constants;
│                       auto-loads calibration on import
├── parallelism.py      Mesh enumeration + TP/PP sharding
├── verdict.py          Per-row fit + tightness classifier
├── bringup.py          Bridge from verdict → MESH_DEVICE + pytest command
├── scaffold.py         First-draft port generator (table inserts + JSON copies)
├── report.py           table / JSON / markdown formatters
├── device.py           TTNN open/close + memory measurement (Phase 2)
├── smoke.py            Hot-op probe at the model's shapes (Phase 2)
├── calibration.py      Measurement DB load/save (Phase 2)
└── data/
    └── calibration.yaml  Persisted measurements
```

Architecture detection auto-dispatches the memory model:

| If config has… | Detected family | KV cache formula |
|---|---|---|
| `kv_lora_rank` | MLA (DeepSeek) | `(kv_lora_rank + qk_rope_head_dim) × seq × layers` |
| `model_type` in {mamba, rwkv, ...} | SSM | constant (state size, not seq) |
| `sliding_window > 0` | SlidingWindow | capped at `min(seq, window)` |
| `num_local_experts > 0` | MoE wrapper around the above | parent's KV formula |
| (otherwise) | Dense | `2 × batch × seq × kv_heads × head_dim × layers` |

---

## Calibration workflow (the recommended Phase 2 onboarding)

1.  **First-time calibration on your box.**  For each mesh shape you plan to use:
    ```bash
    python -m scripts.tt_hw_planner calibrate --box QB2 --mesh 1,1
    python -m scripts.tt_hw_planner calibrate --box QB2 --mesh 1,4
    ```
    Each call takes ~30–60 seconds.  The first call also warms the kernel
    cache, so subsequent runs are faster.

2.  **Verify the calibration took effect.**
    ```bash
    python -m scripts.tt_hw_planner show-overhead
    # Expect "measured (X.XX GB overhead, from calibration.yaml)" for QB2.
    ```

3.  **Re-plan a model and observe the headroom delta.**
    ```bash
    python -m scripts.tt_hw_planner plan Qwen/Qwen3-32B
    # The QB2 headroom should be larger than before (analytical was pessimistic).
    ```

4.  **Smoke-test before committing.**  For each TIGHT or ROOM verdict in
    your plan output, run the smoke-test to confirm the hot ops actually
    work on your hardware at the model's shapes:
    ```bash
    python -m scripts.tt_hw_planner smoke-test \
      --model Qwen/Qwen3-32B --box QB2 --mesh 1,4
    ```

5.  **Commit the calibration.yaml** to your branch so others on the same
    hardware family get the same calibrated numbers.

---

## What's still TODO (Phase 3+)

- **Expert Parallelism (EP)** for MoE models — currently only TP×PP is
  enumerated.  Adding EP would unlock MoE-specific fits.
- **Per-op memory profiling** — today's smoke-test measures pass/fail and
  wall-clock, but not memory peaks.  Phase 3 would integrate `tt-smi`
  telemetry during the smoke run.
- **Multi-host (TG, multi-Galaxy)** — currently the largest box modelled
  is a single Galaxy.  Trillion-parameter models would need this.
- **Calibration regression suite** — given a calibration.yaml + a list of
  known-working `(model, box, mesh)` triples, verify the planner's
  verdicts agree with the README's documented hardware.
