# tt_hw_planner

Pre-flight memory planner for Tenstorrent hardware.

Given a HuggingFace model ID, this package answers: **which Tenstorrent box
(N150 / N300 / T3K / QB2 / Galaxy) can run this model, at what dtype, at
what mesh shape, and with how much headroom?** No weights downloaded; no
model code run; only HF Hub metadata + `config.json`.

The design follows the structure used by NVIDIA TRT-LLM's model deployment
planner: a static memory model that's calibrated against measured hardware,
plus optional smoke-tests of the hot ops at the model's exact shapes.

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
├── probe.py          HF metadata fetch + architecture detection
├── architecture.py   MemoryModel hierarchy:
│                       Dense / MLA / SlidingWindow / SSM / MoE
├── hardware.py       Box specs + per-arch overhead constants;
│                     auto-loads calibration on import
├── parallelism.py    Mesh enumeration + TP/PP sharding
├── verdict.py        Per-row fit + tightness classifier
├── report.py         table / JSON / markdown formatters
├── device.py         TTNN open/close + memory measurement (Phase 2)
├── smoke.py          Hot-op probe at the model's shapes (Phase 2)
├── calibration.py    Measurement DB load/save (Phase 2)
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
