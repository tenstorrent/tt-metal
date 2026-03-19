# Matmul N150 benchmark protocol (Milestone 1)

This document describes the **local** scaffolding for model-traced `ttnn.matmul` on Wormhole (e.g. N150): deterministic **smoke / train / holdout** splits over existing sweep vectors, a **report** over `results_export` JSON, and an **acceptance policy** for future gating.

It does **not** change CI workflows or add hard gates. It reuses:

- `tests/sweep_framework/master_config_loader_v2.py` → traced configs
- `tests/sweep_framework/sweeps/model_traced/matmul_model_traced.py` → sweep module
- `tests/sweep_framework/sweeps_parameter_generator.py` → `vectors_export/*.json`
- `tests/sweep_framework/sweeps_runner.py` → hardware runs → `results_export/*.json`

## End-to-end flow

1. **Generate vectors** — `MasterConfigLoader.get_suite_parameters("matmul")` runs at import time; the generator expands the `model_traced` suite and writes `tests/sweep_framework/vectors_export/model_traced.matmul_model_traced.json` (plus optional `__mesh_*` variants).

2. **Partition** — `matmul_n150_protocol.py partition` reads the merged `model_traced` suite and assigns each `input_hash` to **smoke**, **train**, or **holdout** (see below).

3. **Write subset JSON** — `write-json` copies the selected vectors into `benchmark_protocol/generated/matmul_n150_*.json` for `sweeps_runner --vector-source file`.

4. **Run** — `sweeps_runner` executes vectors; results land in `tests/sweep_framework/results_export/` as a JSON **list of OpTest-like records** (metrics include `e2e_perf_ms` when `--perf` is set).

5. **Report** — `matmul_n150_protocol.py report` joins results to the manifest by `input_hash` and prints pass rate, timeouts, p50/p95 e2e, and optional memory p50s.

## Deterministic split

- **Stratum** = `(traced_source, input_a_shape, input_b_shape)` with stable string forms.
- Within each stratum, vectors are sorted by **`input_hash`** (lexicographic).
- Strata are sorted by stratum key; vectors are **round-robin interleaved** across strata (preserves diversity in prefix splits).
- **Smoke** = first `smoke_max` vectors in that order (default 16).
- **Remainder** split: **train** = floor(`train_fraction_of_remainder` × len(remainder)), **holdout** = rest (defaults: fraction `0.58`).

Changing `protocol_version` in the manifest or the partition parameters changes membership; document any change when tuning.

## Commands (N150 / local)

From the repo root, with the project Python environment active and a device available:

```bash
export ARCH_NAME=wormhole_b0          # required by sweeps_runner
export RUNNER_LABEL=N150              # optional; appears in result card_type

# 1) Generate model-traced matmul vectors (requires ttnn / traced_operations data)
#    From repo root; use `python3` if `python` is not on PATH.
( cd tests/sweep_framework && python3 sweeps_parameter_generator.py \
  --module-name model_traced.matmul_model_traced \
  --suite-name model_traced \
  --model-traced all )

# 2) Build manifest + protocol JSON slices
python3 tests/sweep_framework/benchmark_protocol/matmul_n150_protocol.py partition
python3 tests/sweep_framework/benchmark_protocol/matmul_n150_protocol.py write-json

# 3) Run combined protocol set with e2e perf (cwd must be sweep_framework for imports)
( cd tests/sweep_framework && python3 sweeps_runner.py \
  --module-name model_traced.matmul_model_traced \
  --suite-name model_traced \
  --vector-source file \
  --file-path benchmark_protocol/generated/matmul_n150_protocol_all.json \
  --result-dest results_export \
  --perf \
  --summary )

# 4) Summarize (adjust glob if your export prefix differs)
python3 tests/sweep_framework/benchmark_protocol/matmul_n150_protocol.py report \
  --results-glob 'tests/sweep_framework/results_export/model_traced_*.json' \
  --json-out tests/sweep_framework/benchmark_protocol/generated/matmul_n150_last_report.json
```

If multiple prior runs exist, prefer `--results <latest_file.json>` to avoid mixing rows from different runs.

**One-shot** (same steps; optional `MEASURE_MEMORY=1` adds graph memory capture):

```bash
chmod +x tests/sweep_framework/benchmark_protocol/run_matmul_n150_protocol.sh
./tests/sweep_framework/benchmark_protocol/run_matmul_n150_protocol.sh all
```

If `build_metal.sh` has not been run, step (1) may fail until the Python package and `model_tracer/traced_operations` inputs are available—fix the build/environment first.

## Machine-readable outputs

- **Manifest**: `tests/sweep_framework/benchmark_protocol/generated/matmul_n150_protocol_manifest.json`
- **Protocol vectors**: `matmul_n150_{smoke,train,holdout}.json`, `matmul_n150_protocol_all.json`
- **Report**: `--json-out` from the `report` subcommand (full structure includes per-split aggregates and train vs holdout p95 comparison)

## Related docs

- [Acceptance policy](./matmul_n150_acceptance_policy.md) — non-binding criteria for Milestone 1; intended for later optional gating.

## Follow-up milestones (proposal)

- **Milestone 2** — Check in a **baseline** `results_export` summary JSON (or manifest + metric digest) from a known-good N150 run; add optional CLI flag to **exit non-zero** on holdout p95 regression vs baseline; add **stratified** reporting (e.g. by `traced_source` bucket); keep **off by default** for shared CI until stable.
- **Milestone 3** — Land a **first matmul change** (e.g. program config or factory tweak) with before/after protocol runs; require **holdout** p95 non-regression and **no new** timeouts/hangs per acceptance policy.
