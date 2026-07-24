# Sweep Matrix Computation

This document describes `compute_sweep_matrix.py`, which generates the GitHub Actions matrix for sweep test execution.

## Overview

The script reads generated vector artifacts and emits a matrix that maps test modules to hardware runners. All routing and runner policy lives in `matrix_runner_config.py`; this script orchestrates the computation.

## Run Types

Run type is determined by `SWEEP_NAME` (authoritative in CI) or, when absent, by matching `GITHUB_EVENT_SCHEDULE` against `SCHEDULE_TYPES` in `matrix_runner_config.py`. The default is `nightly`.

| Run Type        | `SWEEP_NAME` value              | Cron schedule     | Suite          |
| --------------- | ------------------------------- | ----------------- | -------------- |
| Lead Models     | `ALL SWEEPS (Lead Models)`      | `0 2 * * *`       | `model_traced` |
| Model Traced    | `ALL SWEEPS (Model Traced)`     | `0 3 * * *`       | `model_traced` |
| Comprehensive   | `ALL SWEEPS (Comprehensive)`    | *(dispatch only)* | All suites     |
| Nightly         | `ALL SWEEPS (Nightly)`          | *(default)*       | `nightly`      |

**Batch size**: 10 modules per batch for most run types; 3 for comprehensive or when device perf measurement is enabled.

### Run Type Strategies

- **Lead Models**: In CI (`--group-by hw`), hardware-grouped files are routed via `get_lead_models_test_group_name_for_hardware_group()` into two lanes: `lead-models-single-chip` and `lead-models-galaxy`. With `--group-by mesh` (local default), routing uses `LEAD_MODELS_MESH_TEST_GROUPS` instead.
- **Model Traced**: In CI (`--group-by hw`), hardware-grouped files are routed via `get_test_group_name_for_hardware_group()` across standard hardware lanes. `vector_grouping_mode` from the manifest tells `compute_sweep_matrix.py` which path to take.
- **Nightly / Comprehensive**: All modules on `wormhole-n150-sweeps`. CCL modules are split to `n300-llmbox-ccl` with suite `generality_suite_fabric_1d`.

## Routing Policy

All runner assignment logic lives in `matrix_runner_config.py`:

- `TEST_GROUPS` — logical lane names and their associated runner profiles
- `RUNNER_PROFILES` — physical runner properties (`runs_on`, `runner_label`, `arch`, `tt_smi_cmd`)
- `LEAD_MODELS_MESH_TEST_GROUPS` / `MODEL_TRACED_MESH_TEST_GROUPS` — mesh shape → test group mappings
- `LEAD_MODELS_BATCH_POLICY` — per-lane batch overrides for lead models
- `HW_GROUP_MATRIX_KEYS` — which test groups belong to each per-hardware output bucket

To change runner assignment for any run type, update `matrix_runner_config.py`.

## Artifacts: `generation_manifest.json`

When present in `VECTORS_DIR`, `generation_manifest.json` is the authoritative list of vector files. It also records `vector_grouping_mode` (`mesh` or `hw`), which determines how `compute_model_traced_matrix` routes modules and how `vector_source.py` filters vectors at runtime.

If the manifest is absent, the script falls back to scanning `*.json` files in `VECTORS_DIR`.

## Outputs

The script prints GitHub Actions output lines to stdout:

```
matrix={"module":[...],"batches":[...],"ccl_batches":[...],"include":[...]}
n150-matrix={"include":[...]}
n300-matrix={"include":[...]}
p150b-matrix={"include":[...]}
t3k-matrix={"include":[...]}
galaxy-matrix={"include":[...]}
```

Each `include` entry contains: `test_group_name`, `arch`, `runs_on`, `runner_label`, `tt_smi_cmd`, `module_selector`, `suite_name`, `batch_display`.

The workflow reads the per-hardware outputs (`n150-matrix`, etc.) to fan work out to the correct runner pools.

## Environment Variables

| Variable               | Description                                  | Example                        |
| ---------------------- | -------------------------------------------- | ------------------------------ |
| `SWEEP_NAME`           | Run type (takes precedence over schedule)    | `ALL SWEEPS (Lead Models)`     |
| `GITHUB_EVENT_SCHEDULE`| Cron expression (fallback run type lookup)   | `0 3 * * *`                    |
| `GITHUB_EVENT_NAME`    | GitHub event type                            | `schedule`                     |
| `MEASURE_DEVICE_PERF`  | Reduces batch size to 3                      | `true`                         |
| `VECTORS_DIR`          | Path to vector JSON files                    | `/tmp/vectors`                 |

## Usage

### In CI

The workflow sets `SWEEP_NAME` explicitly via `resolve-inputs`, so schedule-based fallback is only relevant for manually triggered or locally simulated runs.

### Local Testing

```bash
export VECTORS_DIR="tests/sweep_framework/vectors_export"

# Simulate a lead models run
export SWEEP_NAME="ALL SWEEPS (Lead Models)"
python3 tests/sweep_framework/framework/compute_sweep_matrix.py | python3 -m json.tool

# Simulate a nightly run
export SWEEP_NAME="ALL SWEEPS (Nightly)"
python3 tests/sweep_framework/framework/compute_sweep_matrix.py | python3 -m json.tool
```

## Module ownership, categories, and time budgets

Every matrix entry is tagged with a functional **category**, an **owner** (Slack
member id), and a **team** so a failing sweep job is attributable to the team
that owns it (consumed by `ttnn-sweeps-slack-notify.yaml`). This policy lives in
`matrix_runner_config.py` and is produced natively by the matrix builder — there
is no separate static category list to keep in sync.

- **Category** is the leading segment of the module stem, i.e. the top-level
  directory under `tests/sweep_framework/sweeps/` (`eltwise.unary.relu` →
  `eltwise`). Ownership is `SWEEP_CATEGORY_OWNERS`; unlisted categories fall back
  to `DEFAULT_CATEGORY_METADATA`.
- **Attributable jobs**: `compute_standard_matrix` groups regular modules by
  category *before* batching, so a single job never straddles two owners. Mixed
  batches (possible on the mesh/hardware-routed lead-models / model-traced paths)
  fall back to the default owner.
- **Time budgets** (opt-in): set `CATEGORY_TIME_BUDGET_SECONDS[<category>]` and
  that category is packed into batches whose estimated runtime stays under the
  budget (`_budget_pack`), instead of a fixed module count. Per-module estimates
  come from `MODULE_RUNTIME_ESTIMATES` (seed from CI history), defaulting to
  `DEFAULT_MODULE_RUNTIME_SECONDS`. Categories with no budget are unchanged.

## Related Files

- `.github/workflows/ttnn-run-sweeps.yaml` — workflow that calls this script
- `tests/sweep_framework/framework/matrix_runner_config.py` — all routing and runner policy
- `tests/sweep_framework/sweeps_parameter_generator.py` — generates vector files and manifest
- `tests/sweep_framework/framework/vector_source.py` — loads and filters vectors at runtime
