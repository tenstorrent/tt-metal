# Sweep Framework

The sweep framework tests TTNN operations across large parameter spaces. It generates all permutations of input parameters, runs each one on device, and records pass/fail status, PCC accuracy, and optional performance/memory metrics.

## Quick Start

```bash
# 1. Install dependencies (if not already in your venv)
uv pip install -r tests/sweep_framework/requirements-sweeps.txt

# 2. Generate test vectors for one module
python tests/sweep_framework/sweeps_parameter_generator.py --module-name eltwise.unary.relu.relu

# 3. Run the tests
python tests/sweep_framework/sweeps_runner.py \
  --module-name eltwise.unary.relu.relu \
  --vector-source vectors_export \
  --result-dest results_export
```

Generated vectors go to `tests/sweep_framework/vectors_export/`. Results go to `tests/sweep_framework/results_export/`.

## Directory Layout

```
tests/sweep_framework/
├── sweeps/                             # Sweep test definitions (one .py per op)
│   ├── eltwise/                        # Element-wise ops (unary, binary, ternary + backwards)
│   ├── matmul/                         # Matmul variants (short, full, sparse, generality)
│   ├── data_movement/                  # Concat, permute, reshape, slice, etc.
│   ├── reduction/                      # Argmax, mean, topk, var, etc.
│   ├── ccl/                            # Collective communication ops
│   ├── conv2d/ conv_transpose2d/       # Convolution ops
│   ├── normalization/                  # Batch norm, softmax, etc.
│   ├── transformer/                    # Attention, rotary embedding, etc.
│   ├── model_traced/                   # Ops with parameters traced from real models
│   └── ...
├── framework/                          # Core framework internals
│   ├── constants.py                    # LEAD_MODELS, mesh suffix helpers
│   ├── compute_sweep_matrix.py         # CI matrix generation for GitHub Actions
│   ├── permutations.py                 # Cartesian product of parameters
│   ├── serialize.py                    # Vector serialization/deserialization
│   ├── statuses.py                     # TestStatus enum (PASS, FAIL_*, XFAIL, etc.)
│   ├── vector_source.py               # Load vectors from JSON files or DB
│   ├── result_destination.py           # Write results to JSON files or Superset
│   ├── device_fixtures.py             # Default device setup
│   ├── tt_smi_util.py                 # Device reset after hangs
│   └── ...
├── sweep_utils/                        # Shared helpers for sweep test files
│   ├── utils.py                        # General utilities
│   ├── sharding_utils.py              # Shard spec generation
│   ├── mesh_tensor_utils.py           # Multi-chip tensor helpers
│   ├── ccl_common.py                  # CCL test helpers
│   ├── conv2d_common.py               # Conv2d parameter helpers
│   └── ...
├── sweeps_parameter_generator.py       # CLI: generate test vectors
├── sweeps_runner.py                    # CLI: execute tests
├── load_ttnn_ops_data_v2.py            # CLI: load model-traced configs into DB
├── master_config_loader_v2.py          # Extract configs from master JSON
├── operation_parameter_extractors.py   # Op-specific parameter extraction registry
├── validate_sweep_pipeline.py          # End-to-end pipeline validation
├── sweep_categories.py                 # Op → category mapping
└── requirements-sweeps.txt             # Python dependencies
```

## Writing a Sweep Test

Each sweep test is a `.py` file under `tests/sweep_framework/sweeps/`. The file path determines the module name (e.g., `sweeps/eltwise/unary/relu/relu.py` → module name `eltwise.unary.relu.relu`).

A sweep test file has up to four components:

### 1. `parameters` (required)

A dict of named suites. Each suite maps parameter names to lists of values. All permutations within a suite are generated automatically.

```python
parameters = {
    "suite_1": {
        "batch_sizes": [(1,)],
        "height": [384, 1024],
        "width": [1024, 4096],
        "broadcast": [None, "h", "w", "hw"],
        "input_a_dtype": [ttnn.bfloat16],
        "input_b_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
}
```

**Rules:**
- Each suite must have **at most 10,000 permutations**. Split into multiple suites if needed.
- **All TTNN types must be top-level parameters.** Do not nest `ttnn.*` objects inside tuples or dicts — they will not serialize correctly. See [Correct vs Incorrect Nesting](#correct-vs-incorrect-nesting) below.

Suite names control how tests are grouped. Common patterns:
- `default` — a single general suite
- `nightly`, `weekly` — frequency-based suites for CI scheduling
- `xfail_*` — suites where failures are expected (triggers XFAIL/XPASS status tracking)

### 2. `run()` (required)

Called once per test vector. Receives all parameter values as keyword arguments plus `device`.

```python
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time

def run(
    batch_sizes,
    height,
    width,
    broadcast,
    input_a_dtype,
    input_b_dtype,
    input_a_layout,
    input_b_layout,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    # 1. Create torch reference
    torch_input_a = torch.randn(*batch_sizes, height, width)
    torch_output = torch.add(torch_input_a, torch_input_a)

    # 2. Run on device
    input_tensor = ttnn.from_torch(torch_input_a, dtype=input_a_dtype, layout=input_a_layout,
                                    device=device, memory_config=input_a_memory_config)

    start_time = start_measuring_time()
    output = ttnn.add(input_tensor, input_tensor, memory_config=output_memory_config)
    e2e_perf = stop_measuring_time(start_time)

    output = ttnn.to_torch(output)

    # 3. Return [check_result, e2e_perf]
    return [check_with_pcc(torch_output, output, 0.999), e2e_perf]
```

**Return format** — one of:
- `(pass: bool, message: Optional[str])` — e.g., `(True, "0.999")`
- `[(pass, message), e2e_perf_ns]` — includes end-to-end performance in nanoseconds

### 3. `invalidate_vector()` (optional)

Pre-filters invalid parameter combinations before execution.

```python
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["broadcast"] in {"w", "hw"} and test_vector["input_b_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Broadcasting along width is not supported for row major layout"
    return False, None
```

Invalidated vectors are skipped (status `NOT_RUN`) with the reason recorded.

**Restriction:** This function runs on CPU only — no device code or TTNN device calls.

### 4. `mesh_device_fixture()` (optional)

Override the default single-chip device with a custom device configuration (e.g., multi-chip mesh).

```python
def mesh_device_fixture():
    assert ttnn.get_num_devices() >= 8, "Not T3000!"
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
    yield (mesh_device, "T3000 Mesh")
    ttnn.close_mesh_device(mesh_device)
    del mesh_device
```

The yielded tuple is `(device_object, label_string)`. The `device_object` is passed to `run()` as the `device` argument.

### 5. `TIMEOUT` (optional)

Override the default 30-second per-test timeout:

```python
TIMEOUT = 60  # seconds
```

## Correct vs Incorrect Nesting

TTNN types (`ttnn.CoreGrid`, `ttnn.ShardStrategy`, etc.) must be top-level parameters, not nested inside tuples or dicts.

**Incorrect** — TTNN types buried in tuples:
```python
parameters = {
    "default": {
        "matmul_specs": [
            ((2, 3), (1600, 224, 896), False,
             dict(core_grid=ttnn.CoreGrid(y=5, x=7), strategy=ttnn.ShardStrategy.BLOCK), None),
        ],
    }
}
```

**Correct** — TTNN types as separate top-level keys, split into meaningful suites:
```python
parameters = {
    "mcast_2d": {
        "batch_sizes": [(2, 3)],
        "input_shapes": [(1600, 224, 896)],
        "batch_matrix_multiply": [False],
        "input_a_sharded_core_grid": [ttnn.CoreGrid(y=5, x=7)],
        "input_a_sharded_strategy": [ttnn.ShardStrategy.BLOCK],
        "input_b_sharded_memory_config_specs": [None],
    },
}

# Append shared parameters to all suites
general = {
    "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "input_b_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "output_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "input_layout": [ttnn.TILE_LAYOUT],
}
for suite in parameters.values():
    suite.update(general)
```

You can also use generator functions to produce parameter lists programmatically — just pass the resulting list into the suite dict.

## Vector Generation

```bash
# Generate vectors for all modules
python tests/sweep_framework/sweeps_parameter_generator.py

# Generate vectors for one module
python tests/sweep_framework/sweeps_parameter_generator.py --module-name matmul.short.matmul

# Generate only model-traced vectors (all or lead models)
python tests/sweep_framework/sweeps_parameter_generator.py --model-traced all
python tests/sweep_framework/sweeps_parameter_generator.py --model-traced lead

# Custom tag (default: your username)
python tests/sweep_framework/sweeps_parameter_generator.py --module-name matmul.short.matmul --tag my-experiment

# Randomize vector order with a seed
python tests/sweep_framework/sweeps_parameter_generator.py --module-name matmul.short.matmul --randomize 42
```

**Options:**

| Flag | Description |
|------|-------------|
| `--module-name <name>` | Module to generate (omit for all) |
| `--tag <tag>` | Tag for separating vector sets (default: `$USER`) |
| `--randomize <seed>` | Shuffle vector order reproducibly |
| `--skip-modules <a,b>` | Comma-separated modules to skip |
| `--model-traced [all\|lead]` | Generate only model-traced ops |
| `--suite-name <name>` | Generate a specific suite only |
| `--mesh-shape <RxC>` | Filter to a specific mesh shape (e.g., `2x4`) |

Output goes to `tests/sweep_framework/vectors_export/`. For multi-chip ops, vectors are grouped by mesh shape into separate files (e.g., `module__mesh_2x4.json`) for CI runner routing.

## Test Runner

```bash
# Run a specific module
python tests/sweep_framework/sweeps_runner.py \
  --module-name eltwise.unary.relu.relu \
  --vector-source vectors_export \
  --result-dest results_export

# Run a specific suite
python tests/sweep_framework/sweeps_runner.py \
  --module-name eltwise.unary.relu.relu \
  --suite-name suite_1 \
  --vector-source vectors_export \
  --result-dest results_export

# Run a single vector by ID (hang detection disabled for debugging)
python tests/sweep_framework/sweeps_runner.py \
  --module-name eltwise.unary.relu.relu \
  --vector-id abc123 \
  --vector-source vectors_export \
  --result-dest results_export

# Dry run — see what would execute without running
python tests/sweep_framework/sweeps_runner.py \
  --module-name eltwise.unary.relu.relu \
  --vector-source vectors_export \
  --result-dest results_export \
  --dry-run --summary

# Run from an arbitrary JSON file
python tests/sweep_framework/sweeps_runner.py \
  --module-name eltwise.unary.relu.relu \
  --vector-source file \
  --file-path /path/to/vectors.json \
  --result-dest results_export
```

**Key options:**

| Flag | Description |
|------|-------------|
| `--module-name <name>` | Module(s) to run (comma-separated, or omit for all) |
| `--suite-name <name>` | Run only this suite |
| `--vector-id <hash>` | Run a single vector (disables hang detection for debugging) |
| `--vector-source` | `vectors_export` (default) or `file` |
| `--file-path <path>` | JSON path (required when `--vector-source file`) |
| `--result-dest` | `results_export` (default) or `superset` |
| `--tag <tag>` | Reserved; tag field is stored in vectors but not used for filtering yet |
| `--skip-modules <a,b>` | Skip these modules when running all |
| `--perf` | Measure end-to-end performance |
| `--device-perf` | Measure device-level performance (requires profiler build) |
| `--measure-memory` | Capture per-core L1 memory usage via graph trace |
| `--watcher` | Enable watcher for memory/exception monitoring |
| `--skip-on-timeout` | Abort remaining suite tests after a timeout |
| `--keep-invalid` | Include invalid vectors as NOT_RUN (default: exclude them) |
| `--main-proc-verbose` | Run in main process (not subprocess) for easier debugging |
| `--dry-run` | Plan without executing |
| `--summary` | Print execution summary |

### Test Statuses

| Status | Meaning |
|--------|---------|
| `PASS` | Test met expected PCC or other criteria |
| `FAIL_ASSERT_EXCEPTION` | Assertion failure, bad PCC, or unhandled exception |
| `FAIL_CRASH_HANG` | Test timed out (assumed hang) |
| `FAIL_L1_OUT_OF_MEM` | L1 memory allocation failure |
| `FAIL_WATCHER` | Watcher-raised exception (requires `--watcher`) |
| `FAIL_UNSUPPORTED_DEVICE_PERF` | Device perf requested (`--device-perf`) but no profiler data available |
| `NOT_RUN` | Skipped due to `invalidate_vector` |
| `XFAIL` | Expected failure (suite name starts with `xfail`) |
| `XPASS` | Unexpected pass in an xfail suite |

### Hang Detection and Recovery

- Default timeout: 30 seconds per test (override with `TIMEOUT` in test file).
- On timeout, the test subprocess is killed and `tt-smi` resets the device before the next test.
- Set `TT_SMI_RESET_COMMAND` env var for your system (e.g., `TT_SMI_RESET_COMMAND="tt-smi -tr 0"`).
- When running a single vector (`--vector-id`), hang detection is disabled so you can attach debuggers.

### Memory Profiling

Use `--measure-memory` to capture per-core and device-level L1 memory usage without execution overhead (uses graph trace in `NO_DISPATCH` mode).

```bash
python tests/sweep_framework/sweeps_runner.py \
  --module-name matmul.short.matmul \
  --vector-source vectors_export \
  --result-dest results_export \
  --measure-memory
```

Captured metrics:

| Metric | Description |
|--------|-------------|
| `peak_l1_memory_per_core_bytes` | Peak total (CB + L1) per core |
| `peak_cb_per_core_bytes` | Peak circular buffer per core |
| `peak_l1_buffers_per_core_bytes` | Peak L1 buffer per core |
| `num_cores` | Number of cores used |
| `peak_l1_memory_aggregate_bytes` | Worst-case if all cores peak simultaneously (per-core × num_cores) |
| `peak_l1_memory_device_bytes` | Actual observed peak across device |

If `aggregate ≈ device`, cores peak together (parallel execution). If `aggregate >> device`, execution is sequential (only a few cores active at a time).

## Model-Traced Sweeps

Sweep tests can use parameters traced from real model executions rather than hand-written parameter lists. These live under `sweeps/model_traced/`.

Model-traced sweeps pull configurations from a master JSON file (produced by `model_tracer/`) via `master_config_loader_v2.py`. The loader extracts op-specific parameters using registered extractors in `operation_parameter_extractors.py`.

### Lead Models

Lead models are prioritized models whose traced configurations get dedicated CI treatment, including automatic routing to multi-chip runners. They are defined in `model_tracer/sweep_manifest.yaml` (with a fallback in `framework/constants.py`).

Generate vectors for lead models only:

```bash
python tests/sweep_framework/sweeps_parameter_generator.py --model-traced lead --tag ci-main
```

### Multi-Chip Runner Assignment

For lead model runs, vectors are grouped by `mesh_device_shape` and routed to appropriate hardware in CI:

| Mesh Shape | Runner |
|------------|--------|
| `1x1` | N150 (single-chip) |
| `1x2`, `1x4`, `2x4` | Galaxy (multi-chip) |
| `4x8`, `8x4` | Galaxy topology-6u (32-chip) |

The mapping is defined in `framework/compute_sweep_matrix.py`.

## CI Execution

Sweeps run automatically via the [ttnn - run sweeps](https://github.com/tenstorrent/tt-metal/actions/workflows/ttnn-run-sweeps.yaml) workflow.

| Run Type | Schedule | Suite | Description |
|----------|----------|-------|-------------|
| **Nightly** | Mon, Tue, Thu, Fri @ 4:30 AM UTC | `nightly` | Standard parameter sweeps |
| **Comprehensive** | Wed, Sat @ 4:00 AM UTC | All suites | Exhaustive testing |
| **Model Traced** | Daily @ 4:00 AM UTC | `model_traced` | Configs from real model traces |
| **Lead Models** | Daily @ 3:00 AM UTC | `model_traced` | Lead models with multi-chip routing |

Before merging new or modified sweep tests, verify locally and also run the [ttnn - run sweeps](https://github.com/tenstorrent/tt-metal/actions/workflows/ttnn-run-sweeps.yaml) workflow against your branch.

## Pipeline Validation

`validate_sweep_pipeline.py` runs the full generate → execute → trace → split pipeline for a module:

```bash
python tests/sweep_framework/validate_sweep_pipeline.py \
  --model-trace /path/to/model_trace.json \
  --module-name model_traced.all_gather_async_model_traced \
  --suite model_traced \
  --mesh-shape 4x8
```

## FAQ / Troubleshooting

**`ModuleNotFoundError: No module named 'beautifultable'`**
Install sweep dependencies: `uv pip install -r tests/sweep_framework/requirements-sweeps.txt`

**TTNN types not serializing correctly**
TTNN nanobind classes need the `tt_nanobind_class` wrapper for serialization support. Enum types work without it. See the template in `tt_lib_bindings_tensor.cpp`.

**`invalidate_vector` or `parameters` code fails with device errors**
These run on CPU only — no device code or TTNN device calls allowed. To filter by device architecture, use a `mesh_device_fixture` instead.

**Tests hang or timeout unexpectedly**
Set `TT_SMI_RESET_COMMAND` for your system. Debug individual vectors with `--vector-id` (disables timeout) and `--main-proc-verbose` (runs in-process for full stack traces).
