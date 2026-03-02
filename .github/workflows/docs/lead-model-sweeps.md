# Lead Model Sweep Runs in TTNN Workflow

This document describes how lead model sweep runs are executed in the `ttnn-run-sweeps.yaml` GitHub Actions workflow.

## Overview

Lead model sweeps are a specialized category of sweeps that test TTNN operations traced from production models. They run daily and use a PostgreSQL database as the source of truth for test configurations.

## Triggering Lead Model Sweeps

Lead model sweeps can be triggered in two ways:

### 1. Scheduled (Automatic)
- **Cron Schedule**: `0 2 * * *` (2:00 AM UTC daily)
- **Run Name**: `ttnn - run sweeps lead models (scheduled)`

### 2. Manual Dispatch
- **Selection**: Choose `ALL SWEEPS (Lead Models)` from the `sweep_name` dropdown
- **Run Name**: `ttnn - run sweeps lead models`

## Execution Flow

```
┌─────────────────────────────────────┐
│      build-artifact                 │
│  (builds with tracy if scheduled)   │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│   generate-master-json-from-db      │
│  (reconstructs JSON from database)  │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│       ttnn-generate-sweeps          │
│   (generates sweep vectors with     │
│    --model-traced lead flag)        │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│    ttnn-compute-module-matrix       │
│  (creates parallel execution plan)  │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│     ttnn-run-sweeps-parallel        │
│   (executes sweeps across runners)  │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│    ttnn-homogenize-run-result       │
│  (aggregates results + pushes to    │
│   database for lead models)         │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│       ttnn-upload-results           │
│     (uploads to SFTP server)        │
└─────────────────────────────────────┘
```

## Job Details

### 1. build-artifact
- Builds the TTNN wheel and artifacts
- Enables Tracy profiler for scheduled runs (device performance measurement)
- Configuration: `tracy: true` for scheduled runs

### 2. generate-master-json-from-db
**Condition**: Runs when `use_database == 'true'` OR schedule is `0 2 * * *` (lead models)

Generates two JSON files from the PostgreSQL database:
- **V1 Format** (legacy): `ttnn_operations_master_reconstructed.json` from `ttnn_ops` schema
- **V2 Format** (new): `ttnn_operations_master_v2_reconstructed.json` from `ttnn_ops_v2` schema (per-tensor placement)

```bash
# V1 generation
python tests/sweep_framework/load_ttnn_ops_data.py reconstruct \
  model_tracer/traced_operations/ttnn_operations_master_reconstructed.json \
  ttnn_ops

# V2 generation
python tests/sweep_framework/load_ttnn_ops_data_v2.py reconstruct \
  model_tracer/traced_operations/ttnn_operations_master_v2_reconstructed.json \
  ttnn_ops_v2
```

### 3. ttnn-generate-sweeps
**For Lead Models**: Generates vectors using `--model-traced lead` flag

```bash
python3 tests/sweep_framework/sweeps_parameter_generator.py --model-traced lead --tag ci-main
```

- Downloads the database-generated JSON from previous job
- Runs on the specified runner (default: N150)
- Uploads vectors to `sweeps-vectors-all` artifact

### 4. ttnn-compute-module-matrix
Creates the execution matrix for parallel sweep runs by:
1. Downloading generated vector JSON files
2. Chunking into batches (3 for perf runs, 10 for nightly)
3. Determining suite name (`lead_models` for this run type)
4. Generating runner configurations

### 5. ttnn-run-sweeps-parallel
Executes sweeps in parallel across multiple runners.

**Lead Model Specific Settings**:
- `LEAD_MODELS_RUN=1` environment variable is set when test group name starts with `lead-models`
- Database URL is provided via `TTNN_OPS_DATABASE_URL` secret

**Default Flags for Scheduled Runs**:
- `--skip-on-timeout`: Skip remaining tests after first timeout
- `--device-perf`: Measure device performance
- `--measure-memory`: Capture peak L1 memory usage

```bash
python3 tests/sweep_framework/sweeps_runner.py \
  --module-name "$MODULE_SELECTOR" \
  --vector-source vectors_export \
  --result-dest results_export \
  --tag ci-main \
  --summary \
  --skip-on-timeout \
  --device-perf \
  --measure-memory \
  --suite-name "$SUITE_NAME"
```

### 6. ttnn-homogenize-run-result
Aggregates results from all parallel runners.

**Lead Model Specific**: Pushes results to database

```bash
# Homogenize results
python3 tests/sweep_framework/run_collective_update.py --run-type "lead_models"

# Push to database (lead models only)
python /work/.github/actions/sweep-run-analysis/scripts/push_sweep_results.py \
  /work/tests/sweep_framework/results_export/ \
  "lead models"
```

Environment variables for database push:
- `TTNN_OPS_DATABASE_URL`: Database connection string
- `GITHUB_RUN_ID`: CI run identifier
- `ARCH_NAME`: Target architecture
- `GITHUB_SHA`: Commit hash
- `GITHUB_REF_NAME`: Branch name

### 7. ttnn-upload-results
Uploads final results to SFTP server for external consumption (Superset dashboards).

## Key Differences from Other Sweep Types

| Aspect | Lead Models | Nightly | Comprehensive | Model Traced |
|--------|-------------|---------|---------------|--------------|
| Schedule | 2:00 AM daily | 4:30 AM (Sun-Fri) | 4:00 AM (Wed/Sat) | 3:00 AM daily |
| Database Source | Yes | No | No | Yes |
| Database Push | Yes | No | No | No |
| CCL Modules | No | Yes | Yes | No |
| Vector Generation | `--model-traced lead` | Default | Default | `--model-traced all` |

## Configuration Options (Manual Dispatch)

When running lead model sweeps manually, the following options are available:

| Option | Default | Description |
|--------|---------|-------------|
| `skip_on_timeout` | true | Skip remaining tests after first timeout |
| `upload_results` | true | Upload results to Superset |
| `measure_device_perf` | true | Measure device performance with Tracy |
| `measure_e2e_perf` | false | Measure end-to-end performance |
| `measure_memory` | false | Capture peak L1 memory usage |
| `arch` | wormhole_b0 | Target architecture |
| `runner-label` | N150 | Runner type |
| `log-level` | INFO | Log verbosity |
| `use_database` | false (auto-enabled for lead models) | Load configs from database |

## Secrets Required

- `TTNN_OPS_LEAD_MODELS_DATABASE_URL`: PostgreSQL connection string for lead models database
- `SFTP_CICD_WRITER_KEY`: SSH key for SFTP upload

---

## Real-World Example: Run #22559650190

This section documents an actual lead model sweep run to illustrate how the workflow executes in practice.

**Run Details:**
- **Run URL**: https://github.com/tenstorrent/tt-metal/actions/runs/22559650190
- **Trigger**: Manual dispatch (`workflow_dispatch`)
- **Display Title**: `ttnn - run sweeps lead models`
- **Branch**: `Aswinmcw/sweeps_v2_p1`
- **Overall Duration**: ~12 hours (03:04 UTC - 14:56 UTC on 2026-03-02)
- **Conclusion**: Success

### Mesh-Based Runner Routing

Lead model sweeps use **mesh shape suffixes** in vector filenames to route tests to appropriate hardware:

```
Vector File Naming Convention:
  model_traced.add__mesh_1x1.json   → Routes to N150 (single-chip)
  model_traced.add__mesh_2x4.json   → Routes to Galaxy (multi-chip)
  model_traced.add__mesh_8x4.json   → Routes to Galaxy (multi-chip)
```

The `compute_sweep_matrix.py` script parses these suffixes and assigns runners:

| Mesh Shape | Runner Assignment | Hardware |
|------------|-------------------|----------|
| `1x1` | `lead-models-single-chip` | N150 (single Wormhole chip) |
| `1x2`, `1x4`, `1x8`, `2x4`, `4x8`, `8x4`, `2x16`, `16x2` | `lead-models-galaxy` | topology-6u (32-chip Galaxy) |

### Jobs Executed in This Run

This run spawned **34 parallel jobs** across two runner types:

| Job Category | Runner Type | Job Count | Description |
|-------------|-------------|-----------|-------------|
| `lead-models-single-chip` | N150 | 31 jobs | Single-chip tests (1x1 mesh shape) |
| `lead-models-galaxy` | topology-6u (Galaxy) | 3 jobs | Multi-chip tests (1x2, 1x4, 1x8, 2x4, 4x8, 8x4, 2x16, 16x2 mesh shapes) |

### Example Job 1: Single-Chip Sweep (N150) - No Vectors Executed

**Job Name**: `Run sweeps in parallel (lead-models-single-chip, 1x1:model_traced.gelu_model_traced,model_traced.global_avg_pool2d_model_traced,model_traced.group_norm_model_traced)`

- **Runner**: N150 (`tt-ubuntu-2204-n150-stable`)
- **Mesh Shape**: 1x1 (single chip)
- **Modules Assigned**: 3 modules
- **Vectors Executed**: **0** (no `__mesh_1x1` vector files existed)
- **Duration**: ~3-4 minutes (job setup/teardown only)

These jobs were created expecting single-chip vectors, but the lead models database only contains multi-chip configurations from DeepSeek V3 Galaxy traces.

### Example Job 2: Multi-Chip Sweep (Galaxy)

**Job URL**: https://github.com/tenstorrent/tt-metal/actions/runs/22559650190/job/65344906969

**Job Name**: `Run sweeps in parallel (lead-models-galaxy, 1x2+1x4+1x8+2x4+4x8+8x4+2x16+16x2:model_traced.add_model_traced,...)`

- **Runner**: topology-6u Galaxy (`g04glx03`)
- **Mesh Shapes**: 1x2, 1x4, 1x8, 2x4, 4x8, 8x4, 2x16, 16x2 (all multi-chip configurations)
- **Duration**: ~17 minutes

These jobs test operations that require multiple chips working together. Each module is tested across **all 8 mesh configurations** in the same job. The vector files have suffixes like `__mesh_2x4`, `__mesh_8x4`, etc.

**Modules Tested** (11 modules, each across 8 mesh shapes = 88 test configurations):
- `model_traced.add_model_traced`
- `model_traced.all_gather_async_model_traced`
- `model_traced.concat_model_traced`
- `model_traced.div_model_traced`
- `model_traced.embedding_model_traced`
- `model_traced.exp_model_traced`
- `model_traced.interleaved_to_sharded_model_traced`
- `model_traced.linear_model_traced`
- `model_traced.multiply_model_traced`
- `model_traced.neg_model_traced`
- `model_traced.ones_like_model_traced`

### Step-by-Step Execution Breakdown

| Step | Name | Duration | Description |
|------|------|----------|-------------|
| 1 | Set up job | 3s | Initializes GitHub Actions runner environment |
| 2 | Set up runner | 81s | Runs self-hosted runner setup hook, performs `tt-smi` device reset |
| 3 | Initialize containers | 2s | Starts Docker container with dev image |
| 4 | Checkout Repository | 15s | Clones repository with submodules |
| 5 | Setup Job | 13s | Downloads and extracts build artifacts (wheel, libraries) |
| 6 | Display parallel sweep configuration | <1s | Prints job configuration for debugging |
| 7 | Download sweep vectors | 1s | Downloads pre-generated test vectors from artifacts |
| 8 | Run ttnn sweeps (batch) | ~14.5 min | **Main execution** - runs sweep tests on hardware |
| 9 | Compute artifact suffix | <1s | Generates unique hash for result artifact naming |
| 10 | Upload sweep results | 4s | Uploads test results as GitHub artifact |

### Hardware Environment

The Galaxy runner (`g04glx03`) reported the following configuration:

```
Host Information:
- OS: Ubuntu 22.04.5 LTS
- Kernel: 6.8.0-101-generic
- Memory: 566.12 GB
- Driver: TT-KMD 2.7.0

Software Versions:
- tt_smi: 4.0.0
- pyluwen: 0.8.1
- tt_umd: 0.9.2

Device Information:
- Board Type: tt-galaxy-wh L
- DRAM Speed: 14G
- PCIe Speed: Gen4
- Firmware Bundle: 19.6.0.0
```

### Pre-Job Hook: Device Reset

Before each job, the runner executes a device reset hook:
```bash
/opt/tt_metal_infra/scripts/ci/wormhole_b0/reset.sh
```

This performs:
1. `tt-smi` device reset to ensure clean state
2. Disk space cleanup if needed
3. Device health verification via `tt-smi` JSON output

### Parallel Job Distribution

For this lead models run, the workflow distributed work across **34 total jobs**:

**Single-Chip Jobs (N150)**: 31 parallel jobs
- Runner: `tt-ubuntu-2204-n150-stable`
- Each job was assigned 3 model_traced modules
- Duration: ~3-4 minutes per job
- **Actual vectors executed: 0** (see [Known Issue](#known-issue-single-chip-jobs-execute-zero-vectors) below)

**Multi-Chip Jobs (Galaxy)**: 3 parallel jobs
- Runner: `topology-6u` (32-chip Galaxy system)
- Each job tests 10-11 model_traced modules across mesh configurations (4x8, 2x4)
- Duration: ~6-17 minutes per job depending on module complexity
- **All actual test execution happened here**

**Why were single-chip jobs created but had no vectors?** The matrix computation expected vectors for all mesh shapes, but the lead models database (traced from DeepSeek V3 on Galaxy) only contains multi-chip configurations. See the Known Issue section below for details.

### Result Aggregation

After all parallel jobs complete:
1. `ttnn-homogenize-run-result` downloads all result artifacts
2. Results are merged using `run_collective_update.py --run-type "lead_models"`
3. Combined results are pushed to the PostgreSQL database
4. Final `oprun_*.json` files are uploaded to SFTP for Superset dashboards

---

## Known Issue: Single-Chip Jobs Execute Zero Vectors

In this run, the 31 single-chip N150 jobs completed successfully but **executed 0 test vectors**.

### Root Cause

The lead models database contains traced operations from DeepSeek V3, which runs on multi-chip Galaxy systems. The vector generation only produced files with multi-chip mesh suffixes:

```
Generator output (sample):
- SWEEPS: Generated 9 test vectors for suite model_traced (mesh 4x8)
- SWEEPS: Generated 27 test vectors for suite model_traced (mesh 4x8)
- SWEEPS: Generated 1 test vectors for suite model_traced (mesh 2x4)
```

**No `1x1` mesh vectors were generated** because the source model traces don't contain single-chip configurations.

### Why Single-Chip Jobs Were Created

The `compute_sweep_matrix.py` script:
1. Parses vector filenames for mesh shapes (e.g., `model_traced.add__mesh_4x8.json` → shape `4x8`)
2. Routes modules to runners based on mesh shape
3. Routes modules **without** mesh suffixes to the default N150 runner

The matrix creation logic assumed modules would have vectors for all mesh shapes, but the lead models data only contains multi-chip configurations.

### Evidence from Logs

Single-chip job logs showed:
```
WARNING  | SWEEPS - No vectors found for module model_traced.concat_model_traced, suite model_traced
INFO     | SWEEPS - Total test cases (vectors) executed: 0
```

### Impact

- **31 N150 jobs**: Completed in ~3-4 minutes each, but tested nothing
- **3 Galaxy jobs**: Executed all actual vectors (~17 minutes each)
- **Wasted resources**: ~90 minutes of N150 runner time with no test coverage

### Potential Fixes

1. **Database fix**: Add single-chip (1x1) traced operations to the lead models database
2. **Matrix optimization**: Skip runner configurations that have no matching vectors
3. **Vector generation fix**: Generate vectors for all supported mesh shapes, not just those in the trace data

---

## Local vs CI Vector Generation Differences

When running the generator locally vs in CI, you may see different filename formats due to how mesh shapes are determined.

### How Mesh Suffixes Are Added

The generator extracts mesh shape from `traced_machine_info` in vectors via `get_mesh_shape_from_vector()`:

1. **V2 format (CI)**: Looks for explicit `mesh_device_shape` field (e.g., `[4, 8]`)
2. **Legacy format**: Falls back to inferring from `device_series` + `card_count`

The inference map is limited:
```python
_DEVICE_SERIES_MESH_MAP = {
    ("tt-galaxy-wh", 32): (4, 8),  # Only Galaxy with 32 cards
}
```

### Local Run Example

Running locally with a JSON file that has:
```json
"traced_machine_info": [
  {
    "board_type": "Wormhole",
    "device_series": "n300",
    "card_count": 1
  }
]
```

- No `mesh_device_shape` field exists
- `device_series: "n300"` with `card_count: 1` doesn't match the inference map
- `get_mesh_shape_from_vector()` returns `None`
- **Result**: File exported as `model_traced.reshape_model_traced.json` (no mesh suffix)

### CI Run Example

CI uses database-generated JSON (V2 format) with explicit mesh shapes:
```json
"traced_machine_info": {
  "mesh_device_shape": [4, 8],
  ...
}
```

- `mesh_device_shape: [4, 8]` is found
- **Result**: File exported as `model_traced.reshape__mesh_4x8.json`

### Impact on Test Execution

| Scenario | Vector Filename | Matrix Routing | Runner |
|----------|----------------|----------------|--------|
| Local (no mesh shape) | `module.json` | Routes to default (N150) | N150 |
| CI (mesh 4x8) | `module__mesh_4x8.json` | Routes to Galaxy | topology-6u |
| CI (mesh 1x1) | `module__mesh_1x1.json` | Routes to N150 | N150 |

This explains why local runs may successfully execute vectors on N150, while CI runs with the same module names find no vectors on N150 (because the CI vectors have mesh suffixes that route them elsewhere).

---

## Pipeline Complexity Analysis

This section documents areas of complexity, redundancy, and over-engineering in the model-trace to sweeps pipeline.

### 1. Duplicated Code: V1 vs V2 Format Handling

**Files with near-identical implementations:**

| V1 Version | V2 Version | Purpose |
|------------|------------|---------|
| `load_ttnn_ops_data.py` | `load_ttnn_ops_data_v2.py` | Database loader scripts |
| `master_config_loader.py` | `master_config_loader_v2.py` | Config loading from JSON |
| `ttnn_ops` schema | `ttnn_ops_v2` schema | Database schemas |

**Impact:**
- Bug fixes must be applied to both versions
- CI generates BOTH `ttnn_operations_master_reconstructed.json` AND `ttnn_operations_master_v2_reconstructed.json`
- Unclear which version is authoritative

**Recommendation:** Consolidate into a single loader with format detection, or deprecate V1.

### 2. Workflow Conditional Complexity

The workflow file contains:
- **17 conditional (`if:`) blocks**
- **99 references to `github.event.*`**
- **4 different schedule triggers** with different behavior each

**Example of nested conditions:**
```yaml
if: |
  ((github.event_name == 'workflow_dispatch' && github.event.inputs.sweep_name == 'ALL SWEEPS (Lead Models)') ||
   (github.event_name == 'schedule' && github.event.schedule == '0 2 * * *')) &&
  matrix.generate_mode == 'all'
```

**Impact:**
- Hard to trace which code path executes for a given trigger
- Easy to introduce bugs when modifying conditions
- Testing all paths requires multiple workflow runs

**Recommendation:** Extract run-type detection into a single job that sets output variables, then use those outputs in subsequent jobs.

### 3. Generator Job Duplication

The `ttnn-generate-sweeps` job runs **two matrix entries** for every lead models run:

| Entry | generate_mode | Actual Work for Lead Models |
|-------|--------------|----------------------------|
| 1 | `all` | Runs `--model-traced lead` |
| 2 | `non_ccl` | **Does nothing** (condition not met) |

**Impact:**
- Wasted CI time (~3-4 minutes per empty job)
- Confusing job list in GitHub Actions UI
- Artifact upload step runs even when no vectors generated

**Recommendation:** Make matrix entries conditional on run type, or use a single generator with conditional steps.

### 4. Mesh Shape Inference Brittleness

The mesh shape inference in `sweeps_parameter_generator.py` has a hardcoded map:

```python
_DEVICE_SERIES_MESH_MAP = {
    ("tt-galaxy-wh", 32): (4, 8),  # Only this one case!
}
```

**Impact:**
- N300, N150, and other configurations return `None` (no mesh suffix)
- Local vs CI behavior differs based on JSON format
- Silent failures when mesh shape can't be determined

**Recommendation:**
- Expand the inference map to cover all device types
- Add logging when mesh shape cannot be determined
- Consider making mesh shape required in traced data

### 5. Suite Name Proliferation

Multiple suite names are used throughout the pipeline:

| Suite Name | Purpose | Where Used |
|------------|---------|------------|
| `model_traced` | Real model configs | Vectors, runner |
| `model_traced_sample` | Quick validation | Sweep modules |
| `lead_model_suite` | CCL operations | Generator |
| `generality_suite` | CCL generality | CCL sweeps |
| `generality_suite_fabric_1d` | 1D fabric CCL | CCL sweeps |
| `generality_suite_fabric_2d` | 2D fabric CCL | CCL sweeps |
| `nightly` | Nightly runs | Matrix computation |
| `comprehensive` | Full runs | Matrix computation |

**Impact:**
- Filtering logic spread across multiple files
- Easy to miss vectors when suite name doesn't match
- `--suite-name` flag behavior varies by context

**Recommendation:** Document suite name taxonomy; consider hierarchical naming.

### 6. Artifact Merge Complexity

Vector generation creates multiple artifacts that are later merged:

```
sweeps-vectors-all          → For lead models/model traced
sweeps-vectors-non_ccl      → For nightly/comprehensive (non-CCL)
sweeps-vectors-ccl_only     → For nightly/comprehensive (CCL only)
```

**Impact:**
- Merge step can silently overwrite files with same names
- Empty artifacts still uploaded (wasted storage)
- Download step must use `merge-multiple: true`

**Recommendation:** Use a single artifact per run type, or implement proper conflict detection.

### 7. Boilerplate in Sweep Modules

Each `model_traced` sweep module (~90 files) contains:
- Same imports (~15 lines)
- Same `mesh_device_fixture()` function (~25 lines)
- Same parameter loading pattern (~10 lines)
- Similar `run()` function structure

**Example boilerplate ratio:**
- `add_model_traced.py`: 200 lines total, ~80 lines boilerplate (40%)

**Recommendation:** Extract common fixtures and utilities into a base class or shared module.

### 8. Matrix Computation Creates Empty Jobs

The `compute_sweep_matrix.py` script:
1. Parses vector filenames for mesh shapes
2. Creates jobs for ALL configured mesh shapes
3. Doesn't verify vectors actually exist for each shape

**Result:** 31 N150 jobs with 0 vectors in the example run.

**Recommendation:** Add pre-flight check to skip runner configs with no matching vectors.

### 9. Environment Variable Sprawl

The pipeline uses numerous environment variables:

| Variable | Purpose |
|----------|---------|
| `LEAD_MODELS_RUN` | Flag for lead models |
| `TTNN_LEAD_MODELS_ONLY` | Filter flag |
| `TTNN_OPS_DATABASE_URL` | Database connection |
| `MESH_DEVICE_SHAPE` | Runtime mesh config |
| `TT_METAL_HOME` | Base directory |
| `GITHUB_EVENT_SCHEDULE` | Trigger detection |
| `MEASURE_DEVICE_PERF` | Performance flags |

**Impact:**
- Hard to know which variables are required vs optional
- Behavior changes silently based on environment
- Local vs CI environment differences cause issues

**Recommendation:** Consolidate into a configuration object; document required vs optional.

### 10. Implicit Dependencies Between Jobs

The workflow has implicit dependencies:
- `ttnn-run-sweeps-parallel` assumes vectors have specific filename patterns
- `compute_sweep_matrix.py` assumes vectors are in `/tmp/vectors`
- Results aggregation assumes specific JSON structure

**Impact:**
- Changes to one component can break others silently
- No schema validation between stages
- Debugging requires tracing through multiple jobs

**Recommendation:** Add explicit schema validation at job boundaries; use typed interfaces.

### Summary: Simplification Opportunities

| Area | Current State | Suggested Improvement |
|------|--------------|----------------------|
| V1/V2 duplication | 4+ duplicated files | Consolidate with format detection |
| Workflow conditions | 99 event references | Single run-type detection job |
| Generator matrix | 2 entries, 1 does nothing | Conditional matrix entries |
| Mesh inference | 1 hardcoded case | Comprehensive device map |
| Suite names | 8+ different names | Documented taxonomy |
| Empty jobs | 31 wasted jobs | Pre-flight vector check |
| Sweep boilerplate | 40% per file | Shared base class |

---

## Validating Lead Model Sweep Tests Against DeepSeek V3 Traces

To validate that lead model sweep tests are actually executing configurations from DeepSeek V3 traces, consider the following:

### Data Flow and Transformation Points

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. ORIGINAL TRACE (DeepSeek V3 execution on Galaxy)                         │
│    File: ttnn_operations_master.json                                        │
│    Format: operations → configurations → arguments + executions             │
│    Contains: source path, machine_info, config_hash, argument values        │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ load_ttnn_ops_data.py / load_ttnn_ops_data_v2.py
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. DATABASE (PostgreSQL)                                                    │
│    Schema: ttnn_ops or ttnn_ops_v2                                          │
│    Tables: ttnn_operation, ttnn_configuration, ttnn_mesh_config, etc.       │
│    Transformation: JSON flattened into relational tables                    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ reconstruct command
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. RECONSTRUCTED JSON (CI artifact)                                         │
│    File: ttnn_operations_master_v2_reconstructed.json                       │
│    Format: Similar to original but may have schema differences              │
│    Risk: Lossy transformation during DB round-trip                          │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ sweeps_parameter_generator.py
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. GENERATED VECTORS                                                        │
│    File: model_traced.add_model_traced__mesh_4x8.json                       │
│    Format: suite_name → config_hash → parameter dict                        │
│    Transformations:                                                         │
│      - Type conversions (string → enum)                                     │
│      - Memory config parsing                                                │
│      - Mesh suffix added to filename                                        │
│      - New fields: validity, status, timestamp, tag                         │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ sweeps_runner.py
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. TEST EXECUTION                                                           │
│    - Vector parameters passed to run() function                             │
│    - Mesh device created based on MESH_DEVICE_SHAPE env var                 │
│    - Actual TTNN operation executed                                         │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. RESULTS JSON                                                             │
│    File: oprun_*.json                                                       │
│    Contains: input_hash, success/failure, metrics                           │
│    Risk: Original config may not be fully preserved in results              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Validation Considerations

#### 1. Config Identity Matching

**Challenge:** The `config_hash` changes at multiple points:
- Original trace has `config_hash` from argument hashing
- Generated vectors have `input_hash` from vector hashing
- Results have `input_hash` from test execution

**Approach:**
```python
# Compare by semantic equivalence, not hash
original_config = {
    "shape": [256, 1, 32, 576],
    "dtype": "DataType.BFLOAT8_B",
    ...
}
vector_config = {
    "input_a_shape": "(256, 1, 32, 576)",  # String format!
    "input_a_dtype": "DataType.BFLOAT8_B",
    ...
}
```

#### 2. Parameter Name Transformations

The generator transforms parameter names:

| Original Trace | Generated Vector |
|----------------|------------------|
| `shape` | `input_a_shape` |
| `dtype` | `input_a_dtype` |
| `layout` | `input_a_layout` |
| `memory_config` | `input_a_memory_config` |
| `other_tensor` | `input_b_*` |

**Validation must account for these mappings.**

#### 3. Type Representation Differences

```python
# Original trace
{"dtype": {"type": "DataType", "repr": "DataType.BFLOAT8_B"}}

# Generated vector
{"input_a_dtype": "DataType.BFLOAT8_B"}

# Executed (Python object)
ttnn.bfloat8_b
```

#### 4. Mesh Shape Filtering

Vectors are filtered/grouped by mesh shape:
- Only vectors matching runner's mesh shape are executed
- A single original config may generate vectors for multiple mesh shapes
- Or may only exist for specific mesh shapes

**Validation should check:**
- Which mesh shapes exist in original trace
- Which mesh shapes have generated vectors
- Which mesh shapes actually executed

#### 5. Source Path Verification

The `traced_source` field should contain DeepSeek V3 path:

```python
# Expected for lead models
"traced_source": "models/demos/deepseek_v3/demo/demo.py"

# Or list format
"traced_source": ["models/demos/deepseek_v3/demo/demo.py", ...]
```

**Validation:** Every executed vector should have DeepSeek V3 in its source.

#### 6. Lead Models Filter

The `LEAD_MODELS` constant in `constants.py` controls filtering:

```python
LEAD_MODELS = [
    "deepseek_v3",
]
```

**Risk:** If this list is incorrect or source paths don't match, wrong configs get included/excluded.

### Validation Script Approach

```python
def validate_lead_model_coverage():
    """
    Compare original DeepSeek V3 traces against executed sweep tests.
    """

    # 1. Load original traces
    original_configs = load_original_trace("ttnn_operations_master.json")
    deepseek_configs = filter_by_source(original_configs, "deepseek_v3")

    # 2. Load generated vectors
    vectors = load_vectors("vectors_export/model_traced.*.json")

    # 3. Load execution results
    results = load_results("results_export/oprun_*.json")

    # 4. Build comparison
    for op_name, orig_configs in deepseek_configs.items():
        vector_configs = vectors.get(op_name, [])
        executed_configs = results.get(op_name, [])

        # Check coverage
        missing_in_vectors = find_missing(orig_configs, vector_configs)
        missing_in_execution = find_missing(vector_configs, executed_configs)

        # Check source attribution
        wrong_source = [v for v in executed_configs
                        if "deepseek_v3" not in v.get("traced_source", "")]
```

### Specific Things to Watch For

| Risk | Description | How to Detect |
|------|-------------|---------------|
| **Config Loss** | Original configs not making it to vectors | Compare config counts per operation |
| **Wrong Source** | Non-DeepSeek configs being tested | Check `traced_source` field |
| **Mesh Mismatch** | Configs generated for wrong mesh | Compare `traced_machine_info.mesh_device_shape` |
| **Type Corruption** | Parameter values changed during transformation | Semantic comparison of values |
| **Silent Filtering** | Configs silently dropped by filters | Log filtering decisions |
| **Hash Collision** | Different configs with same hash | Compare full parameter dictionaries |

### Recommended Validation Steps

1. **Count Comparison**
   ```
   Original DeepSeek traces: X operations, Y configs
   Generated vectors: X' operations, Y' configs
   Executed tests: X'' operations, Y'' configs

   Expected: X ≈ X' ≈ X'', Y ≈ Y' ≈ Y''
   ```

2. **Source Attribution Check**
   ```
   All executed configs should have traced_source containing "deepseek_v3"
   ```

3. **Parameter Sampling**
   ```
   For N random configs, verify parameter values match original trace
   ```

4. **Mesh Shape Audit**
   ```
   Original mesh shapes: {4x8: N1, 2x4: N2, ...}
   Generated mesh shapes: {4x8: M1, 2x4: M2, ...}
   Executed mesh shapes: {4x8: P1, 2x4: P2, ...}
   ```

5. **End-to-End Hash Tracking**
   ```
   Add logging to track config_hash through entire pipeline
   Original → DB → Reconstructed → Vector → Execution → Result
   ```
