# Find Operation Tests and Pipelines

Find which unit tests in `tests/ttnn/unit_tests/operations` test a given operation, and identify which CI/CD pipelines run those tests.

## Usage

**Quick Start:**
```bash
python .cursor/commands/ttnn/find_operation_tests.py <operation_name>
```

**With operation path:**
```bash
python .cursor/commands/ttnn/find_operation_tests.py <operation_name> --operation-path <path>
```

**Examples:**
```bash
python .cursor/commands/ttnn/find_operation_tests.py slice_write
python .cursor/commands/ttnn/find_operation_tests.py conv2d --operation-path conv/conv2d
python .cursor/commands/ttnn/find_operation_tests.py dropout
```

When you need to find tests for an operation and see which pipelines run them, use this command and provide:
- The operation name you're looking for (e.g., 'slice_write', 'conv2d', 'matmul', 'dropout')
- Optionally, the operation path if you know it (e.g., 'experimental/slice_write', 'data_movement/slice')

## Overview

This command helps you:
1. **Find test files** that test a specific operation by searching for the operation name in test files
2. **Identify test directories** that contain tests for the operation
3. **Map to CI/CD pipelines** that run those tests based on the test directory structure
4. **Show pipeline details** including workflow files, test groups, and execution commands

## How It Works

### Step 1: Search for Test Files

The command searches for test files that reference the operation:
- Searches in `tests/ttnn/unit_tests/operations/` directory
- Looks for files containing the operation name (case-insensitive)
- Checks both direct matches and common variations (e.g., `slice_write` matches `test_slice_write.py`)

### Step 2: Identify Test Directories

Based on the operation location and test file locations, identifies which test directory categories apply:
- `data_movement/` - for data movement operations
- `eltwise/` - for element-wise operations
- `conv/` - for convolution operations
- `matmul/` - for matrix multiplication operations
- `pool/` - for pooling operations
- `fused/` - for fused operations
- `reduce/` - for reduction operations
- `ccl/` - for collective communication operations
- `transformers/` - for transformer operations
- `rand/` - for random operations
- `debug/` - for debug operations
- `ssm/` - for state space model operations

### Step 3: Map to CI/CD Pipelines

Maps test directories to CI/CD pipelines based on workflow configurations:

**Post-Commit Pipelines** (`.github/workflows/ttnn-post-commit.yaml`):
- `ttnn eltwise group 1-8` - runs `tests/ttnn/unit_tests/operations/eltwise`
- `ttnn data_movement/ccl group 1-3` - runs `tests/ttnn/unit_tests/operations/data_movement`, `ccl`, `point_to_point`
- `ttnn conv group` - runs `tests/ttnn/unit_tests/operations/conv`
- `ttnn pool group` - runs `tests/ttnn/unit_tests/operations/pool`
- `ttnn matmul group` - runs `tests/ttnn/unit_tests/operations/matmul`
- `ttnn fused group` - runs `tests/ttnn/unit_tests/operations/fused`
- `ttnn transformers group` - runs `tests/ttnn/unit_tests/operations/transformers`
- `ttnn reduce and misc ops group` - runs `tests/ttnn/unit_tests/operations/reduce`, `rand`, `debug`, `ssm`

**Nightly Pipelines** (`.github/workflows/tt-metal-l2-nightly-impl.yaml`):
- Similar structure but with nightly-specific test paths in `tests/ttnn/nightly/unit_tests/operations/`

**Blackhole Multi-Card Tests** (`.github/workflows/blackhole-multi-card-unit-tests-impl.yaml`):
- Special CCL tests for Blackhole hardware

## Example Usage

### Example 1: Finding Tests for `slice_write`

**Operation:** `slice_write`
**Operation Path:** `experimental/slice_write` (in `ttnn/cpp/ttnn/operations/experimental/slice_write/`)

**Expected Results:**
- Test file: `tests/ttnn/unit_tests/operations/data_movement/test_slice_write.py`
- Test directory: `data_movement/`
- Pipelines:
  - **Post-Commit:** `ttnn data_movement/ccl group 1`, `group 2`, `group 3`
  - **Nightly:** `data_movement` group (if exists)

### Example 2: Finding Tests for `conv2d`

**Operation:** `conv2d`
**Operation Path:** `conv/conv2d` (in `ttnn/cpp/ttnn/operations/conv/conv2d/`)

**Expected Results:**
- Test files: `tests/ttnn/unit_tests/operations/conv/test_conv2d.py`
- Test directory: `conv/`
- Pipelines:
  - **Post-Commit:** `ttnn conv group`
  - **Nightly:** `conv` group

### Example 3: Finding Tests for `dropout`

**Operation:** `dropout`
**Operation Path:** `experimental/dropout` (in `ttnn/cpp/ttnn/operations/experimental/dropout/`)

**Expected Results:**
- Test file: `tests/ttnn/unit_tests/operations/data_movement/test_dropout.py`
- Test directory: `data_movement/`
- Pipelines:
  - **Post-Commit:** `ttnn data_movement/ccl group 1`, `group 2`, `group 3`

## Implementation Steps

### Step 1: Search for Operation in Codebase

1. Search for the operation name in `ttnn/cpp/ttnn/operations/` to find the operation path
2. Note the operation category (experimental, data_movement, eltwise, etc.)

### Step 2: Search for Test Files

1. Search in `tests/ttnn/unit_tests/operations/` for files containing the operation name
2. Look for patterns like:
   - `test_{operation_name}.py`
   - `test_{operation_name}_*.py`
   - Files that import or use the operation

### Step 3: Identify Test Directory

1. Determine which test directory contains the test files
2. Map operation categories to test directories:
   - `experimental/slice_write` → `data_movement/` (based on test location)
   - `experimental/dropout` → `data_movement/` (based on test location)
   - `conv/conv2d` → `conv/`
   - `eltwise/unary/` → `eltwise/`
   - `matmul/` → `matmul/`

### Step 4: Find Pipelines

1. Search `.github/workflows/` for workflows that run pytest on the identified test directory
2. Key workflows to check:
   - `.github/workflows/ttnn-post-commit.yaml` - main post-commit tests
   - `.github/workflows/tt-metal-l2-nightly-impl.yaml` - nightly tests
   - `.github/workflows/blackhole-multi-card-unit-tests-impl.yaml` - Blackhole multi-card tests
   - `.github/workflows/sanity-tests.yaml` - orchestrates sanity tests

### Step 5: Extract Pipeline Details

For each matching pipeline, extract:
- **Workflow file:** The YAML file that defines the pipeline
- **Test group name:** The name of the test group/job
- **Command:** The exact pytest command used
- **Trigger conditions:** When the pipeline runs (post-commit, nightly, manual, etc.)
- **Runner labels:** Which hardware runners are used

## Output Format

The command should output:

```
Operation: {operation_name}
Operation Path: {operation_path}

Test Files Found:
  - {test_file_path}
  - ...

Test Directory: {test_directory}

Pipelines That Run These Tests:

1. Pipeline: {pipeline_name}
   Workflow: {workflow_file}
   Test Group: {test_group_name}
   Command: {pytest_command}
   Triggers: {trigger_conditions}
   Runners: {runner_labels}

2. Pipeline: {pipeline_name}
   ...
```

## Common Operation-to-Test-Directory Mappings

| Operation Category | Test Directory | Notes |
|-------------------|----------------|-------|
| `experimental/slice_write` | `data_movement/` | Tested alongside other data movement ops |
| `experimental/dropout` | `data_movement/` | Tested alongside other data movement ops |
| `data_movement/slice` | `data_movement/` | Direct mapping |
| `data_movement/concat` | `data_movement/` | Direct mapping |
| `eltwise/unary/` | `eltwise/` | Direct mapping |
| `eltwise/binary/` | `eltwise/` | Direct mapping |
| `eltwise/binary/bias_gelu` | `eltwise/` | Bias GELU is a fused binary element‑wise op; tests live in the eltwise suite |
| `conv/conv1d` | `conv/` | Direct mapping |
| `conv/conv2d` | `conv/` | Direct mapping |
| `conv/conv3d` | `conv/` | Direct mapping |
| `matmul/` | `matmul/` | Direct mapping |
| `pool/` | `pool/` | Direct mapping |
| `fused/` | `fused/` | Direct mapping |
| `reduction/` | `reduce/` | Note: operation uses "reduction", tests use "reduce" |
| `ccl/` | `ccl/` | Direct mapping |
| `transformer/` | `transformers/` | Note: operation uses "transformer", tests use "transformers" |
| `rand/` | `rand/` | Direct mapping |
| `debug/` | `debug/` | Direct mapping |

## Implementation Notes

- The script is deliberately tolerant of variations in naming and layout to surface relevant tests even when operation names differ slightly.
- It parses workflow YAML files in a lightweight manner, extracting `name:` and `cmd:` fields to correlate test groups with the directories they exercise.
- Special handling for `data_movement/ccl` groups ensures that tests in the `data_movement` directory are correctly associated with the multiple CCL test groups defined in the post‑commit workflow.

---

*This documentation was updated to include the `bias_gelu` operation mapping, reflecting recent changes to expose its approximation mode and ensure proper test coverage.*