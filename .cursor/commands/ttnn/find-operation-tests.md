# Find Operation Tests and Pipelines

Find which unit tests test a given operation, generate local testing commands, and get CI/CD configuration for running tests via APC.

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
python .cursor/commands/ttnn/find_operation_tests.py transpose
python .cursor/commands/ttnn/find_operation_tests.py slice_write
python .cursor/commands/ttnn/find_operation_tests.py conv2d --operation-path conv/conv2d
python .cursor/commands/ttnn/find_operation_tests.py dropout
```

## What This Command Provides

### 1. Test Files Discovery

Finds all test files related to the operation:
- **Primary Tests**: Direct test files for the operation (e.g., `test_transpose.py`)
- **Related Tests**: Tests that use the operation (e.g., `ttnn.transpose` calls)
- **Legacy Tests**: Old tt_eager tests that may still be relevant

### 2. Local Testing Commands

Generates ready-to-run pytest commands:
```bash
source python_env/bin/activate

# Primary tests:
pytest tests/ttnn/unit_tests/base_functionality/test_reshape_transpose.py -v

# Related tests:
pytest tests/ttnn/unit_tests/base_functionality/test_reshape.py -v
```

### 3. CI Pipeline Information

Shows which post-commit pipelines run the tests.

### 4. APC Configuration

Generates JSON configuration for running tests via APC (Automated Pre-Commit):

**Workflow URL:** https://github.com/tenstorrent/tt-metal/actions/workflows/apc-select-tests.yaml

**JSON Configuration (copy-paste ready):**
```json
{"sd-unit-tests":false,"fast-dispatch-unit-tests":false,"fabric-unit-tests":false,"cpp-unit-tests":false,"ttnn-unit-tests":true,"models-unit-tests":false,"tt-train-cpp-unit-tests":false,"run-profiler-regression":false,"t3000-apc-fast-tests":false,"test-ttnn-tutorials":false,"triage-tests":false}
```

## Example Output

```
Found operation path: data_movement/transpose

Searching for tests for operation: transpose

================================================================================
Operation: transpose
Operation Path: data_movement/transpose
================================================================================

Test Files Found:

  Primary Tests:
    - tests/ttnn/unit_tests/base_functionality/test_reshape_transpose.py

  Tests that use ttnn.transpose:
    - tests/ttnn/unit_tests/base_functionality/test_reshape.py
    - tests/ttnn/unit_tests/operations/conv/data_movement/test_fold_op.py

  Legacy Tests:
    - tests/tt_eager/python_api_testing/unit_testing/misc/test_transpose.py
    - tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_transpose.py

================================================================================
Local Testing Commands
================================================================================

# Activate environment and run tests
source python_env/bin/activate

# Primary tests:
pytest tests/ttnn/unit_tests/base_functionality/test_reshape_transpose.py -v

# Related tests (use ttnn.transpose):
pytest tests/ttnn/unit_tests/base_functionality/test_reshape.py -v

================================================================================
Pipelines That Run These Tests (Post-Commit)
================================================================================

1. Pipeline: ttnn data_movement/ccl group 1
   Workflow: .github/workflows/ttnn-post-commit.yaml
   Command: pytest -n auto --timeout 300 tests/ttnn/unit_tests/operations/data_movement ...
   Test Directory: data_movement/

================================================================================
CI Testing via APC (Automated Pre-Commit)
================================================================================

Workflow URL: https://github.com/tenstorrent/tt-metal/actions/workflows/apc-select-tests.yaml

JSON Configuration (copy-paste ready):
{"sd-unit-tests":false,"fast-dispatch-unit-tests":false,"fabric-unit-tests":false,"cpp-unit-tests":false,"ttnn-unit-tests":true,"models-unit-tests":false,"tt-train-cpp-unit-tests":false,"run-profiler-regression":false,"t3000-apc-fast-tests":false,"test-ttnn-tutorials":false,"triage-tests":false}
```

## Search Directories

The command searches in these directories:
- `tests/ttnn/unit_tests/operations/` - Main operation tests
- `tests/ttnn/unit_tests/base_functionality/` - Base functionality tests
- `tests/tt_eager/python_api_testing/unit_testing/` - Legacy unit tests
- `tests/tt_eager/python_api_testing/sweep_tests/pytests/` - Legacy sweep tests

## Test Directory Categories

| Test Directory | Pipeline | Notes |
|----------------|----------|-------|
| `data_movement/` | ttnn data_movement/ccl groups 1-3 | Includes transpose, slice, concat, etc. |
| `eltwise/` | ttnn eltwise groups 1-8 | Element-wise operations |
| `conv/` | ttnn conv group | Convolution operations |
| `matmul/` | ttnn matmul group | Matrix multiplication |
| `pool/` | ttnn pool group | Pooling operations |
| `fused/` | ttnn fused group | Fused operations |
| `reduce/` | ttnn reduce and misc ops | Reduction operations |
| `ccl/` | ttnn data_movement/ccl groups | Collective communication |
| `transformers/` | ttnn transformers group | Transformer operations |
| `base_functionality/` | ttnn base_functionality | Base functionality tests |

## APC Test Categories

The APC workflow supports these test categories:

| APC Option | Description |
|------------|-------------|
| `ttnn-unit-tests` | All TTNN unit tests |
| `cpp-unit-tests` | C++ unit tests |
| `models-unit-tests` | Model tests |
| `fast-dispatch-unit-tests` | Fast dispatch tests |
| `fabric-unit-tests` | Fabric tests |
| `sd-unit-tests` | SD tests |

For most operation testing, use `"ttnn-unit-tests": true`.

## Running APC Tests

1. Go to: https://github.com/tenstorrent/tt-metal/actions/workflows/apc-select-tests.yaml
2. Click "Run workflow"
3. Select your branch
4. Paste the JSON configuration
5. Click "Run workflow"

## Related Commands

- `migrate-device-operation.md` - Migrate device operations to TMP pattern
- `verify-device-operation-hash.md` - Verify device operation hash implementation

## Troubleshooting

### No Tests Found

If no tests are found:
1. Check if the operation name is spelled correctly
2. Search for variations (e.g., `slice_write` vs `slice` vs `write`)
3. Check if tests might be in a different directory structure
4. Look for integration tests that might test the operation indirectly

### Multiple Test Files Found

If multiple test files are found, all are relevant. The operation may have:
- Direct test files covering specific functionality
- Integration tests that use the operation
- Legacy tests from tt_eager
