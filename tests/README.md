# TT-Metal Unit and Sweep Tests

This README applies to:

1. **Sweep Test Framework** (`tests/sweep_framework/`) - For running multiple parameterized sweep tests across ops
2. **Unit Test Runner** (`tests/ttnn/unit_test_runner.py`) - For running multiple pytest-based metal ops unit tests at a time

## Sweep Test Framework

The sweep test framework runs parameterized tests across large parameter spaces to evaluate test coverage and performance along that coverage. The sweep test framework separates where vectors are loaded from (vector source) and where results are written to (result destination):

### Steps to run sweep tests

#### 1. First Generate Test Vectors

```bash
# Generate vectors for all sweep modules and export to Elasticsearch
python tests/sweep_framework/sweeps_parameter_generator.py

# Generate vectors for all sweep modules and export to drive -> tests/sweep_framework/vectors_export
python tests/sweep_framework/sweeps_parameter_generator.py --dump-file

# Generate vectors for a specific module and export to Elasticsearch
python tests/sweep_framework/sweeps_parameter_generator.py --module-name eltwise.unary.relu.relu

# Generate vectors for a specific module and export to drive -> tests/sweep_framework/vectors_export
python tests/sweep_framework/sweeps_parameter_generator.py --module-name eltwise.unary.relu.relu --dump-file
```

- **If exporting to Elasticsearch:** (requires credentials):
  ```bash
  export ELASTIC_USERNAME="your-elastic-username"
  export ELASTIC_PASSWORD="your-elastic-password"
  ```

#### 2. Run Tests

- **Vector sources**: `elastic`, `file`, or `vectors_export`
- **Result destinations**: `postgres`, `elastic`, or `results_export`

- **If vector source OR result destination is elastic** (requires credentials):
  ```bash
  export ELASTIC_USERNAME="your-elastic-username"
  export ELASTIC_PASSWORD="your-elastic-password"
  ```

- **If result destination is postgres** (requires credentials):
  ```bash
  export POSTGRES_HOST="your-postgres-host"
  export POSTGRES_DATABASE="your-database-name"
  export POSTGRES_USER="your-username"
  export POSTGRES_PASSWORD="your-password"
  export POSTGRES_PORT="5432"  # Optional, defaults to 5432
  ```

- **If result destination is results_export**: no database credentials are required (results are written to JSON files under `tests/sweep_framework/results_export/`).

**To run sweeps you must specify the source of the vectors source and destination of the results**
```bash
# Run all available sweep tests
python tests/sweep_framework/sweeps_runner.py --vector-source {elastic,file,vectors_export} --result-dest {elastic,postgres,results_export}

# Run specific module
python tests/sweep_framework/sweeps_runner.py --module-name eltwise.unary.relu.relu --vector-source {elastic,file,vectors_export} --result-dest {elastic,postgres,results_export}

# Run specific suite within a module
python tests/sweep_framework/sweeps_runner.py --module-name eltwise.unary.relu.relu --suite-name suite_1 --vector-source {elastic,file,vectors_export} --result-dest {elastic,postgres,results_export}

# Run multiple modules (comma-separated)
python tests/sweep_framework/sweeps_runner.py --module-name "eltwise.unary.relu.relu,matmul.short.matmul" --vector-source {elastic,file,vectors_export} --result-dest {elastic,postgres,results_export}
```

### Sweep Runner CLI Reference (key options)
- **Tag filter**: Vectors fetched from Elasticsearch are filtered by `--tag` (defaults to your `$USER`). Ensure your tag matches the one used during vector generation.
```bash
- `--module-name`: Module name or comma-separated list (comma-separated supported for `elastic` and `vectors_export` sources)
- `--suite-name`: Suite to run within a module
- `--vector-source`: One of `elastic` (default), `file`, `vectors_export`
- `--file-path`: Path to vectors JSON (required when `--vector-source file`)
- `--vector-id`: Run a single vector by id (requires `--module-name`)
- `--result-dest`: One of `postgres` (default), `elastic`, `results_export`
- `--tag`: Tag to filter vectors in Elasticsearch (defaults to `$USER`)
- `--skip-modules`: Comma-separated modules to skip when running all modules
- `--skip-on-timeout`: Skip remaining tests in a suite if a test times out
- `--keep-invalid`: Include invalid vectors in results with NOT_RUN status (default: exclude invalid vectors from results entirely)
- `--watcher`: Enable watcher
- `--perf`: Measure end-to-end perf for ops that support it
- `--device-perf`: Measure device perf (requires profiler build)
- `--dry-run`: Plan without executing
- `--summary`: Print an execution (or dry-run) summary
```

For information on writing sweep tests, see [`tests/sweep_framework/README.md`](sweep_framework/README.md).

## Unit Test Framework

The unit test framework runs pytest-based tests and provides database integration for result tracking.

### Features

- **Pytest Integration**: Run multiple standard pytests
- **Database Reporting**: Stores test results in PostgreSQL

### Prerequisites
The unit test framework **requires** PostgreSQL credentials to run. Set these environment variables to enable Postgres export:

```bash
export POSTGRES_HOST="your-postgres-host"
export POSTGRES_DATABASE="your-database-name"
export POSTGRES_USER="your-username"
export POSTGRES_PASSWORD="your-password"
export POSTGRES_PORT="5432"  # Optional, defaults to 5432
```

### Usage

```bash
python tests/ttnn/unit_test_runner.py <test_paths> [options]
```

#### Basic Examples

**⚠️ Important: Unit tests require PostgreSQL credentials to be set in environment variables before running.**

```bash
# Set required PostgreSQL environment variables first
export POSTGRES_HOST="your-postgres-host"
export POSTGRES_DATABASE="your-database-name"
export POSTGRES_USER="your-username"
export POSTGRES_PASSWORD="your-password"

# Run a single test file
python tests/ttnn/unit_test_runner.py tests/ttnn/unit_tests/operations/test_concat.py

# Run multiple test files
python tests/ttnn/unit_test_runner.py "tests/ttnn/unit_tests/operations/test_concat.py,tests/ttnn/unit_tests/operations/test_add.py"

# Run all tests in a directory
python tests/ttnn/unit_test_runner.py tests/ttnn/unit_tests/operations/

# Dry run to see what tests would be executed
python tests/ttnn/unit_test_runner.py tests/ttnn/unit_tests/operations/ --dry-run
```

#### Options

- `--dry-run`: Perform a dry run to count test cases without executing them
- `test_paths`: Comma-separated list of paths to test files or directories

### Test Results

When database integration is enabled, results are stored with:
- **Run metadata**: Git hash, author, branch, timestamp, host information
- **Test details**: Individual test outcomes, execution times, error messages
- **Status classification**: PASS, FAIL, ERROR, SKIP, XFAIL (expected failure)


## Database Synchronicity

Both frameworks use a shared PostgreSQL database schema (managed by `tests/sweep_framework/framework/database.py`) when the result destination is PostgreSQL.

### Database Schema

#### Shared Tables
- **`runs`**: High-level test run information (git info, timestamps, overall status)
- **`tests`**: Individual test file/module execution details

#### Framework-Specific Tables
- **`unit_testcases`**: Individual unit test case results
- **`sweep_testcases`**: Individual sweep test vector results


## Test Status Classification

Both frameworks use consistent status classification:

### Unit Test Statuses
- **PASS**: Test passed successfully
- **FAIL**: Test failed due to assertion or logic error
- **ERROR**: Test failed due to unexpected exception
- **SKIP**: Test was skipped
- **XFAIL**: Expected failure (test marked with `@pytest.mark.xfail`)

### Sweep Test Statuses
- **PASS**: Test met expected criteria (usually PCC validation)
- **FAIL_ASSERT_EXCEPTION**: Test failed due to assertion or exception
- **FAIL_L1_OUT_OF_MEM**: Test failed due to L1 memory exhaustion
- **FAIL_WATCHER**: Test failed due to watcher exception
- **FAIL_CRASH_HANG**: Test timed out or crashed
- **NOT_RUN**: Test skipped due to invalid vector
- **FAIL_UNSUPPORTED_DEVICE_PERF**: Device perf requested but unsupported or missing data
