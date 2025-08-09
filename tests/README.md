# TT-Metal Unit Tests and Sweep Test Runners

This README applies to two testing frameworks for in this directory:

1. **Unit Test Framework** (`tests/ttnn/unit_test_runner.py`) - For running pytest-based metal ops unit tests
2. **Sweep Test Framework** (`tests/sweep_framework/`) - For running parameterized sweep tests across ops

Both frameworks support PostgreSQL database integration for result tracking and analysis.

## Quick Start

### Prerequisites

**Database Setup (Required for Database Features):**

#### Unit Test Framework
The unit test framework **requires** PostgreSQL credentials to run. Set these environment variables:

```bash
export POSTGRES_HOST="your-postgres-host"
export POSTGRES_DATABASE="your-database-name"
export POSTGRES_USER="your-username"
export POSTGRES_PASSWORD="your-password"
export POSTGRES_PORT="5432"  # Optional, defaults to 5432
```

**Without these environment variables, the unit test runner will fail to start.**

#### Sweep Test Framework
The sweep test framework separates where vectors are loaded from (vector source) and where results are written to (result destination):

- **Vector sources**: `elastic` (default), `file`, or `vectors_export`
- **Result destinations**: `postgres` (default), `elastic`, or `results_export`

Set credentials based on what you use:

- **If vector source is elastic OR result destination is elastic** (requires credentials):
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

Also note:
- **Tag filter**: Vectors fetched from Elasticsearch are filtered by `--tag` (defaults to your `$USER`). Ensure your tag matches the one used during vector generation.

## Unit Test Framework

The unit test framework runs pytest-based tests and provides database integration for result tracking.

### Features

- **Pytest Integration**: Runs standard pytest test files
- **Database Reporting**: Stores test results in PostgreSQL with detailed metadata
- **Status Tracking**: Supports PASS, FAIL, ERROR, SKIP, XFAIL, XPASS test outcomes
- **Dry Run Mode**: Preview tests without execution
- **Git Integration**: Tracks git hash, author, and branch information
- **Performance Metrics**: Optional e2e and device performance measurement

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
- **Status classification**: PASS, FAIL, ERROR, SKIP, XFAIL (expected failure), XPASS (unexpected pass)

### Example Output

```
=== EXECUTION SUMMARY ===
Total test files: 5
Total test cases executed: 127
Run completed with status: SUCCESS

=== DATABASE STORAGE ===
Run ID: 550e8400-e29b-41d4-a716-446655440000
Results stored in PostgreSQL database
```

## Sweep Test Framework

The sweep test framework runs parameterized tests across large parameter spaces for performance and correctness validation.

### Features

- **Parameterized Testing**: Automatically generates test vectors from parameter combinations
- **Performance Measurement**: Built-in e2e and device performance profiling
- **Result Classification**: Comprehensive test outcome tracking
- **Hang Detection**: Automatic timeout and recovery mechanisms
- **Database Integration**: Stores test vectors and results for analysis
- **Granular Execution**: Run by module, suite, or individual vectors

### Quick Usage

#### 1. Generate Test Vectors

```bash
# Generate vectors for all sweep modules
python tests/sweep_framework/sweeps_parameter_generator.py --dump-file --database {elastic,postgres}

# Generate vectors for a specific module
python tests/sweep_framework/sweeps_parameter_generator.py --module-name eltwise.unary.relu.relu --dump-file
```

#### 2. Run Tests

**Choose your vector source and result destination, then set credentials accordingly:**

**Option A: Default sources (elastic) with PostgreSQL results (default result destination)**
```bash
# Set required credentials
export ELASTIC_USERNAME="your-elastic-username"
export ELASTIC_PASSWORD="your-elastic-password"
export POSTGRES_HOST="your-postgres-host"
export POSTGRES_DATABASE="your-database-name"
export POSTGRES_USER="your-username"
export POSTGRES_PASSWORD="your-password"

# Run all available sweep tests
python tests/sweep_framework/sweeps_runner.py --result-dest postgres --summary

# Run specific module
python tests/sweep_framework/sweeps_runner.py --module-name eltwise.unary.relu.relu --result-dest postgres

# Run specific suite within a module
python tests/sweep_framework/sweeps_runner.py --module-name eltwise.unary.relu.relu --suite-name suite_1 --result-dest postgres

# Run multiple modules (comma-separated)
python tests/sweep_framework/sweeps_runner.py --module-name "eltwise.unary.relu.relu,matmul.short.matmul" --result-dest postgres
```

**Option B: Elasticsearch vectors and results**
```bash
# Set Elasticsearch credentials
export ELASTIC_USERNAME="your-elastic-username"
export ELASTIC_PASSWORD="your-elastic-password"

# Run all tests, storing results in Elasticsearch
python tests/sweep_framework/sweeps_runner.py --result-dest elastic --summary
```

**Option C: Local JSON results export (no DB required)**
```bash
# Set Elasticsearch credentials if using elastic vector source
export ELASTIC_USERNAME="your-elastic-username"
export ELASTIC_PASSWORD="your-elastic-password"

# Export results to tests/sweep_framework/results_export/<module>.json
python tests/sweep_framework/sweeps_runner.py --result-dest results_export --summary
```

**Option D: File-based vectors**
```bash
# Read vectors from a JSON file
python tests/sweep_framework/sweeps_runner.py --vector-source file --file-path /abs/path/to/vectors.json --result-dest results_export --summary
```

**Option E: Vectors from local export directory**
```bash
# Read vectors from tests/sweep_framework/vectors_export/<module>.json
python tests/sweep_framework/sweeps_runner.py --vector-source vectors_export --result-dest results_export --summary
```

**Dry Run (works with either backend):**
```bash
# Dry run to see what would be executed (requires appropriate credentials based on vector source)
python tests/sweep_framework/sweeps_runner.py --dry-run --result-dest postgres
# or, with results exported to JSON
python tests/sweep_framework/sweeps_runner.py --dry-run --result-dest results_export
```

#### 3. Advanced Options

```bash
# Run with performance measurement
python tests/sweep_framework/sweeps_runner.py --module-name mymodule --perf --result-dest postgres

# Run with device profiling (requires profiler build)
python tests/sweep_framework/sweeps_runner.py --module-name mymodule --device-perf --result-dest postgres

# Run with watcher enabled
python tests/sweep_framework/sweeps_runner.py --module-name mymodule --watcher --result-dest postgres

# Run single test vector for debugging
python tests/sweep_framework/sweeps_runner.py --module-name mymodule --vector-id abc123def --result-dest postgres

# Run all modules but skip some (only valid when not specifying --module-name)
python tests/sweep_framework/sweeps_runner.py --skip-modules "eltwise.unary.relu.relu,matmul.short.matmul" --result-dest postgres --summary

# Skip remaining tests in a suite after a timeout
python tests/sweep_framework/sweeps_runner.py --module-name mymodule --skip-on-timeout --result-dest postgres

# Print a detailed summary at the end
python tests/sweep_framework/sweeps_runner.py --module-name mymodule --summary --result-dest postgres
```

For detailed information on writing sweep tests, see [`tests/sweep_framework/README.md`](sweep_framework/README.md).

## Database Integration

Both frameworks can use a shared PostgreSQL database schema (managed by `tests/sweep_framework/framework/database.py`) when the result destination is PostgreSQL. The sweep framework can also write results to Elasticsearch or to JSON files (when using `results_export`).

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
- **XPASS**: Unexpected pass (xfail test that passed)

### Sweep Test Statuses
- **PASS**: Test met expected criteria (usually PCC validation)
- **FAIL_ASSERT_EXCEPTION**: Test failed due to assertion or exception
- **FAIL_L1_OUT_OF_MEM**: Test failed due to L1 memory exhaustion
- **FAIL_WATCHER**: Test failed due to watcher exception
- **FAIL_CRASH_HANG**: Test timed out or crashed
- **NOT_RUN**: Test skipped due to invalid vector
- **FAIL_UNSUPPORTED_DEVICE_PERF**: Device perf requested but unsupported or missing data

### Sweep Runner CLI Reference (key options)
- `--module-name`: Module name or comma-separated list (comma-separated supported for `elastic` and `vectors_export` sources)
- `--suite-name`: Suite to run within a module
- `--vector-source`: One of `elastic` (default), `file`, `vectors_export`
- `--file-path`: Path to vectors JSON (required when `--vector-source file`)
- `--vector-id`: Run a single vector by id (requires `--module-name`)
- `--result-dest`: One of `postgres` (default), `elastic`, `results_export`
- `--tag`: Tag to filter vectors in Elasticsearch (defaults to `$USER`)
- `--skip-modules`: Comma-separated modules to skip when running all modules
- `--skip-on-timeout`: Skip remaining tests in a suite if a test times out
- `--watcher`: Enable watcher
- `--perf`: Measure end-to-end perf for ops that support it
- `--device-perf`: Measure device perf (requires profiler build)
- `--dry-run`: Plan without executing
- `--summary`: Print an execution (or dry-run) summary
