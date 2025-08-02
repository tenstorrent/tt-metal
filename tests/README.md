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
The sweep test framework has two database options:

1. **PostgreSQL** (requires credentials):
   ```bash
   export POSTGRES_HOST="your-postgres-host"
   export POSTGRES_DATABASE="your-database-name"
   export POSTGRES_USER="your-username"
   export POSTGRES_PASSWORD="your-password"
   export POSTGRES_PORT="5432"  # Optional, defaults to 5432
   ```

2. **Elasticsearch** (requires credentials):
   ```bash
   export ELASTIC_USERNAME="your-elastic-username"
   export ELASTIC_PASSWORD="your-elastic-password"
   ```

For sweep tests, you must have credentials for **either** PostgreSQL **or** Elasticsearch depending on which `--database` option you choose.

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

**Choose your database backend and set appropriate credentials:**

**Option A: Using PostgreSQL**
```bash
# Set PostgreSQL credentials first
export POSTGRES_HOST="your-postgres-host"
export POSTGRES_DATABASE="your-database-name"
export POSTGRES_USER="your-username"
export POSTGRES_PASSWORD="your-password"

# Run all available sweep tests
python tests/sweep_framework/sweeps_runner.py --database postgres

# Run specific module
python tests/sweep_framework/sweeps_runner.py --module-name eltwise.unary.relu.relu --database postgres

# Run specific suite within a module
python tests/sweep_framework/sweeps_runner.py --module-name eltwise.unary.relu.relu --suite-name suite_1 --database postgres

# Run multiple modules (comma-separated)
python tests/sweep_framework/sweeps_runner.py --module-name "eltwise.unary.relu.relu,matmul.short.matmul" --database postgres
```

**Option B: Using Elasticsearch**
```bash
# Set Elasticsearch credentials first
export ELASTIC_USERNAME="your-elastic-username"
export ELASTIC_PASSWORD="your-elastic-password"

# Run tests using Elasticsearch (default database backend)
python tests/sweep_framework/sweeps_runner.py --module-name eltwise.unary.relu.relu --elastic cloud

# Run with custom Elasticsearch URL
python tests/sweep_framework/sweeps_runner.py --module-name eltwise.unary.relu.relu --elastic "https://your-elastic-url"
```

**Dry Run (works with either backend):**
```bash
# Dry run to see what would be executed (requires appropriate credentials)
python tests/sweep_framework/sweeps_runner.py --dry-run --database postgres
# or
python tests/sweep_framework/sweeps_runner.py --dry-run --elastic cloud
```

#### 3. Advanced Options

```bash
# Run with performance measurement
python tests/sweep_framework/sweeps_runner.py --module-name mymodule --perf --database postgres

# Run with device profiling (requires profiler build)
python tests/sweep_framework/sweeps_runner.py --module-name mymodule --device-perf --database postgres

# Run with watcher enabled
python tests/sweep_framework/sweeps_runner.py --module-name mymodule --watcher --database postgres

# Run single test vector for debugging
python tests/sweep_framework/sweeps_runner.py --module-name mymodule --vector-id abc123def --database postgres
```

For detailed information on writing sweep tests, see [`tests/sweep_framework/README.md`](sweep_framework/README.md).

## Database Integration

Both frameworks use a shared PostgreSQL database schema managed by the common database module at `tests/sweep_framework/framework/database.py`.

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
