# TT-Metal Testing Framework

This directory contains two main testing frameworks for TT-Metal:

1. **Unit Test Framework** (`tests/ttnn/unit_test_runner.py`) - For running pytest-based unit tests
2. **Sweep Test Framework** (`tests/sweep_framework/`) - For running parameterized performance and correctness tests

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

### Database Configuration

#### Environment Variables
```bash
export POSTGRES_HOST="localhost"          # Database host
export POSTGRES_DATABASE="ttmetal_tests"  # Database name
export POSTGRES_USER="testuser"           # Username
export POSTGRES_PASSWORD="password"       # Password
export POSTGRES_PORT="5432"               # Port (optional)
```

#### Environments
Both frameworks support multiple database environments:
- **`dev`**: Development environment (default for unit tests)
- **`prod`**: Production environment (default for sweep tests)

### Querying Results

Example SQL queries for analyzing test results:

```sql
-- Get recent test runs
SELECT id, initiated_by, device, status, start_time_ts, end_time_ts
FROM runs
ORDER BY start_time_ts DESC
LIMIT 10;

-- Get failing tests from latest run
SELECT t.name, tc.status, tc.exception
FROM tests t
JOIN unit_testcases tc ON t.id = tc.test_id
WHERE t.run_id = (SELECT id FROM runs ORDER BY start_time_ts DESC LIMIT 1)
  AND tc.status LIKE 'fail%';

-- Performance trends for a specific test
SELECT DATE(start_time_ts) as date, AVG(e2e_perf) as avg_perf
FROM sweep_testcases
WHERE name LIKE '%matmul%'
  AND e2e_perf IS NOT NULL
GROUP BY DATE(start_time_ts)
ORDER BY date;
```

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

## Performance Measurement

### Unit Tests
Performance measurement in unit tests is handled through pytest plugins and can include:
- Test execution time
- Memory usage patterns
- Custom performance markers

### Sweep Tests
Sweep tests offer comprehensive performance measurement:

```python
# In your sweep test run() function
from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time

def run(..., *, device):
    # Test setup...

    start_time = start_measuring_time()
    output_tensor = ttnn.operation(input_tensor, ...)
    e2e_perf = stop_measuring_time(start_time)

    # Validation...

    return [check_with_pcc(expected, actual, 0.999), e2e_perf]
```

## Troubleshooting

### Common Issues

#### Database Connection Errors

**Unit Test Runner:**
```
ValueError: Missing required PostgreSQL environment variables: POSTGRES_HOST, POSTGRES_USER
```
**Solution**: Set the required PostgreSQL environment variables. Unit tests cannot run without database credentials:
```bash
export POSTGRES_HOST="your-postgres-host"
export POSTGRES_DATABASE="your-database-name"
export POSTGRES_USER="your-username"
export POSTGRES_PASSWORD="your-password"
```

**Sweep Test Runner:**
```
RuntimeError: The psycopg2 library is required but not installed
```
**Solution**: Either install PostgreSQL dependencies or use Elasticsearch instead:
```bash
# Option 1: Install PostgreSQL support
pip install psycopg2-binary

# Option 2: Use Elasticsearch (set credentials and omit --database postgres)
export ELASTIC_USERNAME="your-username"
export ELASTIC_PASSWORD="your-password"
python tests/sweep_framework/sweeps_runner.py --elastic corp
```

#### Import Errors
```
ModuleNotFoundError: No module named 'psycopg2'
```
**Solution**: Install PostgreSQL dependencies:
```bash
pip install psycopg2-binary
```

#### Memory Issues
```
FAIL_L1_OUT_OF_MEM: Not enough space to allocate
```
**Solution**: Reduce batch sizes, tensor dimensions, or use different memory configurations in your test parameters.

#### Timeout Issues
```
FAIL_CRASH_HANG: TEST TIMED OUT (CRASH / HANG)
```
**Solution**:
- Increase timeout in test files by setting `TIMEOUT = <seconds>`
- Check for infinite loops or deadlocks in test code
- Verify hardware setup and driver status

### Getting Help

1. **Check logs**: Both frameworks provide detailed logging
2. **Use dry-run**: Preview what tests will be executed
3. **Database queries**: Analyze historical results for patterns
4. **Incremental testing**: Start with small parameter sets and scale up

### Environment Setup

Ensure your environment is properly configured:

```bash
# Create and activate Python environment
./create_venv.sh
source python_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up database credentials (REQUIRED)
# For unit tests - PostgreSQL is mandatory:
export POSTGRES_HOST="your-postgres-host"
export POSTGRES_DATABASE="your-database-name"
export POSTGRES_USER="your-username"
export POSTGRES_PASSWORD="your-password"

# For sweep tests - choose one database backend:
# Option A: PostgreSQL (same as above)
# Option B: Elasticsearch
export ELASTIC_USERNAME="your-elastic-username"
export ELASTIC_PASSWORD="your-elastic-password"
```

## Best Practices

### Unit Tests
- Use descriptive test names
- Group related tests in the same file
- Use pytest fixtures for setup/teardown
- Mark slow tests appropriately
- Use `@pytest.mark.xfail` for known issues

### Sweep Tests
- Keep parameter combinations reasonable (< 10,000 per suite)
- Use meaningful suite names
- Implement proper vector validation
- Include performance assertions where appropriate
- Test edge cases in separate suites

### Database Usage
- Use consistent run descriptions
- Tag runs appropriately for later analysis
- Query results regularly to identify trends
- Archive old data periodically

This testing framework provides comprehensive coverage for both focused unit testing and broad parameter sweep validation, ensuring robust testing across the TT-Metal ecosystem.
