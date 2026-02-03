# AI Prompt: Reproduce Non-Deterministic Failures

## Context

You are tasked with creating a stress test that reproduces a non-deterministic failure from CI logs within 20 minutes. You're running on the same machine/environment where the test failed.

## Directory Structure

```
.github/scripts/reproduce-ND-failures/
  <failure-name>/
    logs/          # Contains CI error logs (created by user)
    tests/         # You will write tests here
```

## Input

The user has created a subfolder named `<failure-name>` with:
- `logs/` - Contains one or more error logs from CI failures
- `tests/` - Empty folder where you will write reproducible tests

## Your Task

Create a standalone stress test in the `tests/` folder that reproduces the failure within 20 minutes.

## Step-by-Step Process

### 1. Analyze the Error Logs

Read all logs in `logs/` and identify:
- What failed? (timeout, assertion, crash, hang)
- When did it fail? (timing information, which rank/process)
- What were the symptoms? (error messages, stack traces)
- What test was running? (test name, parameters)

### 2. Find and Analyze the Original Test

**Critical: You must read and understand the original test code before creating a stress test.**

#### 2a. Extract Test Information from Logs

From the error logs, identify:
- Test name (e.g., `test_ccl_all_gather`, `test_reduce_scatter_post_commit`)
- Test file path (e.g., `tests/ttnn/unit_tests/operations/ccl/test_reduce_scatter.py`)
- Test parameters/configuration used
- Command that ran the test (pytest invocation, script call, etc.)

#### 2b. Find the Test File

Search for the test in the codebase:
```bash
# Search by test name
grep -r "def test_reduce_scatter" tests/

# Or use glob to find test files
find tests/ -name "*reduce_scatter*.py"
```

**Read the entire test file** to understand:
- What the test does
- Setup/teardown logic
- Parameters and configuration
- Dependencies (imports, fixtures, helper functions)

#### 2c. Find the YAML Configuration (if provided)

The user may provide YAML files that show how CI ran the test. Look for:
- `.github/workflows/*.yaml` - Workflow definitions
- `tests/scripts/*.yaml` - Test configurations
- Pipeline configs that specify environment variables, hardware, test invocation

**Extract from YAML:**
- Environment variables set for the test
- Hardware requirements (device type, number of chips)
- Test command and arguments
- Timeout values
- Any special setup steps

#### 2d. Understand the Full Test Context

Before proceeding, you must understand:
- What does this test do? (functionality being tested)
- How is it invoked? (command, args, env vars)
- What hardware does it need? (T3000, Wormhole, Galaxy, etc.)
- What are the dependencies? (other tests, setup scripts, data files)
- What is the expected flow? (setup → test → teardown)

### 3. Identify the Root Cause

Determine what made this failure likely:
- **Race condition**: Timing skew between processes/ranks/threads
- **Resource contention**: Multiple processes accessing same resource
- **Synchronization issue**: Deadlock, livelock, or missing sync
- **Hardware edge case**: Specific chip behavior, thermal, power
- **Timeout**: Operation taking longer than expected
- **State corruption**: Accumulated state from previous operations

### 4. Amplification Strategy

Choose strategies to make the failure more likely:

**For Race Conditions:**
- Add artificial delays at synchronization points
- Increase the number of parallel processes/ranks
- Remove or reduce safety margins

**For Resource Contention:**
- Increase parallelism (more workers, more processes)
- Reduce buffer sizes or resource limits
- Add competing background operations

**For Timing Issues:**
- Reduce timeouts to narrow the window
- Add delays before critical operations
- Increase iteration counts

**For Hardware Edge Cases:**
- Target specific device IDs that failed
- Stress specific execution paths
- Run operations back-to-back without cleanup

**General Amplifiers:**
- Loop the test (50-100 iterations minimum)
- Add logging to confirm we hit the failure condition
- Use environment variables to control behavior
- Make it easy to adjust amplification parameters (CLI args)

### 5. Write the Stress Test

Create a new file in `tests/` with:

**File naming:** `test_<descriptive_name>_stress.py` or similar

**Code structure:**
```python
"""
Stress test to reproduce: <failure description>

Original failure: <job name> - <date>
Error: <brief error description>

This test amplifies <specific condition> to reproduce the failure.
Run with: <command to run>
"""

import ...

def setup():
    """Setup test environment"""
    pass

def stress_test(iterations=100, **amplification_params):
    """
    Main stress test loop

    Args:
        iterations: Number of times to run (default: 100)
        amplification_params: Parameters to control amplification
    """
    for i in range(iterations):
        # Run amplified version of original test
        pass

if __name__ == "__main__":
    # CLI args for easy parameter tuning
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    # Add amplification parameters
    args = parser.parse_args()

    stress_test(iterations=args.iterations, ...)
```

**Must include:**
- Clear docstring explaining what failure we're reproducing
- Comments explaining the amplification strategy
- Logging to show progress and when failure occurs
- CLI arguments for tuning parameters
- Exit with non-zero code on failure

### 6. Verify Reproduction

**IMPORTANT: Activate the virtual environment before running tests**

```bash
source python_env/bin/activate  # Activate venv
python tests/test_*_stress.py   # Run your test
```

Run the test multiple times to verify:
- Does it reproduce the failure within 20 minutes?
- Is it consistent (reproduces >50% of the time)?
- Does it fail with the same error as the original?

If not reproducing:
- Increase amplification (more delay, more iterations)
- Try different amplification strategies
- Check if environment matches CI (hardware, env vars)
- Verify you're in the virtual environment

### 7. Document

Add a README in the `<failure-name>` folder:

**File:** `<failure-name>/README.md`

```markdown
# <Failure Name>

## Original Failure

- **Job**: <CI job name>
- **Date**: <when it failed>
- **Frequency**: <how often it fails>
- **Error**: <brief description>

## Root Cause

<What causes this failure>

## Reproduction

The stress test in `tests/` reproduces this by:
- <Amplification strategy 1>
- <Amplification strategy 2>

### Run Test

```bash
# Activate virtual environment first
source python_env/bin/activate

# Run the stress test
<command to run the test>
```

### Expected Behavior

- **Success**: <what success looks like>
- **Failure**: <what the reproduced failure looks like>

## Results

<Document your reproduction attempts and results>
```

## Important Notes

- **Activate venv first** - Always run tests inside `python_env/bin/activate`
- **Don't guess** - Read the actual test code and logs
- **Start simple** - Try minimal amplification first, then increase
- **Be specific** - Target the exact code path that failed
- **Make it tunable** - Use CLI args so parameters can be adjusted
- **Log everything** - Make it easy to see what's happening
- **Fail fast** - Don't wait for natural timeouts, reduce them

## Example: T3K Reduce Scatter Race

See `.github/scripts/reproduce-ND-failures/T3K-reduce-scatter-race/` for a complete example of this workflow.

## Success Criteria

Your stress test successfully reproduces the failure if:
1. It runs standalone without manual setup
2. It reproduces the failure within 20 minutes
3. The failure mode matches the original error
4. It can be run repeatedly for verification
5. Parameters can be tuned via CLI arguments
