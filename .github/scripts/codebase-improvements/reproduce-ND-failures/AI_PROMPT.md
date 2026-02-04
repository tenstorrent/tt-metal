# AI Prompt: Reproduce Non-Deterministic Failures

## Context

You are tasked with creating a stress test that reproduces a non-deterministic failure from CI logs within 5 minutes. You're running on the same machine/environment where the test failed.

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

Create a standalone stress test in the `tests/` folder that reproduces the failure within 5 minutes.

## CRITICAL CHECKLIST - Don't Skip These

Before writing any code, ensure you have:

- [ ] Found and read the **exact CI YAML configuration** for the failing job
- [ ] Extracted **ALL environment variables** from YAML and logs
- [ ] Found any **timeout or special settings** relevant to the failure type (e.g., hang-detection for timeouts)
- [ ] Identified the **failure type**: device timeout, test timeout, assertion, crash, race condition, etc.
- [ ] Located the **exact line of code** where the failure occurs
- [ ] Determined the **minimal failing operation** (don't reproduce the whole test suite)
- [ ] Listed what **CANNOT be replicated** locally and how to mock it
- [ ] Chosen an **amplification strategy** (parallel workers, iterations, etc.)
- [ ] Verified your **device setup matches the original test** (fixture vs direct open, MeshDevice vs single device)

**If you haven't done ALL of the above, STOP and go back. Missing any of these causes reproduction to fail.**

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

### 3. Analyze CI Environment in EXTREME DETAIL

**CRITICAL: This is where most reproduction attempts fail. You MUST extract EVERY detail.**

#### 3a. Find the Exact CI Job Configuration

Search for the workflow YAML file that ran the failing test:
```bash
grep -r "job-name-from-logs" .github/workflows/
```

**Read the ENTIRE job definition** and extract:

1. **All Environment Variables** (under `env:` in the YAML):
   - `PYTHONPATH`, `LD_LIBRARY_PATH`, `TT_METAL_HOME`
   - `ARCH_NAME`, `LOGURU_LEVEL`
   - `TEST_BUDGET_SECONDS`, timeout settings
   - **SPECIAL TIMEOUT SETTINGS** like `hang-detection-timeout`, `TT_METAL_TIMEOUT`, etc.
   - Any custom environment variables specific to this test type

2. **Container/Docker Configuration**:
   - Image used
   - Mounted volumes
   - Device access flags
   - Working directory

3. **Test Execution Details**:
   - Exact pytest command (including ALL flags)
   - Test splitting (`--splits`, `--group`)
   - Parallel execution (`-n auto` or sequential)
   - Markers (`-m "..."`)
   - Timeout flags (`--timeout=N`)

4. **Pre-test Steps**:
   - What commands ran BEFORE the failing test?
   - Did other test suites run first?
   - Any setup scripts or initialization?

#### 3b. Search Logs for Hidden Configuration

**CRITICAL: Logs often contain configuration that's not in YAML files.**

Search the logs for:
```bash
# Look for timeout settings
grep -i "timeout" logs/*.log | head -20

# Look for environment variable dumps
grep -i "env\|export\|ARCH\|TT_METAL" logs/*.log | head -50

# Look for special settings
grep -i "hang-detection\|watchdog\|timeout-detection" logs/*.log
```

**Common hidden settings to look for:**
- Hang detection timeouts (often 5-30 seconds)
- Device watchdog timers
- Memory limits or buffer sizes
- Debug/trace flags
- Fast dispatch modes

#### 3c. Identify What CANNOT Be Replicated

**Be realistic about what you can't replicate locally:**

- Mount points that don't exist (`/mnt/MLPerf`, etc.)
- Specific Docker images or containers
- Multiple physical devices (if you only have one)
- Specific hardware states or thermal conditions
- Previous test suite execution (accumulated state)

**For each thing you can't replicate, decide:**
1. **Mock it**: Set dummy values that won't break the test
2. **Skip it**: If it's not relevant to the failure
3. **Simplify it**: Replace complex setup with minimal equivalent

#### 3d. Extract the EXACT Failure Point

**Don't just know the test failed - know EXACTLY where and how.**

From the error logs, identify:

1. **Type of Failure**:
   - Test timeout (pytest `--timeout` exceeded)?
   - Device timeout (internal watchdog/hang detection)?
   - Assertion failure?
   - Exception/crash?

2. **Exact Location**:
   - Which function/operation failed?
   - What was the stack trace?
   - Which line of code in the test?

3. **Failure Mechanism**:
   - Is it a timeout WAITING for something?
   - Is it a hang DURING an operation?
   - Is it an error AFTER an operation?

**Example: Device Timeout vs Test Timeout**

```
# Device timeout (internal watchdog):
RuntimeError: TT_THROW @ /project/tt_metal/impl/dispatch/system_memory_manager.cpp:627
TIMEOUT: device timeout, potential hang detected
→ This is NOT a pytest timeout
→ This happens INSIDE the test during a specific operation
→ Device has internal timeout (often 5s if hang-detection is enabled)

# Test timeout (pytest):
FAILED tests/test.py::test_name - pytest_timeout.Timeout: test exceeded timeout
→ This is pytest's --timeout flag
→ Entire test took too long
```

### 4. Isolate the Failing Component

**CRITICAL: Don't reproduce the entire test suite - isolate the EXACT failing operation.**

#### 4a. Identify the Minimal Failing Operation

From your analysis, determine the SMALLEST piece of code that can trigger the failure.

**Example: If the failure is in tensor conversion:**
```python
# Don't reproduce the entire test suite ❌
# Don't even reproduce the entire test ❌
# Just reproduce the specific operation ✓

# Minimal failing code:
ttnn_tensor = ttnn.gather(input, dim, index=indices)
result = ttnn.to_torch(ttnn_tensor)  # ← Fails here
```

#### 4b. Strip Away Unnecessary Context

Remove everything not directly related to the failure:
- Complex test fixtures → Simple setup
- Multiple test variants → Just the failing case
- Validation logic → Minimal assertions
- Logging/debugging → Just failure detection

#### 4c. Design Focused Stress Test

Create a test that:
1. **Runs ONLY the failing operation**
2. **Repeats it many times** (50-100+ iterations)
3. **Uses amplification** to increase failure probability
4. **Completes within 5 minutes**

### 5. Identify the Root Cause

Determine what made this failure likely:
- **Race condition**: Timing skew between processes/ranks/threads
- **Resource contention**: Multiple processes accessing same resource
- **Synchronization issue**: Deadlock, livelock, or missing sync
- **Hardware edge case**: Specific chip behavior, thermal, power
- **Timeout**: Operation taking longer than expected
- **State corruption**: Accumulated state from previous operations

### 6. Amplification Strategy

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
- **Use `pytest -n auto` for parallel worker stress** (if race condition suspected)

### 7. Set Up Environment Variables

**CRITICAL: Missing environment variables is a common reason reproduction fails.**

Create a setup script or document the required environment:

```bash
#!/bin/bash
# Set ALL CI environment variables

# Core paths
export PYTHONPATH="/tt-metal"
export LD_LIBRARY_PATH="/tt-metal/build/lib"
export TT_METAL_HOME="/tt-metal"

# Architecture (from CI job name or logs)
export ARCH_NAME="wormhole_b0"  # or grayskull, blackhole, etc.

# Logging
export LOGURU_LEVEL="INFO"

# Timeouts - CRITICAL for reproducing timeout issues
export TEST_BUDGET_SECONDS="600"

# HANG DETECTION - Often the key to reproducing device timeouts
# Search logs for: "hang-detection-timeout", "watchdog", "timeout-detection"
export TT_METAL_HANG_DETECTION_TIMEOUT="5"  # Often 5 seconds in CI
# Or similar variables like:
# export TT_METAL_DEVICE_TIMEOUT="5"
# export TT_METAL_DISPATCH_TIMEOUT="5"

# Mock missing paths
export HF_HOME="${HF_HOME:-/tmp/hf_home}"  # Mock if /mnt/MLPerf doesn't exist

# Any other variables from CI YAML or logs
```

**How to Find Timeout Settings:**

1. Search YAML files:
```bash
grep -i "timeout\|hang\|watchdog" .github/workflows/*.yaml
```

2. Search logs for environment setup:
```bash
grep -i "timeout.*:" logs/*.log | grep -v "pytest"
grep -i "hang-detection" logs/*.log
```

3. Look for device initialization logs:
```bash
grep -i "device.*timeout\|dispatch.*timeout" logs/*.log
```

### 8. Write the Focused Stress Test

**CRITICAL: You must create TWO files:**
1. `tests/test_<descriptive_name>_stress.py` - The Python test
2. `run_test.sh` - Bash script to set up environment and run the test

#### 8a. Create the Bash Runner Script

**File naming:** `run_test.sh`

**This is MANDATORY** - The bash script is the primary way to run the test.

```bash
#!/bin/bash

# Runner script for <test name> stress test
# Sets up environment and runs the test

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "<Test Name> Stress Test"
echo "=========================================="
echo ""

# Set ALL required environment variables
echo "Setting up environment..."

# CRITICAL: Set any timeout/hang detection variables
export TT_METAL_OPERATION_TIMEOUT_SECONDS=5  # Example: adjust as needed

# Architecture
export ARCH_NAME=${ARCH_NAME:-wormhole_b0}

# Metal paths
export TT_METAL_HOME=${TT_METAL_HOME:-/tt-metal}
export PYTHONPATH=${PYTHONPATH:-/tt-metal}

# Logging
export LOGURU_LEVEL=${LOGURU_LEVEL:-INFO}

# Add any other environment variables from CI
echo "  TT_METAL_OPERATION_TIMEOUT_SECONDS=$TT_METAL_OPERATION_TIMEOUT_SECONDS"
echo "  ARCH_NAME=$ARCH_NAME"
echo ""

# Activate virtual environment
if [ -f "/opt/venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source /opt/venv/bin/activate
elif [ -f "$TT_METAL_HOME/python_env/bin/activate" ]; then
    source "$TT_METAL_HOME/python_env/bin/activate"
fi

echo ""
echo "Running stress test..."
echo "=========================================="
echo ""

# Change to test directory
cd "$SCRIPT_DIR/tests"

# Run the test with pytest
# For stress tests: use -n auto for parallel workers (amplification)
# Use -x to stop on first failure
# Use --timeout to prevent hanging
pytest test_<name>_stress.py -n auto -x -v --timeout=300 "$@"

TEST_RESULT=$?

echo ""
echo "=========================================="
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ Test PASSED (no failures reproduced)"
else
    echo "❌ Test FAILED (failure reproduced!)"
fi
echo "=========================================="

exit $TEST_RESULT
```

**Make it executable:**
```bash
chmod +x run_test.sh
```

#### 8b. Create the Python Stress Test

**File naming:** `tests/test_<descriptive_name>_stress.py`

**Design Principles:**
1. **Isolate**: Test ONLY the failing operation, not the whole test
2. **Simplify**: Remove unnecessary setup, validation, dependencies
3. **Amplify**: Use parallelism (`pytest -n auto`) or high iteration counts
4. **Focus**: Target the exact failure mechanism (device timeout, race condition, etc.)
5. **Fast**: Complete within 5 minutes
6. **Match device setup**: Use the same device initialization as the original test (fixture, direct open, etc.)

**Code structure:**
```python
"""
Stress test to reproduce: <failure description>

Original failure: <job name> - <date>
Error: <brief error description>

This test amplifies <specific condition> to reproduce the failure.

Run with:
    cd <failure-folder>
    ./run_test.sh
"""

import pytest
import torch
import ttnn

@pytest.mark.parametrize("iteration", range(50))
def test_<name>_stress(iteration, device):
    """
    Stress test that amplifies <failure condition>.

    Each iteration runs the failing operation. With pytest -n auto,
    multiple workers run in parallel, increasing failure probability.
    """
    # Minimal setup - just what's needed
    input_tensor = torch.randn([batch, size], dtype=torch.bfloat16)

    # The exact failing operation
    ttnn_input = ttnn.from_torch(input_tensor, device=device)
    result = ttnn.operation_that_fails(ttnn_input)  # ← Fails here

    # Basic validation
    assert result is not None
```

**Must include:**
- Clear docstring with instructions to use run_test.sh
- Comments explaining the amplification strategy
- Parametrized iterations (50-100+)
- Logging to show progress and when failure occurs
- Exit with non-zero code on failure

### 9. Verify Reproduction

**IMPORTANT: Set environment variables AND activate the virtual environment before running tests**

---
⚠️ **MANDATORY: RUN TESTS IN BACKGROUND AND MONITOR OUTPUT** ⚠️

**You MUST run tests in background mode** so you can monitor progress, detect errors early, and react without waiting for timeouts. If you run a blocking command and the test hangs or fails, you will be stuck waiting and unable to help the user.

**WRONG (BLOCKING - DO NOT DO THIS):**
```bash
pytest tests/test_stress.py -v 2>&1 | tee logs/run_1.log
```
This blocks until completion. If the test hangs for 5 minutes, you sit there doing nothing.

**RIGHT (BACKGROUND + MONITOR):**
```bash
# Run in background
pytest tests/test_stress.py -v 2>&1 | tee logs/run_1.log &
TEST_PID=$!

# Monitor the output file periodically
sleep 10 && tail -50 logs/run_1.log
```

Or use the Bash tool's `run_in_background: true` parameter and then use `Read` or `tail` to check the output file.

---

### 9a. Background Execution Workflow (MANDATORY)

**Every test run MUST follow this pattern:**

1. **Start test in background:**
   ```bash
   # Using Bash tool with run_in_background: true
   pytest tests/test_stress.py -x -v --timeout=60 2>&1 | tee logs/run_1.log
   ```

2. **Wait briefly, then check output:**
   ```bash
   # After 10-15 seconds, check what's happening
   tail -50 logs/run_1.log
   ```

3. **Look for these signals:**
   - `PASSED` / `FAILED` - Test completed
   - `TIMEOUT` / `RuntimeError` - Error occurred (may need board reset)
   - `Error` / `Exception` - Something went wrong
   - No new output for 30+ seconds - Possible hang

4. **React appropriately:**
   - **Test passed:** Continue to next test or finish
   - **Test failed with expected error:** SUCCESS! You reproduced it. Run sequential for clean logs.
   - **Test failed with unexpected error:** Analyze, fix test, reset board if needed, try again
   - **Test hanging:** Cancel the process, reset board, try different approach

5. **Reset board after device errors:**
   ```bash
   # If you see device errors, reset before trying again
   tt-smi -r
   sleep 5
   ```

### 9b. Detecting and Handling Errors Early

**Check output every 10-15 seconds.** Don't wait for the full timeout.

**Signs you need to cancel and reset:**
- `Read unexpected run_mailbox value` - Device contention, wrong test setup
- `TIMEOUT: device timeout` - You may have reproduced it! Or device is stuck.
- `RuntimeError: TT_THROW` - Device error, may need reset
- Multiple workers failing simultaneously - Likely device contention issue
- No output for 30+ seconds after tests started - Possible hang

**When you see errors:**
1. Cancel the running test (kill the process or stop the background task)
2. Read the full log to understand what happened
3. Reset the board: `tt-smi -r`
4. Analyze what went wrong
5. Fix the test or try a different approach
6. Run again

### 9c. Example: Proper Background Test Execution

```bash
# Step 1: Set environment
export TT_METAL_OPERATION_TIMEOUT_SECONDS=5
export ARCH_NAME=wormhole_b0
export PYTHONPATH=/tt-metal
export TT_METAL_HOME=/tt-metal
source /opt/venv/bin/activate

# Step 2: Start test in background (using run_in_background: true in Bash tool)
cd /path/to/tests
pytest test_stress.py -x -v --timeout=60 2>&1 | tee ../logs/run_1.log

# Step 3: After 10-15 seconds, check progress
tail -100 ../logs/run_1.log

# Step 4: If you see errors or success, react
# - If FAILED with expected error: Great! Run sequential for clean trace
# - If FAILED with unexpected error: Analyze, reset board, fix test
# - If still running: Wait and check again

# Step 5: If device errors occurred, reset before next attempt
tt-smi -r
sleep 5
```

---

**Why background execution is mandatory:**
- Tests can hang or timeout, blocking you for minutes
- Early error detection lets you fix issues faster
- You can reset the board and retry without waiting
- The user sees progress instead of wondering if you're stuck
- Device errors require board reset before continuing

---

**Why save to logs/?**
- The user needs to see what happened during the test run
- Long-running tests may time out or be interrupted
- Errors and stack traces must be preserved for debugging
- Multiple runs should be numbered (run_1.log, run_2.log, etc.)

**Run using the bash script:**

```bash
# Simply run the script - it handles everything
cd <failure-name>
./run_test.sh 2>&1 | tee logs/run_1.log
```

The bash script will:
- Set all environment variables (including critical timeouts)
- Activate the virtual environment
- Run the test with appropriate pytest flags
- Report pass/fail status

**What to look for:**
- Device timeout exception: `RuntimeError: ... TIMEOUT: device timeout`
- Specific error message matching the CI logs
- Hang during the specific operation you identified

Run the test multiple times to verify:
- Does it reproduce the failure within 5 minutes?
- Does it fail with the same error as the original?
- If using `pytest -n auto`, does parallel execution increase failure rate?

**If not reproducing after 10 runs:**

1. **Check environment variables in run_test.sh** - This is the #1 reason reproduction fails:
   ```bash
   # Verify hang detection timeout is set
   grep TIMEOUT run_test.sh
   # Search for other timeout variables you might have missed
   grep -i "timeout" logs/*.log | grep -v "pytest"
   ```

2. **Increase amplification:**
   - Add `pytest -n auto` for parallel workers
   - Increase iteration count (100+, 500+)
   - Reduce delays between operations
   - Test multiple tensor sizes/parameters in parallel

3. **Verify you're testing the right thing:**
   - Are you reproducing the exact failing operation?
   - Are you looking for the right error (device timeout vs test timeout)?
   - Did you isolate to just the failing component?

**The test MUST complete within 5 minutes** - design it to maximize stress and failure reproduction within this time budget.

### 10. Document

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
source /opt/venv/bin/activate

# Run the stress test
<command to run the test>
```

### Expected Behavior

- **Success**: <what success looks like>
- **Failure**: <what the reproduced failure looks like>

## Results

<Document your reproduction attempts and results>
```

## Common Mistakes to AVOID

These are the most common reasons reproduction attempts fail:

### 1. Missing Environment Variables ❌
**Wrong:** Running test without setting CI environment variables
**Right:** Extract ALL variables from YAML and logs - search for `export`, `env:`, variable names in error messages

### 2. Missing Timeout Settings (for timeout failures) ❌
**Wrong:** Not setting device timeout → test hangs forever or doesn't trigger the timeout error
**Right:** Search YAML/logs for timeout variables and set them (e.g., `TT_METAL_OPERATION_TIMEOUT_SECONDS`, `TT_METAL_HANG_DETECTION_TIMEOUT`, `hang-detection-timeout`)

### 3. Reproducing Wrong Thing ❌
**Wrong:** Reproducing the entire test suite or test
**Right:** Isolate and reproduce ONLY the failing operation

### 4. Wrong Failure Type ❌
**Wrong:** Looking for pytest timeout when it's a device timeout exception
**Right:** Understand if it's a test timeout vs device timeout vs assertion vs crash

### 5. Not Simplifying ❌
**Wrong:** Trying to replicate entire CI environment including missing mounts
**Right:** Mock or skip things that don't affect the failure

### 6. No Amplification ❌
**Wrong:** Running the test once and expecting it to fail
**Right:** Use `pytest -n auto`, high iteration counts, or other amplification

### 7. Ignoring Logs ❌
**Wrong:** Only reading the YAML file
**Right:** Search logs for hidden config, timeout settings, environment dumps

### 8. Not Creating Bash Runner Script ❌ (CRITICAL)
**Wrong:** Only creating the Python test file
**Right:** Create both `run_test.sh` AND `test_*_stress.py`

The bash script is MANDATORY because:
- Sets up environment variables automatically (especially timeouts!)
- Activates virtual environment
- Can be run from any directory
- Works in CI without manual setup
- Documents all required configuration
- Makes the test truly portable

### 9. Not Saving Test Output ❌ (CRITICAL)
**Wrong:** `pytest tests/test_stress.py -v` - user CANNOT see what happened
**Right:** `./run_test.sh 2>&1 | tee logs/run_1.log` - output saved to file

**EVERY test run MUST save output to logs/** - no exceptions. The user cannot see your terminal output. If you don't save to a log file, the user has no way to verify results or see errors.

### 9. Mismatching Device Setup ❌
**Wrong:** Using different device initialization than the original test
**Right:** Match EXACTLY how the original test creates/obtains the device

**Read the original test** to see how it handles devices:
- Does it use a `device` fixture from conftest.py? → Use that fixture
- Does it call `ttnn.open_device()` directly? → Do the same
- Does it use `MeshDevice` or single device? → Match it

The key is **matching the original test's device handling**, not following a fixed pattern.

### 10. Not Stopping on First Failure ❌
**Wrong:** Running 50+ test iterations without `-x` flag, continuing after error reproduced
**Right:** Use `pytest -x` to stop on first failure once you're trying to reproduce

**When to use `-x`:**
- When running stress tests to reproduce a failure
- When you want to capture the exact error and stack trace
- When continuing after failure wastes time (device may be in bad state)

**When NOT to use `-x`:**
- Initial exploration to see how many tests fail
- Verifying a fix works across all test cases

```bash
# For reproduction - stop on first failure
pytest tests/test_*_stress.py -n auto -x -v --timeout=300 2>&1 | tee logs/run.log
```

### 11. Parallel Execution Log Capture Issues ❌
**Wrong:** Expecting clean log output with `pytest -n auto` (parallel workers)
**Right:** Understand that parallel output is interleaved and may need sequential run for clean logs

With `pytest -n auto`:
- Output from different workers is interleaved
- Stack traces may be split across multiple lines
- Some output may be buffered or lost

**Best practice:**
1. First reproduce with `-n auto` (matches CI, higher stress)
2. If error is reproduced, run AGAIN with sequential execution for clean logs:
   ```bash
   # Clean sequential run to capture exact error
   pytest tests/test_*_stress.py -x -v --timeout=300 2>&1 | tee logs/sequential_run.log
   ```

### 12. Running Tests in Blocking Mode ❌ (CRITICAL)
**Wrong:** Running pytest as a blocking command and waiting for completion
**Right:** Run tests in background, monitor output, detect errors early, react quickly

**Why this matters:**
- If a test hangs, you're stuck waiting for the timeout (potentially minutes)
- You can't see errors as they happen
- You can't reset the board or try a different approach
- The user sees you "doing nothing" while waiting

**Wrong:**
```bash
# Blocking - you wait for the entire test to complete or timeout
pytest tests/test_stress.py -v --timeout=300 2>&1 | tee logs/run.log
```

**Right:**
```bash
# Background execution with monitoring
# Use Bash tool with run_in_background: true, then check output with tail/Read

# Start test in background
pytest tests/test_stress.py -v --timeout=300 2>&1 | tee logs/run.log &

# Check progress every 10-15 seconds
tail -50 logs/run.log

# If errors detected: cancel, analyze, reset board, try again
```

**The pattern:**
1. Start test in background
2. Wait 10-15 seconds
3. Check output file for errors/completion
4. React: continue waiting, cancel and fix, or celebrate success
5. Reset board after device errors before retrying

### 13. Device Contention with Parallel Workers ❌
**Wrong:** Using `pytest -n auto` with a fixture that creates a new device per test
**Right:** Understand that hardware tests with parallel workers need shared device management

With `-n auto`, pytest creates multiple worker processes. If each test tries to open the device:
- Multiple processes fight for the same physical hardware
- You get errors like `Read unexpected run_mailbox value`
- This is NOT the error you're trying to reproduce

**Solutions:**
1. **Run sequentially first** to verify the test works: `pytest test.py -x -v`
2. **Use session-scoped fixtures** if parallel execution is needed
3. **Loop iterations inside a single test** instead of parametrizing with parallel workers
4. **Match the original CI's approach** - check if CI uses `-n auto` with the same test

## Important Notes

- **⚠️ CREATE BASH RUNNER SCRIPT** - ALWAYS create `run_test.sh` that sets up environment, not just the Python test
- **⚠️ ALWAYS RUN TESTS IN BACKGROUND** - Use `run_in_background: true` in Bash tool, then monitor with `tail`/`Read`. Never block waiting for a test to complete.
- **⚠️ MONITOR OUTPUT EVERY 10-15 SECONDS** - Check logs for errors, don't wait for timeouts
- **⚠️ RESET BOARD AFTER DEVICE ERRORS** - Run `tt-smi -r` before retrying after device failures
- **⚠️ EVERY pytest command MUST save output to logs/** - Use `2>&1 | tee logs/run_N.log` - the user CANNOT see your terminal
- **Set environment variables in run_test.sh** - Extract ALL variables from CI YAML and logs
- **Match the original test's device setup** - Use the same fixture or initialization method as the failing test
- **Use `-x` flag to stop on first failure** - Once reproducing, don't waste time running more iterations
- **Don't guess** - Read the actual test code, YAML, and logs thoroughly
- **Isolate and simplify** - Don't reproduce the whole test suite
- **Be specific** - Target the exact failing operation and error type
- **Run sequential first** - Verify test works before adding `-n auto` parallel stress
- **Run sequential for clean logs** - After reproducing with parallel, run sequential to get clean stack traces
- **Make script executable** - Always `chmod +x run_test.sh`

## Success Criteria

Your stress test successfully reproduces the failure if:
1. **Has run_test.sh** - Bash script that sets up environment and runs test
2. It runs standalone with one command: `./run_test.sh`
3. It completes within 5 minutes (hard requirement)
4. It reproduces the failure within that 5 minute window
5. The failure mode matches the original error (same exception/message)
6. It can be run repeatedly for verification
7. Parameters can be tuned via command-line arguments to run_test.sh

## Example: Device Timeout Reproduction

**Scenario:** Test fails with `RuntimeError: TIMEOUT: device timeout, potential hang detected`

### ❌ WRONG Approach

```bash
# Missing environment variables
# Running full test suite
# No amplification
pytest tests/ttnn/unit_tests/operations/data_movement -v
```

**Problems:**
- No hang detection timeout set → takes forever or doesn't timeout
- Running entire test suite → takes > 5 minutes
- No amplification → low chance of reproducing rare issue
- Not isolated → can't tell what specific operation fails

### ✅ RIGHT Approach

**1. Extract from CI:**
```yaml
# From .github/workflows/*.yaml
env:
  hang-detection-timeout: 5
  ARCH_NAME: wormhole_b0
  LOGURU_LEVEL: INFO
```

**2. Analyze logs:**
```
RuntimeError: TT_THROW @ system_memory_manager.cpp:627
TIMEOUT: device timeout, potential hang detected
  at ttnn.to_torch(ttnn_gather)  ← Exact failure point
```

**3. Write isolated test:**
```python
# test_gather_device_timeout.py
import pytest
import torch
import ttnn

# NOTE: This example uses the `device` fixture because the original test did.
# Always check how the original test handles device setup and match it.

@pytest.mark.parametrize("iteration", range(50))
def test_gather_timeout(iteration, device):
    """Isolated test: just the failing gather + to_torch operation."""
    # The exact failing parameters from CI
    input_shape = [1, 151936]
    index_shape = [1, 151936]

    input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    index = torch.randint(0, 151936, index_shape, dtype=torch.int64)

    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint32, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.gather(ttnn_input, dim=-1, index=ttnn_index)
    result = ttnn.to_torch(ttnn_gather)  # ← This is where timeout happens
```

**4. Run with environment + amplification (SAVE OUTPUT TO LOGS!):**
```bash
# Set environment (CRITICAL for this timeout example!)
export TT_METAL_OPERATION_TIMEOUT_SECONDS=5  # Device operation timeout
export ARCH_NAME=wormhole_b0
export LOGURU_LEVEL=INFO
export TT_METAL_HOME=/tt-metal

source /opt/venv/bin/activate

# Run with parallel workers for amplification - SAVE OUTPUT TO LOGS!
# Use -x to stop on first failure (don't waste time after reproducing)
pytest test_gather_device_timeout.py -n auto -x -v --timeout=60 2>&1 | tee logs/stress_run_1.log

# After reproducing, run sequential for clean stack traces:
pytest test_gather_device_timeout.py -x -v --timeout=60 2>&1 | tee logs/sequential_run.log
```

**This approach:**
- ✅ Sets the relevant environment variables (timeout settings for this example)
- ✅ Isolates to just the failing operation
- ✅ Uses 50 iterations × multiple workers = high stress
- ✅ Uses `-x` to stop on first failure (saves time)
- ✅ Matches device setup from original test (fixture in this case)
- ✅ Saves ALL output to logs/ folder with `| tee logs/...`
- ✅ Completes in ~3-5 minutes
- ✅ Targets the exact error (device timeout during to_torch)

---

## Final Checklist Before Finishing

Before telling the user you're done, verify:

- [ ] **Created run_test.sh bash script** - Sets up environment and runs the test
- [ ] **Made run_test.sh executable** - `chmod +x run_test.sh`
- [ ] **Ran tests in BACKGROUND mode** - Used `run_in_background: true` and monitored output, didn't block waiting
- [ ] **Monitored output and reacted to errors** - Checked logs every 10-15 seconds, reset board after device errors
- [ ] **ALL test runs used `./run_test.sh 2>&1 | tee logs/<name>.log`** - The user CANNOT see your terminal. If you didn't save output, go back and re-run with logging.
- [ ] **Device setup matches original test** - Used same fixture or initialization method
- [ ] **Used `-x` flag** - Tests stop on first failure
- [ ] **Ran sequential first before parallel** - Verified test works before adding `-n auto`
- [ ] **Log files exist in logs/ folder** - Run `ls -la logs/` to verify files were created
- [ ] **README.md documents how to run** - Including `./run_test.sh` command
