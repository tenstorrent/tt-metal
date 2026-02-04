# AI Prompt: Reproduce Deterministic Failures

## Context

You are tasked with creating a minimal reproduction test for a deterministic failure from CI logs. The failure happens every time, so your goal is to create the simplest possible test that reproduces it for debugging.

## Directory Structure

```
.github/scripts/reproduce-deterministic-failures/
  <failure-name>/
    logs/          # Contains CI error logs (created by user)
    tests/         # You will write tests here
```

## Input

The user has created a subfolder named `<failure-name>` with:
- `logs/` - Contains error logs from CI failures
- `tests/` - Empty folder where you will write a minimal reproduction test

## Your Task

Create a standalone minimal test in the `tests/` folder that reproduces the failure immediately and reliably.

## CRITICAL CHECKLIST

Before writing any code, ensure you have:

- [ ] Found and read the **exact CI YAML configuration** for the failing job
- [ ] Extracted **ALL environment variables** from YAML and logs
- [ ] Located the **exact line of code** where the failure occurs
- [ ] Identified the **failure type**: assertion, exception, crash, shape mismatch, etc.
- [ ] Understood the **expected vs actual behavior**
- [ ] Determined the **minimal failing operation** (don't reproduce the whole test)
- [ ] Verified your **device setup matches the original test**
- [ ] Listed what **CANNOT be replicated** locally and how to mock it

**If you haven't done ALL of the above, STOP and go back.**

## Step-by-Step Process

### 1. Analyze the Error Logs

Read all logs in `logs/` and identify:
- **What failed?** (assertion, exception type, error message)
- **Where did it fail?** (exact line, function, stack trace)
- **What was the error?** (expected vs actual values)
- **What test was running?** (test name, parameters)

### 2. Find and Analyze the Original Test

#### 2a. Extract Test Information from Logs

From the error logs, identify:
- Test name (e.g., `test_bert_attention`, `test_gather_negative_index`)
- Test file path (e.g., `tests/ttnn/unit_tests/operations/test_gather.py`)
- Test parameters/configuration used
- Command that ran the test

#### 2b. Find the Test File

Search for the test in the codebase:
```bash
# Search by test name
grep -r "def test_bert_attention" tests/

# Or use glob to find test files
find tests/ -name "*bert*.py"
```

**Read the entire test file** to understand:
- What the test does
- Setup/teardown logic
- Parameters and configuration
- Dependencies (imports, fixtures, helper functions)

#### 2c. Find the YAML Configuration

Look for workflow files that show how CI ran the test:
- `.github/workflows/*.yaml` - Workflow definitions
- `tests/scripts/*.yaml` - Test configurations

**Extract from YAML:**
- Environment variables set for the test
- Hardware requirements (device type, number of chips)
- Test command and arguments
- Any special setup steps

### 3. Analyze CI Environment

#### 3a. Find the Exact CI Job Configuration

Search for the workflow YAML file that ran the failing test:
```bash
grep -r "job-name-from-logs" .github/workflows/
```

**Read the job definition** and extract:

1. **All Environment Variables**:
   - `PYTHONPATH`, `LD_LIBRARY_PATH`, `TT_METAL_HOME`
   - `ARCH_NAME`, `LOGURU_LEVEL`
   - Any custom environment variables

2. **Test Execution Details**:
   - Exact pytest command (including ALL flags)
   - Test markers (`-m "..."`)
   - Any special pytest plugins or configurations

3. **Pre-test Steps**:
   - Setup scripts or initialization
   - Build commands
   - Environment preparation

#### 3b. Search Logs for Hidden Configuration

Look for configuration that's not in YAML files:
```bash
# Look for environment variable dumps
grep -i "env\|export\|ARCH\|TT_METAL" logs/*.log | head -50

# Look for pytest invocation
grep -i "pytest" logs/*.log | head -20
```

#### 3c. Identify What CANNOT Be Replicated

**Be realistic about what you can't replicate locally:**
- Mount points that don't exist
- Specific Docker images or containers
- Multiple physical devices (if you only have one)
- Specific hardware configurations

**For each thing you can't replicate:**
1. **Mock it**: Set dummy values that won't affect the failure
2. **Skip it**: If it's not relevant to the failure
3. **Simplify it**: Replace complex setup with minimal equivalent

### 4. Isolate the Minimal Failing Operation

**CRITICAL: Create the SMALLEST possible test that reproduces the failure.**

#### 4a. Identify the Exact Failing Line

From the stack trace, determine exactly what operation fails:

```python
# Example from logs:
# AssertionError: Tensor shape mismatch
#   Expected: torch.Size([32, 128])
#   Got: torch.Size([32, 64])
#   at result = model.forward(input) ← Fails here
```

#### 4b. Strip Away Everything Unnecessary

Remove everything not directly related to the failure:
- Complex test fixtures → Simple setup
- Multiple test variants → Just the failing case
- Validation logic → Just the assertion that fails
- Logging/debugging → Minimal output
- Multiple operations → Just the one that fails

#### 4c. Create Focused Test

Your test should:
1. **Run ONLY the failing operation**
2. **Fail immediately** (within seconds)
3. **Have clear error message** matching the original
4. **Be completely standalone** (no external dependencies if possible)

### 5. Identify the Root Cause

Based on the error, determine what's wrong:
- **Logic bug**: Incorrect calculation or condition
- **Shape mismatch**: Tensor dimensions don't match
- **Type error**: Wrong dtype or data structure
- **Index error**: Out of bounds access
- **Assertion failure**: Expected behavior not met
- **Missing validation**: Edge case not handled

Document your hypothesis in the test docstring.

### 6. Set Up Environment Variables

Create a clear list of required environment variables:

```bash
#!/bin/bash
# Required environment variables

# Core paths
export PYTHONPATH="/tt-metal"
export LD_LIBRARY_PATH="/tt-metal/build/lib"
export TT_METAL_HOME="/tt-metal"

# Architecture (from CI job name or logs)
export ARCH_NAME="wormhole_b0"  # or grayskull, blackhole, etc.

# Logging
export LOGURU_LEVEL="INFO"

# Any other variables from CI YAML or logs
```

### 7. Write the Minimal Reproduction Test

**CRITICAL: You must create TWO files:**
1. `test_<descriptive_name>_repro.py` - The Python test
2. `run_test.sh` - Bash script to set up environment and run the test

#### 7a. Create the Bash Runner Script

**File naming:** `run_test.sh`

**This is MANDATORY** - The bash script is the primary way to run the test.

```bash
#!/bin/bash

# Runner script for <test name> reproduction
# Sets up environment and runs the test

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "<Test Name> Reproduction Test"
echo "=========================================="
echo ""

# Set ALL required environment variables
echo "Setting up environment..."

# Example: Set timeout if needed
export TT_METAL_OPERATION_TIMEOUT_SECONDS=5

# Architecture
export ARCH_NAME=${ARCH_NAME:-wormhole_b0}

# Metal paths
export TT_METAL_HOME=${TT_METAL_HOME:-/tt-metal}
export PYTHONPATH=${PYTHONPATH:-/tt-metal}

# Logging
export LOGURU_LEVEL=${LOGURU_LEVEL:-INFO}

# Add any other environment variables from CI
# export SOME_CUSTOM_VAR=value

echo "  TT_METAL_HOME=$TT_METAL_HOME"
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
echo "Running test..."
echo "=========================================="
echo ""

# Change to test directory
cd "$SCRIPT_DIR/tests"

# Run the test
pytest test_<name>_repro.py -x -v --timeout=60 "$@"

TEST_RESULT=$?

echo ""
echo "=========================================="
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ Test PASSED"
else
    echo "❌ Test FAILED (exit code: $TEST_RESULT)"
fi
echo "=========================================="

exit $TEST_RESULT
```

**Make it executable:**
```bash
chmod +x run_test.sh
```

#### 7b. Create the Python Test

**File naming:** `tests/test_<descriptive_name>_repro.py`

**Design Principles:**
1. **Minimal**: Absolute minimum code to reproduce
2. **Clear**: Obvious what's being tested
3. **Fast**: Runs in seconds
4. **Standalone**: Can be run independently (via run_test.sh)
5. **Documented**: Clear explanation of the failure

**Code structure:**
```python
"""
Minimal reproduction for: <failure description>

Original failure: <job name> - <date>
Error: <exact error message>

This test reproduces the exact failure with minimal setup.

Expected behavior: <what should happen>
Actual behavior: <what actually happens>

Run with:
    cd <failure-folder>
    ./run_test.sh
"""

import pytest
import torch
import ttnn

def test_<name>_repro(device):
    """
    Minimal reproduction of <failure>.

    This test isolates the exact failing operation from the original test.
    It should fail with: <expected error message>
    """
    # Minimal setup - only what's needed to reproduce
    input_tensor = torch.randn([32, 128], dtype=torch.bfloat16)

    # The exact failing operation
    ttnn_input = ttnn.from_torch(input_tensor, device=device)
    result = ttnn.operation_that_fails(ttnn_input)  # ← Fails here

    # This assertion should fail with the original error
    assert result.shape == torch.Size([32, 128]), f"Expected [32, 128], got {result.shape}"
```

**Must include:**
- Clear docstring with instructions to use run_test.sh
- Comment marking the exact failing line
- Expected vs actual behavior
- Minimal setup (no unnecessary code)
- Clear assertion that reproduces the error

### 8. Verify Reproduction

**Run the test using the bash script:**

```bash
# Simply run the script - it handles everything
cd <failure-name>
./run_test.sh 2>&1 | tee logs/reproduction.log
```

The bash script will:
- Set all environment variables
- Activate the virtual environment
- Run the test
- Report pass/fail status

**Verify:**
- Does it fail with the same error as CI?
- Does the error message match?
- Does it fail at the expected line?
- Can you run it repeatedly and get the same failure?

**If not reproducing:**
1. Check the environment variables in run_test.sh - are they all correct?
2. Verify you're testing the exact same parameters as CI
3. Check device setup in the Python test - does it match the original?
4. Review logs for missing configuration
5. Try running manually to isolate environment vs test issues

### 9. Document

Create a README in the `<failure-name>` folder:

**File:** `<failure-name>/README.md`

```markdown
# <Failure Name>

## Original Failure

- **Job**: <CI job name>
- **Date**: <when it failed>
- **Error**: <exact error message>

## Root Cause

<Explanation of what's wrong and why it fails>

## Minimal Reproduction

The test in `tests/test_<name>_repro.py` reproduces this failure by:
- <Brief explanation of what the test does>

### Run Test

```bash
# Activate virtual environment and set variables
source /opt/venv/bin/activate
export TT_METAL_HOME=/tt-metal
export PYTHONPATH=/tt-metal
export ARCH_NAME=wormhole_b0

# Run the reproduction test
cd tests
pytest test_<name>_repro.py -v
```

### Expected Output

The test should fail with:
```
<exact error message expected>
```

## Fix Strategy

<Suggestions for how to fix this issue>

## Verification

After fixing, this test should pass. Run:
```bash
pytest test_<name>_repro.py -v
```

Expected output: `PASSED`
```

## Common Mistakes to AVOID

### 1. Including Unnecessary Code ❌
**Wrong:** Copying the entire original test with all its setup
**Right:** Strip down to the absolute minimum needed to reproduce

### 2. Missing Environment Variables ❌
**Wrong:** Running test without setting CI environment variables
**Right:** Extract ALL variables from YAML and logs and document them

### 3. Not Matching Device Setup ❌
**Wrong:** Using different device initialization than the original test
**Right:** Match EXACTLY how the original test creates/obtains the device

**Read the original test** to see how it handles devices:
- Does it use a `device` fixture? → Use that fixture
- Does it call `ttnn.open_device()` directly? → Do the same
- Does it use `MeshDevice`? → Match it

### 4. Testing Multiple Things ❌
**Wrong:** Including multiple operations when only one fails
**Right:** Isolate to just the single failing operation

### 5. Complex Test Logic ❌
**Wrong:** Loops, conditionals, multiple test cases in one test
**Right:** Single, linear execution path that hits the failure

### 6. Poor Documentation ❌
**Wrong:** No explanation of what's being tested or why
**Right:** Clear docstring with expected vs actual behavior

### 7. Not Creating Bash Runner Script ❌
**Wrong:** Only creating the Python test file
**Right:** Create both `run_test.sh` AND `test_*_repro.py`

The bash script is MANDATORY because:
- Sets up environment variables automatically
- Activates virtual environment
- Can be run from any directory
- Works in CI without manual setup
- Documents all required configuration

### 8. Not Saving Test Output ❌
**Wrong:** `pytest test_repro.py -v` - output lost if terminal closes
**Right:** `./run_test.sh 2>&1 | tee logs/reproduction.log`

### 9. Ignoring Error Messages ❌
**Wrong:** Creating a test that fails differently than the original
**Right:** Verify the error message and stack trace match exactly

### 10. Over-Engineering ❌
**Wrong:** Adding CLI args, configuration, multiple test variants
**Right:** Single test function with hardcoded values that reproduce the issue

### 11. Not Testing the Fix ❌
**Wrong:** Creating reproduction but not verifying the fix works
**Right:** After fixing, re-run the same test to confirm it passes

## Important Notes

- **Create bash runner** - ALWAYS create run_test.sh, not just the Python test
- **Keep it simple** - The goal is easy debugging, not comprehensive testing
- **Match the original** - Device setup, environment, parameters must match
- **Save all output** - Use `2>&1 | tee logs/...` for every command
- **Document clearly** - Others should understand the failure immediately
- **Fast iteration** - Test should run in seconds for quick debugging
- **Verify the fix** - Same test should pass after the issue is resolved
- **Make script executable** - Always `chmod +x run_test.sh`

## Success Criteria

Your reproduction test is successful if:
1. **Has run_test.sh** - Bash script that sets up environment and runs test
2. It runs standalone with one command: `./run_test.sh`
3. It fails within seconds with the same error as CI
4. The error message and stack trace match the original
5. The Python test is under 50 lines of code (ideally under 30)
6. It can be easily modified to test potential fixes
7. After fixing the bug, the same test passes

## Example: Shape Mismatch

**Scenario:** Test fails with `AssertionError: Expected shape [32, 128], got [32, 64]`

### ❌ WRONG Approach

```python
# test_complex_repro.py - TOO COMPLEX
import pytest
import torch
import ttnn
from models.bert import BertModel
from utils.data_loader import load_test_data

@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("seq_len", [64, 128, 256])
def test_bert_all_configs(batch_size, seq_len, device):
    """Test all BERT configurations"""
    config = load_config("bert_base.yaml")
    model = BertModel(config)
    data = load_test_data(batch_size, seq_len)

    for layer in model.layers:
        output = layer(data)
        validate_output(output)
```

**Problems:**
- Tests multiple configurations (only one fails)
- Includes unnecessary model loading
- Complex setup with external dependencies
- Takes minutes to run
- Hard to debug which specific operation fails

### ✅ RIGHT Approach

**File 1: `run_test.sh`**

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "BERT Attention Shape Mismatch Test"
echo "=========================================="

# Set environment
export TT_METAL_HOME=${TT_METAL_HOME:-/tt-metal}
export PYTHONPATH=${PYTHONPATH:-/tt-metal}
export ARCH_NAME=${ARCH_NAME:-wormhole_b0}

# Activate venv
if [ -f "/opt/venv/bin/activate" ]; then
    source /opt/venv/bin/activate
fi

echo "Running test..."
cd "$SCRIPT_DIR/tests"
pytest test_bert_attention_shape_repro.py -v "$@"

TEST_RESULT=$?
echo "=========================================="
[ $TEST_RESULT -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED"
exit $TEST_RESULT
```

**File 2: `tests/test_bert_attention_shape_repro.py`**

```python
"""
Minimal reproduction for: BERT attention output shape mismatch

Original failure: bert-tests / test_bert_attention[batch32-seq128]
Error: AssertionError: Expected shape [32, 128], got [32, 64]

The attention layer incorrectly halves the sequence dimension.

Expected: Output shape should match input shape [32, 128]
Actual: Output shape is [32, 64] (sequence dim halved)

Run with:
    ./run_test.sh
"""

import pytest
import torch
import ttnn

def test_bert_attention_shape_repro(device):
    """
    Minimal reproduction of BERT attention shape mismatch.

    The attention operation should preserve sequence length but instead
    outputs half the expected sequence dimension.
    """
    batch_size = 32
    seq_len = 128
    hidden_dim = 768

    # Minimal input
    input_tensor = torch.randn([batch_size, seq_len, hidden_dim], dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(input_tensor, device=device)

    # The exact failing operation - attention with num_heads=12
    output = ttnn.scaled_dot_product_attention(
        ttnn_input, ttnn_input, ttnn_input,
        num_heads=12
    )  # ← Output shape is wrong here

    result = ttnn.to_torch(output)

    # This assertion fails: expected [32, 128, 768], got [32, 64, 768]
    expected_shape = torch.Size([batch_size, seq_len, hidden_dim])
    assert result.shape == expected_shape, (
        f"Shape mismatch: expected {expected_shape}, got {result.shape}"
    )
```

**Run and verify:**
```bash
# Make executable
chmod +x run_test.sh

# Run - should fail with shape mismatch
./run_test.sh 2>&1 | tee logs/repro.log
```

**This approach:**
- ✅ Has bash runner script (run_test.sh)
- ✅ Sets up environment automatically
- ✅ Minimal Python code (under 50 lines)
- ✅ Single failing operation isolated
- ✅ Clear expected vs actual documented
- ✅ Runs in seconds with one command
- ✅ Easy to modify for testing fixes
- ✅ Saves output to logs
- ✅ Matches device setup from original test
- ✅ Works in any environment (local, CI, etc.)

---

## Final Checklist Before Finishing

Before telling the user you're done, verify:

- [ ] **Test reproduces the exact same error as CI**
- [ ] **Error message and stack trace match**
- [ ] **Test is minimal** (under 50 lines ideally)
- [ ] **Environment variables documented**
- [ ] **Device setup matches original test**
- [ ] **Output saved to logs/** - Used `2>&1 | tee logs/...`
- [ ] **README.md documents the failure and fix strategy**
- [ ] **Test runs in seconds, not minutes**
- [ ] **Clear documentation** of expected vs actual behavior
- [ ] **Verification plan** for testing the fix
