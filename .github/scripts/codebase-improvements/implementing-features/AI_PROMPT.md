# AI Prompt: Implementing Features and Fixes

## Context

You are in the **Fix Implementation Phase**. A reproduction test has been created and confirmed to demonstrate the issue. Your task is to analyze the root cause, implement a fix, verify it works, and prepare for PR creation.

**Time Limit: 30 minutes**

If you cannot make meaningful progress within this time, document your findings and give up gracefully.

## 🚨 CRITICAL RULES - READ FIRST

1. **PROFILE FIRST** - Add timing instrumentation BEFORE attempting ANY fix
2. **TEST EXISTING CODE** - If there's an existing "fix" on the branch, TEST it first - don't assume it works
3. **NEVER MAKE UNIVERSAL CHANGES** - ALL optimizations must be CONDITIONAL (e.g., only for large transfers)
4. **VERIFY ASSUMPTIONS** - Log actual data sizes, iteration counts, etc. - don't guess
5. **DO NOT CREATE THE PR** - The user or orchestration script will do it
6. **ALWAYS build Metal** - Run `./build_metal.sh` after EVERY code change
7. **USE the bash script** - Run tests via `./run_test.sh`, NOT pytest directly
8. **USE /opt/venv** - The bash script activates it automatically
9. **DON'T DELETE TESTS** - Keep test files until fix is FULLY validated
10. **WRITE a report** - Document what you did in outputs/

## ⚠️ COMMON MISTAKES TO AVOID

Based on past failures, DO NOT:

❌ **Skip profiling** - "I can see from the stack trace where the bottleneck is"
   → Stack traces show WHERE timeout happens, not WHERE time is spent

❌ **Trust existing "fix" commits** - "There's already a fix on the branch, I'll build on it"
   → Test it first. Many "fixes" don't actually work.

❌ **Make universal changes** - `timeout = timeout * 10` or `sleep(100ms)`
   → Use conditional logic that only affects the problematic case

❌ **Guess at values** - "The tensor is X bytes" or "There are Y iterations"
   → Log the ACTUAL values. Your assumptions may be wrong by 10x or more.

❌ **Give up without proof** - "This is a fundamental hardware limitation"
   → Show specific code/evidence. What have you actually tried?

❌ **Optimize without measuring** - Spend hours tweaking code paths
   → Profile first. You might be optimizing the wrong thing.

## Input

You will receive:
- **Reproduction test path**: Location of the test that demonstrates the issue
- **User prompt**: Description of what needs to be fixed
- **Branch name**: Current development branch with the test
- **Failure logs**: Output from running the reproduction test

## Your Task (IN THIS ORDER)

1. **TEST the existing branch state** - Don't assume anything works
2. **PROFILE to find the bottleneck** - Add timing instrumentation FIRST
3. Analyze the root cause based on MEASURED data
4. Implement fixes iteratively (CONDITIONAL changes only!)
5. Test thoroughly using ./run_test.sh (2x for deterministic, 5x for non-deterministic)
6. Push the fix branch (keep test files for validation)
7. Write PR description for user
8. Write detailed execution report

## 🔬 MANDATORY FIRST STEP: PROFILING

**DO NOT SKIP THIS SECTION. Profile BEFORE attempting any fix.**

### Why Profiling is Mandatory

In past attempts, agents have:
- Spent hours optimizing the wrong code path
- Made incorrect assumptions about data sizes, iteration counts, etc.
- Given up claiming "fundamental limitations" without measuring
- Implemented fixes that didn't address the actual bottleneck

**Profiling reveals where time is actually spent - often different from what stack traces suggest.**

### Step 1: Add Python Timing Instrumentation

Add this to the test file BEFORE running it:

```python
import time

# At the start of test function:
t0 = time.perf_counter()

# After first operation (e.g., moving to device):
t1 = time.perf_counter()

# After main operation (e.g., gather):
t2 = time.perf_counter()

# After reading back (e.g., to_torch):
t3 = time.perf_counter()

# Log the breakdown:
print(f"\n{'='*60}")
print(f"TIMING BREAKDOWN:")
print(f"  to_device:  {(t1-t0)*1000:7.2f} ms")
print(f"  operation:  {(t2-t1)*1000:7.2f} ms  <<< is this the bottleneck?")
print(f"  to_torch:   {(t3-t2)*1000:7.2f} ms  <<< or is this?")
print(f"  TOTAL:      {(t3-t0)*1000:7.2f} ms")
print(f"{'='*60}\n")

# Also write to file for easy retrieval:
with open("/tmp/profiling_results.log", "a") as f:
    f.write(f"to_device={t1-t0:.3f}s, op={t2-t1:.3f}s, to_torch={t3-t2:.3f}s\n")
```

### Step 2: Add C++ Timing (if needed)

If the bottleneck is in C++ code, add timing there too:

```cpp
#include <chrono>
#include <tt_metal/common/logger.hpp>

// At start of function:
auto start = std::chrono::high_resolution_clock::now();

// After key operations:
auto checkpoint = std::chrono::high_resolution_clock::now();
auto duration_ms = std::chrono::duration<double, std::milli>(checkpoint - start).count();

// Log timing (add conditions to reduce noise if needed):
log_info(LogMetal, "operation took {:.2f}ms", duration_ms);
```

### Step 3: Log Actual Data Sizes

**CRITICAL**: Don't assume you know the transfer size. LOG IT:

```cpp
// In the transfer function:
log_info(LogMetal, "Transferring {} bytes ({:.2f} MB)", size, size / (1024.0 * 1024.0));
```

```python
# In the test:
print(f"Tensor size: {tensor.numel() * tensor.element_size()} bytes")
print(f"Shape: {tensor.shape}")
```

### Step 4: Run and Analyze

```bash
./run_test.sh 2>&1 | tee /tmp/profiling_run.log
cat /tmp/profiling_results.log
```

**ONLY AFTER you have timing data should you proceed to implement fixes.**

### Example: Why Profiling Matters

Before profiling, you might assume based on stack trace:
- "Error in function X = function X is slow"

After profiling:
```
TIMING BREAKDOWN:
  setup:       50 ms  (1%)
  operation:   30 ms  (1%)   <<< Stack trace pointed here, but it's FAST
  cleanup:   4920 ms  (98%)  <<< THIS is where time actually goes!
  TOTAL:     5000 ms
```

The stack trace showed where the *error* occurred, not where the *time* was spent.
Profiling prevents you from optimizing the wrong thing.

## CRITICAL CHECKLIST

### Before Starting Implementation

- [ ] **Checked for existing "fix" commits** on the branch
- [ ] **Tested current state** - Verified test actually fails (don't assume!)
- [ ] **Added profiling instrumentation** - Will measure where time is spent
- [ ] **You understand the expected vs actual behavior**
- [ ] **You have identified the failing operation/function**

### After Profiling (Before Writing Fix Code)

- [ ] **You have MEASURED timing data** - Know exactly where time is spent
- [ ] **You know ACTUAL data sizes** - Verified, not assumed
- [ ] **You have a hypothesis based on measurements** - Not stack trace guesses
- [ ] **You know what files need CONDITIONAL changes**

**If you don't have measured data for the above, STOP and profile first.**

### Implementation Principles

- [ ] **All changes will be CONDITIONAL** - Only affect specific cases
- [ ] **Will not make universal changes** - Won't slow down small operations
- [ ] **Will test after EACH change** - Build + run_test.sh

## Step-by-Step Process

### Phase 0: Test Existing State (MANDATORY - 2 min)

**DO NOT SKIP THIS PHASE.**

#### 0a. Check for Existing "Fix" Commits

```bash
# Check current branch
git log --oneline -10

# Look for existing fix attempts
git log --oneline --all | grep -i "fix\|optim\|timeout"
```

If there's already a "fix" commit on this branch or a related fix branch:
1. **DO NOT assume it works**
2. **TEST IT FIRST** before building on it

#### 0b. Test the Current State

```bash
# Build Metal first (CRITICAL)
cd /tt-metal
./build_metal.sh

# Navigate to test directory
cd <path-to-test-parent-directory>

# Run the test
./run_test.sh 2>&1 | tee current_state_test.txt
```

**Document the result:**
- Does the test PASS or FAIL?
- What is the exact error?
- How long does it take to fail?

**If the test PASSES:**
- The issue may already be fixed
- Verify by running 2-3 times
- Document and report this finding

**If the test FAILS:**
- Proceed to profiling (Phase 1)
- You now have a baseline

### Phase 1: Root Cause Analysis (5 min)

**IMPORTANT: This phase REQUIRES profiling. Do not skip to implementation.**

#### 1a. Add Profiling Instrumentation

**BEFORE running any analysis, add timing to the test file:**

```python
import time

def test_...(device):
    t0 = time.perf_counter()

    # ... existing setup code ...

    t1 = time.perf_counter()  # After setup/to_device

    # ... main operation ...

    t2 = time.perf_counter()  # After main operation

    # ... read back (to_torch) ...

    t3 = time.perf_counter()  # After read back

    # Log timing
    print(f"\n>>> PROFILING: setup={t1-t0:.3f}s, op={t2-t1:.3f}s, readback={t3-t2:.3f}s, total={t3-t0:.3f}s\n")

    # Write to file
    with open("/tmp/profiling.log", "a") as f:
        f.write(f"setup={t1-t0:.3f}, op={t2-t1:.3f}, readback={t3-t2:.3f}\n")
```

#### 1b. Run the Profiling Test

```bash
# Clear old logs
rm -f /tmp/profiling.log

# Run test with profiling
./run_test.sh 2>&1 | tee profiling_run.txt

# Check timing results
cat /tmp/profiling.log
grep -i "profiling\|timing\|ms\|seconds" profiling_run.txt
```

#### 1c. Analyze the Timing Data

**Document the breakdown:**
```markdown
## Profiling Results

| Phase | Time | % of Total |
|-------|------|------------|
| Setup/to_device | X.XX s | XX% |
| Main operation | X.XX s | XX% |
| Read back | X.XX s | XX% |
| **TOTAL** | X.XX s | 100% |

**Bottleneck**: The XXXXX phase takes XX% of time. This is where optimization should focus.
```

**ONLY AFTER you have this data should you form a hypothesis.**

#### 1d. Verify Data Sizes (Don't Guess!)

Add logging to verify actual transfer sizes:

```bash
# Search for where to add logging
grep -r "completion_queue\|transfer\|copy" tt_metal/impl/
```

If optimizing data transfer, you MUST know the actual size:
- Add `log_info` statements in C++ code
- Add `print(f"size={tensor.numel() * tensor.element_size()}")` in Python
- **DO NOT assume sizes based on tensor shapes without verification**

#### 1e. Analyze the Failure (Based on Profiling)

Now that you have MEASURED data, determine:

1. **What operation is the bottleneck?**
   - Look at your profiling results - which phase takes the most time?
   - The stack trace shows where *failure* occurred, not where *time* is spent

2. **Where specifically is time being spent?**
   - If profiling pointed to a particular phase, dig deeper
   - Add more granular timing if needed

3. **Why is it slow/failing?**
   - **Performance**: Too slow (timeout, benchmark miss)
   - **Correctness**: Wrong output (shape, values, exception)
   - **Resource**: OOM, device hang, buffer overflow

4. **What's the root cause?**
   - Inefficient algorithm
   - Missing optimization
   - Logic bug
   - Incorrect parameters
   - Resource limitation

#### 1f. Identify Files to Modify

Based on the profiling results, locate the relevant source files:

```bash
# Find operation implementation (replace <operation> with actual name)
grep -r "def <operation>" ttnn/
grep -r "class <Operation>" tt_eager/

# Find kernel code
find tt_metal/ ttnn/ -name "*<operation>*.cpp"

# Find dispatch/device code
find tt_metal/impl/dispatch/ -name "*.cpp"
```

**Create a list of candidate files** that likely need changes.

#### 1g. Form a Hypothesis

Document your hypothesis for the root cause:

```markdown
## Root Cause Hypothesis

**Issue**: <Brief description>

**Why it happens**: <Explanation>

**Expected fix**: <What needs to change>

**Files to modify**:
- `path/to/file1.cpp`: <What to change>
- `path/to/file2.cpp`: <What to change>

**Verification**: Test should pass after these changes
```

### Phase 2: Create Fix Branch (1 min)

#### 2a. Create Branch Off Main

```bash
# Save current branch name
OLD_BRANCH=$(git branch --show-current)

# Create fix branch from latest main
git checkout main
git pull origin main
git checkout -b fix/<descriptive-name>

# Example: fix/gather-wide-tensor-timeout
# Example: fix/llama70b-prefill-performance
```

Branch naming:
- `fix/<issue>` - For bug fixes
- `opt/<feature>` - For optimizations
- `feat/<feature>` - For new features

#### 2b. Copy Reproduction Test

The test should already be on this branch (copied by run.sh script).
Verify it exists:

```bash
ls -la <path-to-test-directory>/
```

If not present, the script should have copied it. Check git log.

#### 2c. Build Metal (CRITICAL)

**Before running ANY tests, you MUST build Metal:**

```bash
# Navigate to Metal root
cd /tt-metal

# Build Metal (required for all tests)
./build_metal.sh

# This takes ~5-10 minutes
# Wait for it to complete before proceeding
```

**DO NOT:**
- ❌ Run cmake commands directly
- ❌ Use build_python_venv (wrong venv)
- ❌ Skip the build step

**DO:**
- ✅ Use `./build_metal.sh`
- ✅ Use `/opt/venv/bin/activate` for venv
- ✅ Wait for build to complete

#### 2d. Verify Test Fails on New Branch

**Use the bash script runner (not pytest directly):**

```bash
# Navigate to test directory
cd <path-to-test-parent-directory>

# Run using the bash script
./run_test.sh 2>&1 | tee baseline_failure.txt

# Check the output
tail -50 baseline_failure.txt
```

**Expected**: Test should FAIL with the error from raw-logs

If test PASSES unexpectedly:
- Check if you're on the right branch
- Verify Metal was built
- Document this and STOP

### Phase 3: Implement Fix (10 min)

#### 3a. CRITICAL: Make CONDITIONAL Changes Only

**🚨 NEVER make UNIVERSAL changes that affect ALL operations.**

Any optimization MUST be conditional based on clear criteria that identify the problematic case.

**❌ BAD - Universal change:**
```cpp
// This affects EVERYTHING - unacceptable!
auto timeout = base_timeout * 10;
```

**✅ GOOD - Conditional change:**
```cpp
// Only affects the specific problematic case
auto timeout = base_timeout;
if (condition_that_identifies_the_problem) {
    timeout = base_timeout * 10;  // Extended only for problematic case
}
```

**❌ BAD - Unconditional slowdown:**
```cpp
std::this_thread::sleep_for(std::chrono::milliseconds(100));  // ALL operations now slower
```

**✅ GOOD - Conditional optimization:**
```cpp
if (is_problematic_case) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
// Normal cases unaffected
```

#### Why Conditional Changes Matter

Universal changes can:
- Slow down ALL operations (not just the problematic ones)
- Mask the real issue (e.g., increasing timeout hides performance problems)
- Break other tests that depend on original behavior
- Make the codebase harder to maintain

#### 3b. Make Small, Targeted Changes

**DO NOT** make large, sweeping changes. Instead:

1. **Start with the most likely fix based on profiling data**
2. **Make ONE change at a time**
3. **Test after EACH change**
4. **Ensure changes are CONDITIONAL** - only affect the problematic case
5. **Document what each change does and WHY**

**Example: Conditional Fix**

```cpp
// Before: All cases use same behavior
while (!done) {
    check_status();
    process();
}

// After: Problematic cases get special handling
while (!done) {
    check_status();
    if (is_problematic_case) {
        // Special handling for the case that was failing
        process_with_workaround();
    } else {
        // Normal cases keep original behavior
        process();
    }
}
```

**Example: Bug Fix**

```python
# Before: Incorrect dimension calculation
output_shape = [batch, seq_len // 2, hidden]  # Bug: halves sequence

# After: Correct dimension
output_shape = [batch, seq_len, hidden]  # Fix: preserves sequence
```

#### 3b. Test After Each Change

After EVERY code change, you MUST:

1. **Rebuild Metal**:
```bash
cd /tt-metal
./build_metal.sh
```

2. **Run test using bash script**:
```bash
# Navigate to test directory
cd <path-to-test-parent-directory>

# Run using bash script (not pytest directly!)
./run_test.sh 2>&1 | tee test_run_1.txt

# Check result
tail -50 test_run_1.txt
```

**CRITICAL:**
- ❌ DO NOT run `pytest` directly
- ❌ DO NOT skip the build step
- ❌ DO NOT use build_python_venv
- ✅ USE `./build_metal.sh` after each change
- ✅ USE `./run_test.sh` to run tests
- ✅ USE `/opt/venv` (activated by run_test.sh)

**Track your attempts**:
- `test_run_1.txt` - First attempt
- `test_run_2.txt` - Second attempt
- etc.

#### 3c. Iterate Until Test Passes

Keep making changes until:
1. Test passes consistently
2. Error is resolved
3. Performance meets requirements

**If stuck after 3-4 attempts:**
- Re-examine your hypothesis
- Try a different approach
- Consider giving up and documenting findings

#### 3d. Verify Stability (MANDATORY)

**CRITICAL**: You MUST verify the fix works reliably before finishing.

Once test passes, verify it's stable:

```bash
# Navigate to test directory
cd <path-to-test-parent-directory>

# For DETERMINISTIC failures: Run test 2 times
# For NON-DETERMINISTIC failures: Run test 5 times
RUNS=2  # Use 2 for deterministic, 5 for non-deterministic

for i in $(seq 1 $RUNS); do
    echo "========== Run $i =========="
    ./run_test.sh 2>&1 | tee verify_run_${i}.txt

    # Check result
    if [ $? -ne 0 ]; then
        echo "❌ Test failed on run $i"
        break
    fi
    echo "✅ Test passed run $i"
done
```

For performance fixes, measure consistency:
```bash
# Measure performance (2x for deterministic, 5x for non-deterministic)
for i in $(seq 1 $RUNS); do
    echo "Run $i:"
    ./run_test.sh 2>&1 | grep -E "samples/s|duration|time|PASSED|FAILED"
done
```

**Success criteria (deterministic failures):**
- ✅ Test passes 2/2 times (deterministic should be consistent)
- ✅ Error is resolved both times
- ✅ No device errors or hangs

**Success criteria (non-deterministic failures):**
- ✅ Test passes 5/5 times (higher bar for race conditions/intermittent issues)
- ✅ No intermittent failures
- ✅ Stress test runs without triggering the original failure

**How to know which you have:**
Check the task file - it will say "deterministic: true" or "deterministic: false"

**If any run fails:**
- ❌ DO NOT proceed to PR
- ❌ Fix is not stable
- 🔄 Debug and try again

### Phase 4: Prepare for PR (2 min)

#### 4a. Clean Up Commits

Review your commits:
```bash
git log --oneline -10
```

**Remove the reproduction test**:
```bash
# The test stays on the old branch, not in the PR
git rm path/to/test_repro.py
git rm run_test.sh 2>/dev/null || true  # Remove bash script if present
git commit -m "Remove reproduction test (kept on dev branch)"
```

**CRITICAL: Verify PR contents are clean**:
```bash
# Check what will be in the PR (should only be source files)
git diff main...HEAD --name-only

# Should NOT include:
# - info.json
# - .github/scripts/ files
# - test files
# - run_test.sh

# Should ONLY include:
# - Modified source files (*.cpp, *.hpp, *.py in main codebase)
```

**Clean up any debug code**:
- Remove print statements
- Remove temporary hacks
- Remove commented-out code

#### 4b. Create Well-Documented Commits

If you have messy commits, squash them:
```bash
git rebase -i HEAD~5  # Interactive rebase last 5 commits
# Squash into logical commits
```

Each commit should:
- Have a clear, descriptive message
- Explain WHY the change was made
- Reference the issue if applicable

**Good commit message:**
```
Optimize completion queue read for wide tensors

The existing implementation copied data element-by-element, which
was inefficient for wide tensors (>100k elements). This change
uses bulk memcpy for better performance.

Fixes timeout in gather operations with shape [1, 151936].
```

### Phase 5: Document Your Work (2 min)

**CRITICAL: DO NOT CREATE THE PR**

The user or the orchestration script will create the PR after you finish.
Your job is to:
1. ✅ Implement and test the fix
2. ✅ Push the branch
3. ✅ Write a report describing what you did
4. ❌ DO NOT run `gh pr create`

#### 5a. Push Branch

```bash
git push origin fix/<descriptive-name>
```

#### 5b. Write PR Description (for the user to use)

Create a file with the PR description that the user can use:

```bash
cat > /tmp/pr_description.md <<'EOF'
## Title
<Short description of fix>

## Body
## Summary

Brief description of the issue and the fix.

## Root Cause

Explanation of what was wrong and why it failed.

## Changes

- **file1.cpp**: Description of changes
- **file2.py**: Description of changes

## Performance Impact

- **Before**: <measurement>
- **After**: <measurement>
- **Improvement**: <percentage>

## Testing

Reproduction test verified fix works:
- Test: `path/to/test_repro.py`
- Results: Passed 5/5 runs after fix
- Build: Rebuilt Metal with ./build_metal.sh

## Recommended CI Workflows

- [x] `all-post-commit` (required)
- [ ] `<specific-workflow-1>`
- [ ] `<specific-workflow-2>`

## Relevant Developers

@developer1 @developer2

---

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF

echo "PR description written to /tmp/pr_description.md"
echo "Branch pushed to: fix/<descriptive-name>"
echo ""
echo "User can create PR with:"
echo "gh pr create --draft --base main --head fix/<descriptive-name> --title \"...\" --body-file /tmp/pr_description.md"
```

**DO NOT run gh pr create yourself!** The user or script will do it.

#### 5c. Add Recommended Workflows

Based on what you changed, recommend specific CI workflows:

**Always include:**
- `all-post-commit`

**Add based on changes:**
- Modified TTNN ops → `ttnn-unit-tests`
- Modified models → `model-perf-tests`
- Modified device code → `device-tests`
- Modified dispatch → `dispatch-tests`
- Performance changes → `perf-tests`

### Phase 6: Write Execution Report (2 min)

Create a markdown file in `outputs/` with timestamp and description:

**Filename**: `outputs/YYYY-MM-DD_HH-MM-SS_<short-desc>.md`

**Template**:

```markdown
# Fix Implementation Report: <Short Description>

**Generated**: YYYY-MM-DD HH:MM:SS
**Status**: ✅ Success / ⚠️ Partial / ❌ Failed
**Duration**: X minutes

---

## Summary

Brief summary of what was attempted and the outcome.

## Input

- **Reproduction Test**: `path/to/test_repro.py`
- **User Prompt**: "<original prompt>"
- **Failure Type**: Performance/Correctness/Timeout/etc.

## Root Cause Analysis

### What Failed
<Description of the failing operation>

### Why It Failed
<Explanation of the root cause>

### Hypothesis
<Your initial hypothesis>

## Implementation

### Approach
<Description of the fix approach>

### Files Modified
1. **path/to/file1.cpp**
   - Change: <what was changed>
   - Reason: <why it was needed>

2. **path/to/file2.py**
   - Change: <what was changed>
   - Reason: <why it was needed>

### Iterations
- **Attempt 1**: <what you tried> → Result: <pass/fail>
- **Attempt 2**: <what you tried> → Result: <pass/fail>
- **Final**: <successful approach> → Result: ✅ Pass

## Verification

### Test Results
```
<Output from successful test runs>
```

### Performance Impact
- **Before**: <measurement>
- **After**: <measurement>
- **Improvement**: <percentage or absolute change>

### Stability
- Ran test 5 times: <N passes / N runs>
- Consistent behavior: Yes/No

## Pull Request

**Status**: Created / Not Created
**Link**: https://github.com/tenstorrent/tt-metal/pull/XXXXX
**Branch**: fix/<name>
**Base**: main

### PR Details
- **Title**: <PR title>
- **Recommended Workflows**:
  - all-post-commit
  - <workflow-1>
  - <workflow-2>

## Next Steps

### If Successful
- [ ] Review PR and address comments
- [ ] Run recommended CI workflows
- [ ] Merge after approval

### If Partial
- [ ] <What still needs to be done>
- [ ] <Known limitations>

### If Failed
- [ ] <Why it failed>
- [ ] <What to try next>
- [ ] <Who to consult>

## Relevant Developers

Based on the area of change:
- **Primary**: @developer1 (area expertise)
- **Secondary**: @developer2 (code owner)
- **Reviewer**: @developer3 (similar fixes)

## Technical Details

### Stack Trace (Original Failure)
```
<Relevant parts of stack trace>
```

### Key Code Changes
```cpp
// Example of main change
<Before and after code>
```

## Lessons Learned

- <What worked well>
- <What didn't work>
- <Recommendations for future>

---

**Report Generated by Claude Sonnet 4.5**
```

## Failure Modes and How to Handle Them

### 1. Cannot Reproduce on New Branch

**Symptoms**: Test passes on fix branch but failed on dev branch

**Causes**:
- Branch-specific issue
- Environment differences
- Test not accurate

**Action**:
- Document this in report
- Status: ❌ Failed
- Next step: Manual investigation needed

### 2. Cannot Identify Root Cause

**Symptoms**: Multiple attempts, no clear cause

**Time limit**: 3 minutes of analysis

**Action**:
- Document what you found
- List possible causes
- Status: ❌ Failed - Needs expert analysis
- Recommend developers to consult

### 3. Fix Doesn't Work

**Symptoms**: Test still fails after changes

**Time limit**: 5-6 attempts over 8 minutes

**Action**:
- Try alternative approaches
- If no progress after 6 attempts, give up
- Document all attempts
- Status: ❌ Failed - Could not fix
- Explain blockers

### 4. Fix Breaks Other Things

**Symptoms**: Build fails, other tests fail, crashes

**Action**:
- Revert changes
- Try more targeted fix
- If still breaking, give up
- Status: ⚠️ Partial - Fix causes regressions
- Document side effects

### 5. Fix Is Incomplete

**Symptoms**: Test passes sometimes, performance still below target

**Action**:
- Document what works and what doesn't
- Create PR with current progress
- Status: ⚠️ Partial - Improvement but not complete
- List remaining work

## Time Management (Total: 30 min)

**Strict timeline:**

| Phase | Time | Cumulative | Notes |
|-------|------|------------|-------|
| Test existing state | 2 min | 2 min | Check for existing fixes |
| Build Metal (initial) | 5 min | 7 min | Required before any tests |
| **PROFILING** | 5 min | 12 min | **MANDATORY - DO NOT SKIP** |
| Root cause analysis | 3 min | 15 min | Based on profiling data |
| Implement fix (iterative) | 10 min | 25 min | CONDITIONAL changes only |
| Verify stability | 2-5 min | 30 min | 2x deterministic, 5x non-deterministic |

**Verify stability time**:
- Deterministic: 2 runs × ~1 min = 2 min
- Non-deterministic: 5 runs × ~1 min = 5 min

**Note**: Build time can vary. Each iteration requires rebuild (~5 min).
**At 30 minutes, STOP regardless of status** and write the report.

**Profiling is NON-NEGOTIABLE**: The 5 minutes for profiling will SAVE time by ensuring you optimize the right thing. Skipping it leads to hours of wasted effort.

**Realistically**: Expect 2-3 fix iterations, so ~25-35 minutes total.

## Success Criteria

A successful implementation should:
1. ✅ **Profiled FIRST** - Have timing data showing where time is spent
2. ✅ **Tested existing state** - Verified the baseline behavior
3. ✅ **Changes are CONDITIONAL** - Only affect specific cases (e.g., large transfers)
4. ✅ Reproduction test passes reliably using ./run_test.sh
   - Deterministic: 2/2 runs pass
   - Non-deterministic: 5/5 runs pass
5. ✅ Metal rebuilt after each code change (./build_metal.sh)
6. ✅ Changes are well-documented in commits
7. ✅ Fix branch pushed to origin
8. ✅ PR description written for user (in /tmp/pr_description.md)
9. ✅ Recommended CI workflows listed
10. ✅ Execution report written to outputs/ (including profiling data)
11. ✅ No obvious regressions introduced
12. ❌ **DID NOT create PR directly** (user/script does this)
13. ❌ **DID NOT delete test files** until fix is fully validated

## Giving Up Gracefully - STRICT REQUIREMENTS

**DO NOT give up easily.** In past attempts, agents claimed "fundamental limitations" without evidence, then found fixes when pushed to continue.

### Before You Can Give Up, You MUST Have:

**✅ ALL of these requirements must be met:**

1. **Profiled the operation** - You have MEASURED timing data showing exactly where time is spent
2. **Verified assumptions** - You have LOGGED actual data sizes, iteration counts, etc.
3. **Tested at least 5 different approaches** - Not just variations, but fundamentally different strategies
4. **Searched alternative code paths** - Looked for DMA, async, streaming, or other mechanisms
5. **Examined BOTH host AND device code** - Not just one side
6. **Documented WHY each approach failed** - With specific evidence, not speculation
7. **Spent at least 20 minutes actively trying** - Not just analyzing

**❌ You CANNOT give up if:**

- You haven't profiled yet
- You've only tried 2-3 approaches
- You're assuming something is a "hardware limitation" without evidence
- You've only looked at host code (or only device code)
- You're guessing at bottleneck locations instead of measuring

### Evidence Required for "Fundamental Limitation" Claims

If you claim something is a hardware/firmware limitation:

1. **Show the specific code** that cannot be optimized
2. **Show the measurement** proving where time is spent
3. **Explain why** no software change can improve it
4. **Provide the math** - actual throughput vs theoretical limits

**Example of VALID "fundamental limitation" evidence:**
```
Profiling shows:
- Device writes 9.27 MB at 1.22 MB/s
- PCIe theoretical limit: 4 GB/s
- Actual throughput is 3000x below hardware capability

Tried:
1. Reduced polling frequency → No improvement
2. Batched NOC writes → No improvement
3. Increased burst size → Crashes (hardware limit)
4. Removed synchronization waits → Crashes (required for correctness)
5. Async DMA transfer → Not available in this code path

CONCLUSION: Device-side write speed is the bottleneck, not host or software.
The 1.22 MB/s suggests device firmware issue, not software optimization opportunity.
```

**Example of INVALID "fundamental limitation" claim:**
```
The timeout appears to be fundamental to how wide tensors are handled.
This issue is deeper than a simple optimization.
```
(This has no evidence, no measurements, no specific code references)

### Give-Up Checklist

Before writing a failure report, verify:

- [ ] **I have profiling data** showing time breakdown
- [ ] **I know the actual data sizes** (not guesses)
- [ ] **I tried at least 5 approaches** with documented results
- [ ] **I examined device-side code** (not just host)
- [ ] **I searched for alternative APIs/mechanisms**
- [ ] **I can cite specific code** that cannot be changed
- [ ] **I spent at least 20 minutes** on active implementation attempts

### If Requirements Not Met

If you cannot check all boxes above:
- **DO NOT give up**
- Continue investigating
- Ask for more time if needed
- Try more approaches

### Example give-up report (With Required Evidence):

```markdown
## Status: ❌ Failed - With Evidence

### Profiling Data (REQUIRED)
```
TIMING BREAKDOWN:
  setup:      0.05s (0.6%)
  gather:     0.02s (0.2%)
  to_torch:   8.62s (99.2%)  <<< BOTTLENECK IDENTIFIED
```

### Verified Data Sizes (REQUIRED)
- Actual transfer: 9.27 MB (not 300KB as stack trace suggested)
- Chunks: 1159 × 8KB = 9.27 MB
- Throughput: 1.22 MB/s

### What Was Tried (5+ approaches REQUIRED)
1. ✅ Reduced polling sleep (10µs → 100µs → 1ms) → No improvement
2. ✅ Batched NOC flushes (every 8 chunks) → No improvement
3. ✅ Moved flush outside loop → Test crashes (buffer overflow)
4. ✅ Increased NOC burst size → Test crashes (hardware limit)
5. ✅ Conditional timeout scaling → Test passes but doesn't fix speed
6. ✅ Pre-buffering sleep → Slight improvement but still slow

### Evidence of Limitation
The device writes at 1.22 MB/s. PCIe theoretical is 4 GB/s.
This 3000x gap suggests hardware/firmware issue, not software.

Specific code at `cq_common.hpp:130` uses `CQ_NOC_WAIT` which MUST wait
(removing it causes crashes - attempt #3). This serialization appears
required for correctness.

### Recommended Developers
- **@dispatch-lead**: Why is device write so slow?
- **@device-expert**: Is this a firmware limitation?
```

## Important Notes

### Methodology
- **PROFILE FIRST** - This is non-negotiable. 5 minutes of profiling saves hours of guessing.
- **TEST existing code** - Don't assume existing "fixes" work. Verify first.
- **MEASURE, don't guess** - Log actual data sizes, don't assume from tensor shapes.
- **CONDITIONAL changes only** - Never make universal changes that affect all operations.

### Giving Up
- **DON'T give up easily** - Past agents claimed "fundamental limitations" without evidence, then found fixes when pushed.
- **Require evidence** - If claiming something is unfixable, show the code, measurements, and math.
- **Try 5+ approaches** - Not variations of the same approach, but fundamentally different strategies.
- **Examine both sides** - Check device code, not just host code (or vice versa).

### Process
- **Work methodically** - 30 minutes is enough if you profile first
- **Document as you go** - Don't wait until the end
- **Always create the report** - Even if you give up (with required evidence)
- **Keep test files** - Don't delete until fix is fully validated
- **Include profiling data** - In all reports

### PR Preparation
- **DO NOT create PR directly** - User/script will do it
- **Include relevant developers** - Tag appropriate reviewers
- **Write clear PR description** - Save to /tmp/pr_description.md

## Final Checklist

Before finishing, verify:

### Methodology Requirements
- [ ] **Tested existing state FIRST** - Didn't assume existing code works
- [ ] **Added profiling instrumentation** - Have timing data
- [ ] **Know where bottleneck is** - Based on MEASURED data, not assumptions
- [ ] **Verified data sizes** - Logged actual transfer sizes

### Implementation Requirements
- [ ] **All changes are CONDITIONAL** - Only affect specific cases (e.g., large transfers)
- [ ] **No universal slowdowns** - Small operations still fast
- [ ] **Built Metal with ./build_metal.sh** after all changes
- [ ] **Ran test using ./run_test.sh** (not pytest directly)
   - [ ] Deterministic: 2/2 passes
   - [ ] Non-deterministic: 5/5 passes
- [ ] **Test passed all runs** (or documented failure with required evidence)

### Documentation Requirements
- [ ] **Fix branch pushed** to origin
- [ ] **PR description written** to /tmp/pr_description.md
- [ ] **DID NOT create PR** (user/script will do it)
- [ ] **DID NOT delete test files** (kept for validation)
- [ ] **Execution report written** to outputs/
- [ ] **Report includes profiling data** showing time breakdown
- [ ] **Report includes** relevant developer contacts

### If Giving Up
- [ ] **Profiled the operation** - Have timing data
- [ ] **Tried at least 5 approaches** - Documented each
- [ ] **Examined both host AND device code**
- [ ] **Have specific evidence** for why fix isn't possible
- [ ] **Spent at least 20 minutes** on active attempts

### Time
- [ ] **Total time** <= 30 minutes
