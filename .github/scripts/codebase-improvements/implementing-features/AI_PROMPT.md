# AI Prompt: Implementing Features and Fixes

## Context

You are in the **Fix Implementation Phase**. A reproduction test has been created and confirmed to demonstrate the issue. Your task is to analyze the root cause, implement a fix, verify it works, and prepare for PR creation.

**Time Limit: 15 minutes**

If you cannot make meaningful progress within this time, document your findings and give up gracefully.

## 🚨 CRITICAL RULES - READ FIRST

1. **DO NOT CREATE THE PR** - The user or orchestration script will do it
2. **ALWAYS build Metal** - Run `./build_metal.sh` after EVERY code change
3. **USE the bash script** - Run tests via `./run_test.sh`, NOT pytest directly
4. **USE /opt/venv** - The bash script activates it automatically
5. **TEST thoroughly** - Run test 5 times before declaring success
6. **PUSH the branch** - So user can create PR
7. **WRITE a report** - Document what you did in outputs/

## Input

You will receive:
- **Reproduction test path**: Location of the test that demonstrates the issue
- **User prompt**: Description of what needs to be fixed
- **Branch name**: Current development branch with the test
- **Failure logs**: Output from running the reproduction test

## Your Task

1. Analyze the root cause of the failure (read raw-logs for error details)
2. Verify test fails on the fix branch (build Metal first!)
3. Implement fixes iteratively (build after each change!)
4. Test thoroughly using ./run_test.sh (5 successful runs minimum)
5. Remove test from branch (stays on dev branch)
6. Push the fix branch
7. Write PR description for user
8. Write detailed execution report

## CRITICAL CHECKLIST

Before starting, verify:

- [ ] **Reproduction test runs and fails consistently** on current branch
- [ ] **You understand the expected vs actual behavior**
- [ ] **You have identified the failing operation/function**
- [ ] **You know what files likely need changes**
- [ ] **You have a hypothesis for the root cause**

**If you don't have clear answers to the above, STOP and analyze more.**

## Step-by-Step Process

### Phase 1: Root Cause Analysis (2 min)

#### 1a. Run the Reproduction Test

```bash
# Set required environment variables (from test docs)
export TT_METAL_HOME=/tt-metal
export PYTHONPATH=/tt-metal
export ARCH_NAME=wormhole_b0
source /opt/venv/bin/activate

# Run the test to see the failure
cd <test-directory>
pytest test_*_repro.py -v -s 2>&1 | tee failure_log.txt
```

#### 1b. Analyze the Failure

From the test output and code, determine:

1. **What operation fails?**
   - Is it a TTNN operation? (`ttnn.gather`, `ttnn.to_torch`)
   - Is it a model forward pass?
   - Is it a kernel execution?

2. **Where does it fail?**
   - Extract the stack trace
   - Identify the exact function/line
   - Understand the call chain

3. **Why does it fail?**
   - **Performance**: Too slow (timeout, benchmark miss)
   - **Correctness**: Wrong output (shape, values, exception)
   - **Resource**: OOM, device hang, buffer overflow

4. **What's the root cause?**
   - Inefficient algorithm
   - Missing optimization
   - Logic bug
   - Incorrect parameters
   - Resource limitation

#### 1c. Identify Files to Modify

Based on the failure, locate the relevant source files:

```bash
# Find operation implementation
grep -r "def gather" ttnn/
grep -r "class Gather" tt_eager/

# Find kernel code
find tt_metal/ ttnn/ -name "*gather*.cpp"

# Find dispatch/device code
find tt_metal/impl/dispatch/ -name "*.cpp"
```

**Create a list of candidate files** that likely need changes.

#### 1d. Form a Hypothesis

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

### Phase 3: Implement Fix (8 min)

#### 3a. Make Small, Targeted Changes

**DO NOT** make large, sweeping changes. Instead:

1. **Start with the most likely fix**
2. **Make ONE change at a time**
3. **Test after EACH change**
4. **Document what each change does**

**Example: Performance Fix**

```cpp
// Before: Inefficient memory copy
void copy_data(uint32_t* src, uint32_t* dst, size_t size) {
    for (size_t i = 0; i < size; i++) {
        dst[i] = src[i];  // Slow element-by-element
    }
}

// After: Optimized bulk copy
void copy_data(uint32_t* src, uint32_t* dst, size_t size) {
    memcpy(dst, src, size * sizeof(uint32_t));  // Fast bulk operation
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

## Time Management (Total: 15 min)

**Strict timeline:**

| Phase | Time | Cumulative |
|-------|------|------------|
| Root cause analysis | 2 min | 2 min |
| Build Metal (initial) | 5 min | 7 min |
| Implement fix (iterative) | 15 min | 22 min |
| Verify stability | 2-5 min | 24-27 min |
| Prepare branch & docs | 3 min | 27-30 min |

**Verify stability time**:
- Deterministic: 2 runs × ~1 min = 2 min
- Non-deterministic: 5 runs × ~1 min = 5 min

**Note**: Build time can vary. Each iteration requires rebuild (~5 min).
**At 30 minutes, STOP regardless of status** and write the report.

**Realistically**: Expect 2-3 fix iterations, so ~20-30 minutes total.

## Success Criteria

A successful implementation should:
1. ✅ Reproduction test passes reliably using ./run_test.sh
   - Deterministic: 2/2 runs pass
   - Non-deterministic: 5/5 runs pass
2. ✅ Metal rebuilt after each code change (./build_metal.sh)
3. ✅ Changes are well-documented in commits
4. ✅ Fix branch pushed to origin
5. ✅ PR description written for user (in /tmp/pr_description.md)
6. ✅ Recommended CI workflows listed
7. ✅ Execution report written to outputs/
8. ✅ No obvious regressions introduced
9. ❌ **DID NOT create PR directly** (user/script does this)

## Giving Up Gracefully

If you need to give up:

1. **Document everything you tried**
2. **Explain what didn't work and why**
3. **Provide your best hypothesis**
4. **List what would be needed to succeed**
5. **Recommend specific developers/experts**
6. **Write a complete report with Status: Failed**

**Example give-up report:**

```markdown
## Status: ❌ Failed

### What Was Tried
1. Optimized completion queue read → Still times out
2. Increased buffer size → OOM error
3. Changed dispatch path → Test hangs
4. Reduced timeout threshold → Different error

### Why It Failed
The timeout appears to be fundamental to how wide tensors are handled
in the dispatch system. The issue is deeper than a simple optimization.

### What's Needed
- Expert analysis of dispatch architecture
- Possibly redesign of completion queue for wide tensors
- May need hardware team input on buffer limitations

### Recommended Next Steps
1. Consult @dispatch-expert for architecture review
2. Profile the exact bottleneck in completion queue
3. Consider if this is a hardware limitation

### Recommended Developers
- **@dispatch-lead**: Completion queue architecture
- **@device-expert**: Hardware buffer limits
- **@perf-team**: Wide tensor optimization expertise
```

## Important Notes

- **Work quickly but carefully** - 15 minutes is tight
- **Don't overthink** - Make reasonable changes and test
- **Document as you go** - Don't wait until the end
- **Give up if stuck** - Better to document failure than waste time
- **Always create the report** - Even if you give up
- **Remove test from PR** - It stays on dev branch only
- **Use draft PRs** - Always create as draft first
- **Include relevant developers** - Tag appropriate reviewers

## Final Checklist

Before finishing, verify:

- [ ] **Built Metal with ./build_metal.sh** after all changes
- [ ] **Ran test 5/5 times using ./run_test.sh** (not pytest directly)
- [ ] **Test passed all 5 runs** (or documented failure)
- [ ] **Reproduction test removed** from PR branch
- [ ] **Fix branch pushed** to origin
- [ ] **PR description written** to /tmp/pr_description.md
- [ ] **DID NOT create PR** (user/script will do it)
- [ ] **Execution report written** to outputs/
- [ ] **Report includes** relevant developer contacts
- [ ] **Total time** <= 15 minutes
