# AI Prompt: Implementing Features and Fixes

## Context

You are in the **Fix Implementation Phase**. A reproduction test has been created and confirmed to demonstrate the issue. Your task is to analyze the root cause, implement a fix, verify it works, and create a PR.

**Time Limit: 15 minutes**

If you cannot make meaningful progress within this time, document your findings and give up gracefully.

## Input

You will receive:
- **Reproduction test path**: Location of the test that demonstrates the issue
- **User prompt**: Description of what needs to be fixed
- **Branch name**: Current development branch with the test
- **Failure logs**: Output from running the reproduction test

## Your Task

1. Analyze the root cause of the failure
2. Create a fix branch off main
3. Copy the reproduction test to the new branch
4. Implement fixes iteratively
5. Verify the test passes reliably
6. Create a draft PR (excluding the test)
7. Write a detailed report

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

**IMPORTANT**: Cherry-pick or copy the test from the old branch:

```bash
# Find the commit with the test
git log $OLD_BRANCH --oneline | head -5

# Cherry-pick the test commit
git cherry-pick <commit-hash>

# Or manually copy
git checkout $OLD_BRANCH -- path/to/test_repro.py
git add path/to/test_repro.py
git commit -m "Add reproduction test for <issue>"
```

#### 2c. Verify Test Fails on New Branch

```bash
# Run test to confirm it still fails
pytest path/to/test_repro.py -v -s 2>&1 | tee baseline_failure.txt
```

If test PASSES on new branch:
- Issue might be branch-specific
- Environment might be different
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

After EVERY code change:

```bash
# Quick test run
pytest path/to/test_repro.py -v -x 2>&1 | tee test_run_1.txt

# Check result
tail -20 test_run_1.txt
```

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

#### 3d. Verify Stability

Once test passes, verify it's reliable:

```bash
# Run test 5 times
for i in {1..5}; do
    echo "Run $i"
    pytest path/to/test_repro.py -v -x || break
done
```

For performance fixes:
```bash
# Measure performance multiple times
for i in {1..3}; do
    pytest path/to/test_repro.py -v -s | grep "samples/s\|duration\|time"
done
```

All runs should pass consistently.

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
git commit -m "Remove reproduction test (kept on dev branch)"
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

### Phase 5: Create Draft PR (2 min)

#### 5a. Push Branch

```bash
git push origin fix/<descriptive-name>
```

#### 5b. Create Draft PR

```bash
gh pr create \
  --draft \
  --base main \
  --head fix/<descriptive-name> \
  --title "<Short description of fix>" \
  --body "$(cat <<'EOF'
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

Reproduction test created on branch `<old-branch-name>` at:
- Path: `path/to/test_repro.py`
- Status: Passes reliably after fix

## Recommended CI Workflows

- [x] `all-post-commit` (required)
- [ ] `<specific-workflow-1>`
- [ ] `<specific-workflow-2>`

## Relevant Developers

@developer1 @developer2

---

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

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
| Create fix branch | 1 min | 3 min |
| Implement fix (iterative) | 8 min | 11 min |
| Prepare for PR | 2 min | 13 min |
| Create PR | 2 min | 15 min |

**At 15 minutes, STOP regardless of status** and write the report.

## Success Criteria

A successful implementation should:
1. ✅ Reproduction test passes reliably (5/5 runs)
2. ✅ Changes are well-documented
3. ✅ Draft PR created with clear description
4. ✅ Recommended CI workflows listed
5. ✅ Execution report written
6. ✅ No obvious regressions introduced

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

- [ ] Reproduction test passed 5/5 times after fix (or documented failure)
- [ ] Commits are clean and well-documented
- [ ] Reproduction test removed from PR branch
- [ ] Draft PR created (or failure documented)
- [ ] PR description includes recommended workflows
- [ ] Execution report written to outputs/
- [ ] Report includes relevant developer contacts
- [ ] Total time <= 15 minutes
