# Implementing Features

This directory contains prompts and workflows for implementing fixes and features after a reproduction test has been created.

## Purpose

Once you have a reproduction test (from reproduce-deterministic-failures or reproduce-ND-failures), this workflow helps you:
1. Analyze the root cause
2. Implement a fix
3. Verify the fix works
4. Create a PR with the changes

## When to Use This

Use this workflow when:
- You have a working reproduction test
- You need to fix the underlying issue
- You want to optimize performance
- You need to implement a feature

## Workflow Overview

### Input Requirements

Before starting, you must have:
- A reproduction test that reliably demonstrates the issue
- Clear understanding of expected vs actual behavior
- Access to the codebase on a development branch

### Phase 1: Analysis (2 min)

1. **Run the reproduction test** to confirm the issue
2. **Analyze the failure**:
   - What operation is failing?
   - Where in the codebase does it fail?
   - What's the root cause? (logic bug, performance issue, missing optimization)
3. **Identify files to modify**:
   - Core implementation files
   - Related utilities or helpers
   - Build configuration if needed

### Phase 2: Fix Development (8 min)

1. **Create fix branch** off latest main:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b fix/<descriptive-name>
   ```

2. **Copy reproduction test** to the new branch:
   ```bash
   git cherry-pick <commit-with-test>
   # Or manually copy the test file
   ```

3. **Implement fixes iteratively**:
   - Make small, targeted changes
   - Run test after each change
   - Verify test passes reliably
   - Document what each change does

4. **Verify stability**:
   - Run test multiple times (3-5 times)
   - For performance fixes: verify improvement is consistent
   - For bug fixes: verify test passes 100%

### Phase 3: PR Creation (5 min)

1. **Prepare commits**:
   - Remove the reproduction test from the branch (it stays on dev branch)
   - Clean up any debug code
   - Ensure commits are well-documented

2. **Create draft PR**:
   ```bash
   git push origin fix/<descriptive-name>
   gh pr create --draft --base main --head fix/<descriptive-name>
   ```

3. **Write PR description** with:
   - Summary of the issue
   - Root cause explanation
   - Description of the fix
   - Performance impact (if applicable)
   - Recommended CI workflows to run

4. **Tag relevant reviewers**

## Fix Strategies

### Performance Optimization

**Goal**: Make operation faster/more efficient

**Approach**:
1. Profile the slow operation
2. Identify bottleneck (memory, compute, dispatch)
3. Apply targeted optimizations:
   - Reduce memory transfers
   - Improve kernel efficiency
   - Optimize data layout
   - Use faster dispatch paths

**Verification**:
- Measure before/after performance
- Ensure improvement meets requirements
- Check for regressions in other tests

### Bug Fixes

**Goal**: Fix incorrect behavior

**Approach**:
1. Understand why the bug occurs
2. Find the incorrect logic/calculation
3. Implement fix
4. Add validation if needed

**Verification**:
- Test passes consistently
- Edge cases handled
- No new bugs introduced

### Feature Implementation

**Goal**: Add new functionality

**Approach**:
1. Understand requirements
2. Design the implementation
3. Add new code/modify existing
4. Add proper error handling

**Verification**:
- Feature works as expected
- Existing functionality not broken
- Performance acceptable

## PR Description Template

```markdown
## Summary

Brief description of what was changed and why.

## Root Cause

Explanation of the underlying issue that needed fixing.

## Changes

- **File 1**: Description of changes
- **File 2**: Description of changes

## Performance Impact

- Before: X ms/samples per second
- After: Y ms/samples per second
- Improvement: Z%

## Testing

- [ ] Reproduction test passes consistently
- [ ] No regressions in related tests
- [ ] Performance improvement verified

## Recommended CI Workflows

- `all-post-commit` (required)
- `<specific-workflow-1>`
- `<specific-workflow-2>`

## Reviewers

@developer1 @developer2
```

## Time Management

**Total: 15 minutes for fix + PR**

- 0-2 min: Analyze root cause
- 2-10 min: Implement and test fix
- 10-15 min: Create PR and documentation

If you cannot make progress within this timeframe:
- Document what you tried
- Explain the blocker
- Recommend next steps

## Success Criteria

A successful fix should:
1. Make the reproduction test pass consistently
2. Not break existing functionality
3. Be well-documented in code and PR
4. Include recommended CI workflows
5. Have clear performance impact (if applicable)

## Failure Modes

### Cannot Reproduce Issue

If the test doesn't fail on the new branch:
- The issue may be branch-specific
- Environment might be different
- Test might not be accurate

**Action**: Document this and recommend manual investigation

### Cannot Find Root Cause

If the cause is unclear after analysis:
- The issue might be too complex
- More expertise needed
- Insufficient information

**Action**: Document findings, recommend experts to consult

### Fix Doesn't Work

If changes don't make test pass:
- Approach might be wrong
- Issue might be elsewhere
- Multiple issues present

**Action**: Try alternative approaches, document attempts

### Fix Breaks Other Things

If changes cause regressions:
- Fix too aggressive
- Side effects not considered
- Need more targeted approach

**Action**: Revert problematic changes, try narrower fix

## Developer Contacts

Based on the area of the fix, relevant developers to tag:

### Performance Issues
- @developer-perf-1
- @developer-perf-2

### TTNN Operations
- @developer-ops-1
- @developer-ops-2

### Device/Dispatch
- @developer-device-1
- @developer-device-2

### Models
- @developer-models-1
- @developer-models-2

*Note: Update this list based on your team structure*

## Examples

### Example 1: Gather Timeout Fix

**Issue**: `ttnn.to_torch()` times out on wide tensors `[1, 151936]`

**Root Cause**: Completion queue read operation doesn't handle wide tensors efficiently

**Fix Approach**:
1. Optimize `copy_completion_queue_data_into_user_space()`
2. Increase buffer size for wide tensor reads
3. Add timeout handling for edge cases

**Files Modified**:
- `tt_metal/impl/dispatch/system_memory_manager.cpp`
- `tt_metal/impl/dispatch/command_queue.cpp`

**Verification**: Test passes in <5 seconds (was timing out at 5s)

### Example 2: Llama 70B Performance

**Issue**: Prefill time exceeds benchmark (400ms target, getting 450ms)

**Root Cause**: Attention kernel not optimized for large context

**Fix Approach**:
1. Profile attention operation
2. Optimize kernel grid configuration
3. Improve memory access patterns

**Files Modified**:
- `ttnn/cpp/ttnn/operations/transformer/attention.cpp`
- `ttnn/cpp/ttnn/operations/transformer/kernels/attention_kernel.cpp`

**Verification**: Prefill time reduced to 380ms (5% improvement over target)

## Integration with run.sh

This workflow is automatically invoked by `run.sh` after reproduction test is created:

```bash
# run.sh calls implementing-features workflow
cd implementing-features
# AI reads AI_PROMPT.md
# AI follows the workflow
# AI creates PR
# AI writes report
```

See `AI_PROMPT.md` for the detailed AI instructions.
