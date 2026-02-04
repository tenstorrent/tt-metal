# Example Usage

This document shows practical examples of using the automated fix implementation system.

## Example 1: Fix Deterministic Timeout (Using Existing Test)

We have an existing reproduction test for the wide tensor timeout issue. This example shows how to skip reproduction and go straight to fix implementation.

### Step 1: Configure info.json

```json
{
  "deterministic": true,
  "url": "",
  "prompt": "The gather operation with wide tensors [1, 151936] times out during ttnn.to_torch(). The completion queue read is too slow for wide tensors. Optimize the underlying copy operation.",
  "raw-logs": "",
  "existing-test-path": "reproduce-deterministic-failures/timeout-in-datamovement/tests/test_gather_timeout_stress.py"
}
```

By providing `existing-test-path`, the script automatically skips the reproduction phase.

### Step 2: Run

```bash
cd .github/scripts/codebase-improvements
./run.sh
```

### Step 3: What Happens

1. **Reproduction Phase**: ✅ **SKIPPED** - Using existing test
2. **Fix Implementation**: AI analyzes the test, then optimizes `system_memory_manager.cpp` and `command_queue.cpp`
3. **PR Creation**: Draft PR with optimized completion queue read
4. **Report**: Details in `outputs/YYYY-MM-DD_HH-MM-SS_gather-timeout.md`

**Time Saved**: ~5 minutes by skipping reproduction

### Expected Changes

```cpp
// Before: Element-by-element copy
for (uint32_t i = 0; i < size; i++) {
    dst[i] = src[i];
}

// After: Bulk memcpy
memcpy(dst, src, size * sizeof(uint32_t));
```

## Example 2: Optimize Model Performance (From CI URL)

### Step 1: Configure info.json

```json
{
  "deterministic": true,
  "url": "https://github.com/tenstorrent/tt-metal/actions/runs/12345678",
  "prompt": "Llama 70B prefill time is 450ms but needs to be under 400ms to meet benchmark. The test is passing but slow. Optimize the attention operations to reduce prefill time by at least 50ms.",
  "raw-logs": ""
}
```

### Step 2: Run

```bash
./run.sh
```

### Step 3: What Happens

1. **Fetch Logs**: Downloads logs from GitHub Actions run
2. **Create Test**: Creates a performance test that measures prefill time and fails if >400ms
3. **Analyze**: Profiles attention operations to find bottlenecks
4. **Optimize**: Implements kernel optimizations or changes grid configuration
5. **Verify**: Runs test 5 times to confirm consistent <400ms
6. **PR**: Creates draft PR with optimizations

### Expected Report Section

```markdown
## Performance Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Prefill time | 450ms | 380ms | -70ms (-15.5%) |
| Meets target | ❌ No | ✅ Yes | +50ms margin |
```

## Example 3: Fix Non-Deterministic Race Condition (Raw Logs)

### Step 1: Collect Logs

Save failing CI logs to a file, then configure:

```json
{
  "deterministic": false,
  "url": "",
  "prompt": "Reduce scatter operation occasionally hangs on T3K with 8 chips. The failure is intermittent (fails ~20% of runs). Find and fix the race condition in the CCL synchronization.",
  "raw-logs": "... paste entire log here including stack traces ..."
}
```

### Step 2: Run

```bash
./run.sh
```

### Step 3: What Happens

1. **Create Stress Test**: Uses `reproduce-ND-failures` workflow
   - Parametrized test with 50+ iterations
   - Parallel execution with `pytest -n auto`
   - Amplifies race condition to fail within 5 minutes

2. **Analyze**: Identifies synchronization issue in CCL barrier

3. **Fix**: Adds proper synchronization primitives

4. **Verify**: Runs stress test 5 times with no failures

### Expected Test

```python
@pytest.mark.parametrize("iteration", range(50))
def test_reduce_scatter_race_stress(iteration, mesh_device):
    """Stress test that amplifies race condition in reduce scatter."""
    # Run reduce scatter with minimal delays
    # Parallel workers increase race condition probability
    result = ttnn.reduce_scatter(...)
    assert result is not None  # Would hang/timeout on race
```

## Example 4: Implement New Feature

### Step 1: Configure

```json
{
  "deterministic": true,
  "url": "",
  "prompt": "Add support for dynamic shapes in the gather operation. Currently only static shapes are supported, but we need to handle shape inference at runtime. Create a test that verifies dynamic shape support, then implement it.",
  "raw-logs": ""
}
```

### Step 2: What Happens

1. **Create Test**: Test that calls gather with dynamic shapes (fails with NotImplementedError)
2. **Implement**: Adds dynamic shape inference to gather operation
3. **Verify**: Test passes with various dynamic shapes
4. **PR**: Draft PR with new feature

## Example 5: Debug Cryptic Failure

### Step 1: Configure with Raw Logs

```json
{
  "deterministic": true,
  "url": "",
  "prompt": "CI test 'test_bert_large_inference' fails with 'Read unexpected run_mailbox value: 0x123abc'. Determine root cause and fix.",
  "raw-logs": "
=========================== FAILURES ===========================
_______ test_bert_large_inference[batch32-seq512] ________

... stack trace ...
RuntimeError: Read unexpected run_mailbox value: 0x123abc
Expected: 0x0
Location: tt_metal/impl/dispatch/dispatch_core.cpp:456
... more logs ...
"
}
```

### Step 2: AI Investigation

1. **Create Reproduction**: Minimal test that triggers mailbox error
2. **Analyze**: Investigates dispatch state, device synchronization
3. **Root Cause**: Identifies race in dispatch core initialization
4. **Fix**: Adds proper barrier before mailbox read
5. **Verify**: Test passes reliably

## Handling Failures

### When Reproduction Fails

If AI cannot create a working reproduction test:

**Report Output:**
```markdown
## Status: ❌ Failed - Could Not Reproduce

### Attempts Made
1. Created test based on stack trace - No failure
2. Added parameters from CI config - Still passes
3. Tried with different device setup - Still passes

### Why It Failed
- Issue may be environment-specific (CI container)
- May require specific device state from previous tests
- Logs might not contain complete information

### Recommended Next Steps
1. Provide more detailed logs from CI
2. Check if issue is hardware-specific
3. Try running the full test suite sequence
4. Consult @test-expert for environment differences

### Developers to Contact
- @ci-infra - CI environment questions
- @device-team - Device setup questions
```

### When Fix Fails

If AI cannot implement a working fix:

**Report Output:**
```markdown
## Status: ❌ Failed - Could Not Fix

### Root Cause Identified
Device timeout occurs in completion queue read for wide tensors.

### Attempts Made
1. Optimized copy operation → Still times out
2. Increased buffer size → OOM error
3. Changed dispatch path → Different error
4. Reduced tensor size → Works, but not a real fix

### Why It Failed
The issue appears to be fundamental to the dispatch architecture.
Wide tensor support may require redesigning completion queue handling.

### Recommended Next Steps
1. Consult @dispatch-architect about completion queue redesign
2. Consider if hardware limitation exists
3. May need to profile at lower level (kernel traces)
4. Might require multi-stage fix across several PRs

### Developers to Contact
- @dispatch-lead - Completion queue expert
- @perf-team - Low-level profiling
- @hardware-team - Buffer limitations
```

## Testing Individual Phases

You can test each phase independently:

### Test Reproduction Only

```bash
cd reproduce-deterministic-failures
# Manually create failure folder and logs
# Invoke Claude with AI_PROMPT.md
```

### Test Fix Implementation Only

```bash
# Assuming you have a reproduction test
cd implementing-features
# Create .impl_task.md with test path
# Invoke Claude with AI_PROMPT.md
```

### Test Full Workflow

```bash
cd codebase-improvements
./run.sh
# Follow prompts for manual Claude invocation
```

## GitHub Actions Integration

### Basic Workflow

```yaml
name: Auto-Fix Failure

on:
  workflow_dispatch:
    inputs:
      prompt:
        description: 'What to fix'
        required: true
      deterministic:
        description: 'Is failure deterministic?'
        required: true
        type: boolean
      ci_run_url:
        description: 'CI run URL with logs'
        required: false

jobs:
  auto-fix:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Configure info.json
        run: |
          cd .github/scripts/codebase-improvements
          cat > info.json <<EOF
          {
            "deterministic": ${{ inputs.deterministic }},
            "url": "${{ inputs.ci_run_url }}",
            "prompt": "${{ inputs.prompt }}",
            "raw-logs": ""
          }
          EOF

      - name: Run automated fix
        run: |
          cd .github/scripts/codebase-improvements
          ./run.sh

      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: fix-report
          path: .github/scripts/codebase-improvements/outputs/*.md
```

### Triggered on Repeated Failures

```yaml
name: Auto-Fix on Repeated Failure

on:
  workflow_run:
    workflows: ["CI Tests"]
    types: [completed]

jobs:
  check-failures:
    if: ${{ github.event.workflow_run.conclusion == 'failure' }}
    runs-on: ubuntu-latest
    steps:
      - name: Check if failure is repeated
        # Logic to check if same test failed multiple times

      - name: Trigger auto-fix
        if: repeated_failure
        # Configure and run ./run.sh
```

## Tips for Success

### 1. Provide Detailed Prompts

**Bad:**
```json
{"prompt": "Fix the test"}
```

**Good:**
```json
{"prompt": "The gather operation times out with wide tensors [1, 151936]. The completion queue read is slow. Optimize the copy operation in system_memory_manager.cpp to use bulk transfer instead of element-wise copy."}
```

### 2. Include Relevant Logs

Make sure logs contain:
- Full stack trace
- Error messages
- Test parameters
- Environment variables
- Device information

### 3. Set Realistic Expectations

- Simple fixes: 15-20 minutes total
- Complex issues: May not complete in time limit
- Architecture changes: May need multiple iterations

### 4. Review AI Output

- Always review the generated test
- Verify the fix makes sense
- Check for potential side effects
- Run additional tests if needed

### 5. Iterate if Needed

If first attempt fails:
- Update prompt with more context
- Provide additional logs
- Try more specific fix suggestions
- Consider breaking into smaller tasks

## Monitoring and Metrics

Track automation success:

```bash
# Count successful runs
grep "Status: ✅ Success" outputs/*.md | wc -l

# Count failed runs
grep "Status: ❌ Failed" outputs/*.md | wc -l

# Find recent PRs created
grep "PR URL" outputs/*.md | tail -5
```

## Future Enhancements

Planned improvements:
- Automatic log fetching from GitHub API
- Parallel testing of multiple fix approaches
- Integration with error tracking systems
- Automatic PR review requests
- Success rate tracking and learning
