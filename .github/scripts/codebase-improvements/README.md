# Codebase Improvements

This directory contains AI-powered workflows for reproducing failures and implementing fixes.

## Overview

```
codebase-improvements/
├── info.json                              # Input configuration for automated runs
├── run.sh                                 # Main orchestration script
├── outputs/                               # Run reports (one per execution)
│   └── YYYY-MM-DD_HH-MM-SS_<desc>.md
├── reproduce-deterministic-failures/      # For 100% reproducible failures
├── reproduce-ND-failures/                 # For non-deterministic failures
├── analyze-nd-failures/                   # For analyzing failure patterns
└── implementing-features/                 # For feature implementation and fixes
```

## Quick Start

### 1. Configure Your Task

Edit `info.json`:

```json
{
  "deterministic": true,
  "url": "https://github.com/tenstorrent/tt-metal/actions/runs/...",
  "prompt": "This test fails because gather with wide tensors times out. Fix the underlying operation.",
  "raw-logs": "",
  "existing-test-path": ""
}
```

**Fields:**
- `deterministic`: `true` if failure happens every time, `false` if intermittent
- `url`: GitHub Actions run URL (or leave empty if using raw-logs)
- `prompt`: Description of what to fix or improve
- `raw-logs`: Raw error logs/stack trace (or leave empty if using url)
- `existing-test-path`: Path to existing reproduction test (skips test creation phase)

**Note:**
- **ALWAYS provide either `url` OR `raw-logs`** - Claude needs error context to fix
- Optionally provide `existing-test-path` to skip test creation
- If using `existing-test-path`, you still need logs/url for error information

### 2. Run the Automation

```bash
cd .github/scripts/codebase-improvements
./run.sh
```

The script will:
1. Fetch logs from the URL or use raw-logs
2. Create a reproduction test
3. Confirm the test reproduces the issue
4. Create a fix branch off main
5. Iteratively fix the codebase
6. Create a draft PR with the fix
7. Generate a report in `outputs/`

### 3. Review Results

Check `outputs/` for a markdown file with:
- Success/failure status
- PR link (if successful)
- Detailed execution log
- Next steps and developer contacts

## Use Cases

### Example 1: Fix Slow Model Performance

```json
{
  "deterministic": true,
  "url": "https://github.com/tenstorrent/tt-metal/actions/runs/12345",
  "prompt": "Llama 70B prefill is too slow. Optimize ops to meet benchmark requirements.",
  "raw-logs": ""
}
```

### Example 2: Fix Intermittent Timeout

```json
{
  "deterministic": false,
  "url": "",
  "prompt": "Reduce scatter occasionally hangs on T3K. Find and fix the race condition.",
  "raw-logs": "... paste logs here ..."
}
```

### Example 3: Use Existing Test

```json
{
  "deterministic": true,
  "url": "",
  "prompt": "Fix gather timeout with wide tensors by optimizing completion queue.",
  "raw-logs": "",
  "existing-test-path": "reproduce-deterministic-failures/timeout-in-datamovement/tests/test_gather_timeout_stress.py"
}
```

### Example 4: Improve Passing Test

```json
{
  "deterministic": true,
  "url": "https://github.com/tenstorrent/tt-metal/actions/runs/12345",
  "prompt": "This test passes but is slow. Improve efficiency by 20%.",
  "raw-logs": "",
  "existing-test-path": ""
}
```

## Workflow Details

### Phase 0: Configuration
- Parse info.json
- Validate inputs
- Check if existing test provided (skip Phase 1 if so)

### Phase 1: Reproduction (5 min) - *Optional*
- Skipped if `existing-test-path` is provided
- Fetch logs from URL or use raw-logs
- Route to reproduce-deterministic-failures or reproduce-ND-failures
- Create minimal/stress test
- Verify test reproduces the issue
- Commit test to current branch

### Phase 2: Fix Development (10 min)
- Create new branch off main
- Copy test to new branch
- Analyze root cause
- Implement fixes iteratively
- Run test after each change
- Verify fix is stable

### Phase 3: PR Creation (5 min)
- Create draft PR (excluding the test file)
- Generate PR description with:
  - Summary of changes
  - Root cause explanation
  - Recommended CI workflows to run
  - Performance impact
- Write report to outputs/

## Time Limits

- **Total runtime**: 20 minutes maximum
- **Reproduction**: 5 minutes
- **Fix development**: 10 minutes
- **PR creation**: 5 minutes

If the AI cannot make progress within these limits, it will:
- Document what it tried
- Explain why it failed
- Recommend next steps
- List relevant developers to contact

## Output Reports

Each run generates a markdown file in `outputs/` with format:

```
outputs/2026-02-04_16-30-00_gather-timeout-fix.md
```

Contents:
- **Status**: Success/Partial/Failed
- **PR Link**: (if successful)
- **Test Created**: Path to reproduction test
- **Changes Made**: Summary of code changes
- **Execution Log**: Detailed timeline
- **Next Steps**: What to do next
- **Developers**: Relevant contacts

## Integration with CI

This framework is designed to be run in GitHub Actions:

```yaml
- name: Auto-fix failure
  run: |
    cd .github/scripts/codebase-improvements
    ./run.sh
```

Anyone can trigger this by:
1. Pushing a branch with configured info.json
2. Using workflow_dispatch with inputs
3. Automatically on repeated CI failures

## Subdirectories

### reproduce-deterministic-failures/
For failures that happen 100% of the time. Creates minimal reproduction tests for debugging.

### reproduce-ND-failures/
For non-deterministic failures (race conditions, timeouts). Creates stress tests that amplify failure conditions.

### analyze-nd-failures/
For analyzing patterns in non-deterministic failures across multiple runs.

### implementing-features/
For feature implementation and complex fixes. Contains prompts and workflows for the fix development phase.

## Example: Existing Test

See `reproduce-deterministic-failures/timeout-in-datamovement/` for a complete example:
- Test: `tests/test_gather_timeout_stress.py`
- Reproduces: `ttnn.to_torch()` timeout on wide tensors `[1, 151936]`
- Status: Successfully reproduced, ready for fixing

## Notes

- Tests created for reproduction stay on the original branch
- PRs exclude these tests (they're just for development)
- All runs are logged to outputs/
- AI can give up if no progress is made
- Modular design allows testing each phase independently
