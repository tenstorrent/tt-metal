# Reproduce Deterministic Failures

This directory contains minimal reproducible tests for deterministic CI failures.

## Purpose

When a test fails deterministically in CI (fails every time), we need a simple, isolated test case that reproduces the failure locally for efficient debugging.

## Workflow

### 1. User: Create Failure Folder

When you encounter a deterministic failure:

```bash
cd .github/scripts/reproduce-deterministic-failures
mkdir <failure-name>
cd <failure-name>
mkdir logs tests
```

### 2. User: Collect Logs

Copy the error logs from CI into the `logs/` folder:

```bash
# Example
cp ~/Downloads/ci-error-log.txt logs/
```

Include the full error output with stack traces.

### 3. User: Invoke AI

Provide the AI with the prompt from `AI_PROMPT.md` and tell it which failure folder to work on:

```
Read AI_PROMPT.md and create a minimal reproduction test for reproduce-deterministic-failures/<failure-name>/
```

### 4. AI: Create Minimal Reproduction

The AI will:
- Read the logs in `logs/`
- Find the original test code
- Identify the root cause
- Create a minimal, isolated test in `tests/`
- Document the issue and reproduction steps in a README

### 5. User: Run and Debug

Run the test to verify it reproduces the failure:

```bash
cd <failure-name>/tests
pytest test_*_repro.py -v
```

The test should fail immediately with the same error as CI.

### 6. User: Fix and Verify

- Debug and fix the issue
- Re-run the reproduction test to verify the fix
- Run the original test suite to ensure no regressions

## Directory Structure

```
reproduce-deterministic-failures/
├── AI_PROMPT.md              # Generic prompt for AI
├── README.md                 # This file
└── <failure-name>/           # One folder per failure
    ├── README.md             # Failure-specific documentation
    ├── logs/                 # CI error logs (user provides)
    │   └── *.txt
    └── tests/                # Minimal reproduction (AI creates)
        └── test_*_repro.py
```

## Tips

- **Name failures descriptively**: `bert-attention-assertion`, `gather-shape-mismatch`, etc.
- **Keep tests minimal**: Strip away everything not needed to reproduce
- **Fast feedback**: Tests should run in seconds, not minutes
- **Document expected vs actual**: Clear description of what's wrong
- **Include fix verification**: After fixing, the same test should pass

## When to Use This

Use this workflow for:
- Deterministic test failures (fails every time)
- Assertion failures with clear error messages
- Shape mismatches, type errors, or logic bugs
- Failures that need debugging and fixing

Don't use this for:
- Non-deterministic failures (use `reproduce-ND-failures/` instead)
- Failures already reproducible with a simple command
- Issues that need hardware unavailable locally

## Difference from Non-Deterministic Failures

| Aspect | Deterministic | Non-Deterministic |
|--------|---------------|-------------------|
| **Failure rate** | 100% | Intermittent |
| **Goal** | Minimal reproduction for debugging | Amplification to trigger failure |
| **Test complexity** | Simple, isolated | Stress test with iterations |
| **Runtime** | Seconds | Up to 5 minutes |
| **Amplification** | Not needed | Critical (parallel, iterations) |
| **Use case** | Fix bugs | Find race conditions |
