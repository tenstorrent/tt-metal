# Reproduce Non-Deterministic Failures

This directory contains stress tests for reproducing non-deterministic CI failures.

## Purpose

When a test fails non-deterministically in CI, it's hard to debug. This workflow helps create reproducible stress tests that amplify the conditions causing the failure, making it fail reliably within 20 minutes.

## Workflow

### 1. User: Create Failure Folder

When you encounter a non-deterministic failure:

```bash
cd .github/scripts/reproduce-ND-failures
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

You can include multiple logs if the failure happened multiple times.

### 3. User: Invoke AI

Provide the AI with the prompt from `AI_PROMPT.md` and tell it which failure folder to work on:

```
Read AI_PROMPT.md and create a stress test for reproduce-ND-failures/<failure-name>/
```

### 4. AI: Create Stress Test

The AI will:
- Read the logs in `logs/`
- Find the original test code
- Identify the root cause
- Create an amplified stress test in `tests/`
- Document the approach in a README

### 5. User: Run and Verify

Run the stress test to verify it reproduces the failure:

```bash
cd <failure-name>/tests
python test_*_stress.py
```

### 6. User: Iterate

If it doesn't reproduce:
- Ask AI to increase amplification
- Provide more logs or context
- Try different amplification strategies

## Directory Structure

```
reproduce-ND-failures/
├── AI_PROMPT.md              # Generic prompt for AI
├── README.md                 # This file
└── <failure-name>/           # One folder per failure
    ├── README.md             # Failure-specific documentation
    ├── logs/                 # CI error logs (user provides)
    │   └── *.txt
    └── tests/                # Stress tests (AI creates)
        └── test_*_stress.py
```

## Example

See `T3K-reduce-scatter-race/` for a complete example.

## Tips

- **Name failures descriptively**: `T3K-reduce-scatter-race`, `ethernet-hang-wormhole`, etc.
- **Include multiple logs**: More examples help AI identify patterns
- **Run on same hardware**: Tests should run on the same hardware that failed in CI
- **Tune parameters**: Use CLI args to adjust amplification without code changes
- **Document results**: Note reproduction rate, timing, environment details

## When to Use This

Use this workflow for:
- Non-deterministic test failures (pass sometimes, fail sometimes)
- Race conditions and timing bugs
- Hardware-specific failures
- Timeouts and hangs

Don't use this for:
- Deterministic failures (just fix the bug)
- Obvious bugs with clear fix
- Failures that need new hardware unavailable locally
