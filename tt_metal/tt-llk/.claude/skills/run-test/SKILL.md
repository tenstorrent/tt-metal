---
name: run-test
description: Run LLK tests using the test runner agent. Never run pytest directly.
user_invocable: true
---

# /run-test — Run LLK Tests

## Usage

```
/run-test test_pack_untilize.py
/run-test test_pack_untilize.py -k "Float16_b"
/run-test test_math_reduce.py --rerun
/run-test test_eltwise_binary.py --compile-only
```

## What to Do

1. Parse the test file name and options from the user's arguments
2. Determine the scenario:
   - Default: `code-changed` (ENV_SETUP=0, COMPILED=1, RUN_TEST=1)
   - `--rerun`: rerun-only (ENV_SETUP=0, COMPILED=0, RUN_TEST=1)
   - `--compile-only`: compile only (ENV_SETUP=0, COMPILED=1, RUN_TEST=0)
   - `--fresh`: first run (ENV_SETUP=1, COMPILED=1, RUN_TEST=1)
3. Pass any `-k` filter or extra args via PYTEST_ARGS
4. Spawn the **llk-test-runner** agent:

```
Agent tool:
  subagent_type: "llk-test-runner"
  description: "Run tests: {test_file}"
  prompt: |
    Run this test:
    - File: {test_file}
    - Scenario: {scenario}
    - Extra args: {pytest_args or "none"}
```

5. Report the agent's results to the user
6. If tests failed, suggest `/debug-kernel` as next step
