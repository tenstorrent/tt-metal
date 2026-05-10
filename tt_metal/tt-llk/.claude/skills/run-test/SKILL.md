---
name: run-test
description: Run LLK tests using the test runner agent. Never run pytest directly.
user_invocable: true
---

# /run-test — Run LLK Tests

Wraps `.claude/scripts/run_test.sh`, which serialises simulator access (flock), kills stale port processes, sets `TT_UMD_SIMULATOR_PATH`, and exposes `count` / `compile` / `simulate` / `run` subcommands. Agents and Claude must never call `pytest` directly.

## Usage

```
/run-test <test_file> [--arch <arch>] [options]
```

Examples:
```
/run-test test_eltwise_binary_quasar.py --arch quasar
/run-test test_sfpu_square_quasar.py --arch quasar --k Float16
/run-test test_eltwise_binary_quasar.py --arch quasar --rerun
/run-test test_eltwise_binary_quasar.py --arch quasar --compile-only
/run-test test_sfpu_square_quasar.py --arch quasar --no-split
/run-test test_eltwise_binary_quasar.py --arch quasar --maxfail 5
```

## Arguments

- `<test_file>` — required, e.g. `test_sfpu_square_quasar.py`
- `--arch <arch>` — `quasar`, `blackhole`, `wormhole`. Required unless inferable (see below)
- `--k <expr>` — pytest `-k` filter (applied to both compile-producer and consumer so the two phases stay aligned)
- `--test-id <id>` — single parametrize ID (single-quotes/brackets safe)
- `--maxfail <N>` — stop after N failures
- `--rerun` — skip compile, simulate only (uses prior compile-producer artifacts)
- `--compile-only` — compile-producer only, no simulate
- `--no-split` — combined compile+run in one pytest invocation (issue-solver tests)
- `--port <N>` — simulator port (default 5556)
- `--timeout <secs>` — pytest timeout ceiling (default 600)

## Arch Inference

If `--arch` is not provided, infer in this order:
1. **Test path**: file lives in `tests/python_tests/quasar/` → `quasar`
2. **Filename suffix**: `*_quasar.py` / `*_blackhole.py` / `*_wormhole.py` → use that
3. Otherwise (top-level cross-arch tests like `test_matmul.py`): ask the user

## What to Do

1. **Parse** the test file, arch (or infer), and options.

2. **Pick a subcommand** for the script:
   - `--compile-only` → `compile`
   - `--rerun` → `simulate`
   - `--no-split` → `run --no-split` (combined invocation)
   - Default → `run` (compile then simulate)

3. **Spawn the `llk-test-runner` agent** with everything it needs to enforce env setup AND run the script:

   ```
   Agent tool:
     subagent_type: "llk-test-runner"
     description: "Run tests: {test_file} ({arch})"
     prompt: |
       Run this LLK test:
         test_file: {test_file}
         arch:      {arch}
         command:   {count|compile|simulate|run}
         options:   {--k EXPR | --test-id ID | --maxfail N | --no-split | --port N | --timeout S} (whatever applies)

       MANDATORY pre-flight (do these BEFORE invoking run_test.sh):

       1. Verify the venv exists at `tests/.venv/bin/activate`.
          If it does NOT exist, run from the tt-llk worktree root:
              cd tests && CHIP_ARCH={arch} ./setup_testing_env.sh
          The CHIP_ARCH env var is REQUIRED — setup builds arch-specific bits.
          The script picks the test dir from --arch; do not pre-check it.

       2. Then invoke the runner with the worktree path and arch:
              bash .claude/scripts/run_test.sh {command} \
                  --worktree "$(pwd)" \
                  --arch {arch} \
                  --test {test_file} \
                  [other options]
          (Resolve `$(pwd)` to the tt-llk root before passing.)

       Read the script's exit code:
         0 = pass, 1 = test failure, 2 = compile failure,
         3 = env error (venv missing / lock timeout / port stuck), 4 = bad args.

       On failure, surface the failing lines from the script's stdout/stderr
       (the script does not persist logs to disk). Do not modify code.
   ```

4. **Report** the agent's result. If the agent reports exit 1 (test failure) or 2 (compile failure), suggest `/debug-kernel` as the next step. If exit 3 (env error), surface the root cause (venv missing, simulator port stuck, lock timeout) and stop — don't retry blindly.

## Constraints

- **Never** call `pytest` directly. Always go through the script.
- **Never** skip the venv check — the script exits 3 with a clear message, but the skill should detect-and-bootstrap rather than fail.
- **Never** run setup with the wrong `CHIP_ARCH` — wrong-arch venvs silently produce broken builds.
- **Never** run multiple simulator invocations in parallel from the same agent. The script's `flock` protects against concurrent agents, but a single agent should still serialise its own runs.
