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
/run-test test_matmul_quasar.py --arch quasar --test-id 'test_matmul_quasar.py::test_matmul[math_fidelity:LoFi-...-format:Float16->Float16-...]'
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
- `--progress` — manual-debug aid: emit a `[progress] <phase>: elapsed=Ns, last_output=Ns ago` line to stderr every 30s during compile and simulate. Off by default; pass when you suspect a hang and want to see whether a phase is alive but slow vs. truly stuck. Pair with `--progress-interval <secs>` to tune.

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

3. **Spawn the `llk-test-runner` agent.** Its system prompt covers env setup, script invocation, exit-code diagnosis, and output formatting — just pass the inputs:

   ```
   Agent tool:
     subagent_type: "llk-test-runner"
     description: "Run tests: {test_file} ({arch})"
     prompt: |
       test_file: {test_file}
       arch:      {arch}
       command:   {compile|simulate|run|count}
       options:   {whatever applies from --k / --test-id / --maxfail / --no-split / --port / --timeout}
   ```

4. **Relay** the agent's verdict to the user. On FAIL or COMPILE_FAIL, suggest `/debug-kernel`. On ENV_ERROR or HANG, surface the agent's diagnosis and stop — don't auto-retry (the same hang will reappear, and env errors need a human fix).

## LLK Triage on Hang

When `run_test.sh` detects a silicon hang (exit 5), after killing the
consumer tree and before resetting the device, it runs `.claude/scripts/llk_triage.py`.
Output appears in the `RUN_LLK_TESTS_HANG` block as:

```
--- llk-triage ---
=== LLK TRIAGE ===
Arch:     blackhole
Location: 0,0
== Mailbox state ==
  Unpacker @ 0x...  = 0x000000FF (KERNEL_COMPLETE)
  Math     @ 0x...  = 0x00000000 (incomplete)
  ...
Hung kernel threads: Math
== Tensix state (per RISC) ==
  { ... PC, soft-reset, status per RISC ... }
--- end llk-triage ---
```

Standalone use (e.g., to inspect a wedged device manually):

```bash
source tests/.venv/bin/activate
python3 .claude/scripts/llk_triage.py --arch blackhole \
    [--location 0,0] [--device-id 0]
```
