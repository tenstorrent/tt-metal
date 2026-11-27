---
name: run-test
description: Run LLK tests using the test runner agent. Never run pytest directly.
user_invocable: true
---

# /run-test — Run LLK Tests

Wraps `.claude/scripts/run_test.sh`, which serialises simulator and silicon access (flock), sets `TT_UMD_SIMULATOR_PATH`, and exposes `count` / `compile` / `simulate` / `run` subcommands. Agents and Claude must never call `pytest` directly for tests that build kernels or use a device or simulator, functional or perf.

Host-only work is outside this rule: `tests/run_hardware_free.sh` for the hardware-free suite, and `compare_test_and_perf.py`, which only reads `@parametrize`.

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
# --run-tag is single-use; choose a fresh value for every invocation.
/run-test perf_pack_dest_bank.py --arch wormhole --perf --speed-of-light --run-tag pack-dest-bank-sol-1
/run-test perf_matmul_quasar.py --arch quasar --perf --perf-counters
```

## Arguments

- `<test_file>` — required, e.g. `test_sfpu_square_quasar.py`
- `--arch <arch>` — `quasar`, `blackhole`, `wormhole`. Required unless inferable (see below)
- `--k <expr>` — pytest `-k` filter (applied to both compile-producer and consumer so the two phases stay aligned)
- `--test-id <id>` — single parametrize ID (single-quotes/brackets safe)
- `--op <op>` — with `--perf`, keep only SFPU sweep variants for that `MathOperation` name (repeatable; applied to count, compile and consumer)
- `--maxfail <N>` — stop after N failures
- `--rerun` — skip compile, simulate only (uses prior compile-producer artifacts)
- `--compile-only` — compile-producer only, no simulate
- `--no-split` — combined compile+run in one pytest invocation (issue-solver tests)
- `--timeout <secs>` — pytest timeout ceiling (default 600)
- `--speed-of-light` — compile runtime parameters as constants (the build the nightly `LLK perf` run measures; the PR and merge perf gates build without it)
- `--perf` — perf sweep matching each architecture's CI assert mode: `TT_LLK_DISABLE_ASSERTS=1` for Wormhole/Blackhole and `=0` for Quasar, plus `-m perf` (`--mode perf` for SFPU sweep modules) and `--dist loadgroup -n 15` for the silicon consumer. Unlike CI the local runner does not stop at the first failure (no compile `-x`, no default `--maxfail`). Asserts-on and asserts-off builds have distinct ELF variant IDs, so `simulate` cannot reuse the other kind. A module or `--k` with no perf variants exits 4. Prints a `RUN_LLK_TESTS_PERF_REPORT` line naming `perf_data/runs/<tag>` before the verdict; a non-PASS consumer marks it partial. See `perf-report` for validating a PASS result
- `--run-tag <tag>` / `--keep-runs <N>` — perf report directory name and retention (`PERF_RUN_TAG`, `PERF_KEEP_RUNS`; `--keep-runs 0` keeps all); need `--perf`. Tags match `[A-Za-z0-9][A-Za-z0-9._-]*` and are single-use
- `--consumer-jobs <N>` — silicon consumer parallelism (default 15; 0 is serial); needs `--perf`
- `--perf-counters` / `--dump-perf-counters` — counter report (a separate report kind); need `--perf`
- `--keep-asserts` — keep `LLK_ASSERT` in a `--perf` build (already the Quasar default; differs from Wormhole/Blackhole CI)
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
       options:   {whatever applies from --k / --test-id / --maxfail / --no-split / --timeout /
                   --speed-of-light / --perf / --op / --run-tag / --keep-runs / --consumer-jobs /
                   --perf-counters / --dump-perf-counters / --keep-asserts}
   ```

4. **Relay** the agent's verdict to the user. For `--perf`, include the report directory; a PASS says nothing about whether the numbers are plausible, so hand off to `perf-report` validation. On FAIL or COMPILE_FAIL, suggest `/debug-kernel`. On ENV_ERROR or HANG, surface the agent's diagnosis and stop — don't auto-retry (the same hang will reappear, and env errors need a human fix).

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
