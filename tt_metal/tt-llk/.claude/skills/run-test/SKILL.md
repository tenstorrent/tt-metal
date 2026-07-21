---
name: run-test
description: Run LLK tests using the test runner agent. Never run pytest directly.
user_invocable: true
---

# /run-test — Run LLK Tests

Wraps `.claude/scripts/run_test.sh`, which serialises every invocation on one global lock (flock), fetches SFPI when missing, sets `TT_UMD_SIMULATOR_PATH`, watches the run and kills a hang gracefully, reaps stale emulator jobs, and exposes `count` / `compile` / `simulate` / `run` subcommands. It is a single blocking call — invoke it like pytest and wait for the verdict. Agents and Claude must never call `pytest` directly.

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
- `--test-id <id>` — single parametrize ID (single-quotes/brackets safe); a leading `<arch>/` rootdir prefix, as emitted by `count`/`--collect-only`, is stripped automatically
- `--maxfail <N>` — stop after N failures
- `--rerun` — skip compile, simulate only (uses prior compile-producer artifacts)
- `--compile-only` — compile-producer only, no simulate
- `--no-split` — combined compile+run in one pytest invocation (issue-solver tests)
- `--port <N>` — simulator port (default 5556)
- `--stall <secs>` — log-stall seconds that mark a hang (default 180 emulator, 300 silicon)

## Blocking call, hang-bounded

`run_test.sh` is a single synchronous call — invoke it and wait for the verdict; there is no resume loop and no timeout to tune. The only wait is on the global lock (first-come, first-served, unbounded). Once a run starts a watcher bounds it: a post-ready log stall (emulator) or `TENSIX TIMED OUT` (silicon) is detected, killed gracefully so tt-exalens releases the remote emulator, and reported as HANG (exit 5). The `llk-test-runner` agent passes the Bash-tool maximum `timeout: 600000` as a backstop only.

## Sandbox

Every `run_test.sh` call must run with `dangerouslyDisableSandbox: true` — the script needs network to the remote emulator and writes the build cache under `/tmp`, both of which a sandboxed Bash blocks (surfacing as a false `ENV_ERROR`). The flag is a no-op when the run is already un-sandboxed, e.g. the headless codegen launchers (`batch_generate.sh` / `run_5_kernels.sh`) run `claude -p --dangerously-skip-permissions`. The `llk-test-runner` agent applies it automatically.

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

3. **Spawn the `llk-test-runner` agent** — synchronously, in the foreground. Never set `run_in_background`, and never end your turn while it runs: a sim orphaned by an ended turn leaks a remote emulator job. Its system prompt covers env setup, script invocation, exit-code diagnosis, and output formatting — just pass the inputs:

   ```
   Agent tool:
     subagent_type: "llk-test-runner"
     description: "Run tests: {test_file} ({arch})"
     prompt: |
       test_file: {test_file}
       arch:      {arch}
       command:   {compile|simulate|run|count}
       options:   {whatever applies from --k / --test-id / --maxfail / --no-split / --port}
   ```

4. **Relay** the agent's verdict to the user. On FAIL or COMPILE_FAIL, suggest `/debug-kernel`. On HANG, surface the diagnosis and stop — don't auto-retry (the same hang reappears; the script already reset the device / reaped the emulator job). On ENV_ERROR, one retry is acceptable (the emulator may have been transiently congested); a persistent ENV_ERROR needs a human.

## Inspecting a wedged device manually

On a silicon (BH/WH) hang the script resets the device with `tt-smi -r`. To read Tensix state before a reset, run the triage script standalone:

```bash
source tests/.venv/bin/activate
python3 .claude/scripts/llk_triage.py --arch blackhole \
    [--location 0,0] [--device-id 0]
```
