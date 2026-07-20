# Running LLK Quasar tests from tt-metal

This tutorial describes the local developer flow for running `tt-llk` functional
and performance tests against the Quasar 1x3 Zebu emulator. It also explains the
local `tt-umd-simulators` launcher changes that make long runs and SSH failures
easier to handle.

The flow is:

```text
pytest in tt-metal/tt_metal/tt-llk
  -> tt-exalens on the local development machine
  -> tt-umd-simulators/build/emu-quasar-1x3/run.sh
  -> quasar-1x3_run_dev.sh
  -> SSH to a reachable soc-l-* host
  -> Aether schedules and runs the test on Zebu
```

## 1. Prerequisites

Before starting, make sure that:

- The development machine is an IRD reservation with access to the Toronto
  `soc-l-*` hosts and Zebu pool.
- The machine is on the corporate network/VPN.
- Passwordless public-key SSH works to at least one `soc-l-*` host.
- `tt-metal` and `tt-umd-simulators` are checked out next to each other.
- The `tt-llk` Python environment and SFPI toolchain are installed.
- The callback host and DEBUDA port for the IRD reservation are known.

The examples below use:

```bash
export TT_METAL_ROOT=/proj_sw/user_dev/$USER/tt-metal
export LLK_ROOT=$TT_METAL_ROOT/tt_metal/tt-llk
export UMD_SIM_ROOT=/proj_sw/user_dev/$USER/tt-umd-simulators
```

## 2. Build the Quasar emulator wrapper

The local build does not build the Zebu image. It creates the descriptor and
launcher directory that tt-exalens uses to reach Aether/Zebu.

```bash
cd "$UMD_SIM_ROOT"
cmake -B build -G Ninja
ninja -C build
```

Verify the generated files:

```bash
test -f "$UMD_SIM_ROOT/build/emu-quasar-1x3/soc_descriptor.yaml"
test -x "$UMD_SIM_ROOT/build/emu-quasar-1x3/run.sh"
test -x "$UMD_SIM_ROOT/build/emu-quasar-1x3/quasar-1x3_run_dev.sh"
```

Re-run `ninja -C build` after changing
`emu/quasar-1x3/quasar-1x3_run_dev.sh`; CMake copies it into the build
directory.

### Optional: install the instrumented launcher

The untracked
`emu/quasar-1x3/quasar-1x3_run_dev.instrumented.sh` is not copied by CMake.
Install it explicitly when diagnosing bring-up failures:

```bash
SIM_BUILD="$UMD_SIM_ROOT/build/emu-quasar-1x3"
cp -a "$SIM_BUILD/quasar-1x3_run_dev.sh" \
      "$SIM_BUILD/quasar-1x3_run_dev.sh.pre_instrument"
cp -a "$UMD_SIM_ROOT/emu/quasar-1x3/quasar-1x3_run_dev.instrumented.sh" \
      "$SIM_BUILD/quasar-1x3_run_dev.sh"
```

It can also be installed while preserving existing logs:

```bash
cd "$LLK_ROOT/tests/python_tests/quasar"
./preserve_exalens_logs.sh
```

Restore the generated standard launcher with:

```bash
cp -a "$SIM_BUILD/quasar-1x3_run_dev.sh.pre_instrument" \
      "$SIM_BUILD/quasar-1x3_run_dev.sh"
```

## 3. Configure the callback and LLK environment

Emulation uses TCP because Zebu runs remotely. Obtain the callback host and
DEBUDA port from `ird list` or `P_USER_DBD_PORT`, then export them:

```bash
export CHIP_ARCH=quasar
export TT_METAL_SIMULATOR="$UMD_SIM_ROOT/build/emu-quasar-1x3"
export TT_UMD_SIMULATOR_PATH="$TT_METAL_SIMULATOR"
export NNG_SOCKET_ADDR="tcp://<IRD-host-visible-to-Zebu>:<DEBUDA-port>"
export NNG_SOCKET_LOCAL_PORT=5555
```

For example, an existing reservation may use a value such as
`tcp://tensix-l-01:54948`. Do not copy that port blindly: it belongs to one
specific reservation.

`TT_METAL_SIMULATOR` is the canonical variable used by the LLK pytest
configuration. Setting `TT_UMD_SIMULATOR_PATH` too keeps older helper scripts
compatible.

To prefer a particular Aether login host, set:

```bash
export SSH_MACHINE_NAME=soc-l-01
```

The modified launcher tries that host first and then searches the default pool.
The pool can be restricted or reordered:

```bash
export SSH_MACHINE_CANDIDATES="soc-l-03 soc-l-07 soc-l-09"
```

The remote Aether checkout defaults to
`/proj_sw/user_dev/$USER/work/aether`. Override it only if needed:

```bash
export AETHER_WORKSPACE=/proj_sw/user_dev/$USER/work/aether
```

The checkout on the selected remote host must be clean enough to fetch and
checkout the launcher's pinned `AETHER_TAG`.

## 4. Prepare the LLK test environment

Outside the standard LLK development container, use the full bootstrap. It
creates `tests/.venv`, installs the Python requirements (including tt-exalens),
and then installs SFPI:

```bash
cd "$LLK_ROOT/tests"
./setup_external_testing_env.sh
source .venv/bin/activate
```

In the LLK development container, where the Python environment is already
provided, refresh SFPI with:

```bash
cd "$LLK_ROOT/tests"
./setup_testing_env.sh
source .venv/bin/activate
```

`setup_testing_env.sh` installs or refreshes SFPI; it does not create the
Python virtual environment. If `tests/.venv` is absent, use the normal
tt-metal/tt-llk environment bootstrap for the current development image before
continuing.

Confirm the important values:

```bash
command -v pytest
command -v tt-exalens
printf 'simulator=%s\ncallback=%s\nlocal_port=%s\n' \
  "$TT_METAL_SIMULATOR" "$NNG_SOCKET_ADDR" "$NNG_SOCKET_LOCAL_PORT"
```

## 5. Run one functional test

The LLK helper is the safest routine entry point. It compiles in parallel,
serializes emulator access with `flock`, removes stale processes on port 5556,
and runs simulation without xdist:

```bash
bash "$LLK_ROOT/.cursor/scripts/run_test.sh" run \
  --worktree "$LLK_ROOT" \
  --arch quasar \
  --test test_unary_broadcast_quasar.py \
  --maxfail 1 \
  --verbose
```

Use `--k '<expression>'` to select a smaller set of parametrized variants.
Use `count` instead of `run` to inspect how many variants would be collected:

```bash
bash "$LLK_ROOT/.cursor/scripts/run_test.sh" count \
  --worktree "$LLK_ROOT" \
  --arch quasar \
  --test test_unary_broadcast_quasar.py
```

### Equivalent direct pytest flow

For a combined compile-and-run invocation:

```bash
cd "$LLK_ROOT/tests/python_tests/quasar"
pytest -x --run-simulator --port=5556 --timeout=1000 \
  test_unary_broadcast_quasar.py
```

For repeated development, compile first and then consume the existing
artifacts:

```bash
pytest --compile-producer -n 15 -x test_unary_broadcast_quasar.py
pytest --compile-consumer -x --run-simulator --port=5556 --timeout=1000 \
  test_unary_broadcast_quasar.py
```

Do not use parallel pytest workers for the simulator-consuming phase. Only one
process should own the Quasar emulator connection.

### Run the functional regression

The regression wrapper runs matching `test_*.py` files and manages tt-exalens
through pytest. It requires `EXALENS_PORT` in addition to the variables exported
earlier:

```bash
export EXALENS_PORT=5556
cd "$LLK_ROOT/tests"
./run_quasar_regression.sh
```

Restrict the run by filename glob and pass extra pytest arguments after `--`:

```bash
./run_quasar_regression.sh -t "test_reduce_*.py" -- -v --tb=short
```

## 6. Run one Quasar performance test

Performance tests normally use `--speed-of-light`. Select a run type with the
fully qualified enum name because pytest `-k` uses substring matching; bare
`PACK_ISOLATE` also matches `UNPACK_ISOLATE`.

Using the helper:

```bash
bash "$LLK_ROOT/.cursor/scripts/run_test.sh" run \
  --worktree "$LLK_ROOT" \
  --arch quasar \
  --test perf_unary_broadcast_quasar.py \
  --speed-of-light \
  --k "PerfRunType.MATH_ISOLATE" \
  --maxfail 1 \
  --verbose
```

Using pytest directly:

```bash
cd "$LLK_ROOT/tests/python_tests/quasar"
pytest -x --run-simulator --port=5556 --timeout=1000 \
  --speed-of-light \
  -k "PerfRunType.MATH_ISOLATE" \
  perf_unary_broadcast_quasar.py
```

`--speed-of-light` promotes runtime parameters to compile-time constants, so a
large parameter sweep can produce many builds and take substantially longer
than a functional run.

## 7. Run the performance suite

The per-run-type suite runner installs the instrumented launcher, retries
tt-exalens bring-up timeouts, detects single-test hangs, and writes a Markdown
report.

Start with one suite entry:

```bash
cd "$LLK_ROOT/tests/python_tests/quasar"
bash ./run_perf_suite_and_report.sh \
  --run-type MATH_ISOLATE \
  --suite-ids 10
```

Run all mapped tests for one run type:

```bash
bash ./run_perf_suite_and_report.sh --run-type MATH_ISOLATE
```

Run the complete per-run-type sequence:

```bash
nohup bash ./run_perf_suite_orchestrator.sh \
  > perf_suite_orchestrator.stdout 2>&1 &
```

The current suite scripts contain developer-specific absolute paths and a
fixed `NNG_SOCKET_ADDR` near the top. Review and update those values for the
active checkout and IRD reservation before starting a long run.

`run_perf_suite_all_modes.sh` is a newer work-in-progress path. It runs all
enabled `PerfRunType` values in one pytest case and records known hangs for
retry. Prefer the per-run-type commands above unless intentionally validating
the all-modes infrastructure.

## 8. Logs and results

Run commands from `tests/python_tests/quasar` so all bring-up logs are together:

- `tt-exalens.log` contains local tt-exalens startup and server output.
- `emu_<timestamp>_<socket>.log` contains launcher, SSH, Aether, and Zebu output.
- `/tmp/tt-llk-build/` contains generated LLK variants, `build.h`, and ELF files.
- `$LLK_ROOT/perf_data/` contains combined performance CSV reports.
- `perf_output_<run_type>_<suite_id>.txt` contains suite pytest output.
- `perf_suite_results_<run_type>.md` contains the per-run-type suite summary.

Before a manual retry, preserve logs because a new tt-exalens process can
truncate `tt-exalens.log`:

```bash
cd "$LLK_ROOT/tests/python_tests/quasar"
./preserve_exalens_logs.sh
```

During a long bring-up, inspect both logs:

```bash
tail -f tt-exalens.log
tail -f "$(ls -1t emu_*.log | sed -n '1p')"
```

`Waiting for ack msg from remote...` can be normal while the scheduler allocates
Zebu or while the test is running.

## 9. What changed in the local emulator launchers

The changes reported by `git status` in `tt-umd-simulators` have two scopes.

First, these launchers changed `EMULATOR_TIMEOUT` from 180 seconds to 36000
seconds:

- `emu/quasar-1x3/quasar-1x3_run_dev.sh`
- `emu/quasar-2x3/quasar-2x3_run.sh`
- `emu/quasar-2x3_DISPATCH/quasar-2x3_DISPATCH_run.sh`
- `emu/quasar-9x4_DM/quasar-9x4_DM_run.sh`

The LLK flow in this tutorial uses Quasar 1x3; the other timeout edits affect
their corresponding emulator configurations only.

Second, the Quasar 1x3 developer launcher now:

- Tries an explicit `SSH_MACHINE_NAME` first, then `soc-l-01` through
  `soc-l-12`.
- Uses batch public-key authentication and bounded connection/keepalive
  timeouts.
- Treats OpenSSH exit code 255 as a connection/session failure instead of
  incorrectly treating it as a missing Aether workspace.
- Writes a `FATAL: SSH ...` marker when no usable host is available.

The instrumented launcher adds timestamps, retries after host selection,
explicit SSH1/SSH2/SSH3 phase results, and fail-fast dependency ordering.
The LLK `ExalensServer` recognizes `FATAL: SSH` in a new emulator log and aborts
bring-up instead of waiting for its full 600-second readiness timeout.

The 36000-second Aether timeout is not the only timeout in the stack. A direct
pytest `--timeout`, the 600-second tt-exalens readiness timeout, and suite
hang-watch thresholds can still end a run earlier.

## 10. Troubleshooting

### `FATAL: SSH to <none> failed`

Test SSH independently:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 soc-l-01 true
```

Then set `SSH_MACHINE_NAME` or a smaller `SSH_MACHINE_CANDIDATES` list to hosts
that are valid for the current reservation.

### SSH succeeds but Aether setup fails

Check the remote workspace:

```bash
ssh "$SSH_MACHINE_NAME" \
  "cd '$AETHER_WORKSPACE' && git status --short && git branch --show-current"
```

Local modifications or an unavailable pinned tag can prevent `git fetch` or
`git checkout` from completing.

### tt-exalens never becomes ready

Verify:

```bash
test -d "$TT_METAL_SIMULATOR"
test -f "$TT_METAL_SIMULATOR/soc_descriptor.yaml"
printf '%s\n' "$NNG_SOCKET_ADDR" "$NNG_SOCKET_LOCAL_PORT"
lsof -i :5556
```

Inspect the newest emulator log for host selection, SSH return codes, Aether
checkout, `setup_emu_env.sh`, and `make test` status.

### Port 5556 is held by a stale server

The LLK helper clears stale users of its selected port. For direct pytest runs,
inspect the owner before terminating it:

```bash
lsof -i :5556
pgrep -af 'tt-exalens|quasar-1x3_run_dev'
```

Only terminate processes belonging to the current run. The suite scripts use
targeted `pkill` cleanup after a timeout or hang.

### A run appears stuck after Ctrl+C

During startup, the LLK server may wait for tt-exalens to become ready before
performing graceful shutdown so the emulator resource can be released. Follow
`tt-exalens.log` and the newest `emu_*.log` before forcing termination.

## 11. Minimal checklist

For subsequent runs, the essential sequence is:

```bash
export LLK_ROOT=/proj_sw/user_dev/$USER/tt-metal/tt_metal/tt-llk
export UMD_SIM_ROOT=/proj_sw/user_dev/$USER/tt-umd-simulators
export CHIP_ARCH=quasar
export TT_METAL_SIMULATOR="$UMD_SIM_ROOT/build/emu-quasar-1x3"
export TT_UMD_SIMULATOR_PATH="$TT_METAL_SIMULATOR"
export NNG_SOCKET_ADDR="tcp://<IRD-host-visible-to-Zebu>:<DEBUDA-port>"
export NNG_SOCKET_LOCAL_PORT=5555

source "$LLK_ROOT/tests/.venv/bin/activate"
cd "$LLK_ROOT/tests/python_tests/quasar"
pytest -x --run-simulator --port=5556 --timeout=1000 \
  test_unary_broadcast_quasar.py
```
