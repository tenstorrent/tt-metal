# Reusing a tt-exalens Simulator Server for LLK Pytest Runs

This guide explains how to keep one RTL simulator-backed `tt-exalens` server
running and use separate short `pytest` invocations to run LLK tests against it.
This is useful when simulator startup dominates iteration time.

The standard `pytest --run-simulator` flow owns the whole simulator lifecycle:

1. `pytest` starts `tt-exalens`.
2. `pytest` waits until the simulator is ready.
3. The test runs.
4. `pytest` stops `tt-exalens` during session teardown.

That is simple and safe, but each new `pytest` command pays the full simulator
startup cost. The reusable server flow splits the owner process from the client
process:

1. One terminal starts `tt-exalens --server` and leaves it running.
2. Another terminal runs `pytest --run-simulator --reuse-simulator-server`.
3. Each pytest process connects to the existing server and exits without
   stopping it.
4. You stop `tt-exalens` manually when done.

## When to Use This

Use this flow for tight development loops where you repeatedly run the same LLK
test or a small set of tests on the same simulator instance.

Good examples:

- Performance microbenchmarks.
- Debugging one test while changing Python or C++ test code.
- Running many short pytest invocations where the simulator state is known to be
  clean enough between runs.

Avoid this flow when:

- You need a fresh simulator process for every test.
- The test leaves persistent device state that affects later test runs.
- You are using ttsim through a `.so` simulator path. Reuse mode is for the RTL
  `tt-exalens --server` flow, not in-process ttsim.
- You need `--reset-simulator-per-test`.

## Required Environment Variables

Set these in every terminal involved in the flow unless your shell startup
already exports them.

```bash
export CHIP_ARCH=quasar
export TT_METAL_SIMULATOR="/proj_sw/user_dev/${USER}/tt-umd-simulators/build/emu-quasar-1x3/"
export TT_UMD_SIMULATOR_PATH="$TT_METAL_SIMULATOR"
export NNG_SOCKET_ADDR="tcp://tensix-l-01:54948"
export NNG_SOCKET_LOCAL_PORT=5555
```

Variable meanings:

- `CHIP_ARCH` selects the LLK architecture. For Quasar tests, use `quasar`.
- `TT_METAL_SIMULATOR` is the simulator build path used by the LLK test harness.
  For this reuse flow it should point to an RTL simulator directory, not a `.so`.
- `TT_UMD_SIMULATOR_PATH` is accepted as an older alias. Keeping it equal to
  `TT_METAL_SIMULATOR` is useful for compatibility with existing scripts.
- `NNG_SOCKET_ADDR` is the NNG endpoint used by the simulator environment.
- `NNG_SOCKET_LOCAL_PORT` is the local NNG port used by `tt-exalens`.

`LLK_HOME` is normally initialized by the pytest harness if it is not already
set, so you usually do not need to export it manually.

The `--port` pytest option and `tt-exalens --port` value are the remote
`tt-exalens` server port. This is separate from `NNG_SOCKET_LOCAL_PORT`.

## One-Time Setup in Each Terminal

From the repository root or any LLK test directory:

```bash
export CHIP_ARCH=quasar
export TT_METAL_SIMULATOR="/proj_sw/user_dev/${USER}/tt-umd-simulators/build/emu-quasar-1x3/"
export TT_UMD_SIMULATOR_PATH="$TT_METAL_SIMULATOR"
export NNG_SOCKET_ADDR="tcp://tensix-l-01:54948"
export NNG_SOCKET_LOCAL_PORT=5555
```

Then go to the Python tests directory containing the test you want to run. For
example:

```bash
cd /proj_sw/user_dev/${USER}/tt-metal/tt_metal/tt-llk/tests/python_tests/quasar
```

## Terminal 1: Start the Reusable Server

Start `tt-exalens` manually and leave this terminal running:

```bash
tt-exalens --port=5556 --server -s "$TT_METAL_SIMULATOR"
```

Wait until the output shows the simulator is ready. The ready marker currently
includes:

```text
[4B MODE]
```

Keep this process running while you use reuse mode from another terminal.

## Terminal 2: Run Pytest Clients

Run the test with `--reuse-simulator-server` and the same port:

```bash
pytest -x --run-simulator --reuse-simulator-server --port=5556 \
  perf_sfpu_nop_quasar.py -vv
```

You can run the same command repeatedly. Each invocation is a new pytest process,
but it connects to the same long-lived `tt-exalens` server.

Expected pytest log shape:

```text
Connecting to existing tt-exalens server (port=5556)...
```

You should not see pytest logs like these in reuse mode:

```text
Starting tt-exalens server ...
Stopping tt-exalens ...
```

Those messages belong to the normal owned mode.

## Comparing Normal Mode and Reuse Mode

Normal owned mode:

```bash
pytest -x --run-simulator --port=5556 perf_sfpu_nop_quasar.py -vv
```

Use this when you want pytest to start and stop the simulator automatically.

Reusable server mode:

```bash
# Terminal 1
tt-exalens --port=5556 --server -s "$TT_METAL_SIMULATOR"

# Terminal 2
pytest -x --run-simulator --reuse-simulator-server --port=5556 \
  perf_sfpu_nop_quasar.py -vv
```

Use this when you want to avoid restarting the simulator for every pytest
command.

Do not run normal owned mode on the same port while a reusable server is active.
The normal owner path intentionally cleans up stale `tt-exalens` processes on
the selected port before starting a new one.

## Running Multiple Repetitions

If you only need to repeat the same test within one pytest process, you can also
use `pytest-repeat`:

```bash
pytest -x --run-simulator --reuse-simulator-server --port=5556 \
  --count=10 perf_sfpu_nop_quasar.py -vv
```

This still uses the long-lived external server, but it keeps the repeated test
executions inside one pytest session.

## Stopping tt-exalens When Done

The reusable server is externally managed, so pytest will not stop it. Stop it
from the terminal that started it by pressing `Ctrl+C`.

If the server does not exit, find it:

```bash
pgrep -af "tt-exalens.*--port=5556"
```

Terminate it gracefully:

```bash
pkill -TERM -f "[t]t-exalens --port=5556 --server"
```

Check whether anything remains:

```bash
pgrep -af "tt-exalens|emu-quasar-1x3"
```

If a simulator child process remains after `tt-exalens` exits, terminate it by
PID:

```bash
kill -TERM <pid>
```

If it still does not stop after a short wait:

```bash
kill -KILL <pid>
```

Use `SIGKILL` only as a last resort because it does not give the process a
chance to clean up.

## Supported and Unsupported Flag Combinations

Required:

```bash
--run-simulator --reuse-simulator-server
```

Unsupported:

```bash
--reuse-simulator-server
```

Reuse mode requires `--run-simulator`.

```bash
--run-simulator --reuse-simulator-server --reset-simulator-per-test
```

Reuse mode cannot reset the simulator between tests because pytest does not own
the external `tt-exalens` server.

Reuse mode is also unsupported when `TT_METAL_SIMULATOR` points to a `.so`
ttsim library. In-process ttsim does not use `ExalensServer`.

## Troubleshooting

### Pytest Cannot Connect

Check that the server is running on the same port:

```bash
pgrep -af "tt-exalens.*--port=5556"
```

Make sure the pytest command uses the same port:

```bash
pytest --run-simulator --reuse-simulator-server --port=5556 ...
```

Also confirm both terminals have the same simulator environment:

```bash
printf 'CHIP_ARCH=%s\nTT_METAL_SIMULATOR=%s\nTT_UMD_SIMULATOR_PATH=%s\nNNG_SOCKET_ADDR=%s\nNNG_SOCKET_LOCAL_PORT=%s\n' \
  "$CHIP_ARCH" \
  "$TT_METAL_SIMULATOR" \
  "$TT_UMD_SIMULATOR_PATH" \
  "$NNG_SOCKET_ADDR" \
  "$NNG_SOCKET_LOCAL_PORT"
```

### Pytest Starts tt-exalens Anyway

Check that the pytest command includes:

```bash
--reuse-simulator-server
```

Without that flag, `pytest --run-simulator` uses the normal owner mode and will
start and stop `tt-exalens` itself.

### A Later Run Behaves Differently

The simulator process is reused, so persistent simulator state can matter. If a
test looks stateful or flaky, stop the external server and start a fresh one:

```bash
pkill -TERM -f "[t]t-exalens --port=5556 --server"
tt-exalens --port=5556 --server -s "$TT_METAL_SIMULATOR"
```

For tests that require isolation, use normal owned mode instead:

```bash
pytest -x --run-simulator --port=5556 perf_sfpu_nop_quasar.py -vv
```
