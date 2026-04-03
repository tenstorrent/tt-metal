# Job Lifecycle

How to compose, route, execute, and monitor commands. Loaded after workspace
detection when tt-run needs to actually run something.

## Routing Decision

| Command type | Route | Example |
|---|---|---|
| Build (cmake, pip install) | Bash | `./build_metal.sh` |
| Git operations | Bash | `git pull`, `git status` |
| File I/O | Bash | `mkdir`, `cp`, read files |
| Pytest on device | tt-device-mcp | `tt_device_job_run` |
| Server start on device | tt-device-mcp | `tt_device_job_run_bg` |
| Device reset | tt-device-mcp | `tt_device_reset` |
| Health check (HTTP) | Bash | `curl -sf http://localhost:8000/health` |
| Benchmark against server | Bash or MCP | Depends on where server runs |

**Rule:** If the command requires a TT device to be attached to the process,
use tt-device-mcp. Everything else uses Bash.

## tt-device-mcp Tools

### Foreground job (blocks until done)

```
tt_device_job_run(command="pytest models/tt_transformers/tests/test_mlp.py --timeout 300")
```

Returns job ID. Blocks until completion. Use for tests and short commands.

### Background job (returns immediately)

```
tt_device_job_run_bg(command="python examples/server_example_tt.py --model ...")
```

Returns job ID immediately. Use for servers and long-running processes.

### Monitor

```
tt_device_job_status(job_id="...")   # Check if running/completed/failed
tt_device_job_wait(job_id="...")     # Block until completion
tt_device_job_logs(job_id="...")     # Stream stdout/stderr
```

### Cleanup

```
tt_device_job_kill(job_id="...")     # Terminate a running job
tt_device_reset()                    # Reset device state (last resort)
```

### Direct execution (no job wrapping)

```
tt_device_exec(command="echo $ARCH_NAME")
```

For quick one-off commands that don't need job tracking.

## Execution Patterns

### Pattern: Run a tt-metal model test

1. Load `knowledge/recipes/tt-metal/env.md` — set required env vars
2. Load `knowledge/recipes/tt-metal/build.md` — build if needed (incremental)
3. Compose: `pytest <test_path> --timeout <timeout>` with env vars
4. Route: `tt_device_job_run` (needs device)
5. Monitor: wait for completion, stream logs on failure

### Pattern: vLLM server + benchmark

1. Load `knowledge/recipes/vllm/env.md` — set required env vars
2. Load `knowledge/recipes/vllm/build.md` — install vLLM if needed
3. Load `knowledge/recipes/vllm/server.md` — start server
4. Compose: server start command with model and TT config
5. Route: `tt_device_job_run_bg` (server is long-running)
6. Health check: `curl -sf http://localhost:8000/health` in retry loop (Bash)
7. Load `knowledge/recipes/vllm/benchmark.md` — run benchmark
8. Compose: `vllm bench serve ...` command
9. Route: Bash (benchmark talks to server over HTTP, not device)
10. Cleanup: `tt_device_job_kill` to stop server

## Failure Recovery

### Build failure

Report the error with the failing command and output. Do not retry automatically.
Build failures usually need human diagnosis (missing deps, wrong env).

### Test failure

Stream logs via `tt_device_job_logs`. Report:
- Exit code
- Last 50 lines of output
- Whether it's a timeout vs assertion vs crash

### Device hang

If a job doesn't respond to `tt_device_job_status` or appears stuck:
1. `tt_device_job_kill(job_id)`
2. If kill fails: `tt_device_reset()`
3. Report the hang with job ID and last known state

### Server won't start

If health check times out:
1. Check `tt_device_job_status` — is the server process still alive?
2. Stream `tt_device_job_logs` — look for crash or OOM
3. Report with log tail
