# Execution

Routing, environment, and failure recovery for command execution.

## Routing Rule

**Never run tests or device commands directly via Bash.** Multiple agents may
share the device — the MCP queue prevents conflicts.

| Route | When | Tools |
|---|---|---|
| tt-device-mcp | Anything that needs a TT device | `tt_device_job_run`, `tt_device_job_run_bg`, `tt_device_exec`, `tt_device_reset` |
| Bash | Host-only work | build, git, file I/O, pip install, curl health check |

Use `[claude]$USER` as owner. Wait for job completion before proceeding.
If a job seems stuck (>5 min), ask the user before killing it.

## Environment

Write a YAML env file for every device job — don't rely on server defaults.
Pass it via tt-device-mcp's `env` parameter so the environment is explicit and
inspectable on disk.

Build the env file from three sources, in order:
1. **Workspace-detect**: paths, `$USER`, `HF_HOME`, `HF_TOKEN`
2. **tt:learn note** (`~/.tt-agent/notes/context-<target-slug>-params.md`): required
   env vars, optional vars, numeric constraint relationships, relevant paths.
   This is the authoritative source — extracted from actual source code, not recipe tables.
   If the note doesn't exist yet, invoke `tt:learn("<target> env vars and config params")`
   where target is whatever is being run (model, kernel, op, submodule, server).
3. **User request**: specific values — model name, mesh topology, batch size, etc.

Before writing the file, validate numeric constraints from the note. Constraints vary
by target — the note will describe them. Common examples:
- Size/length flags that are totals (input + output) rather than per-item
- Timeouts that must account for compile time on first run
- Concurrency flags that must match the intended load pattern

For path variables (e.g. `TT_CACHE_PATH`): check the path exists on disk before
setting — setting a non-existent path causes errors at load time, not at submission.

## Failure Recovery

**Build failure**: Report error with command and output. Don't retry — usually
needs human diagnosis.

**Test failure**: Stream `tt_device_job_logs`. Report exit code, last 50 lines,
and failure type (timeout vs assertion vs crash).

**Device hang**: `tt_device_job_kill` → if that fails, `tt_device_reset` →
report with job ID and last known state.

**Server won't start**: Check `tt_device_job_status`, stream logs, report.
