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

The agent composes the full environment for every device job — don't rely
on server defaults. Write a YAML env file and pass it via tt-device-mcp's
`env` parameter. This makes the job's environment explicit and inspectable
on disk.

Build the env file from:
- Workspace-detect output (paths, username, HF_HOME, HF_TOKEN)
- The recipe's `env.md` for the target repo
- The user's request (model, mesh topology, etc.)

## Failure Recovery

**Build failure**: Report error with command and output. Don't retry — usually
needs human diagnosis.

**Test failure**: Stream `tt_device_job_logs`. Report exit code, last 50 lines,
and failure type (timeout vs assertion vs crash).

**Device hang**: `tt_device_job_kill` → if that fails, `tt_device_reset` →
report with job ID and last known state.

**Server won't start**: Check `tt_device_job_status`, stream logs, report.
