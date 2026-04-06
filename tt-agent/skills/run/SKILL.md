---
name: run
description: "Build, test, and execute commands on Tenstorrent devices — handles workspace detection, recipe loading, MCP routing, and job lifecycle. Use for any build/run/test action."
metadata:
  layer: tool
---

# TT Run

## Purpose

Execution engine for tt-agent. Detects the workspace, loads the right recipe,
composes commands, and routes them to the right execution target (Bash for host
commands, tt-device-mcp for device commands). Both directly invocable and used
by workflow skills as their execution backend.

## When to Invoke

- "Run this test on device"
- "Build tt-metal"
- "Start the vLLM server and run a benchmark"
- Any workflow skill that needs to build or execute
- Orchestrator dispatches here for "build, run on device"

## Pipeline

```
detect workspace → load recipe → compose command → route → execute → report
```

1. **Detect workspace**: Load `workspace-detect.md`. Identify repo, platform, arch.
2. **Load recipe**: Read `knowledge/recipes/<repo>/` files relevant to the action.
   If no recipe exists, rely on explicit user commands or tt-learn.
3. **Compose command**: Build command from recipe + user intent + env vars.
4. **Route & execute**: Device → tt-device-mcp. Host → Bash. See `execution.md`.
5. **Report**: Summarize result. On failure, include actionable context.

## MCP Routing Rule

**Device commands go through tt-device-mcp, not Bash.** This is a safety invariant.

- Device: `tt_device_job_run`, `tt_device_job_run_bg`, `tt_device_exec`, `tt_device_reset`
- Host: Bash — build, git, file I/O, pip install, server health checks

When in doubt: if the command needs a TT device attached, use MCP.

## Progressive Load Table

| Sub-task | Load |
|---|---|
| Detect workspace, repo, platform, arch | `workspace-detect.md` |
| Workspace setup and activation | `knowledge/recipes/workspace.md` |
| Command routing, env handoff, recovery | `execution.md` |
| Build steps for detected repo | `knowledge/recipes/<repo>/build.md` |
| Test invocation for detected repo | `knowledge/recipes/<repo>/test.md` |
| Server lifecycle (vLLM) | `knowledge/recipes/vllm/server.md` |
| Benchmark (vLLM) | `knowledge/recipes/vllm/benchmark.md` |
| Environment variables | `knowledge/recipes/<repo>/env.md` |
