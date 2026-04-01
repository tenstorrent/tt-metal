---
name: orchestrator
description: "Route, plan, and decompose high-level Tenstorrent hardware development requests — new kernels, ops, models, CI failures, regressions, optimizations — into sequenced skill dispatches with tracked plans"
---

# TT Orchestrator

## Purpose

Translates a developer's high-level intent into a structured plan, then dispatches to
the right tt-agent skills in the right order. The orchestrator never does the work
itself — it scopes, decomposes, sequences, and verifies.

## When to Invoke

Trigger on high-level requests like:
- "Implement a new op for X"
- "Optimize the performance of Y"
- "This CI job is failing, fix it"
- "Debug why model Z is producing wrong results"
- "Profile the matmul kernel and improve throughput"
- "Add a new model to tt-metal"

If the request maps directly to a single tool (e.g., "run the profiler"), go there
directly. Use the orchestrator when the request requires multiple steps or skills.

## Pipeline

```
analyze → scope → decompose → dispatch → verify → iterate
```

1. **Analyze**: Read the request. Classify as: new build, optimize existing, fix failure,
   or investigate regression. Load `decomposer.md` for decomposition patterns.

2. **Scope**: Identify the target — kernel, op, model, or pipeline. Determine which
   hardware tier is involved (bare-metal kernel, ttnn op, model layer).

3. **Decompose**: Break the request into ordered sub-tasks. Write PLAN.md to
   `~/.tt-agent/notes`. See `decomposer.md` for standard patterns.

4. **Dispatch**: Execute sub-tasks by invoking the appropriate skills (see table below).
   Track progress in STATUS.md in `~/.tt-agent/notes`.

5. **Verify**: After each major step, confirm outputs meet the TT quality bar:
   PCC > 0.999 vs PyTorch reference, CB sizing fits L1, tile alignment correct.

6. **Iterate**: If verification fails, re-enter at step 4 with updated context.
   If stuck after 3 iterations, escalate with a detailed status report.

## Skill Dispatch Table

| Situation | Dispatch to |
|---|---|
| Profile bottleneck, find hot kernels | `/tt:profiler` |
| Optimize an existing op or kernel | `/tt:iterator` |
| CI job is failing | `/tt:ci-fixer` |
| Performance regression bisect | `/tt:bisect` |
| Build, flash, or run on device | `/tt:device` |
| Write or run tests for an op | `/tt:tester` |
| Debug wrong outputs or crashes | `/tt:debugger` |
| Design a new op or kernel | `/tt:designer` |
| Review code before merging | `/tt:code-review` |
| Need codebase context before proceeding | `/tt:learn` |

## Document Protocol

All plans and status are written to `~/.tt-agent/notes`:

| Document | Name pattern | Purpose |
|---|---|---|
| Plan | `plan-<task>.md` | Ordered sub-tasks, decision rationale |
| Status | `status-<task>.md` | Current step, blockers, findings so far |

Each document must include: date, tt-metal commit hash, skills invoked.

## Progressive Load Table

| Sub-task | Load |
|---|---|
| Decomposing new vs optimize vs fix | `decomposer.md` |
