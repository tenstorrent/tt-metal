---
name: optimizer
description: "Autonomous kernel/op optimization loop — profiles, hypothesizes, implements, measures, commits per iteration. Operates single or multi-workspace. Use to push a specific op's device time toward a goal."
metadata:
  layer: workflow
---

# TT Optimizer

## Purpose

Drive a developer's performance goal on a specific op or kernel to convergence
without supervision. The loop: profile → hypothesize → implement → profile
again → record → repeat. Every iteration lands as a git commit on a dedicated
branch so the full trajectory is inspectable and any point is recoverable.

Two loop modes:
- **Parameter search** — structured sweep over a discrete config space (block
  sizes, shard configs, CB depths). Single workspace, one build, many runs.
- **Data flow optimization** — open-ended code changes (barrier placement,
  CB sizing, NOC batching, fusion). One workspace per hypothesis. Can run
  multiple hypotheses in parallel as separate workspaces.

**Why `layer: workflow`:** runs until a convergence criterion fires; iterates
through phases; owns long-running state.

## When to Invoke

- "Optimize <op> for device time"
- "Get <kernel> to within X% of roofline"
- "Make this matmul faster"
- `tt:orchestrator` dispatches here for "Profile bottleneck, optimize throughput"

## Preflight: developer-rule conflict

This skill commits per iteration and may create new workspaces. Before starting
work, confirm:

1. State plainly: *"I will commit every iteration to a dedicated branch and
   may create N parallel workspaces. I will never push to a remote."*
2. Check the developer's CLAUDE.md (global and project) for rules conflicting
   with autonomous commits or workspace creation. If found, surface the
   conflict, quote the rule, ask the developer to override or adjust scope.
3. Wait for explicit confirmation before proceeding.

See `skills/skill-creator/tt-guidelines.md` Developer-Rule Conflict Protocol.

## Inputs

- **Target**: op or kernel name, and a test that exercises it (pytest path +
  optional `-k` filter). If the user only names a model, the Extract phase
  handles isolating the bottleneck.
- **Goal**: one of
  - absolute device time in ns (e.g., "under 8ms")
  - relative improvement (e.g., "30% faster than baseline")
  - % of roofline (e.g., "reach 70% of theoretical peak")
- **Mode** (optional): `parameter-search` or `dataflow-optimize`. If omitted,
  the skill picks based on the target: an op with an explicit config struct
  and a well-defined parameter space defaults to parameter search; otherwise
  data flow.
- **Parallelism** (optional, default 1): number of hypotheses to run
  concurrently in separate workspaces. >1 only valid for `dataflow-optimize`.

## Phase Table

| Phase | Loads / dispatches | Produces |
|---|---|---|
| Preflight | Developer-rule check (see above) | Go / no-go |
| Prepare | `skills/run/workspace-detect.md` (via tt:run), invoke `tt:learn("<target>")` | Workspace context note, target research note |
| Extract (if no unit test exists) | dispatch `extract.md` subagent | Extracted unit test, saved input tensors, ±10% verification pass |
| Baseline | invoke `tt:profiler` on the unit test | Baseline profile note, initial row in `trend-<scope>.md` |
| Spawn workspaces (multi-hypothesis only) | `workspaces.md` | N workspaces, N branches, ccache shared |
| Iterate | dispatch `parameter-search.md` OR `dataflow-optimize.md` subagent(s); each calls `tt:run` + `tt:profiler` per trial; writes commit + trend row per trial | Per-iteration commits, profile notes, updated trend file |
| Converge | `convergence.md` | Success / stall-asks-user / PCC-abort |
| Review | invoke `tt:code-review` on the winning branch's diff | Review note, final summary |

After each phase, summarize in 3-5 lines in the trend file and move on. Phase
inputs are consumed, not carried forward.

## Outputs

Written to `~/.tt-agent/notes/`:

| File | Purpose |
|---|---|
| `trend-<scope>.md` | Single-glance trajectory. Overwritten every iteration. Columns: iter / commit / workspace / metric / Δ baseline / Δ best / notes. |
| `profile-<scope>-<ts>.md` | One per iteration, written by `tt:profiler`. |
| `findings-optimizer-<scope>-<ts>.md` | Final report on convergence or stop. Summary, winning commit(s), path(s) to workspaces/branches, cleanup instructions. |

Written to the repo (commits):

- One commit per iteration on branch `optimizer/<scope>-<YYYY-MM-DD>[-<letter>]`.
- Commit message format: `opt(<scope>): <short hypothesis> — <metric> (<Δ%> vs best)`.
- Never pushed.

## Per-Iteration Claude Output

One line per iteration, visible to the developer in real time:

```
Iter <n> [<workspace-letter>] <commit-sha>: <metric> (baseline <B>, Δbest <X%>, best@iter <m>)
```

Example:
```
Iter 7 [a] abc1234: 8.2ms (baseline 12.1ms, Δbest -3%, best@iter 5)
```

## Convergence

See `convergence.md`. Summary:

- Keep going if best-so-far improved ≥ 2% in the last 5 iterations.
- At 10 iterations without sufficient improvement, write a checkpoint note and
  ask the developer for further direction.
- Abort immediately if a trial's PCC drops below 0.999 — commit is kept for
  forensics; trend is rolled back to previous best.

No hard iteration cap. The developer can interrupt at any time; the trend
file and commits are up to date after every iteration.

## Progressive Load Table

| Sub-task | Load |
|---|---|
| Per-iteration rules, trend file format, commit protocol | `convergence.md` |
| Spawn parallel workspaces, ccache setup, cleanup reporting | `workspaces.md` |
| Model → unit test + tensor capture | `extract.md` |
| Structured parameter sweep loop | `parameter-search.md` |
| Open-ended hypothesis loop | `dataflow-optimize.md` |
| Build, test, profile execution | invoke `tt:run` and `tt:profiler` |
| Target research, volatile APIs | invoke `tt:learn("<question>")` |
| Final review on winning branch | invoke `tt:code-review` |

## Caller Contract

- **Developer**: reads `trend-<scope>.md` anytime. On stall prompt, responds
  with new direction or stop. At success, cherry-picks or merges the winning
  commit; otherwise deletes the branches and workspaces per the cleanup
  instructions in the findings note.
- **Orchestrator**: dispatches here with target + goal; reads the findings
  note on completion to feed back into a larger plan.
