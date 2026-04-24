---
name: optimizer
description: "Autonomous kernel/op optimization loop — profiles, hypothesizes, implements, measures, commits per iteration. Operates single or multi-workspace. Use to push a specific op's device time toward a goal."
metadata:
  layer: workflow
---

# TT Optimizer

## Purpose

Drive a performance goal on a specific op or kernel to convergence without
supervision. The loop: profile → hypothesize → implement → profile →
record → repeat. Every iteration lands as a git commit on a dedicated
branch so the trajectory is inspectable and any point recoverable.

Two iteration modes (in `iterate.md`):
- **parameter-search** — discrete sweep over a config space. Single
  workspace, one build, many runs.
- **dataflow-optimize** — open-ended code changes (barriers, CB sizing,
  NOC batching, fusion). One workspace per hypothesis; parallel across
  workspaces is supported.

## When to Invoke

- "Optimize <op> for device time"
- "Get <kernel> to within X% of roofline"
- "Make this matmul faster"
- `tt:orchestrator` dispatches here for "Profile bottleneck, optimize throughput"

## Preflight: developer-rule conflict

This skill commits per iteration and may create new workspaces. Before
starting:

1. State plainly: *"I will commit every iteration to a dedicated branch
   and may create N parallel workspaces. I will never push to a remote."*
2. Check the developer's CLAUDE.md (global and project) for rules
   conflicting with autonomous commits or workspace creation. Surface
   the conflict, quote the rule, ask for override or scope adjustment.
3. Wait for explicit confirmation before proceeding.

## Preflight: tool preload

Fetch these schemas via `ToolSearch` before baseline — loading mid-session
during a CCL hang blocks for the full test timeout (10+ min) before the
reset tool is callable:

- `select:tt_device_reset` — for CCL hangs or watchdog locks.
- `select:tt_device_job_run,tt_device_job_run_bg,tt_device_job_wait,tt_device_job_kill,tt_device_job_logs` — tt:run dispatches through these.

Mandatory when a CCL hypothesis is on the queue. Record load in the first
trend-file entry.

## Inputs

- **Target**: op or kernel name + test (pytest path + optional `-k`). If
  the user names only a model, the Extract phase isolates the bottleneck.
- **Goal**: absolute ns, relative %, roofline %, or utilization %. Formulas
  and guidance on picking the right type live in `convergence.md` §
  Success criterion.
- **Mode** (optional): `parameter-search` or `dataflow-optimize`. Default:
  parameter-search if op has an explicit config struct and bounded
  parameter space; else dataflow-optimize.
- **Parallelism** (optional, default 1): concurrent hypotheses in separate
  workspaces. >1 only valid for `dataflow-optimize`.

## Phase Table

| Phase | What happens | Procedure | Note produced |
|---|---|---|---|
| Preflight | Check dev-rule conflicts, preload tools, sweep docs | inline above | — |
| Prepare | Research target, sibling-diff scan, dim validation, doc sweep | `prepare.md` | `prepare-<scope>-<ts>.md` |
| Extract* | Capture model inputs, write unit test, verify ±10% | `extract.md` | test path + tensor path |
| Baseline | Profile the unit test | `tt:profiler` | `profile-<scope>-<ts>.md`, baseline row |
| Spawn* | Clone + branch + build per hypothesis | `workspaces.md` | N workspaces, N branches |
| Iterate | Hypothesize → implement → build → profile → commit → record | `iterate.md` (+ `playbook.md`, `convergence.md`) | commits, profile notes, trend rows |
| Review | Review-to-done loop on winning branch | `skills/code-review/review-loop.md` | `findings-review-<ts>-<scope>.md` |

\* conditional — Extract runs only if no unit test exists; Spawn only for parallelism > 1.

After each phase, summarize in 3-5 lines in the trend file and move on.
Phase inputs are consumed, not carried forward.

## Outputs

`~/.tt-agent/notes/`:

| File | Purpose |
|---|---|
| `trend-<scope>.md` | Single-glance trajectory. Overwritten every iteration. See `convergence.md` for format. |
| `profile-<scope>-<ts>.md` | One per iteration, written by `tt:profiler`. |
| `findings-optimizer-<scope>-<ts>.md` | Final report: summary, winning commit(s), workspace paths, cleanup instructions. |

Repo:

- One commit per iteration on branch `optimizer/<scope>-<YYYY-MM-DD>[-<letter>]`.
- Commit subject format: see `iterate.md` § Record.
- Never pushed.

## Convergence

See `convergence.md`. Summary:

- Continue while best improved ≥ 2% in the last 5 iterations.
- At 10 iterations without improvement, ask the developer.
- PCC < 0.999 → immediate abort. Commit kept; trend rolls back to prior best.

No hard iteration cap. Developer can interrupt anytime; trend file and
commits are current after every iteration.

## Caller Contract

- **Developer**: reads `trend-<scope>.md` anytime. On stall prompt,
  responds with direction or stop. At success, cherry-picks or merges
  the winning commit; otherwise deletes branches and workspaces per
  the findings note.
- **Orchestrator**: dispatches here with target + goal; reads the
  findings note on completion.
