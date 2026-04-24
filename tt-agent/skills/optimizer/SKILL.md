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

## Preflight: tool preload

Before the first iteration, load the deferred tools this loop will need. If
loaded mid-session, a CCL deadlock or hung test will block for the full
test timeout (10+ minutes) before the reset tool becomes callable.

Required schemas to fetch via `ToolSearch` before baseline:
- `select:tt_device_reset` — for CCL hangs or watchdog-triggered device locks.
- `select:tt_device_job_run,tt_device_job_run_bg,tt_device_job_wait,tt_device_job_kill,tt_device_job_logs` — tt:run dispatches through these.

State that these are loaded in the first trend-file entry. If a CCL or CCL-
tuning hypothesis is on the queue, this is mandatory — not optional.

## Inputs

- **Target**: op or kernel name, and a test that exercises it (pytest path +
  optional `-k` filter). If the user only names a model, the Extract phase
  handles isolating the bottleneck.
- **Goal**: one of
  - absolute device time in ns (e.g., "under 8ms")
  - relative improvement (e.g., "30% faster than baseline")
  - % of roofline (e.g., "reach 70% of theoretical peak")
  - **utilization target** (e.g., "≥70% FLOPs% on target matmul with matmul
    as THE bottleneck") — requires the `DRAM %` / `FLOPs %` columns from
    `tt-perf-report` (see `skills/profiler/interpretation.md`). Prefer this
    over relative-improvement when the baseline is already low-utilization;
    a 30% speedup that leaves the op at 25% FLOPs hasn't actually fixed it.
- **Mode** (optional): `parameter-search` or `dataflow-optimize`. If omitted,
  the skill picks based on the target: an op with an explicit config struct
  and a well-defined parameter space defaults to parameter search; otherwise
  data flow.
- **Parallelism** (optional, default 1): number of hypotheses to run
  concurrently in separate workspaces. >1 only valid for `dataflow-optimize`.

## Phase Table

| Phase | Loads / dispatches | Produces |
|---|---|---|
| Preflight | Developer-rule check (see above); ask about internal docs (see Prepare §5) | Go / no-go + doc pointers |
| Prepare | Expanded sub-steps (see "Prepare phase" section below); `skills/run/workspace-detect.md` (via tt:run); invoke `tt:learn("<target>")` (REQUIRED, not optional) | Workspace context note, target research note, dim-validation findings, authoritative-doc pointers |
| Extract (if no unit test exists) | dispatch `extract.md` subagent | Extracted unit test, saved input tensors, ±10% verification pass |
| Baseline | invoke `tt:profiler` on the unit test | Baseline profile note, initial row in `trend-<scope>.md` |
| Spawn workspaces (multi-hypothesis only) | `workspaces.md` | N workspaces, N branches, ccache shared |
| Iterate | dispatch `parameter-search.md` OR `dataflow-optimize.md` subagent(s); each calls `tt:run` + `tt:profiler` per trial; writes commit + trend row per trial | Per-iteration commits, profile notes, updated trend file |
| Converge | `convergence.md` | Success / stall-asks-user / PCC-abort |
| Review | invoke `skills/code-review/review-loop.md` (not a single review — the escalating done-gate) on the winning branch's diff | Clean-review findings note or documented abort, then final summary |

After each phase, summarize in 3-5 lines in the trend file and move on. Phase
inputs are consumed, not carried forward.

## Prepare phase (not optional)

Before the first baseline profile:

1. **Research the target op.** Invoke `tt:learn("<target> — config knobs,
   valid dim constraints, kernel variants")`. Record the note path in the
   trend file. Do not skip — the first iteration's hypothesis queue depends
   on this.

2. **Sibling-implementation scan.** If the target file is "reused from X
   with changes" (very common in tt-metal — most multimodal and model
   variants extend a Llama reference), read X and diff it against the
   target. The original's config patterns may not apply to the new model's
   dimensions. Note any divergent dims, config lambdas, or cached `args.*`
   fields.

   **Check X's mode branch** before borrowing structural choices. tt-metal
   models often branch on `mode=="prefill" | "decode"` with very different
   memory layouts. If the target runs in PREFILL, do not borrow X's DECODE
   progcfg or sharding. `grep` the sibling for the mode keyword and trace
   which progcfg the target's shape would select. See `common-wrong-turns.md`
   § "Check the sibling implementation's mode before committing to a
   structural change".

3. **Actual-vs-derived dim check.** For every `args.<dim>` used in a
   progcfg, compare to `self.<weight>.shape[-1]` of the loaded tensor.
   `ModelArgs` fields are computed at config time and can truncate
   (int-div) or drift from actual weight shape. Prefer the weight-shape
   value in progcfgs. If a mismatch exists, flag it in the trend file —
   a standalone fix PR may be warranted.

4. **Authoritative docs sweep.** Ask the developer whether internal guides
   exist (Confluence pages, `tech_reports/`, PDF exports in the workspace).
   For matmul the PSE Matmul Configuration Guide governs variant selection,
   L1 budgeting, and subblock rules — reading it before iterating reshapes
   the hypothesis queue substantially.

5. **Known anti-patterns.** Skim `common-wrong-turns.md`. Entries there
   cost other sessions full iterations to discover.

Output: a short Prepare note in `~/.tt-agent/notes/` listing the research
pointer, sibling diff highlights, dim-validation findings, and doc pointers.
The Baseline phase reads this.

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
Iter <n> [<workspace-letter>] <commit-sha>: <metric> (baseline <B>, Δbest <X%>, best@iter <m>) · <FLOPs%>F / <DRAM%>D / <Bound>
```

Example:
```
Iter 7 [a] abc1234: 8.2ms (baseline 12.1ms, Δbest -3%, best@iter 5) · 44%F / 18%D / overhead
```

**Rule: anything computed for display lands in the trend file in the same
iteration.** Utilization snapshots, contribution breakdowns, per-op timing
comparisons — if they're worth showing in chat, they're worth preserving
in the trend file. Ephemeral tables in chat rot; trend-file tables survive.
See `convergence.md` for the exact columns.

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
| Known anti-patterns and debug techniques (read before iterating) | `common-wrong-turns.md` |
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
