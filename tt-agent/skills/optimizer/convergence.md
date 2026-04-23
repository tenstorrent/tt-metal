# Convergence Rules

Governs when to stop, when to ask, when to abort. Applies to both
`parameter-search` and `dataflow-optimize` subagents.

## State tracked per session

- **baseline**: device time from the Baseline phase. Fixed for the session.
- **best**: lowest device time seen so far across all iterations. Updates when
  a trial beats the previous best.
- **history**: list of `(iter, commit, workspace, metric, pcc)` per trial.
- **stall counter**: iterations since `best` last improved by ≥ threshold.

## Thresholds (defaults, overridable at session start)

| Name | Default | Meaning |
|---|---|---|
| `improvement_threshold` | 2% | A trial counts as "meaningful improvement" if `(best_prev - trial) / best_prev >= 2%`. |
| `window` | 5 | Rolling window size. If any of the last N trials improved best by ≥ threshold, keep going. |
| `stall_ask` | 10 | If best has not improved by ≥ threshold for this many iterations in a row, ask developer. |
| `pcc_abort` | 0.999 | Any trial PCC below this triggers immediate abort. |

## Per-iteration decision

After each trial:

1. Run the unit test under profiling. Record metric (device FW duration) and PCC.
2. Commit the code change with message `opt(<scope>): <hypothesis> — <metric> (<Δ%> vs best)`.
3. Append a row to `trend-<scope>.md`.
4. Emit the one-line Claude output.
5. Evaluate in order:
   - `pcc < pcc_abort` → **abort**. Keep commit for forensics. Write findings
     note with the failing commit SHA and last-good-best SHA. Stop.
   - trial beats `best` by ≥ `improvement_threshold` → update `best`, reset
     `stall counter` to 0. **Continue.**
   - trial does not beat `best` (or beats by < threshold) → increment
     `stall counter`. Check if any trial in the last `window` iterations beat
     best by ≥ threshold:
     - Yes → **continue**.
     - No and `stall counter < stall_ask` → **continue**, but no meaningful
       progress this window.
     - No and `stall counter >= stall_ask` → **ask developer**.

## Success criterion

A session ends successfully when `best` crosses the user-supplied goal:

- Absolute goal: `best <= target_ns`
- Relative goal: `best <= baseline * (1 - target_pct)`
- Roofline goal: `best / roofline_ns <= 1 / target_pct` — requires the caller
  to supply `roofline_ns` at session start (see `tt:learn` for how to compute
  it per op).

On success: write `findings-optimizer-<scope>-<ts>.md`, invoke
`tt:code-review` on the winning branch, report to the developer.

## Stall — asking the developer

Prompt template:

```
Stalled at iteration <N>. Best so far: <best> (baseline <baseline>, Δ <pct>%).
Last <stall_ask> iterations improved best by less than <threshold>%.

Recent trials:
<table of last 5 iterations: commit, hypothesis, metric>

Trend file: ~/.tt-agent/notes/trend-<scope>.md
Workspaces: <list>

Options:
  1. Continue for another <stall_ask> iterations (may stall again).
  2. Switch mode (parameter-search ↔ dataflow-optimize).
  3. Narrow scope (suggest a different hotspot from the baseline profile).
  4. Stop and accept current best.

What would you like to do?
```

Wait for the developer's choice. On timeout (no response within a
session-configurable budget, default: just wait), keep the state and stay
idle until they answer.

## PCC abort — forensics

When PCC drops below 0.999:

1. Keep the failing commit. Do not revert.
2. In the findings note, include: failing commit SHA, its diff vs the
   previous-best commit, the exact PCC value, and a note that the developer
   should treat this commit as evidence (not a fix).
3. The trend file's "best" row remains the last passing commit. The
   workspace's HEAD is at the failing commit, so the developer can check
   out or diff easily.

## Trend file format

`~/.tt-agent/notes/trend-<scope>.md` — overwritten every iteration.

```markdown
# Trend: <scope>

**Session started:** YYYY-MM-DD HH:MM:SS
**Repo:** tt-metal
**Baseline commit:** <short-sha>
**Target branch(es):** optimizer/<scope>-<date>[-a|-b|...]
**CCACHE_DIR:** <resolved path>
**Goal:** <absolute | relative | roofline>

## Current state
- Baseline: <ns>
- Best: <ns> at iter <m> on branch <letter> (commit <sha>)
- Δ baseline: -<pct>%
- Iterations: <n>
- Stall counter: <s> / <stall_ask>

## History

| Iter | WS | Commit | Metric | PCC | Δ best | Hypothesis |
|---|---|---|---|---|---|---|
| 0 (baseline) | a | abc1234 | 12.1ms | 1.0000 | — | — |
| 1 | a | def5678 | 11.5ms | 0.9999 | -5% | batch noc reads |
| ... |
```

Keep under 100 rows by truncating the oldest iterations once the table
passes that length — preserve baseline, current best, and the last 50.

## Interruption behavior

The trend file and commits are written synchronously per iteration. If the
developer interrupts (Ctrl-C, tool cancel) at any point, the session can
resume by re-reading `trend-<scope>.md` and the branch HEAD. A resume is
implicit — invoking the optimizer with the same scope and goal picks up from
the current best.
