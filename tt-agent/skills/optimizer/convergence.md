# Convergence Rules

Governs when to stop, ask, or abort. Applies to both modes of `iterate.md`.

## State tracked per session

- **baseline**: device time from the Baseline phase. Fixed.
- **best**: lowest device time seen. Updates when a trial beats it.
- **history**: list of `(iter, commit, workspace, metric, pcc)`.
- **stall counter**: iterations since `best` last improved by ≥ threshold.

## Thresholds (defaults, overridable at session start)

| Name | Default | Meaning |
|---|---|---|
| `improvement_threshold` | 2% | Trial is "meaningful" if `(best_prev − trial) / best_prev ≥ 2%`. |
| `window` | 5 | Rolling window. If any of the last N trials beat best by ≥ threshold, keep going. |
| `stall_ask` | 10 | Iterations with no ≥threshold improvement → ask developer. |
| `pcc_abort` | 0.999 | Trial PCC below this → immediate abort. |

## Decision after each trial

- `pcc < pcc_abort` → **abort**. Keep commit for forensics. Findings
  note with failing SHA and last-good-best SHA. Stop.
- trial beats `best` by ≥ `improvement_threshold` → update `best`, reset
  stall counter. **Continue.**
- else → increment stall counter. If any trial in the last `window` beat
  best by ≥ threshold → **continue**. Else if `stall counter < stall_ask`
  → **continue** (no progress this window). Else → **ask developer**.

## Success criterion

`best` crosses the supplied goal:

- **Absolute**: `best ≤ target_ns`
- **Relative**: `best ≤ baseline × (1 − target_pct)`
- **Roofline**: `best / roofline_ns ≤ 1 / target_pct` (caller supplies
  `roofline_ns`; see `tt:learn`).
- **Utilization**: `flops_pct ≥ target_flops_pct` AND `target_op_fw_ns ≥
  sum(non_target_op_fw_ns)` (target op is THE bottleneck). `flops_pct` /
  `dram_pct` come from `tt-perf-report` (see
  `skills/profiler/interpretation.md`). Use when baseline is low-utilization
  — a 30% speedup at 25% FLOPs hasn't fixed the op.

On success: write `findings-optimizer-<scope>-<ts>.md`, invoke
`tt:code-review` via `review-loop.md` on the winning branch, report.

## Bound-ceiling exit (utilization-goal only)

If the op's bound class stays `overhead / sync-bound` with FLOPs% < 40%
after 3 productive iterations (FLOPs% under 35% in every trial, and
remaining levers — `in0_block_w`, `per_core_M`, L1 sharding — tried or
ruled out by L1 budget), write a `kernel-family-ceiling` checkpoint and
ask:

```
Reached kernel-family ceiling at iter <N>. Best <best> at <flops%>F /
<dram%>D / overhead. Current variant (<2D MC | 1D ring | DRAM-sharded>)
appears structurally overhead-bound on this shape.

Options:
  1. Switch to <alternative variant> as a parameter-search sweep.
  2. Accept current best (structural change out of scope).
  3. Broaden scope to adjacent ops (AllGather, tilize, ...).
```

Prevents long subblock fiddling on a misclassified goal.

## Stall prompt

```
Stalled at iteration <N>. Best: <best> (baseline <baseline>, Δ <pct>%).
Last <stall_ask> iterations improved best by less than <threshold>%.

Recent trials:
<table of last 5: commit, hypothesis, metric>

Trend file: ~/.tt-agent/notes/trend-<scope>.md
Workspaces: <list>

Options:
  1. Continue for another <stall_ask> iterations.
  2. Switch mode (parameter-search ↔ dataflow-optimize).
  3. Narrow scope.
  4. Stop and accept current best.
```

Wait for developer choice. On timeout, keep state and stay idle.

## PCC abort forensics

1. Keep the failing commit — do not revert.
2. Findings note: failing SHA, diff vs previous-best, exact PCC value,
   note to treat as evidence (not a fix).
3. Trend `best` row stays at last passing commit. Workspace HEAD is at
   failing commit for easy inspection.

## Trend file format

`~/.tt-agent/notes/trend-<scope>.md` — overwritten every iteration.

```markdown
# Trend: <scope>

**Session started:** YYYY-MM-DD HH:MM:SS
**Repo:** tt-metal
**Baseline commit:** <short-sha>
**Target branch(es):** optimizer/<scope>-<date>[-a|-b|...]
**CCACHE_DIR:** <resolved path>
**Goal:** <absolute | relative | roofline | utilization>

## Current state
- Baseline: <ns> · Best: <ns> at iter <m> on branch <letter> (<sha>)
- Δ baseline: -<pct>% · Utilization (best): <flops%>F / <dram%>D / <bound> / <cores> cores
- Iterations: <n> · Stall counter: <s> / <stall_ask>

## History

| Iter | WS | Commit | Metric | PCC | Δ best | FLOPs% | DRAM% | Bound | Cores | Hypothesis |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 (baseline) | a | abc1234 | 12.1ms | 1.0000 | — | 36% | 11% | overhead | 64 | — |

## Per-iteration contribution breakdown

| # | Change | Saved | % of baseline | Running total |
|---|---|---|---|---|

## Forensic failures

| # | Iter | What was tried | Result |
|---|---|---|---|

## Parameter sweeps (per knob, when non-monotonic)

| Knob | Values tried | Best value | Notes |
|---|---|---|---|

## Op-level timing (context, optional)

| Op | Baseline | Current | Δ |
|---|---|---|---|
```

Rule: anything shown in chat must also land here. Ephemeral chat tables rot.

Truncate History at 100 rows (preserve baseline, current best, and the
last 50). Contribution and forensic tables never truncated.

## Interruption

Trend file and commits are written synchronously per iteration. Resume by
re-reading `trend-<scope>.md` and the branch HEAD — invoking the optimizer
with the same scope and goal picks up from current best.
