# 10 · Progress Report — Tracking and Communicating the Optimization Loop

This file is **model-agnostic**. It describes *how to run and report* an iterative
optimization loop, not what to change in any specific model. Read 01–09 for the levers.

---

## Why two artifacts

An optimization campaign is a loop, not a one-shot. To stay honest, resumable, and
recordable, keep two artifacts current **every iteration**:

1. a **results log** (machine-readable, the loop's memory), and
2. a **progress report** (human-readable, regenerated from the log).

If you only keep one, keep the log — the report is derived from it.

---

## 1. The results log (TSV)

One row per iteration. Suggested columns:

```
iteration  timestamp  commit  metric  delta  guard  guard-metric  status  description
```

Header comments record the invariants so they never drift as context grows:

```
# metric_direction: lower_is_better | higher_is_better
# target: <number>
```

`status` ∈ `baseline | keep | discard | crash | no-op`. Before starting a new
iteration, **read the last 10–20 rows + `git log --oneline`** — that is your memory of
what worked, what failed, and what is still untried.

---

## 2. The progress report (HTML)

A **self-contained** HTML file, **regenerated from the TSV** (never hand-edited — if it
is generated it cannot lie). Regenerate at each eval checkpoint and at loop end. Include:

- **Summary cards:** baseline, current, best-so-far, target, improvement %, distance to
  target, kept / discarded counts.
- **Per-iteration chart:** one bar per iteration with the **target line marked**, so the
  trajectory toward (or away from) the goal is visible at a glance.
- **Iteration table:** every row of the TSV, status colour-coded, with the one-line
  description of each change.

Keep it dependency-free (inline CSS) so it opens anywhere and travels with the repo.

---

## 3. The loop itself (high level)

```
0. Baseline   — one reproducible number from a fixed workload. Use a low-noise
                profile/device measurement to DECIDE, and a wall-clock number as the
                shipped deliverable. Record iteration 0.
1. Profile    — attribute the metric to components (ops / functions / stages), ranked
                by cost. The biggest bucket is your budget.
2. One change — a single atomic change targeting the top bucket. Commit it.
3. Verify     — re-measure. The improvement must exceed the run-to-run NOISE FLOOR to
                count as real.
4. Guard      — a correctness check (accuracy / tolerance / tests) must ALWAYS pass.
                If it fails, revert regardless of the speed win.
5. Decide     — keep (improved AND guard passed) or revert (worse / within noise /
                guard failed). Append the row to the log.
6. Re-profile — fixing one bucket promotes the next. Repeat until the target is met or
                the top buckets are irreducible (at the math/bandwidth floor).
```

---

## 4. Process hygiene (hard-won)

- **Don't keep a change inside the noise floor** — re-run to confirm before believing it.
- **One variable per iteration** — batched changes can't be attributed when results move.
- **Don't bundle tooling/results into experiment commits.** If the harness, logs, or
  report files are committed alongside a code experiment, reverting that experiment
  deletes them too. Keep them out of the tracked experiment surface, or in their own commit.
- **A sweep winner is not a win** until it is validated in the full system *and* passes
  the guard.
- **Detach long runs and sweeps in a `tmux` (or otherwise persistent) session** so an
  SSH/connection drop can't kill a multi-minute profile or a multi-hour config sweep; log
  to a file so progress stays observable while detached. Poll the log/session instead of
  blocking on the run.
- **Stop on observable state** — the measured number, the guard, the irreducible floor —
  never on a feeling that "it looks done."
