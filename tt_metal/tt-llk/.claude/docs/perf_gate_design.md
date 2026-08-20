# LLK Perf Regression Gate — Design

**Status:** Design proposal for discussion and team approval.

**Goal:** Catch performance regressions in LLK kernels at PR merge time, before they land on main.

---

## Overview

The gate compares perf measurements from a PR branch against a baseline from main, flags regressions, and blocks the merge if any are found. The comparison is **per measurement point** — one row per `(marker, run_type, sweep_config)` — so all test variants are independently gated, not just pass/fail on the whole suite.

**Rule:** A point is a regression when it is **more than 2% slower AND more than 30 cycles slower** than the baseline. Both clauses must hold.

---

## Why this rule

The thresholds come from five-run noise baselines on both Blackhole and Wormhole architectures:

- **2% threshold:** Smallest slow-down we can reliably detect. Across 108,377 identical runs of Blackhole's `L1_TO_L1` (no code change), the worst movement was 1.88% — so 2% sits above all noise and catches real regressions.
- **30 cycles floor:** Suppresses false positives on small markers. `INIT` and `UNINIT` are ~350 cycles and jitter by ±25 cycles, which looks like ±7% — a catastrophic false-positive rate if we gated on percentage alone. The 30-cycle clause blocks jitter while letting real regressions through. On the big markers (`TILE_LOOP`, `KERNEL`), the worst noise was 5,110 cycles at 1.88%, so the cycle floor is no constraint.

**Reference:** See `tt_metal/tt-llk/docs/perf_evaluation/results/blackhole-nonsol/README.md` and `wormhole-nonsol/README.md` for full baseline derivation and per-run-type breakdowns.

---

## Architecture

### Data Flow

1. **Baseline:** Query Snowflake for the latest perf run of main (nightly or post-merge)
2. **Current:** Run perf sweep on the PR branch (compile-producer + compile-consumer)
3. **Compare:** Run `perf_regression_compare.py` — reads both CSV sets, compares medians, flags regressions
4. **Report:** Write Markdown report + CSV of all points, regressions, improvements to the PR

### Pipeline Integration

The gate is a **separate CI step** that runs **after** `pytest --compile-consumer` completes and `perf_data/` is written.

```
[ compile-producer ] ──> [ compile-consumer ] ──> [ perf-regression-gate ] ──> [merge decision]
   (parallel -n 10)        (on hardware -n 15)      (query Snowflake,        (hard block if
                                                     compare CSVs,            regressions)
                                                     report results)
```

The gate uses the same perf measurement that developers and on-call already run, so there is no new hardware time — it is a post-processing step on the CSVs produced by `perf_data`.

### Baseline Management

Baselines live in **Snowflake**, keyed by `(arch, commit_sha)`. The schema captures one row per perf run:

```
archive {
  arch: 'wormhole' | 'blackhole' | 'quasar',
  commit_sha: '59602f...',
  run_type: 'nightly' | 'pr_merge' | 'pr_gate',
  timestamp: '2026-08-20T...',
  perf_data: { test_name -> marker -> run_type -> mean(...) }  # or ref to GCS blob
}
```

A PR gate query: `SELECT perf_data FROM archive WHERE arch = ? AND commit_sha = ? ORDER BY timestamp DESC LIMIT 1` — gets the latest measurement for the branch point on main.

**Who updates baselines:**
- **Nightly runs:** Automated cron, every day at end of shift
- **Post-merge:** CI job after a PR merges to main, captures the merged commit
- **PR gate:** Reads, does not write

### Regression Report

When regressions are found, the gate writes:

1. **Markdown report** — short summary + top 25 regressions + improvements, posted to the PR
2. **CSV of all points** — full output, every point that was compared
3. **CSV of regressions only** — the points that failed

Example output:

```
# Perf compare — llk

❌ REGRESSIONS FOUND

Rule: a point is a regression when it is **more than 2% slower AND more than 30 cycles slower**.
Both must hold. Comparison is median-vs-median, per (marker, run_type, sweep config).

- baseline (main): `abc123` — 1 iteration(s)
- current (branch): `def456` — 5 iteration(s)
- 108,377 points compared, **3 regression(s)**, 0 improvement(s), 0 new point(s)

## Top 3 regressions (slower on current)

| marker | run type | current | baseline | Δ | Δ cycles | config |
|---|---|--:|--:|--:|--:|---|
| TILE_LOOP | L1_TO_L1 | 3200 | 3100 | +3.2% | +100 | tile_cnt=8, loop_factor=4 |
| ... |
```

---

## Open Questions for Rose & Team

### 1. Baseline Versioning

When do we snapshot main as the new baseline? Options:

- **Post-merge (current proposal):** Every PR that merges to main triggers a baseline snapshot. Gate always compares against the absolute latest main.
  - Pro: Gate is always tight, catches regressions as soon as they land
  - Con: A PR that lands with a real regression pollutes the baseline, and subsequent PRs inherit it

- **Nightly only:** Baselines update once per day (e.g., 5pm EOD). Gate always compares against the last nightly.
  - Pro: Baselines are stable within the day, regressions don't propagate
  - Con: If a regression lands early in the day, it sits in main for hours before the baseline is updated

- **Both:** Post-merge captures fast feedback (gate blocks), but nightly is the "official" baseline used for dashboards/reports.

**Recommendation:** Post-merge, with a dashboard showing "since last baseline update" vs "vs yesterday's nightly" — so we catch regressions fast but also know how we stand vs stable state.

### 2. Measurement Count

The baseline query returns **one measurement per commit** (the latest run). The current PR run uses **5 iterations** (from perf suite), so the comparison is **1 iteration vs 5 iterations** — the median of 5 beats the median of 1.

Is this acceptable, or should we:

- Require baseline runs to also be 5 iterations (more expensive, more stable)?
- Use a single iteration for both (cheaper, noisier)?
- Use a weighted comparison (current = mean of 5, baseline = single measurement scaled by noise envelope)?

---

## Implementation Roadmap

### Phase 1 (this PR)
- [ ] Design document (this file) — frozen for team review
- [ ] `perf_regression_compare.py` — already in repo, ready to use
- [ ] Snowflake schema draft — schema for baseline table, queries, retention policy
- [ ] CI integration sketch — `ci.yml` template showing where the step goes
- [ ] **Decision:** Answer the three questions above

### Phase 2 (follow-up PR)
- [ ] Snowflake schema + loader
- [ ] Baseline snapshot job (on post-merge to main)
- [ ] Gate CI step (queries Snowflake, runs compare, posts report)
- [ ] Draft PR gate policy (when gate is enabled, which configs are gated, severity handling)

### Phase 3 (after team approval)
- [ ] Enable gate on all LLK PRs
- [ ] Monitor for false positives / tuning
- [ ] Dashboard integration (show baseline age, gate health, regressions caught)

---

## Example: A Real Regression

A developer optimizes `llk_math_eltwise` — shaves 50 cycles off the kernel. But they introduce a subtle race in the pack thread, and under contention, pack stalls. On average it's +60 cycles, but TILE_LOOP moves from 50,000 to 50,120 cycles — that is +0.24% and +120 cycles.

The gate fires: `0.24% < 2%` ✗ but `+120 > 30` ✓ → **regression**. PR is blocked, developer debugs, finds the race, fixes it.

Without the gate, the 0.24% regression lands on main and accumulates. Three months later, no single change broke anything — but 50 small regressions add up, and the kernel is 10% slower.

---

## Thresholds for Other Architectures

- **Wormhole:** Same rule holds — 0 false positives on four of five measured configurations. 53 outliers on `L1_TO_L1` matmul (different run each time, suggesting measurement concurrency effects, not hardware). **Decision needed:** Gate separately on matmul, or accept these as-is?
- **Quasar:** Not yet measured. Noise baseline needed before gate can be enabled.

---

## Non-Goals (Out of Scope)

- **Power, area, or build-time regressions** — orthogonal tools, different metrics
- **Correctness gates** — existing test harness covers this
- **Relative comparison across architectures** — Blackhole vs Wormhole are different parts of the stack
- **Real-time dashboard updates** — handled separately by the data / observability team
