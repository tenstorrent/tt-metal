# LLK Perf Regression Gate — Design

**Status:** Design proposal for discussion and team approval. Ready for implementation once team aligns on open questions.

**Goal:** Catch performance regressions in LLK kernels at PR merge time, before they land on main.

---

## Executive Summary

**Rule:** Flag a regression when a measurement is **more than 2% slower AND more than 30 cycles slower** than main.

**Cost:** ~9 minutes for `L1_TO_L1`, ~19 minutes for isolates on Wormhole (the slower architecture).

**Threshold maturity:**
- `MATH_ISOLATE`, `UNPACK_ISOLATE`: **Gate-ready** (zero false positives on both Blackhole and Wormhole)
- `L1_TO_L1`: **Ready on Blackhole**, carries 53 matmul false positives on Wormhole
- `PACK_ISOLATE`: **Not ready** — blocks on investigation of run-to-run instability (see "The one blocker" below)

---

## Overview

The gate compares perf measurements from a PR branch against a baseline from main, flags regressions, and blocks the merge if any are found. The comparison is **per measurement point** — one row per `(marker, run_type, sweep_config)` — so all test variants are independently gated, not just pass/fail on the whole suite.

**Rule:** A point is a regression when it is **more than 2% slower AND more than 30 cycles slower** than the baseline. Both clauses must hold.

---

## Measured baselines

We ran identical code five times on one card (Speed of Light off) and measured how much each number moved. Nothing changed between runs, so every difference is noise.

### Cost: Gate latency (wall clock, cold build)

| config | Blackhole | Wormhole |
|---|--:|--:|
| full — every run type | 19:41 | 30:14 |
| isolates — unpack, math, pack | 12:40 | 18:45 |
| L1_TO_L1 only | 6:17 | 9:19 |

The two architectures run on separate CI runners in parallel, so gate latency is the Wormhole figure: about **9 minutes** for `L1_TO_L1`, **19 minutes** for isolates. Compile is 65-79% of every configuration, so build caching on the gate runner would save more than any run-type choice.

### Threshold: How much is noise?

| arch | configuration | measurements | rule fires on unchanged code |
|---|---|--:|--:|
| Blackhole | L1_TO_L1 | 108,377 | **0** |
| Blackhole | MATH_ISOLATE | 101,180 | **0** |
| Blackhole | UNPACK_ISOLATE | 103,424 | **0** |
| Blackhole | PACK_ISOLATE | 106,748 | 2 |
| Wormhole | L1_TO_L1 | 100,971 | 53 |
| Wormhole | MATH_ISOLATE | 94,692 | **0** |
| Wormhole | UNPACK_ISOLATE | 96,846 | **0** |
| Wormhole | PACK_ISOLATE | 99,414 | 1,457 |

Both clauses are necessary:
- **Percentage clause alone** fails on `INIT`, which is a few hundred cycles and wobbles by 20 — looks like 5-7%, a catastrophic false-positive rate
- **Cycle count alone** fails on `TILE_LOOP`, which moves thousands of cycles and is still under 2%

**Repeating runs does not help.** Averaging two runs per side does not improve the typical case at all, only the extreme tail. Use one run per side.

---

## Architecture

### Data Flow

1. **Baseline:** Query Snowflake for the latest perf run of main (nightly or post-merge)
2. **Current:** Run perf sweep on the PR branch (compile-producer + compile-consumer)
3. **Compare:** Run `perf_regression_compare.py` — reads both CSV sets, compares medians, flags regressions
4. **Report:** Write Markdown report + CSV of all points, regressions, improvements to the PR

### Pipeline Integration

The gate is a **separate CI step** that runs **after** `pytest --compile-consumer` completes and `perf_data/` is written. It compares against the Snowflake baseline and reports regressions.

```
[ compile-producer ] ──> [ compile-consumer ] ──> [ perf-regression-gate ] ──> [merge decision]
   (parallel -n 10)        (on hardware -n 15)      (query Snowflake,        (hard block if
                                                     compare CSVs,            regressions)
                                                     report results)
```

After merge to main, the **sanity workflow** re-runs perf tests and publishes the results as the new baseline to Snowflake. This ensures:
- Baseline measurements are taken after the code is merged (not speculatively from a PR)
- Time budget is generous enough for multiple iterations if needed
- New baseline is ready for the next PR's gate comparison

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

## What is gate-ready today

| configuration | verdict |
|---|---|
| MATH_ISOLATE, UNPACK_ISOLATE | **Ready.** Zero false positives on both architectures |
| L1_TO_L1 | Ready on Blackhole. On Wormhole it carries 53 matmul points that trigger false positives |
| PACK_ISOLATE | **Blocked.** See "The one blocker" below |

### The one blocker: the packer

Every failure in the study is the packer path.

- **Wormhole PACK_ISOLATE:** 1,457 failures, up to 24% and 41,091 cycles. 1,246 of them are `perf_matmul`.
- **Wormhole L1_TO_L1:** 53 failures, all matmul. That path runs unpack, math and pack in sequence, so the instability leaks in.
- **Blackhole PACK_ISOLATE:** 2 failures, one test configuration (`perf_pack_dest_bank`), bimodal between the first run and the rest.

On Wormhole it is not a first-run artefact: any of the five runs can be the odd one out. It is genuine run-to-run instability.

**No usable threshold can absorb a 24% swing.** This needs an investigation before the packer can be gated.

---

## Open Questions for Rose & Team

### 1. Why is the packer unstable, and worse on Wormhole?

One untested hypothesis: the measure phase runs 15 tests concurrently on different Tensix cores, so a long kernel overlaps with its neighbours and sees whatever L1 and NoC contention occurs. The affected points are the largest ones. Measuring serially would confirm or eliminate it.

**Proposed resolution:** Run one serial L1_TO_L1 noise baseline on Wormhole with `-n 1` to test the hypothesis. ~10 minutes card time.

### 2. How big are real regressions?

These threshold numbers constrain the gate from below only (we know noise is below 2% AND 30 cycles on most configs). The upper bound has to come from the commit history: what is the smallest regression that actually mattered, and when did it land on main?

**Proposed resolution:** Team review of recent LLK commits to find examples.

### 3. Cross-machine and cross-day drift

Every run here was one card in one session. A real gate compares different runners on different days. These baseline numbers only apply to one machine on one day.

**Decision needed:** Do we:
- Re-measure monthly to account for hardware drift?
- Snapshot baselines per machine and per architecture?
- Build a tolerance band around the measurements?

### 4. Packer blocking

**Decision needed:** Do we:
- Start with MATH_ISOLATE and UNPACK_ISOLATE only (safe, narrow scope)?
- Include L1_TO_L1 on Blackhole, skip Wormhole L1_TO_L1 for now?
- Gate all three (zero false positives on math/unpack, 53 on wormhole L1_TO_L1)?
- Exclude perf_matmul explicitly, gate the rest?

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
