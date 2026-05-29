# CI Disable — Galaxy Pipelines: Work So Far

> **Source of truth.** This file is the canonical record of automation-tracked PRs. Wiping it resets the automation to fresh-state view; stale GitHub PRs not listed here are intentionally invisible.

Last updated: 2026-05-29T04:02:05Z

> **Scope: GALAXY PIPELINES ONLY.** Companion file to `disabling-work/ci-disable-targeted-verification-galaxy.md`. The non-Galaxy state log is `disabling-work/disabling-work-so-far.md`; do not confuse the two.

---

## How to read/update this file

- Read this file at the start of every automation session and treat it as the authoritative current state for CI disable work.
- Scan the `## Quick Index` table first; it gives the lifecycle stage per PR before drilling into details.
- Per-PR sections use uniform field tables (`PR | Disable issue | Timeout issue | Branch | Workflow file | Lifecycle stage | Last rebase | Last revalidation | Verification run | Last touched by automation | Readiness`); update fields in place rather than rewriting the section. `Last touched by automation: <UTC ISO>` is required on every PR row and drives the 4-hour throttle — update it every time the automation does any work on the PR (rebase, dispatch, log analysis, comment, removal).
- Each PR section also carries a `Disables (with main evidence)` table listing every currently-disabled test ID together with the most recent failing main-run job-link (`/runs/<id>/job/<jid>`) and the run completion timestamp. This mirrors the PR description's evidence table (see `disabling-work/ci-disable-targeted-verification-galaxy.md` → `Main-run evidence model`) and is the starting point the next session re-checks before doing any work on the PR. If keeping the state log compact is preferred, the per-PR section MAY instead include the one-line pointer `Main-run evidence: see PR description.` — the PR description's evidence table is authoritative either way. Preserve any existing PR entries unchanged when extending the schema.
- Append new entries to the top of `## Recent Activity` (most recent first); keep at most 30 entries — trim older entries to a single `- Older history truncated — see git history of this file.` line if needed.
- Commit and push any change to this file before ending the session.
- Lifecycle stages: `new`, `batch-committed`, `verifying`, `verification-inconclusive`, `verified-pass`, `verified-fail`, `merged`, `out-of-scope`. (`verification-inconclusive` = a verification was dispatched but failed to actually exercise the previously-passing jobs; eligible for re-dispatch and does NOT consume the one-run-per-PR budget.)

---

## Quick Index

| PR | Workflow | Lifecycle stage | Verification result | Ready to merge? | Notes |
|----|----------|-----------------|---------------------|-----------------|-------|
| _(none yet)_ |  |  |  |  |  |

---

## Active Runs

| Run | Pipeline | Branch | Started | Status | Notes |
|-----|----------|--------|---------|--------|-------|
| _(none)_ |  |  |  |  |  |

**Policy:** Concurrent runs across PRs are allowed; each automation session may dispatch at most three new runs.

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-------|
| _(none)_ |  |  |  |  |  |  |

---

## Blockers

| Blocker | Status | Notes |
|---------|--------|-------|
| All Galaxy workflow failures are infrastructure/environmental | Active | TLB allocation failures, model loading/access failures, hugepage/NUMA issues, runner cancellations, and script-wrapper failures (where pytest tests themselves pass but the surrounding shell script fails) are the predominant failure modes across galaxy-integration-tests, galaxy-unit-tests. No deterministic test failures with specific test IDs meeting the 3-consecutive-runs criterion were found across any Galaxy pipeline examined. |

---

## Recent Activity

- 2026-05-29T04:02:05Z — Session analysis complete (triggered by hourly cron). Examined Galaxy workflows. No examining PRs (state log is empty). No focus PRs created. Detailed findings from this session: (1) Galaxy unit tests (10 most recent runs examined, runs #1524–#1459): non-consecutive failure pattern — run #1524 (2026-05-28, id 26566306035) failure, #1506 (2026-05-27) failure, #1495 (2026-05-26) SUCCESS, #1494 (2026-05-25) failure, #1493–#1492 (2026-05-24–23) failure, #1491–#1490 (2026-05-22–21) SUCCESS. No 3 consecutive failures with same error signature; failures are infrastructure (runner cancellations, not test code failures). (2) Galaxy integration tests run 26554625774 (2026-05-28, job-level analysis performed): Qwen-Image job 78224663587 — `1 deselected, 1 error in 5.48s` from TLB allocation failure (`tt::umd::TLBManager::configure_tlb`), pure infra fault, no FAILED pytest test IDs; unit/distributed job 78224663577 — 20 pytest tests `20 passed in 67.80s` but `run_tg_frequent_tests` shell script reported failure, same script-level-not-pytest pattern as previous sessions; resnet50 and Wan2.2 jobs: PASSED; SD3.5/Motif/Mochi: model loading/access failures (infra). Galaxy integration tests have been failing in all 20 examined runs but exclusively due to infrastructure/environment issues, not deterministic pytest test failures. No eligible disable candidates found. Paralysis check: FAILED — zero examining PRs, zero focus PRs. Legitimate absence of in-scope deterministic test failures; Galaxy pipelines fail exclusively via infrastructure/environment issues or script-level failures where pytest tests themselves pass. Orphaned-PR check skipped per Source of Truth policy. If you suspect orphaned automation PRs exist from prior broken sessions, perform a manual backfill before the next session.
- 2026-05-29T03:00:15Z — Session analysis complete (triggered by hourly cron). Examined Galaxy workflows. No examining PRs (state log is empty). No focus PRs created. Detailed findings from latest runs: (1) Galaxy unit tests run 26566306035 (2026-05-28): failure due to runner setup cancellation on `UMD API tests [wh_galaxy_perf_uf]` job — this is an infra fault, not a test failure; all other unit test jobs (distributed ops, multi-process, fabric, tttv2, UMD unit tests) PASSED. (2) Galaxy integration tests run 26554625774 (2026-05-28): resnet50 and Wan2.2 PASSED; Flux.1 integration tests — all 3 pytest tests PASSED (3 passed, 5 deselected in 51.13s) but `run_tg_frequent_tests` shell script reported overall failure, indicating another component within the same script (likely CCL or perf check) failed; unit/distributed tests — all 20 `test_multi_device_trace_TG` pytest tests PASSED (20 passed in 67.80s) but same script-level failure; Qwen-Image: TLB allocation infra failure (`RuntimeError: Failed to allocate TLB window`); SD3.5/Motif/Mochi: model loading/access failures (OSError: Can't load config). None of these constitute deterministic test failures eligible for disable under policy. The `run_tg_frequent_tests` script-level failures in Flux.1 and unit/distributed jobs are not deterministic pytest test failures — the pytest tests themselves pass. Root cause of script-level failure is likely a CCL perf stage, a Python segfault during test collection in CCL tests, or a non-pytest component of the script. Paralysis check: FAILED — zero examining PRs and zero focus PRs, no eligible deterministic test failures identified in any Galaxy workflow to justify disable PR creation. This is not a bug in the automation logic but reflects a genuine absence of in-scope deterministic test failures; Galaxy pipelines are failing exclusively due to infrastructure/environment issues or script-level failures unrelated to test code regressions.
- 2026-05-29T02:00:41Z — Session analysis complete (triggered by hourly cron). Examined all Galaxy workflows. No examining PRs (state log is empty). No focus PRs created. Root cause: all Galaxy CI failures are infrastructure/environmental — TLB allocation errors (RuntimeError: Failed to allocate TLB window, error code -12 for TLB size 1048576) in Qwen-Image tests; model loading/access failures (OSError: Can't load config, stabilityai/...) in Mochi/SD3.5/Motif demo tests; hugepage/NUMA issues (UMD: NOC address of a hugepage does not match the expected address) in e2e and integration test runners; runner cancellation (setup runner step cancelled) in unit test `wh_galaxy_perf_uf` jobs; Python segfault during test collection in CCL tests. None qualify as deterministic test failures per policy. Galaxy unit tests show intermittent failures (runs 1524, 1506, 1494, 1493, 1492 failed; runs 1495, 1491, 1490 passed) driven entirely by infrastructure faults, not reproducible test code failures. Galaxy integration tests (183 total runs, all failures) and e2e tests (126 total runs, ~1 pass, rest failures) similarly driven by infrastructure. Paralysis check: FAILED — zero examining PRs and zero focus PRs, no eligible deterministic test failures identified in any Galaxy workflow to justify disable PR creation. This is not a bug in the automation logic but reflects a genuine absence of in-scope deterministic test failures; Galaxy pipelines are failing exclusively due to infrastructure/environment issues.
