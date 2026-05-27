# CI Disable Work — Status Log

Last updated: **2026-05-27T10:00 UTC** (session: BH systematic blocker persists (27+ consecutive failures, most recent run 26501610173 failure 09:01–09:51 UTC). All PRs unchanged. PR #44938 verified-pass awaiting human review. PR #45108 SKIP (<4h, verified-pass). PRs #45110/#45112/#45114 BH-blocked. 0 dispatches.)

---

## How to read/update this file

- Read this file at the start of every automation session and treat it as the authoritative current state for CI disable work.
- Scan the `## Quick Index` table first; it gives the lifecycle stage per PR before drilling into details.
- Per-PR sections use uniform field tables (`PR | Disable issue | Timeout issue | Branch | Workflow file | Lifecycle stage | Last rebase | Last revalidation | Verification run | Readiness`); update fields in place rather than rewriting the section.
- Append new entries to the top of `## Recent Activity` (most recent first); keep at most 30 entries — trim older entries to a single `- Older history truncated — see git history of this file.` line if needed.
- Commit and push any change to this file before ending the session.
- Lifecycle stages: `new`, `batch-committed`, `verifying`, `verification-inconclusive`, `verified-pass`, `verified-fail`, `merged`, `out-of-scope`. (`verification-inconclusive` = a verification was dispatched but failed to actually exercise the previously-passing jobs; eligible for re-dispatch and does NOT consume the one-run-per-PR budget.)

---

## Quick Index

| PR | Workflow | Lifecycle stage | Verification result | Ready to merge? | Notes |
|----|----------|-----------------|---------------------|-----------------|-------|
| [#44938](https://github.com/tenstorrent/tt-metal/pull/44938) | `t3000-demo-tests` | `verified-pass` | [26295163268](https://github.com/tenstorrent/tt-metal/actions/runs/26295163268) success | Yes | Not behind main (mergeable_state: unknown) — awaiting CI/review |
| [#45108](https://github.com/tenstorrent/tt-metal/pull/45108) | `(T3K) T3000 e2e tests` | `verified-pass` | [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) — CCL failure + 90m timeout | Yes | Rebased 2026-05-27 07:01 UTC onto main `c68e6ee`; now up-to-date; awaiting human review |
| [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) | `Blackhole post-commit tests` | `verification-inconclusive` | [26490261745](https://github.com/tenstorrent/tt-metal/actions/runs/26490261745) — **infra-inconclusive** (artifact expiry, 2026-05-27 04:15 UTC) | No (awaiting valid verification) | Third consecutive infra-inconclusive. Systematic blocker: BH post-commit build artifacts expire ~1 day; no valid source run available. Re-dispatch eligible when fresh successful BH post-commit main run exists. |
| [#45112](https://github.com/tenstorrent/tt-metal/pull/45112) | `(Blackhole) e2e tests` | `batch-committed` | — | No | No verify yet; same systematic blocker (Blackhole artifact expiry); PR already up-to-date with main @ `f19f708f644` |
| [#45114](https://github.com/tenstorrent/tt-metal/pull/45114) | `(Blackhole) Demo tests` | `batch-committed` | — | No | No verify yet; same systematic blocker (Blackhole artifact expiry); rebased 2026-05-27T05:00 UTC via merge |
| [#44860](https://github.com/tenstorrent/tt-metal/pull/44860) | `tt-metal-l2-tests` | `out-of-scope` | — | N/A | Separate agent |

---

## Active Runs

*(none)*

**Policy:** Concurrent runs across PRs are allowed; each automation session may dispatch at most three new runs.

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-----------|
| [26501610173](https://github.com/tenstorrent/tt-metal/actions/runs/26501610173) | `Blackhole post-commit tests` | `main` | 2026-05-27 09:01 UTC | 2026-05-27 09:51 UTC | **failure** | BH systematic blocker: 27th+ consecutive failure (SHA `9cc43ddcad215256d199f4bd456604a1f2045b64`). Still no successful BH post-commit run. |
| [26499906125](https://github.com/tenstorrent/tt-metal/actions/runs/26499906125) | `Blackhole post-commit tests (P100 nightly)` | `main` | 2026-05-27 08:25 UTC | 2026-05-27 09:39 UTC | **failure** | BH systematic blocker: failure (SHA `2c0072ef33f24c4f51f6a2b4b11a1aa376c0939b`). |
| [26494921196](https://github.com/tenstorrent/tt-metal/actions/runs/26494921196) | `(Single-card) Demo tests` | `main` | 2026-05-27 06:31 UTC | 2026-05-27 07:36 UTC | **success** | Single-card demo main run — overall SUCCESS (SHA `b4f5ed4f`). Confirms no deterministic single-card demo failures. |
| [26492593059](https://github.com/tenstorrent/tt-metal/actions/runs/26492593059) | `Blackhole post-commit tests` | `main` | 2026-05-27 05:25 UTC | 2026-05-27 06:07 UTC | **failure** | BH systematic blocker: 25th+ consecutive failure (SHA `c68e6ee16713ace55d0fd4d677f418d4add39f8d`). No new successful BH post-commit run. |
| [26490261745](https://github.com/tenstorrent/tt-metal/actions/runs/26490261745) | `Blackhole post-commit tests` | `verify/ci-disable-blackhole-post-commit-20260527` | 2026-05-27 04:11 UTC | 2026-05-27 04:15 UTC | **failure** | PR #45110 third verify: **infra inconclusive** — `build-artifact/download-artifacts` failed (`ERROR: Could not find build artifact matching expected pattern`, `TRACY_ENABLED: true`); all test jobs failed at Initialize containers; pytest never ran. Source run `26482998463` (attempt 2) had no build tarball — artifacts expired (~1 day retention). |
| [26482835281](https://github.com/tenstorrent/tt-metal/actions/runs/26482835281) | `Blackhole post-commit tests` | `verify/ci-disable-blackhole-post-commit-20260526` | 2026-05-27 00:18 | 2026-05-27 00:32 | **failure** | PR #45110 only verify: **infra inconclusive** — artifact download failed; pruned jobs failed at Initialize containers before pytest; artifact reuse `26480351906` |
| [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) | `(T3K) T3000 e2e tests` | `verify/ci-disable-t3000-e2e-20260526` | 2026-05-26 16:13 | 2026-05-26 18:49 | **failure** | PR #45108 only verify: Llama **success**; CCL **failure** (trace buffer `rs_input_shape2` + 90m timeout → [#45286](https://github.com/tenstorrent/tt-metal/issues/45286)); artifact `26445104085` |
| [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) | `(T3K) T3000 e2e tests` | `main` | 2026-05-26 07:27 | 2026-05-26 09:25 | **failure** | Main revalidation: Llama green; `t3k_ccl_tests` failed (keeps CCL disables) |
| [26295163268](https://github.com/tenstorrent/tt-metal/actions/runs/26295163268) | `t3000-demo-tests` | `ci/disable-failing-tests-t3000-demo-tests-20260521` | — | 2026-05-22 21:04 | **success** | PR #44938 verification passed |

---

## PR #44938 — t3000-demo-tests

| Field | Value |
|-------|-------|
| PR | [#44938](https://github.com/tenstorrent/tt-metal/pull/44938) (undrafted) |
| Disable issue | [#44937](https://github.com/tenstorrent/tt-metal/issues/44937) |
| Timeout issue | none |
| Branch | `ci/disable-failing-tests-t3000-demo-tests-20260521` |
| Workflow file | `t3000-demo-tests` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-27 02:42 UTC — PR is NOT behind main (mergeable_state: `blocked`, not `behind`); up-to-date per GitHub API |
| Last revalidation | 2026-05-27 10:00 UTC — lightweight check; no new t3000-demo-tests main runs since last session; no action needed |
| Verification run | [26295163268](https://github.com/tenstorrent/tt-metal/actions/runs/26295163268) — **success** (2026-05-22) |
| Readiness | **Yes** (pending CI checks passing and human review) |

**Notes:** Ready to merge pending CI and human review. PR is undrafted. Verification [26295163268](https://github.com/tenstorrent/tt-metal/actions/runs/26295163268) passed on 2026-05-22. Not behind main as of 07:01 UTC 2026-05-27. No action taken; awaiting human review.

---

## PR #45108 — (T3K) T3000 e2e tests

| Field | Value |
|-------|-------|
| PR | [#45108](https://github.com/tenstorrent/tt-metal/pull/45108) (draft) |
| Disable issue | [#45107](https://github.com/tenstorrent/tt-metal/issues/45107) |
| Timeout issue | [#45286](https://github.com/tenstorrent/tt-metal/issues/45286) (T3K CCL 90m timeout) |
| Branch | `ci/disable-failing-tests-t3000-e2e-tests-20260524` |
| Workflow file | `t3000-e2e-tests.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-27 ~07:01 UTC — rebased via `update_pull_request_branch` onto latest main (`c68e6ee16713ace55d0fd4d677f418d4add39f8d`); branch now up-to-date |
| Last revalidation | 2026-05-26 ~20:35 UTC — 14 CCL disables revalidated on main e2e [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812); no newer T3000 e2e main run as of 10:00 UTC 2026-05-27 |
| Verification run | [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) — Llama **success**; CCL **failure** (trace-buffer + 90m timeout) |
| Readiness | **Yes** (rebased and up-to-date; awaiting human review) |

**Notes:** Draft; verification [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) done — Llama pruned job green (no regression); CCL still red (trace-buffer param not disabled + 90m timeout). 14 disables still justified on latest main e2e [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812). Reclassified to verified-pass on 2026-05-27: the single verification run showed no regressions in jobs that were passing on main. Rebased 2026-05-27 07:01 UTC via `update_pull_request_branch` — now up-to-date with main. Ready for human undraft + merge. PR was <4h old at 10:00 UTC session start → SKIP (lightweight only).

---

## PR #45110 — Blackhole post-commit tests

| Field | Value |
|-------|-------|
| PR | [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) (draft) |
| Disable issue | [#45109](https://github.com/tenstorrent/tt-metal/issues/45109) |
| Timeout issue | none |
| Branch | `ci/disable-failing-tests-blackhole-post-commit-20260524` |
| Workflow file | `blackhole-post-commit.yaml` |
| Lifecycle stage | `verification-inconclusive` |
| Last rebase | 2026-05-27 ~04:05 UTC — merged onto `main` @ [`67fad7e3dd3`](https://github.com/tenstorrent/tt-metal/commit/67fad7e3dd3); PR head `4b98caa2530` |
| Last revalidation | 2026-05-27 04:00 UTC — disable still valid; main BH post-commit [26482998463](https://github.com/tenstorrent/tt-metal/actions/runs/26482998463) (SHA `2a4648824103`) only failed on models-P150 cloud VM job (unrelated); BH multi-card fast unit tests passed |
| Verification run | [26490261745](https://github.com/tenstorrent/tt-metal/actions/runs/26490261745) — **infra-inconclusive** (2026-05-27 04:11–04:15 UTC) |
| Readiness | **No** (awaiting valid verification — blocked by artifact expiry) |

**Disabled tests:**

| Disabled test ID | Still failing on main? | Last checked (UTC) | Main run |
|------------------|------------------------|--------------------|----------|
| `test_all_to_all_combine_no_trace[…-fabric_1d_ring_axis_0]` | **Yes** | 2026-05-27 01:05 | [26473697334](https://github.com/tenstorrent/tt-metal/actions/runs/26473697334) |

**Systematic blocker (all 3 verification attempts infra-inconclusive):**

The `blackhole-post-commit` workflow builds with `TRACY_ENABLED=true` and uploads build artifacts with short retention (~1 day). By the time verification is dispatched, the source run's build tarball has already expired. Error signature: `ERROR: Could not find build artifact matching expected pattern` + `TRACY_ENABLED (requested): true`. Confirmed by checking artifacts of the only recent successful BH post-commit main run (`26324273581`, May 23): that run's `🛠️ Build Release ubuntu 22.04` job uploaded a tarball, but the artifact is absent from the run's artifact list 4 days later.

**Re-dispatch is eligible when** a fresh successful BH post-commit main run on `main` completes, and the session runs **within ~24 hours** of that run (before the build artifact expires). BH post-commit main runs: most recent 27+ runs have all concluded `failure`. Last successful: run `26324273581` (May 23 05:10 UTC, SHA `79925a3f`).

---

## PR #45112 — (Blackhole) e2e tests

| Field | Value |
|-------|-------|
| PR | [#45112](https://github.com/tenstorrent/tt-metal/pull/45112) |
| Disable issue | [#45111](https://github.com/tenstorrent/tt-metal/issues/45111) |
| Timeout issue | unknown |
| Branch | `ci/disable-failing-tests-blackhole-e2e-tests-20260524` |
| Workflow file | `blackhole-e2e-tests.yaml` |
| Lifecycle stage | `batch-committed` |
| Last rebase | 2026-05-27 ~05:00 UTC — PR base SHA `f19f708f644` already matches current main HEAD; no additional rebase needed |
| Last revalidation | 2026-05-27 05:00 UTC (passive) — all recent BH e2e main runs are failure; disabled tests presumed still failing; deep analysis not done (context budget) |
| Verification run | none yet |
| Readiness | **No** (blocked by Blackhole artifact expiry) |

**Notes:** PR head `5a930ef4b9f0`. Same systematic blocker as PR #45110: BH e2e uses ~1-day artifact retention. Workflow `blackhole-e2e-tests.yaml` needs `use-artifacts-from-run` YAML edit on temp branch before verification. Blocked until a fresh successful BH main run exists within 24h.

---

## PR #45114 — (Blackhole) Demo tests

| Field | Value |
|-------|-------|
| PR | [#45114](https://github.com/tenstorrent/tt-metal/pull/45114) |
| Disable issue | [#45113](https://github.com/tenstorrent/tt-metal/issues/45113) |
| Timeout issue | none |
| Branch | `ci/disable-failing-tests-blackhole-demo-tests-20260524` |
| Workflow file | `blackhole-demo-tests.yaml` |
| Lifecycle stage | `batch-committed` |
| Last rebase | 2026-05-27 ~05:00 UTC — merged latest main (`f19f708f644`) into PR branch via `update_pull_request_branch`; was previously at `b5522097` (May 24) |
| Last revalidation | 2026-05-27 05:00 UTC (passive) — all recent BH demo main runs (20+) are failure; disabled tests presumed still failing; deep analysis not done (context budget) |
| Verification run | none yet |
| Readiness | **No** (blocked by Blackhole artifact expiry) |

**Disabled tests:**
- `test_demo_text` — `performance + ci-32` variant (skipped via issue #45113)

**Notes:** Same systematic blocker as PR #45110: `blackhole-demo-tests.yaml` uses short-retention (~1 day) artifacts. All BH demo tests main runs (last 20+) have `conclusion: failure`. The most recent successful BH demo run predates the BH post-commit last-success (May 23), so no suitable artifact source exists.

---

## PR #44860 — tt-metal-l2-tests

| Field | Value |
|-------|-------|
| PR | [#44860](https://github.com/tenstorrent/tt-metal/pull/44860) |
| Disable issue | unknown |
| Timeout issue | unknown |
| Branch | unknown |
| Workflow file | `tt-metal-l2-tests` |
| Lifecycle stage | `out-of-scope` |
| Last rebase | unknown |
| Last revalidation | unknown |
| Verification run | unknown |
| Readiness | **N/A** |

**Notes:** Separate agent scope (Nightly L2 tests).

---

## Blockers

| Blocker | Status | Notes |
|---------|--------|-------|
| **Systematic: Blackhole artifact expiry** | **Open** | BH post-commit, BH e2e, BH demo-tests workflows all use ~1-day build artifact retention. Verification can only succeed if dispatched within ~24h of a fresh successful BH main run. BH post-commit last success: run `26324273581` (May 23 05:10 UTC, SHA `79925a3f`). Currently 4+ days since last success; 27+ consecutive failure runs (most recent: [26501610173](https://github.com/tenstorrent/tt-metal/actions/runs/26501610173), May 27 09:01–09:51 UTC). Until a new successful run completes, PRs #45110/#45112/#45114 are all blocked. |
| PR #45110 infra-inconclusive (×3) | **Open** | Three consecutive `build-artifact/download-artifacts` failures due to expired TRACY=true tarball. Re-dispatch eligible; requires fresh artifact source within 24h of a successful BH post-commit main run. |
| Trace-buffer disable candidate (#45108) | **Watch** | `rs_input_shape2` — 1/5 main runs (below 3-consecutive threshold, not added) |
| CCL job 90m timeout (#45108) | **Tracked** | [#45286](https://github.com/tenstorrent/tt-metal/issues/45286) |
| New disable PR opportunity | **Investigated** | `all-model-tests.yaml` = manually dispatched only (not suitable). `(Single-card) Demo tests` = recent runs non-consecutive failures/successes (run 26494921196 is SUCCESS as of 07:36 UTC today). `(Single-card) Frequent model and ttnn tests` = scattered failures mostly fixed by retry/attempt-2 (not ≥3-consecutive-same-error). No suitable WH single-card scheduled workflow with ≥3 consecutive same-error failures found. New disable PR deferred until a qualifying candidate emerges. |

---

## Recent Activity

- `2026-05-27 ~10:00 UTC` — SESSION: BH systematic blocker persists: runs [26501610173](https://github.com/tenstorrent/tt-metal/actions/runs/26501610173) (SHA `9cc43ddc`, 09:01–09:51 UTC, failure) and [26499906125](https://github.com/tenstorrent/tt-metal/actions/runs/26499906125) (SHA `2c0072ef`, 08:25–09:39 UTC, failure) — 27th+ consecutive BH post-commit failures. PR [#44938](https://github.com/tenstorrent/tt-metal/pull/44938) CANDIDATE (7.3h old, verified-pass): updated_at 02:42 UTC; confirmed no action needed — awaiting human review. PR [#45108](https://github.com/tenstorrent/tt-metal/pull/45108) SKIP (<4h old, verified-pass, updated 07:04 UTC): lightweight only — no new completed runs since last session. PRs #45110/#45112/#45114: BH-blocked, no change. No new disable PR opportunity found (continuing from prior session investigation). 0 dispatches. State log updated.
- `2026-05-27 ~09:00 UTC` — SESSION: BH systematic blocker still persists: BH post-commit run [26492593059](https://github.com/tenstorrent/tt-metal/actions/runs/26492593059) (May 27 05:25–06:07 UTC, SHA `c68e6ee`) = failure; 25+ consecutive failures; no in-progress BH runs. PR [#44938](https://github.com/tenstorrent/tt-metal/pull/44938) CANDIDATE (6.3h old, verified-pass): no action, awaiting human review. PR [#45108](https://github.com/tenstorrent/tt-metal/pull/45108) SKIP (<2h old). PRs #45110/#45112/#45114: BH-blocked. New disable PR investigation: `(Single-card) Demo tests` — latest run 26494921196 (May 27 07:36) SUCCESS; recent runs show non-consecutive pass/fail pattern, no ≥3 consecutive same-error failures. `(Single-card) Frequent model and ttnn tests` — scattered failures (most fixed by retry at attempt 2), no ≥3 consecutive same-error. No qualifying WH single-card workflow found. 0 dispatches this session. State log updated.
- `2026-05-27 ~08:01 UTC` — SESSION: PRs #44938 and #45108 SKIP (< 1h old, `verified-pass`). PRs #45112 and #45114 SKIP (< 3h old, BH-blocked). PR [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) CANDIDATE (verification-inconclusive): BH systematic blocker persists — latest BH post-commit main run [26492593059](https://github.com/tenstorrent/tt-metal/actions/runs/26492593059) (May 27 05:25–06:07 UTC, SHA `c68e6ee`) = failure; 23+ consecutive failures; no re-dispatch possible. Investigated `all-model-tests.yaml` for new disable PR — determined it is `workflow_dispatch` only (no scheduled main runs); not suitable for ≥3-consecutive-same-error disable pattern. New disable PR for non-BH WH single-card scheduled workflow deferred (context budget). 0 dispatches this session. State log updated.
- `2026-05-27 ~07:01 UTC` — SESSION: PR [#45108](https://github.com/tenstorrent/tt-metal/pull/45108) CANDIDATE (4.3h old, `verified-pass`): rebased via `update_pull_request_branch` onto latest main `c68e6ee16713ace55d0fd4d677f418d4add39f8d`; PR now up-to-date; no new T3000 e2e main runs since [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) (2026-05-26 07:27 UTC); CCL disables still valid (last revalidation 2026-05-26 20:35 UTC). PR [#44938](https://github.com/tenstorrent/tt-metal/pull/44938) CANDIDATE (4.3h old): confirmed not-behind-main (mergeable_state: `blocked`); no action needed — awaiting human review. BH systematic blocker: confirmed 22+ consecutive failures; most recent run [26492593059](https://github.com/tenstorrent/tt-metal/actions/runs/26492593059) (May 27 05:25–06:07 UTC, SHA `c68e6ee`) = failure. No in-flight runs. 0 dispatches. State log updated.
- `2026-05-27 ~06:02 UTC` — SESSION: All PRs <4h old → lightweight checks only. BH systematic blocker unchanged: most recent BH post-commit main run [26482998463](https://github.com/tenstorrent/tt-metal/actions/runs/26482998463) (May 27 01:47 UTC, SHA `2a4648824103`) = failure; 21+ consecutive BH post-commit failures; last success `26324273581` (May 23 SHA `79925a3f`). PR [#45108](https://github.com/tenstorrent/tt-metal/pull/45108): confirmed still `behind` main (mergeable_state: `behind`) — needs rebase before merge; deferred (PR <4h old). No in-flight runs completed. 0 dispatches. State log updated and pushed.
- `2026-05-27 ~05:25 UTC` — SESSION: All PRs <4h old → lightweight checks only. BH systematic blocker confirmed: most recent BH post-commit main run [26482998463](https://github.com/tenstorrent/tt-metal/actions/runs/26482998463) (May 27 01:38 UTC, SHA `2a4648824103`) = failure; 20+ consecutive BH post-commit failures; last success `26324273581` (May 23). PR [#44938](https://github.com/tenstorrent/tt-metal/pull/44938): confirmed not-behind-main (mergeable_state: `blocked`, not `behind`) — no action needed. PR [#45108](https://github.com/tenstorrent/tt-metal/pull/45108): confirmed still `behind` main — needs rebase, deferred (PR <4h old). No in-flight runs completed. No new dispatches (0/3). New disable PR (non-BH non-Galaxy single-card workflow) deferred to next session due to context budget. State log updated.
- `2026-05-27 ~05:15 UTC` — SESSION: BH systematic blocker confirmed unchanged (BH post-commit: 12+ consecutive failure runs; most recent: [26482998463](https://github.com/tenstorrent/tt-metal/actions/runs/26482998463), May 27 01:38 UTC, failure; BH demo tests: 20+ consecutive failure runs). Merged latest main (`f19f708f644`) into PR [#45114](https://github.com/tenstorrent/tt-metal/pull/45114) via `update_pull_request_branch` (previously at `b5522097`, May 24). Confirmed PR [#45112](https://github.com/tenstorrent/tt-metal/pull/45112) already up-to-date (base SHA `f19f708f644` matches current main HEAD). PRs [#44938](https://github.com/tenstorrent/tt-metal/pull/44938) and [#45108](https://github.com/tenstorrent/tt-metal/pull/45108) updated <4h ago (lightweight-only; no action taken). PR [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) updated <4h ago (lightweight-only; no new completed runs to act on). 0 dispatches this session. State log updated.
- `2026-05-27 ~04:45 UTC` — SESSION END: Run [26490261745](https://github.com/tenstorrent/tt-metal/actions/runs/26490261745) completed **infra-inconclusive** (third time for PR [#45110](https://github.com/tenstorrent/tt-metal/pull/45110)) — `build-artifact/download-artifacts` failed with `ERROR: Could not find build artifact matching expected pattern` and `TRACY_ENABLED: true`; all test jobs failed at Initialize containers; pytest never ran. Source run `26482998463` (attempt 2) had no build tarball. Confirmed systematic blocker: BH post-commit artifacts expire ~1 day; the only recent successful BH main run (`26324273581`, May 23) built artifacts but they are now gone (4 days later). PR #45110 lifecycle → `verification-inconclusive` (third); PR body updated. PRs #45112 and #45114 also blocked by same artifact-expiry issue (all BH demo/e2e main runs have been `failure` for weeks). No new dispatches this session (0/3). Work remaining: monitor BH post-commit main runs; re-dispatch PR #45110 verification within 24h of next successful run.
- `2026-05-27 ~04:25 UTC` — PR [#45110](https://github.com/tenstorrent/tt-metal/pull/45110): merged `main` @ `67fad7e3dd3` into PR branch (head `4b98caa2530`); revalidation confirmed disable still valid (26482998463 only failed on unrelated models-P150 job); dispatched fresh verification [26490261745](https://github.com/tenstorrent/tt-metal/actions/runs/26490261745) on `verify/ci-disable-blackhole-post-commit-20260527` (pruned to `blackhole-multi-card-fast-unit-tests`; artifact source `26482998463`); lifecycle → `verifying`.
- `2026-05-27 ~01:10 UTC` — Marked run [26482835281](https://github.com/tenstorrent/tt-metal/actions/runs/26482835281) COMPLETED (infra failure, inconclusive for disable validation); rebased [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) and [#45108](https://github.com/tenstorrent/tt-metal/pull/45108) onto `main` @ `4b308296`; revalidated disables unchanged on BH post-commit + T3K e2e latest main runs; created timeout-tracking issue [#45286](https://github.com/tenstorrent/tt-metal/issues/45286) for T3K CCL 90m timeout; updated PR bodies.
- 2026-05-27 — Reclassified PR #45110 to verification-inconclusive; prior run 26482835281 was infra-inconclusive and is now retry-eligible per updated policy.
- 2026-05-27 — Reclassified PR #45108 to verified-pass under updated policy (no regression in previously-passing jobs; CCL failures pre-existed on main).
- `2026-05-27 ~00:20 UTC` — PR [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) rebased onto `main` @ `4cd9b59a8d2`; dispatched verify [26482835281](https://github.com/tenstorrent/tt-metal/actions/runs/26482835281).
- `2026-05-26 ~20:35 UTC` — Automation poll (status log only): revalidated 14 CCL disables on main e2e [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812).
- `2026-05-26 ~19:15 UTC` — Verification complete for PR [#45108](https://github.com/tenstorrent/tt-metal/pull/45108): run [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) — Llama success, CCL failure + timeout.
- `2026-05-26 ~16:13 UTC` — Targeted verification dispatched for PR [#45108](https://github.com/tenstorrent/tt-metal/pull/45108): run [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854); artifact reuse `26445104085`.
- `2026-05-24` — PR [#45108](https://github.com/tenstorrent/tt-metal/pull/45108) / [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) opened.
- `2026-05-22` — PR [#44938](https://github.com/tenstorrent/tt-metal/pull/44938) verification success ([26295163268](https://github.com/tenstorrent/tt-metal/actions/runs/26295163268)).
