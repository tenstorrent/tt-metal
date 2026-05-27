# CI Disable Work — Status Log

Last updated: **2026-05-27T01:10 UTC** (PR #45110 verify complete; rebases; timeout issue #45286)

---

## How to read/update this file

- Read this file at the start of every automation session and treat it as the authoritative current state for CI disable work.
- Scan the `## Quick Index` table first; it gives the lifecycle stage per PR before drilling into details.
- Per-PR sections use uniform field tables (`PR | Disable issue | Timeout issue | Branch | Workflow file | Lifecycle stage | Last rebase | Last revalidation | Verification run | Readiness`); update fields in place rather than rewriting the section.
- Append new entries to the top of `## Recent Activity` (most recent first); keep at most 30 entries — trim older entries to a single `- Older history truncated — see git history of this file.` line if needed.
- Commit and push any change to this file before ending the session.
- Lifecycle stages: `new`, `batch-committed`, `verifying`, `verified-pass`, `verified-fail`, `merged`, `out-of-scope`.

---

## Quick Index

| PR | Workflow | Lifecycle stage | Verification result | Ready to merge? | Notes |
|----|----------|-----------------|---------------------|-----------------|-------|
| [#44938](https://github.com/tenstorrent/tt-metal/pull/44938) | `t3000-demo-tests` | `verified-pass` | [26295163268](https://github.com/tenstorrent/tt-metal/actions/runs/26295163268) success | Yes | Pending fresh rebase before merge |
| [#45108](https://github.com/tenstorrent/tt-metal/pull/45108) | `(T3K) T3000 e2e tests` | `verified-fail` | [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) — CCL failure + 90m timeout | No | Timeout tracked in [#45286](https://github.com/tenstorrent/tt-metal/issues/45286) |
| [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) | `Blackhole post-commit tests` | `verified-fail` | [26482835281](https://github.com/tenstorrent/tt-metal/actions/runs/26482835281) — infra inconclusive | No | No second verify; main-failure evidence only |
| [#45112](https://github.com/tenstorrent/tt-metal/pull/45112) | `(Blackhole) e2e tests` | `batch-committed` | — | No | No verify yet; needs rebase |
| [#45114](https://github.com/tenstorrent/tt-metal/pull/45114) | `(Blackhole) Demo tests` | `batch-committed` | — | No | No verify yet; needs rebase |
| [#44860](https://github.com/tenstorrent/tt-metal/pull/44860) | `tt-metal-l2-tests` | `out-of-scope` | — | N/A | Separate agent |

---

## Active Runs

_None — global run lock free._

**Policy:** Concurrent runs across PRs are allowed; each automation session may dispatch at most one new run.

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-------|
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
| Last rebase | 2026-05-24 — **needs fresh rebase** before merge vs `main` @ `4b308296` |
| Last revalidation | unknown — needs investigation next session |
| Verification run | [26295163268](https://github.com/tenstorrent/tt-metal/actions/runs/26295163268) — **success** (2026-05-22) |
| Readiness | **Yes** (pending fresh rebase) |

**Notes:** Ready to merge pending rebase/review. Verification [26295163268](https://github.com/tenstorrent/tt-metal/actions/runs/26295163268) passed on 2026-05-22. PR has been undrafted.

---

## PR #45108 — (T3K) T3000 e2e tests

| Field | Value |
|-------|-------|
| PR | [#45108](https://github.com/tenstorrent/tt-metal/pull/45108) (draft) |
| Disable issue | [#45107](https://github.com/tenstorrent/tt-metal/issues/45107) |
| Timeout issue | [#45286](https://github.com/tenstorrent/tt-metal/issues/45286) (T3K CCL 90m timeout) |
| Branch | `ci/disable-failing-tests-t3000-e2e-tests-20260524` |
| Workflow file | `t3000-e2e-tests.yaml` |
| Lifecycle stage | `verified-fail` |
| Last rebase | 2026-05-27 ~01:05 UTC — rebased onto [`4b308296`](https://github.com/tenstorrent/tt-metal/commit/4b308296cb6a65b3ba8c27f1f10b0efef1443876); PR head [`d4b91a7`](https://github.com/tenstorrent/tt-metal/commit/d4b91a7ac90322b1bb76a4c64096376ee05fa14a) |
| Last revalidation | 2026-05-26 ~20:35 UTC — 14 CCL disables revalidated on main e2e [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) |
| Verification run | [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) — Llama **success**; CCL **failure** (trace-buffer + 90m timeout) |
| Readiness | **No** |

**Notes:** Draft; verification [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) done — Llama pruned job green (no regression); CCL still red (trace-buffer param not disabled + 90m timeout). 14 disables still justified on latest main e2e [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812). No new disables — `rs_input_shape2` still 1/5 main runs. Pending disable candidate (NOT added): `test_reduce_scatter_async_sharded_to_interleaved[…-rs_input_shape2-…]` — 1/5 main runs. Next: Watch main for 3× trace-buffer signature; no second verify. Consider merge-readiness only if team accepts CCL residual failure as out-of-scope.

---

## PR #45110 — Blackhole post-commit tests

| Field | Value |
|-------|-------|
| PR | [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) (draft) |
| Disable issue | [#45109](https://github.com/tenstorrent/tt-metal/issues/45109) |
| Timeout issue | none |
| Branch | `ci/disable-failing-tests-blackhole-post-commit-20260524` |
| Workflow file | `blackhole-post-commit.yaml` |
| Lifecycle stage | `verified-fail` |
| Last rebase | 2026-05-27 ~01:05 UTC — rebased onto [`4b308296`](https://github.com/tenstorrent/tt-metal/commit/4b308296cb6a65b3ba8c27f1f10b0efef1443876); PR head [`3ae25c8`](https://github.com/tenstorrent/tt-metal/commit/3ae25c8ebf9e8c94c19b95dc7dd7b564bba9aba8) |
| Last revalidation | 2026-05-27 01:05 UTC — disable still valid; latest main BH post-commit [26473697334](https://github.com/tenstorrent/tt-metal/actions/runs/26473697334) still shows same assertion |
| Verification run | [26482835281](https://github.com/tenstorrent/tt-metal/actions/runs/26482835281) — **failure (infra inconclusive)** |
| Readiness | **No** |

**Disabled tests:**

| Disabled test ID | Still failing on main? | Last checked (UTC) | Main run |
|------------------|------------------------|--------------------|----------|
| `test_all_to_all_combine_no_trace[…-fabric_1d_ring_axis_0]` | **Yes** | 2026-05-27 01:05 | [26473697334](https://github.com/tenstorrent/tt-metal/actions/runs/26473697334) |

**Notes:** Draft; single verification [26482835281](https://github.com/tenstorrent/tt-metal/actions/runs/26482835281) completed failure (infra) — cannot mark ready; no second verify. Main revalidation confirms disable still valid. Next: Human review / accept disable on main-failure evidence only; no re-verify.

---

## PR #45112 — (Blackhole) e2e tests

| Field | Value |
|-------|-------|
| PR | [#45112](https://github.com/tenstorrent/tt-metal/pull/45112) |
| Disable issue | unknown — needs investigation next session |
| Timeout issue | unknown — needs investigation next session |
| Branch | unknown — needs investigation next session |
| Workflow file | `(Blackhole) e2e tests` |
| Lifecycle stage | `batch-committed` |
| Last rebase | unknown — needs investigation next session |
| Last revalidation | unknown — needs investigation next session |
| Verification run | none yet |
| Readiness | **No** |

**Notes:** Needs rebase; no verify yet; `use-artifacts-from-run` not on `main` workflow.

---

## PR #45114 — (Blackhole) Demo tests

| Field | Value |
|-------|-------|
| PR | [#45114](https://github.com/tenstorrent/tt-metal/pull/45114) |
| Disable issue | unknown — needs investigation next session |
| Timeout issue | unknown — needs investigation next session |
| Branch | unknown — needs investigation next session |
| Workflow file | `(Blackhole) Demo tests` |
| Lifecycle stage | `batch-committed` |
| Last rebase | unknown — needs investigation next session |
| Last revalidation | unknown — needs investigation next session |
| Verification run | none yet |
| Readiness | **No** |

**Notes:** Needs rebase; no verify yet; `use-artifacts-from-run` not on `main` workflow.

---

## PR #44860 — tt-metal-l2-tests

| Field | Value |
|-------|-------|
| PR | [#44860](https://github.com/tenstorrent/tt-metal/pull/44860) |
| Disable issue | unknown — needs investigation next session |
| Timeout issue | unknown — needs investigation next session |
| Branch | unknown — needs investigation next session |
| Workflow file | `tt-metal-l2-tests` |
| Lifecycle stage | `out-of-scope` |
| Last rebase | unknown — needs investigation next session |
| Last revalidation | unknown — needs investigation next session |
| Verification run | unknown — needs investigation next session |
| Readiness | **N/A** |

**Notes:** Separate agent scope (Nightly L2 tests).

---

## Blockers

| Blocker | Status | Notes |
|---------|--------|-------|
| Active verification run blocks new dispatch | **Resolved** | No active runs |
| BH post-commit verify infra failure (#45110) | **Open** | Run 26482835281 — artifact download + container init; no re-verify |
| Trace-buffer disable candidate (#45108) | **Watch** | `rs_input_shape2` — 1/5 main runs |
| CCL job 90m timeout (#45108) | **Tracked** | [#45286](https://github.com/tenstorrent/tt-metal/issues/45286) |
| PRs behind main | **Resolved** | #45108, #45110 rebased onto `4b308296` 2026-05-27 |

---

## Recent Activity

- `2026-05-27 ~01:10 UTC` — Marked run [26482835281](https://github.com/tenstorrent/tt-metal/actions/runs/26482835281) COMPLETED (infra failure, inconclusive for disable validation); rebased [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) and [#45108](https://github.com/tenstorrent/tt-metal/pull/45108) onto `main` @ `4b308296` (heads `3ae25c8`, `d4b91a7`); revalidated disables unchanged on BH post-commit + T3K e2e latest main runs; created timeout-tracking issue [#45286](https://github.com/tenstorrent/tt-metal/issues/45286) for T3K CCL 90m timeout; updated PR bodies; no verification dispatch (budgets exhausted on #45108/#45110).
- `2026-05-27 ~00:20 UTC` — PR [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) rebased onto `main` @ `4cd9b59a8d2`; dispatched verify [26482835281](https://github.com/tenstorrent/tt-metal/actions/runs/26482835281).
- `2026-05-26 ~20:35 UTC` — Automation poll (status log only): revalidated 14 CCL disables on main e2e [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812).
- `2026-05-26 ~19:15 UTC` — Verification complete for PR [#45108](https://github.com/tenstorrent/tt-metal/pull/45108): run [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) — Llama success, CCL failure + timeout.
- `2026-05-26 ~16:13 UTC` — Targeted verification dispatched for PR [#45108](https://github.com/tenstorrent/tt-metal/pull/45108): run [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854); artifact reuse `26445104085`.
- `2026-05-24` — PR [#45108](https://github.com/tenstorrent/tt-metal/pull/45108) / [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) opened.
- `2026-05-22` — PR [#44938](https://github.com/tenstorrent/tt-metal/pull/44938) verification success ([26295163268](https://github.com/tenstorrent/tt-metal/actions/runs/26295163268)).
