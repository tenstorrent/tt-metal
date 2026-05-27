# CI Disable Work — Running Status Log

Last updated: **2026-05-27T01:10 UTC** (PR #45110 verify complete; rebases; timeout issue #45286)

Operational policy: **one workflow run at a time**, draft PRs only, artifact reuse for verification, Galaxy out of scope.

**Legend — main-failure checks:** Each disabled test row records the last time it was confirmed still failing (or passing / removed) on `main`, with a link to the main pipeline run used for that check.

**Legend — rebase tracking:** Each PR records the last rebase onto `main` (UTC timestamp + base commit SHA rebased onto).

---

## Active Runs

_None — global run lock free._

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-------|
| [26482835281](https://github.com/tenstorrent/tt-metal/actions/runs/26482835281) | `Blackhole post-commit tests` | `verify/ci-disable-blackhole-post-commit-20260526` | 2026-05-27 00:18 | 2026-05-27 00:32 | **failure** | PR #45110 only verify: **infra inconclusive** — artifact download failed; pruned jobs failed at Initialize containers before pytest; artifact reuse `26480351906` |
| [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) | `(T3K) T3000 e2e tests` | `verify/ci-disable-t3000-e2e-20260526` | 2026-05-26 16:13 | 2026-05-26 18:49 | **failure** | PR #45108 only verify: Llama **success**; CCL **failure** (trace buffer `rs_input_shape2` + 90m timeout → [#45286](https://github.com/tenstorrent/tt-metal/issues/45286)); artifact `26445104085` |
| [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) | `(T3K) T3000 e2e tests` | `main` | 2026-05-26 07:27 | 2026-05-26 09:25 | **failure** | Main revalidation: Llama green; `t3k_ccl_tests` failed (keeps CCL disables) |
| [26295163268](https://github.com/tenstorrent/tt-metal/actions/runs/26295163268) | `t3000-demo-tests` | `ci/disable-failing-tests-t3000-demo-tests-20260521` | — | 2026-05-22 21:04 | **success** | PR #44938 verification passed |

---

## In-Progress Pipelines / PRs

### `Blackhole post-commit tests` — verify complete (inconclusive infra); stay draft

| Item | Link |
|------|------|
| Draft PR | [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) |
| Tracking issue | [#45109](https://github.com/tenstorrent/tt-metal/issues/45109) |
| Branch | `ci/disable-failing-tests-blackhole-post-commit-20260524` |
| Workflow | `blackhole-post-commit.yaml` |

**Last rebase on main:** 2026-05-27 ~01:05 UTC — rebased onto [`4b308296`](https://github.com/tenstorrent/tt-metal/commit/4b308296cb6a65b3ba8c27f1f10b0efef1443876); PR head [`3ae25c8`](https://github.com/tenstorrent/tt-metal/commit/3ae25c8ebf9e8c94c19b95dc7dd7b564bba9aba8).

**Status:** Draft; single verification [26482835281](https://github.com/tenstorrent/tt-metal/actions/runs/26482835281) **completed failure (infra)** — cannot mark ready; **no second verify**. **Main revalidation:** disable still valid (latest main BH post-commit [26473697334](https://github.com/tenstorrent/tt-metal/actions/runs/26473697334) still shows same assertion).

**Currently disabled — main status (issue #45109):**

| Disabled test ID | Still failing on main? | Last checked (UTC) | Main run |
|------------------|------------------------|--------------------|----------|
| `test_all_to_all_combine_no_trace[…-fabric_1d_ring_axis_0]` | **Yes** | 2026-05-27 01:05 | [26473697334](https://github.com/tenstorrent/tt-metal/actions/runs/26473697334) |

**Next:** Human review / accept disable on main-failure evidence only; no re-verify.

---

### `(T3K) T3000 e2e tests` — verify complete; watch trace-buffer; timeout tracked

| Item | Link |
|------|------|
| Draft PR | [#45108](https://github.com/tenstorrent/tt-metal/pull/45108) |
| Disable-tracking issue | [#45107](https://github.com/tenstorrent/tt-metal/issues/45107) |
| **Timeout-tracking issue** | [#45286](https://github.com/tenstorrent/tt-metal/issues/45286) |
| Branch | `ci/disable-failing-tests-t3000-e2e-tests-20260524` |
| Workflow | `t3000-e2e-tests.yaml` |

**Last rebase on main:** 2026-05-27 ~01:05 UTC — rebased onto [`4b308296`](https://github.com/tenstorrent/tt-metal/commit/4b308296cb6a65b3ba8c27f1f10b0efef1443876); PR head [`d4b91a7`](https://github.com/tenstorrent/tt-metal/commit/d4b91a7ac90322b1bb76a4c64096376ee05fa14a).

**Status:** Draft; verification [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) done — Llama pruned job **green** (no regression); CCL still red (trace-buffer param not disabled + 90m timeout). **14 disables** still justified on latest main e2e [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812). **No new disables** — `rs_input_shape2` still **1/5** main runs.

**Pending disable candidate (NOT added):** `test_reduce_scatter_async_sharded_to_interleaved[…-rs_input_shape2-…]` — **1/5** main runs.

**Next:** Watch main for 3× trace-buffer signature; no second verify. Consider merge-readiness only if team accepts CCL residual failure as out-of-scope.

---

### `t3000-demo-tests` — **ready to merge**

| Item | Link |
|------|------|
| PR | [#44938](https://github.com/tenstorrent/tt-metal/pull/44938) (undrafted) |
| Tracking issue | [#44937](https://github.com/tenstorrent/tt-metal/issues/44937) |
| Branch | `ci/disable-failing-tests-t3000-demo-tests-20260521` |

**Last rebase on main:** 2026-05-24 — **Needs fresh rebase** before merge vs `main` @ `4b308296`.

**Status:** Ready to merge pending rebase/review. Verification [26295163268](https://github.com/tenstorrent/tt-metal/actions/runs/26295163268) **success** (2026-05-22).

---

### Other open disable draft PRs

| Pipeline | PR | Notes |
|----------|-----|-------|
| `tt-metal-l2-tests` | [#44860](https://github.com/tenstorrent/tt-metal/pull/44860) | Separate agent scope |
| `(Blackhole) e2e tests` | [#45112](https://github.com/tenstorrent/tt-metal/pull/45112) | Needs rebase; no verify yet; `use-artifacts-from-run` not on `main` workflow |
| `(Blackhole) Demo tests` | [#45114](https://github.com/tenstorrent/tt-metal/pull/45114) | Needs rebase; no verify yet; `use-artifacts-from-run` not on `main` workflow |

---

## Pipeline Status Summary

| Pipeline | Status | PR |
|----------|--------|-----|
| `t3000-perf-tests` | Done | #44771 (merged) |
| Runtime unit tests | Done | merged |
| `t3000-demo-tests` | Ready to merge | #44938 |
| `(T3K) T3000 e2e tests` | Verify done; draft | #45108 |
| `Blackhole post-commit tests` | Verify inconclusive (infra); draft | #45110 |
| Nightly L2 tests | Out of scope | #44860 (separate agent) |
| Galaxy workflows | Out of scope | — |

---

## Blockers / Resolved

| Blocker | Status | Notes |
|---------|--------|-------|
| Active verification run blocks new dispatch | **Resolved** | No active runs |
| BH post-commit verify infra failure (#45110) | **Open** | Run 26482835281 — artifact download + container init; no re-verify |
| Trace-buffer disable candidate (#45108) | **Watch** | `rs_input_shape2` — 1/5 main runs |
| CCL job 90m timeout (#45108) | **Tracked** | [#45286](https://github.com/tenstorrent/tt-metal/issues/45286) |
| PRs behind main | **Resolved** | #45108, #45110 rebased onto `4b308296` 2026-05-27 |

---

## Change Log (append-only)

### 2026-05-27 ~01:10 UTC — automation: verify complete, rebases, timeout issue
- Marked run [26482835281](https://github.com/tenstorrent/tt-metal/actions/runs/26482835281) **COMPLETED** (infra failure, inconclusive for disable validation)
- Rebased [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) and [#45108](https://github.com/tenstorrent/tt-metal/pull/45108) onto `main` @ `4b308296`; heads `3ae25c8`, `d4b91a7`
- Revalidated disables unchanged (BH post-commit + T3K e2e latest main runs)
- Created timeout-tracking issue [#45286](https://github.com/tenstorrent/tt-metal/issues/45286) for T3K CCL 90m timeout
- Updated PR bodies; no verification dispatch (budgets exhausted on #45108/#45110)

### 2026-05-27 ~00:20 UTC — PR #45110 rebase + verification dispatch
- Rebased onto `main` @ `4cd9b59a8d2`; dispatched verify [26482835281](https://github.com/tenstorrent/tt-metal/actions/runs/26482835281)

### 2026-05-26 ~20:35 UTC — automation poll (status log only)
- Revalidated 14 CCL disables on main e2e [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812)

### 2026-05-26 ~19:15 UTC — verification complete (PR #45108)
- Run 26460410854: Llama success, CCL failure + timeout

### 2026-05-26 ~16:13 UTC — targeted verification dispatched (PR #45108)
- Run 26460410854; artifact reuse 26445104085

### 2026-05-24 — PR #45108 / #45110 opened

### 2026-05-22 — PR #44938 verification success
