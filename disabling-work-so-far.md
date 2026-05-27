# CI Disable Work — Running Status Log

Last updated: **2026-05-27T00:20 UTC** (PR #45110 rebase + verification dispatch)

Operational policy: **one workflow run at a time**, draft PRs only, artifact reuse for verification, Galaxy out of scope.

**Legend — main-failure checks:** Each disabled test row records the last time it was confirmed still failing (or passing / removed) on `main`, with a link to the main pipeline run used for that check.

**Legend — rebase tracking:** Each PR records the last rebase onto `main` (UTC timestamp + base commit SHA rebased onto).

---

## Active Runs

| Run | Pipeline | Branch | Started (UTC) | Status | Notes |
|-----|----------|--------|---------------|--------|-------|
| [26482835281](https://github.com/tenstorrent/tt-metal/actions/runs/26482835281) | `Blackhole post-commit tests` | `verify/ci-disable-blackhole-post-commit-20260526` | 2026-05-27 00:18 | **in_progress** | PR #45110 first/only verify: pruned to `blackhole-multi-card-fast-unit-tests`; artifact reuse Merge Gate `26480351906` |

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-------|
| [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) | `(T3K) T3000 e2e tests` | `verify/ci-disable-t3000-e2e-20260526` | 2026-05-26 16:13 | 2026-05-26 18:49 | **failure** | Pruned verify for PR #45108: Llama job **success**; CCL job **failure** — one pytest failure (trace buffer overflow on `rs_input_shape2` fabric_ring param) then **90m job timeout**; artifact reuse `26445104085` |
| [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) | `(T3K) T3000 e2e tests` | `main` | 2026-05-26 07:27 | 2026-05-26 09:25 | **failure** | Main revalidation: Llama job green; `t3k_ccl_tests` failed (keeps CCL disables) |
| [26368616671](https://github.com/tenstorrent/tt-metal/actions/runs/26368616671) | `(T3K) T3000 e2e tests` | verify branch (prior) | — | — | — | Prior targeted verification for PR #45108 (see PR body) |
| [26295163268](https://github.com/tenstorrent/tt-metal/actions/runs/26295163268) | `t3000-demo-tests` | `ci/disable-failing-tests-t3000-demo-tests-20260521` | — | 2026-05-22 21:04 | **success** | PR #44938 verification passed |

---

## In-Progress Pipelines / PRs

### `Blackhole post-commit tests` — **ACTIVE** (verification in flight)

| Item | Link |
|------|------|
| Draft PR | [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) |
| Tracking issue | [#45109](https://github.com/tenstorrent/tt-metal/issues/45109) |
| Branch | `ci/disable-failing-tests-blackhole-post-commit-20260524` |
| Workflow | `blackhole-post-commit.yaml` |

**Last rebase on main:** 2026-05-27 ~00:15 UTC — rebased onto [`4cd9b59a8d2`](https://github.com/tenstorrent/tt-metal/commit/4cd9b59a8d25b72bb9e43b16496d98ef6d3aafa5); PR head [`ba24f484ab5`](https://github.com/tenstorrent/tt-metal/commit/ba24f484ab58338cd71625d74f6dd9dedd909f62).

**Status:** Draft; awaiting single verification run [26482835281](https://github.com/tenstorrent/tt-metal/actions/runs/26482835281) (ACTIVE). **Main revalidation (2026-05-27):** `test_all_to_all_combine_no_trace[…-fabric_1d_ring_axis_0]` still fails with `AssertionError: Equal check failed` (zero tensor) on 3 consecutive `main` runs [26473697334](https://github.com/tenstorrent/tt-metal/actions/runs/26473697334), [26462959668](https://github.com/tenstorrent/tt-metal/actions/runs/26462959668), [26449034552](https://github.com/tenstorrent/tt-metal/actions/runs/26449034552).

**Currently disabled — main status (issue #45109):**

| Disabled test ID | Still failing on main? | Last checked (UTC) | Main run |
|------------------|------------------------|--------------------|----------|
| `test_all_to_all_combine_no_trace[…-fabric_1d_ring_axis_0]` | **Yes** — zero-output accuracy assertion | 2026-05-27 00:15 | [26473697334](https://github.com/tenstorrent/tt-metal/actions/runs/26473697334) |

**Next:** When run 26482835281 completes, evaluate merge-readiness (no second verify). Do not add disables after first batch.

---

### `(T3K) T3000 e2e tests` — verification complete; watch trace-buffer candidate

| Item | Link |
|------|------|
| Draft PR | [#45108](https://github.com/tenstorrent/tt-metal/pull/45108) |
| Tracking issue | [#45107](https://github.com/tenstorrent/tt-metal/issues/45107) |
| Branch | `ci/disable-failing-tests-t3000-e2e-tests-20260524` |
| Workflow | `t3000-e2e-tests.yaml` |

**Last rebase on main:** 2026-05-26 ~20:30 UTC — confirmed up to date with [`ff4b73d01`](https://github.com/tenstorrent/tt-metal/commit/ff4b73d01641cc35e43d994372d3127445375150) (`main` tip at that time); PR head [`7a1c5d55f1f`](https://github.com/tenstorrent/tt-metal/commit/7a1c5d55f1f1f52d3cf56d4b0941e412cb79d9e0). **May need rebase** vs current `main` @ `4cd9b59a8d2`.

**Status:** Draft; single verification [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) **completed failure** — Llama pruned job **green**; CCL pruned job still red. All 14 CCL disables still justified on [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812). **No new disables** — trace-buffer `rs_input_shape2` still **1/5** recent main runs (3× rule).

**Currently disabled — main status (issue #45107):** (unchanged — see prior log rows for full 14-test table)

**Pending disable candidate (NOT added — 3× rule):** `test_reduce_scatter_async_sharded_to_interleaved[…-rs_input_shape2-…]` trace-buffer `TT_FATAL` — **1/5** main runs.

**Next:** Wait for 2 more consecutive main runs with same trace-buffer error; no second verification dispatch for this PR.

---

### `t3000-demo-tests` — **ready to merge**

| Item | Link |
|------|------|
| PR | [#44938](https://github.com/tenstorrent/tt-metal/pull/44938) (undrafted) |
| Tracking issue | [#44937](https://github.com/tenstorrent/tt-metal/issues/44937) |
| Branch | `ci/disable-failing-tests-t3000-demo-tests-20260521` |

**Last rebase on main:** 2026-05-24 — **Needs fresh rebase** before merge vs `main` @ `4cd9b59a8d2`.

**Status:** Ready to merge pending rebase/review. Verification [26295163268](https://github.com/tenstorrent/tt-metal/actions/runs/26295163268) **success** (2026-05-22).

---

### Other open disable draft PRs

| Pipeline | PR | Notes |
|----------|-----|-------|
| `tt-metal-l2-tests` | [#44860](https://github.com/tenstorrent/tt-metal/pull/44860) | Separate agent scope |
| `(Blackhole) e2e tests` | [#45112](https://github.com/tenstorrent/tt-metal/pull/45112) | Needs rebase; no verify yet; `use-artifacts-from-run` blocker on workflow |
| `(Blackhole) Demo tests` | [#45114](https://github.com/tenstorrent/tt-metal/pull/45114) | Needs rebase; no verify yet; `use-artifacts-from-run` blocker on workflow |

---

## Pipeline Status Summary

| Pipeline | Status | PR |
|----------|--------|-----|
| `t3000-perf-tests` | Done | #44771 (merged) |
| Runtime unit tests | Done | merged |
| `t3000-demo-tests` | Ready to merge | #44938 |
| `(T3K) T3000 e2e tests` | Verify done; watch trace-buffer 3× | #45108 |
| `Blackhole post-commit tests` | Verify in flight | #45110 |
| Nightly L2 tests | Out of scope | #44860 (separate agent) |
| Galaxy workflows | Out of scope | — |

---

## Blockers / Resolved

| Blocker | Status | Notes |
|---------|--------|-------|
| Active verification run blocks new dispatch | **ACTIVE** | Run [26482835281](https://github.com/tenstorrent/tt-metal/actions/runs/26482835281) in progress for PR #45110 |
| BH post-commit missing `use-artifacts-from-run` on `main` | **Resolved (verify branch)** | Added on `verify/ci-disable-blackhole-post-commit-20260526` only |
| Trace-buffer disable candidate (#45108) | **Watch** | `rs_input_shape2` — 1/5 main runs |
| CCL job 90m timeout (#45108) | **Out of scope** | Timeout-tracking only |
| PR #45110 behind main | **Resolved** | Rebased 2026-05-27 onto `4cd9b59a8d2` |

---

## Change Log (append-only)

### 2026-05-27 ~00:20 UTC — PR #45110 rebase + verification dispatch
- Rebased `ci/disable-failing-tests-blackhole-post-commit-20260524` onto `main` @ `4cd9b59a8d2`; head `ba24f484ab5`
- Revalidated disable: `test_all_to_all_combine_no_trace[…-fabric_1d_ring_axis_0]` still failing on 3 consecutive main runs
- Dispatched verify run [26482835281](https://github.com/tenstorrent/tt-metal/actions/runs/26482835281) on `verify/ci-disable-blackhole-post-commit-20260526` (artifact reuse `26480351906`; pruned to `blackhole-multi-card-fast-unit-tests`)
- Updated PR #45110 body and issue #45109 disable list

### 2026-05-26 ~20:35 UTC — automation poll (status log only)
- Confirmed PR #45108 rebased on `main` @ `ff4b73d01`; head `7a1c5d55f1f`; still draft/blocked
- Revalidated 14 CCL disables against latest `main` e2e run [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) — still failing
- Trace-buffer `rs_input_shape2` still 1/5 main runs — not disabled
- No disable edits; no verification dispatch

### 2026-05-26 ~19:15 UTC — verification complete (no code changes)
- Run 26460410854 **completed failure**: Llama **success**, CCL **failure** (trace buffer `rs_input_shape2` + 90m timeout)
- Log analysis: 14 existing disables still appropriate; trace-buffer param **not** disabled (1/5 main runs only)
- PR #45108 rebased/current with `main` @ `4b466c68cf3`; issue/PR bodies updated
- No new workflow dispatch this cycle

### 2026-05-26 ~16:13 UTC — targeted verification dispatched (PR #45108)
- Started run 26460410854 on `verify/ci-disable-t3000-e2e-20260526`
- Pruned jobs: `t3k_ccl_tests`, `models_tttv2_llama31_8B_tests`
- Artifact reuse from run 26445104085

### 2026-05-24 — PR #45108 / #45110 opened
- Initial disable batches for T3K e2e and Blackhole post-commit

### 2026-05-22 — PR #44938 verification success
- `t3000-demo-tests` targeted verification green (run 26295163268)
