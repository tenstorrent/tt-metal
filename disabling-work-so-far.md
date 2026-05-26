# CI Disable Work — Running Status Log

Last updated: **2026-05-26T20:35 UTC** (revalidation poll; no code/disable changes)

Operational policy: **one workflow run at a time**, draft PRs only, artifact reuse for verification, Galaxy out of scope.

**Legend — main-failure checks:** Each disabled test row records the last time it was confirmed still failing (or passing / removed) on `main`, with a link to the main pipeline run used for that check.

**Legend — rebase tracking:** Each PR records the last rebase onto `main` (UTC timestamp + base commit SHA rebased onto).

---

## Active Runs

_None — safe to dispatch at most one new verification run when needed._

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

### `(T3K) T3000 e2e tests` — **ACTIVE** (priority)

| Item | Link |
|------|------|
| Draft PR | [#45108](https://github.com/tenstorrent/tt-metal/pull/45108) |
| Tracking issue | [#45107](https://github.com/tenstorrent/tt-metal/issues/45107) |
| Branch | `ci/disable-failing-tests-t3000-e2e-tests-20260524` |
| Workflow | `t3000-e2e-tests.yaml` |

**Last rebase on main:** 2026-05-26 ~20:30 UTC — confirmed up to date with [`ff4b73d01`](https://github.com/tenstorrent/tt-metal/commit/ff4b73d01641cc35e43d994372d3127445375150) (`main` tip); PR head [`7a1c5d55f1f`](https://github.com/tenstorrent/tt-metal/commit/7a1c5d55f1f1f52d3cf56d4b0941e412cb79d9e0). No rebase commit needed this cycle.

**Status:** Draft; `mergeable_state: blocked` (CI). Targeted verification [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) **completed failure** — Llama pruned job **green**; CCL pruned job still red. **Main revalidation (2026-05-26 ~20:30 UTC):** latest `main` `(T3K) T3000 e2e tests` run still [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812); all 14 CCL disables still justified (CCL job failed; AI summary lists same six failure classes including disabled tests). **No new disables** — trace-buffer `rs_input_shape2` still **1/5** recent main runs (3× rule).

**Latest disable-set change (2026-05-26):**
- **Removed:** Llama 3.1 batch-32 demo skips — passing on main (see removed-test row below)
- **Kept:** All CCL disables below (still failing on main as of last check)

**Main check source run:** [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) on `main` @ [`213a10be7cb`](https://github.com/tenstorrent/tt-metal/commit/213a10be7cb02a2f47a1af33f1fc0720b1452533) — checked **2026-05-26 20:30 UTC** (still latest completed `main` e2e run; `t3k_ccl_tests [wh_llmbox]` failed; `models_tttv2_llama31_8B_tests [wh_llmbox]` passed)

**Currently disabled — main status (issue #45107):**

| Disabled test ID | Still failing on main? | Last checked (UTC) | Main run |
|------------------|------------------------|--------------------|----------|
| `test_all_gather_matmul_async[ag_output_shape0-perf-no_barrier_with_persistent-chunking]` | **Yes** — dtype mismatch (Float vs BFloat16) | 2026-05-26 20:30 | [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) |
| `test_all_gather_matmul_async[ag_output_shape0-check-barrier_with_persistent-default]` | **Yes** — same error class | 2026-05-26 20:30 | [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) |
| `test_all_gather_matmul_async[ag_output_shape0-perf-barrier_without_persistent-default]` | **Yes** — same error class | 2026-05-26 20:30 | [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) |
| `test_all_gather_matmul_async[ag_output_shape1-check-barrier_with_persistent-default]` | **Yes** — same error class | 2026-05-26 20:30 | [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) |
| `test_all_gather_matmul_async[ag_output_shape1-perf-no_barrier_with_persistent-chunking]` | **Yes** — same error class | 2026-05-26 20:30 | [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) |
| `test_all_gather_matmul_async[ag_output_shape1-check-barrier_without_persistent-chunking]` | **Yes** — same error class | 2026-05-26 20:30 | [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) |
| `test_ring_joint_sdpa_program_cache[bf16-sd35-no_trace]` | **Yes** — `cache_entries_counter` / `CacheEntriesCounter` | 2026-05-26 20:30 | [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) |
| `test_ring_joint_sdpa_program_cache[bf8_b-sd35-no_trace]` | **Yes** — same error class | 2026-05-26 20:30 | [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) |
| `test_ring_joint_sdpa_program_cache[bf4_b-sd35-no_trace]` | **Yes** — same error class | 2026-05-26 20:30 | [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) |
| `test_all_broadcast_sharded_2x4` (ROW_MAJOR + bfloat16) | **Yes** — `CacheEntriesCounter` undefined | 2026-05-26 20:30 | [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) |
| `test_all_broadcast_sharded_2x4` (TILE + bfloat16) | **Yes** — same error class | 2026-05-26 20:30 | [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) |
| `test_all_broadcast_sharded_2x4` (TILE + bfloat8_b) | **Yes** — same error class | 2026-05-26 20:30 | [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) |
| `test_decode_perf[wormhole_b0-…-2x4_grid-True-device_params0]` | **Yes** — zero-output accuracy assertion | 2026-05-26 20:30 | [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) |
| `test_all_to_all_combine_no_trace_submesh[…-fabric_1d_line_axis_1]` | **Yes** — zero-output accuracy assertion | 2026-05-26 20:30 | [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) |

**Removed from disable set (was disabled, now passing on main):**

| Test ID | Main status at removal | Last checked (UTC) | Main run |
|---------|------------------------|--------------------|----------|
| `test_mlp1d_llama_demo[…-batch-32-performance-…]` | **Passing** — job green | 2026-05-26 09:25 | [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) |
| `test_mlp1d_llama_demo[…-batch-32-accuracy-…]` | **Passing** — job green | 2026-05-26 09:25 | [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) |

**Pending disable candidate (NOT added — 3× rule):**

| Test ID | Error (same on verify + main 26438570812) | Consecutive main failures | Last checked (UTC) |
|---------|-------------------------------------------|---------------------------|--------------------|
| `test_reduce_scatter_async_sharded_to_interleaved[wormhole_b0-fabric_ring-rs_input_shape2-…-HEIGHT_SHARDED-…-L1-…]` | `TT_FATAL: Creating trace buffers of size 1409024B … only 1271456B allocated` | **1/5** recent main runs (26438570812 only; absent in 26389759743, 26354956826, 26326515951, 26274534025) | 2026-05-26 20:30 |

**Next:** Wait for 2 more consecutive main runs with same trace-buffer error before disabling; then CCL-only pruned verify with artifact reuse. Do not chase the 90m global job timeout on CCL (pytest continued after first failure).

---

### `t3000-demo-tests` — **ready to merge**

| Item | Link |
|------|------|
| PR | [#44938](https://github.com/tenstorrent/tt-metal/pull/44938) (undrafted) |
| Tracking issue | [#44937](https://github.com/tenstorrent/tt-metal/issues/44937) |
| Branch | `ci/disable-failing-tests-t3000-demo-tests-20260521` |

**Last rebase on main:** 2026-05-24 (per context doc) — exact base commit not recorded in automation log; PR head at last update: [`a2fc1b13fb1`](https://github.com/tenstorrent/tt-metal/commit/a2fc1b13fb11de19fe18350c19e08f4dd6cb5421). **Needs fresh rebase** before merge (mergeable_state: blocked; base SHA stale vs current main).

**Status:** Ready to merge pending rebase/review. Verification run [26295163268](https://github.com/tenstorrent/tt-metal/actions/runs/26295163268) **success** (2026-05-22).

**Disabled test — main status:**

| Disabled test ID | Still failing on main? | Last checked (UTC) | Main run / evidence |
|------------------|------------------------|--------------------|-----------------------|
| `test_sd35_pipeline[wormhole_b0-no_traced-device_params0-2x4cfg1sp0tp1-large-1024-1024-3.5-28-True]` | **Yes** — exit 134 / device hang | 2026-05-24 (context doc) | 5/5 consecutive main runs since 2026-05-15; not re-checked in 2026-05-26 automation cycle |

**Queue:** Advance after PR #45108 verification completes and branch is stable.

---

### Other open disable draft PRs (not active this cycle)

| Pipeline | PR | Last rebase on main | Main-failure check | Notes |
|----------|-----|---------------------|--------------------|-------|
| `tt-metal-l2-tests` | [#44860](https://github.com/tenstorrent/tt-metal/pull/44860) | Unknown — last PR activity 2026-05-22 | Not checked this cycle | Separate agent scope |
| Blackhole post-commit | [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) | Unknown — created 2026-05-24 | Not checked this cycle | Not advanced |
| `(Blackhole) e2e tests` | [#45112](https://github.com/tenstorrent/tt-metal/pull/45112) | Unknown — created 2026-05-24 | Not checked this cycle | Not advanced |
| `(Blackhole) Demo tests` | [#45114](https://github.com/tenstorrent/tt-metal/pull/45114) | Unknown — created 2026-05-24 | Not checked this cycle | Not advanced |

---

## Pipeline Status Summary

| Pipeline | Status | PR |
|----------|--------|-----|
| `t3000-perf-tests` | Done | #44771 (merged) |
| Runtime unit tests | Done | merged |
| `t3000-demo-tests` | Ready to merge | #44938 |
| `(T3K) T3000 e2e tests` | In progress — verify done; pending 3× trace-buffer or next disable cycle | #45108 |
| Nightly L2 tests | Out of scope | #44860 (separate agent) |
| Galaxy workflows | Out of scope | — |

---

## Blockers / Resolved

| Blocker | Status | Notes |
|---------|--------|-------|
| Active verification run blocks new dispatch | **Resolved** | Run 26460410854 completed 2026-05-26 18:49 UTC |
| Trace-buffer disable candidate | **Watch** | `rs_input_shape2` / fabric_ring — only 1/5 main runs; need 3× same error before skip |
| CCL job 90m timeout after first failure | **Out of scope** | Global job timeout; not a disable target |
| GitHub REST `Bad credentials` with Bearer token | **Workaround** | Use `gh` CLI or `Authorization: token` header |
| PR #45108 behind main | **Resolved** | Up to date with `main` @ `ff4b73d01` (confirmed 2026-05-26 20:30 UTC) |
| No compatible artifact source | **Resolved** | Used Merge Gate run `26445104085` for verification |

---

## Change Log (append-only)

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

### 2026-05-26 ~18:02 UTC — automation poll (no code changes)
- Polled run 26460410854: workflow `in_progress`; Llama job **success**; CCL job **in_progress** (pytest)
- No new workflow dispatch; no disable edits on PR #45108
- Recorded current `main` tip `231c7223899` for post-verify rebase planning

### 2026-05-26 ~18:00 UTC — status log format update
- Added per-test main-failure check timestamps (linked to run 26438570812)
- Added PR rebase tracking (timestamp + base commit SHA) for #45108 and #44938
- No new workflow runs; reporting-only update

### 2026-05-26 ~17:00 UTC — automation cycle (PR #45108)
- Removed Llama batch-32 skips after main revalidation (job green on 26438570812)
- Kept all CCL disables; confirmed ring_joint still failing on main
- Updated issue #45107 and PR #45108 descriptions
- Pushed commit `32e4d9b9c29` to `ci/disable-failing-tests-t3000-e2e-tests-20260524`
- Verification run 26460410854 still ACTIVE (Llama pruned job passed; CCL running)

### 2026-05-26 ~16:13 UTC — targeted verification dispatched
- Started run 26460410854 on `verify/ci-disable-t3000-e2e-20260526`
- Pruned jobs: `t3k_ccl_tests`, `models_tttv2_llama31_8B_tests`
- Artifact reuse from run 26445104085

### 2026-05-24 — PR #45108 opened
- Initial CCL + Llama disables for `(T3K) T3000 e2e tests`
- Tracking issue #45107 created

### 2026-05-22 — PR #44938 verification success
- `t3000-demo-tests` targeted verification green (run 26295163268)
