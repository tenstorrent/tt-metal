# CI Disable Work — Running Status Log

Last updated: **2026-05-26T17:30 UTC** (automation backfill after status-log protocol added)

Operational policy: **one workflow run at a time**, draft PRs only, artifact reuse for verification, Galaxy out of scope.

---

## Active Runs

| Run | Pipeline | Branch | Started (UTC) | Jobs | Artifact source | Status |
|-----|----------|--------|---------------|------|-----------------|--------|
| [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) | `(T3K) T3000 e2e tests` | `verify/ci-disable-t3000-e2e-20260526` | 2026-05-26 16:13 | `t3k_ccl_tests [wh_llmbox]` (**in progress**), `models_tttv2_llama31_8B_tests [wh_llmbox]` (**success**) | `26445104085` (Merge Gate Release, main `5e9f894`) | **ACTIVE** — pruned targeted verification for PR #45108 |

> Do **not** dispatch another workflow until this run completes.

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-------|
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

**Status:** Awaiting targeted verification (run [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) active). PR rebased/synced 2026-05-26; mergeable_state was behind main at last check — rebase again if needed before undraft.

**Latest disable-set change (2026-05-26):**
- **Removed:** Llama 3.1 batch-32 demo skips in `models/common/demos/llama31_8B_demo.py` — main run `26438570812` shows `models_tttv2_llama31_8B_tests [wh_llmbox]` **success**
- **Kept:** All CCL disables below (still failing on main)

**Currently disabled (issue #45107):**

| File | Disabled test(s) |
|------|------------------|
| `tests/nightly/t3000/ccl/test_minimal_all_gather_matmul_async.py` | 6× `test_all_gather_matmul_async[…]` params (ag_output_shape0/1, various barrier/chunking combos) |
| `tests/nightly/t3000/ccl/test_ring_joint_attention.py` | `test_ring_joint_sdpa_program_cache[bf16\|bf8_b\|bf4_b-sd35-no_trace]` |
| `tests/nightly/t3000/ccl/test_new_all_broadcast.py` | `test_all_broadcast_sharded_2x4` — ROW_MAJOR+bf16, TILE+bf16, TILE+bf8_b |
| `tests/nightly/t3000/ccl/test_all_to_all_dispatch.py` | `test_decode_perf[…-2x4_grid-True-device_params0]` |
| `tests/nightly/t3000/ccl/test_all_to_all_combine.py` | `test_all_to_all_combine_no_trace_submesh[…-fabric_1d_line_axis_1]` |

**Main revalidation (run 26438570812):** CCL job still fails with deterministic errors including dtype mismatch in all_gather_matmul_async, `cache_entries_counter` / `CacheEntriesCounter` in ring_joint and all_broadcast, zero-output accuracy in combine/dispatch. Ring joint skips **not** removed despite PR #45104 on main — failures persist on latest main.

**Next after verification:** If CCL job passes on verify branch, update PR/issue and consider undraft handoff. If CCL fails, analyze logs for minimal additional disables (3× same-error rule only). Do not dispatch a second run while 26460410854 is active.

---

### `t3000-demo-tests` — **ready to merge**

| Item | Link |
|------|------|
| Draft PR | [#44938](https://github.com/tenstorrent/tt-metal/pull/44938) |
| Tracking issue | [#44937](https://github.com/tenstorrent/tt-metal/issues/44937) |
| Branch | `ci/disable-failing-tests-t3000-demo-tests-20260521` |

**Status:** Ready to merge. Verification run [26295163268](https://github.com/tenstorrent/tt-metal/actions/runs/26295163268) **success** (2026-05-22). One disable: `test_sd35_pipeline[…-2x4cfg1sp0tp1-…]`. Remaining main failures classified as infra (DRAM training, GH download timeout) — not disableable.

**Queue:** Advance after PR #45108 verification completes and branch is stable.

---

### Other open disable draft PRs (not active this cycle)

| Pipeline | PR | Issue | Notes |
|----------|-----|-------|-------|
| `tt-metal-l2-tests` | [#44860](https://github.com/tenstorrent/tt-metal/pull/44860) | — | Stale since 2026-05-22; separate agent scope per context doc |
| Blackhole post-commit | [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) | — | Created 2026-05-24; not advanced |
| `(Blackhole) e2e tests` | [#45112](https://github.com/tenstorrent/tt-metal/pull/45112) | — | Created 2026-05-24; not advanced |
| `(Blackhole) Demo tests` | [#45114](https://github.com/tenstorrent/tt-metal/pull/45114) | — | Created 2026-05-24; not advanced |

---

## Pipeline Status Summary

| Pipeline | Status | PR |
|----------|--------|-----|
| `t3000-perf-tests` | Done | #44771 (merged) |
| Runtime unit tests | Done | merged |
| `t3000-demo-tests` | Ready to merge | #44938 |
| `(T3K) T3000 e2e tests` | In progress — awaiting verification | #45108 |
| Nightly L2 tests | Out of scope | #44860 (separate agent) |
| Galaxy workflows | Out of scope | — |

---

## Blockers / Resolved

| Blocker | Status | Notes |
|---------|--------|-------|
| Active verification run blocks new dispatch | **Active** | Run 26460410854 in progress since 2026-05-26 16:13 UTC |
| GitHub REST `Bad credentials` with Bearer token | **Workaround** | Use `gh` CLI or `Authorization: token` header |
| PR #45108 behind main | **Watch** | Rebased/synced 2026-05-26; re-check before undraft |
| No compatible artifact source | **Resolved** | Used Merge Gate run `26445104085` for current verification |

---

## Change Log (append-only)

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
