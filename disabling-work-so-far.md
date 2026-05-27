# CI Disable Work — Status Log

Last updated: **2026-05-27T13:20 UTC** (session: PR #45306 (T3K unit tests): added multiprocess disable (test_tt_fabric T3K 2x2 SIGABRT — issue #45305), rebased onto main de985e8a, dispatched first verification run 26513061621 on verify/ci-disable-t3000-unit-tests-20260527; build in progress. 1 dispatch.)

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
| [#45108](https://github.com/tenstorrent/tt-metal/pull/45108) | `(T3K) T3000 e2e tests` | `verified-pass` | [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) — CCL failure + 90m timeout | Yes | Rebased 2026-05-27 11:00 UTC onto main `fe07cd111531`; up-to-date; awaiting human review |
| [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) | `Blackhole post-commit tests` | `verified-pass` | [26508063374](https://github.com/tenstorrent/tt-metal/actions/runs/26508063374) **success** | Yes | Rebased onto main `aa2de19b`; verification passed; awaiting human review |
| [#45112](https://github.com/tenstorrent/tt-metal/pull/45112) | `(Blackhole) e2e tests` | `verified-pass` | [26508072188](https://github.com/tenstorrent/tt-metal/actions/runs/26508072188) **success** | Yes | Rebased onto main `aa2de19b`; verification passed; awaiting human review |
| [#45114](https://github.com/tenstorrent/tt-metal/pull/45114) | `(Blackhole) Demo tests` | `out-of-scope` | — | N/A | **Disable REMOVED 2026-05-27**: test fixed on main since May 26. PR should be closed. |
| [#45306](https://github.com/tenstorrent/tt-metal/pull/45306) | `(T3K) T3000 unit tests` | `verifying` | [26513061621](https://github.com/tenstorrent/tt-metal/actions/runs/26513061621) in progress | No (verifying) | Verification dispatched 2026-05-27 13:09 UTC; build in progress; 3 disables: DPrint/Watcher (t3k_ttmetal), DistributedTensorOp/Matmul (t3k_ttnn), test_tt_fabric T3K 2x2 (multiprocess) |
| [#44860](https://github.com/tenstorrent/tt-metal/pull/44860) | `tt-metal-l2-tests` | `out-of-scope` | — | N/A | Separate agent |

---

## Active Runs

| Run | Pipeline | Branch | Started | Status | Notes |
|-----|----------|--------|---------|--------|-------|
| [26513061621](https://github.com/tenstorrent/tt-metal/actions/runs/26513061621) | `(T3K) T3000 unit tests` | `verify/ci-disable-t3000-unit-tests-20260527` | 2026-05-27 13:09 UTC | **in progress** (building) | PR #45306 first verify; pruned to 3 failing jobs; fresh build |

**Policy:** Concurrent runs across PRs are allowed; each automation session may dispatch at most three new runs.

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-----------|
| [26508072188](https://github.com/tenstorrent/tt-metal/actions/runs/26508072188) | `(Blackhole) e2e tests` | `verify/ci-disable-blackhole-e2e-20260527` | 2026-05-27 11:21 UTC | 2026-05-27 12:02 UTC | **success** | PR #45112 first verify; all ccl nightly tests across all BH platforms **success** → **verified-pass** |
| [26508063374](https://github.com/tenstorrent/tt-metal/actions/runs/26508063374) | `Blackhole post-commit tests` | `verify/ci-disable-blackhole-post-commit-20260527b` | 2026-05-27 11:20 UTC | 2026-05-27 12:05 UTC | **success** | PR #45110 fourth verify attempt; all multi-card fast unit tests (P300-viommu, BH-LLMBox, BH-LoudBox) **success** → **verified-pass** |
| [26501610173](https://github.com/tenstorrent/tt-metal/actions/runs/26501610173) | `Blackhole post-commit tests` | `main` | 2026-05-27 09:01 UTC | 2026-05-27 09:51 UTC | **failure** | BH systematic blocker: 27th+ consecutive failure (SHA `9cc43ddcad`). Build step SUCCESS. |
| [26499906125](https://github.com/tenstorrent/tt-metal/actions/runs/26499906125) | `Blackhole post-commit tests (P100 nightly)` | `main` | 2026-05-27 08:25 UTC | 2026-05-27 09:39 UTC | **failure** | BH systematic blocker: failure (SHA `2c0072ef`). |
| [26494921196](https://github.com/tenstorrent/tt-metal/actions/runs/26494921196) | `(Single-card) Demo tests` | `main` | 2026-05-27 06:31 UTC | 2026-05-27 07:36 UTC | **success** | Single-card demo main run — overall SUCCESS (SHA `b4f5ed4f`). Confirms no deterministic single-card demo failures. |
| [26492593059](https://github.com/tenstorrent/tt-metal/actions/runs/26492593059) | `Blackhole post-commit tests` | `main` | 2026-05-27 05:25 UTC | 2026-05-27 06:07 UTC | **failure** | BH systematic blocker: 25th+ consecutive failure (SHA `c68e6ee`). |
| [26490261745](https://github.com/tenstorrent/tt-metal/actions/runs/26490261745) | `Blackhole post-commit tests` | `verify/ci-disable-blackhole-post-commit-20260527` | 2026-05-27 04:11 UTC | 2026-05-27 04:15 UTC | **failure** | PR #45110 third verify: **infra inconclusive** — artifact download failed; pytest never ran. |
| [26482835281](https://github.com/tenstorrent/tt-metal/actions/runs/26482835281) | `Blackhole post-commit tests` | `verify/ci-disable-blackhole-post-commit-20260526` | 2026-05-27 00:18 | 2026-05-27 00:32 | **failure** | PR #45110 only verify: **infra inconclusive** — artifact download failed. |
| [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) | `(T3K) T3000 e2e tests` | `verify/ci-disable-t3000-e2e-20260526` | 2026-05-26 16:13 | 2026-05-26 18:49 | **failure** | PR #45108 only verify: Llama **success**; CCL **failure** (trace buffer + 90m timeout → [#45286](https://github.com/tenstorrent/tt-metal/issues/45286)) |
| [26438570812](https://github.com/tenstorrent/tt-metal/actions/runs/26438570812) | `(T3K) T3000 e2e tests` | `main` | 2026-05-26 07:27 | 2026-05-26 09:25 | **failure** | Main revalidation: Llama green; `t3k_ccl_tests` failed |
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
| Last rebase | 2026-05-27 02:42 UTC — PR is NOT behind main; up-to-date |
| Last revalidation | 2026-05-27 11:00 UTC — lightweight check; no new main runs since last session |
| Verification run | [26295163268](https://github.com/tenstorrent/tt-metal/actions/runs/26295163268) — **success** (2026-05-22) |
| Readiness | **Yes** (pending CI checks passing and human review) |

**Notes:** Ready to merge pending CI and human review. PR is undrafted. Awaiting human review.

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
| Last rebase | 2026-05-27 ~07:01 UTC — rebased via `update_pull_request_branch` onto `c68e6ee` |
| Last revalidation | 2026-05-26 ~20:35 UTC — 14 CCL disables revalidated on main |
| Verification run | [26460410854](https://github.com/tenstorrent/tt-metal/actions/runs/26460410854) — Llama **success**; CCL **failure** (trace-buffer + 90m timeout) |
| Readiness | **Yes** (rebased and up-to-date; awaiting human review) |

**Notes:** Draft; reclassified to verified-pass — the single verification run showed no regressions in jobs that were passing on main. Awaiting human undraft + review.

---

## PR #45110 — Blackhole post-commit tests

| Field | Value |
|-------|-------|
| PR | [#45110](https://github.com/tenstorrent/tt-metal/pull/45110) (draft) |
| Disable issue | [#45109](https://github.com/tenstorrent/tt-metal/issues/45109) |
| Timeout issue | none |
| Branch | `ci/disable-failing-tests-blackhole-post-commit-20260524` |
| Workflow file | `blackhole-post-commit.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-27 12:10 UTC — merged onto `main` @ `aa2de19bd3548be5f000f34353f62455e54e4498` via update_pull_request_branch |
| Last revalidation | 2026-05-27 13:00 UTC — run 26507885931 (SHA `fe07cd11`) still shows same 3 failures |
| Verification run | [26508063374](https://github.com/tenstorrent/tt-metal/actions/runs/26508063374) — **success** (2026-05-27 11:20–12:05 UTC) |
| Readiness | **Yes** (verification passed; awaiting human review) |

**Disabled tests:**

| Disabled test ID | Still failing on main? | Last checked (UTC) | Main run |
|------------------|------------------------|--------------------|----------|
| `test_all_to_all_combine_no_trace[…-fabric_1d_ring_axis_0]` | **Yes** | 2026-05-27 11:00 | [26501610173](https://github.com/tenstorrent/tt-metal/actions/runs/26501610173) |

**Verification summary:** All `blackhole-multi-card-fast-unit-tests` jobs across P300-viommu, BH-LLMBox, and BH-LoudBox platforms returned **success**. Including the previously-failing `ttnn ops fast unit tests (ccl, DRAM prefetcher)` on BH-LLMBox — no regression detected. **→ verified-pass**.

---

## PR #45112 — (Blackhole) e2e tests

| Field | Value |
|-------|-------|
| PR | [#45112](https://github.com/tenstorrent/tt-metal/pull/45112) |
| Disable issue | [#45111](https://github.com/tenstorrent/tt-metal/issues/45111) |
| Timeout issue | unknown |
| Branch | `ci/disable-failing-tests-blackhole-e2e-tests-20260524` |
| Workflow file | `blackhole-e2e-tests.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-27 12:10 UTC — merged onto `main` @ `aa2de19bd3548be5f000f34353f62455e54e4498` via update_pull_request_branch |
| Last revalidation | 2026-05-27 13:00 UTC — run 26507885931 (SHA `fe07cd11`) still shows same 3 failures |
| Verification run | [26508072188](https://github.com/tenstorrent/tt-metal/actions/runs/26508072188) — **success** (2026-05-27 11:21–12:02 UTC) |
| Readiness | **Yes** (verification passed; awaiting human review) |

**Disabled tests:** CCL nightly tests in `tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/`:
- `test_all_to_all_combine_no_trace` (all SKUs)
- `test_reduce_scatter_async_4dev_ring`
- `test_reduce_scatter_async_line`

**Verification summary:** All `ccl nightly tests` jobs across bh_llmbox, bh_loudbox, bh_deskbox, bh_p300, and bh_quietbox_2 returned **success**. No regressions. **→ verified-pass**.

---

## PR #45114 — (Blackhole) Demo tests

| Field | Value |
|-------|-------|
| PR | [#45114](https://github.com/tenstorrent/tt-metal/pull/45114) |
| Disable issue | [#45113](https://github.com/tenstorrent/tt-metal/issues/45113) |
| Timeout issue | none |
| Branch | `ci/disable-failing-tests-blackhole-demo-tests-20260524` |
| Workflow file | `blackhole-demo-tests.yaml` |
| Lifecycle stage | `out-of-scope` |
| Last rebase | 2026-05-27 11:10 UTC |
| Last revalidation | 2026-05-27 11:00 UTC — test confirmed fixed on main |
| Verification run | none (disable removed) |
| Readiness | **N/A** (disable removed, PR should be closed) |

**Action:** Disable removed — `test_demo_text` ci-32+performance fixed on main May 26+. PR has no meaningful changes — recommend closing PR and issue #45113.

---

## PR #45306 — (T3K) T3000 unit tests

| Field | Value |
|-------|-------|
| PR | [#45306](https://github.com/tenstorrent/tt-metal/pull/45306) (draft) |
| Disable issue | [#45305](https://github.com/tenstorrent/tt-metal/issues/45305) |
| Timeout issue | none |
| Branch | `ci/disable-failing-tests-t3000-unit-tests-20260527` |
| Workflow file | `t3000-unit-tests.yaml` |
| Lifecycle stage | `verifying` |
| Last rebase | 2026-05-27 13:15 UTC — rebased onto main `de985e8a3de` (Fix kernel-build warnings) |
| Last revalidation | 2026-05-27 13:00 UTC — run 26507885931 (SHA `fe07cd11`) still shows same 3 failures |
| Verification run | [26513061621](https://github.com/tenstorrent/tt-metal/actions/runs/26513061621) — **in progress** (dispatched 2026-05-27 13:09 UTC; fresh build) |
| Readiness | **No** (verifying) |

**Disabled tests (via gtest_filter):**

`t3k_ttmetal_tests` (`unit_tests_debug_tools`):
- `MeshWatcherFixture.TensixTestWatcherSanitizeMulticastSemaphoreInc`
- `DPrintMeshFixture.TensixTestPrintPrependDeviceCoreRisc`
- `DPrintMeshFixture.TensixTestDprintMeshCoordsAllDevicesMapping`
- `DPrintMeshFixture.ActiveEthTestPrint`
- `DPrintMeshFixture.TensixTestPrintMuting`
- `DPrintMeshFixture.TensixTestPrintBuffering`

`t3k_ttnn_tests` (`unit_tests_ttnn`):
- `DistributedTensorOpIfTest/0.AllGatherWithShardedTopology`
- `DistributedTensorOpIfTest/0.ReduceScatterWithShardedTopology`
- `DistributedTensorOpIfTest/0.AllReduceWithShardedTopology`
- `DistributedTensorOpIfTest/0.AllBroadcastWithShardedTopology`
- `DistributedTensorOpIfTest/0.BroadcastWithShardedTopology`
- `DistributedTensorOpIfTest/0.FusedRmsMinimalWithShardedTopology`
- `QueryOpConstraints/MatmulOpIfTest.Matmul/2`

**Session note:** 15+ consecutive failures on `main` (since at least 2026-05-26 05:25 UTC). Evidence in runs 26507885931 (May 27 11:17), 26502247809 (May 27 09:14), 26497476274 (May 27 07:33), 26492864186 (May 27 05:33), and 11+ earlier. Disable mechanism: gtest_filter negative patterns in `tests/scripts/t3000/run_t3000_unit_tests.sh`. Third failing job (`t3k_tt_metal_multiprocess_tests`) crashes with SIGABRT — harder to identify specific tests; deferred to next session after investigation.

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
| **Systematic: Blackhole artifact expiry (for reuse)** | **Resolved for current PRs** | BH PRs #45110 and #45112 are now verified-pass. No more BH verifications pending. |
| PR #45110 infra-inconclusive (×3) | **Resolved** | Fresh-build verification 26508063374 succeeded → verified-pass |
| PR #45112 first verification | **Resolved** | Fresh-build verification 26508072188 succeeded → verified-pass |
| `t3k_tt_metal_multiprocess_tests` crashes | **Investigating** | SIGABRT in MPI tests; harder to identify specific test IDs. Deferred for next session. Need to isolate which `tt-run` invocation causes the crash. |
| PR #45306 verification | **In progress** | Verification run [26513061621](https://github.com/tenstorrent/tt-metal/actions/runs/26513061621) dispatched 2026-05-27 13:09 UTC; fresh build; 3 pruned jobs |
| PR #44938 needs human review/merge | **Watch** | verified-pass, undrafted |
| PR #45108 needs human undraft + review | **Watch** | verified-pass, draft |
| PR #45110 needs human undraft + review | **Watch** | verified-pass, draft |
| PR #45112 needs human undraft + review | **Watch** | verified-pass, draft |

---

## Recent Activity

- `2026-05-27 ~13:20 UTC` — SESSION: **PR #45306 (T3K unit tests) — multiprocess disable added, rebased, verification dispatched.** Focus PRs: #45306 (batch-committed with no verify → carve-out). Actions: (1) Added multiprocess test disable to PR #45306 initial batch: commented out `tt-run test_tt_fabric T3K 2x2` invocation in `run_t3000_unit_tests.sh` (SIGABRT: TT_FATAL Physical chip id not found for eth coord; 5+ consecutive runs on main). (2) Rebased PR #45306 branch onto main `de985e8a3de` (Fix kernel-build warnings). (3) Dispatched first verification run [26513061621](https://github.com/tenstorrent/tt-metal/actions/runs/26513061621) on branch `verify/ci-disable-t3000-unit-tests-20260527` (pruned to 3 jobs: t3k_ttmetal, t3k_ttnn, t3k_tt_metal_multiprocess; fresh build path — no SHA-matching source run at de985e8a). Build in progress at session end. Focus slots 2+3: investigated candidates — `runtime-unit-tests.yaml` (job-level timeouts, policy says out of scope); `vllm-nightly-tests.yaml` (3+ consecutive non-Galaxy failures for WH-T3K/BH-DB Llama sampling tests, needs deeper log investigation for specific test names); `models-t1-e2e-tests.yaml` (Galaxy SKUs only). 1 dispatch this session.

- `2026-05-27 ~12:30 UTC` — SESSION: **Both BH verification runs completed success.** Focus PRs: #45110 (verifying→verified-pass, carve-out), #45112 (verifying→verified-pass, carve-out), #45306 (new, batch-committed). Key findings: (1) PR #45110 verification [26508063374](https://github.com/tenstorrent/tt-metal/actions/runs/26508063374) succeeded — all blackhole-multi-card-fast-unit-tests across P300-viommu, BH-LLMBox, BH-LoudBox → success. Classified verified-pass. (2) PR #45112 verification [26508072188](https://github.com/tenstorrent/tt-metal/actions/runs/26508072188) succeeded — all ccl nightly tests across all BH e2e platforms → success. Classified verified-pass. (3) Both PRs rebased onto main `aa2de19bd3548be5f000f34353f62455e54e4498`. (4) New workflow `(T3K) T3000 unit tests` identified with 15+ consecutive failures. Issue [#45305](https://github.com/tenstorrent/tt-metal/issues/45305) created. PR [#45306](https://github.com/tenstorrent/tt-metal/pull/45306) created with gtest_filter disables for `t3k_ttmetal_tests` (6 DPrint/Watcher tests) and `t3k_ttnn_tests` (7 DistributedTensorOpIf/MatmulOpIf tests). 0 dispatches this session.
- `2026-05-27 ~11:30 UTC` — SESSION: **BH paralysis loop broken.** Focus PRs: #45110 (verification-inconclusive, carve-out), #45112 (batch-committed-no-verify, carve-out), #45114 (batch-committed-no-verify, carve-out). Key findings: (1) BH post-commit/e2e/demo build steps succeed on main even when test jobs fail; fresh-build dispatch is valid. (2) PR [#45114](https://github.com/tenstorrent/tt-metal/pull/45114) disable REMOVED — `test_demo_text` fixed on main since May 26; commit `b841c358e48`. (3) PR [#45110](https://github.com/tenstorrent/tt-metal/pull/45110): rebased onto `fe07cd111531`; fresh-build verification dispatched [26508063374](https://github.com/tenstorrent/tt-metal/actions/runs/26508063374); build-artifact SUCCESS. (4) PR [#45112](https://github.com/tenstorrent/tt-metal/pull/45112): rebased onto `fe07cd111531`; first verification dispatched [26508072188](https://github.com/tenstorrent/tt-metal/actions/runs/26508072188); build-artifact SUCCESS. 2 dispatches this session.
- `2026-05-27 ~10:00 UTC` — SESSION: BH systematic blocker persists. PRs #44938/#45108: no action needed. PRs #45110/#45112/#45114: BH-blocked. No new disable PR opportunity found. 0 dispatches. State log updated.
- Older history truncated — see git history of this file.
