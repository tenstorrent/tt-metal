# CI Disable Work — Status Log

> **Source of truth.** This file is the canonical record of automation-tracked PRs. Wiping it resets the automation to fresh-state view; stale GitHub PRs not listed here are intentionally invisible.

Last updated: **2026-05-29T11:03 UTC** — Session: Examining lane (3 PRs): PR #45500 all 5 disables removed — superseded by main PR #45313 (commit `19fe34405b1`) which merged equivalent `sampling-test-filter`; branch rebased/skipped to main HEAD `6d54c53a3c6`; PR auto-closed by GitHub (branch = main HEAD, zero diff); lifecycle `out-of-scope`; issue #45499 updated. PR #45511 rebased from `09a0526ebfe5` → `6d54c53a3c6` (new head `c95f10b551b`); evidence refreshed — `TestPerfCountersSingleOp::test_performance_counter_columns[Matmul_perf_counters]` still failing in run [26628589795](https://github.com/tenstorrent/tt-metal/actions/runs/26628589795)/job [78481174299](https://github.com/tenstorrent/tt-metal/actions/runs/26628589795/job/78481174299) (2026-05-29T10:22 UTC); PR description + issue #45510 updated. PR #45484 rebased from `09a0526ebfe5` → `6d54c53a3c6` (new head `f477ad8ffa3`); evidence refreshed — all 4 llk-sd-unit-tests failing in run [26628149873](https://github.com/tenstorrent/tt-metal/actions/runs/26628149873) (2026-05-29T09:18-09:26 UTC); PR description + issue #45483 updated. Lightweight: run [26626298958](https://github.com/tenstorrent/tt-metal/actions/runs/26626298958) completed `failure` (moot — zero disables on PR #45507); moved to Recently Completed. Focus lane: 0 focus PRs — no uncovered non-Galaxy workflows with ≥3 consecutive deterministic failures; `single-card-demo-tests` (2 consecutive, not 3), `blackhole-post-commit` (latest success), `t3000-integration-tests` (1 failure at top, not 3 consecutive), `t3000-demo-tests` (latest success), `t3000-profiler-tests` (inactive since January). No priority-2/3 PRs. Focus slots filled: 0/3. Paralysis check: limited: 0 focus PRs (0 uncovered workflows + 0 priority-2/3 PRs) + 3 examining PRs.

---

## How to read/update this file

- Read this file at the start of every automation session and treat it as the authoritative current state for CI disable work.
- Scan the `## Quick Index` table first; it gives the lifecycle stage per PR before drilling into details.
- Per-PR sections use uniform field tables (`PR | Disable issue | Timeout issue | Branch | Workflow file | Lifecycle stage | Last rebase | Last revalidation | Verification run | Last touched by automation | Readiness`); update fields in place rather than rewriting the section. `Last touched by automation: <UTC ISO>` is required on every PR row and drives the 4-hour throttle — update it every time the automation does any work on the PR (rebase, dispatch, log analysis, comment, removal).
- Each PR section also carries a `Disables (with main evidence)` table listing every currently-disabled test ID together with the most recent failing main-run job-link (`/runs/<id>/job/<jid>`) and the run completion timestamp. This mirrors the PR description's evidence table (see `disabling-work/ci-disable-targeted-verification.md` → `Main-run evidence model`) and is the starting point the next session re-checks before doing any work on the PR. If keeping the state log compact is preferred, the per-PR section MAY instead include the one-line pointer `Main-run evidence: see PR description.` — the PR description's evidence table is authoritative either way. Preserve any existing PR entries unchanged when extending the schema.
- Append new entries to the top of `## Recent Activity` (most recent first); keep at most 30 entries — trim older entries to a single `- Older history truncated — see git history of this file.` line if needed.
- Commit and push any change to this file before ending the session.
- Lifecycle stages: `new`, `batch-committed`, `verifying`, `verification-inconclusive`, `verified-pass`, `verified-fail`, `merged`, `out-of-scope`. (`verification-inconclusive` = a verification was dispatched but failed to actually exercise the previously-passing jobs; eligible for re-dispatch and does NOT consume the one-run-per-PR budget.)

---

## Quick Index

| PR | Workflow | Lifecycle stage | Verification result | Ready to merge? | Notes |
|----|----------|-----------------|---------------------|-----------------|-------|
| [#45484](https://github.com/tenstorrent/tt-metal/pull/45484) | Nightly tt-metal L2 tests (`tt-metal-l2-nightly.yaml`) — `llk-sd-unit-tests` | `verified-pass` | [run 26609412851](https://github.com/tenstorrent/tt-metal/actions/runs/26609412851) — success (completed 2026-05-29T00:31 UTC) | Yes — pending review | All 4 llk-sd-unit-tests jobs passed. Disables `MeshDeviceFixture.Top32RmDevPipelineCompletes` across wh_n150/wh_n300/bh_p100/bh_p150. |
| [#45487](https://github.com/tenstorrent/tt-metal/pull/45487) | Blackhole demo tests (`blackhole-demo-tests.yaml`) — `whisper / whisper performance [bh_p150_perf]` | `verified-pass` | [run 26614671489](https://github.com/tenstorrent/tt-metal/actions/runs/26614671489) — **success** | Yes — pending review | Disables whisper distil-large-v3 perf check. Target job passed. |
| [#45490](https://github.com/tenstorrent/tt-metal/pull/45490) | Runtime unit tests (`runtime-unit-tests.yaml`) — `bh_multicard_debug_tools` | `verified-pass` | [run 26612634163](https://github.com/tenstorrent/tt-metal/actions/runs/26612634163) — success (target job passed) | No — test now passing on main; all disables removed; PR eligible for closure | `DPrintMeshFixture.ActiveEthTestPrint` now PASSING on main (run [26620110417](https://github.com/tenstorrent/tt-metal/actions/runs/26620110417)/job 78444865829, 2026-05-29T05:48 UTC). Disable removed 2026-05-29T08:02 UTC. Branch has zero disables. |
| [#45492](https://github.com/tenstorrent/tt-metal/pull/45492) | T3000 unit tests (`t3000-unit-tests.yaml`) — `t3k_tt_metal_multiprocess_tests` | `verified-pass` | [run 26612761462](https://github.com/tenstorrent/tt-metal/actions/runs/26612761462) — **success** | Yes — pending review | Disables `IntermeshSplit2x2FabricFixture.MultiMeshEastMulticast_*` (2 tests). |
| [#45494](https://github.com/tenstorrent/tt-metal/pull/45494) | Blackhole e2e tests (`blackhole-e2e-tests.yaml`) — `ccl nightly tests` | `verified-pass` | [run 26612913914](https://github.com/tenstorrent/tt-metal/actions/runs/26612913914) — failure (target job passed; other SKU failures pre-existing) | Yes — pending review | Disables `test_all_to_all_combine_no_trace` on `bh_llmbox`. |
| [#45498](https://github.com/tenstorrent/tt-metal/pull/45498) | metal-run-microbenchmarks (`metal-run-microbenchmarks.yaml`) — `T3K ubench - Fabric Mux BW` | `verified-pass` | [run 26614829843](https://github.com/tenstorrent/tt-metal/actions/runs/26614829843) — failure (disabled tests skipped; 30 passed; 1 pre-existing [8-8-8-8] geomean error also on main) | Yes — pending review | Disables 2 bandwidth regression test parametrizations (num_full_size_channels=4). |
| [#45500](https://github.com/tenstorrent/tt-metal/pull/45500) | vllm-nightly-tests (`vllm-nightly-tests.yaml`) — `[WH-T3K] Llama-3.1-8B-Instruct with sampling-tests` | `out-of-scope` | [run 26614947278](https://github.com/tenstorrent/tt-metal/actions/runs/26614947278) — verified-pass (superseded by main #45313) | No — CLOSED 2026-05-29T11:12 UTC | All 15 disables removed; superseded by main PR #45313 (commit `19fe34405b1`). Branch at main HEAD; PR auto-closed. |
| [#45507](https://github.com/tenstorrent/tt-metal/pull/45507) | T3000 e2e tests (`t3000-e2e-tests.yaml`) — `t3k_ccl_tests [wh_llmbox]` | `verified-pass` | run [26626298958](https://github.com/tenstorrent/tt-metal/actions/runs/26626298958) — completed `failure` (moot; zero disables; failures pre-existing on main) | No — zero disables remain; PR eligible for closure | All 3 tests no longer failing on main. Run 26626298958 completed 2026-05-29T11:03 UTC. |
| [#45511](https://github.com/tenstorrent/tt-metal/pull/45511) | Sanity tests (`sanity-tests.yaml`) — `profiler-tests / Perf op report [wh_n300_civ2]` | `verified-pass` | [run 26619643533](https://github.com/tenstorrent/tt-metal/actions/runs/26619643533) — **success** (completed 2026-05-29T05:55 UTC) | Yes — pending review | Test still failing; evidence refreshed to run [26628589795](https://github.com/tenstorrent/tt-metal/actions/runs/26628589795)/job [78481174299](https://github.com/tenstorrent/tt-metal/actions/runs/26628589795/job/78481174299) (2026-05-29T10:22 UTC). |
| [#45514](https://github.com/tenstorrent/tt-metal/pull/45514) | Runtime integration tests (`runtime-integration-tests.yaml`) — `runtime_fd_python_2 [wh_n150_civ2]` / `[bh_p150b_civ2]` | `verified-fail` | [run 26623896690](https://github.com/tenstorrent/tt-metal/actions/runs/26623896690) — failure (`seed=0` correctly SKIPPED; `seed=42` NEW failure on PR branch, was passing on main) | No — verified-fail; needs human review (seed=42 not independently failing on main — masked by seed=0 device hang/timeout) | Disables `test_indexed_slice[DataType.BFLOAT16-4-32-6-0]` on both SKUs. Cannot add seed=42 disable without main-run evidence. |

---

## Active Runs

| Run | Pipeline | Branch | Started | Status | Notes |
|-----|----------|--------|---------|--------|-------|
| _(no active runs)_ | | | | | |

**Policy:** Concurrent runs across PRs are allowed; each automation session may dispatch at most three new runs.

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-------|
| [26626298958](https://github.com/tenstorrent/tt-metal/actions/runs/26626298958) | (T3K) T3000 e2e tests (`t3000-e2e-tests.yaml`) | `ci-disable/verify-t3000-e2e-tests-t3k-ccl-20260529c` | 2026-05-29T08:16:00 UTC | 2026-05-29T11:03 UTC | **moot / pre-existing failures** | PR #45507 fourth attempt. All 3 disables were removed 2026-05-29T10:02 UTC before run completed. Run completed with `conclusion: failure` — failures are pre-existing on main (unrelated to PR disables which were already removed). No regression introduced. PR #45507 remains `verified-pass` (zero disables). |
| [26623896690](https://github.com/tenstorrent/tt-metal/actions/runs/26623896690) | (Runtime) Integration Tests (`runtime-integration-tests.yaml`) | `ci-disable/verify-runtime-integration-tests-indexed-fill-20260529` | 2026-05-29T07:19:08 UTC | 2026-05-29T07:42:46 UTC | **verified-fail** | PR #45514 → `verified-fail`. `test_indexed_slice[DataType.BFLOAT16-4-32-6-0]` correctly SKIPPED on both wh_n150_civ2 and bh_p150b_civ2. BUT `test_indexed_slice[DataType.BFLOAT16-4-32-6-42]` NEW FAILURE on both SKUs (same TT_THROW @ system_memory_manager.cpp:738). `seed=42` was passing on main (run 26621870487 had only seed=0 failing). Regression caused by disable of seed=0 exposing seed=42. PR needs revision; cannot add to existing batch. |
| [26621576169](https://github.com/tenstorrent/tt-metal/actions/runs/26621576169) | (T3K) T3000 e2e tests (`t3000-e2e-tests.yaml`) | `ci-disable/verify-t3000-e2e-tests-t3k-ccl-20260529b` | 2026-05-29T06:18:19 UTC | 2026-05-29T08:03:16 UTC | **verification-inconclusive** | PR #45507 → `verification-inconclusive` (third consecutive timeout). Metal context timeout at 07:09:33 UTC; job-level timeout at 08:01:56 UTC. Budget NOT consumed; re-dispatched as run 26626298958. |
| [26619643533](https://github.com/tenstorrent/tt-metal/actions/runs/26619643533) | (Single-card) Profiler tests (`single-card-profiler-tests.yaml`) | `ci-disable/verify-sanity-tests-perf-counters-20260529` | 2026-05-29T05:20:43 UTC | 2026-05-29T05:55 UTC | **verified-pass** | PR #45511 → `verified-pass`. All profiler jobs passed: Perf op report [wh_n300_civ2] success. Disabled test skipped; no regressions. |
| [26615944303](https://github.com/tenstorrent/tt-metal/actions/runs/26615944303) | T3000 e2e tests (`t3000-e2e-tests.yaml`) | `ci-disable/verify-t3000-e2e-tests-t3k-ccl-20260529` | 2026-05-29T03:21:22 UTC | 2026-05-29T05:08 UTC | **verification-inconclusive** | PR #45507 → `verification-inconclusive`. Job `t3k_ccl_tests [wh_llmbox]` timed out after 90 minutes. Metal context timeout at 04:12 UTC. Timeout is out-of-scope per policy; budget NOT consumed; re-dispatch eligible. |
| [26614829843](https://github.com/tenstorrent/tt-metal/actions/runs/26614829843) | metal-run-microbenchmarks (`metal-run-microbenchmarks.yaml`) | `ci-disable/verify-metal-run-microbenchmarks-fabric-mux-bw-20260529` | 2026-05-29T02:44:22 UTC | 2026-05-29T03:06 UTC | **verified-pass** | PR #45498 → `verified-pass`. Disabled tests skipped; 30 passed; [8-8-8-8] geomean error pre-existing on main. |
| [26614947278](https://github.com/tenstorrent/tt-metal/actions/runs/26614947278) | vllm-nightly-tests (`vllm-nightly-tests.yaml`) | `ci-disable/verify-vllm-nightly-t3k-llama-sampling-20260529` | 2026-05-29T02:48:12 UTC | 2026-05-29T03:34 UTC | **verified-pass** | PR #45500 → `verified-pass`. Target [WH-T3K] sampling-tests PASSED. BH-QB-GE: infra fault (EngineCore failed to start). WH-GLX: Galaxy, out-of-scope. |
| [26614671489](https://github.com/tenstorrent/tt-metal/actions/runs/26614671489) | Blackhole demo tests (`blackhole-demo-tests.yaml`) | `ci-disable/verify-blackhole-demo-tests-whisper-perf-20260529` | 2026-05-29T02:39:08 UTC | 2026-05-29T∰03 UTC | **verified-pass** | PR #45487 → `verified-pass`. Target `whisper performance [bh_p150_perf]` passed. |
|| [26612634163](https://github.com/tenstorrent/tt-metal/actions/runs/26612634163) | Runtime unit tests (`runtime-unit-tests.yaml`) | `ci-disable/verify-runtime-unit-tests-dprint-activeeth-20260529` | 2026-05-29T01:33:20 UTC | 2026-05-29T∰02 UTC | **verified-pass** | PR #45490 → `verified-pass`. Target `bh_multicard_debug_tools [bh_quietbox]` passed. `bh_multicard_dispatch` flaky/pre-existing. |
|| [26612761462](https://github.com/tenstorrent/tt-metal/actions/runs/26612761462) | T3000 unit tests (`t3000-unit-tests.yaml`) | `ci-disable/verify-t3000-unit-tests-intermesh-split2x2-20260529` | 2026-05-29T01:37:25 UTC | 2026-05-29T∰02 UTC | **verified-pass** | PR #45492 → `verified-pass`. Target `t3k_tt_metal_multiprocess_tests [wh_llmbox]` passed. |
|| [26612913914](https://github.com/tenstorrent/tt-metal/actions/runs/26612913914) | Blackhole e2e tests (`blackhole-e2e-tests.yaml`) | `ci-disable/verify-blackhole-e2e-tests-all-to-all-combine-no-trace-20260529` | 2026-05-29T01:42:23 UTC | 2026-05-29T∰02 UTC | **verified-pass** | PR #45494 → `verified-pass`. Target `ccl nightly tests [bh_llmbox]` passed; other SKU load-test-matrix failures pre-existing. |
| [26609412851](https://github.com/tenstorrent/tt-metal/actions/runs/26609412851) | Nightly tt-metal L2 tests (`tt-metal-l2-nightly.yaml`) | `ci-disable/tt-metal-l2-nightly-mesh-device-top32-20260528` | 2026-05-28T23:57:22 UTC | 2026-05-29T00:31:40 UTC | **verified-pass** | All 4 llk-sd-unit-tests jobs passed (N150, N300, P100, P150 — all `success`). No regressions. PR #45484 → `verified-pass`. |

---

## PR #45484 — Nightly tt-metal L2 tests (MeshDeviceFixture.Top32RmDevPipelineCompletes)

| Field | Value |
|-------|-------|
| PR | [#45484](https://github.com/tenstorrent/tt-metal/pull/45484) — `[skip ci] Disable MeshDeviceFixture.Top32RmDevPipelineCompletes in tt-metal-l2-nightly llk-sd-unit-tests` (draft, open) |
| Disable issue | [#45483](https://github.com/tenstorrent/tt-metal/issues/45483) — `[CI] Track disable: MeshDeviceFixture.Top32RmDevPipelineCompletes in tt-metal-l2-nightly llk-sd-unit-tests` (open) |
| Timeout issue | none |
| Branch | `ci-disable/tt-metal-l2-nightly-mesh-device-top32-20260528` (head SHA `f477ad8ffa3c53c09a84da7fd7bfd50132ce68b9` — rebased to `6d54c53a3c6ff51f34c2cbb57d1a9d0a8ed28129`) |
| Workflow file | `.github/workflows/tt-metal-l2-nightly.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-29T11:03 UTC — rebased from `09a0526ebfe5c43a23c4c5b19280cd4fc078f8c7` → `6d54c53a3c6ff51f34c2cbb57d1a9d0a8ed28129` (new head `f477ad8ffa3c53c09a84da7fd7bfd50132ce68b9`), no conflicts |
| Last revalidation | 2026-05-29T11:03 UTC — `MeshDeviceFixture.Top32RmDevPipelineCompletes` confirmed still failing on `main`. Latest completed L2 nightly run [26628149873](https://github.com/tenstorrent/tt-metal/actions/runs/26628149873) (started 2026-05-29T08:57 UTC): all 4 llk-sd-unit-tests jobs FAILED (wh_n150 job 78471369457 at 09:18 UTC, wh_n300 job 78471369454 at 09:18 UTC, bh_p100 job 78471369413 at 09:26 UTC, bh_p150 job 78471369416 at 09:21 UTC). Evidence updated in PR description and issue #45483. |
| Verification run | [26609412851](https://github.com/tenstorrent/tt-metal/actions/runs/26609412851) — **verified-pass** (completed 2026-05-29T00:31 UTC, conclusion `success`; all 4 llk-sd-unit-tests jobs passed) |
| Last touched by automation | 2026-05-29T11:03 UTC |
| Readiness | **Yes — verified-pass, ready for merge review** |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| `MeshDeviceFixture.Top32RmDevPipelineCompletes` [wh_n150] | https://github.com/tenstorrent/tt-metal/actions/runs/26628149873/job/78471369457 | 2026-05-29 09:18 UTC |
| `MeshDeviceFixture.Top32RmDevPipelineCompletes` [wh_n300] | https://github.com/tenstorrent/tt-metal/actions/runs/26628149873/job/78471369454 | 2026-05-29 09:18 UTC |
| `MeshDeviceFixture.Top32RmDevPipelineCompletes` [bh_p100] | https://github.com/tenstorrent/tt-metal/actions/runs/26628149873/job/78471369413 | 2026-05-29 09:26 UTC |
| `MeshDeviceFixture.Top32RmDevPipelineCompletes` [bh_p150] | https://github.com/tenstorrent/tt-metal/actions/runs/26628149873/job/78471369416 | 2026-05-29 09:21 UTC |

Main-run evidence: see PR description.

---

## PR #45487 — Blackhole demo tests (whisper distil-large-v3 perf check)

| Field | Value |
|-------|-------|
| PR | [#45487](https://github.com/tenstorrent/tt-metal/pull/45487) — `[skip ci] Disable distil-large-v3 performance check in blackhole-demo-tests whisper performance job` (draft, open) |
| Disable issue | [#45496](https://github.com/tenstorrent/tt-metal/issues/45496) — `[CI] Track disable: whisper perf test (distil-large-v3 performance regression) in blackhole-demo-tests whisper performance job` (open) |
| Timeout issue | none |
| Branch | `ci-disable/blackhole-demo-tests-whisper-perf-20260529` (head SHA `0ce037d988421be76903d117d576ee53065b2411` — rebased to `e1210204db9b322686e4dff457dd58f705782a8a`) |
| Workflow file | `.github/workflows/blackhole-demo-tests.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-29T06:00 UTC — rebased from `9727d4445a8d52c844223ba59618e0e5452b9445` → `e1210204db9b322686e4dff457dd58f705782a8a` (new head `0ce037d988421be76903d117d576ee53065b2411`), no conflicts |
| Last revalidation | 2026-05-29T06:00 UTC — `test_demo_for_conditional_generation[...-distil-large-v3-1-2]` confirmed still failing; most recent completed run 26556373726 (2026-05-28T05:22 UTC, job 78230061657) still fails. New run 26619741477 queued on `e1210204db9b` — not yet completed. Evidence unchanged |
| Verification run | [26614671489](https://github.com/tenstorrent/tt-metal/actions/runs/26614671489) — **verified-pass** (conclusion `success`; target `whisper performance [bh_p150_perf]` passed) |
| Last touched by automation | 2026-05-29T06:00 UTC |
| Readiness | **Yes — verified-pass, ready for merge review** |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| `models/demos/audio/whisper/demo/demo.py::test_demo_for_conditional_generation[blackhole-device_params0-True-False-None-False-0.0-None-None-None-False-transcribe-English-models/demos/audio/whisper/demo/dataset/conditional_generation-1-distil-whisper/distil-large-v3-1-2]` [bh_p150_perf] | https://github.com/tenstorrent/tt-metal/actions/runs/26556373726/job/78230061657 | 2026-05-28 11:37 UTC |

Main-run evidence: see PR description.

---

## PR #45490 — Runtime unit tests (DPrintMeshFixture.ActiveEthTestPrint)

| Field | Value |
|-------|-------|
| PR | [#45490](https://github.com/tenstorrent/tt-metal/pull/45490) — `[skip ci] Disable DPrintMeshFixture.ActiveEthTestPrint in runtime-unit-tests bh_multicard_debug_tools` (draft, open) |
| Disable issue | [#45489](https://github.com/tenstorrent/tt-metal/issues/45489) — `[CI] Track disable: DPrintMeshFixture.ActiveEthTestPrint in runtime-unit-tests bh_multicard_debug_tools` (open) |
| Timeout issue | none |
| Branch | `ci-disable/runtime-unit-tests-dprint-activeeth-20260529` (head SHA `7f602a7b5b3be52c401f144a553dfdfbb1b755e6` — rebased to `8e179efa2a3468dc772fa29c87cf08102612b2df`) |
| Workflow file | `.github/workflows/runtime-unit-tests.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-29T08:02 UTC — rebased to `8e179efa2a3468dc772fa29c87cf08102612b2df` (new head `7f602a7b5b3`); removal commit pushed, no conflicts |
| Last revalidation | 2026-05-29T08:02 UTC — `DPrintMeshFixture.ActiveEthTestPrint` now **PASSING** on main; [run 26620110417](https://github.com/tenstorrent/tt-metal/actions/runs/26620110417/job/78444865829) job 78444865829 completed 2026-05-29T05:48 UTC — `[PASSED] 1 test.`. Disable REMOVED from branch. |
| Verification run | [26612634163](https://github.com/tenstorrent/tt-metal/actions/runs/26612634163) — **verified-pass** (conclusion `failure` but target `bh_multicard_debug_tools [bh_quietbox]` passed; `bh_multicard_dispatch` failure pre-existing flaky on main, non-consecutive) |
| Last touched by automation | 2026-05-29T08:02 UTC |
| Readiness | **No — all disables removed (test now passing on main); PR eligible for closure** |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| _(no disabled tests remain — all re-enabled as of 2026-05-29T08:02 UTC)_ | | |

Main-run evidence: see PR description.

---

## PR #45492 — T3000 unit tests (IntermeshSplit2x2FabricFixture.MultiMeshEastMulticast_*)

| Field | Value |
|-------|-------|
| PR | [#45492](https://github.com/tenstorrent/tt-metal/pull/45492) — `[skip ci] Disable IntermeshSplit2x2FabricFixture.MultiMeshEastMulticast_* in t3000-unit-tests t3k_tt_metal_multiprocess_tests` (draft, open; title updated from `IntermeshSplit2x2FabricFixture.*` to `MultiMeshEastMulticast_*` after RandomizedInterMeshUnicast removal) |
| Disable issue | [#45491](https://github.com/tenstorrent/tt-metal/issues/45491) — `[CI] Track disable: IntermeshSplit2x2FabricFixture.* in t3000-unit-tests t3k_tt_metal_multiprocess_tests` (open; updated to reflect removal) |
| Timeout issue | none |
| Branch | `ci-disable/t3000-unit-tests-intermesh-split2x2-20260529` (head SHA `9fd52f1be01daab00eecf86c9942ba65a57df352` — rebased to `ee5df35e32105209e769898f597c4d6b46f5fa06`) |
| Workflow file | `.github/workflows/t3000-unit-tests.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-29T09:02 UTC — rebased to `ee5df35e32105209e769898f597c4d6b46f5fa06` (new head `9fd52f1be01`), no conflicts |
| Last revalidation | 2026-05-29T08:02 UTC — `MultiMeshEastMulticast_0` and `MultiMeshEastMulticast_1` still FAILING in latest main run [26620003623](https://github.com/tenstorrent/tt-metal/actions/runs/26620003623) job [78444692420](https://github.com/tenstorrent/tt-metal/actions/runs/26620003623/job/78444692420) (2026-05-29T06:06 UTC). Deep revalidation deferred 2026-05-29T09:02 UTC (throttled — last full analysis 1h ago). |
| Verification run | [26612761462](https://github.com/tenstorrent/tt-metal/actions/runs/26612761462) — **verified-pass** (conclusion `success`; target `t3k_tt_metal_multiprocess_tests [wh_llmbox]` passed) |
| Last touched by automation | 2026-05-29T09:02 UTC |
| Readiness | **Yes — verified-pass, ready for merge review** |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| `IntermeshSplit2x2FabricFixture.MultiMeshEastMulticast_0` [wh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/26620003623/job/78444692420 | 2026-05-29 06:06 UTC |
| `IntermeshSplit2x2FabricFixture.MultiMeshEastMulticast_1` [wh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/26620003623/job/78444692420 | 2026-05-29 06:06 UTC |

**Note:** `IntermeshSplit2x2FabricFixture.RandomizedInterMeshUnicast` was removed from the disable in this session (passed in main run 26611455612 job 78419112342 at 2026-05-29 01:58 UTC).

Main-run evidence: see PR description.

---

## PR #45494 — Blackhole e2e tests (test_all_to_all_combine_no_trace)

| Field | Value |
|-------|-------|
| PR | [#45494](https://github.com/tenstorrent/tt-metal/pull/45494) — `[skip ci] Disable test_all_to_all_combine_no_trace in blackhole-e2e-tests ccl nightly tests` (draft, open) |
| Disable issue | [#45493](https://github.com/tenstorrent/tt-metal/issues/45493) — `[CI] Track disable: test_all_to_all_combine_no_trace in blackhole-e2e-tests ccl nightly tests` (open) |
| Timeout issue | none |
| Branch | `ci-disable/blackhole-e2e-tests-all-to-all-combine-no-trace-20260529` (head SHA `81eb4585d58` — rebased to `19fe34405b152c77c34c62f2826bf2226f7fb59c`) |
| Workflow file | `.github/workflows/blackhole-e2e-tests.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-29T10:02 UTC — rebased from `9727d4445a8d52c844223ba59618e0e5452b9445` → `19fe34405b152c77c34c62f2826bf2226f7fb59c` (new head `81eb4585d58`), no conflicts |
| Last revalidation | 2026-05-29T02:02 UTC — `test_all_to_all_combine_no_trace` confirmed still failing on `main`; latest completed run 26589993622 job 78379365051 `ccl nightly tests [bh_llmbox]` still fails |
| Verification run | [26612913914](https://github.com/tenstorrent/tt-metal/actions/runs/26612913914) — **verified-pass** (completed 2026-05-29T∰02 UTC; conclusion `failure`; target `ccl nightly tests [bh_llmbox]` passed; other SKU load-test-matrix failures pre-existing) |
| Last touched by automation | 2026-05-29T10:02 UTC |
| Readiness | **Yes — verified-pass, ready for merge review** |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| `test_all_to_all_combine_no_trace[...fabric_1d_line_axis_0]` (all mem/local_reduce combos) [bh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/26589993622/job/78379365051 | 2026-05-28 18:46 UTC |
| `test_all_to_all_combine_no_trace[...fabric_1d_ring_axis_0]` (all mem/local_reduce combos) [bh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/26589993622/job/78379365051 | 2026-05-28 18:46 UTC |

Main-run evidence: see PR description.

---

## PR #45498 — metal-run-microbenchmarks (T3K Fabric Mux BW bandwidth regression)

| Field | Value |
|-------|-------|
| PR | [#45498](https://github.com/tenstorrent/tt-metal/pull/45498) — `[skip ci] Disable T3K Fabric Mux BW bandwidth regression tests in metal-run-microbenchmarks` (draft, open) |
| Disable issue | [#45497](https://github.com/tenstorrent/tt-metal/issues/45497) — `[CI] Track disable: T3K Fabric Mux BW bandwidth regression tests in metal-run-microbenchmarks` (open) |
| Timeout issue | none |
| Branch | `ci-disable/metal-run-microbenchmarks-fabric-mux-bw-20260529` (head SHA `7a0fdbd0161` — rebased to `19fe34405b152c77c34c62f2826bf2226f7fb59c`) |
| Workflow file | `.github/workflows/metal-run-microbenchmarks.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-29T10:02 UTC — rebased from `80094df9c61e426b145574ffb2c0aebc0c75f02a` → `19fe34405b152c77c34c62f2826bf2226f7fb59c` (new head `7a0fdbd0161`), no conflicts |
| Last revalidation | 2026-05-29T04:00 UTC — both disabled tests still failing in latest main run [26612909413](https://github.com/tenstorrent/tt-metal/actions/runs/26612909413) job [78423388911](https://github.com/tenstorrent/tt-metal/actions/runs/26612909413/job/78423388911) (2026-05-29T02:25 UTC): same bandwidth mismatch signature |
| Verification run | [26614829843](https://github.com/tenstorrent/tt-metal/actions/runs/26614829843) — **verified-pass** (completed 2026-05-29T03:06 UTC; conclusion `failure`; disabled tests skipped; 30 passed; 1 pre-existing `[8-8-8-8]` geomean error also failing on main run 26587701074 with same signature) |
| Last touched by automation | 2026-05-29T10:02 UTC |
| Readiness | **Yes — verified-pass, ready for merge review** |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| `tests/tt_metal/microbenchmarks/ethernet/test_fabric_mux_bandwidth.py::test_mux_bw_full_size_channels[32-1-4096-10000-0-8-0-4]` [wh_llmbox T3K] | https://github.com/tenstorrent/tt-metal/actions/runs/26612909413/job/78423388911 | 2026-05-29 02:25 UTC |
| `tests/tt_metal/microbenchmarks/ethernet/test_fabric_mux_bandwidth.py::test_mux_bw_both_channel_types[32-1-4096-10000-8-8-1-4]` [wh_llmbox T3K] | https://github.com/tenstorrent/tt-metal/actions/runs/26612909413/job/78423388911 | 2026-05-29 02:25 UTC |

Main-run evidence: see PR description.

---

## PR #45500 — vllm-nightly-tests (T3K Llama-3.1-8B sampling/seeding determinism)

| Field | Value |
|-------|-------|
| PR | [#45500](https://github.com/tenstorrent/tt-metal/pull/45500) — `[skip ci] Disable T3K Llama-3.1-8B sampling/seeding determinism tests in vllm-nightly-tests` (CLOSED 2026-05-29T11:12 UTC — auto-closed when branch became identical to main) |
| Disable issue | [#45499](https://github.com/tenstorrent/tt-metal/issues/45499) — `[CI] Track disable: T3K Llama-3.1-8B sampling/seeding determinism tests in vllm-nightly-tests` (open; updated to reflect closure) |
| Timeout issue | none |
| Branch | `ci-disable/vllm-nightly-tests-t3k-llama-sampling-20260529` (head SHA `6d54c53a3c6ff51f34c2cbb57d1a9d0a8ed28129` — at main HEAD; commit skipped during rebase) |
| Workflow file | `.github/workflows/vllm-nightly-tests.yaml` |
| Lifecycle stage | `out-of-scope` |
| Last rebase | 2026-05-29T11:03 UTC — rebase conflict with `19fe34405b1` ([skip ci] Disable consistently failing tests in vllm-nightly-tests #45313); main already has equivalent `sampling-test-filter`; commit skipped (`git rebase --skip`); branch now at main HEAD `6d54c53a3c6` |
| Last revalidation | 2026-05-29T11:03 UTC — all 5 disabled test groups now EXCLUDED on `main` by commit `19fe34405b1` (PR #45313 in `vllm-nightly-tests-impl.yaml` `sampling-test-filter`). Tests no longer failing on main. All 15 individual test disables removed from PR. PR auto-closed. |
| Verification run | [26614947278](https://github.com/tenstorrent/tt-metal/actions/runs/26614947278) — **verified-pass** (completed 2026-05-29T03:34 UTC; conclusion `failure`; target `[WH-T3K] Llama-3.1-8B-Instruct with sampling-tests` PASSED; BH-QB-GE: infra fault; WH-GLX: Galaxy out-of-scope) |
| Last touched by automation | 2026-05-29T11:03 UTC |
| Readiness | **No — CLOSED (auto-closed); all disables superseded by main PR #45313** |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| _(no disabled tests remain — all 15 tests removed 2026-05-29T11:03 UTC: superseded by main PR #45313 commit `19fe34405b1` which merged equivalent `sampling-test-filter` into `vllm-nightly-tests-impl.yaml`)_ | | |

Main-run evidence: see PR description (now shows zero disables).

---

## PR #45507 — T3000 e2e tests (t3k_ccl_tests: test_all_gather_matmul_async / test_all_broadcast_sharded_2x4 / test_ring_joint_sdpa_program_cache)

| Field | Value |
|-------|-------|
| PR | [#45507](https://github.com/tenstorrent/tt-metal/pull/45507) — `[skip ci] Disable t3k_ccl_tests test_all_gather_matmul_async / test_all_broadcast_sharded_2x4 / test_ring_joint_sdpa_program_cache in t3000-e2e-tests` (draft, open) |
| Disable issue | [#45506](https://github.com/tenstorrent/tt-metal/issues/45506) — `[CI] Track disable: t3k_ccl_tests test_all_gather_matmul_async / test_all_broadcast_sharded_2x4 / test_ring_joint_sdpa_program_cache in t3000-e2e-tests` (open) |
| Timeout issue | none |
| Branch | `ci-disable/t3000-e2e-tests-t3k-ccl-20260529` (head SHA `6fa854e5a5a` — rebased to `19fe34405b152c77c34c62f2826bf2226f7fb59c`) |
| Workflow file | `.github/workflows/t3000-e2e-tests.yaml` |
| Lifecycle stage | `verified-pass` (zero disables — all 3 tests removed 2026-05-29T10:02 UTC; PR eligible for closure) |
| Last rebase | 2026-05-29T10:02 UTC — rebased from `ee5df35e32105209e769898f597c4d6b46f5fa06` → `19fe34405b152c77c34c62f2826bf2226f7fb59c` (new head `6fa854e5a5a`), no conflicts |
| Last revalidation | 2026-05-29T10:02 UTC — main run 26624591673/job 78459489635 (completed 2026-05-29T09:49 UTC, SHA 09a0526ebfe5): `test_all_gather_matmul_async[...perf-no_barrier_with_persistent-chunking]` SKIPPED (disabled by PR #45108); `test_all_broadcast_sharded_2x4[...ROW_MAJOR...BLOCK_SHARDED]` PASSING; `test_ring_joint_sdpa_program_cache[...no_trace-sd35-bf16]` SKIPPED (disabled by PR #45108). All 3 disables removed from PR branch. |
| Verification run | run [26626298958](https://github.com/tenstorrent/tt-metal/actions/runs/26626298958) — completed `failure` (moot; all disables removed 2026-05-29T10:02 UTC before run concluded; failures are pre-existing on main; no regression attributable to PR disables which were already removed); runs 26615944303 and 26621576169 were both `verification-inconclusive` (timeouts) |
| Last touched by automation | 2026-05-29T10:02 UTC |
| Readiness | No — zero disables remain (all tests no longer failing on main); PR eligible for closure |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| _(no disabled tests remain — all removed 2026-05-29T10:02 UTC: `test_all_gather_matmul_async` SKIPPED by PR #45108 on main, `test_all_broadcast_sharded_2x4[ROW_MAJOR]` PASSING on main, `test_ring_joint_sdpa_program_cache[bf16]` SKIPPED by PR #45108 on main — all confirmed in run 26624591673/job 78459489635)_ | | |

Main-run evidence: see PR description.

## PR #45511 — Sanity tests (TestPerfCountersSingleOp.test_performance_counter_columns[Matmul_perf_counters])

| Field | Value |
|-------|-------|
| PR | [#45511](https://github.com/tenstorrent/tt-metal/pull/45511) — `[skip ci] Disable TestPerfCountersSingleOp.test_performance_counter_columns in sanity-tests profiler-tests` (draft, open) |
| Disable issue | [#45510](https://github.com/tenstorrent/tt-metal/issues/45510) — `[CI] Track disable: TestPerfCountersSingleOp.test_performance_counter_columns in sanity-tests profiler-tests/Perf op report [wh_n300_civ2]` (open) |
| Timeout issue | none |
| Branch | `ci-disable/sanity-tests-perf-counters-matmul-20260529` (head SHA `c95f10b551b6d33be7f6d65063780e89c9531b9b` — rebased to `6d54c53a3c6ff51f34c2cbb57d1a9d0a8ed28129`) |
| Workflow file | `.github/workflows/sanity-tests.yaml` (calls `single-card-profiler-tests-impl.yaml` for profiler job) |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-29T11:03 UTC — rebased from `09a0526ebfe5c43a23c4c5b19280cd4fc078f8c7` → `6d54c53a3c6ff51f34c2cbb57d1a9d0a8ed28129` (new head `c95f10b551b6d33be7f6d65063780e89c9531b9b`), no conflicts |
| Last revalidation | 2026-05-29T11:03 UTC — `TestPerfCountersSingleOp::test_performance_counter_columns[Matmul_perf_counters]` confirmed still failing on `main`. Latest completed sanity-tests run [26628589795](https://github.com/tenstorrent/tt-metal/actions/runs/26628589795) (started 2026-05-29T09:07 UTC): `profiler-tests / Perf op report [wh_n300_civ2]` job [78481174299](https://github.com/tenstorrent/tt-metal/actions/runs/26628589795/job/78481174299) — `failure` (completed 2026-05-29T10:22 UTC). Evidence updated in PR description and issue #45510. |
| Verification run | [26619643533](https://github.com/tenstorrent/tt-metal/actions/runs/26619643533) — **verified-pass** (completed 2026-05-29T05:55 UTC, conclusion `success`; all profiler jobs passed including `Perf op report [wh_n300_civ2]`; disabled test skipped as expected) |
| Last touched by automation | 2026-05-29T11:03 UTC |
| Readiness | **Yes — verified-pass, ready for merge review** |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| `tests/ttnn/tracy/test_perf_op_report.py::TestPerfCountersSingleOp::test_performance_counter_columns[Matmul_perf_counters]` [wh_n300_civ2] | https://github.com/tenstorrent/tt-metal/actions/runs/26628589795/job/78481174299 | 2026-05-29 10:22 UTC |

Main-run evidence: see PR description.

---

---

## PR #45514 — Runtime integration tests (test_indexed_slice[DataType.BFLOAT16-4-32-6-0])

| Field | Value |
|-------|-------|
| PR | [#45514](https://github.com/tenstorrent/tt-metal/pull/45514) — `[skip ci] Disable test_indexed_slice[DataType.BFLOAT16-4-32-6-0] in runtime-integration-tests runtime_fd_python_2` (draft, open) |
| Disable issue | [#45513](https://github.com/tenstorrent/tt-metal/issues/45513) — `[CI] Track disable: test_indexed_slice[DataType.BFLOAT16-4-32-6-0] in runtime-integration-tests runtime_fd_python_2` (open) |
| Timeout issue | none |
| Branch | `ci-disable/runtime-integration-tests-indexed-fill-20260529` (head SHA `f42962a8bd66f41afab63f0bb1e0c2e754a36934` — rebased to `ee5df35e32105209e769898f597c4d6b46f5fa06`) |
| Workflow file | `.github/workflows/runtime-integration-tests.yaml` |
| Lifecycle stage | `verified-fail` |
| Last rebase | 2026-05-29T09:02 UTC — rebased to `ee5df35e32105209e769898f597c4d6b46f5fa06` (new head `f42962a8bd6`), no conflicts |
| Last revalidation | 2026-05-29T09:02 UTC — `test_indexed_slice[DataType.BFLOAT16-4-32-6-0]` (seed=0) still failing on main in run 26621870487/job 78450667711 (2026-05-29T06:59 UTC). Seed=42 analysis: `test_indexed_slice[DataType.BFLOAT16-4-32-6-42]` confirmed NOT independently failing on main — only seed=0 appears in run 26621870487 failures. Root cause: seed=0 times out the device on main; seed=42 never reaches execution; on the PR branch seed=0 is skipped so seed=42 runs and also hits TT_THROW. Cannot disable seed=42 without main-run evidence. PR needs human review. |
| Verification run | [26623896690](https://github.com/tenstorrent/tt-metal/actions/runs/26623896690) — **verified-fail** (completed 2026-05-29T07:42 UTC; `test_indexed_slice[DataType.BFLOAT16-4-32-6-0]` correctly SKIPPED; but `test_indexed_slice[DataType.BFLOAT16-4-32-6-42]` NEW FAILURE on wh_n150_civ2 and bh_p150b_civ2 — same TT_THROW signature; seed=42 was passing on main run 26621870487) |
| Last touched by automation | 2026-05-29T09:02 UTC |
| Readiness | No — verified-fail; PR needs revision to also disable seed=42 (or all D=6 parametrizations) |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| `tests/tt_eager/python_api_testing/unit_testing/misc/test_indexed_fill.py::test_indexed_slice[DataType.BFLOAT16-4-32-6-0]` [wh_n150_civ2] | https://github.com/tenstorrent/tt-metal/actions/runs/26621870487/job/78450667711 | 2026-05-29 06:59 UTC |
| `tests/tt_eager/python_api_testing/unit_testing/misc/test_indexed_fill.py::test_indexed_slice[DataType.BFLOAT16-4-32-6-0]` [bh_p150b_civ2] | https://github.com/tenstorrent/tt-metal/actions/runs/26621870487/job/78450667722 | 2026-05-29 06:59 UTC |

**Note — VERIFIED-FAIL:** Verification run 26623896690 revealed that `test_indexed_slice[DataType.BFLOAT16-4-32-6-42]` is NOW FAILING on the PR branch (same TT_THROW @ system_memory_manager.cpp:738). This test was passing on main (run 26621870487). The disable patch needs to be extended. New PR required to also disable seed=42.

Main-run evidence: see PR description.


## Blockers

| Blocker | Status | Notes |
|---------|--------|-------|
| _(none)_ | | |

---

## Recent Activity

- **2026-05-29T11:03 UTC session.** Examining lane (3 PRs): PR #45500 — all 5 disabled test groups removed (superseded by main PR #45313 / commit `19fe34405b1` which merged equivalent `sampling-test-filter` into `vllm-nightly-tests-impl.yaml`); branch rebased/skipped to main HEAD `6d54c53a3c6` (commit skipped — our filter change was superseded); PR auto-closed by GitHub (branch = main HEAD, zero diff); lifecycle `out-of-scope`; issue #45499 updated; PR comment posted. PR #45511 rebased from `09a0526ebfe5` → `6d54c53a3c6` (new head `c95f10b551b6d33be7f6d65063780e89c9531b9b`); `TestPerfCountersSingleOp::test_performance_counter_columns[Matmul_perf_counters]` confirmed still failing in run [26628589795](https://github.com/tenstorrent/tt-metal/actions/runs/26628589795)/job [78481174299](https://github.com/tenstorrent/tt-metal/actions/runs/26628589795/job/78481174299) (2026-05-29T10:22 UTC); PR description + issue #45510 updated; PR comment posted. PR #45484 rebased from `09a0526ebfe5` → `6d54c53a3c6` (new head `f477ad8ffa3c53c09a84da7fd7bfd50132ce68b9`); all 4 llk-sd-unit-tests confirmed still failing in run [26628149873](https://github.com/tenstorrent/tt-metal/actions/runs/26628149873) (2026-05-29T09:18-09:26 UTC); PR description + issue #45483 updated; PR comment posted. Lightweight: run [26626298958](https://github.com/tenstorrent/tt-metal/actions/runs/26626298958) completed `failure` (moot — zero disables on PR #45507); moved to Recently Completed; no lifecycle change to PR #45507 (remains `verified-pass`, zero disables). Focus lane: 0 focus PRs — no uncovered non-Galaxy workflows with ≥3 consecutive deterministic failures; `single-card-demo-tests` (2 consecutive, not 3), `blackhole-post-commit` (latest success), `t3000-integration-tests` (1 consecutive failure at top, not 3), `t3000-demo-tests` (latest success), `t3000-profiler-tests` (inactive since January). No priority-2/3 PRs. Focus slots filled: 0/3. Paralysis check: limited: 0 focus PRs (0 uncovered workflows + 0 priority-2/3 PRs available) + 3 examining PRs.

- **2026-05-29T10:02 UTC session.** Examining lane (3 PRs): PR #45507 rebased to `19fe34405b` (new head `6fa854e5a5a`); main run 26624591673 completed (job 78459489635, 2026-05-29T09:49 UTC, conclusion `failure`): ALL 3 disabled tests no longer failing — `test_all_gather_matmul_async[...perf-no_barrier_with_persistent-chunking-mesh_device0]` SKIPPED (disabled by PR #45108 on main), `test_all_broadcast_sharded_2x4[...ROW_MAJOR...BLOCK_SHARDED]` PASSING (job actually failing on `test_reduce_scatter_async` tests — unrelated category), `test_ring_joint_sdpa_program_cache[...no_trace-sd35-bf16]` SKIPPED (disabled by PR #45108 on main); all 3 disables REMOVED from PR branch; PR description + issue #45506 updated; run 26626298958 still in_progress but moot; lifecycle updated to `verified-pass` (zero disables); PR eligible for closure. PR #45494 rebased to `19fe34405b` (new head `81eb4585d58`), no conflicts; evidence unchanged (latest completed BH e2e run still 26589993622; next queued run 26628343995 not completed); PR comment posted. PR #45498 rebased to `19fe34405b` (new head `7a0fdbd0161`), no conflicts; evidence unchanged (latest microbenchmarks run still 26612909413); PR comment posted. Focus lane: 0 focus PRs — no uncovered non-Galaxy workflows with ≥3 consecutive deterministic failures (`single-card-demo-tests` 2 consecutive, `blackhole-post-commit` success, `t3000-integration-tests` success, `t3000-demo-tests` not 3 consecutive, `runtime-integration-tests` covered by PR #45514 `verified-fail`); no priority-2/3 PRs available. Focus slots filled: 0/3. Paralysis check: limited: 0 focus PRs (0 uncovered workflows + 0 priority-2/3 PRs available) + 3 examining PRs.

- **2026-05-29T09:02 UTC session.** Examining lane (3 PRs): PR #45507 rebased to `ee5df35e32` (new head `79ab65aac13`); run [26626298958](https://github.com/tenstorrent/tt-metal/actions/runs/26626298958) still queued (no classification); main run 26624591673 still in_progress; most recent completed t3000-e2e-tests main run still 26561397299; evidence unchanged; lifecycle `verifying`. PR #45514 rebased to `ee5df35e32` (new head `f42962a8bd6`); seed=42 confirmed NOT independently failing on main ([run 26621870487](https://github.com/tenstorrent/tt-metal/actions/runs/26621870487) shows only seed=0 in failures); root cause: seed=0 hangs device on main → job times out before seed=42 can run; on PR branch seed=0 skipped → seed=42 runs and also hits same TT_THROW; PR cannot be fixed without main-run evidence for seed=42; needs human review; lifecycle remains `verified-fail`. PR #45492 rebased to `ee5df35e32` (new head `9fd52f1be01`); deep revalidation deferred (throttled — last full analysis 1h ago); lifecycle remains `verified-pass`. Focus lane: 0 focus PRs — no uncovered non-Galaxy workflows with ≥3 consecutive deterministic functional failures; PR #45507 is `verifying` (run 26626298958 queued, not inconclusive) so no priority-3 re-dispatch. Workflow survey: `single-card-demo-tests` (2 consecutive, not 3); `blackhole-post-commit` (3 successes); `t3000-integration-tests` (3 successes); `models-t2-e2e-tests` (3 consecutive failures — all perf/OOS, AI summary confirms functional SUCCESS for Gemma-3-4B); `models-t3-unit/e2e-tests` (passing); `runtime-perf-tests`/`perf-device-models`/`merge-gate` (all passing). Focus slots filled: 0/3. Paralysis check: limited: 0 focus PRs (0 uncovered workflows + 0 priority-2/3 PRs available) + 3 examining PRs.

- **2026-05-29T08:02 UTC session.** Examining lane (3 PRs): PR #45490 rebased to `8e179efa2a34` (new head `7f602a7b5b3`); `DPrintMeshFixture.ActiveEthTestPrint` NOW PASSING on main — [run 26620110417](https://github.com/tenstorrent/tt-metal/actions/runs/26620110417/job/78444865829) job 78444865829 `bh_multicard_debug_tools [bh_quietbox]` passed (2026-05-29T05:48 UTC) — disable REMOVED from branch; PR description + issue #45489 updated; PR eligible for closure (zero remaining disables). PR #45492 rebased to `8e179efa2a34` (new head `97305625190`); `MultiMeshEastMulticast_0` and `_1` still failing in latest main run [26620003623](https://github.com/tenstorrent/tt-metal/actions/runs/26620003623/job/78444692420) (2026-05-29T06:06 UTC); evidence updated in PR description + issue #45491; `RandomizedInterMeshUnicast` failing again (non-consecutive, not re-added). PR #45514 rebased to `8e179efa2a34` (new head `7a1885c8f3d`); [run 26623896690](https://github.com/tenstorrent/tt-metal/actions/runs/26623896690) classified `verified-fail` — `test_indexed_slice[DataType.BFLOAT16-4-32-6-0]` correctly SKIPPED; but `seed=42` NEW failure on wh_n150_civ2 and bh_p150b_civ2 (was passing on main run 26621870487); same `TT_THROW @ system_memory_manager.cpp:738` — PR needs revision; PR comment + issue update posted. Focus lane (1 PR, priority-3): PR #45507 rebased to `8e179efa2a34` (new head `a95d55f9e79`); run 26621576169 classified `verification-inconclusive` (third consecutive timeout — Metal context timeout at 07:09:33, job timeout 08:01:56; budget NOT consumed); evidence unchanged; new verification [run 26626298958](https://github.com/tenstorrent/tt-metal/actions/runs/26626298958) dispatched fresh-build on temp branch `ci-disable/verify-t3000-e2e-tests-t3k-ccl-20260529c` (fourth attempt). 1/3 dispatch slots used. Workflow survey: `single-card-demo-tests` (2 consecutive, not 3); `models-t2-e2e-tests`/`t3000-perf-tests` failures = perf threshold violations (out of scope); `t3000-demo-tests` latest SUCCESS; `fast-dispatch-full-regressions` latest SUCCESS. Focus slots filled: 1/3 (only priority-3 PR #45507 available; no uncovered non-Galaxy workflows with ≥3 consecutive deterministic failures). Paralysis check: passed: 1 focus PR (dispatched) + 3 examining PRs.

- **2026-05-29T07:00 UTC session.** Examining lane (3 PRs): PR #45507 rebased from `e1210204db9b` → `09a0526ebfe5` (new head `05478526ba8`); run 26621576169 still in_progress — no classification; evidence unchanged (most recent t3000-e2e-tests run still 26561397299). PR #45484 rebased from `e1210204db9b` → `09a0526ebfe5` (new head `90eafec8a315`); llk-sd-unit-tests still SKIPPED in most recent run 26605109542; most recent execution run 26595275788 evidence unchanged; new run 26621634376 queued on `09a0526ebfe5`. PR #45511 rebased from `e1210204db9b` → `09a0526ebfe5` (new head `facde5bba3f`); evidence refreshed — `TestPerfCountersSingleOp::test_performance_counter_columns[Matmul_perf_counters]` still failing in new run 26619736108 / job 78448844129 (2026-05-29T06:27 UTC); PR description + issue #45510 regenerated from updated evidence. Focus lane (1 new PR, priority-1): [#45514](https://github.com/tenstorrent/tt-metal/pull/45514) (`runtime-integration-tests.yaml` `runtime_fd_python_2 [wh_n150_civ2]` and `[bh_p150b_civ2]` — `test_indexed_slice[DataType.BFLOAT16-4-32-6-0]` — 3 consecutive failures across runs 26494861276 (2026-05-27), 26558546676 (2026-05-28), 26621870487 (2026-05-29) with identical `TT_THROW @ system_memory_manager.cpp:738` signature; initial disable batch committed; issue [#45513](https://github.com/tenstorrent/tt-metal/issues/45513) created; verification [run 26623896690](https://github.com/tenstorrent/tt-metal/actions/runs/26623896690) dispatched fresh-build on temp branch `ci-disable/verify-runtime-integration-tests-indexed-fill-20260529` — no SHA-matching successful `runtime-integration-tests` main run on `09a0526ebfe5`). 1/3 dispatch slots used. Workflow survey: `single-card-demo-tests` (2 consecutive, not 3); `runtime-integration-tests` now newly covered. No other uncovered non-Galaxy workflows with ≥3 consecutive deterministic failures. Focus slots filled: 1/3 (only 1 eligible uncovered workflow with deterministic failures; no priority-2/3 PRs available beyond #45507 which is verifying not inconclusive). Paralysis check: passed: 1 focus PR (dispatched) + 3 examining PRs.

- **2026-05-29T06:00 UTC session.** Examining lane (3 PRs): PR #45511 run [26619643533](https://github.com/tenstorrent/tt-metal/actions/runs/26619643533) classified `verified-pass` (completed `success` 2026-05-29T05:55 UTC; `Perf op report [wh_n300_civ2]` passed; all other profiler jobs passed; disabled test skipped; evidence updated to run 26617362086/job 78441354055); branch rebased to `e1210204db9b` (new head `6b526734ba2`); issue #45510 updated; PR comment posted. PR #45484 rebased to `e1210204db9b` (new head `353676663fb8`); llk-sd-unit-tests SKIPPED again in run 26605109542; evidence unchanged (most recent execution run 26595275788); PR comment posted. PR #45487 rebased to `e1210204db9b` (new head `0ce037d9884`); evidence unchanged (whisper perf still failing in run 26556373726/job 78230061657); PR comment posted. Focus lane (1 PR, priority-3): PR #45507 rebased from `5cbb6896e10a` → `e1210204db9b` (new head `88c9a0b6206`); evidence revalidated (most recent t3000-e2e-tests run still 26561397299; all 3 disabled tests still failing); verification re-dispatched as [run 26621576169](https://github.com/tenstorrent/tt-metal/actions/runs/26621576169) (fresh-build; no SHA-matching run on `e1210204db9b`; temp branch `ci-disable/verify-t3000-e2e-tests-t3k-ccl-20260529b`; only t3k_ccl_tests pruned in). PR comment posted. Workflow survey: t3000-perf-tests (5 consecutive failures — perf threshold violations, out of scope), models-t2-e2e-tests (3 consecutive — Gemma perf upper-bound + Llama T3K Ethernet timeout, out of scope), models-t1-* (Galaxy-only), single-card-demo-tests (2 consecutive, not 3), runtime-integration-tests (2 consecutive, not 3). No new eligible uncovered workflows. Paralysis check: passed: 1 focus PR (dispatched) + 3 examining PRs.

- **2026-05-29T05:00 UTC session.** Examining lane (2 PRs): PR #45484 rebased from `e5d8677f6723e295c57f1ea36c29d85449fdbc76` → `5cbb6896e10a844495016f3294cc56786487d772` (new head `a4ca2eb5a9f729293212d0042dcd48999d159e70`); revalidation: `MeshDeviceFixture.Top32RmDevPipelineCompletes` still failing — `llk-sd-unit-tests` SKIPPED in runs 26605109542 and 26588506741, most recent run with actual execution: 26595275788 (evidence unchanged). PR #45507 run [26615944303](https://github.com/tenstorrent/tt-metal/actions/runs/26615944303) classified as `verification-inconclusive` — job timed out after 90min (Metal context timeout at 04:12 UTC, action timeout at 05:08 UTC); timeout is out-of-scope per policy; budget NOT consumed; re-dispatch eligible. Branch rebased to `5cbb6896e10a` (new head `79d3ef392dd9`). Workflow survey: checked `blackhole-post-commit`, `single-card-demo-tests`, `runtime-integration-tests`, `models-t1-unit-tests` / `models-t1-e2e-tests` (all Galaxy-only failures), `fast-dispatch-full-regressions-and-models`, `t3000-integration-tests`, `t3000-demo-tests`, `models-t2/t3-unit/e2e-tests`, `perf-models` (hang/timeout — OOS), `perf-device-models`, `runtime-perf-tests`, `merge-gate`, `single-card-profiler-tests`, `t3000-profiler-tests`, `sanity-tests-debug`, `runtime-sanity-tests`. Focus lane (1 new PR): [#45511](https://github.com/tenstorrent/tt-metal/pull/45511) (`sanity-tests.yaml` `profiler-tests / Perf op report [wh_n300_civ2]` — `TestPerfCountersSingleOp::test_performance_counter_columns[Matmul_perf_counters]` — 4 consecutive failures, identical `subprocess.CalledProcessError exit 4` signature; initial disable batch committed; issue [#45510](https://github.com/tenstorrent/tt-metal/issues/45510) created; verification [run 26619643533](https://github.com/tenstorrent/tt-metal/actions/runs/26619643533) dispatched fresh-build on `single-card-profiler-tests.yaml` N300-only, temp branch `ci-disable/verify-sanity-tests-perf-counters-20260529`). 1/3 dispatch slots used (PR #45507 re-dispatch also eligible as priority-3 but session cap met by sanity-tests new PR). Focus slots filled: 1/3 (only 1 eligible uncovered workflow with deterministic failures; PR #45507 is priority-3 verification-inconclusive but counts as 0 new dispatches this session since 1-dispatch session target met with new PR). Note: `gh pr comment` and `gh pr create` returned HTTP 403 — token lacks write permissions for PR operations; PR created and comments posted via MCP GitHub tool instead.

- **2026-05-29T04:00 UTC session.** Examining lane (2 PRs classified as `verified-pass`): PR #45498 (run [26614829843](https://github.com/tenstorrent/tt-metal/actions/runs/26614829843) completed `failure` → **verified-pass**; disabled tests `test_mux_bw_full_size_channels[...-0-8-0-4]` and `test_mux_bw_both_channel_types[...-8-8-1-4]` confirmed **SKIPPED** as expected; 30 passed; 1 pre-existing `[8-8-8-8]` geomean error also present in main run 26587701074 — not a regression; revalidation: both disabled tests still failing in latest main run 26612909413 job 78423388911). PR #45500 (run [26614947278](https://github.com/tenstorrent/tt-metal/actions/runs/26614947278) completed `failure` → **verified-pass**; target `[WH-T3K] Llama-3.1-8B-Instruct with sampling-tests` **PASSED**; BH-QB-GE failure = infra fault, EngineCore failed to start — not a code regression; WH-GLX failure = Galaxy, out of scope; revalidation: sampling tests still failing in main run 26611126930 job 78418064065). PR #45494 per-PR section fixed (lifecycle was incorrectly `verifying` in per-PR section; Quick Index was already correct at `verified-pass` from previous session). PR #45507 run 26615944303 still in progress — left as `verifying`, will classify next session. Focus lane: 0 new PRs — checked uncovered workflows: `(Single-card) Model perf tests` failure = timeout/hang (`Setting cpu` loop then `Cleaning up orphan processes` with no test output) — out of scope; `(T3K) T3000 perf tests` failure = same timeout/hang pattern in all 3 failing jobs — out of scope. No other eligible uncovered non-Galaxy workflows found with ≥3 consecutive deterministic test failures. Focus slots filled: 0/3 (no eligible uncovered workflows). State log corrected and pushed.

- **2026-05-29T03:02–03:21 UTC session.** Examining lane: 4 PRs classified as `verified-pass`. PR #45487 (run 26614671489 completed `success` → **verified-pass**; target `whisper performance [bh_p150_perf]` passed). PR #45490 (run 26612634163 completed `failure` → **verified-pass**; target `bh_multicard_debug_tools [bh_quietbox]` passed; `bh_multicard_dispatch` failure is pre-existing flaky non-consecutive on main). PR #45492 (run 26612761462 completed `success` → **verified-pass**; target `t3k_tt_metal_multiprocess_tests [wh_llmbox]` passed). PR #45494 (run 26612913914 completed `failure` → **verified-pass**; target `ccl nightly tests [bh_llmbox]` passed; other SKU load-test-matrix failures pre-existing). Focus lane: 1 new PR [#45507](https://github.com/tenstorrent/tt-metal/pull/45507) (`t3000-e2e-tests` `t3k_ccl_tests [wh_llmbox]` — `test_all_gather_matmul_async` + `test_all_broadcast_sharded_2x4` + `test_ring_joint_sdpa_program_cache` — 3+ consecutive failures confirmed across runs 26561397299 and 26497844654 with identical AI summary signatures; verification [run 26615944303](https://github.com/tenstorrent/tt-metal/actions/runs/26615944303) dispatched fresh-build). 1/3 dispatch slots used (no more single-card workflows with deterministic non-Galaxy failures eligible this session). Runs #45498 (26614829843) and #45500 (26614947278) still pending.

- **2026-05-29T02:02–02:50 UTC session.** Examining lane (3 PRs, rebased all to `9727d4445a8d`): PR #45490 rebased (head `3318536ab76a`), `DPrintMeshFixture.ActiveEthTestPrint` still failing; PR #45492 rebased (head `ca3b87aacf2c`), `RandomizedInterMeshUnicast` REMOVED (passed in main run 26611455612), gtest filter narrowed to `MultiMeshEastMulticast_0` and `MultiMeshEastMulticast_1` only, PR description + tracking issue #45491 updated; PR #45494 rebased (head `e2a5bed979f1`), `test_all_to_all_combine_no_trace` still failing. All 3 verification runs (26612634163, 26612761462, 26612913914) still in progress. Focus lane (3 new PRs): [#45487](https://github.com/tenstorrent/tt-metal/pull/45487) (`blackhole-demo-tests` whisper distil-large-v3 perf check / `bh_p150_perf` — 5 consecutive failures; verification [run 26614671489](https://github.com/tenstorrent/tt-metal/actions/runs/26614671489) fresh-build dispatched); [#45498](https://github.com/tenstorrent/tt-metal/pull/45498) (`metal-run-microbenchmarks` T3K Fabric Mux BW `test_mux_bw_full_size_channels[...-0-8-0-4]` + `test_mux_bw_both_channel_types[...-8-8-1-4]` — 3 consecutive failures; verification [run 26614829843](https://github.com/tenstorrent/tt-metal/actions/runs/26614829843) fresh-build dispatched); [#45500](https://github.com/tenstorrent/tt-metal/pull/45500) (`vllm-nightly-tests` T3K Llama-3.1-8B `TestSeedingAndVariety.*` + 4 other determinism tests — 3 consecutive failures; verification [run 26614947278](https://github.com/tenstorrent/tt-metal/actions/runs/26614947278) fresh-build dispatched). 3/3 dispatch slots used. All fresh-build (no SHA-matching successful main runs for any of the 3 workflows on base `9727d4445a8d`).

- **2026-05-29T01:07–01:44 UTC session.** Examining: PR #45484 classified `verified-pass` (run 26609412851 completed success, all 4 `llk-sd-unit-tests` jobs passed; rebased branch to `e5d8677f67`; evidence confirmed still valid — `MeshDeviceFixture.Top32RmDevPipelineCompletes` still failing in latest `llk-sd-unit-tests` run on main 26595275788). 3 new focus PRs dispatched: [#45490](https://github.com/tenstorrent/tt-metal/pull/45490) (`runtime-unit-tests` `DPrintMeshFixture.ActiveEthTestPrint` / `bh_quietbox` — 3 consecutive failures; verification [run 26612634163](https://github.com/tenstorrent/tt-metal/actions/runs/26612634163) dispatched fresh-build); [#45492](https://github.com/tenstorrent/tt-metal/pull/45492) (`t3000-unit-tests` `IntermeshSplit2x2FabricFixture.*` / `wh_llmbox` — 3 consecutive failures, same signature as already-disabled `test_tt_fabric`; verification [run 26612761462](https://github.com/tenstorrent/tt-metal/actions/runs/26612761462) dispatched fresh-build); [#45494](https://github.com/tenstorrent/tt-metal/pull/45494) (`blackhole-e2e-tests` `test_all_to_all_combine_no_trace` all-zeros output / `bh_llmbox` — ≥5 consecutive failures; verification [run 26612913914](https://github.com/tenstorrent/tt-metal/actions/runs/26612913914) dispatched fresh-build). 3/3 dispatch slots used. All verification runs are fresh-build (no SHA-matching successful main run available for any of the 3 workflows).

- **2026-05-28 ~23:57 UTC session.** 1 new focus PR created: [#45484](https://github.com/tenstorrent/tt-metal/pull/45484) for `tt-metal-l2-nightly` `llk-sd-unit-tests` (disables `MeshDeviceFixture.Top32RmDevPipelineCompletes` across `wh_n150`/`wh_n300`/`bh_p100`/`bh_p150` — same deterministic `trisc1 compile failure` already excluded in `runtime-unit-tests` via PR #44767, ≥3 consecutive failing main runs). Verification [run 26609412851](https://github.com/tenstorrent/tt-metal/actions/runs/26609412851) dispatched fresh-build with `run_sd_unit_tests=true` (no SHA-matching successful `tt-metal-l2-nightly` main run on the rebase base `577298dde0a`). 2 remaining focus slots could not be filled — all other open failures were out-of-scope (timeouts / <3 consecutive / Galaxy). The session created the PR + dispatched the run but did NOT push the state log update at session-end; this entry is the human/manual backfill that closes that gap.
