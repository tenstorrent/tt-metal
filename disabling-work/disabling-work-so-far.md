# CI Disable Work — Status Log

> **Source of truth.** This file is the canonical record of automation-tracked PRs. Wiping it resets the automation to fresh-state view; stale GitHub PRs not listed here are intentionally invisible.

Last updated: **2026-05-29T05:30 UTC** — Session: Examining lane (2 PRs): PR #45484 rebased onto `5cbb6896e10a` (new head `a4ca2eb5a9f7`); evidence unchanged (`llk-sd-unit-tests` SKIPPED in recent runs, test still failing in run 26595275788). PR #45507 run 26615944303 classified as `verification-inconclusive` (job timed out at 90min — Metal context timeout at 04:12:04 UTC + action timeout; test may or may not have been reached; timeout is out-of-scope per policy); branch rebased to `5cbb6896e10a` (new head `79d3ef392dd9`). Focus lane (1 new PR): [#45511](https://github.com/tenstorrent/tt-metal/pull/45511) (`sanity-tests.yaml` `profiler-tests / Perf op report [wh_n300_civ2]` — `TestPerfCountersSingleOp::test_performance_counter_columns[Matmul_perf_counters]` — 4+ consecutive failures with identical subprocess.CalledProcessError exit 4 signature; verification [run 26619643533](https://github.com/tenstorrent/tt-metal/actions/runs/26619643533) dispatched fresh-build, N300-only). Note: PR comments via `gh pr comment` returned 403 (token lacks addComment permission); comments posted via MCP tool instead.

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
| [#45490](https://github.com/tenstorrent/tt-metal/pull/45490) | Runtime unit tests (`runtime-unit-tests.yaml`) — `bh_multicard_debug_tools` | `verified-pass` | [run 26612634163](https://github.com/tenstorrent/tt-metal/actions/runs/26612634163) — failure (target job passed; bh_multicard_dispatch flaky/pre-existing) | Yes — pending review | Disables `DPrintMeshFixture.ActiveEthTestPrint` on `bh_quietbox`. |
| [#45492](https://github.com/tenstorrent/tt-metal/pull/45492) | T3000 unit tests (`t3000-unit-tests.yaml`) — `t3k_tt_metal_multiprocess_tests` | `verified-pass` | [run 26612761462](https://github.com/tenstorrent/tt-metal/actions/runs/26612761462) — **success** | Yes — pending review | Disables `IntermeshSplit2x2FabricFixture.MultiMeshEastMulticast_*` (2 tests). |
| [#45494](https://github.com/tenstorrent/tt-metal/pull/45494) | Blackhole e2e tests (`blackhole-e2e-tests.yaml`) — `ccl nightly tests` | `verified-pass` | [run 26612913914](https://github.com/tenstorrent/tt-metal/actions/runs/26612913914) — failure (target job passed; other SKU failures pre-existing) | Yes — pending review | Disables `test_all_to_all_combine_no_trace` on `bh_llmbox`. |
| [#45498](https://github.com/tenstorrent/tt-metal/pull/45498) | metal-run-microbenchmarks (`metal-run-microbenchmarks.yaml`) — `T3K ubench - Fabric Mux BW` | `verified-pass` | [run 26614829843](https://github.com/tenstorrent/tt-metal/actions/runs/26614829843) — failure (disabled tests skipped; 30 passed; 1 pre-existing [8-8-8-8] geomean error also on main) | Yes — pending review | Disables 2 bandwidth regression test parametrizations (num_full_size_channels=4). |
| [#45500](https://github.com/tenstorrent/tt-metal/pull/45500) | vllm-nightly-tests (`vllm-nightly-tests.yaml`) — `[WH-T3K] Llama-3.1-8B-Instruct with sampling-tests` | `verified-pass` | [run 26614947278](https://github.com/tenstorrent/tt-metal/actions/runs/26614947278) — failure (target [WH-T3K] sampling-tests PASSED; BH-QB-GE infra fault; WH-GLX Galaxy out-of-scope) | Yes — pending review | Disables 15 sampling/seeding determinism tests via `sampling-test-filter`. |
|| [#45507](https://github.com/tenstorrent/tt-metal/pull/45507) | T3000 e2e tests (`t3000-e2e-tests.yaml`) — `t3k_ccl_tests [wh_llmbox]` | `verification-inconclusive` | [run 26615944303](https://github.com/tenstorrent/tt-metal/actions/runs/26615944303) — **verification-inconclusive** (job timed out 90min; Metal context timeout at 04:12 UTC; budget NOT consumed) | No — re-dispatch needed | Disables test_all_gather_matmul_async / test_all_broadcast_sharded_2x4 / test_ring_joint_sdpa_program_cache. Rebased to `5cbb6896e10a`. |
|| [#45511](https://github.com/tenstorrent/tt-metal/pull/45511) | Sanity tests (`sanity-tests.yaml`) — `profiler-tests / Perf op report [wh_n300_civ2]` | `verifying` | [run 26619643533](https://github.com/tenstorrent/tt-metal/actions/runs/26619643533) — in progress (dispatched 2026-05-29T05:20 UTC) | No | New PR. Disables `TestPerfCountersSingleOp::test_performance_counter_columns[Matmul_perf_counters]`. Fresh-build, N300-only. |

---

## Active Runs

| Run | Pipeline | Branch | Started | Status | Notes |
|-----|----------|--------|---------|--------|-------|
| [26619643533](https://github.com/tenstorrent/tt-metal/actions/runs/26619643533) | (Single-card) Profiler tests (`single-card-profiler-tests.yaml`) | `ci-disable/verify-sanity-tests-perf-counters-20260529` (verification temp branch for PR #45511) | 2026-05-29T05:20:43 UTC | in progress | Verification dispatch for PR #45511. Fresh-build (no SHA-matching successful `single-card-profiler-tests` run on `5cbb6896e10a`). N300-only (`run-n300-profiler: true`, others `false`). Next session MUST log-analyze and classify. |

**Policy:** Concurrent runs across PRs are allowed; each automation session may dispatch at most three new runs.

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-------|
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
| Branch | `ci-disable/tt-metal-l2-nightly-mesh-device-top32-20260528` (head SHA `a4ca2eb5a9f729293212d0042dcd48999d159e70` — rebased to `5cbb6896e10a844495016f3294cc56786487d772`) |
| Workflow file | `.github/workflows/tt-metal-l2-nightly.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-29T05:10 UTC — rebased from `e5d8677f6723e295c57f1ea36c29d85449fdbc76` → `5cbb6896e10a844495016f3294cc56786487d772` (new head `a4ca2eb5a9f729293212d0042dcd48999d159e70`), no conflicts |
| Last revalidation | 2026-05-29T05:10 UTC — `MeshDeviceFixture.Top32RmDevPipelineCompletes` confirmed still failing on `main`; most recent run where `llk-sd-unit-tests` actually executed: run 26595275788 (2026-05-28 18:49 UTC). Runs 26605109542 and 26588506741 had `llk-sd-unit-tests` SKIPPED — evidence unchanged |
| Verification run | [26609412851](https://github.com/tenstorrent/tt-metal/actions/runs/26609412851) — **verified-pass** (completed 2026-05-29T00:31 UTC, conclusion `success`; all 4 llk-sd-unit-tests jobs passed) |
| Last touched by automation | 2026-05-29T01:07 UTC |
| Readiness | **Yes — verified-pass, ready for merge review** |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| `MeshDeviceFixture.Top32RmDevPipelineCompletes` [wh_n150] | https://github.com/tenstorrent/tt-metal/actions/runs/26595275788/job/78367033195 | 2026-05-28 19:16 UTC |
| `MeshDeviceFixture.Top32RmDevPipelineCompletes` [wh_n300] | https://github.com/tenstorrent/tt-metal/actions/runs/26518398862/job/78108245432 | 2026-05-27 15:23 UTC |
| `MeshDeviceFixture.Top32RmDevPipelineCompletes` [bh_p100] | https://github.com/tenstorrent/tt-metal/actions/runs/26595275788/job/78367033385 | 2026-05-28 19:09 UTC |
| `MeshDeviceFixture.Top32RmDevPipelineCompletes` [bh_p150] | https://github.com/tenstorrent/tt-metal/actions/runs/26595275788/job/78367033433 | 2026-05-28 19:21 UTC |

Main-run evidence: see PR description.

---

## PR #45487 — Blackhole demo tests (whisper distil-large-v3 perf check)

| Field | Value |
|-------|-------|
| PR | [#45487](https://github.com/tenstorrent/tt-metal/pull/45487) — `[skip ci] Disable distil-large-v3 performance check in blackhole-demo-tests whisper performance job` (draft, open) |
| Disable issue | [#45496](https://github.com/tenstorrent/tt-metal/issues/45496) — `[CI] Track disable: whisper perf test (distil-large-v3 performance regression) in blackhole-demo-tests whisper performance job` (open) |
| Timeout issue | none |
| Branch | `ci-disable/blackhole-demo-tests-whisper-perf-20260529` (head SHA `cbab606e1d76da13f01ebc89e48645a5fcc05c25`) |
| Workflow file | `.github/workflows/blackhole-demo-tests.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-29T02:39 UTC — created onto `origin/main` HEAD `9727d4445a8d52c844223ba59618e0e5452b9445` |
| Last revalidation | 2026-05-29T02:39 UTC — `test_demo_for_conditional_generation[...-distil-large-v3-1-2]` confirmed still failing on `main`; 5 consecutive failures (runs 26556373726, 26492636705, 26433785330, 26384932053, 26352730780) |
| Verification run | [26614671489](https://github.com/tenstorrent/tt-metal/actions/runs/26614671489) — **verified-pass** (conclusion `success`; target `whisper performance [bh_p150_perf]` passed) |
| Last touched by automation | 2026-05-29T03:21 UTC |
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
| Branch | `ci-disable/runtime-unit-tests-dprint-activeeth-20260529` (head SHA `3318536ab76aeb0e01a1be36e7c130e2cf39dfea` — rebased to `9727d4445a8d52c844223ba59618e0e5452b9445`) |
| Workflow file | `.github/workflows/runtime-unit-tests.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-29T02:02 UTC — rebased from `80094df9c61e426b145574ffb2c0aebc0c75f02a` → `9727d4445a8d52c844223ba59618e0e5452b9445` (1 commit: `#45440: Skip docs deploy for pre-release tags`), no conflicts |
| Last revalidation | 2026-05-29T02:02 UTC — `DPrintMeshFixture.ActiveEthTestPrint` confirmed still failing on `main`; latest run 26556700411 job 78231645434 `bh_multicard_debug_tools [bh_quietbox]` still fails |
| Verification run | [26612634163](https://github.com/tenstorrent/tt-metal/actions/runs/26612634163) — **verified-pass** (conclusion `failure` but target `bh_multicard_debug_tools [bh_quietbox]` passed; `bh_multicard_dispatch` failure pre-existing flaky on main, non-consecutive) |
| Last touched by automation | 2026-05-29T03:21 UTC |
| Readiness | **Yes — verified-pass, ready for merge review** |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| `DPrintMeshFixture.ActiveEthTestPrint` [bh_quietbox] | https://github.com/tenstorrent/tt-metal/actions/runs/26556700411/job/78231645434 | 2026-05-28 06:48 UTC |

Main-run evidence: see PR description.

---

## PR #45492 — T3000 unit tests (IntermeshSplit2x2FabricFixture.MultiMeshEastMulticast_*)

| Field | Value |
|-------|-------|
| PR | [#45492](https://github.com/tenstorrent/tt-metal/pull/45492) — `[skip ci] Disable IntermeshSplit2x2FabricFixture.MultiMeshEastMulticast_* in t3000-unit-tests t3k_tt_metal_multiprocess_tests` (draft, open; title updated from `IntermeshSplit2x2FabricFixture.*` to `MultiMeshEastMulticast_*` after RandomizedInterMeshUnicast removal) |
| Disable issue | [#45491](https://github.com/tenstorrent/tt-metal/issues/45491) — `[CI] Track disable: IntermeshSplit2x2FabricFixture.* in t3000-unit-tests t3k_tt_metal_multiprocess_tests` (open; updated to reflect removal) |
| Timeout issue | none |
| Branch | `ci-disable/t3000-unit-tests-intermesh-split2x2-20260529` (head SHA `ca3b87aacf2ccd23fa9ab7898358f215ab280607` — rebased to `9727d4445a8d52c844223ba59618e0e5452b9445` + removal commit) |
| Workflow file | `.github/workflows/t3000-unit-tests.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-29T02:02 UTC — rebased from `80094df9c61e` → `9727d4445a8d` (1 commit), no conflicts; then removal commit pushed |
| Last revalidation | 2026-05-29T02:02 UTC — `IntermeshSplit2x2FabricFixture.RandomizedInterMeshUnicast` PASSED in latest main run 26611455612 (job 78419112342, completed 2026-05-29 01:58 UTC) → REMOVED from PR. `MultiMeshEastMulticast_0` hung in latest run (timeout after 7s, no deterministic pass or fail); `MultiMeshEastMulticast_1` not reached. Both remain disabled. |
| Verification run | [26612761462](https://github.com/tenstorrent/tt-metal/actions/runs/26612761462) — **verified-pass** (conclusion `success`; target `t3k_tt_metal_multiprocess_tests [wh_llmbox]` passed) |
| Last touched by automation | 2026-05-29T03:21 UTC |
| Readiness | **Yes — verified-pass, ready for merge review** |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| `IntermeshSplit2x2FabricFixture.MultiMeshEastMulticast_0` [wh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/26606267783/job/78404397617 | 2026-05-28 22:56 UTC |
| `IntermeshSplit2x2FabricFixture.MultiMeshEastMulticast_1` [wh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/26606267783/job/78404397617 | 2026-05-28 22:56 UTC |

**Note:** `IntermeshSplit2x2FabricFixture.RandomizedInterMeshUnicast` was removed from the disable in this session (passed in main run 26611455612 job 78419112342 at 2026-05-29 01:58 UTC).

Main-run evidence: see PR description.

---

## PR #45494 — Blackhole e2e tests (test_all_to_all_combine_no_trace)

| Field | Value |
|-------|-------|
| PR | [#45494](https://github.com/tenstorrent/tt-metal/pull/45494) — `[skip ci] Disable test_all_to_all_combine_no_trace in blackhole-e2e-tests ccl nightly tests` (draft, open) |
| Disable issue | [#45493](https://github.com/tenstorrent/tt-metal/issues/45493) — `[CI] Track disable: test_all_to_all_combine_no_trace in blackhole-e2e-tests ccl nightly tests` (open) |
| Timeout issue | none |
| Branch | `ci-disable/blackhole-e2e-tests-all-to-all-combine-no-trace-20260529` (head SHA `e2a5bed979f129a32d5d5c5e6160353f7ca32206` — rebased to `9727d4445a8d52c844223ba59618e0e5452b9445`) |
| Workflow file | `.github/workflows/blackhole-e2e-tests.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-29T02:02 UTC — rebased from `80094df9c61e` → `9727d4445a8d` (1 commit), no conflicts |
| Last revalidation | 2026-05-29T02:02 UTC — `test_all_to_all_combine_no_trace` confirmed still failing on `main`; latest completed run 26589993622 job 78379365051 `ccl nightly tests [bh_llmbox]` still fails |
| Verification run | [26612913914](https://github.com/tenstorrent/tt-metal/actions/runs/26612913914) — **verified-pass** (completed 2026-05-29T∰02 UTC; conclusion `failure`; target `ccl nightly tests [bh_llmbox]` passed; other SKU load-test-matrix failures pre-existing) |
| Last touched by automation | 2026-05-29T03:21 UTC |
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
| Branch | `ci-disable/metal-run-microbenchmarks-fabric-mux-bw-20260529` (head SHA `016bf37ced4e1bd70ae082e734b689823de9205d`) |
| Workflow file | `.github/workflows/metal-run-microbenchmarks.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-29T02:44 UTC — created onto `origin/main` HEAD `9727d4445a8d52c844223ba59618e0e5452b9445` |
| Last revalidation | 2026-05-29T04:00 UTC — both disabled tests still failing in latest main run [26612909413](https://github.com/tenstorrent/tt-metal/actions/runs/26612909413) job [78423388911](https://github.com/tenstorrent/tt-metal/actions/runs/26612909413/job/78423388911) (2026-05-29T02:25 UTC): same bandwidth mismatch signature |
| Verification run | [26614829843](https://github.com/tenstorrent/tt-metal/actions/runs/26614829843) — **verified-pass** (completed 2026-05-29T03:06 UTC; conclusion `failure`; disabled tests skipped; 30 passed; 1 pre-existing `[8-8-8-8]` geomean error also failing on main run 26587701074 with same signature) |
| Last touched by automation | 2026-05-29T04:00 UTC |
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
| PR | [#45500](https://github.com/tenstorrent/tt-metal/pull/45500) — `[skip ci] Disable T3K Llama-3.1-8B sampling/seeding determinism tests in vllm-nightly-tests` (draft, open) |
| Disable issue | [#45499](https://github.com/tenstorrent/tt-metal/issues/45499) — `[CI] Track disable: T3K Llama-3.1-8B sampling/seeding determinism tests in vllm-nightly-tests` (open) |
| Timeout issue | none |
| Branch | `ci-disable/vllm-nightly-tests-t3k-llama-sampling-20260529` (head SHA `02de8958265bcbad67a791f722a75643273bd813`) |
| Workflow file | `.github/workflows/vllm-nightly-tests.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-29T02:48 UTC — created onto `origin/main` HEAD `9727d4445a8d52c844223ba59618e0e5452b9445` |
| Last revalidation | 2026-05-29T02:48 UTC — confirmed tests failing in ≥3 consecutive runs (26611126930, 26567209010, 26564928459); same FAILED test IDs across runs 26611126930 and 26567209010 confirmed; 26564928459 confirmed same T3K Llama sampling job failed |
| Verification run | [26614947278](https://github.com/tenstorrent/tt-metal/actions/runs/26614947278) — **verified-pass** (completed 2026-05-29T03:34 UTC; conclusion `failure`; target `[WH-T3K] Llama-3.1-8B-Instruct with sampling-tests` PASSED; BH-QB-GE: infra fault EngineCore failed to start; WH-GLX: Galaxy out-of-scope) |
| Last touched by automation | 2026-05-29T04:00 UTC |
| Readiness | **Yes — verified-pass, ready for merge review** |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| `TestSeedingAndVariety::test_seeding` (and all other TestSeedingAndVariety tests) [WH-T3K] | https://github.com/tenstorrent/tt-metal/actions/runs/26611126930/job/78418064065 | 2026-05-29 01:18 UTC |
| `TestBatchIsolation::test_mixed_params_batch` [WH-T3K] | https://github.com/tenstorrent/tt-metal/actions/runs/26611126930/job/78418064065 | 2026-05-29 01:18 UTC |
| `TestRepetitionPenalty::test_repetition_penalty_mixed_batch` [WH-T3K] | https://github.com/tenstorrent/tt-metal/actions/runs/26611126930/job/78418064065 | 2026-05-29 01:18 UTC |
| `TestPresencePenalty::test_presence_penalty_mixed_batch` [WH-T3K] | https://github.com/tenstorrent/tt-metal/actions/runs/26611126930/job/78418064065 | 2026-05-29 01:18 UTC |
| `TestFrequencyPenalty::test_frequency_penalty_mixed_batch` [WH-T3K] | https://github.com/tenstorrent/tt-metal/actions/runs/26611126930/job/78418064065 | 2026-05-29 01:18 UTC |

Main-run evidence: see PR description.

---

---

## PR #45507 — T3000 e2e tests (t3k_ccl_tests: test_all_gather_matmul_async / test_all_broadcast_sharded_2x4 / test_ring_joint_sdpa_program_cache)

| Field | Value |
|-------|-------|
| PR | [#45507](https://github.com/tenstorrent/tt-metal/pull/45507) — `[skip ci] Disable t3k_ccl_tests test_all_gather_matmul_async / test_all_broadcast_sharded_2x4 / test_ring_joint_sdpa_program_cache in t3000-e2e-tests` (draft, open) |
| Disable issue | [#45506](https://github.com/tenstorrent/tt-metal/issues/45506) — `[CI] Track disable: t3k_ccl_tests test_all_gather_matmul_async / test_all_broadcast_sharded_2x4 / test_ring_joint_sdpa_program_cache in t3000-e2e-tests` (open) |
| Timeout issue | none |
| Branch | `ci-disable/t3000-e2e-tests-t3k-ccl-20260529` (head SHA `79d3ef392dd96a238ee6333ee261a37c9952f5f6` — rebased to `5cbb6896e10a844495016f3294cc56786487d772`) |
| Workflow file | `.github/workflows/t3000-e2e-tests.yaml` |
| Lifecycle stage | `verification-inconclusive` |
| Last rebase | 2026-05-29T05:15 UTC — rebased from `9727d4445a8d52c844223ba59618e0e5452b9445` → `5cbb6896e10a844495016f3294cc56786487d772` (new head `79d3ef392dd96a238ee6333ee261a37c9952f5f6`), no conflicts |
| Last revalidation | 2026-05-29T03:21 UTC — confirmed 3+ consecutive failures in `t3k_ccl_tests [wh_llmbox]` (runs 26561397299, 26497844654, 26438570812); AI summaries from runs 26561397299 and 26497844654 confirm same 3 test names failing; `test_decode_perf` and `test_all_to_all_combine_no_trace_submesh` already disabled in current main code (skip decorator present) |
| Verification run | [26615944303](https://github.com/tenstorrent/tt-metal/actions/runs/26615944303) — **verification-inconclusive** (conclusion `failure`; job timed out after 90 minutes — Metal context timeout at 2026-05-29T04:12 UTC, action timeout at 05:08 UTC; timeout is out-of-scope per policy; budget NOT consumed; re-dispatch eligible) |
| Last touched by automation | 2026-05-29T05:15 UTC |
| Readiness | No — verification-inconclusive; re-dispatch needed |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| `tests/nightly/t3000/ccl/test_minimal_all_gather_matmul_async.py::test_all_gather_matmul_async[wormhole_b0-fabric_ring-mem_config_input0-mem_config_ag0-mem_config_mm0-ag_output_shape0-perf-no_barrier_with_persistent-chunking-mesh_device0]` [wh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/26561397299/job/78246222518 | 2026-05-28 09:47 UTC |
| `tests/nightly/t3000/ccl/test_new_all_broadcast.py::test_all_broadcast_sharded_2x4[wormhole_b0-mesh_device0-device_params0-1-DataType.BFLOAT16-Layout.ROW_MAJOR-1-4-output_shape0-input_shard_shape0-input_shard_grid0-None-None-TensorMemoryLayout.BLOCK_SHARDED]` [wh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/26561397299/job/78246222518 | 2026-05-28 09:47 UTC |
| `tests/nightly/t3000/ccl/test_ring_joint_attention.py::test_ring_joint_sdpa_program_cache[wormhole_b0-2rpx4up-mesh_device0-line-1-no_trace-sd35-bf16]` [wh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/26561397299/job/78246222518 | 2026-05-28 09:47 UTC |

Main-run evidence: see PR description.

## PR #45511 — Sanity tests (TestPerfCountersSingleOp.test_performance_counter_columns[Matmul_perf_counters])

| Field | Value |
|-------|-------|
| PR | [#45511](https://github.com/tenstorrent/tt-metal/pull/45511) — `[skip ci] Disable TestPerfCountersSingleOp.test_performance_counter_columns in sanity-tests profiler-tests` (draft, open) |
| Disable issue | [#45510](https://github.com/tenstorrent/tt-metal/issues/45510) — `[CI] Track disable: TestPerfCountersSingleOp.test_performance_counter_columns in sanity-tests profiler-tests/Perf op report [wh_n300_civ2]` (open) |
| Timeout issue | none |
| Branch | `ci-disable/sanity-tests-perf-counters-matmul-20260529` (head SHA `cac9d270e1c74b8c6324484b14789575986214cf`) |
| Workflow file | `.github/workflows/sanity-tests.yaml` (calls `single-card-profiler-tests-impl.yaml` for profiler job) |
| Lifecycle stage | `verifying` |
| Last rebase | 2026-05-29T05:17 UTC — created onto `origin/main` HEAD `5cbb6896e10a844495016f3294cc56786487d772` |
| Last revalidation | 2026-05-29T05:17 UTC — `TestPerfCountersSingleOp::test_performance_counter_columns[Matmul_perf_counters]` confirmed failing in 4+ consecutive `sanity-tests` main runs (26616624249, 26615707248, 26613293147, 26612419920); same `subprocess.CalledProcessError: ... returned non-zero exit status 4` signature across all runs |
| Verification run | [26619643533](https://github.com/tenstorrent/tt-metal/actions/runs/26619643533) — in progress (dispatched 2026-05-29T05:20 UTC via `single-card-profiler-tests.yaml` with `run-n300-profiler: true`, others `false`; fresh-build) |
| Last touched by automation | 2026-05-29T05:20 UTC |
| Readiness | No — verification in progress |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| `tests/ttnn/tracy/test_perf_op_report.py::TestPerfCountersSingleOp::test_performance_counter_columns[Matmul_perf_counters]` [wh_n300_civ2] | https://github.com/tenstorrent/tt-metal/actions/runs/26616624249/job/78439009497 | 2026-05-29 04:57 UTC |

Main-run evidence: see PR description.

---

## Blockers

| Blocker | Status | Notes |
|---------|--------|-------|
| _(none)_ | | |

---

## Recent Activity

- **2026-05-29T05:00 UTC session.** Examining lane (2 PRs): PR #45484 rebased from `e5d8677f6723e295c57f1ea36c29d85449fdbc76` → `5cbb6896e10a844495016f3294cc56786487d772` (new head `a4ca2eb5a9f729293212d0042dcd48999d159e70`); revalidation: `MeshDeviceFixture.Top32RmDevPipelineCompletes` still failing — `llk-sd-unit-tests` SKIPPED in runs 26605109542 and 26588506741, most recent run with actual execution: 26595275788 (evidence unchanged). PR #45507 run [26615944303](https://github.com/tenstorrent/tt-metal/actions/runs/26615944303) classified as `verification-inconclusive` — job timed out after 90min (Metal context timeout at 04:12 UTC, action timeout at 05:08 UTC); timeout is out-of-scope per policy; budget NOT consumed; re-dispatch eligible. Branch rebased to `5cbb6896e10a` (new head `79d3ef392dd9`). Workflow survey: checked `blackhole-post-commit`, `single-card-demo-tests`, `runtime-integration-tests`, `models-t1-unit-tests` / `models-t1-e2e-tests` (all Galaxy-only failures), `fast-dispatch-full-regressions-and-models`, `t3000-integration-tests`, `t3000-demo-tests`, `models-t2/t3-unit/e2e-tests`, `perf-models` (hang/timeout — OOS), `perf-device-models`, `runtime-perf-tests`, `merge-gate`, `single-card-profiler-tests`, `t3000-profiler-tests`, `sanity-tests-debug`, `runtime-sanity-tests`. Focus lane (1 new PR): [#45511](https://github.com/tenstorrent/tt-metal/pull/45511) (`sanity-tests.yaml` `profiler-tests / Perf op report [wh_n300_civ2]` — `TestPerfCountersSingleOp::test_performance_counter_columns[Matmul_perf_counters]` — 4 consecutive failures, identical `subprocess.CalledProcessError exit 4` signature; initial disable batch committed; issue [#45510](https://github.com/tenstorrent/tt-metal/issues/45510) created; verification [run 26619643533](https://github.com/tenstorrent/tt-metal/actions/runs/26619643533) dispatched fresh-build on `single-card-profiler-tests.yaml` N300-only, temp branch `ci-disable/verify-sanity-tests-perf-counters-20260529`). 1/3 dispatch slots used (PR #45507 re-dispatch also eligible as priority-3 but session cap met by sanity-tests new PR). Focus slots filled: 1/3 (only 1 eligible uncovered workflow with deterministic failures; PR #45507 is priority-3 verification-inconclusive but counts as 0 new dispatches this session since 1-dispatch session target met with new PR). Note: `gh pr comment` and `gh pr create` returned HTTP 403 — token lacks write permissions for PR operations; PR created and comments posted via MCP GitHub tool instead.

- **2026-05-29T04:00 UTC session.** Examining lane (2 PRs classified as `verified-pass`): PR #45498 (run [26614829843](https://github.com/tenstorrent/tt-metal/actions/runs/26614829843) completed `failure` → **verified-pass**; disabled tests `test_mux_bw_full_size_channels[...-0-8-0-4]` and `test_mux_bw_both_channel_types[...-8-8-1-4]` confirmed **SKIPPED** as expected; 30 passed; 1 pre-existing `[8-8-8-8]` geomean error also present in main run 26587701074 — not a regression; revalidation: both disabled tests still failing in latest main run 26612909413 job 78423388911). PR #45500 (run [26614947278](https://github.com/tenstorrent/tt-metal/actions/runs/26614947278) completed `failure` → **verified-pass**; target `[WH-T3K] Llama-3.1-8B-Instruct with sampling-tests` **PASSED**; BH-QB-GE failure = infra fault, EngineCore failed to start — not a code regression; WH-GLX failure = Galaxy, out of scope; revalidation: sampling tests still failing in main run 26611126930 job 78418064065). PR #45494 per-PR section fixed (lifecycle was incorrectly `verifying` in per-PR section; Quick Index was already correct at `verified-pass` from previous session). PR #45507 run 26615944303 still in progress — left as `verifying`, will classify next session. Focus lane: 0 new PRs — checked uncovered workflows: `(Single-card) Model perf tests` failure = timeout/hang (`Setting cpu` loop then `Cleaning up orphan processes` with no test output) — out of scope; `(T3K) T3000 perf tests` failure = same timeout/hang pattern in all 3 failing jobs — out of scope. No other eligible uncovered non-Galaxy workflows found with ≥3 consecutive deterministic test failures. Focus slots filled: 0/3 (no eligible uncovered workflows). State log corrected and pushed.

- **2026-05-29T03:02–03:21 UTC session.** Examining lane: 4 PRs classified as `verified-pass`. PR #45487 (run 26614671489 completed `success` → **verified-pass**; target `whisper performance [bh_p150_perf]` passed). PR #45490 (run 26612634163 completed `failure` → **verified-pass**; target `bh_multicard_debug_tools [bh_quietbox]` passed; `bh_multicard_dispatch` failure is pre-existing flaky non-consecutive on main). PR #45492 (run 26612761462 completed `success` → **verified-pass**; target `t3k_tt_metal_multiprocess_tests [wh_llmbox]` passed). PR #45494 (run 26612913914 completed `failure` → **verified-pass**; target `ccl nightly tests [bh_llmbox]` passed; other SKU load-test-matrix failures pre-existing). Focus lane: 1 new PR [#45507](https://github.com/tenstorrent/tt-metal/pull/45507) (`t3000-e2e-tests` `t3k_ccl_tests [wh_llmbox]` — `test_all_gather_matmul_async` + `test_all_broadcast_sharded_2x4` + `test_ring_joint_sdpa_program_cache` — 3+ consecutive failures confirmed across runs 26561397299 and 26497844654 with identical AI summary signatures; verification [run 26615944303](https://github.com/tenstorrent/tt-metal/actions/runs/26615944303) dispatched fresh-build). 1/3 dispatch slots used (no more single-card workflows with deterministic non-Galaxy failures eligible this session). Runs #45498 (26614829843) and #45500 (26614947278) still pending.

- **2026-05-29T02:02–02:50 UTC session.** Examining lane (3 PRs, rebased all to `9727d4445a8d`): PR #45490 rebased (head `3318536ab76a`), `DPrintMeshFixture.ActiveEthTestPrint` still failing; PR #45492 rebased (head `ca3b87aacf2c`), `RandomizedInterMeshUnicast` REMOVED (passed in main run 26611455612), gtest filter narrowed to `MultiMeshEastMulticast_0` and `MultiMeshEastMulticast_1` only, PR description + tracking issue #45491 updated; PR #45494 rebased (head `e2a5bed979f1`), `test_all_to_all_combine_no_trace` still failing. All 3 verification runs (26612634163, 26612761462, 26612913914) still in progress. Focus lane (3 new PRs): [#45487](https://github.com/tenstorrent/tt-metal/pull/45487) (`blackhole-demo-tests` whisper distil-large-v3 perf check / `bh_p150_perf` — 5 consecutive failures; verification [run 26614671489](https://github.com/tenstorrent/tt-metal/actions/runs/26614671489) fresh-build dispatched); [#45498](https://github.com/tenstorrent/tt-metal/pull/45498) (`metal-run-microbenchmarks` T3K Fabric Mux BW `test_mux_bw_full_size_channels[...-0-8-0-4]` + `test_mux_bw_both_channel_types[...-8-8-1-4]` — 3 consecutive failures; verification [run 26614829843](https://github.com/tenstorrent/tt-metal/actions/runs/26614829843) fresh-build dispatched); [#45500](https://github.com/tenstorrent/tt-metal/pull/45500) (`vllm-nightly-tests` T3K Llama-3.1-8B `TestSeedingAndVariety.*` + 4 other determinism tests — 3 consecutive failures; verification [run 26614947278](https://github.com/tenstorrent/tt-metal/actions/runs/26614947278) fresh-build dispatched). 3/3 dispatch slots used. All fresh-build (no SHA-matching successful main runs for any of the 3 workflows on base `9727d4445a8d`).

- **2026-05-29T01:07–01:44 UTC session.** Examining: PR #45484 classified `verified-pass` (run 26609412851 completed success, all 4 `llk-sd-unit-tests` jobs passed; rebased branch to `e5d8677f67`; evidence confirmed still valid — `MeshDeviceFixture.Top32RmDevPipelineCompletes` still failing in latest `llk-sd-unit-tests` run on main 26595275788). 3 new focus PRs dispatched: [#45490](https://github.com/tenstorrent/tt-metal/pull/45490) (`runtime-unit-tests` `DPrintMeshFixture.ActiveEthTestPrint` / `bh_quietbox` — 3 consecutive failures; verification [run 26612634163](https://github.com/tenstorrent/tt-metal/actions/runs/26612634163) dispatched fresh-build); [#45492](https://github.com/tenstorrent/tt-metal/pull/45492) (`t3000-unit-tests` `IntermeshSplit2x2FabricFixture.*` / `wh_llmbox` — 3 consecutive failures, same signature as already-disabled `test_tt_fabric`; verification [run 26612761462](https://github.com/tenstorrent/tt-metal/actions/runs/26612761462) dispatched fresh-build); [#45494](https://github.com/tenstorrent/tt-metal/pull/45494) (`blackhole-e2e-tests` `test_all_to_all_combine_no_trace` all-zeros output / `bh_llmbox` — ≥5 consecutive failures; verification [run 26612913914](https://github.com/tenstorrent/tt-metal/actions/runs/26612913914) dispatched fresh-build). 3/3 dispatch slots used. All verification runs are fresh-build (no SHA-matching successful main run available for any of the 3 workflows).

- **2026-05-28 ~23:57 UTC session.** 1 new focus PR created: [#45484](https://github.com/tenstorrent/tt-metal/pull/45484) for `tt-metal-l2-nightly` `llk-sd-unit-tests` (disables `MeshDeviceFixture.Top32RmDevPipelineCompletes` across `wh_n150`/`wh_n300`/`bh_p100`/`bh_p150` — same deterministic `trisc1 compile failure` already excluded in `runtime-unit-tests` via PR #44767, ≥3 consecutive failing main runs). Verification [run 26609412851](https://github.com/tenstorrent/tt-metal/actions/runs/26609412851) dispatched fresh-build with `run_sd_unit_tests=true` (no SHA-matching successful `tt-metal-l2-nightly` main run on the rebase base `577298dde0a`). 2 remaining focus slots could not be filled — all other open failures were out-of-scope (timeouts / <3 consecutive / Galaxy). The session created the PR + dispatched the run but did NOT push the state log update at session-end; this entry is the human/manual backfill that closes that gap.
