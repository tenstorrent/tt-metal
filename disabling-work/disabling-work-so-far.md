# CI Disable Work — Status Log

> **Source of truth.** This file is the canonical record of automation-tracked PRs. Wiping it resets the automation to fresh-state view; stale GitHub PRs not listed here are intentionally invisible.

Last updated: **2026-05-29T01:44 UTC** — Session: PR #45484 classified `verified-pass` (run 26609412851 completed success, all 4 llk-sd-unit-tests jobs passed); branch rebased to `e5d8677f67`; 3 new focus PRs created (#45490 `runtime-unit-tests`, #45492 `t3000-unit-tests`, #45494 `blackhole-e2e-tests`); 3 verification runs dispatched (26612634163, 26612761462, 26612913914).

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
| [#45484](https://github.com/tenstorrent/tt-metal/pull/45484) | Nightly tt-metal L2 tests (`tt-metal-l2-nightly.yaml`) — `llk-sd-unit-tests` | `verified-pass` | [run 26609412851](https://github.com/tenstorrent/tt-metal/actions/runs/26609412851) — success (completed 2026-05-29T00:31 UTC) | Yes — pending review | All 4 llk-sd-unit-tests jobs passed. Branch rebased to `e5d8677f67`. Disables `MeshDeviceFixture.Top32RmDevPipelineCompletes` across wh_n150/wh_n300/bh_p100/bh_p150. |
| [#45490](https://github.com/tenstorrent/tt-metal/pull/45490) | Runtime unit tests (`runtime-unit-tests.yaml`) — `bh_multicard_debug_tools` | `verifying` | [run 26612634163](https://github.com/tenstorrent/tt-metal/actions/runs/26612634163) — in progress (started 2026-05-29T01:33 UTC) | No | New PR. Disables `DPrintMeshFixture.ActiveEthTestPrint` on `bh_quietbox`. Fresh-build dispatch (no SHA-matching successful `runtime-unit-tests` main run). |
| [#45492](https://github.com/tenstorrent/tt-metal/pull/45492) | T3000 unit tests (`t3000-unit-tests.yaml`) — `t3k_tt_metal_multiprocess_tests` | `verifying` | [run 26612761462](https://github.com/tenstorrent/tt-metal/actions/runs/26612761462) — in progress (started 2026-05-29T01:37 UTC) | No | New PR. Disables `IntermeshSplit2x2FabricFixture.*` (3 tests) on `wh_llmbox`. Fresh-build dispatch. |
| [#45494](https://github.com/tenstorrent/tt-metal/pull/45494) | Blackhole e2e tests (`blackhole-e2e-tests.yaml`) — `ccl nightly tests` | `verifying` | [run 26612913914](https://github.com/tenstorrent/tt-metal/actions/runs/26612913914) — in progress (started 2026-05-29T01:42 UTC) | No | New PR. Disables `test_all_to_all_combine_no_trace` (all 16 parametrizations) on `bh_llmbox`. Fresh-build dispatch. |

---

## Active Runs

| Run | Pipeline | Branch | Started | Status | Notes |
|-----|----------|--------|---------|--------|-------|
| [26612634163](https://github.com/tenstorrent/tt-metal/actions/runs/26612634163) | Runtime unit tests (`runtime-unit-tests.yaml`) | `ci-disable/verify-runtime-unit-tests-dprint-activeeth-20260529` (verification temp branch for PR #45490) | 2026-05-29T01:33:20 UTC | in progress | Verification dispatch for PR #45490. Fresh-build (no SHA-matching successful `runtime-unit-tests` main run for rebase base `80094df9c61e`). Pruned to `bh_quietbox` SKU only. Next session MUST log-analyze and classify. |
| [26612761462](https://github.com/tenstorrent/tt-metal/actions/runs/26612761462) | T3000 unit tests (`t3000-unit-tests.yaml`) | `ci-disable/verify-t3000-unit-tests-intermesh-split2x2-20260529` (verification temp branch for PR #45492) | 2026-05-29T01:37:25 UTC | in progress | Verification dispatch for PR #45492. Fresh-build (all recent `t3000-unit-tests` main runs failed). Pruned to `t3k_tt_metal_multiprocess_tests` only via modified `t3k_unit_tests.yaml`. Next session MUST log-analyze and classify. |
| [26612913914](https://github.com/tenstorrent/tt-metal/actions/runs/26612913914) | Blackhole e2e tests (`blackhole-e2e-tests.yaml`) | `ci-disable/verify-blackhole-e2e-tests-all-to-all-combine-no-trace-20260529` (verification temp branch for PR #45494) | 2026-05-29T01:42:23 UTC | in progress | Verification dispatch for PR #45494. Fresh-build (all recent 10 `blackhole-e2e-tests` main runs failed). Pruned to `ccl nightly tests [bh_llmbox]` only via modified `blackhole_e2e_tests.yaml`. Next session MUST log-analyze and classify. |

**Policy:** Concurrent runs across PRs are allowed; each automation session may dispatch at most three new runs.

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-------|
| [26609412851](https://github.com/tenstorrent/tt-metal/actions/runs/26609412851) | Nightly tt-metal L2 tests (`tt-metal-l2-nightly.yaml`) | `ci-disable/tt-metal-l2-nightly-mesh-device-top32-20260528` | 2026-05-28T23:57:22 UTC | 2026-05-29T00:31:40 UTC | **verified-pass** | All 4 llk-sd-unit-tests jobs passed (N150, N300, P100, P150 — all `success`). No regressions. PR #45484 → `verified-pass`. |

---

## PR #45484 — Nightly tt-metal L2 tests (MeshDeviceFixture.Top32RmDevPipelineCompletes)

| Field | Value |
|-------|-------|
| PR | [#45484](https://github.com/tenstorrent/tt-metal/pull/45484) — `[skip ci] Disable MeshDeviceFixture.Top32RmDevPipelineCompletes in tt-metal-l2-nightly llk-sd-unit-tests` (draft, open) |
| Disable issue | [#45483](https://github.com/tenstorrent/tt-metal/issues/45483) — `[CI] Track disable: MeshDeviceFixture.Top32RmDevPipelineCompletes in tt-metal-l2-nightly llk-sd-unit-tests` (open) |
| Timeout issue | none |
| Branch | `ci-disable/tt-metal-l2-nightly-mesh-device-top32-20260528` (head SHA `1fc7f56ced4018bf87ae92251711f408f5153e61` — rebased to `e5d8677f6723e295c57f1ea36c29d85449fdbc76`) |
| Workflow file | `.github/workflows/tt-metal-l2-nightly.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-05-29T01:07 UTC — rebased from `577298dde0ac8bfb943e44997162ee14e9b0069b` → `e5d8677f6723e295c57f1ea36c29d85449fdbc76` (new head `1fc7f56ced4018bf87ae92251711f408f5153e61`), no conflicts |
| Last revalidation | 2026-05-29T01:07 UTC — `MeshDeviceFixture.Top32RmDevPipelineCompletes` confirmed still failing on `main`; most recent `llk-sd-unit-tests` run with tests actually executing: run 26595275788 (2026-05-28 18:49 UTC); two most recent runs (26605109542, 26588506741) had `llk-sd-unit-tests` skipped entirely |
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

## PR #45490 — Runtime unit tests (DPrintMeshFixture.ActiveEthTestPrint)

| Field | Value |
|-------|-------|
| PR | [#45490](https://github.com/tenstorrent/tt-metal/pull/45490) — `[skip ci] Disable DPrintMeshFixture.ActiveEthTestPrint in runtime-unit-tests bh_multicard_debug_tools` (draft, open) |
| Disable issue | [#45489](https://github.com/tenstorrent/tt-metal/issues/45489) — `[CI] Track disable: DPrintMeshFixture.ActiveEthTestPrint in runtime-unit-tests bh_multicard_debug_tools` (open) |
| Timeout issue | none |
| Branch | `ci-disable/runtime-unit-tests-dprint-activeeth-20260529` (head SHA `5d3245c208542d256f59e19cd3c7cb05f37a6b79`) |
| Workflow file | `.github/workflows/runtime-unit-tests.yaml` |
| Lifecycle stage | `verifying` |
| Last rebase | 2026-05-29 — created today onto `origin/main` HEAD `80094df9c61e426b145574ffb2c0aebc0c75f02a` |
| Last revalidation | session-start (2026-05-29T01:07 UTC) — `DPrintMeshFixture.ActiveEthTestPrint` confirmed still failing on `main`; 3 consecutive failures in runs 26492962341, 26548660473, 26556700411 (job 78231645434 confirmed same test failing with same signature) |
| Verification run | [26612634163](https://github.com/tenstorrent/tt-metal/actions/runs/26612634163) — in progress (dispatched 2026-05-29T01:33 UTC, fresh-build, pruned to `bh_quietbox` SKU) |
| Last touched by automation | 2026-05-29T01:33 UTC |
| Readiness | No — verification in progress |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| `DPrintMeshFixture.ActiveEthTestPrint` [bh_quietbox] | https://github.com/tenstorrent/tt-metal/actions/runs/26556700411/job/78231645434 | 2026-05-28 05:53 UTC |

Main-run evidence: see PR description.

---

## PR #45492 — T3000 unit tests (IntermeshSplit2x2FabricFixture.*)

| Field | Value |
|-------|-------|
| PR | [#45492](https://github.com/tenstorrent/tt-metal/pull/45492) — `[skip ci] Disable IntermeshSplit2x2FabricFixture.* in t3000-unit-tests t3k_tt_metal_multiprocess_tests` (draft, open) |
| Disable issue | [#45491](https://github.com/tenstorrent/tt-metal/issues/45491) — `[CI] Track disable: IntermeshSplit2x2FabricFixture.* in t3000-unit-tests t3k_tt_metal_multiprocess_tests` (open) |
| Timeout issue | none |
| Branch | `ci-disable/t3000-unit-tests-intermesh-split2x2-20260529` (head SHA `e64ae450d573ee3dfa20d462f39dd0e56c7e5abe`) |
| Workflow file | `.github/workflows/t3000-unit-tests.yaml` |
| Lifecycle stage | `verifying` |
| Last rebase | 2026-05-29 — created today onto `origin/main` HEAD `80094df9c61e426b145574ffb2c0aebc0c75f02a` |
| Last revalidation | session-start (2026-05-29T01:07 UTC) — `IntermeshSplit2x2FabricFixture.*` confirmed still failing on `main` across ≥3 consecutive `t3000-unit-tests` runs; same `TT_FATAL: Physical chip id not found for eth coord` signature as already-disabled `test_tt_fabric` (issue #45305) |
| Verification run | [26612761462](https://github.com/tenstorrent/tt-metal/actions/runs/26612761462) — in progress (dispatched 2026-05-29T01:37 UTC, fresh-build, pruned to `t3k_tt_metal_multiprocess_tests` only) |
| Last touched by automation | 2026-05-29T01:37 UTC |
| Readiness | No — verification in progress |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| `IntermeshSplit2x2FabricFixture.RandomizedInterMeshUnicast` [wh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/26606267783/job/78404397617 | 2026-05-28 22:56 UTC |
| `IntermeshSplit2x2FabricFixture.MultiMeshEastMulticast_0` [wh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/26606267783/job/78404397617 | 2026-05-28 22:56 UTC |
| `IntermeshSplit2x2FabricFixture.MultiMeshEastMulticast_1` [wh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/26606267783/job/78404397617 | 2026-05-28 22:56 UTC |

Main-run evidence: see PR description.

---

## PR #45494 — Blackhole e2e tests (test_all_to_all_combine_no_trace)

| Field | Value |
|-------|-------|
| PR | [#45494](https://github.com/tenstorrent/tt-metal/pull/45494) — `[skip ci] Disable test_all_to_all_combine_no_trace in blackhole-e2e-tests ccl nightly tests` (draft, open) |
| Disable issue | [#45493](https://github.com/tenstorrent/tt-metal/issues/45493) — `[CI] Track disable: test_all_to_all_combine_no_trace in blackhole-e2e-tests ccl nightly tests` (open) |
| Timeout issue | none |
| Branch | `ci-disable/blackhole-e2e-tests-all-to-all-combine-no-trace-20260529` (head SHA `0023866c6da41d4d7b3b2a3a0b2b6d2a3b7e4a12`) |
| Workflow file | `.github/workflows/blackhole-e2e-tests.yaml` |
| Lifecycle stage | `verifying` |
| Last rebase | 2026-05-29 — created today onto `origin/main` HEAD `80094df9c61e426b145574ffb2c0aebc0c75f02a` |
| Last revalidation | session-start (2026-05-29T01:07 UTC) — `test_all_to_all_combine_no_trace` confirmed still failing on `main` across ≥5 consecutive `blackhole-e2e-tests` runs (ccl nightly tests [bh_llmbox]); all 16 parametrizations produce `AssertionError: Equal check failed k=0,b=0,s=0 test_tensor=[0.,0.,...]` consistently |
| Verification run | [26612913914](https://github.com/tenstorrent/tt-metal/actions/runs/26612913914) — in progress (dispatched 2026-05-29T01:42 UTC, fresh-build, pruned to `ccl nightly tests [bh_llmbox]` only) |
| Last touched by automation | 2026-05-29T01:42 UTC |
| Readiness | No — verification in progress |

### Disables (with main evidence)

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| `test_all_to_all_combine_no_trace[...fabric_1d_line_axis_0]` (all mem/local_reduce combos) [bh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/26589993622/job/78379365051 | 2026-05-28 18:46 UTC |
| `test_all_to_all_combine_no_trace[...fabric_1d_ring_axis_0]` (all mem/local_reduce combos) [bh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/26589993622/job/78379365051 | 2026-05-28 18:46 UTC |

Main-run evidence: see PR description.

---

## Blockers

| Blocker | Status | Notes |
|---------|--------|-------|
| _(none)_ | | |

---

## Recent Activity

- **2026-05-29T01:07–01:44 UTC session.** Examining: PR #45484 classified `verified-pass` (run 26609412851 completed success, all 4 `llk-sd-unit-tests` jobs passed; rebased branch to `e5d8677f67`; evidence confirmed still valid — `MeshDeviceFixture.Top32RmDevPipelineCompletes` still failing in latest `llk-sd-unit-tests` run on main 26595275788). 3 new focus PRs dispatched: [#45490](https://github.com/tenstorrent/tt-metal/pull/45490) (`runtime-unit-tests` `DPrintMeshFixture.ActiveEthTestPrint` / `bh_quietbox` — 3 consecutive failures; verification [run 26612634163](https://github.com/tenstorrent/tt-metal/actions/runs/26612634163) dispatched fresh-build); [#45492](https://github.com/tenstorrent/tt-metal/pull/45492) (`t3000-unit-tests` `IntermeshSplit2x2FabricFixture.*` / `wh_llmbox` — 3 consecutive failures, same signature as already-disabled `test_tt_fabric`; verification [run 26612761462](https://github.com/tenstorrent/tt-metal/actions/runs/26612761462) dispatched fresh-build); [#45494](https://github.com/tenstorrent/tt-metal/pull/45494) (`blackhole-e2e-tests` `test_all_to_all_combine_no_trace` all-zeros output / `bh_llmbox` — ≥5 consecutive failures; verification [run 26612913914](https://github.com/tenstorrent/tt-metal/actions/runs/26612913914) dispatched fresh-build). 3/3 dispatch slots used. All verification runs are fresh-build (no SHA-matching successful main run available for any of the 3 workflows).

- **2026-05-28 ~23:57 UTC session.** 1 new focus PR created: [#45484](https://github.com/tenstorrent/tt-metal/pull/45484) for `tt-metal-l2-nightly` `llk-sd-unit-tests` (disables `MeshDeviceFixture.Top32RmDevPipelineCompletes` across `wh_n150`/`wh_n300`/`bh_p100`/`bh_p150` — same deterministic `trisc1 compile failure` already excluded in `runtime-unit-tests` via PR #44767, ≥3 consecutive failing main runs). Verification [run 26609412851](https://github.com/tenstorrent/tt-metal/actions/runs/26609412851) dispatched fresh-build with `run_sd_unit_tests=true` (no SHA-matching successful `tt-metal-l2-nightly` main run on the rebase base `577298dde0a`). 2 remaining focus slots could not be filled — all other open failures were out-of-scope (timeouts / <3 consecutive / Galaxy). The session created the PR + dispatched the run but did NOT push the state log update at session-end; this entry is the human/manual backfill that closes that gap. The policy doc has been hardened in the same commit to forbid `gh pr list` / `git log` / `git show` / web-UI history reconstruction of state-log content and to add a hard `Session-End Invariants (BLOCKING)` checklist that names the missing-state-log-push case as a broken session.
