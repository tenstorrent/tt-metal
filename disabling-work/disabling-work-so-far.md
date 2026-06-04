# CI Disable Work — Status Log

**Last updated:** 2026-06-04T03:15 UTC

> **Source of truth.** This file is the canonical record of automation-tracked PRs. Wiping it resets the automation to fresh-state view; stale GitHub PRs not listed here are intentionally invisible.

---

## How to read/update this file

- Read this file at the start of every automation session and treat it as the authoritative current state for CI disable work.
- Scan the `## Quick Index` table first; it gives the lifecycle stage per PR before drilling into details.
- Per-PR sections use uniform field tables (`PR | Disable issue | Timeout issue | Branch | Workflow file | Lifecycle stage | Last rebase | Last revalidation | Verification run | Last touched by automation | Readiness`); update fields in place rather than rewriting the section. `Last touched by automation: <UTC ISO>` is required on every PR row and drives the 4-hour throttle — update it every time the automation does any work on the PR (rebase, dispatch, log analysis, comment, removal).
- Each PR section also carries a `Disables (with main evidence)` table listing every currently-disabled test ID together with the most recent failing main-run job-link (`/runs/<id>/job/<jid>`), the commit SHA that run was built from (rendered as a link to `https://github.com/tenstorrent/tt-metal/commit/<sha>`), and the run completion timestamp. This mirrors the PR description's evidence table (see `disabling-work/ci-disable-targeted-verification.md` → `Main-run evidence model`) and is the starting point the next session re-checks before doing any work on the PR. If keeping the state log compact is preferred, the per-PR section MAY instead include the one-line pointer `Main-run evidence: see PR description.` — the PR description's evidence table is authoritative either way. Preserve any existing PR entries unchanged when extending the schema.
- Append new entries to the top of `## Recent Activity` (most recent first); keep at most 30 entries — trim older entries to a single `- Older history truncated — see git history of this file.` line if needed.
- Commit and push any change to this file before ending the session.
- Lifecycle stages: `new`, `batch-committed`, `verifying`, `verification-inconclusive`, `verified-pass`, `verified-fail`, `merged`, `out-of-scope`. (`verification-inconclusive` = a verification was dispatched but failed to actually exercise the previously-passing jobs; eligible for re-dispatch and does NOT consume the one-run-per-PR budget.)
- **Multiple Quick Index rows MAY exist for the same workflow file** — one row per open automation PR (see `disabling-work/ci-disable-targeted-verification.md` → `## Multi-PR per workflow`). To compute the test-level disable set for a workflow, take the union of the `Disables (with main evidence)` table contents across every Quick Index row whose `Workflow` column references that workflow file. A workflow has "uncovered failing tests" iff its deterministic-main-failure set contains at least one test not in that union. Cross-PR conflict prevention: a new PR's initial batch MUST exclude every test already in any currently-OPEN automation PR's disable set for that workflow.
- **Per-PR sections** are appended below `## Recently Completed Runs` as PRs are created — one `## PR #<num> — <workflow> (<test summary>)` section per tracked PR, in the schema described above. On a freshly-cleared state log there are no per-PR sections yet; they will be created by the next session as PRs are opened.

---

## Quick Index

> Multiple PRs per workflow are allowed (see `disabling-work/ci-disable-targeted-verification.md` → `## Multi-PR per workflow`). Coverage is computed test-by-test by taking the union of every open automation PR's `Disables (with main evidence)` table for that workflow; a workflow has "uncovered failing tests" iff at least one deterministic main failure for that workflow is missing from that union.

| PR | Workflow | Lifecycle stage | Verification result | Ready to merge? | Notes |
|----|----------|-----------------|---------------------|-----------------|-------|
| [#46023](https://github.com/tenstorrent/tt-metal/pull/46023) | `runtime-unit-tests.yaml` | `out-of-scope` | pass (but PR closed) | No (closed) | Run 26921420038 completed SUCCESS (verified-pass); BUT test now SKIPPED on main (fixture fix by msudumbrekar); disable removed; PR #46023 closed, issue #46022 closed 2026-06-04T01:15Z |
| [#46017](https://github.com/tenstorrent/tt-metal/pull/46017) | `models-t2-e2e-tests.yaml` | `verified-pass` | pass | Yes | Run 26917123038 completed SUCCESS 2026-06-03T23:03Z; batch1-trace SKIPPED; rebased to 275272621c5 2026-06-04T03:15Z |
| [#46010](https://github.com/tenstorrent/tt-metal/pull/46010) | `blackhole-post-commit.yaml` | `verified-pass` | pass | Yes | Run 26913761687 completed; all 8 sdpa_tail SKIPPED; other failures pre-existing; rebased to 275272621c5 2026-06-04T03:15Z |
| [#45999](https://github.com/tenstorrent/tt-metal/pull/45999) | `runtime-unit-tests.yaml` | `verified-pass` | pass | Yes | Run 26910857102 completed failure but verified-pass; rebased to 275272621c5 2026-06-04T03:15Z |
| [#45993](https://github.com/tenstorrent/tt-metal/pull/45993) | `blackhole-demo-tests.yaml` | `verified-pass` | pass | Yes | Run 26908439442 completed SUCCESS; rebased to 275272621c5 2026-06-04T03:15Z |
| [#45991](https://github.com/tenstorrent/tt-metal/pull/45991) | `runtime-integration-tests.yaml` | `verified-pass` | pass | Yes | Run 26907813785 completed failure but verified-pass; rebased to 275272621c5 2026-06-04T03:15Z |
| [#45990](https://github.com/tenstorrent/tt-metal/pull/45990) | `blackhole-post-commit.yaml` | `verified-pass` | pass | Yes | Run 26907794205 completed failure but verified-pass; rebased to 275272621c5 2026-06-04T03:15Z |
| [#45981](https://github.com/tenstorrent/tt-metal/pull/45981) | `t3000-unit-tests.yaml` | `verified-pass` | pass | Yes | Run 26904737723 completed failure but verified-pass; rebased to 275272621c5; evidence refreshed to run 26913119973 2026-06-04T03:15Z |
| [#45979](https://github.com/tenstorrent/tt-metal/pull/45979) | `runtime-unit-tests.yaml` | `verified-pass` | pass | Yes | Run 26904326397 completed SUCCESS; rebased to 275272621c5; evidence refreshed to run 26920844625 2026-06-04T03:15Z |

---

## Active Runs

| Run | Pipeline | Branch | Started | Status | Notes |
|-----|----------|--------|---------|--------|-------|
| _(none)_ | | | | | |

**Policy:** Concurrent runs across PRs are allowed; each automation session may dispatch at most three new runs.

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-------|
| [26921420038](https://github.com/tenstorrent/tt-metal/actions/runs/26921420038) | `runtime-unit-tests.yaml` | `ci-disable/runtime-unit-tests-dispatch-telemetry-2026-06-03-verify` | 2026-06-04T00:14 UTC | 2026-06-04T00:41 UTC | **verified-pass** (SUCCESS) | All 4 SKUs success; DispatchTelemetryHostL1WaitTest.WorkerWaitReportsUpstreamBlockedState SKIPPED (fixture skip, test fixed upstream); PR #46023 then closed (test no longer failing on main) |
| [26919824610](https://github.com/tenstorrent/tt-metal/actions/runs/26919824610) | `runtime-unit-tests.yaml` | `ci-disable/runtime-unit-tests-dispatch-telemetry-2026-06-03-verify` | 2026-06-03T23:34 UTC | 2026-06-03T23:45 UTC | **verification-inconclusive** | load-test-matrix failed: `Verify test timeouts against budget` — dummy anchor entry missing `team` field; pytest never ran; budget NOT consumed; re-dispatched as run 26921420038 |
| [26917123038](https://github.com/tenstorrent/tt-metal/actions/runs/26917123038) | `models-t2-e2e-tests.yaml` | `ci-disable/models-t2-e2e-tests-vision-batch1-2026-06-03-verify` | 2026-06-03T22:31 UTC | 2026-06-03T23:03 UTC | **verified-pass** | conclusion: SUCCESS; Llama 3.2-90B-Vision batch1-trace test SKIPPED ("Disabled: see #46016"); 1 skipped 4 deselected |
| [26913761687](https://github.com/tenstorrent/tt-metal/actions/runs/26913761687) | `blackhole-post-commit.yaml` | `ci-disable/blackhole-post-commit-sdpa-tail-2026-06-03-verify` | 2026-06-03T21:21 UTC | 2026-06-03T22:26 UTC | **verified-pass** | conclusion: failure but no regression; all 8 test_sdpa_tail SKIPPED; remaining failures (chunked-trace) are pre-existing (same as on main); job was already failing on main before PR #46010 |
| [26910857102](https://github.com/tenstorrent/tt-metal/actions/runs/26910857102) | `runtime-unit-tests.yaml` | `ci-disable/runtime-unit-tests-descriptor-merger-2026-06-03-verify` | 2026-06-03T20:24 UTC | 2026-06-03T20:59 UTC | **verified-pass** | conclusion: failure but no regression; runtime_debug_tools was already failing on main (same job); disabled DescriptorMergerTests are SKIPPED; remaining failure is device hang in other tests (pre-existing) |
| [26908439442](https://github.com/tenstorrent/tt-metal/actions/runs/26908439442) | `blackhole-demo-tests.yaml` | `ci-disable/blackhole-demo-tests-mistral-2026-06-03-verify` | 2026-06-03T19:37 UTC | 2026-06-03T20:24 UTC | **verified-pass** | conclusion: success; mistral-small e2e tests [bh_loudbox] PASSED |
| [26907813785](https://github.com/tenstorrent/tt-metal/actions/runs/26907813785) | `runtime-integration-tests.yaml` | `ci-disable/runtime-integration-tests-indexed-fill-2026-06-03-verify` | 2026-06-03T19:25 UTC | 2026-06-03T20:19 UTC | **verified-pass** | conclusion: failure but no regression; runtime_fd_python_2 was already failing on main; D=4 disabled (SKIPPED), D=16 also fails with same underlying bug (pre-existing, masked by D=4 abort) |
| [26907794205](https://github.com/tenstorrent/tt-metal/actions/runs/26907794205) | `blackhole-post-commit.yaml` | `ci-disable/blackhole-post-commit-chunked-trace-2026-06-03-verify` | 2026-06-03T19:25 UTC | 2026-06-03T20:39 UTC | **verified-pass** | conclusion: failure but no regression; deepseek blitz op tests (AttributeError ttnn.FaceGeometry) were already failing on main before our PR |
| [26904737723](https://github.com/tenstorrent/tt-metal/actions/runs/26904737723) | `t3000-unit-tests.yaml` | `ci-disable/t3000-unit-tests-attention1d-qwen25-2026-06-03-verify` | 2026-06-03T18:27 UTC | 2026-06-03T20:21 UTC | **verified-pass** | conclusion: failure but no regression; t3k_tttv2_fast_unit_tests PASSED (target job, disabled tests SKIPPED); t3k_tt_metal_multiprocess_tests was already failing on main at base commit (0485c74b235) |
| [26904326397](https://github.com/tenstorrent/tt-metal/actions/runs/26904326397) | `runtime-unit-tests.yaml` | `ci-disable/runtime-unit-tests-data-movement-2026-06-03-verify` | 2026-06-03T18:19 UTC | 2026-06-03T19:15 UTC | **verified-pass** | Both runtime_data_movement jobs (wh_n150_civ2, bh_p150b_civ2) passed; 5 GTest cases now skipped |

---

## Blockers

| Blocker | Status | Notes |
|---------|--------|-------|
| _(none)_ | | |

---

## Recent Activity

- **2026-06-04T03:15 UTC** — Session (03:03 UTC start). Main advanced to 275272621c5 (from 54ccb0525804). Rebase carve-out: all 8 open verified-pass PRs rebased to 275272621c5 and force-pushed: #45979, #45981, #45990, #45991, #45993, #45999, #46010, #46017. Examining lane: all PRs throttled (<4h since 01:30Z last touch) — rebase-only per carve-out. Focus lane: pipeline survey conducted — t3000-perf (run 26923602225): SD3.5 and Motif AssertionError perf failures = 1 consecutive (not 3); QwenImage timeout (OOS); CNN resnet50 all pytest PASS (post-test perf-check failure, not a test ID — cannot disable); falcon7b same pattern. t3k_dits_tests: 2 consecutive completed failing runs (26913119973, 26923416535) with middle run 26917763630 build-failed/skipped — not 3 consecutive. BH e2e tests: device resets (OOS). BH post-commit (run 26922158322): chunked trace + sdpa_tail failures = already covered by PRs #45990 and #46010. Coverage-complete: all deterministic main failures still covered. Focus slots filled: 0/3 (coverage-complete). Examining PRs: 0 deep (all throttled) + 8 rebases. Orphaned-PR check skipped per Source of Truth policy. If you suspect orphaned automation PRs exist from prior broken sessions, perform a manual backfill before the next session.

- **2026-06-04T01:45 UTC** — Session (01:01 UTC start). Examining lane (3 PRs): (1) PR #46023 (runtime-unit-tests dispatch-telemetry): run 26921420038 classified `verified-pass` (SUCCESS; all 4 SKUs; test SKIPPED by fixture). HOWEVER, revalidation against latest main run 26920844625 (sha 5e2561dcd54) shows test now SKIPPED by fixture (`command_queue_fixture.hpp:34`) — no longer FAILING on main. Fixture fix landed (`msudumbrekar/fix-dispatch-telemetry-skip`). Per policy: removed disable, PR #46023 closed, issue #46022 closed (state: out-of-scope). (2) PR #45979 (data-movement): rebased to 54ccb0525804; all 5 tests still FAILING in run 26920844625 (sha 5e2561dcd54, 2026-06-03T23:59 UTC) jobs 79422903173/79422903291; PR comment posted; evidence updated; issue #45978 regenerated. (3) PR #45981 (t3000 attention1d): rebased to 54ccb0525804; both tests still FAILING in run 26913119973 (sha f0d2341794d, 2026-06-03T22:25 UTC) job 79401375600; PR comment posted; evidence updated; issue #45980 regenerated. Lightweight rebases (all other 6 verified-pass PRs): #45990, #45991, #45993, #45999, #46010, #46017 → cherry-pick to 54ccb0525804 via force-push. Focus lane: extensive pipeline survey — no new uncovered deterministic failures found. runtime_llk/runtime_sd_llk failures = device hangs (OOS); t3000-e2e, bh-e2e = device hangs (OOS); models-t2-e2e = infra/device timeouts (OOS); t3000-demo = all SUCCESS; t3000-perf DiT QwenImage = 2 consecutive (not 3); all other failures OOS or covered. Coverage-complete: all deterministic main failures covered by open PRs. Focus slots filled: 0/3 (coverage-complete).

- **2026-06-04T00:14 UTC** — Session (00:01 UTC start). Focus lane (1 PR, priority 3 - verification-inconclusive): PR #46023 (runtime-unit-tests dispatch-telemetry). Run 26919824610 classified `verification-inconclusive` — load-test-matrix step `Verify test timeouts against budget` failed because dummy anchor-definition entry `runtime_debug_tools_anchor_only` was missing mandatory `team` field; pytest never ran; budget NOT consumed. Fix applied: removed dummy entry, inlined `disabled_debug_tools_filter` value directly into `runtime_sd_debug_tools_device_dispatch` entry. Test `DispatchTelemetryHostL1WaitTest.WorkerWaitReportsUpstreamBlockedState` revalidated — still failing on all 4 SKUs in main run 26866206838. No main movement (main still at 5e2561dcd54; no rebase needed). Re-dispatched as run 26921420038 (fresh build; no SHA-matching successful source run). PR comment posted. Examining lane: 0 PRs (all verified-pass PRs throttled <4h; main unchanged; no carve-outs apply). Pipeline survey: blackhole-post-commit covered; t3000-unit-tests covered (device hangs OOS; t3k_dits_tests not 3 consecutive); t3000-e2e-tests device hangs OOS; runtime-integration-tests bh_multicard_eth 1-of-3 (not deterministic); no new uncovered deterministic failures found. Focus slots filled: 1/3 (only priority-3 re-dispatch; no priority-1 or priority-2 PRs).
- **2026-06-03T23:35 UTC** — Session (23:01 UTC start). Examining lane (2 PRs classified): (1) PR #46010 (blackhole-post-commit sdpa-tail): run 26913761687 classified `verified-pass` — all 8 test_sdpa_tail SKIPPED; chunked-trace failures pre-existing (covered by PR #45990); rebased to 5e2561dcd54; evidence updated to run 26912651491 job 79399422886; issue #46009 synced. (2) PR #46017 (models-t2-e2e vision batch1): run 26917123038 classified `verified-pass` — SUCCESS; test_multimodal_demo_text batch1-trace SKIPPED; rebased to 5e2561dcd54; PR comment posted; issue #46016 unchanged. All 6 other `verified-pass` PRs rebased to 5e2561dcd54 (cherry-pick method) and force-pushed: #45979, #45981, #45990, #45991, #45993, #45999. Focus lane (1 new PR, 1 dispatch): New PR #46023 for `runtime-unit-tests.yaml` — `DispatchTelemetryHostL1WaitTest.WorkerWaitReportsUpstreamBlockedState` failing on wh_n150_civ2, wh_n300_civ2, bh_p150b_civ2, bh_p100a_civ2_viommu across 3+ consecutive main runs (26805031959, 26834614992, 26866206838) with GTest failure. Disable: `GTEST_SKIP() << "Disabled: see #46022"` in test_dispatch_telemetry.cpp. Tracking issue #46022. Verification dispatched as run 26919824610 (fresh build; wh+bh; runtime_sd_debug_tools_device_dispatch only). CI survey: extensive survey of blackhole-e2e-tests, t3000-unit-tests, t3000-e2e-tests, runtime-integration-tests, Single-card demo tests, nightly L2 tests, Single-card frequent model tests — no additional uncovered deterministic failures found (all other failures are device hangs/timeouts, infra failures, or non-consecutive). Focus slots filled: 1/3 (only 1 workflow with uncovered deterministic test-level failures found after full survey).
- **2026-06-03T22:35 UTC** — Session (22:01 UTC start). Examining lane (3 PRs with rebase carve-out): (1) PR #45979 (runtime-unit-tests data-movement): rebased to `65a9acd34c9`, force-pushed; 5 GTest cases confirmed still failing on main run 26866206838; PR comment posted. (2) PR #45990 (blackhole-post-commit chunked-trace): rebased to `65a9acd34c9`, force-pushed; 3 tests still failing on main run 26901984996; PR comment posted. (3) PR #45999 (runtime-unit-tests descriptor-merger): rebased to `65a9acd34c9`, force-pushed; 10 DescriptorMergerTest cases still failing on main run 26866206838; PR comment posted. Focus lane (1 new PR): PR #46017 for `models-t2-e2e-tests.yaml` — disabled `test_multimodal_demo_text[wormhole_b0-device_params0-1-batch1-trace-normal-mesh_device0]` in `simple_vision_demo.py` (3 consecutive `AssertionError: Performance regression detected` failures on `wh_llmbox_perf`: runs 26863952291, 26798628102, 26735449307). Tracking issue #46016. Verification dispatched as run 26917123038 (fresh build; model=llama3.2-90b-vision, sku=wh_llmbox_perf). PR #46010 verification run 26913761687 still queued at session start — left as `verifying`. Pipeline survey: blackhole-e2e-tests FP8 dispatch failures have infra failures breaking consecutive streak (only 1 test-level failure); t3k multiprocess tests only 2 consecutive; all other surveyed pipelines either covered by existing PRs or have out-of-scope failures (timeouts, device crashes, Galaxy).
- **2026-06-03T21:30 UTC** — Session (21:00 UTC start). All 5 active verification runs completed. Examining lane (3 PRs): (1) PR #45981 (t3000-unit-tests): run 26904737723 classified `verified-pass` (t3k_tttv2_fast_unit_tests PASSED; t3k_tt_metal_multiprocess_tests was pre-existing failure on main). Rebased to f0d2341794d, force-pushed. Evidence updated to latest main run 26902490775 job 79365805746. PR comment + issue #45980 synced. (2) PR #45993 (blackhole-demo-tests): run 26908439442 classified `verified-pass` (SUCCESS). Rebased to f0d2341794d, force-pushed. PR comment posted. (3) PR #45991 (runtime-integration-tests): run 26907813785 classified `verified-pass` (D=4 SKIPPED, D=16 fails with same underlying bug but job was already failing on main). Rebased to f0d2341794d, force-pushed. PR comment posted. Lightweight classification (no rebase this session): PR #45990 → `verified-pass` (deepseek blitz op tests pre-existing failure), PR comment posted; PR #45999 → `verified-pass` (runtime_debug_tools pre-existing failure, disabled tests SKIPPED), PR comment posted. Focus lane (1 new PR): Discovered that `test_sdpa_tail[*]` in `models/demos/deepseek_v3_b1/tests/unit_tests/test_sdpa_tail.py` failing deterministically across 3 consecutive main runs (26901984996, 26887785807, 26876041545) with `AttributeError: module 'ttnn' has no attribute 'FaceGeometry'` — this was incorrectly classified as out-of-scope in previous session. Created tracking issue #46009. PR #46010 for `blackhole-post-commit.yaml`: disabled all 8 `test_sdpa_tail` parametrizations via `@pytest.mark.skip(reason="Disabled: see #46009")`. Verification dispatched as run 26913761687 (fresh build; run-ops-unit-tests=true, P150 only). Note: PR #45990 and PR #46010 both target `blackhole-post-commit.yaml` with non-overlapping test sets (chunked trace vs sdpa_tail). Note: `test_indexed_slice[DataType.BFLOAT16-16-32-6-0]` (D=16) newly discovered in verification run for PR #45991 — NOT eligible for disable yet (not showing deterministic failures on main, blocked by D=4 abort). Will become eligible after PR #45991 merges and D=16 appears in 3 consecutive main failures.
- **2026-06-03T20:30 UTC** — Session (20:02 UTC start). Examining lane: PR #45979 run 26904326397 completed SUCCESS → classified `verified-pass`. Branch rebased to ce03ba9ee12 and force-pushed. PR comment posted with verification result and revalidation log. Focus lane: 1 new PR created. PR #45999 for `runtime-unit-tests.yaml` `runtime_debug_tools` job: disabled 10 `DescriptorMergerTest` GTest cases failing with protobuf schema mismatch (`NodeDescriptor.Boards` has no field `tray_id`) deterministically on wh_n150_civ2 and bh_p150b_civ2 across 3 consecutive main runs. Tracking issue #45998. Verification dispatched as run 26910857102 (fresh build; no SHA-matching successful source run; pruned to runtime_debug_tools only). Note: extensive workflow survey conducted — all other failing pipelines (blackhole-e2e-tests, t3000-e2e-tests, blackhole-post-commit additional jobs, vllm-nightly T3K) are device timeouts/crashes (out of scope). Focus slots filled: 1/3 (only 1 workflow with uncovered deterministic pytest/GTest failures found after full survey). [NOTE: This session's "out of scope" classification of blackhole-post-commit additional jobs was PARTIALLY WRONG — test_sdpa_tail failures ARE in-scope (AttributeError, not timeout/crash). Corrected in next session.]
- **2026-06-03T19:45 UTC** — Session (19:00 UTC start). Examining lane: PR #45979 and PR #45981 both still in `verifying` state (runs 26904326397 and 26904737723 in_progress/queued, <4hr throttle applies). Focus lane: 3 new PRs created and dispatched. (1) PR #45990 for `blackhole-post-commit.yaml`: disabled 3 pytest tests (`test_run_host_io_decoder_sweep_chunked_trace_smoke/world_size_4/multi_slot_stress[ds-r1-0528-97854ebb-128k]`) failing with "chunked 128K trace not present" on `bh_p150b_civ2`. Tracking issue #45988. (2) PR #45991 for `runtime-integration-tests.yaml`: disabled `test_indexed_slice` D=4 parametrization causing device hang (TT_THROW @ system_memory_manager.cpp:738) on `wh_n150_civ2` and `bh_p150b_civ2`. Tracking issue #45989. (3) PR #45993 for `blackhole-demo-tests.yaml`: disabled 3 mistral-small-3.1-24b vision pipeline tests (`test_e2e_vision_text_pipeline`, `test_mistral_vision_model`, `test_mistral_vision_tower`) with IndexError/RuntimeError on `bh_loudbox`. Tracking issue #45992. Verification runs dispatched: 26907794205 (PR #45990), 26907813785 (PR #45991), 26908439442 (PR #45993).
- **2026-06-03T18:32 UTC** — PR #45981 created for `t3000-unit-tests.yaml` `t3k_tttv2_fast_unit_tests` job: disabled 2 deterministically-failing pytest parametrizations (`test_attention_1d_vs_reference[10-standard-1x2-decode-32-Qwen2.5-7B-1x2]` and `[10-paged-...]`). Fails on `wh_llmbox` only. Error: `TT_FATAL dims must be unique` in `concat_ndim`. Tracking issue #45980. Verification dispatched as run 26904737723 (fresh build; `model=tttv2 modules` filter to target only `t3k_tttv2_fast_unit_tests`). Note: `t3k_ttnn_tests [wh_llmbox]` SIGABRT failure is a device timeout/hang — **out of scope** per policy.
- **2026-06-03T18:25 UTC** — Session start. PR #45979 created for `runtime-unit-tests.yaml` `runtime_data_movement` job: disabled 5 deterministically-failing GTest cases (`TensixDirectWriteMulticast`, `TensixDataMovementOneToAllMulticastSemaphore2x2_2_0`, `TensixDataMovementOneToAllMulticastLinkedSemaphoreLoopback2x2_2_0`, `TensixDataMovementOneToAllMulticastLinkedSemaphore5x5_2_0`, `TensixDataMovementOneToAllUnicastSemaphore2x2_2_0`). All 5 fail on both `wh_n150_civ2` and `bh_p150b_civ2`. Tracking issue #45978. Verification dispatched as run 26904326397.

---

## PR #46010 — blackhole-post-commit.yaml (8 test_sdpa_tail disables)

| Field | Value |
|-------|-------|
| PR | [#46010](https://github.com/tenstorrent/tt-metal/pull/46010) |
| Disable issue | [#46009](https://github.com/tenstorrent/tt-metal/issues/46009) |
| Timeout issue | — |
| Branch | `ci-disable/blackhole-post-commit-sdpa-tail-2026-06-03` |
| Workflow file | `blackhole-post-commit.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-06-04T03:15Z (rebase to `275272621c5`) |
| Last revalidation | 2026-06-03T23:09Z (all 8 sdpa_tail confirmed still failing on main run 26912651491 job 79399422886; headSha `8574aad0975c2679670e5819f039f8630e062975`) |
| Verification run | [26913761687](https://github.com/tenstorrent/tt-metal/actions/runs/26913761687) — completed 2026-06-03T22:26 UTC; **verified-pass** (all 8 test_sdpa_tail SKIPPED; other failures pre-existing) |
| Last touched by automation | 2026-06-04T03:15Z |
| Readiness | **Ready to merge** — verified-pass |

### Disables (with main evidence)

Main-run evidence: see PR description.

Summary (all 8 tests fail on `bh_p150b_civ2` in run 26901984996, completed 2026-06-03T18:39 UTC, head SHA `58783674297daeef3ea1ac72240ea5c8acf72911`):

| Disabled test | SKUs failing | Job link |
|---|---|---|
| `test_sdpa_tail[0.07216878364870322-True-True-16-16-1-True]` | `bh_p150b_civ2` | [79361708858](https://github.com/tenstorrent/tt-metal/actions/runs/26901984996/job/79361708858) |
| `test_sdpa_tail[0.07216878364870322-True-True-16-8-2-False]` | `bh_p150b_civ2` | [79361708858](https://github.com/tenstorrent/tt-metal/actions/runs/26901984996/job/79361708858) |
| `test_sdpa_tail[0.07216878364870322-True-False-16-16-1-True]` | `bh_p150b_civ2` | [79361708858](https://github.com/tenstorrent/tt-metal/actions/runs/26901984996/job/79361708858) |
| `test_sdpa_tail[0.07216878364870322-True-False-16-8-2-False]` | `bh_p150b_civ2` | [79361708858](https://github.com/tenstorrent/tt-metal/actions/runs/26901984996/job/79361708858) |
| `test_sdpa_tail[0.07216878364870322-False-True-16-16-1-True]` | `bh_p150b_civ2` | [79361708858](https://github.com/tenstorrent/tt-metal/actions/runs/26901984996/job/79361708858) |
| `test_sdpa_tail[0.07216878364870322-False-True-16-8-2-False]` | `bh_p150b_civ2` | [79361708858](https://github.com/tenstorrent/tt-metal/actions/runs/26901984996/job/79361708858) |
| `test_sdpa_tail[0.07216878364870322-False-False-16-16-1-True]` | `bh_p150b_civ2` | [79361708858](https://github.com/tenstorrent/tt-metal/actions/runs/26901984996/job/79361708858) |
| `test_sdpa_tail[0.07216878364870322-False-False-16-8-2-False]` | `bh_p150b_civ2` | [79361708858](https://github.com/tenstorrent/tt-metal/actions/runs/26901984996/job/79361708858) |

---

## PR #45993 — blackhole-demo-tests.yaml (3 mistral-small-3.1-24b vision test disables)

| Field | Value |
|-------|-------|
| PR | [#45993](https://github.com/tenstorrent/tt-metal/pull/45993) |
| Disable issue | [#45992](https://github.com/tenstorrent/tt-metal/issues/45992) |
| Timeout issue | — |
| Branch | `ci-disable/blackhole-demo-tests-mistral-2026-06-03` |
| Workflow file | `blackhole-demo-tests.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-06-04T03:15Z (rebase to `275272621c5`) |
| Last revalidation | 2026-06-03T21:05 UTC (all 3 tests confirmed still failing on main run 26865834917) |
| Verification run | [26908439442](https://github.com/tenstorrent/tt-metal/actions/runs/26908439442) — completed SUCCESS 2026-06-03T20:24 UTC; mistral-small-3.1-24b e2e tests [bh_loudbox] PASSED |
| Last touched by automation | 2026-06-04T03:15Z |
| Readiness | **Ready to merge** — verified-pass |

### Disables (with main evidence)

Main-run evidence: see PR description.

Summary (all 3 tests fail on `bh_loudbox` [job 79230013581] in run 26865834917, completed 2026-06-03T06:27 UTC, head SHA `b7f9f3b5bc0ae4d7680cbf70a70ccd4abb58385a`):

| Disabled test | SKUs failing | Job link |
|---|---|---|
| `test_end2end.py::test_e2e_vision_text_pipeline[blackhole-mesh_device0-device_params0-accuracy-8192-1-page_params0-paged_attention-full]` | `bh_loudbox` | [79230013581](https://github.com/tenstorrent/tt-metal/actions/runs/26865834917/job/79230013581) |
| `test_vision_model.py::test_mistral_vision_model[blackhole-device_params0-mesh_device0]` | `bh_loudbox` | [79230013581](https://github.com/tenstorrent/tt-metal/actions/runs/26865834917/job/79230013581) |
| `test_vision_tower.py::test_mistral_vision_tower[blackhole-device_params0-mesh_device0]` | `bh_loudbox` | [79230013581](https://github.com/tenstorrent/tt-metal/actions/runs/26865834917/job/79230013581) |

---

## PR #45991 — runtime-integration-tests.yaml (test_indexed_slice D=4 disable)

| Field | Value |
|-------|-------|
| PR | [#45991](https://github.com/tenstorrent/tt-metal/pull/45991) |
| Disable issue | [#45989](https://github.com/tenstorrent/tt-metal/issues/45989) |
| Timeout issue | — |
| Branch | `ci-disable/runtime-integration-tests-indexed-fill-2026-06-03` |
| Workflow file | `runtime-integration-tests.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-06-04T03:15Z (rebase to `275272621c5`) |
| Last revalidation | 2026-06-03T21:05 UTC (test_indexed_slice D=4 confirmed still failing on main run 26868028159) |
| Verification run | [26907813785](https://github.com/tenstorrent/tt-metal/actions/runs/26907813785) — completed failure 2026-06-03T20:19 UTC; verified-pass (job was already failing on main; no regression) |
| Last touched by automation | 2026-06-04T03:15Z |
| Readiness | **Ready to merge** — verified-pass |

### Disables (with main evidence)

Main-run evidence: see PR description.

Summary (`test_indexed_slice[DataType.BFLOAT16-4-32-6-0]` fails on `wh_n150_civ2` [job 79237084600] and `bh_p150b_civ2` [job 79237084610] in run 26868028159, completed 2026-06-03T07:09 UTC, head SHA `24d9a02f0d25ae4db9f4c08a87804a7edc5f62b4`):

| Disabled test | SKUs failing | Job link |
|---|---|---|
| `test_indexed_fill.py::test_indexed_slice[DataType.BFLOAT16-4-32-6-0]` (D=4) | `wh_n150_civ2`, `bh_p150b_civ2` | [79237084600](https://github.com/tenstorrent/tt-metal/actions/runs/26868028159/job/79237084600) / [79237084610](https://github.com/tenstorrent/tt-metal/actions/runs/26868028159/job/79237084610) |

---

## PR #45990 — blackhole-post-commit.yaml (3 deepseek chunked trace test disables)

| Field | Value |
|-------|-------|
| PR | [#45990](https://github.com/tenstorrent/tt-metal/pull/45990) |
| Disable issue | [#45988](https://github.com/tenstorrent/tt-metal/issues/45988) |
| Timeout issue | — |
| Branch | `ci-disable/blackhole-post-commit-chunked-trace-2026-06-03` |
| Workflow file | `blackhole-post-commit.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-06-04T03:15Z (rebase to `275272621c5`) |
| Last revalidation | 2026-06-03T21:05 UTC (all 3 chunked trace tests confirmed still failing on main run 26901984996) |
| Verification run | [26907794205](https://github.com/tenstorrent/tt-metal/actions/runs/26907794205) — completed failure 2026-06-03T20:39 UTC; verified-pass (deepseek blitz op jobs were already failing on main; no regression from our change) |
| Last touched by automation | 2026-06-04T03:15Z |
| Readiness | **Ready to merge** — verified-pass |

### Disables (with main evidence)

Main-run evidence: see PR description.

Summary (all 3 tests fail on `bh_p150b_civ2` [jobs 79361708858 (slow) and 79361708947 (fast)] in run 26901984996, completed 2026-06-03T18:39 UTC, head SHA `58783674297daeef3ea1ac72240ea5c8acf72911`):

| Disabled test | SKUs failing | Job link |
|---|---|---|
| `test_host_io_decoder_sweep_chunked_trace.py::test_run_host_io_decoder_sweep_chunked_trace_smoke[ds-r1-0528-97854ebb-128k]` | `bh_p150b_civ2` | [79361708858](https://github.com/tenstorrent/tt-metal/actions/runs/26901984996/job/79361708858) |
| `test_host_io_decoder_sweep_chunked_trace.py::test_run_host_io_decoder_sweep_chunked_trace_world_size_4[ds-r1-0528-97854ebb-128k]` | `bh_p150b_civ2` | [79361708858](https://github.com/tenstorrent/tt-metal/actions/runs/26901984996/job/79361708858) |
| `test_host_io_decoder_sweep_chunked_trace.py::test_run_host_io_decoder_sweep_chunked_trace_multi_slot_stress[ds-r1-0528-97854ebb-128k]` | `bh_p150b_civ2` | [79361708858](https://github.com/tenstorrent/tt-metal/actions/runs/26901984996/job/79361708858) |

---

## PR #45981 — t3000-unit-tests.yaml (2 test_attention_1d_vs_reference Qwen2.5-7B disables)

| Field | Value |
|-------|-------|
| PR | [#45981](https://github.com/tenstorrent/tt-metal/pull/45981) |
| Disable issue | [#45980](https://github.com/tenstorrent/tt-metal/issues/45980) |
| Timeout issue | — |
| Branch | `ci-disable/t3000-unit-tests-attention1d-qwen25-2026-06-03` |
| Workflow file | `t3000-unit-tests.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-06-04T03:15Z (rebase to `275272621c5`) |
| Last revalidation | 2026-06-03T21:05 UTC (both tests confirmed still failing on main run 26902490775 job 79365805746) |
| Verification run | [26904737723](https://github.com/tenstorrent/tt-metal/actions/runs/26904737723) — completed failure 2026-06-03T20:21 UTC; verified-pass (t3k_tttv2_fast_unit_tests PASSED; t3k_tt_metal_multiprocess_tests was pre-existing failure on main) |
| Last touched by automation | 2026-06-04T03:15Z |
| Readiness | **Ready to merge** — verified-pass |

### Disables (with main evidence)

Main-run evidence: see PR description.

Summary (both tests fail on `wh_llmbox` [job 79365805746] in run 26902490775, completed 2026-06-03T20:57 UTC, head SHA `0485c74b235826c809cffe553ea4780a1c039e1f`):

| Disabled test | SKUs failing | Job link |
|---|---|---|
| `test_attention_1d_vs_reference[10-standard-1x2-decode-32-Qwen2.5-7B-1x2]` | `wh_llmbox` | [79365805746](https://github.com/tenstorrent/tt-metal/actions/runs/26902490775/job/79365805746) |
| `test_attention_1d_vs_reference[10-paged-1x2-decode-32-Qwen2.5-7B-1x2]` | `wh_llmbox` | [79365805746](https://github.com/tenstorrent/tt-metal/actions/runs/26902490775/job/79365805746) |

---

## PR #45999 — runtime-unit-tests.yaml (10 DescriptorMergerTest gtest disables)

| Field | Value |
|-------|-------|
| PR | [#45999](https://github.com/tenstorrent/tt-metal/pull/45999) |
| Disable issue | [#45998](https://github.com/tenstorrent/tt-metal/issues/45998) |
| Timeout issue | — |
| Branch | `ci-disable/runtime-unit-tests-descriptor-merger-2026-06-03` |
| Workflow file | `runtime-unit-tests.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-06-04T03:15Z (rebase to `275272621c5`) |
| Last revalidation | 2026-06-03T21:00 UTC (all 10 DescriptorMergerTest cases confirmed still failing on main run 26866206838) |
| Verification run | [26910857102](https://github.com/tenstorrent/tt-metal/actions/runs/26910857102) — completed failure 2026-06-03T20:59 UTC; verified-pass (runtime_debug_tools was already failing on main; disabled tests SKIPPED; remaining failure is device hang in other tests) |
| Last touched by automation | 2026-06-04T03:15Z |
| Readiness | **Ready to merge** — verified-pass |

### Disables (with main evidence)

Main-run evidence: see PR description.

Summary (all 10 tests fail on `wh_n150_civ2` [job 79231037025] and `bh_p150b_civ2` [job 79231037066] in run 26866206838, completed 2026-06-03T06:37 UTC, head SHA `15806d0d564eff581a6eb21bea8c56f0b35867d5`):

| Disabled test | SKUs failing | Job link |
|---|---|---|
| `DescriptorMergerTest.MergeXTorusAndYTorusIntoXYTorus` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037025](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037025) / [79231037066](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037066) |
| `DescriptorMergerTest.MergeBHXTorusAndBHYTorusIntoXYTorus` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037025](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037025) / [79231037066](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037066) |
| `DescriptorMergerTest.MergeTwoIdenticalXTorusDescriptors` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037025](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037025) / [79231037066](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037066) |
| `DescriptorMergerTest.MergeXYTorusWithXTorusDescriptors` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037025](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037025) / [79231037066](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037066) |
| `DescriptorMergerTest.SplitAndMerge8x16WhGalaxyXyTorusSuperpod` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037025](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037025) / [79231037066](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037066) |
| `DescriptorMergerTest.SplitAndMerge5WhGalaxyYTorusSuperpod` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037025](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037025) / [79231037066](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037066) |
| `DescriptorMergerTest.SplitAndMerge16N300Cluster` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037025](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037025) / [79231037066](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037066) |
| `DescriptorMergerTest.RejectGraphTemplatesWithDifferentChildren_ForwardPass` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037025](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037025) / [79231037066](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037066) |
| `DescriptorMergerTest.AllowCrossDescriptorConnectionsOnDifferentPorts` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037025](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037025) / [79231037066](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037066) |
| `DescriptorMergerTest.MergeExistingBHTorusDescriptors` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037025](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037025) / [79231037066](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037066) |

---

## PR #45979 — runtime-unit-tests.yaml (5 runtime_data_movement gtest disables)

| Field | Value |
|-------|-------|
| PR | [#45979](https://github.com/tenstorrent/tt-metal/pull/45979) |
| Disable issue | [#45978](https://github.com/tenstorrent/tt-metal/issues/45978) |
| Timeout issue | — |
| Branch | `ci-disable/runtime-unit-tests-data-movement-2026-06-03` |
| Workflow file | `runtime-unit-tests.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-06-04T03:15Z (rebase to `275272621c5`) |
| Last revalidation | 2026-06-04T01:30Z (all 5 tests confirmed still failing on main run 26920844625 jobs 79422903173/79422903291; sha `5e2561dcd54`; completed 2026-06-03T23:59 UTC) |
| Verification run | [26904326397](https://github.com/tenstorrent/tt-metal/actions/runs/26904326397) — completed SUCCESS 2026-06-03T19:15 UTC; both runtime_data_movement jobs passed |
| Last touched by automation | 2026-06-04T03:15Z |
| Readiness | **Ready to merge** — verified-pass |

### Disables (with main evidence)

Main-run evidence: see PR description.

Summary (all 5 tests fail on `wh_n150_civ2` [job 79231037011] and `bh_p150b_civ2` [job 79231037021] in run 26866206838, completed 2026-06-03T06:33–06:37 UTC, head SHA `15806d0d564eff581a6eb21bea8c56f0b35867d5`):

| Disabled test | SKUs failing | Job link |
|---|---|---|
| `GenericMeshDeviceFixture.TensixDirectWriteMulticast` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037011](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037011) / [79231037021](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037021) |
| `GenericMeshDeviceFixture.TensixDataMovementOneToAllMulticastSemaphore2x2_2_0` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037011](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037011) / [79231037021](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037021) |
| `GenericMeshDeviceFixture.TensixDataMovementOneToAllMulticastLinkedSemaphoreLoopback2x2_2_0` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037011](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037011) / [79231037021](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037021) |
| `GenericMeshDeviceFixture.TensixDataMovementOneToAllMulticastLinkedSemaphore5x5_2_0` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037011](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037011) / [79231037021](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037021) |
| `GenericMeshDeviceFixture.TensixDataMovementOneToAllUnicastSemaphore2x2_2_0` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037011](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037011) / [79231037021](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037021) |

---

## PR #46017 — models-t2-e2e-tests.yaml (test_multimodal_demo_text batch1-trace disable)

| Field | Value |
|-------|-------|
| PR | [#46017](https://github.com/tenstorrent/tt-metal/pull/46017) |
| Disable issue | [#46016](https://github.com/tenstorrent/tt-metal/issues/46016) |
| Timeout issue | — |
| Branch | `ci-disable/models-t2-e2e-tests-vision-batch1-2026-06-03` |
| Workflow file | `models-t2-e2e-tests.yaml` |
| Lifecycle stage | `verified-pass` |
| Last rebase | 2026-06-04T03:15Z (rebase to `275272621c5`) |
| Last revalidation | 2026-06-03T23:09Z (test_multimodal_demo_text batch1-trace confirmed still failing on main run 26863952291; no new models-t2-e2e-tests run since previous session) |
| Verification run | [26917123038](https://github.com/tenstorrent/tt-metal/actions/runs/26917123038) — completed SUCCESS 2026-06-03T23:03 UTC; **verified-pass** (test SKIPPED "Disabled: see #46016"; run green) |
| Last touched by automation | 2026-06-04T03:15Z |
| Readiness | **Ready to merge** — verified-pass |

### Disables (with main evidence)

Main-run evidence: see PR description.

Summary (`test_multimodal_demo_text[wormhole_b0-device_params0-1-batch1-trace-normal-mesh_device0]` fails on `wh_llmbox_perf` in run 26863952291, completed 2026-06-03T09:59 UTC, head SHA `bdc8c319d266cf19a08c3cfcba0fe6eef30c3e3b`):

| Disabled test | SKUs failing | Job link |
|---|---|---|
| `models/tt_transformers/demo/simple_vision_demo.py::test_multimodal_demo_text[wormhole_b0-device_params0-1-batch1-trace-normal-mesh_device0]` | `wh_llmbox_perf` | [79224197492](https://github.com/tenstorrent/tt-metal/actions/runs/26863952291/job/79224197492) |

---

## PR #46023 — runtime-unit-tests.yaml (DispatchTelemetryHostL1WaitTest.WorkerWaitReportsUpstreamBlockedState disable)

| Field | Value |
|-------|-------|
| PR | [#46023](https://github.com/tenstorrent/tt-metal/pull/46023) |
| Disable issue | [#46022](https://github.com/tenstorrent/tt-metal/issues/46022) |
| Timeout issue | — |
| Branch | `ci-disable/runtime-unit-tests-dispatch-telemetry-2026-06-03` |
| Workflow file | `runtime-unit-tests.yaml` |
| Lifecycle stage | `out-of-scope` |
| Last rebase | 2026-06-03T23:34Z (created off `origin/main` at `5e2561dcd54`) |
| Last revalidation | 2026-06-03T23:34Z (confirmed still failing on main run 26866206838 job 79231037073; all 4 SKUs failing) |
| Verification run | [26921420038](https://github.com/tenstorrent/tt-metal/actions/runs/26921420038) | `runtime-unit-tests.yaml` | `ci-disable/runtime-unit-tests-dispatch-telemetry-2026-06-03-verify` | 2026-06-04T00:14 UTC | 2026-06-04T00:41 UTC | **verified-pass** (SUCCESS) | All 4 SKUs success; DispatchTelemetryHostL1WaitTest.WorkerWaitReportsUpstreamBlockedState SKIPPED (fixture skip, test fixed upstream); PR #46023 then closed (test no longer failing on main) |
| [26919824610](https://github.com/tenstorrent/tt-metal/actions/runs/26919824610) — completed 2026-06-03T23:45 UTC; **verification-inconclusive** (load-test-matrix infra failure); → [26921420038](https://github.com/tenstorrent/tt-metal/actions/runs/26921420038) re-dispatched 2026-06-04T00:14 UTC (fresh build; fixed YAML) |
| Last touched by automation | 2026-06-04T01:15Z |
| Readiness | **Closed** — test fixed upstream, disable no longer needed |

### Disables (with main evidence)

Main-run evidence: see PR description.

Summary (`DispatchTelemetryHostL1WaitTest.WorkerWaitReportsUpstreamBlockedState` fails on all 4 SKUs in run 26866206838, completed 2026-06-03T05:45 UTC, head SHA `15806d0d564eff581a6eb21bea8c56f0b35867d5`):

| Disabled test | SKUs failing | Job link |
|---|---|---|
| `DispatchTelemetryHostL1WaitTest.WorkerWaitReportsUpstreamBlockedState` | `wh_n150_civ2`, `wh_n300_civ2`, `bh_p150b_civ2`, `bh_p100a_civ2_viommu` | [79231037111](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037111) / [79231037124](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037124) / [79231037073](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037073) / [79231037125](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037125) |
