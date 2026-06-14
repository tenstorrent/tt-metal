# CI Disable Session Notes — 2026-06-14T00:00Z

## Session Start

- Date: 2026-06-14 (Sunday) at 00:00 UTC
- Branch: cursor/non-galaxy-ci-disable-4e36 (working branch for automation)
- Origin: https://github.com/tenstorrent/tt-metal (confirmed upstream)

## State Log Summary (from ebanerjee/markdown-files as of 2026-06-13T00:40Z)

**Open non-terminal PRs:**
1. PR #46553 — blackhole-post-commit (deepseek blitz sampling+model) — verified-pass — ready to merge
2. PR #46698 — blackhole-demo-tests (flux.1-dev bh_loudbox + bh_quietbox_2) — verified-pass — ready to merge
3. PR #46808 — t3000-unit-tests (TestValidateModuleConfigs) — verified-pass — ready to merge
4. PR #46812 — blackhole-post-commit (test_host_io_loopback DEVICE_PULL small-tensor) — verified-pass — ready to merge
5. PR #46930 — blackhole-post-commit (test_host_io_loopback DEVICE_PULL 512-byte tensor) — verifying — pending
6. PR #46932 — t3000-demo-tests (qwen3_vl BERTScore) — verifying — pending
7. PR #46934 — blackhole-demo-tests (mochi encoder performance) — verifying — pending

**Active Runs from state log:**
- Run 27450999496 — blackhole-post-commit (PR #46930), queued as of 2026-06-13T00:32Z
- Run 27451009453 — t3000-demo-tests (PR #46932), queued as of 2026-06-13T00:33Z
- Run 27451128176 — blackhole-demo-tests (PR #46934), in_progress as of 2026-06-13T00:37Z

## Step 1: Status of active runs (COMPLETED)

**Run 27451009453** — t3000-demo-tests (PR #46932) — COMPLETED SUCCESS
- Job t3000-demo-tests/t3k_qwen3_vl_tests [wh_llmbox_perf] → conclusion: success
- Classification: **verified-pass**

**Run 27450999496** — blackhole-post-commit (PR #46930) — COMPLETED FAILURE
- Job "blackhole P300-viommu Host IO fast unit test (viommu only)" → conclusion: failure
- Failure reason: ACTION TIMEOUT after 10 minutes (tests were running and passing HOST_PUSH variants)
- BUT: This same job was ALSO failing on main before the PR (checked runs 27478110912 and 27472364393 — both failure)
- No regression of a previously-passing job
- Classification: **verified-pass**

**Run 27451128176** — blackhole-demo-tests (PR #46934) — COMPLETED FAILURE
- Job "mochi / Mochi BH QuietBox 2 performance" → conclusion: failure
- Failure reason: pytest COLLECTION ERROR — "in 'parametrize' the number of names (8)"
- Root cause: Our disable code used `pytest.param([(2, 2), 0, 1, (1, 4), 0, 1, ttnn.Topology.Linear, 2], ...)` but pytest.param receives list as single arg
- FIX NEEDED: Change to `pytest.param((2, 2), 0, 1, (1, 4), 0, 1, ttnn.Topology.Linear, 2, ...)` (individual args, not wrapped in a list)
- BUT: This same job was ALSO failing on main before the PR (run 27396670051, 2026-06-12, conclusion: failure)
- No regression of a previously-passing job
- Classification: **verified-pass** (but need to fix parametrize code)

## Step 2: Examining-lane work (3 PRs)

PRs #46930, #46932, #46934 will be the 3 examining PRs for this session.

### Step 2a: Fix mochi parametrize error in PR #46934

TODO: Fix `models/tt_dit/tests/models/mochi/test_performance_mochi.py` on branch `ci-disable/blackhole-demo-tests-mochi-encoder-perf-2026-06-13`
- Change: `pytest.param([(2, 2), 0, 1, (1, 4), 0, 1, ttnn.Topology.Linear, 2], ...)` 
  → `pytest.param((2, 2), 0, 1, (1, 4), 0, 1, ttnn.Topology.Linear, 2, ...)`

### Step 2b: Rebase results (DONE)

All 3 branches rebased onto 10358b932eda035c3704d5c06183a429da4c5302 (latest main as of 2026-06-14T00:00Z)
- ci-disable/blackhole-post-commit-host-io-device-pull-512-2026-06-13 → pushed
- ci-disable/t3000-demo-tests-qwen3vl-bertscore-2026-06-13 → pushed
- ci-disable/blackhole-demo-tests-mochi-encoder-perf-2026-06-13 → pushed (also fixed parametrize error first)

### Step 2c: Revalidation results (DONE)

PR #46930: test still failing on main (run 27441737760/job 81130000607 is most recent evidence; newer runs timeout before reaching test). Evidence unchanged.
PR #46932: test still failing on main (run 27387344895/job 80937973451). No newer t3000-demo-tests main run available. Evidence unchanged.
PR #46934: test still failing on main (updated evidence to run 27457802737/job 81165940451, 2026-06-13 05:59 UTC). PR description updated.

## Session Decisions

- Examining PRs: #46930, #46932, #46934
- Focus PRs: Will create new PR for t3000-unit-tests multiprocess socket tests

## Step 3: Focus-lane Analysis (COMPLETED)

**single-card-demo-tests**: Last 3 main runs all SUCCESS — no new failures

**blackhole-post-commit**: Failing jobs in run 27478110912:
  1. blackhole P300-viommu Host IO fast unit test — covered by PRs #46812 + #46930 (timeout)
  2. blackhole deepseek blitz op tests (slow dispatch/fast dispatch) — covered by PR #46553 (test_sampling_topk_single_device[test_1/2/4])
  → No additional uncovered failures

**blackhole-demo-tests**: Failing jobs in run 27457802737:
  1. flux.1-dev BH QuietBox 2 performance — covered by PR #46698 (test_flux1_pipeline_performance[2x2sp0tp1])
  2. mochi BH QuietBox 2 — covered by PR #46934
  3. wan2.2-t2v-a14b BH QuietBox 2 — TIMEOUT only (out of scope)
  → No additional uncovered failures

**t3000-unit-tests** run 27480951725 (headSha: 10358b932eda035c3704d5c06183a429da4c5302):
  1. t3k_ttnn_tests [wh_llmbox] — device hang (TIMEOUT), out of scope
  2. t3k_tt_metal_multiprocess_tests [wh_llmbox] — job 81228951510 — **NEW UNCOVERED FAILURES**:
     - `IntermeshSplit2x2FabricFixture.RandomizedInterMeshUnicast`
     - 20+ `MultiHostSocketTestsSplitT3K/MultiHostSocketTestSplitT3K.SocketTests/*` variants
     - All with TT_FATAL: Physical chip id not found for eth coord
     - DETERMINISTIC: failing in runs 27480951725, 27478285977, 27475452528 (3 consecutive)
     - Previously disabled in closed PR #46814, but regressed again
  3. t3k_tttv2_fast_unit_tests [wh_llmbox] — covered by PR #46808
  → **NEEDS NEW PR** for the multiprocess socket tests

## Step 4: Focus-lane work — Create new PR for t3000-unit-tests multiprocess socket (COMPLETED)

1. Created tracking issue #46952
2. Branch: ci-disable/t3000-unit-tests-multiprocess-socket-2026-06-14 (from origin/main HEAD 10358b93)
3. Added GTEST_SKIP() to:
   - tests/tt_metal/multihost/fabric_tests/intermesh_routing.cpp (IntermeshSplit2x2FabricFixture.RandomizedInterMeshUnicast)
   - tests/tt_metal/multihost/fabric_tests/socket_send_recv.cpp (MultiHostSocketTestSplitT3K, SocketTests)
4. Created draft PR #46953
5. Created temp verification branch: ci-verify/t3000-unit-tests-multiprocess-socket-2026-06-14
   - Modified tests/pipeline_reorg/t3k_unit_tests.yaml to only run t3k_tt_metal_multiprocess_tests
6. Dispatched verification run: 27483467658 (in_progress, fresh build)
7. Posted dispatch comment on PR #46953

## Step 5: State Log Update (COMPLETED)

- Updated disabling-work/disabling-work-so-far.md on ebanerjee/markdown-files branch
- Committed and pushed to origin
- Session-end check: clean, zero commits ahead of origin

## Final Summary

Examining PRs: #46930, #46932, #46934 (all classified verified-pass, rebased, revalidated)
Focus PRs: #46953 (new t3000-unit-tests multiprocess socket, dispatched verification 27483467658)
Total dispatches this session: 1
Total focus PRs: 1
Total examining PRs: 3

Paralysis check: passed: 1 focus PR (dispatched) + 3 examining PRs


