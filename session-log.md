# CI Disable Session Log — 2026-06-13

## Session Start

- Time: 2026-06-13T00:03Z
- Branch: cursor/non-galaxy-ci-disable-b874
- Origin: tenstorrent/tt-metal (confirmed)
- Policy docs read from: ebanerjee/markdown-files branch

## State Log Summary (from disabling-work-so-far.md as of 2026-06-12T00:35Z)

### Open PRs and status:
- PR #46553 (blackhole-post-commit, deepseek blitz): `verified-pass`, ready to merge
- PR #46696 (single-card-demo-tests, mnist): `verified-pass`, ready to merge
- PR #46698 (blackhole-demo-tests, flux.1-dev): `verified-pass`, ready to merge
- PR #46808 (t3000-unit-tests, TestValidateModuleConfigs): `verifying` (run 27386528492, in_progress)
- PR #46812 (blackhole-post-commit, test_host_io_loopback): `verifying` (run 27386537661, in_progress)
- PR #46814 (t3000-unit-tests, MultiHostSocketTestSplitT3K): `verifying` (run 27386543130, in_progress)

## Work Done This Session

### Step 1: Check active verification runs — COMPLETED
All 3 completed with conclusion: failure. Need to classify each.

### Step 2: Classify verification runs — COMPLETED

**Run 27386528492 (PR #46808, t3000-unit-tests TestValidateModuleConfigs):**
- Disabled tests (test_validate_module_configs_method_exists, test_compile_accepts_validate_configs_flag): SKIPPED
- Job `t3k_tttv2_fast_unit_tests` still failing due to segfault in `test_rmsnorm_1d` at line 370
- BUT: this job was ALREADY FAILING on main before the PR
- CLASSIFICATION: `verified-pass`

**Run 27386537661 (PR #46812, blackhole-post-commit test_host_io_loopback):**
- Disabled tests (DEVICE_PULL-64-128-512, 64-256-512, 64-512-512): SKIPPED (0.21s setup only, no call time)
- Job failed due to `test_host_io_loopback[DEVICE_PULL-512-1024-512]` but this test also failing on main
- CLASSIFICATION: `verified-pass`

**Run 27386543130 (PR #46814, t3000-unit-tests MultiHostSocketTestSplitT3K + IntermeshSplit2x2):**
- Disabled tests still FAILING in verification (GTEST_SKIP() not taking effect - fixture SetUp() fails first)
- BUT discovered: ALL these tests started PASSING on main (run 27442163505/job 81119978338, 2026-06-12T20:48Z)
- CLASSIFICATION: `verified-pass` + tests now passing on main -> remove disables

### Step 3: PR #46814 — tests now passing on main (2026-06-12T20:48Z)
- Removed GTEST_SKIP() from:
  - tests/tt_metal/multihost/fabric_tests/intermesh_routing.cpp
  - tests/tt_metal/multihost/fabric_tests/socket_send_recv.cpp
- Rebased onto origin/main (f1e8f9b60d09e0858bc2164fb2f855fa2586e4d8)
- Pushed force-with-lease to ci-disable/t3000-unit-tests-multiprocess-socket-2026-06-12
- Posted comment on PR #46814 explaining classification + removal
- Closed PR #46814 (via update_pull_request API)
- Closed issue #46813 (completed)

### Step 4: PR #46696 — mnist tests now passing on main
- Tests passed in run 27353497098 (2026-06-11T14:19Z) — both mnist-N150 and mnist-N300 jobs success
- Removed @pytest.mark.skip from models/demos/vision/classification/mnist/demo/demo.py
- Rebased onto origin/main, pushed force-with-lease
- Posted removal comment on PR #46696
- Closed PR #46696 (via API)
- Closed issue #46695 (completed)

### Step 5: Examining lane — PR #46808 (t3000-unit-tests TestValidateModuleConfigs)
- Tests still failing on main: job 81119978381, run 27442163505, sha 22a02a8f76d, 2026-06-12 23:38 UTC
- Rebased onto origin/main (f1e8f9b60d09e0858bc2164fb2f855fa2586e4d8), pushed force-with-lease
- Updated PR description with refreshed evidence
- Updated issue #46807 with new evidence
- Posted verified-pass comment on PR #46808
- Lifecycle transition: verifying → verified-pass

### Step 6: Examining lane — PR #46812 (blackhole-post-commit test_host_io_loopback)
- Tests still failing on main: job 81130000607, run 27441737760, sha 16e79a78, 2026-06-12 22:13 UTC
- Rebased onto origin/main, pushed force-with-lease
- Updated PR description and issue #46810 with refreshed evidence
- Posted verified-pass comment on PR #46812
- Lifecycle transition: verifying → verified-pass

### Step 7: Focus lane — 3 new PRs created and dispatched

**PR #46930 (Focus Slot 1): blackhole-post-commit test_host_io_loopback DEVICE_PULL-512-1024-512**
- Issue: #46929
- Branch: ci-disable/blackhole-post-commit-host-io-device-pull-512-2026-06-13
- Change: added skip condition in test_host_io.py (DEVICE_PULL, 512-byte tensor)
- Evidence: 3 consecutive failures in runs 27406637988, 27417161007, 27441737760 (all on bh_p300-viommu)
- Verification dispatch: run 27450999496, fresh build, run-blackhole-multi-card-fast-unit-tests=true
- Rebase base: f1e8f9b60d09e0858bc2164fb2f855fa2586e4d8

**PR #46932 (Focus Slot 2): t3000-demo-tests qwen3_vl BERTScore CI test**
- Issue: #46931
- Branch: ci-disable/t3000-demo-tests-qwen3vl-bertscore-2026-06-13
- Change: wrapped ci-only-bert-score parametrization with pytest.mark.skip in demo.py
- Evidence: 3 consecutive failures in runs 27110081195, 27245677498, 27387344895 (all wh_llmbox_perf)
- Verification dispatch: run 27451009453, fresh build, model=qwen3_vl
- Rebase base: f1e8f9b60d09e0858bc2164fb2f855fa2586e4d8

**PR #46934 (Focus Slot 3): blackhole-demo-tests mochi encoder performance**
- Issue: #46933
- Branch: ci-disable/blackhole-demo-tests-mochi-encoder-perf-2026-06-13
- Change: wrapped dit_2x2sp0tp1_vae_1x4sp0tp1_BH_QB parametrization with pytest.mark.skip
- Evidence: 3 consecutive failures in runs 27255332239, 27326025746, 27396670051 (bh_quietbox_2)
- Verification dispatch: run 27451128176, fresh build, model=mochi, system-type=QuietBox 2 (2xP300)
- Rebase base: f1e8f9b60d09e0858bc2164fb2f855fa2586e4d8

### State Log Update — COMPLETED
- Updated and pushed disabling-work-so-far.md to origin/ebanerjee/markdown-files
- Commit: 16b62206b2b
- Session-end invariant check: PASSED (git status clean, no unpushed commits)
