# CI Disable Session Log — 2026-06-03T20:02 UTC

## Session Start

- Confirmed origin = tenstorrent/tt-metal (upstream)
- Read policy: disabling-work/ci-disable-targeted-verification.md (from ebanerjee/markdown-files)
- Read state log: disabling-work/disabling-work-so-far.md (from ebanerjee/markdown-files)
- State log last updated: 2026-06-03T19:45 UTC

## Tracked PRs at Session Start

| PR | Workflow | Lifecycle | Verification Run | Last Touched |
|----|----------|-----------|-----------------|--------------|
| #45993 | blackhole-demo-tests.yaml | verifying | 26908439442 (queued 19:37) | 19:45Z |
| #45991 | runtime-integration-tests.yaml | verifying | 26907813785 (queued 19:25) | 19:45Z |
| #45990 | blackhole-post-commit.yaml | verifying | 26907794205 (queued 19:25) | 19:45Z |
| #45981 | t3000-unit-tests.yaml | verifying | 26904737723 (started 18:27) | 18:32Z |
| #45979 | runtime-unit-tests.yaml | verifying | 26904326397 (started 18:19) | 18:25Z |

## Run Status Check Results

- Run 26904326397 (PR #45979, runtime-unit-tests.yaml): **COMPLETED - conclusion: success** at 19:15:51Z
- Run 26904737723 (PR #45981, t3000-unit-tests.yaml): still queued
- Run 26907794205 (PR #45990, blackhole-post-commit.yaml): still queued
- Run 26907813785 (PR #45991, runtime-integration-tests.yaml): still queued
- Run 26908439442 (PR #45993, blackhole-demo-tests.yaml): in_progress

## PR #45979 Examining Lane

- Run 26904326397 completed SUCCESS
- Both runtime_data_movement jobs (wh_n150_civ2, bh_p150b_civ2) passed → **verified-pass**
- Branch rebased from 0485c74b235 → ce03ba9ee12 (force-pushed)
- PR comment posted (#4616368582)
- State log updated: lifecycle = verified-pass

## Workflow Survey for Focus Lane

### Out-of-scope failures found (device timeouts/crashes/infra):
- blackhole-e2e-tests.yaml: fabric sanity benchmark (custom binary), bh_qb_DeepSeek_PREFILL (SIGSEGV exit 139), bh_lb_DeepSeek_PREFILL (ethernet core timeout)
- t3000-e2e-tests.yaml: t3k_DeepSeek_PREFILL device timeout (TT_THROW @ system_memory_manager.cpp:738)
- vllm-nightly-tests.yaml [WH-T3K] Gemma3-27B: TT_FATAL but custom framework, not standard pytest
- blackhole-post-commit.yaml additional: models-unit-tests timeout (20min)
- test_sdpa_tail failures: only appeared in 1 of 3 runs (not deterministic yet)
- runtime_llk failures: no GTest FAILED markers (likely hang/segfault before test output)

### Workflows with NO current failures:
- single-card-demo-tests.yaml: most recent run = SUCCESS
- tt-metal-l2-nightly.yaml: most recent run = SUCCESS
- t3000-demo-tests.yaml: most recent 3 runs = SUCCESS
- t3000-integration-tests.yaml: most recent run = SUCCESS
- fast-dispatch-full-regressions-and-models.yaml: all recent SUCCESS

### New actionable failure found:
- runtime-unit-tests.yaml / runtime_debug_tools job:
  - 10 DescriptorMergerTest GTest cases failing deterministically on wh_n150_civ2 AND bh_p150b_civ2
  - 3+ consecutive main runs: 26866206838, 26834614992, 26805031959
  - Error: protobuf schema mismatch (NodeDescriptor.Boards.tray_id field removed)
  - Test file: tools/tests/scaleout/test_descriptor_merger.cpp
  - Failure is hardware-independent → unconditional GTEST_SKIP() appropriate

## Focus Lane: PR #45999

- Created GitHub issue #45998
- Created branch ci-disable/runtime-unit-tests-descriptor-merger-2026-06-03
- Added GTEST_SKIP() << "Disabled: see #45998" to 10 test bodies:
  1. DescriptorMergerTest.MergeXTorusAndYTorusIntoXYTorus (line 392)
  2. DescriptorMergerTest.MergeBHXTorusAndBHYTorusIntoXYTorus (line 417)
  3. DescriptorMergerTest.MergeTwoIdenticalXTorusDescriptors (line 441)
  4. DescriptorMergerTest.MergeXYTorusWithXTorusDescriptors (line 466)
  5. DescriptorMergerTest.SplitAndMerge8x16WhGalaxyXyTorusSuperpod (line 495)
  6. DescriptorMergerTest.SplitAndMerge5WhGalaxyYTorusSuperpod (line 531)
  7. DescriptorMergerTest.SplitAndMerge16N300Cluster (line 565)
  8. DescriptorMergerTest.RejectGraphTemplatesWithDifferentChildren_ForwardPass (line 655)
  9. DescriptorMergerTest.AllowCrossDescriptorConnectionsOnDifferentPorts (line 750)
  10. DescriptorMergerTest.MergeExistingBHTorusDescriptors (line 1113)
- Committed and pushed disable branch
- Created draft PR #45999
- Created verification branch ci-disable/runtime-unit-tests-descriptor-merger-2026-06-03-verify
  - Pruned tests/pipeline_reorg/runtime_unit_tests.yaml to only runtime_debug_tools job
  - Pushed to origin
- No SHA-matching successful main run for runtime-unit-tests.yaml → fresh build
- Dispatched verification run 26910857102 (fresh build)
- PR comment posted (#4616415686)

## State Log Update

- Updated disabling-work/disabling-work-so-far.md on ebanerjee/markdown-files
- Committed (ac9a4a3fdbb) and pushed to origin/ebanerjee/markdown-files

## Session End Invariants

- git status CLEAN on ebanerjee/markdown-files
- git log shows 0 commits ahead of origin
- Self-check PASSED — no BROKEN SESSION

## Session Summary

- Examining PRs: 1 (PR #45979 → verified-pass)
- New PRs created: 1 (PR #45999 for runtime-unit-tests.yaml DescriptorMergerTest)
- Total dispatches: 1 (run 26910857102)
- Focus slots filled: 1/3
  Reason: Only 1 workflow (runtime-unit-tests.yaml runtime_debug_tools job) had uncovered deterministic failures with standard disableable GTest/pytest markers. All other failing pipelines are device timeouts/crashes/infra issues (out of scope).
- Paralysis check: limited: 1 focus PR (only 1 workflow-with-uncovered-failing-tests + 0 priority-2/3 PRs available)

## Orphaned-PR check
Orphaned-PR check skipped per Source of Truth policy. If you suspect orphaned automation PRs exist from prior broken sessions, perform a manual backfill before the next session.
