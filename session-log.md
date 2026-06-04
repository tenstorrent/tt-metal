# CI Disable Session Log — 2026-06-04T03:03 UTC

## Session start
- Time: 2026-06-04T03:03 UTC
- Branch: cursor/non-galaxy-ci-disable-c02c
- Origin: tenstorrent/tt-metal ✓

## Docs read
- ci-disable-targeted-verification.md: read fully ✓
- disabling-work-so-far.md: read fully ✓

## State from state log (Quick Index)
All PRs are `verified-pass` or `out-of-scope`:
- PR #46023 → runtime-unit-tests.yaml → out-of-scope (closed; test fixed upstream)
- PR #46017 → models-t2-e2e-tests.yaml → verified-pass (ready to merge)
- PR #46010 → blackhole-post-commit.yaml → verified-pass (ready to merge)
- PR #45999 → runtime-unit-tests.yaml → verified-pass (ready to merge)
- PR #45993 → blackhole-demo-tests.yaml → verified-pass (ready to merge)
- PR #45991 → runtime-integration-tests.yaml → verified-pass (ready to merge)
- PR #45990 → blackhole-post-commit.yaml → verified-pass (ready to merge)
- PR #45981 → t3000-unit-tests.yaml → verified-pass (ready to merge)
- PR #45979 → runtime-unit-tests.yaml → verified-pass (ready to merge)

All last touched: 2026-06-04T01:30Z (1.5h ago — THROTTLED, <4h)

No active runs.

Previous session (01:45 UTC) declared coverage-complete.

## Throttle analysis
All verified-pass PRs: last touched 01:30Z → 1.5h ago → throttled (< 4h)
No carve-outs apply (none are new/batch-committed-no-verify/verification-inconclusive).
Rebase carve-out: need to check if main has advanced beyond 54ccb0525804.

## Step 1: Check current main HEAD
- main was at 09a9c79f0980 at session start
- After git fetch: main advanced to 275272621c543e530845e82e664ca10c4bbdde18
- All 8 PR branches were on 54ccb0525804 — behind main → rebase carve-out applies

## Step 2: Pipeline survey for new uncovered failures
### New runs analyzed:
- blackhole-post-commit run 26922158322 (sha 2a7934186fac): 
  - chunked trace failures = covered by PR #45990
  - sdpa_tail failures = covered by PR #46010
  - → NO new uncovered failures
  
- t3000-unit-tests run 26923416535 (sha 54ccb0525804):
  - t3k_tttv2_fast_unit_tests: attention1d still FAILED — covered by PR #45981
  - t3k_dits_tests: test_clip_encoder FAILED (2 consecutive; run 26917763630 build-failed/skipped → middle run not counted) — NOT 3 consecutive
  - t3k_ttnn_tests: device hangs (OOS)
  - t3k_ttmetal_tests: device hangs (OOS)
  
- t3000-perf-tests run 26923602225 (sha 54ccb0525804):
  - t3k_LLM_falcon7b: all pytest PASS, post-test perf check fails → not a disableable test
  - t3k_DiT_SD3.5: 1 pytest FAILED (perf tolerance) — only 1 consecutive, not 3
  - t3k_CNN_resnet50: all pytest PASS, post-test perf check fails → not disableable
  - t3k_DiT_QwenImage: FAILED due to timeout (>720s) → OOS
  - t3k_DiT_Motif: 1 pytest FAILED (perf tolerance) — only 1 consecutive, not 3

- blackhole-e2e-tests run 26922178146 (sha 2a7934186fac):
  - All failures: device resets (eth core timeout) → OOS
  - fabric infra unit tests: GitHub action download timeout → infra failure, OOS

### Conclusion: coverage-complete — no new uncovered deterministic failures

## Step 3: Examining lane (throttled - only rebases per carve-out)
- All 8 PRs: last touched 01:30Z → 1.5h ago → throttled
- Main advanced → rebase carve-out applies
- Rebased all 8 PR branches to 275272621c5:
  - #46017 ci-disable/models-t2-e2e-tests-vision-batch1-2026-06-03 ✓
  - #46010 ci-disable/blackhole-post-commit-sdpa-tail-2026-06-03 ✓
  - #45999 ci-disable/runtime-unit-tests-descriptor-merger-2026-06-03 ✓
  - #45993 ci-disable/blackhole-demo-tests-mistral-2026-06-03 ✓
  - #45991 ci-disable/runtime-integration-tests-indexed-fill-2026-06-03 ✓
  - #45990 ci-disable/blackhole-post-commit-chunked-trace-2026-06-03 ✓
  - #45981 ci-disable/t3000-unit-tests-attention1d-qwen25-2026-06-03 ✓
  - #45979 ci-disable/runtime-unit-tests-data-movement-2026-06-03 ✓
- All pushed successfully

## Step 4: Focus lane
- Coverage-complete → no new PRs created
- Focus slots filled: 0/3

## State log
- Updated ebanerjee/markdown-files branch
- Committed: "ci-disable session 2026-06-04T03:15Z: rebase 8 verified-pass PRs to 275272621c5; coverage-complete"
- Pushed to origin successfully
- Session-end self-check: PASSED (clean status, 0 commits ahead)

## Session outcome
- Paralysis check: coverage-complete: 0 examining PRs (full work) + 8 rebases (no new PRs needed; 0 focus PRs in priorities 2/3)
- All PRs: verified-pass, ready to merge
- No new uncovered deterministic failures found
