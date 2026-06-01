# CI Disable Session Log

## Session: 2026-06-01T06:02 UTC

### State at session start
- Origin confirmed: tenstorrent/tt-metal
- Read ci-disable-targeted-verification.md and disabling-work-so-far.md from ebanerjee/markdown-files branch

### Quick Index (from state log at session start)
- PR #45484 - L2 nightly - MERGED
- PR #45487 - BH demo tests - out-of-scope (closed)
- PR #45490 - Runtime unit tests - verified-pass (no disables remain, eligible for closure)
- PR #45492 - T3000 unit tests - MERGED
- PR #45494 - BH e2e tests - MERGED
- PR #45498 - microbenchmarks - out-of-scope (closed)
- PR #45500 - vllm-nightly - out-of-scope (closed)
- PR #45507 - T3000 e2e tests - out-of-scope (closed, zero disables)
- PR #45511 - Sanity tests - out-of-scope (closed)
- PR #45514 - Runtime integration tests - verified-fail (needs human review)
- PR #45529 - Model perf tests - MERGED
- PR #45676 - Runtime unit tests (MeshWatcher*) - verified-pass, last touched 2026-06-01T05:25
- PR #45678 - T3000 integration tests (deepseek vllm) - verified-pass, last touched 2026-06-01T03:05
- PR #45680 - T3000 unit tests (SocketTests) - verification-inconclusive (waiting for #45684)
- PR #45682 - T3000 unit tests (Accessor gtests) - verification-inconclusive, re-dispatched run 26735101071
- PR #45684 - T3000 unit tests (RandomizedInterMeshUnicast) - verifying, run 26735095737

### Active Runs to check
- Run 26735095737 (PR #45684) - dispatched 2026-06-01T04:25 UTC, fresh build t3000-unit-tests
- Run 26735101071 (PR #45682) - dispatched 2026-06-01T04:25 UTC, fresh build t3000-unit-tests

### Steps this session:
1. Check status of runs 26735095737 and 26735101071
2. Classify completed runs
3. Examine PRs in examining lane (up to 3, considering throttle)
4. Survey pipelines for new uncovered failing tests
5. Create new PRs if eligible (up to 3 focus slots)
6. Update state log and push


### Run Analysis (2026-06-01T06:XX UTC)
- Run 26735095737 (PR #45684 - RandomizedInterMeshUnicast): CLASSIFIED verified-pass
  - Target job t3k_tt_metal_multiprocess_tests [wh_llmbox] → SUCCESS
  - Failing: t3k_ttnn_tests (pre-existing), t3k_dits_tests/grok/qwen3 (ml_dtypes PyPI infra fault)
  - No regressions to previously-passing jobs
  - PR comment posted: https://github.com/tenstorrent/tt-metal/pull/45684#issuecomment-4590003080
  
- Run 26735101071 (PR #45682 - Accessor gtests): CLASSIFIED verified-pass
  - Accessor tests correctly SKIPPED "Disabled: see #45681" ✓
  - t3k_ttnn_tests failed at test_ccl_multi_cq_multi_device SIGABRT (pre-existing, masked on main)
  - t3k_ttnn_udm_tests, t3k_ttmetal_tests: ml_dtypes PyPI infra fault (not regressions)
  - PR comment posted: https://github.com/tenstorrent/tt-metal/pull/45682#issuecomment-4590003910

### PR #45680 Re-dispatch Assessment
- PR #45680 lifecycle: verification-inconclusive (SocketTests)
- Previous run poisoned by RandomizedInterMeshUnicast (PR #45684 not yet merged)
- If re-dispatch now: same problem will occur (RandomizedInterMeshUnicast still failing on main, not disabled in PR #45680)
- Decision: DO NOT re-dispatch PR #45680 yet - wait for PR #45684 to merge first
- PR #45680 remains verification-inconclusive, blocked pending PR #45684 merge

### Main Status
- Main HEAD: 97ca6204f5aa (unchanged since last session)
- Latest main t3000-unit-tests run 26737408013: in-progress (started 2026-06-01T05:42 UTC)


### Focus Slot 1: NEW PR #45688 (t3000-e2e-tests)
- Test: test_reduce_scatter_async_sharded_to_interleaved[...-fabric_ring-rs_input_shape2-...-HEIGHT_SHARDED-BufferType.L1-...-1link-mesh_device0]
- Failure: TT_FATAL @ mesh_trace.cpp:78: trace buffer 1605632B > allocated 1271456B
- 3 consecutive runs: 26624591673 (2026-05-29), 26677770587 (2026-05-30), 26706571044 (2026-05-31)
- Issue: #45687
- PR: #45688 (ci-disable/t3000-e2e-tests-reduce-scatter-async-20260601)
- Verification: run 26738723462 dispatched (fresh build, verify branch pruned to t3k_ccl_tests only)
- 1/3 focus dispatch slots used

### Pipeline Survey
- blackhole-post-commit: 3 consecutive failures, but all are PyPI/infra faults (ml_dtypes, graphviz) - NOT deterministic code failures. Run 26723452225 fails in models-unit-tests (P150) - need to check.
- t3000-e2e-tests: Covered by new PR #45688 (t3k_ccl_tests)
- single-card-demo-tests: non-consecutive, not eligible
- runtime-unit-tests: PR #45676 already covers active failures


### State Log Updates
- Updated PR #45684: verifying → verified-pass
- Updated PR #45682: verification-inconclusive → verified-pass
- Updated PR #45680: noted re-dispatch deferred pending #45684 merge
- Added PR #45688: new PR for t3000-e2e-tests reduce scatter async
- Updated Active Runs: removed 26735095737, 26735101071; added 26738723462
- Added to Recently Completed: 26735095737 (verified-pass), 26735101071 (verified-pass)
- Added PR #45688 section
- Added 2026-06-01T06:02 UTC session activity to Recent Activity
- State log committed and pushed to ebanerjee/markdown-files

### Session Summary
- Examining PRs: 2 (PR #45684, PR #45682 - both log-analyzed)
- Focus PRs dispatched: 1 (PR #45688 - new PR for t3000-e2e-tests)
- Paralysis check: passed: 1 focus PR (dispatched) + 2 examining PRs
