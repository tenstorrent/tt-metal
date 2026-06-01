# CI Disable Session Log — 2026-06-01 (non-Galaxy automation)

## Session started: 2026-06-01 ~02:15 UTC

### State read from ebanerjee/markdown-files
- Last session: 2026-05-29T15:00 UTC
- ~2 days have passed since last session
- main HEAD: 97ca6204f5a (fetched 2026-06-01 ~02:20 UTC)
- Previous session base: ~e4ef235d2cef (2026-05-29)

---

## PR state check (2026-06-01 ~02:20 UTC)

MERGED since last session:
- #45484 (Nightly L2 tests, MeshDeviceFixture.Top32RmDevPipelineCompletes) → MERGED
- #45492 (T3000 unit tests, MultiMeshEastMulticast_*) → MERGED
- #45494 (Blackhole e2e tests, test_all_to_all_combine_no_trace) → MERGED
- #45529 (Model perf tests, swin_s test_e2e_performant) → MERGED

CLOSED (not merged) since last session:
- #45487 (BH demo tests, whisper distil-large-v3) → CLOSED 2026-05-30T12:42Z
  Reason: human reviewer "Don't disable. test passed on third attempt" — flaky, non-deterministic
- #45498 (microbenchmarks, T3K Fabric Mux BW) → CLOSED 2026-05-30T12:59Z; reason TBD
- #45507 (T3000 e2e tests) → CLOSED (was zero disables, eligible for closure — auto-closed)
- #45511 (Sanity tests, TestPerfCountersSingleOp) → CLOSED 2026-05-30T12:59Z; reason TBD

STILL OPEN:
- #45490 (Runtime unit tests, DPrintMeshFixture) → OPEN, eligible for closure (all disables removed)
- #45514 (Runtime integration tests, test_indexed_slice seed=0) → OPEN, verified-fail

---

## Examining lane (2 PRs — session capacity 3, only 2 open)

### #45490 examination
- Branch: ci-disable/runtime-unit-tests-dprint-activeeth-20260529
- Status: verified-pass, eligible for closure (all disables removed)
- Action needed: rebase, confirm still eligible for closure, close PR
- TODO: run git operations

### #45514 examination
- Branch: ci-disable/runtime-integration-tests-indexed-fill-20260529
- Status: verified-fail; test_indexed_slice[DataType.BFLOAT16-4-32-6-0] seed=0 failing on main
- Action needed: rebase, revalidate seed=0 evidence, post comment
- Note: verified-fail because seed=42 was also failing after disable (cascading). Needs human review.
- TODO: check latest runtime-integration-tests main run

---

## Focus lane survey (in progress)

### Workflow surveys (2026-06-01 ~02:20 UTC)
| Workflow | Recent run results | Eligible? |
|----------|-------------------|-----------|
| blackhole-post-commit.yaml | fail,fail,success (runs 26728775039, 26723452225 fail, 26717945858 success) | No — 2 consecutive |
| single-card-demo-tests.yaml | TBD | TBD |
| t3000-unit-tests.yaml | TBD | TBD |
| t3000-e2e-tests.yaml | TBD | TBD |
| runtime-unit-tests.yaml | TBD | TBD |
| runtime-integration-tests.yaml | TBD | TBD |
| t3000-demo-tests.yaml | TBD | TBD |
| t3000-integration-tests.yaml | TBD | TBD |
| metal-run-microbenchmarks.yaml | TBD | TBD |
| blackhole-demo-tests.yaml | TBD | TBD |
| sanity-tests.yaml | TBD | TBD |
| tt-metal-l2-nightly.yaml | TBD | TBD |

---

## State log updates needed (for end of session)
- #45484: update to merged
- #45487: update to closed (flaky, not deterministic)
- #45490: rebase + close
- #45492: update to merged
- #45494: update to merged
- #45498: update to closed
- #45500: already out-of-scope/closed
- #45507: already closed in state log (zero disables, eligible for closure)
- #45511: update to closed
- #45514: rebase + update evidence
- #45529: update to merged
