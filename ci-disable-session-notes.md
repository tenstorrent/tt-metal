# CI Disable Session Notes — 2026-06-01T11:00 UTC

## Session Start State

### Current time: 2026-06-01T11:00 UTC
### State log last updated: 2026-06-01T10:20 UTC

### Open PRs (from state log Quick Index):
| PR | Workflow | Lifecycle | Last touched |
|----|----------|-----------|-------------|
| #45514 | runtime-integration-tests | verified-fail | 2026-06-01T07:15 UTC |
| #45676 | runtime-unit-tests | verified-pass | 2026-06-01T10:00 UTC |
| #45678 | t3000-integration-tests | verified-pass | 2026-06-01T09:15 UTC |
| #45680 | t3000-unit-tests | verification-inconclusive | 2026-06-01T09:15 UTC (carve-out) |
| #45682 | t3000-unit-tests | verified-pass | 2026-06-01T06:10 UTC |
| #45684 | t3000-unit-tests | verified-pass | 2026-06-01T06:10 UTC |
| #45688 | t3000-e2e-tests | verified-pass | 2026-06-01T08:15 UTC |
| #45690 | runtime-unit-tests | verified-pass | 2026-06-01T09:01 UTC |
| #45700 | t3000-e2e-tests | verifying | 2026-06-01T09:36 UTC |
| #45704 | blackhole-e2e-tests | verifying | 2026-06-01T10:18 UTC |

### Active Runs:
- Run 26748971279 (PR #45704, blackhole-e2e-tests): dispatched 10:18 UTC (~42min ago) 
- Run 26747001669 (PR #45700, t3000-e2e-tests): dispatched 09:36 UTC (~84min ago)

### Throttle status:
- #45676: touched 10:00 UTC → 1h ago → THROTTLED
- #45678: touched 09:15 UTC → 1h45m ago → THROTTLED
- #45680: verification-inconclusive → CARVE-OUT (never throttled)
- #45682: touched 06:10 UTC → 4h50m ago → ELIGIBLE (>4h)
- #45684: touched 06:10 UTC → 4h50m ago → ELIGIBLE (>4h)
- #45688: touched 08:15 UTC → 2h45m ago → THROTTLED
- #45690: touched 09:01 UTC → 2h ago → THROTTLED
- #45514: touched 07:15 UTC → 3h45m ago → THROTTLED (just under 4h)
- #45700: touched 09:36 UTC → verifying → not a carve-out → THROTTLED
- #45704: touched 10:18 UTC → verifying → not a carve-out → THROTTLED

## Session Work Plan
1. Check if active runs 26748971279 and 26747001669 have completed
2. Check if PR #45684 has merged (would unblock PR #45680 re-dispatch)
3. Examining lane (up to 3):
   - PR #45680 (carve-out, verification-inconclusive)
   - PR #45682 (>4h, verified-pass — rebase + revalidate)
   - PR #45684 (>4h, verified-pass or merged — rebase + revalidate if not merged)
4. Focus lane: 
   - Check for new uncovered workflows with deterministic failures
   - If PR #45684 merged → PR #45680 re-dispatch eligible (priority-3)
5. Update state log

## Work Done This Session

## Work Completed This Session (2026-06-01T11:00 UTC)

### Run Classifications
- Run 26747001669 (PR #45700, t3000-e2e-tests, t3k_ccl_tests): **verified-pass** (completed 2026-06-01T10:50 UTC)
  - Disabled test `test_reduce_scatter_async_training_shapes[...-tt_training_test_six-perf-...]`: correctly SKIPPED ✓
  - 4 "new" failures (`fabric_linear` variants + `tt_training_test_nine-check`) = unmasked cascade-ERRORs from main (not previously passing)
  - `models_tttv2_mlp_modules_full_set_unit_tests` failure = PyPI network infra fault
  - Budget consumed. Lifecycle → verified-pass. PR comment posted on #45700.

### Examining Lane (3 PRs)

**PR #45682 (Accessor gtests, verified-pass)**:
- Rebased to bf2870164f6 (new head b2521b04bef), no conflicts
- Revalidated: accessor tests still FAILING in main run 26747922051/job 78829379467 (2026-06-01T10:43 UTC)
- PR description + issue #45681 updated with new job links
- PR comment posted

**PR #45684 (RandomizedInterMeshUnicast, out-of-scope/CLOSED)**:
- Rebased to bf2870164f6 (new head 04f40d4f91a initially, then b5912bf4f8b after disable removal)
- Revalidated: `IntermeshSplit2x2FabricFixture.RandomizedInterMeshUnicast` PASSING on main run 26747922051/job 78829379476 ([OK] 23757ms, 23777ms)
- REMOVED disable from branch: StrReplace in intermesh_routing.cpp, commit b5912bf4f8b, pushed
- PR description updated (zero disables), issue #45683 CLOSED, PR comment posted
- PR #45684 CLOSED (zero disables). Lifecycle → out-of-scope

**PR #45680 (SocketTests, carve-out, out-of-scope/CLOSED)**:
- Rebased to bf2870164f6 (new head 26e5a28022c initially, then d7c9892a346 after disable removal)
- Revalidated: `MultiHostSocketTestsSplitT3K.SocketTests` PASSING on main run 26747922051/job 78829379476 (all 20 parametrizations [OK])
- REMOVED disable from branch: StrReplace in socket_send_recv.cpp, commit d7c9892a346, pushed
- PR description updated (zero disables), issue #45679 CLOSED, PR comment posted
- PR #45680 CLOSED (zero disables). Lifecycle → out-of-scope

### Main Branch Advance
- Main advanced from 97ca6204f5aa5d6dbee6fe39da6bd468e4ef42d7 → bf2870164f6ce3c6410ecede4bf29215b0923860

### Active Runs Status
- Run 26748971279 (PR #45704, blackhole-e2e-tests): still queued at 11:00 UTC (dispatched 10:18 UTC)

### Focus Lane
- 0/3 dispatches this session
- Surveyed: t3000-e2e-tests (latest completed 26742897792, in_progress 26747352356), t3000-integration-tests (latest 26728843280), runtime-unit-tests (latest 26737509560), blackhole-post-commit (latest 26747343891 = SUCCESS), blackhole-e2e-tests (latest 26717955997)
- No new uncovered workflows found

### State Log Update
- Committed and pushed to ebanerjee/markdown-files branch (commit efc64a4e6de)
- Session-end self-check: CLEAN (no uncommitted/unpushed state log changes)

### Session End Result
- Paralysis check: passed: 0 focus PRs (0 dispatches; no uncovered workflows; no priority-2/3) + 3 examining PRs
