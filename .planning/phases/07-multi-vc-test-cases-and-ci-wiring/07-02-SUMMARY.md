---
phase: 07-multi-vc-test-cases-and-ci-wiring
plan: 02
subsystem: infra
tags: [ci, github-actions, tt-run, multi-mesh, vc2, fabric]

# Dependency graph
requires:
  - phase: 07-multi-vc-test-cases-and-ci-wiring plan 01
    provides: VC2 test YAML with multi-mesh and multi-worker entries
provides:
  - BH multi-card CI workflow entry for VC2 multi-mesh fabric testing
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: [tt-run multi-process CI entry with generate_rank_bindings.py]

key-files:
  created: []
  modified:
    - .github/workflows/blackhole-multi-card-unit-tests-impl.yaml

key-decisions:
  - "test-type: slow for VC2 CI entry to avoid blocking PRs during stabilization"
  - "No TT_METAL_ENABLE_FABRIC_VC2 env var -- test harness reads use_vc2 from YAML"

patterns-established:
  - "VC2 CI pattern: generate_rank_bindings.py + tt-run with bh_4x4_multi_mesh_rank_binding.yaml"

requirements-completed: [CI-01]

# Metrics
duration: 1min
completed: 2026-03-21
---

# Phase 7 Plan 02: CI Wiring Summary

**VC2 multi-mesh fabric test wired into BH multi-card CI workflow as slow nightly entry using tt-run with bh_4x4_multi_mesh_rank_binding.yaml**

## Performance

- **Duration:** 1 min
- **Started:** 2026-03-21T20:22:01Z
- **Completed:** 2026-03-21T20:22:44Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Added VC2 multi-mesh fabric test entry to BH multi-card CI workflow
- Entry runs generate_rank_bindings.py then tt-run with bh_4x4_multi_mesh_rank_binding.yaml
- Set as test-type: slow for nightly CI (not fast-only PRs) during stabilization

## Task Commits

Each task was committed atomically:

1. **Task 1: Add VC2 multi-mesh test entry to BH multi-card CI workflow** - `ff7ccc7def7` (feat)

**Plan metadata:** [pending] (docs: complete plan)

## Files Created/Modified
- `.github/workflows/blackhole-multi-card-unit-tests-impl.yaml` - Added VC2 multi-mesh fabric test entry with tt-run multi-process invocation

## Decisions Made
- test-type: slow chosen to avoid blocking PRs while the VC2 multi-mesh test stabilizes on real BH galaxy hardware
- No TT_METAL_ENABLE_FABRIC_VC2 env var used -- the test harness reads use_vc2: true from the YAML config and enables VC2 programmatically

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- VC2 CI coverage complete -- multi-mesh test runs nightly on BH galaxy hardware
- All Phase 7 plans complete (test YAML entries + CI wiring)

---
*Phase: 07-multi-vc-test-cases-and-ci-wiring*
*Completed: 2026-03-21*
