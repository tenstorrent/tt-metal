---
phase: 07-multi-vc-test-cases-and-ci-wiring
plan: 01
subsystem: testing
tags: [yaml, vc2, fabric, multi-mesh, multi-worker, test-config]

requires:
  - phase: 06-vc2-sender-integration-end-to-end-verification
    provides: VC2 sender integration and e2e test passing with vc_id=2
provides:
  - 5 new YAML test entries covering multi-mesh VC0+VC1+VC2 and multi-worker concurrent VC scenarios
  - Multi-mesh device addressing patterns for cross-mesh VC1 traffic
  - Multi-worker stress test patterns with 4-6 concurrent senders
affects: [07-02-ci-wiring]

tech-stack:
  added: []
  patterns: [multi-mesh linear_chip_id addressing, multi-worker concurrent VC sender patterns]

key-files:
  created: []
  modified:
    - tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_vc2_at_least_2x2_mesh.yaml

key-decisions:
  - "Multi-mesh entries use [mesh_id, linear_chip_id] format matching test_dual_big_mesh_fabric_2d_sanity.yaml"
  - "Single-mesh multi-worker entries use [mesh_id, [row, col]] format matching existing VC2 YAML convention"

patterns-established:
  - "Multi-mesh VC test: use [mesh_id, linear_chip_id] device format with senders on mesh 0 and mesh 1"
  - "Multi-worker stress: 4-6 concurrent senders on mixed VCs between neighboring chips"

requirements-completed: [VC-01, VC-02, VC-03]

duration: 2min
completed: 2026-03-21
---

# Phase 7 Plan 1: Multi-VC Test Cases Summary

**5 new YAML test entries for multi-mesh VC0+VC1+VC2 cross-mesh traffic (3 entries) and multi-worker concurrent VC stress tests (2 entries, 4-6 senders)**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-21T20:18:33Z
- **Completed:** 2026-03-21T20:20:20Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Added 3 multi-mesh test entries (MultiMeshVC0VC1VC2CrossMesh, BidirectionalCrossMesh, VC2OnlyCrossMeshIntraMesh) with cross-mesh device addressing
- Added 2 multi-worker concurrent entries (FourSenderConcurrent, SixSenderStress) with 4 and 6 simultaneous senders on mixed VCs
- All 5 entries have enable_flow_control: true and use_vc2: true
- YAML validates cleanly; total test count now 15

## Task Commits

Each task was committed atomically:

1. **Task 1: Add multi-mesh VC0+VC1+VC2 and multi-worker test entries** - `f70b08af812` (feat)

## Files Created/Modified
- `tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_vc2_at_least_2x2_mesh.yaml` - Added 5 new test entries (219 lines) for multi-mesh and multi-worker VC2 scenarios

## Decisions Made
- Multi-mesh entries use `[mesh_id, linear_chip_id]` format (e.g., `[0, 0]`, `[1, 0]`) matching `test_dual_big_mesh_fabric_2d_sanity.yaml` convention
- Single-mesh multi-worker entries use `[mesh_id, [row, col]]` format (e.g., `[0, [0, 0]]`) matching existing VC2 YAML entries

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All VC2 test entries complete, ready for CI wiring in plan 07-02
- Multi-mesh entries require `tt-run` with `bh_4x4_multi_mesh_rank_binding.yaml` to exercise VC1

---
*Phase: 07-multi-vc-test-cases-and-ci-wiring*
*Completed: 2026-03-21*
