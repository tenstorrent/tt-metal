---
phase: 01-overlay-register-cleanup
plan: 02
subsystem: fabric
tags: [stream-id, template-parameter, worker-adapter, edm]

# Dependency graph
requires:
  - phase: 01-overlay-register-cleanup
    provides: "StreamRegAssignments sub-structs (plan 01)"
provides:
  - "WorkerToFabricEdmSenderBase template with configurable STREAM_ID parameter"
  - "WorkerToFabricEdmSenderImpl using alias preserving backward compatibility"
affects: [05-vc2-sender-wiring]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Template parameter for stream ID decoupling from hardcoded constants"]

key-files:
  created: []
  modified:
    - tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp

key-decisions:
  - "Added static_assert(STREAM_ID <= 31) for compile-time range validation"

patterns-established:
  - "Stream ID as template parameter with default from connection_interface constant"
  - "Base template + using alias pattern for backward-compatible template extension"

requirements-completed: [SREG-02]

# Metrics
duration: 3min
completed: 2026-03-17
---

# Phase 01 Plan 02: Templatize Worker Adapter Summary

**WorkerToFabricEdmSenderBase template with STREAM_ID parameter (default 22) and backward-compatible using alias for all existing callers**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-17T02:13:20Z
- **Completed:** 2026-03-17T02:16:23Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Renamed WorkerToFabricEdmSenderImpl struct to WorkerToFabricEdmSenderBase with new STREAM_ID template parameter
- ACTIVE_ETH fallback path now uses STREAM_ID template parameter instead of hardcoded constant
- Added using alias WorkerToFabricEdmSenderImpl that forwards first two template params with default stream ID
- All 14+ existing callers continue to compile unchanged via the using alias
- Added static_assert for stream ID range validation (0-31)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add stream ID template parameter to adapter** - `26801e0a34a` (feat)

**Plan metadata:** [pending] (docs: complete plan)

## Files Created/Modified
- `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp` - Renamed struct to WorkerToFabricEdmSenderBase, added STREAM_ID template param, added using alias

## Decisions Made
- Added static_assert(STREAM_ID <= 31) inside struct body for compile-time validation (optional per CONTEXT.md, included for safety)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-existing build error in `erisc_datamover_builder.cpp:1025` (missing `get_all_stream_ids` member) prevents full `fabric_unit_tests` build. This is unrelated to plan 01-02 changes and was confirmed to exist on the base branch. Logged to `deferred-items.md`.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- WorkerToFabricEdmSenderBase is ready for VC2 to instantiate with STREAM_ID=30 in Phase 5
- All existing code paths continue using stream ID 22 via the default template parameter

## Self-Check: PASSED

All artifacts verified:
- edm_fabric_worker_adapters.hpp exists with WorkerToFabricEdmSenderBase struct
- WorkerToFabricEdmSenderImpl using alias present
- STREAM_ID template parameter present
- STREAM_ID used in ACTIVE_ETH path
- Commit 26801e0a34a exists

---
*Phase: 01-overlay-register-cleanup*
*Completed: 2026-03-17*
