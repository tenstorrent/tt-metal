---
phase: 01-fabric-auto-packetization
plan: 03
subsystem: api
tags: [fabric, scatter-write, single-packet, rename, passthrough]

# Dependency graph
requires:
  - phase: 01-fabric-auto-packetization
    provides: "Plan 01 test infrastructure scaffolding"
provides:
  - "_single_packet renames + passthrough wrappers for all 4 scatter/fused-scatter families in linear namespace"
affects: [01-07-integration-tests]

# Tech tracking
tech-stack:
  added: []
  patterns: ["passthrough wrapper pattern for scatter functions that cannot be auto-packetized"]

key-files:
  created: []
  modified: ["tt_metal/fabric/hw/inc/linear/api.h"]

key-decisions:
  - "Scatter wrappers are passthrough (no chunking loop) because NocUnicastScatterCommandHeader carries pre-computed NOC scatter addresses that cannot be independently incremented per chunk"
  - "Addrgen internal calls updated to _single_packet to avoid unnecessary wrapper indirection"

patterns-established:
  - "Passthrough wrapper pattern: scatter APIs renamed to _single_packet, wrapper calls _single_packet directly with doc comment explaining limitation"

requirements-completed: [AP-02, AP-03]

# Metrics
duration: 4min
completed: 2026-03-11
---

# Phase 01 Plan 03: Linear Scatter Rename Summary

**Renamed 4 scatter/fused-scatter function families (8 overloads) to _single_packet with 8 passthrough wrappers documenting chunking limitation**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-11T01:25:37Z
- **Completed:** 2026-03-11T01:30:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Renamed all 4 scatter function families to _single_packet suffix (8 overloads total: template + conn mgr for each)
- Added 8 passthrough wrappers under original names with doc comments explaining scatter addresses cannot be chunked
- Updated 2 addrgen internal calls to use _single_packet directly to avoid wrapper indirection
- Verified _with_state and _set_state variants were NOT modified

## Task Commits

Each task was committed atomically:

1. **Task 1: Rename scatter and fused-scatter write families to _single_packet + passthrough wrappers** - `140a473d49` (feat)
2. **Task 2: Verify no addrgen functions were modified** - verification only, no commit needed

**Plan metadata:** (pending final docs commit)

## Files Created/Modified
- `tt_metal/fabric/hw/inc/linear/api.h` - Renamed 8 scatter overloads to _single_packet, added 8 passthrough wrappers, updated 2 addrgen internal calls

## Decisions Made
- Scatter wrappers are passthrough (no chunking loop) because NocUnicastScatterCommandHeader carries pre-computed NOC scatter addresses that cannot be independently chunked
- Addrgen internal calls updated to _single_packet to avoid unnecessary wrapper indirection

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All scatter/fused-scatter renames in linear/api.h complete
- Ready for Plan 05 (mesh scatter renames) and Plan 07 (integration tests)

---
*Phase: 01-fabric-auto-packetization*
*Completed: 2026-03-11*
