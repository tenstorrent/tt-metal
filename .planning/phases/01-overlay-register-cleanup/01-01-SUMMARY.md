---
phase: 01-overlay-register-cleanup
plan: 01
subsystem: fabric
tags: [stream-registers, refactor, type-safety, increment-on-write, scratch-registers]

# Dependency graph
requires: []
provides:
  - StreamRegAssignments with IncrementOnWrite and Scratch tagged sub-structs
  - Type-specific accessors get_inc_on_write_ids() and get_scratch_ids()
  - Explicit dual-use documentation for stream ID 30
affects: [01-overlay-register-cleanup, 02-vc2-stream-id-allocation]

# Tech tracking
tech-stack:
  added: []
  patterns: [nested-sub-struct-for-register-semantics, named-member-access-over-positional-indexing]

key-files:
  created: []
  modified:
    - tt_metal/fabric/erisc_datamover_builder.hpp
    - tt_metal/fabric/erisc_datamover_builder.cpp
    - tt_metal/fabric/fabric_tensix_builder_impl.cpp

key-decisions:
  - "IncrementOnWrite and Scratch sub-structs use hardware register semantics for naming"
  - "Dual-use stream ID 30 appears in both sub-structs with cross-reference comments"
  - "Replaced get_all_stream_ids() entirely with type-specific accessors"

patterns-established:
  - "Sub-struct access pattern: StreamRegAssignments::IncrementOnWrite::member or ::Scratch::member"
  - "Named member access replaces positional array indexing for CT arg emission"

requirements-completed: [SREG-01]

# Metrics
duration: 4min
completed: 2026-03-17
---

# Phase 1 Plan 1: StreamRegAssignments Sub-struct Refactor Summary

**StreamRegAssignments restructured into IncrementOnWrite and Scratch tagged sub-structs with dual-use stream ID 30 explicit in both, all callers updated from positional indexing to named member access**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-17T02:13:25Z
- **Completed:** 2026-03-17T02:17:12Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- StreamRegAssignments now has IncrementOnWrite (IDs 0-30) and Scratch (IDs 30-31) nested sub-structs with semantic comments
- Dual-use stream ID 30 explicitly documented in both sub-structs with cross-reference comments
- All positional stream_ids[N] indexing in erisc_datamover_builder.cpp replaced with named sub-struct member access
- fabric_tensix_builder_impl.cpp updated to use IncrementOnWrite:: paths for sender channel references
- get_all_stream_ids() removed, replaced by get_inc_on_write_ids() and get_scratch_ids()

## Task Commits

Each task was committed atomically:

1. **Task 1: Restructure StreamRegAssignments into tagged sub-structs** - `1d1706e6336` (refactor)
2. **Task 2: Update all callers to use sub-struct paths** - `9fc9a1e523d` (refactor)

## Files Created/Modified
- `tt_metal/fabric/erisc_datamover_builder.hpp` - StreamRegAssignments with IncrementOnWrite and Scratch sub-structs, type-specific accessors
- `tt_metal/fabric/erisc_datamover_builder.cpp` - CT arg emission using named sub-struct member access instead of positional array indexing
- `tt_metal/fabric/fabric_tensix_builder_impl.cpp` - sender_channel references updated to IncrementOnWrite:: paths

## Decisions Made
- Used hardware register semantics (IncrementOnWrite / Scratch) for sub-struct naming, not logical purpose
- Dual-use stream ID 30 appears in both sub-structs rather than creating a third DualUse concept
- get_all_stream_ids() fully removed; callers must explicitly choose register type via get_inc_on_write_ids() or get_scratch_ids()

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Build target `tt-metalium` does not exist; used `tt_metal` instead (trivial, resolved immediately)

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- StreamRegAssignments sub-struct pattern established for VC2 stream ID allocation
- All callers already use explicit register-type paths, making future additions type-safe
- Ready for Plan 01-02 (adapter templatization) or Phase 2 work

---
*Phase: 01-overlay-register-cleanup*
*Completed: 2026-03-17*
