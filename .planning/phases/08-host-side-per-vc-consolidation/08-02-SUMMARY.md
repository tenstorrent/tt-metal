---
phase: 08-host-side-per-vc-consolidation
plan: 02
subsystem: fabric
tags: [per-vc, call-site-migration, erisc-datamover-builder, named-args]

requires:
  - phase: 08-host-side-per-vc-consolidation
    plan: 01
    provides: "per-VC array constants and unified functions"
provides:
  - "All host builder call sites use per-VC arrays and loop-based named_args"
  - "No _vc0/_vc1 split patterns remain in host builder code"
affects: []

tech-stack:
  added: []
  patterns: ["fmt::format loop for per-VC named_args assignment", "per-VC local arrays replacing _vc0/_vc1 scalar pairs"]

key-files:
  created: []
  modified:
    - tt_metal/fabric/erisc_datamover_builder.cpp

key-decisions:
  - "Used fmt::format loop producing identical key strings (NUM_DOWNSTREAM_SENDERS_VC0 etc.) for device compatibility"
  - "enable_first_level_ack_per_vc = {enable_first_level_ack, 0} captures asymmetric VC0/VC1 behavior"
  - "actual_sender_channels_per_vc_sized bridges uint32_t array to size_t array for emit_channel_allocations_ct_args"

patterns-established:
  - "fmt::format per-VC loop pattern for named compile-time args"

requirements-completed: ["Phase 8 goal"]

duration: 4min
completed: 2026-03-14
---

# Phase 8 Plan 02: Migrate Host Builder Call Sites to Per-VC API Summary

**Converted _vc0/_vc1 locals to per_vc arrays and replaced individual named_args assignments with fmt::format loop in erisc_datamover_builder**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-14T16:04:23Z
- **Completed:** 2026-03-14T16:08:37Z
- **Tasks:** 3 (1 code change, 1 already done by plan 01, 1 build/test validation)
- **Files modified:** 1

## Accomplishments
- Replaced num_vc0_downstream_edms/num_vc1_downstream_edms with num_downstream_edms_per_vc array
- Replaced vc0_downstream_edm_size/vc1_downstream_edm_size with downstream_edm_size_per_vc array
- Replaced actual_sender_channels_vc0/vc1 with actual_sender_channels_per_vc array
- Added enable_first_level_ack_per_vc = {enable_first_level_ack, 0} for asymmetric VC behavior
- Replaced 8 individual named_args assignments with 4-line fmt::format loop
- Updated all remaining references to use array indexing
- Build passes cleanly, all 12 latency golden comparisons pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Convert erisc_datamover_builder locals and named_args to per-VC** - `b2fd0ec28f6` (feat)
2. **Task 2: Update fabric_tensix_builder and router_connection_mapping** - No commit needed (already migrated by plan 01)
3. **Task 3: Build and sanity test** - No commit needed (validation only, all passed)

## Files Created/Modified
- `tt_metal/fabric/erisc_datamover_builder.cpp` - Per-VC local arrays, fmt::format named_args loop, array-indexed references

## Decisions Made
- Used fmt::format loop producing identical key strings (e.g., "NUM_DOWNSTREAM_SENDERS_VC0") for device kernel compatibility
- enable_first_level_ack_per_vc = {enable_first_level_ack, 0} captures the asymmetric behavior where only VC0 uses bubble flow control
- Created actual_sender_channels_per_vc_sized (size_t) to bridge the type mismatch when passing to emit_channel_allocations_ct_args

## Deviations from Plan

### Task 2 Already Completed by Plan 01

**fabric_tensix_builder.cpp** and **router_connection_mapping.cpp** call sites were already updated during plan 01 execution (documented in 08-01-SUMMARY.md deviations). No changes needed for Task 2.

No other deviations -- plan executed as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 8 complete: all host builder code migrated to per-VC API
- No _vc0/_vc1 split constants, functions, or locals remain in host builder code
- Ready for Phase 9

## Self-Check: PASSED
