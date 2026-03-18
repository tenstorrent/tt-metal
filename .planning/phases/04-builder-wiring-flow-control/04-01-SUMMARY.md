---
phase: 04-builder-wiring-flow-control
plan: 01
subsystem: fabric
tags: [vc2, stream-registers, ct-args, channel-constants, firmware]

# Dependency graph
requires:
  - phase: 03-channel-mapping-allocation
    provides: VC2 channel mapping with flat indexing and per-VC counts
provides:
  - num_max_sender_channels=10, num_max_receiver_channels=3 for host-side array sizing
  - _without_vc2 variants (9/2) for conditional firmware CT arg emission
  - VC2 stream register IDs 30/31 in StreamRegAssignments
  - Firmware CT arg arrays extended to 10 sender / 3 receiver entries
affects: [04-02-builder-conditional-emission, 04-03-flow-control-wiring]

# Tech tracking
tech-stack:
  added: []
  patterns: [dual-use stream IDs for mutually exclusive modes, _without_vc2 pattern for conditional firmware sizing]

key-files:
  created: []
  modified:
    - tt_metal/fabric/builder/fabric_builder_config.hpp
    - tt_metal/fabric/erisc_datamover_builder.hpp
    - tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp
    - tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp

key-decisions:
  - "Host-side arrays always sized to absolute max (10/3) even before VC2 conditional emission"
  - "VC2 stream IDs dual-use: ID 30 shared with tensix_relay (UDM mutually exclusive), ID 31 shared with multi_risc_teardown scratch"

patterns-established:
  - "_without_vc2 pattern: preserve old maximums for non-VC2 firmware paths"

requirements-completed: [BLDR-01, FLOW-01, FLOW-02]

# Metrics
duration: 8min
completed: 2026-03-18
---

# Phase 4 Plan 01: Builder Wiring Foundation Summary

**Host-side max channel constants bumped to 10/3 with _without_vc2 variants, VC2 stream IDs 30/31 registered, and all firmware CT arg arrays extended to match**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-18T02:56:23Z
- **Completed:** 2026-03-18T03:04:30Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Host-side array constants now accommodate VC2 (num_max_sender_channels=10, num_max_receiver_channels=3)
- Added _without_vc2 variants preserving old 9/2 values for firmware conditional emission in Plan 02
- Registered VC2 stream register IDs (30 for sender free-slots, 31 for receiver free-slots) with dual-use documentation
- Extended all firmware CT arg arrays to match new maximums (16 arrays total)

## Task Commits

Each task was committed atomically:

1. **Task 1: Bump max channel constants and add VC2 stream reg entries** - `df0cab654a7` (feat)
2. **Task 2: Extend firmware CT args arrays to 10 sender / 3 receiver entries** - `0f1db01a507` (feat)

**Plan metadata:** `837022f1dd0` (docs: complete plan)

## Files Created/Modified
- `tt_metal/fabric/builder/fabric_builder_config.hpp` - Added _without_vc2 variants, bumped max constants to 10/3
- `tt_metal/fabric/erisc_datamover_builder.hpp` - Added vc2_sender_free_slots_stream_id=30, vc2_receiver_free_slots_stream_id=31
- `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp` - Extended all sender arrays to 10 entries, receiver arrays to 3 entries
- `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` - Extended firmware router hardcoded arrays to match new maximums

## Decisions Made
- Host-side arrays always sized to absolute max (10/3) even before VC2 conditional emission -- simplifies host code, firmware gets correct size via CT args
- VC2 stream IDs dual-use with tensix_relay (ID 30) and multi_risc_teardown scratch (ID 31) -- mutually exclusive modes make this safe

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Extended sender_channel_free_slots_stream_ids in firmware router**
- **Found during:** Task 2 (firmware CT arg array extension)
- **Issue:** fabric_erisc_router.cpp had hardcoded 9-entry array for sender_channel_free_slots_stream_ids sized to MAX_NUM_SENDER_CHANNELS, would fail JIT compile with new max of 10
- **Fix:** Added 10th entry (0) for VC2 sender channel
- **Files modified:** tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
- **Verification:** Host build succeeds
- **Committed in:** 0f1db01a507 (Task 2 commit)

**2. [Rule 3 - Blocking] Extended local_sender_connection_info_addresses in firmware router**
- **Found during:** Task 2 (firmware CT arg array extension)
- **Issue:** fabric_erisc_router.cpp had hardcoded 9-entry initializer for connection info addresses sized to MAX_NUM_SENDER_CHANNELS
- **Fix:** Added local_sender_channel_9_connection_info_addr as 10th entry
- **Files modified:** tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
- **Verification:** Host build succeeds
- **Committed in:** 0f1db01a507 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both auto-fixes required for JIT compile correctness. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All structural capacity for VC2 is in place (host arrays, firmware arrays, stream IDs)
- Plan 02 can add conditional emission using _without_vc2 variants
- Plan 03 can wire VC2 flow control using the new stream IDs 30/31

---
*Phase: 04-builder-wiring-flow-control*
*Completed: 2026-03-18*
