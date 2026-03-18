---
phase: 04-builder-wiring-flow-control
plan: 02
subsystem: fabric
tags: [vc2, ct-args, builder, firmware, stream-registers, flow-control]

# Dependency graph
requires:
  - phase: 04-builder-wiring-flow-control-01
    provides: Max channel constants (10/3), _without_vc2 variants (9/2), VC2 stream IDs 30/31, extended firmware arrays
provides:
  - Conditional CT arg emission (MAX_NUM_SENDER_CHANNELS=10 or 9, MAX_NUM_RECEIVER_CHANNELS=3 or 2)
  - ACTUAL_VC2_SENDER_CHANNELS CT arg emitted by builder and consumed by firmware
  - Firmware VC boundary derivation from actual per-VC counts (heuristic removed)
  - VC2_SENDER_CHANNEL_START and MAX_NUM_SENDER_CHANNELS_VC2 constants in firmware
  - VC2 sender/receiver free-slots stream IDs wired through builder and firmware
  - DISABLE_RX_CH2_FORWARDING always set to 1
  - Runtime args extended to 10 sender entries
affects: [05-connection-api]

# Tech tracking
tech-stack:
  added: []
  patterns: [conditional CT arg emission based on actual_sender_channels_vc2 > 0, heuristic-to-actual refactor pattern]

key-files:
  created: []
  modified:
    - tt_metal/fabric/erisc_datamover_builder.cpp
    - tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp
    - tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp

key-decisions:
  - "VC2 enablement derived from actual_sender_channels_vc2 > 0 (equivalent to requires_vc2 but avoids FabricBuilderContext dependency)"
  - "emit_channel_allocations_ct_args signature unchanged -- VC2 channels map as padding when VC2 inactive, correct for gated state"

patterns-established:
  - "VC boundary derivation from ACTUAL_VC*_SENDER_CHANNELS instead of magic heuristics"

requirements-completed: [BLDR-02, BLDR-03, BLDR-05, FLOW-03]

# Metrics
duration: 5min
completed: 2026-03-18
---

# Phase 4 Plan 02: Builder Conditional Emission & Firmware VC Boundary Refactor Summary

**Conditional CT arg emission for VC2 (10/3 vs 9/2), firmware heuristic replaced with actual per-VC count derivation, and VC2 stream IDs fully wired through builder-firmware pipeline**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-18T03:08:02Z
- **Completed:** 2026-03-18T03:13:13Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Builder conditionally emits MAX_NUM_SENDER_CHANNELS=10 (VC2) or 9 (no VC2) and MAX_NUM_RECEIVER_CHANNELS=3 or 2
- ACTUAL_VC2_SENDER_CHANNELS CT arg emitted by builder and declared in firmware header
- Firmware heuristic `(MAX_NUM_SENDER_CHANNELS >= 9) ? 5 : 4` replaced with direct ACTUAL_VC*_SENDER_CHANNELS derivation
- VC2 sender free-slots stream IDs (30) for indices 8/9, VC2 receiver free-slots stream ID (31) wired end-to-end
- DISABLE_RX_CH2_FORWARDING always 1 (VC2 receiver never forwards)
- Runtime args extended to 10 sender semaphore entries

## Task Commits

Each task was committed atomically:

1. **Task 1: Conditional CT arg emission, ACTUAL_VC2, VC2 stream IDs, DISABLE_RX_CH2, runtime args** - `c5ba2da6f95` (feat)
2. **Task 2: Replace firmware VC boundary heuristic with actual per-VC count derivation** - `0f62df37a9e` (feat)

## Files Created/Modified
- `tt_metal/fabric/erisc_datamover_builder.cpp` - Conditional MAX channel emission, ACTUAL_VC2 CT arg, VC2 stream IDs, DISABLE_RX_CH2, 10th runtime arg entries
- `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp` - ACTUAL_VC2 declaration, VC boundary refactor, VC2 stream ID declarations
- `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` - Updated sender_channel_free_slots_stream_ids array to use named CT arg stream IDs for indices 8/9

## Decisions Made
- Used `actual_sender_channels_vc2 > 0` as VC2 enablement predicate instead of `fabric_context.requires_vc2()` because the emit_ct_args function has `FabricContext` (from control plane), not `FabricBuilderContext`. The condition is equivalent since channel mapping only assigns VC2 senders when requires_vc2 is true.
- Left `emit_channel_allocations_ct_args` signature unchanged (takes vc0 + vc1 counts). VC2 channels map as padding entries when inactive, which is correct for the current gated state.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated firmware router stream ID array to use named CT args**
- **Found during:** Task 2 (firmware VC boundary refactor)
- **Issue:** `sender_channel_free_slots_stream_ids` array in fabric_erisc_router.cpp had hardcoded 0 for indices 8/9 from Plan 04-01, but now that named CT args are emitted, these should reference the actual stream IDs
- **Fix:** Replaced `0` entries with `sender_channel_8_free_slots_stream_id` and `sender_channel_9_free_slots_stream_id`
- **Files modified:** tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
- **Verification:** Build succeeds
- **Committed in:** 0f62df37a9e (Task 2 commit)

**2. [Rule 1 - Bug] Used actual_sender_channels_vc2 > 0 instead of fabric_context.requires_vc2()**
- **Found during:** Task 1 (conditional CT arg emission)
- **Issue:** Plan referenced `fabric_context.requires_vc2()` but the `fabric_context` variable is `FabricContext` (from control plane), not `FabricBuilderContext`. The `requires_vc2()` method exists only on `FabricBuilderContext`.
- **Fix:** Used `actual_sender_channels_vc2 > 0` which is semantically equivalent
- **Files modified:** tt_metal/fabric/erisc_datamover_builder.cpp
- **Verification:** Build succeeds
- **Committed in:** c5ba2da6f95 (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both auto-fixes necessary for correctness. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- VC2 is fully wired through the builder and firmware CT arg pipeline
- Channel servicing loop already handles VC2 via per-VC iteration (no structural change needed)
- Phase 5 can add the connection API for workers to inject traffic on VC2

---
*Phase: 04-builder-wiring-flow-control*
*Completed: 2026-03-18*
