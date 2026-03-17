---
phase: 03-channel-mapping-allocation
plan: 02
subsystem: testing
tags: [gtest, vc2, channel-mapping, fabric-router, unit-tests]

# Dependency graph
requires:
  - phase: 03-channel-mapping-allocation/01
    provides: "initialize_vc2_mappings() implementation in FabricRouterChannelMapping"
provides:
  - "11 new VC2 channel mapping unit tests"
  - "Fixed 5 Z router tests to match VC0=5 sender count from plan 03-01"
affects: [04-buffer-slot-integration]

# Tech tracking
tech-stack:
  added: []
  patterns: ["IntermeshVCConfig factory + requires_vc2=true manual override for test setup"]

key-files:
  created: []
  modified:
    - "tests/tt_metal/tt_fabric/fabric_router/test_router_channel_mapping.cpp"

key-decisions:
  - "Fixed Z router VC0 sender count from 4 to 5 in existing tests (pre-existing mismatch after plan 03-01)"
  - "VC2 test configs use IntermeshVCConfig::full_mesh() + requires_vc2=true (factory does not set VC2)"

patterns-established:
  - "VC2 test pattern: auto config = IntermeshVCConfig::full_mesh(); config.requires_vc2 = true;"

requirements-completed: [CMAP-01, CMAP-02, CMAP-03, BLDR-04]

# Metrics
duration: 19min
completed: 2026-03-17
---

# Phase 3 Plan 2: VC2 Channel Mapping Tests Summary

**11 VC2 unit tests validating sender/receiver flat indices, Z router sender-only, disabled state, and VC0/VC1 regression**

## Performance

- **Duration:** 19 min
- **Started:** 2026-03-17T16:01:58Z
- **Completed:** 2026-03-17T16:21:07Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Added 11 new VC2 channel mapping test cases covering all CMAP requirements
- Fixed 5 pre-existing Z router test failures (VC0 sender count 4 -> 5 after plan 03-01)
- All 40 RouterChannelMapping tests pass, all ChannelTrimming host tests pass
- Verified full fabric_unit_tests suite: 0 test failures (device fixture setups skipped, expected without hardware)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add VC2 channel mapping unit tests** - `de744b54095` (feat)
2. **Task 2: Full test suite verification** - no code changes, verification only

## Files Created/Modified
- `tests/tt_metal/tt_fabric/fabric_router/test_router_channel_mapping.cpp` - 11 new VC2 tests, 5 Z router test fixes, updated coverage table

## Decisions Made
- Fixed Z router VC0 sender count from 4 to 5 in 5 existing tests: plan 03-01 changed num_sender_channels_z_router_vc0 to 5 but didn't update these tests
- VC2 test configs manually set requires_vc2=true after factory call since factories don't set it (runtime computed in compute_intermesh_vc_config)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed 5 pre-existing Z router test assertions**
- **Found during:** Task 1 (pre-build baseline verification)
- **Issue:** Plan 03-01 changed Z router VC0 from 4 to 5 sender channels but did not update test assertions. Tests ZRouter_VC0_Has4SenderChannels, ZRouter_VC1_SenderChannels_MapToErisc4Through7, ZRouterNoTensix_VC0_VC1_ReceiverChannel_MapsToErisc, ZRouter_CompleteChannelLayout, and GetAllSenderMappings_ZRouter all expected stale values.
- **Fix:** Updated VC0 count 4->5, VC1 base index 4->5, total sender count 8->9 in all affected tests
- **Files modified:** tests/tt_metal/tt_fabric/fabric_router/test_router_channel_mapping.cpp
- **Verification:** All 40 tests pass
- **Committed in:** de744b54095 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential fix -- tests were broken without this correction. No scope creep.

## Issues Encountered
None beyond the pre-existing test failures documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All channel mapping and allocation tests verified
- Phase 03 complete, ready for Phase 04 (buffer slot integration)

---
*Phase: 03-channel-mapping-allocation*
*Completed: 2026-03-17*
