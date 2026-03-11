---
phase: 02-silicon-data-transfer-validation
plan: 03
subsystem: testing
tags: [fabric, auto-packetization, silicon, validation, Fabric2DFixture, Fabric1DFixture, sparse-multicast, tt-smi]

# Dependency graph
requires:
  - phase: 02-silicon-data-transfer-validation
    plan: 01
    provides: unicast silicon tests (4 families), unicast_runner.cpp, test_common.hpp
  - phase: 02-silicon-data-transfer-validation
    plan: 02
    provides: multicast silicon tests (4 families + sparse), multicast_runner.cpp, 5 device kernels
provides:
  - Silicon validation result: 16/17 tests PASSED, 1 SKIPPED (SparseMulticast issue #36581)
  - Confirmed all 8 active families deliver byte-for-byte correct data on hardware
  - GTEST_SKIP restoration for sparse multicast (test hangs, causes device Ethernet core lockup)
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: [tt-smi board reset after device hang, GTEST_SKIP as hang guard for known firmware limitations]

key-files:
  created: []
  modified:
    - tests/tt_metal/tt_fabric/fabric_data_movement/test_auto_packetization.cpp

key-decisions:
  - "SparseMulticast silicon test deferred (GTEST_SKIP restored): removing skip causes Ethernet core lockup requiring board reset via tt-smi, confirming issue #36581 is a real firmware limitation"
  - "8/9 families validated as phase gate: SparseMulticast skip is acceptable, all other families pass"

patterns-established:
  - "Sparse multicast hang recovery: tt-smi -r 0,1,2,3 resets all devices after Ethernet core lockup"

requirements-completed: []

# Metrics
duration: 15min
completed: 2026-03-11
---

# Phase 2 Plan 03: Silicon Test Execution Summary

**16/17 auto-packetization silicon tests PASSED on hardware -- all 8 active families validated byte-for-byte correct across Fabric2DFixture and Fabric1DFixture; SparseMulticast deferred (issue #36581 firmware hang confirmed)**

## Performance

- **Duration:** 15 min
- **Started:** 2026-03-11T17:00:00Z
- **Completed:** 2026-03-11T17:15:00Z
- **Tasks:** 2 (1 auto + 1 human-verify)
- **Files modified:** 1

## Accomplishments
- Ran full auto-packetization silicon test suite on 4-chip TT hardware
- All 8 enabled families passed with byte-for-byte data correctness: UnicastWrite, UnicastScatter, UnicastFusedAtomicInc, UnicastFusedScatterAtomicInc, MulticastWrite, MulticastScatter, MulticastFusedAtomicInc, MulticastFusedScatterAtomicInc
- Both Fabric2DFixture (2D mesh topology) and Fabric1DFixture (linear topology) confirmed working
- Confirmed and documented SparseMulticast firmware limitation -- removing GTEST_SKIP causes Ethernet core lockup requiring board reset

## Task Commits

Each task was committed atomically:

1. **Task 1: Run silicon tests + fix failures** - `bdae1b837e` (fix, prior session) + `88c53d0b7f` (fix, prior session)
2. **Task 2: Human verification** - approved by user

**Prior session silicon fixes included:**
- `88c53d0b7f`: L1 overflow, 1D/2D kernel routing, device adjacency fixes
- `bdae1b837e`: FABRIC_2D ifdef guards for all auto-packetization kernels + 1D test implementations (replaced 6 GTEST_SKIP stubs)

## Files Created/Modified
- `tests/tt_metal/tt_fabric/fabric_data_movement/test_auto_packetization.cpp` - Restored GTEST_SKIP for SparseMulticast (removing it caused device hang/lockup)

## Decisions Made
- Restored GTEST_SKIP for `AutoPacketizationSparseMulticastSilicon`: running without the skip caused the test to hang for 5+ minutes, leaving device Ethernet core (x=31,y=25) in an unresponsive state. Required `tt-smi -r 0,1,2,3` board reset. This confirms issue #36581 is a real firmware limitation, not a test code issue.
- Phase gate accepted at 8/9 families: SparseMulticast skip is documented and tracked in issue #36581.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Restored GTEST_SKIP for SparseMulticast to prevent device hang**
- **Found during:** Task 1 (silicon test execution)
- **Issue:** Working tree had GTEST_SKIP removed from AutoPacketizationSparseMulticastSilicon. Running the test without the guard caused a 5+ minute hang, then device Ethernet core lockup requiring board reset.
- **Fix:** Restored the GTEST_SKIP with the original message referencing issue #36581. Used `tt-smi -r 0,1,2,3` to recover the device.
- **Files modified:** tests/tt_metal/tt_fabric/fabric_data_movement/test_auto_packetization.cpp
- **Verification:** Tests pass 16/17 after board reset, no hang.
- **Committed in:** No separate commit needed -- restoration matched HEAD state (the working tree change was undone)

---

**Total deviations:** 1 auto-fixed (1 bug: device-hang-causing GTEST_SKIP removal reverted)
**Impact on plan:** Fix necessary to prevent device lockup. SparseMulticast remains deferred as intended.

## Issues Encountered
- Sparse multicast test hang corrupted Ethernet core state on Device 0 (port x=31,y=25). Recovery required `tt-smi -r 0,1,2,3`. This confirms the test guard is essential -- the firmware limitation is real.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 8 active auto-packetization families validated on silicon
- Phase 02-silicon-data-transfer-validation is complete
- SparseMulticast tracked in issue #36581 for follow-up when firmware is fixed
- Binary: `./build_Release/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*AutoPacketization*Silicon*"` (16 pass, 1 skip, 0 fail)

## Self-Check: PASSED

All referenced commits verified in git log. test_auto_packetization.cpp confirmed at HEAD state (no working tree diff).

---
*Phase: 02-silicon-data-transfer-validation*
*Completed: 2026-03-11*
