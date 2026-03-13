---
phase: 04-test-infrastructure-cleanup
plan: 02
subsystem: testing
tags: [cpp, gtest, tt-fabric, auto-packetization, refactor]

# Dependency graph
requires:
  - phase: 04-test-infrastructure-cleanup
    provides: test_common.hpp with shared AutoPacketFamily/RawTestParams definitions

provides:
  - make_tx_pattern and verify_payload_words inline helpers in test_common.hpp (shared by all test files)
  - run_silicon_family_test dispatch helper collapsing 16 TEST_F bodies to single-line calls
  - Both unicast_runner.cpp and multicast_runner.cpp stripped of anonymous-namespace duplicates

affects: [04-test-infrastructure-cleanup, 04-03-PLAN.md]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Shared test utilities in test_common.hpp (make_tx_pattern, verify_payload_words)"
    - "Dispatch helper pattern: run_silicon_family_test selects payload sizes from family_is_scatter()"
    - "Function pointer type alias RunnerFn for test dispatch"

key-files:
  created: []
  modified:
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_common.hpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/unicast_runner.cpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/multicast_runner.cpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/test_auto_packetization.cpp

key-decisions:
  - "Placed make_tx_pattern/verify_payload_words inside tt::tt_fabric::test namespace in test_common.hpp (not anonymous namespace) so runners can use them via existing namespace"
  - "run_silicon_family_test placed in anonymous namespace of test_auto_packetization.cpp (not test_common.hpp) because it depends on pick_chip_pair and runner function signatures"
  - "RunnerFn type alias defined in anonymous namespace using function pointer (not std::function) for zero-overhead dispatch"

patterns-established:
  - "Pattern 1: Test utility consolidation — shared helpers in test_common.hpp, not duplicated per runner"
  - "Pattern 2: Dispatch helper pattern — run_silicon_family_test unifies 16 TEST_F bodies by parameterizing family + runner_fn"

requirements-completed: [TEST-02, TEST-03]

# Metrics
duration: 4min
completed: 2026-03-12
---

# Phase 4 Plan 02: Consolidate test utilities and de-duplicate 16 silicon TEST_F bodies Summary

**make_tx_pattern/verify_payload_words moved to test_common.hpp and 16 silicon TEST_F bodies collapsed to single-line dispatch calls via run_silicon_family_test helper**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-12T14:13:34Z
- **Completed:** 2026-03-12T14:17:19Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Eliminated duplicate make_tx_pattern and verify_payload_words from both runner files — now defined once in test_common.hpp
- Added run_silicon_family_test dispatch helper that selects payload sizes based on family_is_scatter(), replacing ~300 lines of repeated boilerplate
- Reduced test_auto_packetization.cpp from 767 to 505 lines (34% reduction)

## Task Commits

Each task was committed atomically:

1. **Task 1: Move shared utilities to test_common.hpp and strip runners** - `7c992d93da` (refactor)
2. **Task 2: De-duplicate 16 silicon TEST_F cases with a dispatch helper** - `5ef36a297d` (refactor)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_common.hpp` - Added `#include <vector>`, `#include <gtest/gtest.h>`, and the two inline helpers
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/unicast_runner.cpp` - Removed anonymous namespace with duplicate definitions
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/multicast_runner.cpp` - Removed anonymous namespace with duplicate definitions
- `tests/tt_metal/tt_fabric/fabric_data_movement/test_auto_packetization.cpp` - Added RunnerFn type alias + run_silicon_family_test helper; replaced 16 TEST_F bodies with single dispatch calls

## Decisions Made
- Placed make_tx_pattern/verify_payload_words in `tt::tt_fabric::test` namespace (not anonymous) so runners can use them via the existing namespace that test_common.hpp defines
- run_silicon_family_test placed in anonymous namespace inside test_auto_packetization.cpp because it depends on the file-local pick_chip_pair helper and the runner function signatures declared at the top
- RunnerFn is a plain function pointer type alias (not std::function) for zero overhead

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Ready for 04-03 build + silicon test validation
- ninja fabric_unit_tests build validation is the gate for this work

## Self-Check: PASSED

- test_common.hpp: FOUND
- unicast_runner.cpp: FOUND
- multicast_runner.cpp: FOUND
- test_auto_packetization.cpp: FOUND
- 04-02-SUMMARY.md: FOUND
- Commit 7c992d93da: FOUND
- Commit 5ef36a297d: FOUND

---
*Phase: 04-test-infrastructure-cleanup*
*Completed: 2026-03-12*
