---
phase: 02-silicon-data-transfer-validation
plan: 01
subsystem: testing
tags: [fabric, auto-packetization, silicon, unicast, scatter, fused-atomic-inc, BaseFabricFixture]

# Dependency graph
requires:
  - phase: 01-fabric-auto-packetization
    provides: auto-packetizing wrapper APIs and compile-probe kernels
provides:
  - test_common.hpp with AutoPacketFamily enum (9 families) and RawTestParams struct
  - BaseFabricFixture-based unicast_runner.cpp for silicon data validation
  - 3 new unicast device kernels (scatter, fused_atomic_inc, fused_scatter_atomic_inc)
  - 4 silicon TEST_F cases for unicast families in test_auto_packetization.cpp
affects: [02-silicon-data-transfer-validation-plan-02]

# Tech tracking
tech-stack:
  added: []
  patterns: [BaseFabricFixture L1 direct I/O, per-chip MeshDevice dispatch, family_kernel_path selection]

key-files:
  created:
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_common.hpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/scatter_unicast_tx_writer_raw.cpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/fused_atomic_inc_unicast_tx_writer_raw.cpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/fused_scatter_atomic_inc_unicast_tx_writer_raw.cpp
  modified:
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/unicast_runner.cpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/test_auto_packetization.cpp
    - tests/tt_metal/tt_fabric/sources.cmake

key-decisions:
  - "L1 direct I/O via WriteToDeviceL1/ReadFromDeviceL1 instead of DRAM MeshBuffer pattern"
  - "GlobalSemaphore created fresh per test call (not static) for BaseFabricFixture per-chip MeshDevice"
  - "family_kernel_path() returns all 9 paths including future multicast kernels for Plan 02"

patterns-established:
  - "BaseFabricFixture runner: get_device(phys_id), generate_worker_mem_map, L1 write/read, RunProgramNonblocking"
  - "Scatter test validation: split verification of first half and second half at different L1 addresses"

requirements-completed: []

# Metrics
duration: 5min
completed: 2026-03-11
---

# Phase 2 Plan 01: Unicast Silicon Tests Summary

**Rewritten unicast runner for BaseFabricFixture with L1 direct I/O, 3 new device kernels (scatter, fused_atomic_inc, fused_scatter_atomic_inc), and 4 silicon TEST_F cases**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-11T14:23:16Z
- **Completed:** 2026-03-11T14:28:02Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Rewrote unicast_runner.cpp from MeshDeviceFixtureBase to BaseFabricFixture pattern with L1 direct I/O
- Created test_common.hpp defining AutoPacketFamily enum (all 9 families), RawTestParams, and helper functions
- Created 3 new unicast device kernels for scatter, fused_atomic_inc, and fused_scatter_atomic_inc families
- Added 4 silicon data-validation TEST_F cases covering all 4 unicast wrapper families

## Task Commits

Each task was committed atomically:

1. **Task 1: Create test_common.hpp and rewrite unicast_runner.cpp** - `abf6ecea5b` (feat)
2. **Task 2: Create 3 device kernels + TEST_F cases** - `4c9eb5f1c8` (feat)

## Files Created/Modified
- `auto_packetization/test_common.hpp` - AutoPacketFamily enum (9 families), RawTestParams, family_kernel_path()
- `auto_packetization/unicast_runner.cpp` - Full rewrite using BaseFabricFixture L1 direct I/O pattern
- `auto_packetization/kernels/scatter_unicast_tx_writer_raw.cpp` - Scatter write + separate atomic_inc
- `auto_packetization/kernels/fused_atomic_inc_unicast_tx_writer_raw.cpp` - Fused write + atomic_inc (auto-packetizing)
- `auto_packetization/kernels/fused_scatter_atomic_inc_unicast_tx_writer_raw.cpp` - Fused scatter + atomic_inc
- `test_auto_packetization.cpp` - 4 new TEST_F silicon cases added alongside existing compile-only tests
- `sources.cmake` - unicast_runner.cpp added to UNIT_TESTS_FABRIC_SRC

## Decisions Made
- Used L1 direct I/O (WriteToDeviceL1/ReadFromDeviceL1) instead of DRAM MeshBuffer to match BaseFabricFixture pattern and avoid the incorrect DRAM-as-L1 bug in the old runner
- Created GlobalSemaphore fresh per test invocation rather than using static optional, since BaseFabricFixture uses per-chip MeshDevice which requires per-device semaphore creation
- family_kernel_path() maps all 9 families including 5 multicast kernel paths that Plan 02 will create, so Plan 02 does not need to modify test_common.hpp

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added missing fabric_context.hpp include**
- **Found during:** Task 1 (unicast_runner.cpp build)
- **Issue:** FabricContext was forward-declared in control_plane.hpp but incomplete for method calls
- **Fix:** Added `#include "tt_metal/fabric/fabric_context.hpp"`
- **Files modified:** unicast_runner.cpp
- **Verification:** Build succeeds
- **Committed in:** abf6ecea5b (Task 1 commit)

**2. [Rule 1 - Bug] Removed unused variable warning**
- **Found during:** Task 1 (build)
- **Issue:** `is_fused` variable was declared but unused (Werror)
- **Fix:** Removed the unused variable (fused logic handled by kernel selection, not host-side branching)
- **Verification:** Build succeeds with -Werror
- **Committed in:** abf6ecea5b (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes trivial and necessary for compilation. No scope change.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- test_common.hpp and unicast runner pattern are ready for Plan 02 to replicate for multicast families
- family_kernel_path() already returns correct paths for all multicast kernels Plan 02 will create
- Build infrastructure (sources.cmake) ready for multicast_runner.cpp addition
- Silicon test execution requires TT hardware -- tests compile but silicon validation deferred to explicit test run

## Self-Check: PASSED

All 8 files verified on disk. Both task commits (abf6ecea5b, 4c9eb5f1c8) verified in git log.

---
*Phase: 02-silicon-data-transfer-validation*
*Completed: 2026-03-11*
