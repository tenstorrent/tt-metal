---
phase: 05-connection-api-testing
plan: 01
subsystem: fabric
tags: [vc2, connection-api, worker-adapter, stream-id, runtime-option]

requires:
  - phase: 04-builder-wiring-flow-control
    provides: "CT arg emission with VC2 conditional paths, per-VC sender channel counts"
provides:
  - "TT_FATAL guard removed, VC2 activatable via RT option"
  - "requires_vc2 gated on RT option AND hardware conditions"
  - "WorkerToFabricEdmSenderVC2 type alias with stream ID 30"
  - "Private VC2 connection API (append_fabric_vc2_connection_rt_args)"
affects: [05-02, testing, worker-kernels]

tech-stack:
  added: []
  patterns: ["Private connection API in separate files from public API", "VC2 sender channel index = VC0_count + VC1_count"]

key-files:
  created:
    - tt_metal/fabric/fabric_vc2_connection.hpp
    - tt_metal/fabric/fabric_vc2_connection.cpp
  modified:
    - tt_metal/fabric/fabric_builder_context.cpp
    - tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp
    - tt_metal/fabric/sources.cmake

key-decisions:
  - "VC2 connection API kept private (not in public fabric.hpp) to avoid premature exposure"
  - "VC2 sender channel computed dynamically from allocator per-VC counts rather than hardcoded index"
  - "CoreCoord include added to header to fix build (auto-fix Rule 3)"

patterns-established:
  - "Private fabric APIs go in separate files under tt_metal/fabric/, not in public API headers"
  - "VC2 flat channel index = get_num_sender_channels(0) + get_num_sender_channels(1)"

requirements-completed: [CONN-01, CONN-02]

duration: 5min
completed: 2026-03-18
---

# Phase 5 Plan 1: Connection API & Testing Summary

**Removed VC2 TT_FATAL guard, gated requires_vc2 on RT option + HW conditions, added VC2 worker adapter alias (stream ID 30), and created private VC2 connection API with per-VC sender channel computation**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-18T16:14:28Z
- **Completed:** 2026-03-18T16:19:22Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Removed TT_FATAL guard that blocked VC2 activation, enabling VC2 when RT option is set
- Gated requires_vc2 on rtoptions.get_enable_fabric_vc2() AND hardware conditions (Blackhole, no UDM/MUX)
- Added WorkerToFabricEdmSenderVC2Impl/WorkerToFabricEdmSenderVC2 type aliases using stream ID 30
- Created fabric_vc2_connection.hpp/.cpp with append_fabric_vc2_connection_rt_args (private, WORKER-only)
- VC2 sender channel index derived from VC0 + VC1 sender counts via static channel allocator
- Full build (fabric_unit_tests) succeeds

## Task Commits

Each task was committed atomically:

1. **Task 1: Remove TT_FATAL guard, gate requires_vc2 on RT option, add VC2 adapter type alias** - `e534834fadc` (feat)
2. **Task 2: Create private VC2 connection API and register in build** - `16c4296a3b0` (feat)

## Files Created/Modified
- `tt_metal/fabric/fabric_builder_context.cpp` - Removed TT_FATAL, gated requires_vc2 on RT option
- `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp` - Added WorkerToFabricEdmSenderVC2 type aliases
- `tt_metal/fabric/fabric_vc2_connection.hpp` - Private VC2 connection API declaration
- `tt_metal/fabric/fabric_vc2_connection.cpp` - VC2 connection API implementation with per-VC channel computation
- `tt_metal/fabric/sources.cmake` - Registered fabric_vc2_connection.cpp in build

## Decisions Made
- VC2 connection API kept private (not added to public fabric.hpp) to avoid premature API exposure
- Used SenderWorkerAdapterSpec overload (explicit addresses) rather than eth_channel overload for VC2 sender
- VC2 sender channel index computed dynamically from allocator get_num_sender_channels(0) + get_num_sender_channels(1)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added missing CoreCoord include to header**
- **Found during:** Task 2 (build verification)
- **Issue:** fabric_vc2_connection.hpp used CoreCoord without including core_coord.hpp
- **Fix:** Added `#include <tt-metalium/core_coord.hpp>` to the header
- **Files modified:** tt_metal/fabric/fabric_vc2_connection.hpp
- **Verification:** Build succeeds after fix
- **Committed in:** 16c4296a3b0 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Minor missing include, essential for compilation. No scope creep.

## Issues Encountered
None beyond the auto-fixed include.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- VC2 connection API ready for integration testing in plan 05-02
- All build targets compile successfully
- Private API pattern established for future VC2 extensions

---
*Phase: 05-connection-api-testing*
*Completed: 2026-03-18*
