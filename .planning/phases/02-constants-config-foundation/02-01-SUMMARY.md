---
phase: 02-constants-config-foundation
plan: 01
subsystem: fabric
tags: [constants, vc, array-init, cpp]

# Dependency graph
requires:
  - phase: 01-overlay-register-cleanup
    provides: stream register restructuring into IncrementOnWrite/Scratch sub-structs
provides:
  - MAX_NUM_VCS = 3 constant propagated through all fabric builder code
  - All MAX_NUM_VCS-sized arrays use {} zero-initialization
  - Legacy sum-of-two getters loop over all VCs dynamically
  - Print methods display all VCs via loop
affects: [02-02, phase-03, phase-04, phase-05]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Loop over MAX_NUM_VCS instead of hardcoded VC indices", "Use {} for zero-init on MAX_NUM_VCS-sized arrays"]

key-files:
  created: []
  modified:
    - tt_metal/fabric/builder/fabric_builder_config.hpp
    - tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp
    - tt_metal/fabric/builder/fabric_remote_channels_allocator.hpp
    - tt_metal/fabric/erisc_datamover_builder.hpp
    - tests/tt_metal/tt_fabric/fabric_router/test_channel_trimming_capture.cpp

key-decisions:
  - "Fixed latent bug in test_channel_trimming_capture.cpp: CaptureResults template used num_max_receiver_channels instead of MAX_NUM_VCS for NUM_VC parameter (coincidentally equal at 2, diverged at 3)"

patterns-established:
  - "Loop-based VC iteration: always use for(vc=0; vc<MAX_NUM_VCS; ++vc) instead of hardcoded indices"
  - "Zero-init pattern: use = {} for MAX_NUM_VCS-sized arrays, never explicit element-count initializers"

requirements-completed: [CNST-01]

# Metrics
duration: 10min
completed: 2026-03-17
---

# Phase 02 Plan 01: Constants and Config Foundation Summary

**MAX_NUM_VCS bumped from 2 to 3 with all initializers, legacy getters, and print methods updated to be VC-count-agnostic**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-17T03:22:39Z
- **Completed:** 2026-03-17T03:32:48Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- MAX_NUM_VCS changed from 2 to 3, auto-propagating through all std::array declarations
- All five {0,0} explicit initializers converted to {} for VC-count independence
- Three legacy sum-of-two getters converted to loop-based summation over all VCs
- Print method in FabricStaticSizedChannelsAllocator updated to dynamically display all VCs

## Task Commits

Each task was committed atomically:

1. **Task 1: Bump MAX_NUM_VCS and fix all {0,0} initializers** - `d17f8c450e5` (feat)
2. **Task 2: Update legacy sum-of-two getters and hardcoded print methods** - `240cfb49a75` (feat)

## Files Created/Modified
- `tt_metal/fabric/builder/fabric_builder_config.hpp` - MAX_NUM_VCS constant changed from 2 to 3
- `tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp` - {0,0} to {}, legacy getters to loops, print method to dynamic loop
- `tt_metal/fabric/builder/fabric_remote_channels_allocator.hpp` - {0,0} to {}, legacy getter to loop
- `tt_metal/fabric/erisc_datamover_builder.hpp` - {0,0} to {} on per-VC channel count arrays
- `tests/tt_metal/tt_fabric/fabric_router/test_channel_trimming_capture.cpp` - Fixed CaptureResults template parameter (num_max_receiver_channels -> MAX_NUM_VCS)

## Decisions Made
- Fixed latent bug in test_channel_trimming_capture.cpp where CaptureResults used num_max_receiver_channels (2) as NUM_VC template parameter instead of MAX_NUM_VCS. With MAX_NUM_VCS=2 they were coincidentally equal; with MAX_NUM_VCS=3 the type mismatch caused a compilation error.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed CaptureResults template parameter in test_channel_trimming_capture.cpp**
- **Found during:** Task 1 (Bump MAX_NUM_VCS)
- **Issue:** Test used `FabricDatapathUsageL1Results<true, builder_config::num_max_receiver_channels, ...>` but the correct second template parameter is `MAX_NUM_VCS`, not `num_max_receiver_channels`. These were both 2 before, masking the bug.
- **Fix:** Changed to `FabricDatapathUsageL1Results<true, builder_config::MAX_NUM_VCS, builder_config::num_max_sender_channels>`
- **Files modified:** tests/tt_metal/tt_fabric/fabric_router/test_channel_trimming_capture.cpp
- **Verification:** Compilation succeeds with no type mismatch errors
- **Committed in:** d17f8c450e5 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential fix for compilation. No scope creep.

## Issues Encountered
- clang-format hook reformatted some lines on both commits (auto-resolved by re-staging)

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- MAX_NUM_VCS=3 is now the foundation constant, all downstream code compiles cleanly
- Ready for Plan 02-02 and subsequent phases that add VC2 logic
- VC2 array entries are zero-initialized, ensuring zero behavioral change until VC2 is explicitly populated

---
*Phase: 02-constants-config-foundation*
*Completed: 2026-03-17*
