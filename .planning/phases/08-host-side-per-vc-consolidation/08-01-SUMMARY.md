---
phase: 08-host-side-per-vc-consolidation
plan: 01
subsystem: fabric
tags: [per-vc, constexpr-array, fabric-builder-config, channel-mapping]

requires:
  - phase: 07-reorganize-buffer-slot-configs-by-vc-in-allocator
    provides: "per-VC array-of-struct pattern in allocator slot configs"
provides:
  - "num_sender_channels_z_router_per_vc and num_downstream_edms_2d_per_vc constexpr arrays"
  - "get_downstream_edm_count_for_vc(vc, is_2D) unified function"
  - "initialize_vc_mappings(vc) unified method"
affects: [08-02-call-site-migration]

tech-stack:
  added: []
  patterns: ["per-VC constexpr array constants with std::array<T, MAX_NUM_VCS>", "unified VC-parameterized functions replacing split vc0/vc1 pairs"]

key-files:
  created: []
  modified:
    - tt_metal/fabric/builder/fabric_builder_config.hpp
    - tt_metal/fabric/builder/fabric_builder_config.cpp
    - tt_metal/fabric/fabric_router_channel_mapping.hpp
    - tt_metal/fabric/fabric_router_channel_mapping.cpp
    - tt_metal/fabric/fabric_tensix_builder.cpp
    - tt_metal/fabric/builder/router_connection_mapping.cpp
    - tests/tt_metal/tt_fabric/fabric_router/test_connection_mapping_logic.cpp
    - tests/tt_metal/tt_fabric/fabric_router/test_connection_registry.cpp
    - tests/tt_metal/tt_fabric/fabric_router/test_z_router_integration.cpp

key-decisions:
  - "Asymmetric constants (num_downstream_edms_vc0, num_downstream_edms_2d_vc1_with_z) kept standalone per CONTEXT.md decision"
  - "initialize_vc_mappings uses if(vc==0)/else branching to preserve distinct VC0/VC1 logic paths"

patterns-established:
  - "Per-VC array pattern: std::array<std::size_t, MAX_NUM_VCS> for symmetric VC constant pairs"
  - "Unified VC function pattern: parameterize by vc index with switch/if branching for VC-specific logic"

requirements-completed: ["Phase 8 goal"]

duration: 7min
completed: 2026-03-14
---

# Phase 8 Plan 01: Merge Symmetric VC Constants and Functions Summary

**Merged _vc0/_vc1 constant pairs into per_vc arrays, unified split downstream EDM count and channel mapping init functions into VC-parameterized APIs**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-14T15:54:40Z
- **Completed:** 2026-03-14T16:01:34Z
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments
- Replaced num_sender_channels_z_router_vc0/_vc1 with num_sender_channels_z_router_per_vc array
- Replaced num_downstream_edms_2d_vc0/_vc1 with num_downstream_edms_2d_per_vc array
- Unified get_vc0/get_vc1_downstream_edm_count into get_downstream_edm_count_for_vc(vc, is_2D)
- Merged initialize_vc0_mappings/initialize_vc1_mappings into initialize_vc_mappings(vc)
- Build passes cleanly, all 12 latency golden comparisons pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Merge constants and function in fabric_builder_config** - `112e668145b` (feat)
2. **Task 2: Merge channel mapping init functions** - `3e47f9051d5` (feat)
3. **Task 3: Build and sanity test** - `32c8a299d8d` (fix - test file constant references)

**Plan metadata:** (pending)

## Files Created/Modified
- `tt_metal/fabric/builder/fabric_builder_config.hpp` - Per-VC array constants, unified function declaration
- `tt_metal/fabric/builder/fabric_builder_config.cpp` - get_downstream_edm_count_for_vc implementation
- `tt_metal/fabric/fabric_router_channel_mapping.hpp` - initialize_vc_mappings(vc) declaration
- `tt_metal/fabric/fabric_router_channel_mapping.cpp` - Unified channel mapping init, updated constant refs
- `tt_metal/fabric/fabric_tensix_builder.cpp` - Updated call from get_vc0 to get_downstream_edm_count_for_vc(0,...)
- `tt_metal/fabric/builder/router_connection_mapping.cpp` - Updated constant ref to per_vc array
- `tests/tt_metal/tt_fabric/fabric_router/test_connection_mapping_logic.cpp` - Updated constant refs
- `tests/tt_metal/tt_fabric/fabric_router/test_connection_registry.cpp` - Updated constant refs
- `tests/tt_metal/tt_fabric/fabric_router/test_z_router_integration.cpp` - Updated constant refs

## Decisions Made
- Asymmetric constants (num_downstream_edms_vc0, num_downstream_edms_2d_vc1_with_z) kept standalone since they have no symmetric counterpart, per CONTEXT.md decision
- initialize_vc_mappings uses if(vc==0)/else branching to preserve distinct VC0 and VC1 logic paths clearly

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed test file references to removed constants**
- **Found during:** Task 3 (Build and sanity test)
- **Issue:** 3 test files referenced removed _vc0/_vc1 constants, causing compilation errors
- **Fix:** Updated all references to use _per_vc array indexing
- **Files modified:** test_connection_mapping_logic.cpp, test_connection_registry.cpp, test_z_router_integration.cpp
- **Verification:** Build passes, all tests compile
- **Committed in:** 32c8a299d8d (Task 3 commit)

**2. [Rule 3 - Blocking] Fixed call sites referencing removed functions**
- **Found during:** Task 1 (during constant/function merge)
- **Issue:** fabric_tensix_builder.cpp and router_connection_mapping.cpp referenced removed get_vc0_downstream_edm_count and num_downstream_edms_2d_vc0
- **Fix:** Updated to use get_downstream_edm_count_for_vc(0,...) and num_downstream_edms_2d_per_vc[0]
- **Files modified:** fabric_tensix_builder.cpp, router_connection_mapping.cpp
- **Verification:** Build passes
- **Committed in:** 112e668145b (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both auto-fixes necessary for compilation. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Per-VC API (arrays + unified functions) established and validated
- Ready for plan 02 call site migration to consume the new per-VC APIs

---
*Phase: 08-host-side-per-vc-consolidation*
*Completed: 2026-03-14*
