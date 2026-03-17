---
phase: 03-channel-mapping-allocation
plan: 01
subsystem: fabric
tags: [vc2, channel-mapping, fabric-builder, constants]

# Dependency graph
requires:
  - phase: 02-constants-config-foundation
    provides: MAX_NUM_VCS=3, IntermeshVCConfig.requires_vc2, PerVcBufferSlots VC2 fields
provides:
  - VC2 channel count constants (num_sender_channels_vc2, num_receiver_channels_vc2, num_sender_channels_z_router_vc2)
  - Updated aggregate channel counts (num_sender_channels_2d=9, num_sender_channels_z_router=10, num_receiver_channels_2d=3)
  - initialize_vc2_mappings() method with Z-router sender-only and mesh sender+receiver paths
  - get_num_virtual_channels() returns 3 when requires_vc2
  - get_num_sender_channels_for_vc(2) returns 1 when VC2 active
  - compute_max_channel_counts() picks up VC2 automatically via existing loop
affects: [03-02, allocator, edm-config]

# Tech tracking
tech-stack:
  added: []
  patterns: [vc2-channel-mapping-gated-by-requires_vc2, z-router-sender-only-for-vc2]

key-files:
  created: []
  modified:
    - tt_metal/fabric/builder/fabric_builder_config.hpp
    - tt_metal/fabric/fabric_router_channel_mapping.hpp
    - tt_metal/fabric/fabric_router_channel_mapping.cpp

key-decisions:
  - "VC2 mesh receiver at internal channel index 2 (after VC0=0, VC1=1)"
  - "VC2 sender always at last flat index after VC0+VC1 senders"
  - "No changes needed in compute_max_channel_counts -- existing loop picks up VC2 via updated get_num_virtual_channels"

patterns-established:
  - "VC2 mapping pattern: gated by intermesh_vc_config_->requires_vc2, Z-router gets sender-only"

requirements-completed: [CMAP-01, CMAP-02, CMAP-03, BLDR-04]

# Metrics
duration: 15min
completed: 2026-03-17
---

# Phase 03 Plan 01: Channel Mapping Allocation Summary

**VC2 channel constants, initialize_vc2_mappings() with Z-router sender-only and mesh sender+receiver, updated query methods returning 3 VCs**

## Performance

- **Duration:** 15 min
- **Started:** 2026-03-17T15:20:51Z
- **Completed:** 2026-03-17T15:35:51Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added VC2 constants (sender=1, receiver=1, z_router_sender=1) and updated aggregates (2d senders 8->9, z_router senders 9->10, 2d receivers 2->3)
- Implemented initialize_vc2_mappings() with correct gating: Z-router creates sender-only at flat index 9, mesh creates sender+receiver
- Updated get_num_virtual_channels() to return 3 when requires_vc2 is true
- Added case 2 to get_num_sender_channels_for_vc() for VC2 handling
- Verified compute_max_channel_counts() picks up VC2 automatically (no code changes needed)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add VC2 constants and update aggregate channel counts** - `4c8a9b4c750` (feat)
2. **Task 2: Add initialize_vc2_mappings() and update query methods** - `884922798f8` (feat)

## Files Created/Modified
- `tt_metal/fabric/builder/fabric_builder_config.hpp` - VC2 constants and updated aggregate totals
- `tt_metal/fabric/fabric_router_channel_mapping.hpp` - initialize_vc2_mappings() declaration
- `tt_metal/fabric/fabric_router_channel_mapping.cpp` - VC2 mapping implementation, updated get_num_virtual_channels and get_num_sender_channels_for_vc

## Decisions Made
- VC2 mesh receiver at internal channel index 2 (after VC0=0, VC1=1) -- follows existing sequential pattern
- VC2 sender placed at last flat index after all VC0+VC1 senders -- consistent with VC1 placement pattern
- No changes needed in compute_max_channel_counts() or fabric_builder_context.cpp -- the existing loop over get_num_virtual_channels() and get_num_sender_channels_for_vc() naturally picks up VC2

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- VC2 channel mappings are complete and ready for plan 03-02 (unit tests)
- All 3 files compile cleanly with fabric_unit_tests target
- Existing VC0/VC1 mappings unchanged (no regression risk)

---
*Phase: 03-channel-mapping-allocation*
*Completed: 2026-03-17*
