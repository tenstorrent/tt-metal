---
phase: 02-constants-config-foundation
plan: 02
subsystem: fabric
tags: [vc2, buffer-slots, intermesh-config, fabric-builder]

# Dependency graph
requires:
  - phase: 02-constants-config-foundation/02-01
    provides: MAX_NUM_VCS=3 constant, VC2 stream register IDs
provides:
  - Extended PerVcBufferSlots struct with 6 fields (VC0/VC1/VC2 sender+receiver)
  - Split buffer slot option tables (vc0_only, vc0_vc1, vc0_vc1_vc2)
  - num_active_vcs-based table selection in get_num_buffer_slots
  - IntermeshVCConfig.requires_vc2 field and predicate
  - FabricBuilderContext::requires_vc2() accessor
affects: [03-channel-allocation, 04-builder-wiring, 05-edm-integration]

# Tech tracking
tech-stack:
  added: []
  patterns: [split-table-by-vc-count, vc2-mirrors-vc1-slots]

key-files:
  created: []
  modified:
    - tt_metal/fabric/builder/fabric_static_sized_channels_allocator.cpp
    - tt_metal/fabric/fabric_builder_context.hpp
    - tt_metal/fabric/fabric_builder_context.cpp

key-decisions:
  - "VC2 buffer slots mirror VC1 values in the vc0_vc1_vc2 table"
  - "VC2 predicate: requires_vc1 AND Blackhole AND not UDM AND not MUX"
  - "Split mesh tables by active VC count instead of mixed VC0-only and VC0+VC1 rows in one table"

patterns-established:
  - "Split buffer slot tables by VC configuration group (vc0_only, vc0_vc1, vc0_vc1_vc2)"
  - "num_active_vcs computed from per-vc channel counts at callsite"

requirements-completed: [CNST-02, CNST-03, CONF-01, CONF-02, CONF-03]

# Metrics
duration: 5min
completed: 2026-03-17
---

# Phase 2 Plan 02: Constants and Config Foundation Summary

**Extended PerVcBufferSlots with VC2 fields, split buffer tables into 3 groups by VC configuration, added requires_vc2 predicate to IntermeshVCConfig with Blackhole+no-UDM/MUX guard**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-17T03:38:30Z
- **Completed:** 2026-03-17T03:43:10Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- PerVcBufferSlots extended to 6 fields (vc0/vc1/vc2 sender + receiver slots)
- Buffer slot option tables split into 3 groups: vc0_only_mesh, vc0_vc1_mesh, vc0_vc1_vc2_mesh
- Table selection driven by num_active_vcs computed from per-vc channel counts
- IntermeshVCConfig gains requires_vc2 field (default false) and VC2 predicate in compute_intermesh_vc_config()
- FabricBuilderContext exposes requires_vc2() accessor
- Zero behavioral change: all existing configs have VC2 channels = 0

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend PerVcBufferSlots and split buffer slot option tables** - `eafd0f95c4a` (feat)
2. **Task 2: Add VC2 enable predicate to IntermeshVCConfig and FabricBuilderContext** - `611d8209815` (feat)

## Files Created/Modified
- `tt_metal/fabric/builder/fabric_static_sized_channels_allocator.cpp` - Extended PerVcBufferSlots, split buffer tables, updated get_optimal_num_slots_per_vc with VC2 params
- `tt_metal/fabric/fabric_builder_context.hpp` - Added requires_vc2 field to IntermeshVCConfig, requires_vc2() accessor
- `tt_metal/fabric/fabric_builder_context.cpp` - VC2 predicate computation in compute_intermesh_vc_config()

## Decisions Made
- VC2 buffer slot values mirror VC1 in the vc0_vc1_vc2 table (same capacity split)
- VC2 predicate requires all of: VC1 active, Blackhole arch, non-UDM, non-MUX tensix config
- Split tables cleanly by VC count rather than mixing VC0-only and multi-VC rows in one table

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All data structures now support 3 VCs
- System knows when VC2 should be enabled via requires_vc2() predicate
- Ready for Phase 3 (channel allocation) which will wire VC2 channels into the allocation path

---
*Phase: 02-constants-config-foundation*
*Completed: 2026-03-17*
