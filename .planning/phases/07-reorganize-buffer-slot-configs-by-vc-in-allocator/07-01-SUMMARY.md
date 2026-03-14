---
phase: 07-reorganize-buffer-slot-configs-by-vc-in-allocator
plan: 01
subsystem: fabric-allocator
tags: [per-vc, refactor, buffer-slots, allocator, cpp]

# Dependency graph
requires:
  - phase: 06-stream-reg-assignment-per-vc
    provides: "Per-VC array pattern established across all fabric subsystems"
provides:
  - "VcSlotConfig struct for per-VC buffer slot configuration"
  - "BufferSlotAllocation struct for aggregate slot allocation result"
  - "Array-of-struct indexed tables replacing PerVcBufferSlots flat struct"
  - "configure_buffer_slots_helper returns value instead of 4 output ref params"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: ["array-of-struct per-VC indexing in allocator slot tables"]

key-files:
  created: []
  modified:
    - "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
    - "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.cpp"

key-decisions:
  - "VcSlotConfig uses sender_slots/receiver_slots field names for brevity"
  - "BufferSlotAllocation uses num_sender_buffer_slots naming to match existing usage patterns"
  - "Constructor uses reference aliases to BufferSlotAllocation fields for minimal downstream diff"

patterns-established:
  - "Per-VC slot config tables use std::array<VcSlotConfig, MAX_NUM_VCS> entries"
  - "Slot selection lambda returns per-VC array instead of writing output references"

requirements-completed: []

# Metrics
duration: 7min
completed: 2026-03-14
---

# Phase 7 Plan 01: Reorganize Buffer Slot Configs Summary

**Replaced PerVcBufferSlots with VcSlotConfig array-of-struct and BufferSlotAllocation return type in allocator**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-14T03:41:36Z
- **Completed:** 2026-03-14T03:48:56Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Eliminated PerVcBufferSlots struct with named vc0/vc1 fields, replaced by VcSlotConfig indexed by VC
- All 3 static slot option tables converted to std::array<VcSlotConfig, MAX_NUM_VCS> format
- get_optimal_num_slots_per_vc returns std::array<VcSlotConfig, MAX_NUM_VCS> instead of writing to 8 output ref scalars
- configure_buffer_slots_helper returns BufferSlotAllocation instead of taking 4 output reference arrays
- CT args wire format verified unchanged by sanity test (all 12 golden comparisons passed)

## Task Commits

Each task was committed atomically:

1. **Task 1: Define VcSlotConfig, convert tables and get_optimal_num_slots_per_vc** - `76cb39618ca` (feat)
2. **Task 2: Restructure configure_buffer_slots_helper and run sanity test** - `beadd4df5a9` (feat)

## Files Created/Modified
- `tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp` - Added VcSlotConfig and BufferSlotAllocation structs, updated configure_buffer_slots_helper declaration
- `tt_metal/fabric/builder/fabric_static_sized_channels_allocator.cpp` - Converted all tables, lambda, helper, and constructor to use new types

## Decisions Made
- VcSlotConfig uses concise `sender_slots`/`receiver_slots` field names
- BufferSlotAllocation uses `num_sender_buffer_slots` etc. for consistency with existing member variable naming
- Constructor creates reference aliases (`auto& num_sender_buffer_slots_per_vc = slot_alloc.num_sender_buffer_slots`) to minimize diff in downstream slot-to-size computation loops
- Used `VcSlotConfigArray` type alias inside configure_buffer_slots_helper for readability of table types

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 7 complete - all per-VC refactor phases (1-7) finished
- Allocator internals now fully consistent with the per-VC array pattern used across all fabric subsystems

---
*Phase: 07-reorganize-buffer-slot-configs-by-vc-in-allocator*
*Completed: 2026-03-14*
