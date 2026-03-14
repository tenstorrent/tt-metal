---
phase: 05-channel-allocator
plan: 01
subsystem: fabric
tags: [fabric, channel-allocator, per-vc, cpp, blackhole]

# Dependency graph
requires:
  - phase: 04-device-sender
    provides: "VC0_SENDER_CHANNEL_START constant and sender VC indexing consistency"
provides:
  - "FabricStaticSizedChannelsAllocator::emit_channel_allocations_ct_args accepts per-VC arrays (no flat scalars)"
  - "FabricRemoteChannelsAllocator::emit_channel_allocations_ct_args accepts per-VC bool array"
  - "erisc_datamover_builder.cpp call sites pass per-VC arrays to both allocator emit calls"
affects: [future allocator API consumers, any phase that calls emit_channel_allocations_ct_args]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Allocator emit methods use per-VC arrays instead of flat scalar counts"
    - "Local scalar derivation at top of function body for backward-compatible internal logic"

key-files:
  created: []
  modified:
    - tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp
    - tt_metal/fabric/builder/fabric_static_sized_channels_allocator.cpp
    - tt_metal/fabric/builder/fabric_remote_channels_allocator.hpp
    - tt_metal/fabric/builder/fabric_remote_channels_allocator.cpp
    - tt_metal/fabric/erisc_datamover_builder.cpp

key-decisions:
  - "Derive local flat scalars at top of function body so downstream logic is unchanged — avoids touching the tag/num_entries/emit_ct_args/index-mapping code"
  - "Remote allocator iterates per-VC bool array to emit sequential entry indices, replacing the old scalar loop — semantically identical for current 2-VC layout"
  - "Call site packs existing local scalars into an array before passing — actual_sender_channels_vc0 and num_receiver_channels locals retained for use elsewhere in the function"

patterns-established:
  - "emit_channel_allocations_ct_args: always per-VC arrays, never flat scalars — all allocator emit methods now consistent with the constructor API"

requirements-completed: [CA-01, CA-02]

# Metrics
duration: 25min
completed: 2026-03-14
---

# Phase 05 Plan 01: Channel Allocator emit_channel_allocations_ct_args Per-VC API Summary

**Replaced flat scalar params with per-VC arrays in both allocator emit methods and updated the single call site, completing allocator API consistency so no method uses mixed flat/per-VC signatures.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-03-13T23:45:00Z
- **Completed:** 2026-03-14T00:02:00Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- `FabricStaticSizedChannelsAllocator::emit_channel_allocations_ct_args` now takes `(ct_args, num_used_sender_channels_per_vc, is_receiver_channel_active_per_vc)` — zero flat scalar params
- `FabricRemoteChannelsAllocator::emit_channel_allocations_ct_args` now takes `(ct_args, is_receiver_channel_active_per_vc)` — zero flat scalar count
- Single call site in `erisc_datamover_builder.cpp` packs scalars into array and passes per-VC arrays to both calls
- Build clean, sanity test passed: all 12 latency tests passed golden comparison, no hangs

## Task Commits

Each task was committed atomically:

1. **Task 1: Update allocator header declarations to per-VC signatures** - `67435625d10` (feat)
2. **Task 2: Update allocator .cpp bodies and builder call site** - `7acf4ce73f4` (feat)
3. **Task 3: Build and sanity test** - (verification only, no code changes)

## Files Created/Modified

- `tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp` - Updated `emit_channel_allocations_ct_args` declaration to per-VC arrays with expanded doc comment
- `tt_metal/fabric/builder/fabric_static_sized_channels_allocator.cpp` - Updated function definition; derives local scalars from per-VC arrays at top of body
- `tt_metal/fabric/builder/fabric_remote_channels_allocator.hpp` - Updated `emit_channel_allocations_ct_args` declaration to per-VC bool array
- `tt_metal/fabric/builder/fabric_remote_channels_allocator.cpp` - Updated definition; iterates per-VC bool array for sequential entry indices
- `tt_metal/fabric/erisc_datamover_builder.cpp` - Packs `actual_sender_channels_vc0/vc1` into array, passes `config.is_receiver_channel_active_per_vc` directly

## Decisions Made

- **Derive locals at function entry:** Rather than rewriting the body of `FabricStaticSizedChannelsAllocator::emit_channel_allocations_ct_args`, local variables `num_used_vc0_sender_channels`, `num_used_vc1_sender_channels`, `num_used_receiver_channels` are derived from the per-VC arrays at the top of the body. This keeps the tag/num_entries/mapping logic unchanged and minimizes diff surface.
- **Remote allocator uses per-VC iteration:** The old `for (i < num_used_receiver_channels)` identity loop is replaced with a per-VC iteration over the bool array, emitting sequential indices only for active VCs. Semantically identical for the current 2-VC layout but now correctly scoped per-VC.
- **Retain `num_receiver_channels` and `actual_sender_channels_vc0/vc1` locals in builder:** These variables are still used above the emit calls (NOC/cmd-buf loops), so they are not removed — only the passing of them into the allocator calls is changed.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- First sanity test run failed with `tt_tlb_alloc failed with error code -12` (TLB resource exhaustion — another process held the `CHIP_IN_USE` lock). This is a hardware resource conflict, not a code regression. Reset with `tt-smi -r` and reran successfully.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Allocator API is now fully per-VC across all methods (constructor + emit)
- All five allocator/builder files updated and verified clean
- Sanity test confirms no behavioral regression
- Ready for Phase 6 or next planned work

---
*Phase: 05-channel-allocator*
*Completed: 2026-03-14*
