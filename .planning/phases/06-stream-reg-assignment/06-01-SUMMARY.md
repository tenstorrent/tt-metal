---
phase: 06-stream-reg-assignment
plan: 01
subsystem: fabric
tags: [cpp, stream-registers, vc, host-side, compile-time-args]

# Dependency graph
requires:
  - phase: 05-channel-allocator
    provides: Per-VC channel allocator emit API used in erisc_datamover_builder.cpp
provides:
  - Per-VC grouping arrays in StreamRegAssignments (five new static constexpr array members)
  - CT-arg emission in get_compile_time_args() using per-VC accessors instead of positional flat-array indexing
affects: [future phases using StreamRegAssignments or CT-arg emission]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-VC array member pattern: std::array<std::array<uint32_t, N_CH>, N_VC> indexed by [vc][vc_relative_channel]"
    - "CT-arg emission via named per-VC accessors rather than positional stream_ids[N]"

key-files:
  created: []
  modified:
    - tt_metal/fabric/erisc_datamover_builder.hpp
    - tt_metal/fabric/erisc_datamover_builder.cpp

key-decisions:
  - "Inner array dimension set to 4 (2D mesh max channels per VC) — z-router VC0 has 5 but uses named constants directly; per-VC arrays sized for 2D case only"
  - "VC1 pkts-acked row is all-zero placeholders since VC1 has no first-level acks — semantically correct and avoids special-casing downstream"
  - "get_all_stream_ids() retained unchanged to preserve backward compatibility with any callers outside CT-arg emission"

patterns-established:
  - "Per-VC array accessors: to_receiver_pkts_sent_ids_per_vc[vc], sender_channel_free_slots_stream_ids_per_vc[vc][ch]"
  - "CT-arg emission names per-VC arrays explicitly rather than flat index offsets"

requirements-completed: [SR-01, SR-02]

# Metrics
duration: 25min
completed: 2026-03-14
---

# Phase 6 Plan 01: Stream Register Assignment Summary

**Per-VC grouping arrays added to StreamRegAssignments; CT-arg emission in get_compile_time_args() rewritten to use named per-VC array accessors eliminating all positional stream_ids[N] flat-index accesses**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-03-14T00:20:00Z
- **Completed:** 2026-03-14T00:41:42Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added five new `static constexpr` per-VC array members to `StreamRegAssignments`: `to_receiver_pkts_sent_ids_per_vc`, `to_sender_pkts_acked_ids_per_vc`, `to_sender_pkts_completed_ids_per_vc`, `vc_free_slots_from_downstream_edge_ids`, `sender_channel_free_slots_stream_ids_per_vc`
- Replaced 33 positional `stream_ids[N]` accesses in the CT-arg emission block with named `StreamRegAssignments::*_per_vc` array references
- All existing named scalar `static constexpr` members preserved (backward compatible)
- `get_all_stream_ids()` preserved
- Wire format preserved: CT arg name strings and stream ID numerical values unchanged
- Build: PASSED (zero new errors)
- Sanity test: PASSED (all 12 latency golden comparisons passed, no hangs)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add per-VC grouping arrays to StreamRegAssignments** - `1735d02ede3` (feat)
2. **Task 2: Update CT-arg emission to use per-VC accessors, then build and sanity test** - `b2c9b76e996` (feat)

## Files Created/Modified
- `tt_metal/fabric/erisc_datamover_builder.hpp` - Added five per-VC grouping array members to `StreamRegAssignments` struct
- `tt_metal/fabric/erisc_datamover_builder.cpp` - Replaced `stream_ids[N]` flat-index CT-arg emission with `StreamRegAssignments::*_per_vc` accessor references

## Decisions Made
- Inner array dimension set to 4 for all per-VC sender arrays (2D mesh max channels per VC); z-router VC0 has 5 sender channels but uses named constants directly and does not need these per-VC arrays
- VC1 `to_sender_pkts_acked_ids_per_vc[1]` row initialized to `{0, 0, 0, 0}` placeholders since VC1 has no first-level acks — avoids breaking the array shape and keeps the pattern uniform
- `get_all_stream_ids()` retained unchanged since plan specified it must remain; the `const auto& stream_ids` local variable in `get_compile_time_args()` was removed (it was only used for the replaced emission block)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- First sanity test run failed with TLB allocation error (`tt_tlb_alloc failed with error code -12`) — this was a device resource exhaustion from a prior test run, not caused by code changes. Board reset via `tt-smi -r 0,1,2,3` resolved it and the rerun passed cleanly.
- clang-format reformatted array initializers (multi-element arrays split to one-element-per-line) on both commits; staged the reformatted result and committed successfully.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `StreamRegAssignments` now expresses per-VC structure at the type level, enabling downstream phases to reference stream IDs by VC and channel index rather than magic flat offsets
- All existing external references (e.g., `fabric_tensix_builder_impl.cpp`) compile unmodified since named scalar members were only added to, never removed

---
*Phase: 06-stream-reg-assignment*
*Completed: 2026-03-14*
