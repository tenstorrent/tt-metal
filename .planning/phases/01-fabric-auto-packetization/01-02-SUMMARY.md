---
phase: 01-fabric-auto-packetization
plan: 02
subsystem: api
tags: [fabric, auto-packetization, linear-api, chunking, noc-write]

# Dependency graph
requires:
  - phase: 01-fabric-auto-packetization/01
    provides: test infrastructure scaffolding for compile-only validation
provides:
  - _single_packet renames for 5 function families in linear/api.h (unicast_write, fused_unicast_with_atomic_inc, multicast_unicast_write, multicast_fused_unicast_with_atomic_inc, sparse_multicast_unicast_write)
  - Auto-packetizing wrappers under original names with chunking loops
affects: [01-fabric-auto-packetization/03, 01-fabric-auto-packetization/04, 01-fabric-auto-packetization/05, 01-fabric-auto-packetization/07]

# Tech tracking
tech-stack:
  added: []
  patterns: [auto-packetizing wrapper with FABRIC_MAX_PACKET_SIZE chunking loop, breadth-first connection manager pattern, fused-op final-chunk-only atomic_inc]

key-files:
  created: []
  modified: [tt_metal/fabric/hw/inc/linear/api.h]

key-decisions:
  - "Sparse multicast wrapper is passthrough (no chunking) with doc comment explaining limitation"
  - "Fused op wrappers send intermediate chunks as regular writes, final chunk as fused with atomic_inc"
  - "Connection manager wrappers use breadth-first: for_each_header inside chunk loop"

patterns-established:
  - "Auto-packetizing wrapper: extract noc_address from command header, while(remaining > FABRIC_MAX_PACKET_SIZE) loop, noc_async_writes_flushed() before every _single_packet call"
  - "Breadth-first conn mgr: PacketHeaderPool::for_each_header inside chunk loop, not outside"
  - "Fused-op chunking: intermediate chunks use regular write _single_packet, final chunk uses fused _single_packet"

requirements-completed: [AP-02, AP-03]

# Metrics
duration: 5min
completed: 2026-03-11
---

# Phase 01 Plan 02: Linear API Unicast + Multicast Renames Summary

**Renamed 5 raw-size function families in linear/api.h to _single_packet (10 overloads) and added 10 auto-packetizing wrappers with FABRIC_MAX_PACKET_SIZE chunking loops**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-11T01:16:45Z
- **Completed:** 2026-03-11T01:22:03Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Renamed all 5 function families (unicast_write, fused_unicast_with_atomic_inc, multicast_unicast_write, multicast_fused_unicast_with_atomic_inc, sparse_multicast_unicast_write) to _single_packet suffix (10 overloads total: 2 per family)
- Added 10 auto-packetizing wrapper functions under original names
- Unicast and multicast wrappers have proper chunking loops; sparse multicast is passthrough with documented limitation
- Fused operation wrappers correctly send intermediate chunks as regular writes, atomic_inc fires only on final chunk
- Connection manager wrappers follow breadth-first pattern (for_each_header inside chunk loop)
- All wrappers call noc_async_writes_flushed() before every _single_packet invocation

## Task Commits

Each task was committed atomically:

1. **Task 1: Rename unicast_write and fused_unicast_with_atomic_inc** - `d866fbf21e` (feat)
2. **Task 2: Rename multicast_write, sparse_multicast, multicast_fused** - `a11ce430b6` (feat)

## Files Created/Modified
- `tt_metal/fabric/hw/inc/linear/api.h` - Renamed 10 functions to _single_packet, added 10 auto-packetizing wrappers in the raw section (before Addrgen API Overloads divider)

## Decisions Made
- Sparse multicast wrapper is a passthrough (calls _single_packet directly, no chunking loop) because sparse multicast uses pre-computed hop bitmasks that cannot be safely chunked. Documented in API comment.
- Fused operation wrappers use regular write _single_packet for intermediate chunks and fused _single_packet for the final chunk, ensuring atomic_inc fires exactly once.
- Connection manager wrappers place for_each_header inside the chunk loop (breadth-first), matching the existing addrgen connection manager pattern.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- linear/api.h unicast + multicast families complete
- Plans 03 (scatter families) can proceed on linear/api.h
- Plans 04-05 can proceed on mesh/api.h
- Device compile validation deferred to wave-level gate after all wave 1 plans complete

## Self-Check: PASSED

---
*Phase: 01-fabric-auto-packetization*
*Completed: 2026-03-11*
