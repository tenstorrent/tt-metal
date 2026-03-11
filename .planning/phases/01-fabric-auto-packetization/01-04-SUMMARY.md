---
phase: 01-fabric-auto-packetization
plan: 04
subsystem: api
tags: [fabric, mesh, auto-packetization, unicast, multicast, fused-atomic-inc, single-packet]

# Dependency graph
requires:
  - phase: 01-fabric-auto-packetization/01
    provides: test infrastructure scaffolding
provides:
  - "_single_packet renames for 4 mesh raw-size function families (8 overloads total)"
  - "auto-packetizing wrappers for mesh unicast write, multicast write, fused unicast+atomic_inc, multicast fused unicast+atomic_inc"
  - "connection manager breadth-first wrappers with route-setup passes (Pattern 5)"
affects: [01-fabric-auto-packetization/05, 01-fabric-auto-packetization/06, 01-fabric-auto-packetization/07]

# Tech tracking
tech-stack:
  added: []
  patterns: [mesh-setroute-false-pattern, mesh-conn-mgr-route-setup-pass, multicast-mcast-route-setup]

key-files:
  created: []
  modified: [tt_metal/fabric/hw/inc/mesh/api.h]

key-decisions:
  - "Updated addrgen overload references to _single_packet (Rule 3 auto-fix, required for compilation)"
  - "Multicast conn mgr wrappers use per-slot ranges[i] for fabric_set_mcast_route in route-setup pass"

patterns-established:
  - "Pattern 5 (mesh conn mgr): route-setup for_each_header pass BEFORE chunk loop, SetRoute=false inside"
  - "Multicast fused wrappers: intermediate chunks as regular multicast unicast writes, final as fused"

requirements-completed: [AP-02, AP-04]

# Metrics
duration: 6min
completed: 2026-03-11
---

# Phase 01 Plan 04: Mesh API Auto-Packetization Summary

**Renamed 8 mesh raw-size functions to _single_packet and added 8 auto-packetizing wrappers with mesh-specific route setup (unicast/multicast SetRoute=false pattern)**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-11T01:16:58Z
- **Completed:** 2026-03-11T01:23:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Renamed 4 raw-size function families (8 overloads: template + conn mgr each) to _single_packet
- Added 8 auto-packetizing wrappers under original names with FABRIC_MAX_PACKET_SIZE chunking loops
- Template wrappers set route once (fabric_set_unicast_route / fabric_set_mcast_route) then use SetRoute=false
- Conn mgr wrappers use Pattern 5: route-setup for_each_header pass before chunk loop, SetRoute=false inside
- Fused wrappers send intermediate chunks as regular writes, final chunk as fused write+atomic_inc

## Task Commits

Each task was committed atomically:

1. **Task 1: Rename unicast_write and fused_unicast_with_atomic_inc** - `ca452c7d1d` (feat)
2. **Task 2: Rename multicast_unicast_write and multicast_fused_unicast_with_atomic_inc** - `15cdcdf499` (feat)

## Files Created/Modified
- `tt_metal/fabric/hw/inc/mesh/api.h` - Renamed 8 raw-size functions to _single_packet, added 8 auto-packetizing wrappers, updated addrgen references

## Decisions Made
- Updated addrgen overload call sites from `fabric_*<FabricSenderType, false>(...)` to `fabric_*_single_packet<FabricSenderType, false>(...)` -- these are single-packet calls within existing addrgen chunking loops, so _single_packet is semantically correct and required for compilation
- Multicast conn mgr wrappers access per-slot ranges via `ranges[i]` in both route-setup pass and chunk loop

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated addrgen overload references to _single_packet**
- **Found during:** Task 1 and Task 2
- **Issue:** Addrgen overloads in mesh/api.h call raw functions with `<FabricSenderType, false>` or `<false>` SetRoute parameter. After rename, these calls would fail because the new wrapper lacks SetRoute template param, and the _single_packet variant has a different name.
- **Fix:** Updated all addrgen references to use _single_packet variants: `fabric_unicast_noc_unicast_write_single_packet<FabricSenderType, false>`, `fabric_unicast_noc_fused_unicast_with_atomic_inc_single_packet<FabricSenderType, false>`, `fabric_multicast_noc_unicast_write_single_packet<FabricSenderType, false>`, `fabric_multicast_noc_fused_unicast_with_atomic_inc_single_packet<FabricSenderType, false>`
- **Files modified:** tt_metal/fabric/hw/inc/mesh/api.h (addrgen section lines 2500+)
- **Verification:** grep confirms no remaining old-name calls with `<..., false>` pattern
- **Committed in:** ca452c7d1d (Task 1), 15cdcdf499 (Task 2)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential for compilation. Addrgen overloads reference renamed functions. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- mesh/api.h unicast and multicast raw-size families complete
- Plan 05 (mesh scatter + fused-scatter renames) can proceed
- Plan 06 (mesh addrgen overloads) depends on _single_packet names from this plan
- Device compile check deferred to wave-level gate

---
*Phase: 01-fabric-auto-packetization*
*Completed: 2026-03-11*
