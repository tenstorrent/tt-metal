---
phase: 03-api-detail-namespace
plan: 02
subsystem: fabric-api
tags: [refactor, namespace, header, linear, detail]
dependency_graph:
  requires: []
  provides: [linear/detail/api.h with 18 _single_packet definitions in detail:: namespace]
  affects: [tt_metal/fabric/hw/inc/linear/api.h, tt_metal/fabric/hw/inc/linear/detail/api.h]
tech_stack:
  added: [linear/detail/api.h]
  patterns: [detail:: namespace isolation, using-declaration for public re-export]
key_files:
  created:
    - tt_metal/fabric/hw/inc/linear/detail/api.h
  modified:
    - tt_metal/fabric/hw/inc/linear/api.h
decisions:
  - "Include detail/api.h after fabric_set_mcast_route in linear/api.h so route helpers are defined before detail:: non-template functions are parsed"
  - "Expose is_addrgen via 'using detail::is_addrgen' rather than duplicating definition"
  - "No 'using namespace detail' -- all calls use explicit detail:: qualification per user decision"
metrics:
  duration: "4m 23s"
  completed_date: "2026-03-12"
  tasks_completed: 2
  files_modified: 2
---

# Phase 3 Plan 02: Linear detail namespace extraction Summary

Extracted all 18 `_single_packet` function definitions from `linear/api.h` into a new `linear/detail/api.h` under the `tt::tt_fabric::linear::experimental::detail` namespace. Public auto-packetizing wrappers in `linear/api.h` now call `detail::`-qualified single-packet functions.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create linear/detail/api.h with _single_packet definitions in detail namespace | f9bb188874 | tt_metal/fabric/hw/inc/linear/detail/api.h (new, 657 lines) |
| 2 | Update linear/api.h to remove definitions and call detail:: | 0c8f189495 | tt_metal/fabric/hw/inc/linear/api.h (modified, -609 lines) |

## Changes Made

### Task 1: linear/detail/api.h (new file, 657 lines)
- Namespace: `tt::tt_fabric::linear::experimental::detail`
- Includes same headers as `linear/api.h` (packet_header_pool, tt_fabric_api, edm_fabric_worker_adapters, routing_plane_connection_manager, tt_fabric_mux, noc_addr, fabric_edm_packet_header, api_common, linear/addrgen_api)
- `is_addrgen` type trait copied into detail namespace
- All 9 function families (18 overloads) with full doc comments:
  - `fabric_unicast_noc_unicast_write_single_packet` (FabricSenderType + ConnectionManager)
  - `fabric_unicast_noc_scatter_write_single_packet` (FabricSenderType + ConnectionManager)
  - `fabric_unicast_noc_fused_scatter_write_atomic_inc_single_packet` (FabricSenderType + ConnectionManager)
  - `fabric_unicast_noc_fused_unicast_with_atomic_inc_single_packet` (FabricSenderType + ConnectionManager)
  - `fabric_multicast_noc_unicast_write_single_packet` (FabricSenderType + ConnectionManager)
  - `fabric_multicast_noc_scatter_write_single_packet` (FabricSenderType + ConnectionManager)
  - `fabric_multicast_noc_fused_unicast_with_atomic_inc_single_packet` (FabricSenderType + ConnectionManager)
  - `fabric_multicast_noc_fused_scatter_write_atomic_inc_single_packet` (FabricSenderType + ConnectionManager)
  - `fabric_sparse_multicast_noc_unicast_write_single_packet` (FabricSenderType + ConnectionManager) [linear-only 9th family]

### Task 2: linear/api.h (modified, -609 lines net)
- Removed all 18 `_single_packet` function definitions and their doc comment blocks
- Added `#include "tt_metal/fabric/hw/inc/linear/detail/api.h"` after `fabric_set_mcast_route` definition (line ~69 in new file) -- placed here intentionally so route helpers are defined before detail:: non-template functions are parsed
- Replaced `is_addrgen` two-part template definition with `using detail::is_addrgen;`
- Added `detail::` prefix to all 28 call sites in wrapper functions
- `fabric_set_unicast_route` and `fabric_set_mcast_route` remain in public namespace
- No `using namespace detail` -- explicit qualification throughout

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Include placement for route helper availability**
- **Found during:** Task 2 planning
- **Issue:** Plan specified adding `detail/api.h` include "before the `using namespace` line" (line 19), but `detail/api.h` contains non-template connection manager overloads that call `fabric_set_unicast_route` and `fabric_set_mcast_route` (defined at lines 29-70). Non-template unqualified name lookup happens at definition time, so those helpers must be declared before the include.
- **Fix:** Placed the include AFTER `fabric_set_mcast_route` definition (line ~69) so route helpers are visible when detail:: non-template functions are parsed.
- **Files modified:** tt_metal/fabric/hw/inc/linear/api.h
- **Commit:** 0c8f189495

## Verification Results

- `linear/detail/api.h` exists: PASS
- `namespace tt::tt_fabric::linear::experimental::detail`: PASS (2 occurrences: open + close)
- `linear/api.h` includes `linear/detail/api.h`: PASS
- Zero `FORCE_INLINE void ..._single_packet` definitions in `linear/api.h`: PASS (0 found)
- 28 `detail::..._single_packet` calls in `linear/api.h`: PASS
- No `using namespace detail` in `linear/api.h`: PASS
- `fabric_set_unicast_route` in public namespace: PASS
- `fabric_set_mcast_route` in public namespace: PASS
- Sparse multicast family in `detail/api.h`: PASS

## Self-Check: PASSED
