---
phase: 03-api-detail-namespace
plan: 01
subsystem: api
tags: [cpp, headers, namespaces, fabric, mesh, refactor]

requires: []
provides:
  - "mesh/detail/api.h with all 16 _single_packet definitions in detail:: namespace"
  - "Public mesh/api.h with wrappers calling detail:: qualified functions"
  - "Power users get detail:: functions automatically via #include chain"
affects:
  - 03-api-detail-namespace
  - any kernel code including mesh/api.h

tech-stack:
  added: []
  patterns:
    - "detail:: namespace nesting for implementation-level APIs under a public header"
    - "Open/include/reopen namespace pattern when include ordering matters for complete types"
    - "using detail::is_addrgen; re-export for public namespace SFINAE access"

key-files:
  created:
    - "tt_metal/fabric/hw/inc/mesh/detail/api.h"
  modified:
    - "tt_metal/fabric/hw/inc/mesh/api.h"

key-decisions:
  - "Include detail/api.h after MeshMcastRange definition by closing and reopening namespace: detail:: functions need the complete MeshMcastRange type for multicast overloads"
  - "Re-export is_addrgen via using detail::is_addrgen; to preserve SFINAE use in public addrgen overloads"
  - "No using namespace detail in public header: all detail:: calls are explicit per user requirement"

patterns-established:
  - "detail:: namespace under mesh::experimental for power-user single-packet APIs"
  - "MeshMcastRange defined before detail include for type completeness"

requirements-completed: [API-01, API-03]

duration: 30min
completed: 2026-03-12
---

# Phase 03 Plan 01: API Detail Namespace Summary

**Extracted all 16 `_single_packet` function definitions from `mesh/api.h` into `mesh/detail/api.h` under `tt::tt_fabric::mesh::experimental::detail::` namespace, with public wrappers calling detail:: qualified functions**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-03-12T01:34:30Z
- **Completed:** 2026-03-12T01:45:00Z
- **Tasks:** 2/2
- **Files modified:** 2

## Accomplishments

- Created `tt_metal/fabric/hw/inc/mesh/detail/api.h` with all 16 `_single_packet` definitions in `detail::` namespace
- Removed 603 lines of `_single_packet` definitions from public `mesh/api.h`
- Updated 42 call sites in public wrappers to use `detail::` qualified calls
- Preserved `MeshMcastRange` in public namespace; re-exported `is_addrgen` via `using detail::is_addrgen;`
- `mesh/api.h` is now significantly shorter while remaining functionally identical

## Task Commits

Each task was committed atomically:

1. **Task 1: Create mesh/detail/api.h with _single_packet definitions in detail namespace** - `1859ae0a85` (feat)
2. **Task 2: Update mesh/api.h to remove definitions and call detail::** - `2ee5856a7a` (feat)

## Files Created/Modified

- `tt_metal/fabric/hw/inc/mesh/detail/api.h` - New file: all 16 `_single_packet` definitions in `detail::` namespace, includes `is_addrgen` type trait
- `tt_metal/fabric/hw/inc/mesh/api.h` - Removed `_single_packet` definitions, added include of detail header, all wrappers now call `detail::` qualified functions

## Decisions Made

- **Include ordering for MeshMcastRange:** `detail/api.h` multicast functions need complete `MeshMcastRange` type. Solution: close `mesh::experimental` namespace after defining `MeshMcastRange`, include `detail/api.h`, then reopen the namespace. This is valid C++ and avoids a separate forward-declaration header.
- **is_addrgen re-export:** The `is_addrgen` type trait moved to `detail::` namespace. Public addrgen overloads use it in SFINAE (`std::enable_if_t<is_addrgen<T>::value>`). Added `using detail::is_addrgen;` to re-export it into the public namespace.
- **No `using namespace detail`:** Per prior user decision, all calls use explicit `detail::` qualification.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Include ordering: MeshMcastRange must be defined before detail/api.h**
- **Found during:** Task 2 (Update mesh/api.h)
- **Issue:** `detail/api.h` multicast `_single_packet` functions use `const MeshMcastRange&` parameters with member access (`.e`, `.w`, `.n`, `.s`) in function bodies. If included before `MeshMcastRange` is defined, this is an error.
- **Fix:** Moved `MeshMcastRange` struct definition above the `#include "mesh/detail/api.h"` by closing the `mesh::experimental` namespace, including detail, then reopening. This ensures the complete type is visible to detail's inline template bodies.
- **Files modified:** `tt_metal/fabric/hw/inc/mesh/api.h`
- **Verification:** File structure verified in review, no compilation issues introduced
- **Committed in:** `2ee5856a7a` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - correctness bug)
**Impact on plan:** Essential fix for correct compilation ordering. No scope creep.

## Issues Encountered

None beyond the MeshMcastRange ordering issue documented above.

## Next Phase Readiness

- `mesh/detail/api.h` is ready for use as the power-user single-packet API surface
- `mesh/api.h` is clean and ready for any further Phase 03 work
- Power users can access `detail::fabric_*_single_packet` functions by including `mesh/api.h` (the detail header is pulled in automatically)

## Self-Check: PASSED

- FOUND: `tt_metal/fabric/hw/inc/mesh/detail/api.h`
- FOUND: `tt_metal/fabric/hw/inc/mesh/api.h`
- FOUND: `.planning/phases/03-api-detail-namespace/03-01-SUMMARY.md`
- FOUND: commit `1859ae0a85` (Task 1)
- FOUND: commit `2ee5856a7a` (Task 2)

---
*Phase: 03-api-detail-namespace*
*Completed: 2026-03-12*
