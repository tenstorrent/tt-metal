---
phase: 04-test-infrastructure-cleanup
plan: 01
subsystem: testing
tags: [tt-fabric, device-kernels, cpp, macros, boilerplate-reduction]

# Dependency graph
requires: []
provides:
  - tx_kernel_common.h shared header with includes, namespace declarations, and TX_KERNEL_PARSE_UNICAST_ARGS/TX_KERNEL_SETUP/TX_KERNEL_TEARDOWN macros
  - All 9 auto-packetization TX kernels reduced to minimal unique content
affects:
  - 04-02-test-infrastructure-cleanup

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Shared device kernel header with FABRIC_2D-conditional macros for RT arg parsing"
    - "#pragma once header guard for device kernel headers"
    - "TX_KERNEL_SETUP / TX_KERNEL_TEARDOWN pattern for common sender lifecycle"

key-files:
  created:
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/tx_kernel_common.h
  modified:
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/unicast_tx_writer_raw.cpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/scatter_unicast_tx_writer_raw.cpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/fused_atomic_inc_unicast_tx_writer_raw.cpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/fused_scatter_atomic_inc_unicast_tx_writer_raw.cpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/multicast_tx_writer_raw.cpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/scatter_multicast_tx_writer_raw.cpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/fused_atomic_inc_multicast_tx_writer_raw.cpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/fused_scatter_atomic_inc_multicast_tx_writer_raw.cpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/sparse_multicast_tx_writer_raw.cpp

key-decisions:
  - "Defined TX_KERNEL_PARSE_UNICAST_ARGS as two separate #ifdef FABRIC_2D branches rather than nested conditional macros to keep each branch linear and readable"
  - "Multicast kernels receive includes/namespace only from the header; kernel_main() is untouched to preserve the complex 4-direction sender/header/hop logic"
  - "sparse_multicast is treated as a multicast kernel (header-only, no unicast macros) because it has its own arg parsing that does not match the unicast pattern"

patterns-established:
  - "TX_KERNEL_PARSE_UNICAST_ARGS / TX_KERNEL_SETUP / TX_KERNEL_TEARDOWN: standard macro pattern for unicast TX device kernels"
  - "Shared header provides both includes and macros; kernel files contain only their unique API logic"

requirements-completed: [TEST-01]

# Metrics
duration: 3min
completed: 2026-03-12
---

# Phase 04 Plan 01: TX Kernel Common Header Summary

**Extracted 17-line boilerplate into tx_kernel_common.h with TX_KERNEL_PARSE_UNICAST_ARGS / TX_KERNEL_SETUP / TX_KERNEL_TEARDOWN macros; all 9 auto-packetization TX kernels now include only the header and their unique fabric API calls**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-12T14:13:32Z
- **Completed:** 2026-03-12T14:17:07Z
- **Tasks:** 2
- **Files modified:** 10 (1 created + 9 modified)

## Accomplishments
- Created `tx_kernel_common.h` with shared includes block, FABRIC_2D-conditional namespace declarations, and three macros (TX_KERNEL_PARSE_UNICAST_ARGS, TX_KERNEL_SETUP, TX_KERNEL_TEARDOWN)
- 4 unicast kernels reduced from ~30-line boilerplate to 2-3 macro calls; unique fabric API body preserved
- 4 multicast kernels reduced from 17-line includes+namespace to single `#include "tx_kernel_common.h"`; complex kernel_main() left intact
- sparse_multicast reduced from 11-line includes+namespace to single include; linear-only, header's else-branch provides correct namespace
- `ninja fabric_unit_tests && ninja install` build clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Create tx_kernel_common.h shared header** - `07b76d3280` (feat)
2. **Task 2: Refactor all 9 kernels to use tx_kernel_common.h** - `d5ac4bd097` (refactor)

## Files Created/Modified
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/tx_kernel_common.h` - Shared header: 7 includes + conditional mesh/api.h + linear/api.h + namespace declarations + 3 macros
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/unicast_tx_writer_raw.cpp` - Now uses TX_KERNEL_PARSE_UNICAST_ARGS + TX_KERNEL_SETUP/TEARDOWN
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/scatter_unicast_tx_writer_raw.cpp` - Now uses macros with scatter_offset between PARSE_UNICAST_ARGS and SETUP
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/fused_atomic_inc_unicast_tx_writer_raw.cpp` - Now uses all three macros
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/fused_scatter_atomic_inc_unicast_tx_writer_raw.cpp` - Now uses macros with scatter_offset
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/multicast_tx_writer_raw.cpp` - Includes header only; kernel_main() unchanged
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/scatter_multicast_tx_writer_raw.cpp` - Includes header only; kernel_main() unchanged
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/fused_atomic_inc_multicast_tx_writer_raw.cpp` - Includes header only; kernel_main() unchanged
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/fused_scatter_atomic_inc_multicast_tx_writer_raw.cpp` - Includes header only; kernel_main() unchanged
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/sparse_multicast_tx_writer_raw.cpp` - Includes header only; kernel_main() unchanged

## Decisions Made
- Defined TX_KERNEL_PARSE_UNICAST_ARGS as two separate `#ifdef FABRIC_2D` / `#else` branches rather than nested conditional sub-macros. This keeps each branch a flat, readable sequence of variable declarations.
- Multicast kernel bodies left entirely intact — their 4-direction sender/header logic is too different from unicast to benefit from the unicast macros.
- sparse_multicast treated identically to multicast kernels (header-only, no unicast macros) since its RT arg layout does not match the unicast pattern.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- tx_kernel_common.h is ready for any future TX kernels added to the auto-packetization test suite
- Pattern is established: new unicast kernels use the three macros; new multicast/sparse kernels include the header for boilerplate and handle their own arg parsing

---
*Phase: 04-test-infrastructure-cleanup*
*Completed: 2026-03-12*
