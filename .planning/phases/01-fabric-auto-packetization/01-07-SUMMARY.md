---
phase: 01-fabric-auto-packetization
plan: 07
subsystem: testing
tags: [gtest, fabric, auto-packetization, raw-size, device-kernel, integration-test]

# Dependency graph
requires:
  - "01-01: test infrastructure scaffolding (test_common.hpp, test_main.cpp, kernel stubs)"
  - "01-02 through 01-06: API header renames and auto-packetizing wrappers"
provides:
  - Raw-size unicast device kernel calling fabric_unicast_noc_unicast_write with oversized payload
  - Raw-size multicast device kernel calling fabric_multicast_noc_unicast_write with per-direction fanout
  - Host-side unicast runner (run_raw_unicast_write_test) dispatching sender + receiver and validating data
  - Host-side multicast runner (run_raw_multicast_write_test) with rectangular sub-mesh receiver set
  - CMake test_auto_packetization binary target
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: [raw-size-kernel-dispatch-without-addrgen, direct-noc-addr-fabric-write]

key-files:
  created:
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/unicast_runner.cpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/multicast_runner.cpp
  modified:
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/unicast_tx_writer_raw.cpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/multicast_tx_writer_raw.cpp
    - tests/tt_metal/tt_fabric/CMakeLists.txt

key-decisions:
  - "Kernel receives NOC address components (x, y, l1_addr) and computes 64-bit NOC addr on device via safe_get_noc_addr rather than host pre-computing it"
  - "Reuses existing rx_addrgen.cpp receiver kernel from addrgen_write tests for completion signaling"
  - "Raw-size kernels do not use CB/reader pattern - send directly from source buffer address"
  - "Multicast kernel follows per-direction fanout pattern (W/E/N/S) matching addrgen multicast tests"

patterns-established:
  - "Raw-size test dispatch: single RISCV_1 writer kernel with no reader/CB, directly referencing DRAM buffer addr"
  - "Completion signaling: fabric_unicast_noc_unicast_atomic_inc after data send, receiver waits on GlobalSemaphore"

requirements-completed: [AP-06]

# Metrics
duration: 7min
completed: 2026-03-11
---

# Phase 01 Plan 07: Integration Test Runners Summary

**Raw-size unicast and multicast device kernels + host runners exercising auto-packetizing fabric write wrappers with oversized payloads**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-11T01:44:33Z
- **Completed:** 2026-03-11T01:51:43Z
- **Tasks:** 1 completed, 1 checkpoint pending
- **Files modified:** 5

## Accomplishments
- Completed unicast and multicast device kernel stubs (from Plan 01) with real API calls to auto-packetizing wrappers
- Created host-side unicast_runner.cpp dispatching unicast_tx_writer_raw kernel and verifying data integrity end-to-end
- Created host-side multicast_runner.cpp with rectangular sub-mesh receiver set and per-direction fabric connections
- Added CMake test_auto_packetization target; binary compiles and links successfully

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement device kernels and host runners** - `5c451e0b7a` (feat)

Task 2 (hardware verification checkpoint) is pending user approval.

## Files Created/Modified
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/unicast_tx_writer_raw.cpp` - Device kernel: sends payload via fabric_unicast_noc_unicast_write + atomic_inc completion
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/multicast_tx_writer_raw.cpp` - Device kernel: per-direction multicast fanout via fabric_multicast_noc_unicast_write
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/unicast_runner.cpp` - Host runner: buffer allocation, kernel dispatch, data validation for unicast
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/multicast_runner.cpp` - Host runner: rectangular sub-mesh setup, per-direction fabric connections, multi-destination validation
- `tests/tt_metal/tt_fabric/CMakeLists.txt` - Added test_auto_packetization executable target

## Decisions Made
- Kernel receives NOC address components (x, y, l1_addr) separately and computes the 64-bit NOC address on-device using safe_get_noc_addr. This matches the established pattern from addrgen_write kernels and avoids host-side NOC address computation.
- Reused the existing rx_addrgen.cpp receiver kernel for completion signaling rather than writing a new one. The receiver simply waits on a GlobalSemaphore, which is protocol-agnostic.
- Raw-size kernels send directly from the source buffer address without a CB/reader pipeline. This is simpler than the addrgen pattern (which uses DRAM->CB->fabric) because raw-size tests focus on the packetization wrapper, not the full pipeline.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Test binary compiles and links. Hardware verification checkpoint (Task 2) is pending.
- Once hardware test passes, Phase 01 is fully complete.

## Self-Check: PASSED
