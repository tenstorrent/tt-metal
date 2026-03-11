---
phase: 01-fabric-auto-packetization
plan: 01
subsystem: testing
tags: [gtest, fabric, compile-probe, device-kernel, auto-packetization]

# Dependency graph
requires: []
provides:
  - RawApiVariant enum with 7 entries for raw-size API variant selection
  - RawTestParams struct mirroring AddrgenTestParams for test parameterization
  - CompileOnlyKernels GTest test that compiles device kernels via detail::CompileProgram
  - Unicast and multicast device kernel stubs including mesh/api.h and linear/api.h
  - Parameterized RawPacketizationTest class with forward-declared runner functions
affects: [01-02-PLAN, 01-03-PLAN, 01-04-PLAN, 01-05-PLAN, 01-06-PLAN, 01-07-PLAN]

# Tech tracking
tech-stack:
  added: []
  patterns: [compile-probe-kernel, raw-size-test-infrastructure]

key-files:
  created:
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_common.hpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_main.cpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/unicast_tx_writer_raw.cpp
    - tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/multicast_tx_writer_raw.cpp
  modified: []

key-decisions:
  - "Used (c) 2025 Tenstorrent AI ULC copyright matching plan spec rather than Tenstorrent Inc. from addrgen_write"
  - "Kernel stubs include both mesh/api.h and linear/api.h to validate both header sets compile"
  - "CompileOnlyKernels test assigns unicast kernel to RISCV_0 and multicast kernel to RISCV_1 to avoid core conflict"

patterns-established:
  - "Compile-probe pattern: TEST_F creates Program, adds kernels, calls detail::CompileProgram without hardware execution"
  - "Raw-size test infrastructure mirrors addrgen_write structure with Fixture2D/Fixture1D and parameterized test classes"

requirements-completed: [AP-01]

# Metrics
duration: 2min
completed: 2026-03-11
---

# Phase 01 Plan 01: Test Infrastructure Summary

**Wave 0 test scaffolding with RawApiVariant enum, CompileOnlyKernels device-toolchain probe, and unicast/multicast kernel stubs**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-11T01:12:31Z
- **Completed:** 2026-03-11T01:14:30Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created test_common.hpp with RawApiVariant enum (7 entries) and RawTestParams struct matching AddrgenTestParams structure
- Created test_main.cpp with CompileOnlyKernels TEST_F that compiles both device kernels via detail::CompileProgram
- Created unicast and multicast device kernel stubs that include mesh/api.h and linear/api.h as compile probes
- Established parameterized RawPacketizationTest class with forward-declared runner functions for plan 07

## Task Commits

Each task was committed atomically:

1. **Task 1: Create test_common.hpp with RawApiVariant enum and RawTestParams** - `eb83fd7241` (feat)
2. **Task 2: Create test_main.cpp stub and device kernel stubs** - `fc1edffdfc` (feat)

## Files Created/Modified
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_common.hpp` - RawApiVariant enum, RawTestParams struct, payload size constants
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_main.cpp` - GTest fixture, CompileOnlyKernels test, parameterized test class
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/unicast_tx_writer_raw.cpp` - Device kernel stub including mesh and linear API headers
- `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/multicast_tx_writer_raw.cpp` - Device kernel stub including mesh and linear API headers

## Decisions Made
- Used `(c) 2025 Tenstorrent AI ULC` SPDX copyright as specified in the plan rather than the `Tenstorrent Inc.` used in some addrgen_write files
- Both kernel stubs include both mesh/api.h and linear/api.h so a single CompileOnlyKernels test validates both header sets
- Assigned unicast kernel to RISCV_0 and multicast kernel to RISCV_1 data movement processors to avoid same-core conflict in CompileProgram

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Test infrastructure is in place for plans 02-06 to modify API headers and validate via CompileOnlyKernels
- After plans 02-06 modify headers, run `--gtest_filter=*CompileOnly*` to validate device-toolchain compilation
- Plan 07 will implement the runner functions (run_raw_unicast_write_test, run_raw_multicast_write_test)

## Self-Check: PASSED

All 4 created files verified on disk. Both task commits (eb83fd7241, fc1edffdfc) verified in git log.

---
*Phase: 01-fabric-auto-packetization*
*Completed: 2026-03-11*
