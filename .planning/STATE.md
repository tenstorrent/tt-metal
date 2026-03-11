# Project State

## Current Position
- **Phase:** 02-silicon-data-transfer-validation
- **Current Plan:** 2 of 2 complete in Phase
- **Status:** Plan 02 complete -- all 9 families have silicon test coverage
- **Next:** Silicon test execution on hardware

## Project Reference

**Core value:** Transparent auto-packetization for fabric APIs -- callers send any size, chunking is invisible
**Current focus:** Silicon data-transfer validation for all 9 auto-packetizing families

## Decisions
- _single_packet suffix for renamed APIs; wrappers keep original names
- Breadth-first multi-connection chunking
- Scatter wrappers are passthrough (pre-computed NOC addresses)
- mesh SetRoute=false pattern
- Fused ops: intermediate chunks as regular writes, atomic_inc only on final chunk
- Tests compiled into fabric_unit_tests binary
- detail::CompileProgram for device kernel compile-only validation
- L1 direct I/O via WriteToDeviceL1/ReadFromDeviceL1 for BaseFabricFixture tests
- GlobalSemaphore created fresh per test call for per-chip MeshDevice pattern
- family_kernel_path() maps all 9 families including future multicast kernels
- MeshGraph::chip_to_coordinate for bounding-box computation in multicast runner
- Sparse multicast test inlined in test file (different dispatch from multicast runner)
- Linear 1D tests reuse mesh multicast kernels (per-direction fanout degenerates in 1D)

## Blockers
None

## Last Session
- **Timestamp:** 2026-03-11
- **Stopped At:** Completed 02-02-PLAN.md (multicast/sparse silicon tests)
