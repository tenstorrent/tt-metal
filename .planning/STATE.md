# Project State

## Current Position
- **Phase:** 02-silicon-data-transfer-validation
- **Current Plan:** 3 of 3 complete in Phase
- **Status:** Phase 02 complete -- all silicon tests run on hardware, 16/17 pass (SparseMulticast deferred issue #36581)
- **Next:** Phase complete

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
- SparseMulticast GTEST_SKIP is essential -- removing it causes Ethernet core lockup (issue #36581 confirmed real)
- Phase gate accepted at 8/9 families validated; SparseMulticast deferred pending firmware fix

## Blockers
None

## Last Session
- **Timestamp:** 2026-03-11
- **Stopped At:** Completed 02-03-PLAN.md (silicon test execution -- phase 02 complete)
