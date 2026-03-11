# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.1 -- Silicon Validation

**Shipped:** 2026-03-11
**Phases:** 1 | **Plans:** 3 | **Sessions:** ~3

### What Was Built
- Complete silicon test suite for 9 auto-packetizing wrapper families
- 8 new device kernels covering unicast, multicast, scatter, fused variants
- 17 TEST_F cases across 2D mesh and 1D linear topologies
- 8/9 families silicon-validated byte-for-byte correct on 4-chip hardware

### What Worked
- Plan decomposition into unicast (Plan 01) -> multicast (Plan 02) -> silicon execution (Plan 03) was clean
- Reusing mesh multicast kernels for 1D linear tests eliminated kernel duplication
- family_kernel_path() forward-declaring all 9 paths in Plan 01 meant Plan 02 needed zero changes to shared infra
- BaseFabricFixture L1 direct I/O pattern was correct from the start -- avoided the DRAM-as-L1 bug

### What Was Inefficient
- Initial Fabric1DFixture tests were GTEST_SKIP stubs that had to be replaced with real 1D implementations after silicon testing revealed the need
- FABRIC_2D ifdef guards were missing from all kernels initially, required a retroactive fix pass
- SparseMulticast test hang caused device lockup requiring board reset -- could have been caught earlier by running with GTEST_SKIP in place first

### Patterns Established
- BaseFabricFixture + L1 direct I/O as the standard test pattern for fabric data validation
- GTEST_SKIP as essential guard for known firmware limitations (not just convenience)
- tt-smi board reset procedure for Ethernet core lockup recovery
- Per-chip MeshDevice dispatch with fresh GlobalSemaphore per test invocation

### Key Lessons
1. Always include FABRIC_2D ifdef guards in device kernels from the start -- 2D-only APIs silently compile but fail at runtime on 1D fixtures
2. GTEST_SKIP for hardware bugs is a safety mechanism, not a workaround -- removing it can cause device-level damage (Ethernet core lockup)
3. Forward-declaring infrastructure (like family_kernel_path for all 9 families) in early plans pays off by eliminating cross-plan modification

### Cost Observations
- Model mix: ~80% sonnet, ~20% opus (opus for silicon execution supervision)
- Sessions: ~3
- Notable: Entire milestone completed in a single day -- tight plan decomposition enabled fast sequential execution

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | ~5 | 1 (8 plans) | Initial API implementation with compile-only validation |
| v1.1 | ~3 | 1 (3 plans) | Silicon validation pattern established |

### Cumulative Quality

| Milestone | Tests | Coverage | Known Gaps |
|-----------|-------|----------|------------|
| v1.0 | 9 compile-only | API completeness | No silicon validation |
| v1.1 | 17 silicon + 9 compile | 8/9 families | SparseMulticast (firmware) |

### Top Lessons (Verified Across Milestones)

1. Plan decomposition by family group (unicast vs multicast) works well for parallel development
2. Forward-declaring shared infrastructure in early plans eliminates cross-plan coupling
