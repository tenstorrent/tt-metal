# Roadmap: Fabric Auto-Packetization

## Milestones

- ✅ **v1.0 Fabric Auto-Packetization** - Phases 1 (shipped 2026-03-11)
- ✅ **v1.1 Silicon Validation** - Phase 2 (shipped 2026-03-11)
- ✅ **v1.2 API & Test Cleanup** - Phases 3-4 (shipped 2026-03-12)

## Phases

<details>
<summary>✅ v1.0 Fabric Auto-Packetization (Phase 1) - SHIPPED 2026-03-11</summary>

### Phase 1: fabric-auto-packetization
**Goal**: Auto-packetizing wrappers for all 9 fabric API families (unicast, multicast, scatter, fused atomic_inc, sparse multicast) for both 2D mesh and 1D linear topologies
**Plans**: 8 plans

Plans:
- [x] 01-01 through 01-08: Implement all 9 auto-packetizing API families with compile-only tests

</details>

<details>
<summary>✅ v1.1 Silicon Validation (Phase 2) - SHIPPED 2026-03-11</summary>

### Phase 2: silicon-data-transfer-validation
**Goal**: Silicon-validate all 9 auto-packetizing wrapper families on 4-chip hardware with byte-for-byte correctness verification
**Plans**: 3 plans

Plans:
- [x] 02-01: Unicast silicon test runner + 4 unicast TEST_F cases
- [x] 02-02: Multicast silicon test runner + 5 multicast TEST_F cases
- [x] 02-03: Linear (1D) silicon tests for all families

</details>

### ✅ v1.2 API & Test Cleanup (Shipped 2026-03-12)

**Milestone Goal:** Reduce code bloat in API headers and test suite by moving internal APIs to detail namespace and consolidating duplicated test infrastructure.

#### Phase 3: api-detail-namespace
**Goal**: Move `_single_packet` APIs to `detail` namespace/header so they are accessible to power users but not cluttering the public API surface
**Depends on**: Phase 2
**Requirements**: API-04
**Success Criteria** (what must be TRUE):
  1. `mesh/detail/api.h` and `linear/detail/api.h` exist with all `_single_packet` definitions in `detail::` namespace
  2. Public `mesh/api.h` and `linear/api.h` call into `detail::` — no duplicate definitions
  3. All compile-only tests pass (CompileOnlyAutoPacketization2D, CompileOnlyAutoPacketization1D)
  4. All silicon tests pass identically (18 PASSED, 1 SKIPPED for SparseMulticast #36581)
**Plans**: 3 plans

Plans:
- [x] 03-01: Mesh detail namespace extraction
- [x] 03-02: Linear detail namespace extraction
- [x] 03-03: Build + test validation

#### Phase 4: test-infrastructure-cleanup
**Goal**: Eliminate test code bloat by consolidating device kernel boilerplate into a shared header, extracting duplicated TEST_F patterns into parameterized tests, and merging runner utility duplication
**Depends on**: Phase 3
**Requirements**: TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. Device kernels share a common boilerplate header — no duplicated includes/arg-parsing across 9 kernels
  2. TEST_F silicon tests are parameterized — no repeated pick_chip_pair/sizes/RawTestParams boilerplate
  3. Unicast and multicast runners share common buffer setup, semaphore, and readback utilities
  4. All 18 silicon tests still pass after refactor (1 SKIPPED for SparseMulticast #36581)
  5. Compile-only tests still pass
**Plans**: 3 plans

Plans:
- [x] 04-01-PLAN.md — Create tx_kernel_common.h shared header and refactor all 8 FABRIC_2D kernels
- [x] 04-02-PLAN.md — Move make_tx_pattern/verify_payload_words to test_common.hpp, de-duplicate 16 TEST_F bodies
- [x] 04-03-PLAN.md — Build + silicon test validation (hardware gate)

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. fabric-auto-packetization | v1.0 | 8/8 | Complete | 2026-03-11 |
| 2. silicon-data-transfer-validation | v1.1 | 3/3 | Complete | 2026-03-11 |
| 3. api-detail-namespace | 2/3 | In Progress|  | 2026-03-12 |
| 4. test-infrastructure-cleanup | v1.2 | 3/3 | Complete | 2026-03-12 |
