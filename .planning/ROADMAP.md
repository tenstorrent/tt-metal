# Roadmap: Fabric Auto-Packetization

## Milestones

- **v1.0 Fabric Auto-Packetization** -- Phase 1 (shipped 2026-03-11)
- **v1.1 Silicon Validation** -- Phase 2 (shipped 2026-03-11)
- **v1.2 API & Test Cleanup** -- Phases 3-6 (in progress)

## Phases

<details>
<summary>v1.0 Fabric Auto-Packetization (Phase 1) -- SHIPPED 2026-03-11</summary>

- [x] Phase 1: fabric-auto-packetization (8/8 plans) -- completed 2026-03-11

</details>

<details>
<summary>v1.1 Silicon Validation (Phase 2) -- SHIPPED 2026-03-11</summary>

- [x] Phase 2: silicon-data-transfer-validation (3/3 plans) -- completed 2026-03-11

</details>

### v1.2 API & Test Cleanup (In Progress)

**Milestone Goal:** Reduce code bloat in API headers and test suite by moving internal APIs to detail namespace and consolidating duplicated test infrastructure. Every phase ends with a full rebuild + retest to confirm no regressions.

- [ ] **Phase 3: api-detail-namespace** - Move `_single_packet` internals out of public headers into detail namespace
- [ ] **Phase 4: test-runner-consolidation** - Extract shared test utilities so both runners use common infrastructure
- [ ] **Phase 5: device-kernel-consolidation** - Consolidate 9 device kernels via shared boilerplate header and templatization
- [ ] **Phase 6: test-case-deduplication** - Replace repetitive TEST_F cases with parameterized/table-driven pattern

## Phase Details

### Phase 3: api-detail-namespace
**Goal**: Public API headers are clean; `_single_packet` implementations live in `detail` namespace/headers, backward-compatible via include
**Depends on**: Phase 2 (complete)
**Requirements**: API-01, API-02, API-03, API-04
**Success Criteria** (what must be TRUE):
  1. `mesh/detail/api.h` and `linear/detail/api.h` exist and contain all `_single_packet` function definitions in `detail::` namespace
  2. Public `mesh/api.h` and `linear/api.h` no longer define `_single_packet` functions directly; they delegate to `detail::`
  3. Existing kernel code that called `_single_packet` APIs still compiles without modification (backward compat via include)
  4. All compile-only and silicon tests pass identically after the restructure (verified by rebuild + retest)
**Plans:** 3 plans

Plans:
- [ ] 03-01-PLAN.md -- Extract mesh _single_packet definitions to mesh/detail/api.h
- [ ] 03-02-PLAN.md -- Extract linear _single_packet definitions to linear/detail/api.h
- [ ] 03-03-PLAN.md -- Full rebuild and retest validation

### Phase 4: test-runner-consolidation
**Goal**: `unicast_runner.cpp` and `multicast_runner.cpp` share a common utility header rather than duplicating payload/L1 helpers
**Depends on**: Phase 3
**Requirements**: TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. A shared test utility header exists containing `make_tx_pattern`, `verify_payload_words`, and L1 setup/validation helpers
  2. `unicast_runner.cpp` and `multicast_runner.cpp` import from the shared header; their own source files no longer define these helpers
  3. All silicon tests pass after runner refactor (verified by rebuild + retest)
**Plans**: TBD

### Phase 5: device-kernel-consolidation
**Goal**: Device kernel count and boilerplate are reduced; a shared device header handles common RT arg parsing, sender setup, header allocation, and completion signaling
**Depends on**: Phase 4
**Requirements**: KERN-01, KERN-02, KERN-03, KERN-04
**Success Criteria** (what must be TRUE):
  1. A shared device kernel header exists and is included by all kernels to handle common boilerplate (RT args, sender setup, header alloc, completion signaling)
  2. The number of separate kernel files is reduced via templatization or parameterization (fewer files covering the same test matrix)
  3. All compile-only and silicon tests pass after kernel consolidation (verified by rebuild + retest)
  4. No behavioral change: silicon test results are byte-for-byte identical before and after
**Plans**: TBD

### Phase 6: test-case-deduplication
**Goal**: The 15 near-identical TEST_F cases are replaced by a parameterized or table-driven pattern that covers the same matrix with less repetition
**Depends on**: Phase 5
**Requirements**: TCASE-01, TCASE-02
**Success Criteria** (what must be TRUE):
  1. TEST_F repetition is eliminated; test coverage matrix is expressed as a parameter table or typed test list rather than copy-pasted test bodies
  2. All test cases that existed before still execute (same test count or equivalent coverage) — no test coverage is lost
  3. Full rebuild + retest confirms all tests pass after de-duplication
**Plans**: TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. fabric-auto-packetization | v1.0 | 8/8 | Complete | 2026-03-11 |
| 2. silicon-data-transfer-validation | v1.1 | 3/3 | Complete | 2026-03-11 |
| 3. api-detail-namespace | v1.2 | 0/3 | Planning complete | - |
| 4. test-runner-consolidation | v1.2 | 0/? | Not started | - |
| 5. device-kernel-consolidation | v1.2 | 0/? | Not started | - |
| 6. test-case-deduplication | v1.2 | 0/? | Not started | - |
