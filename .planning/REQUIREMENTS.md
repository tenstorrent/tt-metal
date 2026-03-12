# Requirements: Fabric Auto-Packetization

**Defined:** 2026-03-11
**Core Value:** Transparent auto-packetization for fabric APIs -- callers send any size, chunking is invisible

## v1.2 Requirements

Requirements for API & Test Cleanup milestone. Each maps to roadmap phases.

### API Cleanup

- [x] **API-01**: Move `_single_packet` function definitions from `mesh/api.h` to `mesh/detail/api.h` in `detail` namespace
- [x] **API-02**: Move `_single_packet` function definitions from `linear/api.h` to `linear/detail/api.h` in `detail` namespace
- [x] **API-03**: Public auto-packetizing wrappers call through to `detail::` single-packet implementations
- [ ] **API-04**: All existing compile-only and silicon tests pass after API restructure

### Test Runner Consolidation

- [ ] **TEST-01**: Extract `make_tx_pattern`, `verify_payload_words`, and L1 setup/validation into shared test utility header
- [ ] **TEST-02**: Refactor `unicast_runner.cpp` and `multicast_runner.cpp` to use shared utilities
- [ ] **TEST-03**: All silicon tests pass after runner refactor

### Device Kernel Consolidation

- [ ] **KERN-01**: Extract shared kernel boilerplate (RT arg parsing, sender setup, header allocation, completion signaling) into common device header
- [ ] **KERN-02**: Refactor existing kernels to use shared boilerplate header
- [ ] **KERN-03**: Templatize/parameterize kernels to reduce the number of separate kernel files
- [ ] **KERN-04**: All compile-only and silicon tests pass after kernel consolidation

### Test Case De-duplication

- [ ] **TCASE-01**: Replace repetitive TEST_F cases with parameterized or table-driven test pattern
- [ ] **TCASE-02**: All test cases continue to produce equivalent test coverage after de-duplication

## Future Requirements

None -- this is a cleanup milestone.

## Out of Scope

| Feature | Reason |
|---------|--------|
| SparseMulticast silicon fix | Blocked on firmware (issue #36581) |
| New API families or features | Cleanup milestone only |
| AddrGen test infrastructure | Separate concern, not part of auto-packetization cleanup |
| Performance optimization | Refactor must be behavior-preserving only |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| API-01 | Phase 3 | Complete |
| API-02 | Phase 3 | Complete |
| API-03 | Phase 3 | Complete |
| API-04 | Phase 3 | Pending |
| TEST-01 | Phase 4 | Pending |
| TEST-02 | Phase 4 | Pending |
| TEST-03 | Phase 4 | Pending |
| KERN-01 | Phase 5 | Pending |
| KERN-02 | Phase 5 | Pending |
| KERN-03 | Phase 5 | Pending |
| KERN-04 | Phase 5 | Pending |
| TCASE-01 | Phase 6 | Pending |
| TCASE-02 | Phase 6 | Pending |

**Coverage:**
- v1.2 requirements: 13 total
- Mapped to phases: 13
- Unmapped: 0 -- all requirements covered

---
*Requirements defined: 2026-03-11*
*Last updated: 2026-03-11 after roadmap creation (traceability complete)*
