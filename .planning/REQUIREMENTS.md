# Requirements: TT-Fabric VC2 Multi-VC Test Coverage

**Defined:** 2026-03-21
**Core Value:** VC2 is correctly exercised alongside VC1 and in isolation, with CI coverage on real multi-mesh hardware

## v1.1 Requirements

### CI Coverage

- [x] **CI-01**: BH galaxy GHA runs `test_fabric_vc2_at_least_2x2_mesh.yaml` as a multi-process test, forcing multi-mesh topology so VC1 is enabled alongside VC2

### VC Combo Test Cases

- [x] **VC-01**: Test cases exist for VC0+VC1+VC2 configuration (multi-mesh topology + use_vc2: true)
- [x] **VC-02**: Test cases exist for VC0+VC2 configuration (single-mesh topology + use_vc2: true)
- [x] **VC-03**: Test cases exercise multiple concurrent workers across VC configurations to validate per-flow bandwidth via existing telemetry

## Future Requirements

### Test Framework Extensions (separate branch, not VC2-specific)

- Per-sender-channel telemetry readback
- Dependency mechanism between senders (model multi-hop neighbor exchange chains)
- Sender kernel command sequence support (send router commands)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Dynamic VC selection (runtime adaptive routing) | Out of scope for user's needs |
| Standalone load-balance test binary | Not needed -- existing framework covers measurement via telemetry |
| Receive-and-forward custom kernels | Deferred to test framework extensions branch |
| Wormhole support | Blackhole only |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CI-01 | Phase 7 | Complete |
| VC-01 | Phase 7 | Complete |
| VC-02 | Phase 7 | Complete |
| VC-03 | Phase 7 | Complete |

**Coverage:**
- v1.1 requirements: 4 total
- Mapped to phases: 4/4
- Unmapped: 0

---
*Requirements defined: 2026-03-21*
*Last updated: 2026-03-21 after roadmap creation*
