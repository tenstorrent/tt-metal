# Roadmap: TT-Fabric VC2

## Milestones

- v1.0 **VC2 Support** -- Phases 1-6 (shipped 2026-03-21)
- v1.1 **VC2 Multi-VC Test Coverage** -- Phase 7 (in progress)

## Phases

<details>
<summary>v1.0 VC2 Support (Phases 1-6) -- SHIPPED 2026-03-21</summary>

- [x] Phase 1: Overlay Register Cleanup (2/2 plans) -- completed 2026-03-17
- [x] Phase 2: Constants & Config Foundation (2/2 plans) -- completed 2026-03-17
- [x] Phase 3: Channel Mapping & Allocation (2/2 plans) -- completed 2026-03-17
- [x] Phase 4: Builder Wiring & Flow Control (2/2 plans) -- completed 2026-03-18
- [x] Phase 5: Connection API & Testing (2/2 plans) -- completed 2026-03-18
- [x] Phase 6: VC2 Sender Integration & End-to-End Verification (2/2 plans) -- completed 2026-03-18

Full phase details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

### v1.1 VC2 Multi-VC Test Coverage (In Progress)

- [ ] **Phase 7: Multi-VC Test Cases & CI Wiring** - YAML test configs for all VC combinations and multi-process CI on BH galaxy

## Phase Details

### Phase 7: Multi-VC Test Cases & CI Wiring
**Goal**: VC2 is exercised in all meaningful VC combinations (VC0+VC1+VC2, VC0+VC2, multi-worker) and verified automatically in CI on real multi-mesh hardware
**Depends on**: Phase 6 (VC2 end-to-end sender integration)
**Requirements**: CI-01, VC-01, VC-02, VC-03
**Success Criteria** (what must be TRUE):
  1. Running `test_tt_fabric` with a VC0+VC1+VC2 YAML config on a multi-mesh BH system completes successfully with all three VCs carrying traffic
  2. Running `test_tt_fabric` with a VC0+VC2 YAML config on a single-mesh BH system completes successfully with VC2 active alongside VC0 (no VC1)
  3. Multi-concurrent-worker test cases exist that inject traffic on multiple VCs simultaneously, and per-flow bandwidth is readable from existing telemetry
  4. BH galaxy GHA workflow runs the VC2 YAML as a multi-process test, exercising multi-mesh topology where VC1 is forced active alongside VC2
  5. All new test cases pass in CI without regressing existing VC0/VC1 test coverage
**Plans:** 2 plans

Plans:
- [ ] 07-01-PLAN.md -- Add multi-mesh VC0+VC1+VC2 and multi-worker test entries to VC2 YAML
- [ ] 07-02-PLAN.md -- Wire VC2 multi-mesh test into BH galaxy CI workflow

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Overlay Register Cleanup | v1.0 | 2/2 | Complete | 2026-03-17 |
| 2. Constants & Config Foundation | v1.0 | 2/2 | Complete | 2026-03-17 |
| 3. Channel Mapping & Allocation | v1.0 | 2/2 | Complete | 2026-03-17 |
| 4. Builder Wiring & Flow Control | v1.0 | 2/2 | Complete | 2026-03-18 |
| 5. Connection API & Testing | v1.0 | 2/2 | Complete | 2026-03-18 |
| 6. VC2 Sender Integration & End-to-End Verification | v1.0 | 2/2 | Complete | 2026-03-18 |
| 7. Multi-VC Test Cases & CI Wiring | v1.1 | 0/2 | Not started | - |
