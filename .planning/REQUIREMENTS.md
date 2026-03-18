# Requirements: TT-Fabric VC2

**Defined:** 2026-03-17
**Core Value:** Third VC correctly wired into fabric infrastructure, invisible to existing VC0/VC1, only active under correct conditions

## v1 Requirements

### Stream Register Cleanup

- [x] **SREG-01**: Stream register assignment refactored into tagged lists -- increment-on-write and scratch usage tracked separately per stream ID (dual-use explicit)
- [x] **SREG-02**: WorkerConnectionAdapter templatized on flow control stream ID; current adapter becomes type alias with stream ID 22 hardcoded

### Constants & Data Structures

- [x] **CNST-01**: `MAX_NUM_VCS` bumped from 2 to 3 with all dependent array initializers updated
- [x] **CNST-02**: `PerVcBufferSlots` extended with vc2_sender_slots and vc2_receiver_slots fields
- [x] **CNST-03**: Buffer slot option tables extended from 4-field to 6-field structs

### Channel Mapping

- [x] **CMAP-01**: `FabricRouterChannelMapping` gains `initialize_vc2_mappings()` for non-Z routers (sender + receiver)
- [x] **CMAP-02**: Z-router VC2 mapping: sender-only at last flat index, forwards to VC0's receiver channel (which on Z routers is VC1's receiver)
- [x] **CMAP-03**: VC2 sender channel indexed as last in flat lookup arrays

### Configuration & Enablement

- [x] **CONF-01**: `FabricConfig` updated with VC2 enable predicate: 2D + Blackhole + no UDM/mux extension
- [x] **CONF-02**: VC2 enablement follows existing `IntermeshVCConfig` gating pattern
- [x] **CONF-03**: VC2 requires VC1 to be active (VC1 requires 2D + multi-mesh)

### Builder Integration

- [x] **BLDR-01**: `FabricBuilderContext` computes VC2 channel counts and allocations
- [x] **BLDR-02**: `EriscDatamoverBuilder` wires VC2 sender→receiver connections on non-Z routers
- [x] **BLDR-03**: `EriscDatamoverBuilder` wires Z-router VC2 worker sender→VC0 receiver forwarding
- [x] **BLDR-04**: `StaticSizedChannelAllocator` allocates L1 buffer space for VC2 channels
- [x] **BLDR-05**: CT arg emission accounts for VC2 channels in index derivation formulas (`MAX_NUM_SENDER_CHANNELS_VC0`, `VC1_SENDER_CHANNEL_START`, etc.)

### Flow Control

- [x] **FLOW-01**: Stream ID 30 increment-on-write register assigned to VC2 sender flow control (free-slots from worker)
- [x] **FLOW-02**: Stream ID 31 increment-on-write register assigned to VC2 receiver flow control (free-slots from sender) -- non-Z routers only
- [x] **FLOW-03**: L1 credit counter arrays grow naturally via `NUM_SENDER_CHANNELS` -- `src_ch_id` is global flat index, no offset remapping needed

### Connection API

- [x] **CONN-01**: Private `append_fabric_connection_rt_args` variant for Metal-layer-only callers (not published under fabric API)
- [x] **CONN-02**: VC2 WorkerConnectionAdapter: same template as existing, type alias with VC2's stream ID hardcoded

### Testing

- [x] **TEST-01**: New tests in `test_tt_fabric` with VC option in sender config for VC2 non-Z router
- [x] **TEST-02**: New tests in `test_tt_fabric` for VC2 Z-router (sender-only, VC0 receiver forwarding)
- [ ] **TEST-03**: Existing regression tests pass after each phase (`test_fabric_ubench` + `test_fabric_sanity`)

## v2 Requirements

### Future Extensions

- **FUTR-01**: Wormhole support for VC2
- **FUTR-02**: Public worker API for VC2 (if needed beyond private data plane usage)
- **FUTR-03**: VC2 performance optimization

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multi-hop routing on VC2 | VC2 is single-hop (neighbour exchange) only -- structural, not routed |
| Router firmware VC2 packet identification | Not needed -- VC2 is structural (separate physical channels) |
| Wormhole support | Blackhole only for initial implementation |
| Public worker API for VC2 | Private/hidden from general workers; Metal-layer only |
| UDM/mux extension compatibility | VC2 disabled when these modes are active |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SREG-01 | Phase 1 | Complete |
| SREG-02 | Phase 1 | Complete |
| CNST-01 | Phase 2 | Complete |
| CNST-02 | Phase 2 | Complete |
| CNST-03 | Phase 2 | Complete |
| CONF-01 | Phase 2 | Complete |
| CONF-02 | Phase 2 | Complete |
| CONF-03 | Phase 2 | Complete |
| CMAP-01 | Phase 3 | Complete |
| CMAP-02 | Phase 3 | Complete |
| CMAP-03 | Phase 3 | Complete |
| BLDR-04 | Phase 3 | Complete |
| BLDR-01 | Phase 4 | Complete |
| BLDR-02 | Phase 4 | Complete |
| BLDR-03 | Phase 4 | Complete |
| BLDR-05 | Phase 4 | Complete |
| FLOW-01 | Phase 4 | Complete |
| FLOW-02 | Phase 4 | Complete |
| FLOW-03 | Phase 4 | Complete |
| CONN-01 | Phase 5 | Complete |
| CONN-02 | Phase 5 | Complete |
| TEST-01 | Phase 5 | Complete |
| TEST-02 | Phase 5 | Complete |
| TEST-03 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 24 total
- Mapped to phases: 24
- Unmapped: 0

---
*Requirements defined: 2026-03-17*
*Last updated: 2026-03-17 after roadmap creation*
