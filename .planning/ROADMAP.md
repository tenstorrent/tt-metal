# Roadmap: TT-Fabric VC2

## Overview

This roadmap delivers a third virtual channel (VC2) for single-hop neighbour exchange traffic in TT-Fabric. The work is structured to allow incremental merges: Phases 1 and 2 are independently mergeable (Phase 1 is a pure refactor, Phase 2 extends constants/config with VC2 channels still at zero — no behavioral change in either). Phases 3-5 progressively wire VC2 into the channel mapping, builder, and connection layers. Existing regression tests must pass after every phase.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Overlay Register Cleanup** - Refactor stream register tracking into tagged lists and templatize WorkerConnectionAdapter on stream ID (completed 2026-03-17)
- [ ] **Phase 2: Constants & Config Foundation** - Bump MAX_NUM_VCS to 3, extend data structures, add VC2 enable predicate
- [ ] **Phase 3: Channel Mapping & Allocation** - Add VC2 channel mappings for non-Z and Z routers, allocate L1 buffer space
- [ ] **Phase 4: Builder Wiring & Flow Control** - Wire VC2 channels through builder context, datamover builder, and stream register assignments
- [ ] **Phase 5: Connection API & Testing** - Private VC2 connection API, VC2 adapter type alias, and end-to-end VC2 tests

## Phase Details

### Phase 1: Overlay Register Cleanup
**Goal**: Stream register usage is explicitly tracked by type (increment-on-write vs scratch), and WorkerConnectionAdapter is generic over flow control stream ID -- enabling VC2 to use different stream IDs without code duplication
**Depends on**: Nothing (first phase)
**Requirements**: SREG-01, SREG-02
**Success Criteria** (what must be TRUE):
  1. Stream register assignments distinguish increment-on-write usage from scratch usage per stream ID, and a stream ID can appear in both lists (dual-use is explicit)
  2. WorkerConnectionAdapter is a template parameterized on flow control stream ID, with the current adapter available as a type alias hardcoded to stream ID 22
  3. All existing regression tests pass unchanged (test_fabric_ubench + test_fabric_sanity on 2x2 mesh)
**Plans:** 2/2 plans complete

Plans:
- [x] 01-01-PLAN.md — Refactor StreamRegAssignments into IncrementOnWrite/Scratch tagged sub-structs
- [x] 01-02-PLAN.md — Templatize WorkerToFabricEdmSenderImpl on flow control stream ID

### Phase 2: Constants & Config Foundation
**Goal**: The fabric infrastructure supports 3 VCs in its data structures and config, with VC2 conditionally enabled based on topology and hardware, but with zero VC2 channels allocated so behavior is unchanged (independently mergeable)
**Depends on**: Phase 1
**Merge strategy**: Independently mergeable — VC2 channels = 0 means no behavioral change
**Requirements**: CNST-01, CNST-02, CNST-03, CONF-01, CONF-02, CONF-03
**Success Criteria** (what must be TRUE):
  1. MAX_NUM_VCS is 3 and all dependent array initializers compile and pass (no {0,0} two-element initializers remain in fabric code)
  2. PerVcBufferSlots has vc2_sender_slots and vc2_receiver_slots fields, and buffer slot option tables are 6-field structs
  3. FabricConfig exposes a VC2 enable predicate that returns true only when 2D + Blackhole + no UDM/mux + VC1 is active
  4. All existing regression tests pass unchanged (VC2 channels = 0 in all current configs)
**Plans:** 2 plans

Plans:
- [x] 02-01-PLAN.md — Bump MAX_NUM_VCS to 3, fix {0,0} initializers, update legacy getters and print methods
- [ ] 02-02-PLAN.md — Extend PerVcBufferSlots, split buffer tables, add VC2 enable predicate

### Phase 3: Channel Mapping & Allocation
**Goal**: VC2 channels are mapped and allocated in L1 for both non-Z routers (sender + receiver) and Z routers (sender-only at last flat index, forwarding to VC0's receiver)
**Depends on**: Phase 2
**Requirements**: CMAP-01, CMAP-02, CMAP-03, BLDR-04
**Success Criteria** (what must be TRUE):
  1. FabricRouterChannelMapping has initialize_vc2_mappings() that assigns non-Z routers a VC2 sender and receiver channel, with the sender indexed last in flat lookup arrays
  2. Z-router VC2 mapping creates a sender-only channel at the last flat index that forwards to VC0's receiver channel (which on Z routers is VC1's receiver)
  3. StaticSizedChannelAllocator allocates L1 buffer space for VC2 sender and receiver channels when VC2 is enabled, and existing VC0/VC1 allocations are unaffected
  4. All existing regression tests pass unchanged
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

### Phase 4: Builder Wiring & Flow Control
**Goal**: VC2 channels are fully wired through the builder pipeline -- FabricBuilderContext computes VC2 counts, EriscDatamoverBuilder connects VC2 senders to receivers, stream IDs 30/31 are assigned for VC2 flow control, and CT args account for VC2
**Depends on**: Phase 3
**Requirements**: BLDR-01, BLDR-02, BLDR-03, BLDR-05, FLOW-01, FLOW-02, FLOW-03
**Success Criteria** (what must be TRUE):
  1. FabricBuilderContext includes VC2 in max channel count computation and config template creation
  2. EriscDatamoverBuilder wires VC2 sender-to-receiver connections on non-Z routers and VC2 sender-to-VC0-receiver forwarding on Z routers
  3. Stream ID 30 is assigned as increment-on-write register for VC2 sender flow control, and stream ID 31 for VC2 receiver flow control (non-Z only)
  4. CT arg emission (MAX_NUM_SENDER_CHANNELS_VC2, VC2_SENDER_CHANNEL_START, etc.) correctly accounts for VC2 channels in index derivation
  5. All existing regression tests pass unchanged
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD
- [ ] 04-03: TBD

### Phase 5: Connection API & Testing
**Goal**: Workers can inject traffic into VC2 channels via a private connection API, and end-to-end tests verify VC2 data flow on both non-Z and Z routers
**Depends on**: Phase 4
**Requirements**: CONN-01, CONN-02, TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. A private append_fabric_connection_rt_args variant exists for Metal-layer-only callers (not in public fabric API headers)
  2. VC2 WorkerConnectionAdapter is a type alias of the templatized adapter with VC2's stream ID hardcoded
  3. VC2 non-Z router test passes: worker injects into VC2 sender, VC2 receiver writes locally, data verified
  4. VC2 Z-router test passes: worker injects into VC2 sender, traffic forwards through VC0's receiver, data verified
  5. All existing regression tests continue to pass (test_fabric_ubench + test_fabric_sanity)
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Overlay Register Cleanup | 2/2 | Complete   | 2026-03-17 |
| 2. Constants & Config Foundation | 1/2 | In progress | - |
| 3. Channel Mapping & Allocation | 0/2 | Not started | - |
| 4. Builder Wiring & Flow Control | 0/3 | Not started | - |
| 5. Connection API & Testing | 0/2 | Not started | - |
