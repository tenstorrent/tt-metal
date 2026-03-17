# TT-Fabric Third Virtual Channel (VC2)

## What This Is

An extension to the TT-Fabric subsystem that adds a third virtual channel (VC2) for single-hop neighbour exchange traffic. This VC is conditionally enabled only in 2D fabric configurations on Blackhole hardware, and is hidden from general worker connections. The work includes builder, router, allocator, adapter, and config changes with full test coverage.

## Core Value

The third VC must be correctly wired into the fabric infrastructure — builder, router, channel allocator, and connection adapter — such that it is invisible to existing VC0/VC1 traffic and only activatable under the correct conditions (2D fabric, Blackhole, no UDM/mux extension).

## Requirements

### Validated

<!-- Existing fabric infrastructure that this work builds on -->

- ✓ Two-VC fabric routing (VC0, VC1) — existing
- ✓ Fabric builder pattern (FabricBuilderContext, EriscDatamoverBuilder, StaticSizedChannelAllocator) — existing
- ✓ Fabric router (fabric_erisc_router.cpp) — existing
- ✓ WorkerConnectionAdapter with flow control — existing
- ✓ CT arg handler for fabric connections — existing
- ✓ FabricConfig for topology/mode configuration — existing
- ✓ 2D fabric topology support — existing

### Active

#### Overlay Register Cleanup (independently mergeable — Phase 1)
- [ ] Stream register assignment refactored into tagged lists: increment-on-write usage and scratch usage tracked separately per stream ID (a stream ID can appear in both)
- [ ] WorkerConnectionAdapter templatized on flow control stream ID; current adapter becomes type alias with stream ID 22 hardcoded

#### Constants & Data Structures
- [ ] `MAX_NUM_VCS` bumped from 2 to 3 with all dependent array initializers updated
- [ ] `PerVcBufferSlots` extended with vc2 fields
- [ ] Buffer slot option tables extended from 4-field to 6-field structs

#### Channel Mapping & Allocation
- [ ] `FabricRouterChannelMapping` gains `initialize_vc2_mappings()` for non-Z routers (sender + receiver)
- [ ] Z-router VC2 mapping: sender-only at last flat index, forwards to VC0's receiver channel (which on Z routers is VC1's receiver)
- [ ] VC2 sender channel indexed as last in flat lookup arrays
- [ ] `StaticSizedChannelAllocator` allocates L1 buffer space for VC2 channels

#### Configuration & Enablement
- [ ] `FabricConfig` updated with VC2 enable predicate: 2D + Blackhole + no UDM/mux extension
- [ ] VC2 enablement follows existing `IntermeshVCConfig` gating pattern
- [ ] VC2 requires VC1 to be active (VC1 requires 2D + multi-mesh; VC2 adds Blackhole + no UDM/mux on top)

#### Builder Integration
- [ ] `FabricBuilderContext` computes VC2 channel counts and allocations
- [ ] `EriscDatamoverBuilder` wires VC2 sender→receiver connections on non-Z routers
- [ ] `EriscDatamoverBuilder` wires Z-router VC2 worker sender→VC0 receiver forwarding
- [ ] CT arg emission accounts for VC2 channels in index derivation formulas

#### Flow Control
- [ ] Stream ID 30: increment-on-write register for VC2 sender flow control (free-slots from worker)
- [ ] Stream ID 31: increment-on-write register for VC2 receiver flow control (free-slots from sender) — non-Z routers only; may reserve on Z routers but unused
- [ ] L1 credit counter arrays grow naturally — `NUM_SENDER_CHANNELS` includes VC2, `src_ch_id` is global flat index, no offset remapping needed

#### Connection API
- [ ] Private `append_fabric_connection_rt_args` variant for Metal-layer-only callers (not published under fabric API)
- [ ] VC2 WorkerConnectionAdapter: same template as existing, type alias with VC2's stream ID

#### Testing
- [ ] New tests in `test_tt_fabric` with VC option in sender config for VC2 non-Z router
- [ ] New tests in `test_tt_fabric` for VC2 Z-router (sender-only, VC0 receiver forwarding)
- [ ] Existing regression tests pass after each phase (`test_fabric_ubench` + `test_fabric_sanity`)

### Out of Scope

- Multi-hop routing on VC2 — VC2 is single-hop (neighbour exchange) only
- Wormhole support — Blackhole only for now
- Public worker API for VC2 — this VC is private/hidden from general workers
- UDM or mux extension mode compatibility — VC2 disabled when these are active
- Performance optimization — correctness first, optimization later
- Router firmware VC2 packet identification — not needed, VC2 is structural (separate physical channels, worker injects, receiver writes locally)

## Context

**Codebase:** tt-metal, a hardware programming framework for Tenstorrent accelerators. The fabric subsystem manages inter-device ethernet routing with virtual channels.

**Existing VC model:** Fabric currently supports 2 VCs (VC0, VC1). Each VC has sender and receiver channels allocated by `StaticSizedChannelAllocator`. The fabric builder constructs router programs on ERISC cores via `EriscDatamoverBuilder`.

**Z vs non-Z routers:** "Z" routers are a special topology configuration. On Z routers, VC2 adds only a sender channel (no receiver). This sender lives at the last flat index but forwards to VC0's receiver channel (which on Z routers is actually VC1's receiver — same destination as other VC0 senders).

**Non-Z routers:** VC2 adds both a sender and receiver channel. Worker injects packets into VC2 sender. VC2 receiver writes locally (no forwarding). This is purely structural — no packet-level VC identification needed.

**Private connection model:** The VC2 connection is not exposed through the public fabric connection API. It's connected by the data plane layer within Metal, using a statically assigned overlay register for flow control.

**Stream register model:** Each stream ID maps to a stream with BOTH a scratch register (index 0) and an increment-on-write register (index 34/64). A single stream ID can serve both purposes simultaneously. Stream IDs 30 and 31 currently used as scratch for ETH_RETRAIN_LINK_SYNC and MULTI_RISC_TEARDOWN_SYNC — their increment-on-write registers are free and will be used for VC2 flow control.

**VC independence:** VC2 does not require VC1. VC1 activates in 2D + multi-mesh mode. VC2 activates in 2D mode (regardless of multi-mesh). They are independently gated.

**Packet-level muxing:** Fabric always muxes at packet granularity. The new VC2 traffic follows this same model.

## Constraints

- **Stream registers**: 32 stream IDs total; each has independent scratch and increment-on-write registers. IDs 30/31 available for VC2 increment-on-write usage
- **Hardware**: Blackhole only — no Wormhole support required
- **Topology**: 2D fabric configs only — VC2 disabled otherwise
- **Mode exclusion**: Must not be enabled when UDM or mux extension modes are active
- **Indexing**: New sender channel must be last index in flat arrays used for lookup
- **Flow control**: Stream ID 30 for sender, stream ID 31 for receiver (non-Z only)
- **Compatibility**: Existing VC0/VC1 behaviour must be completely unchanged

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Templatize existing WorkerConnectionAdapter; current name becomes type alias with stream ID 22 | Avoids code duplication; VC2 adapter is same template with different stream ID | — Pending |
| Stream IDs 30/31 for VC2 flow control (increment-on-write registers) | Scratch registers at these IDs already in use — dual-use is safe since register types are independent | — Pending |
| VC2 sender indexed last in flat arrays | Consistent with existing convention; minimizes disruption to existing index calculations | — Pending |
| Z-router VC2 sender forwards to VC0's receiver channel | Matches physical merge semantics; receiver is shared with other VC0 senders | — Pending |
| Private append_fabric_connection_rt_args in Metal layer | Not published under fabric API; only data plane layer calls it | — Pending |
| Overlay register cleanup as Phase 1 | Can merge independently; makes stream ID dual-use explicit; unblocks VC2 | — Pending |
| VC2 requires VC1 | VC2 activation: 2D + multi-mesh + Blackhole + no UDM/mux (superset of VC1 conditions) | — Pending |

---
*Last updated: 2026-03-17 after deep questioning and stream register clarification*
