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

- [ ] Conditional VC2 channel allocation in fabric builder (only when 2D fabric + Blackhole + no UDM/mux extension)
- [ ] VC2 sender and receiver channels on non-Z routers (purely additive)
- [ ] VC2 sender-only channel on Z routers (merges with VC0 physically, VC2 logically)
- [ ] Additional VC0 worker sender channel on Z routers for private data plane connection
- [ ] Statically assigned overlay register for Z-router private flow control (stateless connect API)
- [ ] Templatized WorkerConnectionAdapter on flow control overlay register stream ID
- [ ] Private `append_fabric_connection_rt_args` variant for Metal-layer-only callers
- [ ] VC2 sender channel indexed as last in flat lookup arrays
- [ ] FabricConfig updated with VC2 enable/disable logic
- [ ] Fabric router updated to handle VC2 packet routing (single-hop only)
- [ ] VC0/VC1 always active when VC1 is active (VC2 is purely additive)
- [ ] Unit/integration tests proving VC2 correctness on both Z and non-Z routers

### Out of Scope

- Multi-hop routing on VC2 — VC2 is single-hop (neighbour exchange) only
- Wormhole support — Blackhole only for now
- Public worker API for VC2 — this VC is private/hidden from general workers
- UDM or mux extension mode compatibility — VC2 disabled when these are active
- Performance optimization — correctness first, optimization later

## Context

**Codebase:** tt-metal, a hardware programming framework for Tenstorrent accelerators. The fabric subsystem manages inter-device ethernet routing with virtual channels.

**Existing VC model:** Fabric currently supports 2 VCs (VC0, VC1). Each VC has sender and receiver channels allocated by `StaticSizedChannelAllocator`. The fabric builder constructs router programs on ERISC cores via `EriscDatamoverBuilder`.

**Z vs non-Z routers:** "Z" routers are a special topology configuration. The new VC2 behaves differently on Z routers — it only adds a sender channel (no receiver), and that sender physically merges into VC0 while being logically addressed as VC2.

**Private connection model:** The new Z-router VC0 worker channel is not exposed through the public fabric connection API. It's connected by the data plane layer within Metal, using a statically assigned overlay register for flow control. This enables a stateless connect API (`append_fabric_connection_rt_args` private variant).

**Packet-level muxing:** Fabric always muxes at packet granularity. The new VC2 traffic follows this same model — no sub-packet interleaving.

## Constraints

- **Hardware**: Blackhole only — no Wormhole support required
- **Topology**: 2D fabric configs only — VC2 disabled otherwise
- **Mode exclusion**: Must not be enabled when UDM or mux extension modes are active
- **Indexing**: New sender channel must be last index in flat arrays used for lookup
- **Flow control**: Overlay register for Z-router private channel must be statically assigned
- **Compatibility**: Existing VC0/VC1 behaviour must be completely unchanged

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Templatize existing WorkerConnectionAdapter (not new class) | Avoids code duplication; only the overlay register stream ID differs | — Pending |
| Static overlay register assignment for Z-router private channel | Enables stateless connect API; no runtime allocation tracking needed | — Pending |
| VC2 sender indexed last in flat arrays | Consistent with existing convention; minimizes disruption to existing index calculations | — Pending |
| Private append_fabric_connection_rt_args in Metal layer | Not published under fabric API; only data plane layer calls it | — Pending |

---
*Last updated: 2026-03-16 after initialization*
