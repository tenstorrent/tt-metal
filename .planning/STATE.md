# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-12)

**Core value:** Each PR is self-contained, correct, and independently reviewable
**Current focus:** Phase 1 — Host Receiver Per-VC (In Progress)

## Current Phase

**Phase 1: Host Receiver Per-VC**

Partially complete. All allocator and builder host-side changes done:
- `FabricStaticSizedChannelsAllocator`, `FabricRemoteChannelsAllocator` — receiver arrays collapsed to per-VC scalars
- `FabricEriscDatamoverConfig` — `is_receiver_channel_active_per_vc` (bool)
- `FabricBuilderContext` — `max_receiver_channels_per_vc_` (bool)
- Dead builder fields removed

Remaining for Phase 1:
- Verify compilation
- Run tests to confirm no regressions

## Next Phase

**Phase 2: Device Receiver Per-VC**
Migrate device-side kernel code to per-VC receiver channel indexing.
Key files: `fabric_erisc_router_ct_args.hpp`, `fabric_erisc_datamover_channels.hpp`, `fabric_erisc_router.cpp`
