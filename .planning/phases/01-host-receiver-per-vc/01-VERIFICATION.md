---
phase: 01-host-receiver-per-vc
status: passed
verified: 2026-03-13
method: git-evidence
commit: bbb7e95b1ec9d7821ebbc37f3cc896e8c6d02c63
---

# Phase 1 Verification: Host Receiver Per-VC

**Status:** passed
**Score:** 6/6 must-haves verified
**Method:** Git evidence (phase executed before GSD phase directories were set up)

## Evidence

**Commit:** `bbb7e95b1ec` — "fabric: migrate host receiver channels to per-VC bool indexing"

**Truth 1 — HR-01: is_receiver_channel_active_per_vc is bool array:**
`FabricEriscDatamoverConfig` and `FabricBuilderContext` updated; `num_used_receiver_channels_per_vc` renamed to `is_receiver_channel_active_per_vc` with type `std::array<bool, MAX_NUM_VCS>`.

**Truth 2 — HR-02: 2D receiver arrays collapsed:**
`fabric_static_sized_channels_allocator` and `fabric_remote_channels_allocator` — receiver arrays collapsed from `[vc][channel_id]` to per-VC scalars.

**Truth 3 — HR-03: receiver_channel_base_address is per-VC scalar:**
`receiver_channel_base_address[vc]` and `remote_receiver_channel_base_address[vc]` are scalars.

**Truth 4 — HR-04: FabricEriscDatamoverConfig and FabricBuilderContext updated:**
Both structs use bool receiver field; `max_receiver_channels_per_vc_` renamed accordingly.

**Truth 5 — HR-05: Dead builder fields removed:**
`receiver_channels_num_buffers`, `local_receiver_channels_buffer_address`, and related fields removed from `FabricEriscDatamoverBuilder`.

**Truth 6 — HR-06: Compiles and tests pass:**
Build passed. Phase 2 commit (`7132db45f5d`) confirms all 12 latency tests pass golden comparison after phase 1+2 changes combined.

## Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|---------|
| HR-01 | satisfied | `is_receiver_channel_active_per_vc` in `erisc_datamover_builder.hpp` |
| HR-02 | satisfied | Allocator 2D arrays collapsed in both allocator files |
| HR-03 | satisfied | Scalar base address fields in both allocators |
| HR-04 | satisfied | `FabricEriscDatamoverConfig` + `FabricBuilderContext` updated |
| HR-05 | satisfied | Dead fields removed from builder (9 files, -231/+145 lines) |
| HR-06 | satisfied | Build + 12/12 latency tests pass |
