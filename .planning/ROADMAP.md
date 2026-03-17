# Roadmap: PR #39538 Decomposition

**10 phases** | **18 requirements mapped** | All v1 requirements covered ✓

| # | Phase | Goal | Requirements | Status |
|---|-------|------|--------------|--------|
| 1 | Host Receiver Per-VC | Migrate host-side receiver channel arrays to per-VC bool/scalar | HR-01 to HR-06 | In Progress |
| 2 | Device Receiver Per-VC | Migrate device-side receiver channel indexing to per-VC | DR-01 to DR-04 | Pending |
| 3 | 1/2 | Complete    | 2026-03-13 | Pending |
| 4 | Device Sender Per-VC | Complete    | 2026-03-13 | Complete (2026-03-13) |
| 5 | 1/1 | Complete    | 2026-03-14 | Pending |
| 6 | 1/1 | Complete    | 2026-03-14 | Pending |
| 7 | Reorganize buffer slot configs by VC | Replace PerVcBufferSlots with VcSlotConfig array pattern | Phase 7 goal | Complete (2026-03-14) |
| 8 | 2/2 | Complete   | 2026-03-14 | Pending |
| 9 | 3/3 | Complete   | 2026-03-14 | Pending |
| 10 | Split channel_allocs into per-VC in device CT args | Separate channel_allocs into per-VC channel allocations in device CT args | Phase 10 goal | Pending |

---

## Phase 1: Host Receiver Per-VC

**Goal:** All host-side receiver channel arrays use per-VC bool/scalar types with no 2D inner arrays.

**Requirements:** HR-01, HR-02, HR-03, HR-04, HR-05, HR-06

**Success criteria:**
1. `is_receiver_channel_active_per_vc` is `std::array<bool, MAX_NUM_VCS>` everywhere in host code
2. No 2D receiver channel array `[vc][channel_id]` exists in allocators or builder
3. `receiver_channel_base_address[vc]` and `remote_receiver_channel_base_address[vc]` are scalars
4. Codebase compiles cleanly

**Status:** In Progress
- ✓ `FabricStaticSizedChannelsAllocator` (hpp + cpp) — receiver arrays collapsed
- ✓ `FabricRemoteChannelsAllocator` (hpp + cpp) — receiver arrays collapsed
- ✓ `FabricEriscDatamoverConfig` — `is_receiver_channel_active_per_vc` (bool)
- ✓ `FabricBuilderContext` — `max_receiver_channels_per_vc_` (bool)
- ✓ Dead fields removed from `FabricEriscDatamoverBuilder`

---

## Phase 2: Device Receiver Per-VC

**Goal:** Device-side kernel code uses per-VC accessors for receiver channels with no flat 2D indexing.

**Requirements:** DR-01, DR-02, DR-03, DR-04

**Success criteria:**
1. No flat receiver channel array indexed by `[vc][channel_id]` in kernel headers
2. CT args wire format unchanged (host/device remain compatible)
3. Receiver channel pointer initialization uses per-VC template parameters
4. Kernels compile and fabric tests pass

---

## Phase 3: Host Sender Per-VC

**Goal:** Host-side sender channel arrays use consistent per-VC indexing.

**Requirements:** HS-01, HS-02

**Plans:** 2/2 plans complete

Plans:
- [ ] 03-01-PLAN.md — Remove dead AllocatorConstructionParams; fix is_sender_channel_serviced_ sizing; document num_used_sender_channels as derived
- [ ] 03-02-PLAN.md — Fix compute_mesh_router_builder.cpp to use per-VC sum; build + sanity test

**Success criteria:**
1. `num_used_sender_channels_per_vc` is consistently typed and named
2. No mixed flat/per-VC sender channel indexing in allocators or builder

---

## Phase 4: Device Sender Per-VC

**Goal:** Device-side kernel code uses per-VC accessors for sender channels.

**Requirements:** DS-01, DS-02

**Plans:** 1/1 plans complete

Plans:
- [x] 04-01-PLAN.md — Add VC0_SENDER_CHANNEL_START constant; replace is_sender_channel_serviced[0] literals; build + sanity test

**Success criteria:**
1. No flat sender channel array indexed by `[vc][channel_id]` in kernel headers
2. CT args wire format unchanged
3. Kernels compile and fabric tests pass

---

## Phase 5: Channel Allocator

**Goal:** Channel allocator uses both per-VC receiver and sender channel data consistently.

**Requirements:** CA-01, CA-02

**Plans:** 1/1 plans complete

Plans:
- [ ] 05-01-PLAN.md — Update emit_channel_allocations_ct_args signatures to per-VC arrays on both allocators; update call site; build + sanity test

**Success criteria:**
1. Allocator API has no mixed flat/per-VC indexing
2. Both sender and receiver use per-VC types throughout the allocator

---

## Phase 6: Stream Reg Assignment

**Goal:** Host stream register assignment table/map uses per-VC indexing for both sender and receiver.

**Requirements:** SR-01, SR-02

**Plans:** 1/1 plans complete

Plans:
- [ ] 06-01-PLAN.md — Add per-VC grouping arrays to StreamRegAssignments; update CT-arg emission to use per-VC accessors; build + sanity test

**Success criteria:**
1. Stream register assignment uses per-VC indexing
2. No flat stream register arrays that mix VC assignment

---

## Phase 7: Reorganize buffer slot configs by VC in allocator

**Goal:** Restructure buffer slot configuration internals in FabricStaticSizedChannelsAllocator to use per-VC array-of-struct indexing, replacing PerVcBufferSlots with VcSlotConfig and collapsing output parameters into return types.
**Requirements**: Phase 7 goal (additive beyond original 18 requirements)
**Depends on:** Phase 6
**Plans:** 1 plan

Plans:
- [ ] 07-01-PLAN.md — Define VcSlotConfig struct; convert tables and get_optimal_num_slots_per_vc to per-VC array return; restructure configure_buffer_slots_helper signature; build + sanity test

**Success criteria:**
1. PerVcBufferSlots struct removed entirely
2. VcSlotConfig struct with per-VC array pattern used by all static tables
3. get_optimal_num_slots_per_vc returns array-of-VcSlotConfig instead of 8 output ref scalars
4. configure_buffer_slots_helper uses struct return instead of 4 output array params
5. CT args wire format unchanged — sanity test passes

---

## Phase 8: Host-side per-VC consolidation

**Goal:** Merge all remaining _vc0/_vc1 named constants and split functions into _per_vc arrays and unified functions across host builder code.
**Requirements**: Phase 8 goal (additive beyond original 18 requirements)
**Depends on:** Phase 7
**Plans:** 2/2 plans complete

Plans:
- [ ] 08-01-PLAN.md — Merge constants to _per_vc arrays in fabric_builder_config; merge channel mapping init functions; build + test
- [ ] 08-02-PLAN.md — Convert erisc_datamover_builder locals/named_args to per-VC; update fabric_tensix_builder and router_connection_mapping call sites; build + test

**Success criteria:**
1. No `_vc0`/`_vc1` suffixed constants or functions remain in `fabric_builder_config`
2. All host builder call sites use unified per-VC function or array indexing
3. `initialize_vc_mappings(vc)` replaces split vc0/vc1 init functions
4. CT args wire format unchanged — sanity test passes

---

## Phase 9: Device-side kernel per-VC templates

**Goal:** Add per-VC template helpers to device kernel headers and refactor router kernel to use templated per-VC functions.
**Requirements**: Phase 9 goal (additive beyond original 18 requirements)
**Depends on:** Phase 8
**Plans:** 3/3 plans complete

Plans:
- [ ] 09-01-PLAN.md — Add per-VC constexpr foundation constants and per-VC sender channel arrays to ct_args header
- [ ] 09-02-PLAN.md — Template any_sender_channels_active/update_telemetry on VC; split runtime arrays into per-VC tuples
- [ ] 09-03-PLAN.md — Migrate all remaining flat sender loops to per-VC; build + sanity test; clean up unused flat arrays

**Success criteria:**
1. Per-VC template helpers exist for sender channel queries
2. `any_sender_channels_active` is templated on VC with per-VC free_slots arrays
3. No flat cross-VC sender channel arrays remain in device kernel code
4. Kernels compile and fabric tests pass

---

## Phase 10: Split channel_allocs into per-VC in device CT args

**Goal:** Split the unified `channel_allocs` in device CT args into per-VC channel allocations, completing the per-VC separation at the host-device boundary.
**Requirements**: Phase 10 goal (additive beyond original 18 requirements)
**Depends on:** Phase 9

**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 10 to break down)

**Success criteria:**
1. Device CT args use per-VC channel allocation structures instead of unified `channel_allocs`
2. Host emission and device parsing agree on per-VC format
3. Kernels compile and fabric tests pass

---
*Roadmap created: 2026-03-12*
