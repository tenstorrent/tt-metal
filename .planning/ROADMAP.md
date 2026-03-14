# Roadmap: PR #39538 Decomposition

**6 phases** | **18 requirements mapped** | All v1 requirements covered ✓

| # | Phase | Goal | Requirements | Status |
|---|-------|------|--------------|--------|
| 1 | Host Receiver Per-VC | Migrate host-side receiver channel arrays to per-VC bool/scalar | HR-01 to HR-06 | In Progress |
| 2 | Device Receiver Per-VC | Migrate device-side receiver channel indexing to per-VC | DR-01 to DR-04 | Pending |
| 3 | 1/2 | Complete    | 2026-03-13 | Pending |
| 4 | Device Sender Per-VC | Complete    | 2026-03-13 | Complete (2026-03-13) |
| 5 | 1/1 | Complete    | 2026-03-14 | Pending |
| 6 | 1/1 | Complete    | 2026-03-14 | Pending |

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
*Roadmap created: 2026-03-12*
