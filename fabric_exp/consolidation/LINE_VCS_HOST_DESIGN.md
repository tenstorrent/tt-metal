# Line VCs on 1D fabrics, host side: design for review (2026-11-02; no code yet)

Scope (user, 2026-11-02): three line VCs as the target; VC1 may be a line VC on 1D fabrics; several workers on one VC
parked. The setting has two levels: the fabric's **format** (1D or 2D headers and routing, fabric-wide, from the
topology) and, per VC, a **routing mode** valid within that format. The host validates the combination; the router
follows per-VC facts from host args (the ownership passes). This note covers the host side only; the router side
follows on the ownership variant. Census and numbers: VC2_LINE_1D_VCS.md. Code cited from this worktree (branch
aagarwal/vc2-line, 8abd80861f7); B: = tt_metal/fabric/erisc_datamover_builder.cpp.

## 1. The setting: representation and home

```
enum class FabricFormat { LINE_1D, MESH_2D };             // from the topology: Linear, Ring (and NeighborExchange) -> 1D
enum class VcRouting { OFF, FULL_2D, LINE, NEIGHBOUR, INTER_MESH };
struct FabricVcConfig {                                    // replaces IntermeshVCConfig's VC fields; one per fabric
    FabricFormat format;
    std::array<VcRouting, builder_config::MAX_NUM_VCS> routing;  // routing[0] is never OFF
    bool mesh_pass_through;                                // as today (requires_vc1_mesh_pass_through)
    // derived, for today's users: requires_vc1() = routing[1] != OFF; requires_vc2() = routing[2] != OFF;
    // vc2_line() = routing[2] == LINE && format == MESH_2D
};
```

- Home: `FabricBuilderContext::compute_vc_config()` (today `compute_intermesh_vc_config`, fabric_builder_context.cpp
  :356-421), stored once per fabric and read by the wiring rules, allocator, stream assignment and builder as today.
  The derived accessors keep every current call site unchanged in the first step.
- Inputs (no bare per-VC list from the user): the format from the topology; on 2D, VC1 = INTER_MESH from the mesh graph
  (as today) and VC2 from the existing options (`TT_METAL_ENABLE_FABRIC_VC2`: NEIGHBOUR; with `TT_METAL_FABRIC_VC2_LINE`:
  LINE); on 1D a new option, the number of line VCs, `TT_METAL_FABRIC_LINE_VCS=N` (rtoption `fabric_line_vcs`, 1..3,
  default 1; test yaml `fabric_setup: line_vcs: N`): VC0..N-1 are LINE. VC0 is FULL_2D on 2D and LINE on 1D always.

## 2. Validation (format against per-VC modes; one TT_FATAL each, in compute_vc_config)

| format | VC0 | VC1 | VC2 | also |
| --- | --- | --- | --- | --- |
| MESH_2D | FULL_2D | OFF, INTER_MESH (multi-mesh with intermesh links; as today) | OFF, NEIGHBOUR, LINE | LINE: no express meshes (today's check); Z: no VC2 forwarding channel (wiring rules) |
| LINE_1D | LINE | OFF, LINE | OFF, LINE | line VCs contiguous from VC0 (N = 1..3); VC1 never INTER_MESH on 1D (1D creates no inter-mesh VC1 today); NeighborExchange: N = 1 only (it has no forwarding channel); no tensix mux / UDM extension with N > 1 (VC2's rule today, fabric_builder_context.cpp:400-404); Blackhole only (as VC2) |

Rings: bubble flow control and first-level acks per LINE VC on a ring (deadlock avoidance), as VC0 and VC2 line rings
have them; `bubble_flow_control_enabled_on_vc(vc, ...)` (fabric_builder_config.hpp:118) becomes "the VC's routing is
FULL_2D or LINE" (VC1 INTER_MESH keeps none).

## 3. 1D fabrics creating VC1/VC2 channels

- `RouterVcShape` (router_wiring_rules.cpp:243-323): the 1D arm (:275-280, "No VC1 or VC2 channels are ever created")
  gives each LINE VC the 1D counts (worker + forwarding, `get_num_used_sender_channel_count(topology)`; one receiver);
  `num_vcs` = the number of VCs not OFF (today it is answered from requires_vc1/vc2 even on 1D).
- Producer wiring (fabric_builder_helpers, RouterProducerSlots): each LINE VC's forwarding channel is fed by the same
  VC's receiver on the router facing the other way, as VC0's on 1D and VC2's in 2D line mode.
- Downstream connections (erisc_datamover_builder.cpp, connect to downstream): per LINE VC, one downstream (the next
  router's same-VC forwarding channel); the per-VC args exist (NUM_DOWNSTREAM_SENDERS_VC<n>, VC<n>_DOWNSTREAM_EDM_SIZE,
  PACKED_DOWNSTREAM_VC<n>_SENDER_CHANNEL_IDS, OPEN_DOWNSTREAM_VC<n>_CONNECTIONS, B:1313-1457).

## 4. Allocation per VC count and modes

- Slots (fabric_static_sized_channels_allocator.cpp): a 1D row per number of line VCs, at 4352 B and 82 slots: receiver a
  power of 2, the senders sharing the rest, the worker taking the remainder: N = 1 today's (W 25, F 25, R 32); N = 2
  (12, 13, 16) per VC; N = 3 (9, 10, 8) per VC. Study switch `TT_FABRIC_EXP_SLOTS_1D_VCS=w,f,r` (one triple for every
  line VC). Every router gets the same row, so receivers match at both ends of a link.
- Stream ids (fabric_stream_assignment): per LINE VC its sender channels' free-slots, its receiver's packets-sent and its
  one downstream free-slots copy, from the requirement-driven assignment; VC2's pinned ids 30/31
  (fabric_stream_assignment.hpp:40-58) only for VC2 NEIGHBOUR (they stay as they are). Credits on L1 counters for every
  VC (the study router's plan). The kernel's name sets already cover three VCs on 1D (VC1's edge-1 name, VC2's one
  line name; three TO_RECEIVER names).
- Transaction IDs: the host computes the count per receiver channel and emits it (the kernel's "TODO: pass from host",
  ct_args:751-770): 16 / (receiver channels), rounded down to a power of 2, at most 8 under deadlock avoidance (16 with
  one receiver and none): N = 1 unchanged, N = 2 8, N = 3 4. Named arg `NUM_TRANSACTION_IDS_PER_RECEIVER_CHANNEL`; the
  router reads it when its side lands (until then its own rule gives the same values for N <= 3).
- Sender-table positions: `sender_flat_base(vc)` from the fabric's per-VC maxima (fabric_stream_assignment); on 1D VC0
  0-1, VC1 2-3, VC2 4-5 (fabric-wide, so every router's map agrees). 2D unchanged.

## 5. Worker connection API per VC

- Generalize VC2's private API (fabric_vc2_connection.cpp:159-191: "the VC2 sender at the last flat index") to
  `append_fabric_connection_rt_args_on_vc(fabric_context, src, dst, link_idx, vc, program, worker_core, args)`: the
  router for (src -> dst, link) as today, then that router's worker channel of `vc` (the VC's first sender channel, from
  the same `is_worker_channel` fact the router reads), the same runtime args as today. VC0's public API stays; VC2's
  becomes a wrapper (vc = 2). Validates that `vc` is not OFF on the fabric.
- Test infra: `ConnectionKey::vc_id` (0 or 2 today, tt_fabric_test_device_setup.hpp:42-48) takes any VC; senders with
  `vc_id: 1` on 1D; yaml `line_vcs`.

## 6. Host args the router will read (aligned with the ownership passes)

| fact | arg | today |
| --- | --- | --- |
| worker role per sender channel | `SENDER_CH_<i>_IS_WORKER_CHANNEL` (sender ownership, option (a)) | not emitted here yet; the main patch marks every VC2 channel a worker: in line mode the rule must be "a VC's first channel" (VC2's second channel forwards) |
| VC serviced | `VC<n>_SERVICED`, format-neutral named args | defines FABRIC_2D_VC1/VC2_SERVICED, emitted only on 2D (compute_mesh_router_builder.cpp:904-938) |
| routes as a line on a 2D fabric | `VC<n>_ROUTES_AS_LINE` | `VC2_LINE_MODE` (kept, equal to VC2's) |
| forwards on VC (crossover) | `RX_CH_<r>_FORWARDS_ON_VC` per receiver channel (receiver ownership: `forwards_on_vc`) | define FABRIC_2D_VC0_CROSSOVER_TO_VC1 (kept); VC2 line crosses over on intermesh routers |
| receiver forwards at all | `DISABLE_RX_CH<n>_FORWARDING` | exists (B:1627-1630) |
| first-level acks per VC | `ENABLE_FIRST_LEVEL_ACK_VC<n>` from the VC's routing and the topology | VC1 fixed at 0 (B:1379); VC2 from line mode |
| bubble injection per channel | `SENDER_CH_<i>_IS_INJECTION` from the per-VC rule | exists, VC-number rule |
| transaction IDs | `NUM_TRANSACTION_IDS_PER_RECEIVER_CHANNEL` | kernel rule |

New args are additive: today's router ignores them, so 2D images and N = 1 1D images stay byte-identical (checked
offline).

## 7. How it is checked without router changes

- Host unit tests (CPU, fabric_unit_tests): wiring shapes for 1D with N = 1..3 (counts, bases), stream ids (unique, <= 29,
  equal maps on every router), slot rows (fit 82, receivers equal at both ends), transaction-ID counts, and the
  validation table (each refused combination throws).
- Mock captures (CPU, this worktree's coverage tooling): P150 1D Linear and Ring and the Galaxy as a 1D ring, N = 1..3,
  with the slot log; the router's named-arg maps dumped from the cache (argdump.py) and checked per router: channel
  counts per VC, slots, stream ids, positions, worker roles, first-level acks, forwards-on-VC. The kernels for N >= 2 do
  not compile until the router side lands (the gate refuses them): expected; the args are written before the compile.
- Identity: N = 1 and every 2D coverage case byte-identical to today's images (offline build compare, the existing
  procedure), so the host change alone moves nothing.

## 8. Questions for the review

1. Input on 1D: a line-VC count (`TT_METAL_FABRIC_LINE_VCS`), or reuse the VC2 option (VC2 LINE on 1D) plus a VC1 option?
2. VC2 NEIGHBOUR stays a 2D mode of its own (today's default VC2), next to LINE?
3. Format-neutral `VC<n>_SERVICED` named args next to the 2D defines, or rename the defines?
4. Should I apply the sender ownership host patch (`SENDER_CH_<i>_IS_WORKER_CHANNEL`) here now, with the first-channel
   rule, so the names agree at the rebase?

## 9. As implemented (2026-11-02; the user approved the design with the four recommendations)

Host side only, offline checks only; the router is unchanged.

- The setting: `builder_config::VcRouting` {OFF, FULL_2D, LINE, NEIGHBOUR, INTER_MESH} (fabric_builder_config.hpp) and
  `IntermeshVCConfig::vc_routing` (one per VC); `vc_routings_of` and `validate_vc_routings` (fabric_builder_context.cpp,
  declared in its header) are the derivation and the one validation table, called once in
  `compute_intermesh_vc_config`. Input: `TT_METAL_FABRIC_LINE_VCS` (rtoption `fabric_line_vcs`, 1..3; test yaml
  `fabric_setup: line_vcs`). `requires_vc1` keeps its meaning (the inter-mesh VC1); `requires_vc2` also covers a 1D
  fabric's third line VC. VC2 NEIGHBOUR stays a 2D mode of its own.
- Rules that followed VC numbers now follow the VC's routing mode: the producer slots (worker channel, a line VC's one
  producer: `RouterProducerSlots(facing, counts, routing)`, the two-argument form keeps the 2D answers), the injection
  policy's participation, the bubble flow control gate (`bubble_flow_control_enabled_on_vc(VcRouting)`), the 1D shape and
  turn set (router_wiring_rules.cpp), the fabric-wide channel maxima, the 1D config's L1 layout (channels and downstream
  teardown semaphores per line VC; N = 1 unchanged), the 1D slot row (a block of its own in the allocator), VC1 on 1D
  in the downstream wiring and runtime args, VC1's first-level acks on 1D rings.
- New format-neutral named args (all routers, read by nothing yet): `VC<n>_SERVICED`, `VC<n>_ROUTES_AS_LINE`,
  `RX_CH_<r>_FORWARDS_ON_VC`, `NUM_TRANSACTION_IDS_PER_RECEIVER_CHANNEL` (`builder_config::
  transaction_ids_per_receiver_channel`, the v4 kernel's current rule), `SENDER_CH_<i>_IS_WORKER_CHANNEL` (the sender
  ownership pass's name; the first sender channel of each VC whose mode has a worker: FULL_2D, LINE, NEIGHBOUR yes,
  INTER_MESH no; every channel position, trimmed or not). The live-check skip uses the same worker rule, so a 1D VC1's
  worker channel keeps its connection check (2D answers unchanged). The FABRIC_2D_* defines stay as they are.
- The worker API per VC: `append_fabric_connection_rt_args_on_vc(..., vc, ...)` (fabric_vc2_connection.*); VC2's private
  API is a wrapper (vc = 2). Test infra: `vc_id` 0, 1 or 2; a non-zero VC connects through the per-VC API; `line_vcs`
  reopens the fabric when it changes.

Checks (CPU only):
- Unit tests (fabric_unit_tests, mock HAL): 73 of 73 in the router-builder suites, 8 new (1D line-VC shapes and turns, the
  routing derivation, every refused combination of the table, transaction IDs, producer slots, injection, stream ids for
  three line VCs, the 1D slot rows).
- Mock captures of P150 Linear and Ring and the Galaxy as a ring at N = 1, 2, 3 (and a one-ERISC ring at N = 3), with the
  host's arg dump (`line_vcs_dump_check.py`): every router ERISC passes (channel counts, worker roles, serviced / line /
  forwards-on facts, acks and injection on rings only, transaction IDs, distinct stream ids, flat bases, every line VC
  opening its downstream where VC0 does, slot rows 13/12/16 at N = 2 and 10/9/8 at N = 3). For N >= 2 the router does
  not compile, as expected: its `brought_up` gate, and its own transaction-ID rule (8 per channel on 1D, 24 > 16).
- Identity (every existing config, N = 1 and every 2D case): 38 coverage cases (1D, ring, 2D, torus, neighbour, VC2
  neighbour and line, VC1 and VC1 + VC2 splits, mux, trimming, telemetry, express; two ERISCs and one; P150 and Galaxy)
  captured on the mock with the host before and after the change (`host_captures.py`, prefixes hold_ / hnew_):
  - inputs: per case and ERISC every captured router config's compile command and named args (the new ones aside) are
    the same in 36 of 38 cases; the other two (the two-rank VC1 splits bh_vc1_vc2_1erisc, bh_vc1_vc2line) differ only by
    configs whose logged commands the two ranks' interleaved log garbled or dropped (`host_identity.py --args`;
    results/line_vcs_host_identity_inputs.txt);
  - images: both captures built with the same router sources (variant vc2l_p3b, -O3; the build keeps one config per
    signature): all 871 configs built on both sides with equal inputs have identical .text and loadable bytes
    (`host_identity.py --pairs`; results/line_vcs_host_identity_images.txt). A config's hash changes with the new args,
    so images are paired by inputs, not by hash.

What the router side needs (on the ownership variant):
1. Read the new args instead of its own derivations: `VC<n>_SERVICED` (not FABRIC_2D_*), `VC<n>_ROUTES_AS_LINE` (for
   `vc_routes_as_line`), `RX_CH_<r>_FORWARDS_ON_VC` (crossover), `NUM_TRANSACTION_IDS_PER_RECEIVER_CHANNEL` (ct_args
   :751-770), `SENDER_CH_<i>_IS_WORKER_CHANNEL` (worker channels per VC; a 1D VC1 has one).
2. The 1D branches per VC: `downstream_sender_free_slots_stream` and `local_downstream_credit_stream` return VC0's names
   on 1D (router_common.hpp:174, :197); a 1D VC1 uses VC1's edge-1 names and its forwarding channel's free-slots stream
   (flat position VC1 base + 1), a 1D VC2 the line names it has in 2D.
3. The gate moves to the host: `!has_vc1 || is_2d` and `vc2_config_supported`'s `is_2d` (fabric_erisc_router_v2.cpp
   :95-119) go; the host's table is the check.
4. Producer-credit command buffers for three worker channels (sender ownership §2 #7: one distinct buffer per serviced
   worker channel; today WR for VC0's and WR_REG for VC2's): a third is needed for a 1D VC1 worker.
5. First-level acks per VC from `ENABLE_FIRST_LEVEL_ACK_VC<n>` on the sender side too (the sender's `vc == 0 ? ... :
   false`), now that a 1D VC1 ring acks.
6. Runtime args: VC1's and VC2's 1D downstream blocks follow VC0's (the host packs them when the VC has a receiver).
7. Workers: the test sender picks VC2's adapter for vc 2 and the plain one otherwise (a 1D VC1 worker's free-slots stream
   is a generic one, not VC2's pinned id 30).
