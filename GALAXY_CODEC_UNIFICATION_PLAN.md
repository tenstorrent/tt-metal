# Fabric Codec Unification Plan

> **Step names:** every numbered step in this document has a short name — see
> **§0 Step index** in `GALAXY_CODEC_UNIFICATION_STATUS.md`. Prefer the names when discussing
> the work; the numbers exist for cross-referencing between these three documents.


Strategy for making **one** 2D packet encoding — the indexed destination-keyed ABI — the only 2D
codec in the fabric, and for removing the express-specific *compile path* wherever the fork is
about the codec rather than about express topology.

Authority: this document plans a change to code governed by
`GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md` (codec/ABI semantics) and
`GALAXY_DEVICE_ROUTER_KERNEL_CONTRACT.md` (ERISC realization). Where this plan and those contracts
disagree about *semantics*, the contracts win. Where they disagree about *implementation status*,
this document wins — it was written against the tree at `42204c541c3` and the contracts still
describe several already-implemented pieces as unimplemented.

Three reference documents, plus one tracker:

| doc | carries | churn |
|---|---|---|
| `GALAXY_CODEC_UNIFICATION_COMPARISON.md` | the two 2D implementations described side by side — wire format, sizing, unicast, multicast, decode, intermesh, and the defects found in each | stable |
| **this document** | scope, the fork ledger, blockers, design decisions, phasing, risks | stable |
| `GALAXY_CODEC_UNIFICATION_IMPLEMENTATION_GUIDE.md` | the ordered, file-by-file edits and their gates | stable |
| `GALAXY_CODEC_UNIFICATION_STATUS.md` | what is implemented, gate results, decision states, measurements, log | **edited constantly** |

If implementation reveals that a decision here is wrong, fix *this* document and note it in the
status log — the tracker is not the source of truth for design.

---

## 0. Executive summary

### 0.1 The headline correction to the contracts

Both contracts read as "target, not implemented." That is out of date. As of `42204c541c3` the
indexed ABI is **substantially implemented and shipping behind `FABRIC_EXPRESS_ENABLED`**:

| Contract claim | Reality in tree |
|---|---|
| codec §0.2 "specified, not implemented" | `IndexedMeshRoutingFields` is complete in `fabric_common.h:223-507` |
| codec §4.1 dest-major 2-bit vectors | `indexed_route_vectors_t`, packed by `compressed_routing_path.cpp:166` |
| codec §5.7.1 per-chip reverse trees | `mcast_reverse_tree.{hpp,cpp}`, embedded at `control_plane.cpp:1976-2006` |
| codec §7.1 unicast widen | `fabric_set_indexed_unicast_route` (`tt_fabric_api.h:419`) |
| codec §7.3 multicast encoder | `fabric_set_indexed_mcast_route` (`tt_fabric_api.h:451`) |
| codec §7.3.1 source multi-inject | `fabric_multicast_source_inject_noc_unicast_write` (`mesh/api.h:1222`) |
| codec §4.5 intermesh landing | `fabric_set_indexed_intermesh_landing_route` (`tt_fabric_api.h:527`) |
| kernel §3.3-§3.5 dense 16-way dispatch | `admit_express_combo` / `forward_express_combo` (`fabric_erisc_router.cpp:993+`) |
| kernel §4.2 exit predicate | `action_is_intermesh_exit` + `fabric_erisc_router.cpp:2178-2210` |

So **this project is not "build the indexed codec." It is "delete the second codec."**

### 0.2 What the work actually is

`ControlPlane::express_routing_enabled(mesh_id)` (`control_plane.cpp:1606`) is one boolean doing
**five unrelated jobs**. Unification means splitting it, promoting exactly one of those jobs to
unconditional, and deleting the alternative that job selects.

```text
express_routing_enabled(mesh_id)   ← ONE predicate, FIVE consumers today
  │
  ├─ G1  packet ABI selection ......... indexed maps vs legacy hop program
  │        encode (fabric.cpp:392), L1 embed (control_plane.cpp:1976),
  │        decode define (compute_mesh_router_builder.cpp:911),
  │        decode CT args (erisc_datamover_builder.cpp:1261)
  │        ► THIS is what gets promoted to unconditional and then deleted
  │
  ├─ G2  express topology ............. Z ports, Z wiring, Z neighbours, chord materialization
  ├─ G3  safety / BFC ................. injection policy, protected rings, deadlock avoidance
  ├─ G4  resource sizing .............. VC0 5-wide, VC1 counter credits, stream budget
  └─ G5  optimization disablement ..... speedy, super-speedy, trim bypass, channel remap
           ► G2-G5 STAY keyed on express. They are genuinely about express.
```

This is exactly what codec §4.5.1 already mandates and what the tree violates:

> The concrete CT/FabricContext selector **must not** be `express_routing_enabled`: that flag
> controls express topology, Z, and associated BFC artifacts, not legacy-vs-indexed packet
> interpretation.

The tree's own comments acknowledge the coupling as deliberate-for-now
(`fabric.cpp:392`: *"All four ABI gates … key on express_routing_enabled"*). This plan unwinds it.

### 0.3 Outcome

When complete:

- **One** 2D encode path, **one** 2D decode path, **one** L1 2D layout, for every 2D mesh.
- `FABRIC_EXPRESS_ENABLED` is no longer a **codec** selector — it survives, under its own name and
  unchanged condition, gating only Z-port capacity. Z is just action bit 4, present in the ABI
  whether or not a given mesh has chords.
- ~1,100 lines of legacy 2D codec deleted (hop program, `hop_index`, `branch_*_offset`,
  `turn_point`, spine/branch multicast, `NOOP`-recompute, sender/RX header mutation, turn-status
  plumbing).
- Galaxy `[32,4]` fits a **96 B** header (not 112 B) because retiring the legacy control fields buys
  back the 5 bytes the wider route buffer costs. See §4.4 — this is load-bearing, not incidental.
- Two live defects fixed as a side effect (§3.1 B5, §3.1 B6).

---

## 1. Scope

### 1.1 In scope

Everything keyed on `FABRIC_EXPRESS_ENABLED` / `express_routing_enabled` **as a codec selector**, on
both sides of the divide:

- the four host gates that pick the ABI: worker encode defines (`fabric.cpp:392`), L1 embed
  (`control_plane.cpp:1976`), kernel define (`compute_mesh_router_builder.cpp:911`), kernel CT args
  (`erisc_datamover_builder.cpp:1261`)
- the kernel-side producers: `tt_fabric_api.h`, `mesh/api.h`, `udm/*`, `cq_relay.hpp`
- the kernel: the four `express_enabled` RX branches, the legacy 2D admit/forward switches,
  `fabric_edge_node_router.hpp`, the turn/header-mutation plumbing
- the kernel-side codec: `fabric_common.h`, `fabric_routing_path_interface.h` — the legacy 2D
  representation and its decode
- the packet-header fields the legacy path needs (`routing_fields`, `is_mcast_active`) and the
  route-buffer tier they pay for
- whatever must change for the indexed path to work on *non-express* fabric at all (§3)

`FABRIC_EXPRESS_ENABLED` is 9 preprocessor sites across 6 kernel-side files, plus one dispatch
kernel:

```text
hw/inc/tt_fabric_api.h                                  producers
hw/inc/mesh/api.h                                       source multi-inject helper
hw/inc/udm/tt_fabric_udm.hpp                            relay-to-mux selection
hw/inc/udm/tt_fabric_udm_impl.hpp                       initial-direction calculation
hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp        express_enabled + shape + ingress CT args
hw/inc/edm_fabric/routing_plane_connection_manager.hpp   ⚠ Z-PORT CAPACITY, not codec — leave untouched
impl/dispatch/kernels/cq_relay.hpp                      route-vs-unicast producer choice
```

Of those nine, **eight are codec** and become `#if defined(FABRIC_2D)` (§9 Q2). Exactly **one** is a
genuine express fact.

#### What survives as an express guard, device-side

| express fact | mechanism after unification | preprocessor guard? |
|---|---|---|
| Z-port capacity — 6 vs 4 connection slots | `TT_FABRIC_MAX_ROUTING_PLANE_CONNECTIONS` | **yes — and it keeps the name `FABRIC_EXPRESS_ENABLED`.** It is an array bound in a type, so it cannot be a CT arg; and its emission condition (`express_routing_enabled`) does not change, so a second spelling would buy nothing. See §4.1 |
| Z sender wired on *this* router | `express_arm_is_realizable<DOWNSTREAM_EDM_SIZE, Z>()` | no — CT-resolved from the downstream array size |
| VC0 five wide | `ACTUAL_VC0_SENDER_CHANNELS` | no |
| VC1 on counter credits | `vc1_uses_counter_credits` | no |
| per-sender BFC role | `SENDER_CH_i_IS_INJECTION` | no |
| speedy / trim disabled | `enable_speedy_vc0`, trim CT args | no |
| deadlock avoidance | `enable_deadlock_avoidance` | no |
| intermesh ingress channels | `IS_RECEIVER_CHANNEL_n_INTERMESH_INGRESS` | no — and it is a 2D fact, not an express one |
| mesh shape | `MESH_{Y,X}_SIZE` CT args, `FABRIC_2D_MESH_{Y,X}_SIZE` defines | 2D, not express |

That express's non-codec effects are already numeric CT args rather than compile forks is by design —
kernel §5: *"the kernel consumes `SENDER_CH_i_IS_INJECTION`; it does not derive it. It must not infer
injection from direction pairs, Z, decode phase, line state, or the self bit."* So there is no hidden
population of express sites waiting to be re-keyed.

Host-side, `express_routing_enabled(mesh_id)` survives untouched for G2-G5.

### 1.2 Out of scope

| Out of scope | Why |
|---|---|
| **Dead code** — `PacketHeader` / `RoutingFields` / `packet_header_t`, the non-low-latency `ROUTING_MODE` arms, `port_direction_table`, the `intra_mesh_routing_path_t<1,true>` stub, `DYNAMIC_ROUTING_ENABLED` | Unrelated to the express divide. Deferred to its own discussion — do not fold it into this project's diffs, even where it looks free. |
| **VC-shape fork axes** — `FABRIC_2D_VC1_ACTIVE`, `FABRIC_2D_VC2_SERVICED`, `FABRIC_2D_VC1_SERVICED`, `FABRIC_2D_VC0_CROSSOVER_TO_VC1` | Larger and independent of the codec. This project must leave them undisturbed. |
| **`UDM_MODE` as a variant** | Only its *producer* forks on express, and that is in scope. UDM as a kernel variant is not. |
| **Arch forks** — `ARCH_WORMHOLE` / `ARCH_BLACKHOLE` | Real silicon differences. |
| **The profiler / noc-tracing decode** — `tools/profiler/fabric_event_profiler.hpp` `recordRoutingFields2D`, and its host side in `profiler.cpp` | Out of scope, owner call. Note only: it reads `routing_fields.branch_west_offset`, so Phase 4.4 makes it a **build break** rather than an optional follow-on. Disposition to be decided then; no work planned here. |
| **1D low-latency codec** (`LowLatencyRoutingFields`, shift/refill) | A different codec on a different header type for a single-downstream topology. Rationale in §1.3. |

### 1.3 Why 1D stays separate

"All fabric traffic uses the same encoding scheme" is delivered here as **all 2D traffic**. 1D keeps
its own codec, deliberately.

**1D's 2-bit hop program is already a positional map** — indexed by distance, via the per-hop shift,
rather than by coordinate. It needs exactly four actions (`NOOP` / `WRITE_ONLY` / `FORWARD_ONLY` /
`WRITE_AND_FORWARD`), so 2 bits is the information-theoretic minimum and the current encoding hits
it. The indexed representation uses 8 bits per entry because 2D needs six one-hot output bits.

The indexed codec's real advantage is **immutability**, and that pays off in 2D specifically because a
multi-output router must hand the *same* header to several senders. A line has exactly one downstream
(`downstream_edm_interfaces[0]`), so there is nothing to share a header across; the shift it would
remove is one instruction.

Cost of folding 1D in, against that one instruction:

| line length | today | indexed (1 B/coord) | delta |
|---|---|---|---|
| 8 (T3K) | **48 B** | 64 B | +33% |
| 16 | **48 B** | 64 B | +33% |
| 32 | 64 B | 80 B | +25% |
| 64 (quad-galaxy) | **64 B** | 112 B | **+75%** |

Header-size sensitivity is empirically established in this tree: the 80 B 2D tier is disabled because
it *"de-stabilized some Mesh benchmarks for 8X4 mesh"* (`fabric_context.hpp:156`). Small-message 1D
traffic — semaphore signalling, small all-reduce, fused atomic-inc — is where the header fraction
matters most.

A packed 2-bit coordinate-indexed 1D map would be header-neutral at every line length, but codec
§4.2.1 rejects packed packet overlays, so it would not share an encoding with 2D anyway — the same two
codecs plus a migration.

Structurally the split is also cheap to keep: 1D is **~104 lines** of the router (2.5%), plus three
overloads (`can_forward_packet_completely`, `receiver_forward_packet`,
`update_packet_header_for_next_hop`). The router is already 67% shared, and both header types derive
from the same `PacketHeaderBase`.

Two 1D improvements sometimes attributed to unification are reachable without it, and should be
pursued separately: widening `encode_1d_sparse_multicast` past `uint16_t` and supporting
`ExtensionWords > 0` (issue #36581), and skipping the per-hop `route_buffer` copy in the non-refill
branch of `update_packet_header_for_next_hop` (`fabric_edm_packet_transmission.hpp:420-427`), which
copies unchanged words every hop.
| **Changing multicast capability** | Express multicast is implemented, tested and working today (`fabric_set_indexed_mcast_route`, the reverse trees, and the `fabric_multicast_source_inject_*` multi-inject helper all exist and ship). Codec §7.3.1's "express meshes ship unicast only" is **stale** — it predates the multi-inject helper, and the contract itself concedes nothing enforces it. Unification must preserve express multicast exactly as it behaves now, and make non-express multicast work through the same path. |
| **Arbitrary-pattern VC1 safety / BFC** | assessment §5.7.5. Untouched. |
| **Re-enabling speedy / trim under express** | kernel §6. Stays disabled (G5). Unification must not silently re-enable it — see §5 risk R4. |
| **New topology or command extensions** | assessment's extension record. |
| **Removing Z from the ABI on non-Z arches** | Z is bit 4 of the action byte on every 2D build. WH has no Z sender; the kernel must reject a Z action rather than the ABI omitting the bit. Keeping the bit is what makes the ABI uniform. |

---

## 2. Current-state fork ledger

Every site where the codec forks today. This is the demolition list.

### 2.1 Host: ABI selection (G1) — 4 gates, all to become unconditional

| # | Site | Current | Target |
|---|---|---|---|
| H1 | `fabric.cpp:392-397` — worker encode defines via `get_express_kernel_defines` | per-mesh, express only | always emit shape defines for 2D |
| H2 | `control_plane.cpp:1976` — L1 layout branch in `compute_and_embed_2d_routing_path_table` | express → indexed vectors; else legacy `compressed_route_2d_t[256]` | always indexed vectors |
| H3 | `compute_mesh_router_builder.cpp:911` — `defines["FABRIC_EXPRESS_ENABLED"]` | express only | delete define; kernel path unconditional |
| H4 | `erisc_datamover_builder.cpp:1261` — `MESH_Y_SIZE` / `MESH_X_SIZE` / `IS_RECEIVER_CHANNEL_n_INTERMESH_INGRESS` | inside `if (express_routing_enabled)` | always emitted for 2D |

Plus one gate that already wants unconditional treatment and just has a stray guard:

| # | Site | Note |
|---|---|---|
| H5 | `fabric_context.cpp:359-364` — `get_max_2d_indexed_route_bytes_from_topology` skips non-express meshes | remove the `continue`; the function is otherwise already shape-generic |
| H6 | `fabric.cpp:704-714` — `get_fabric_kernel_defines()` **TT_FATALs** if any mesh uses express | after unification every 2D mesh needs the node-aware overload, so this becomes a hard requirement for all 2D, not an express carve-out |

### 2.2 Client-facing API layers (what other teams call)

The layer above the producers. Audited for step 2.5, and the result is the most load-bearing fact in
the project:

> **No public 2D fabric API exposes a legacy-shaped signature.** All of them are destination-shaped —
> `(dst_dev_id, dst_mesh_id)`, or that plus `ranges.e/w/n/s`. None takes `direction + num_hops`. The
> client surface is already in the indexed vocabulary; the legacy encoder was the odd one out beneath
> it. **So no public signature changes and no consumer migrations are required.**

| layer | entry points | route path | needs change? |
|---|---|---|---|
| `hw/inc/mesh/api.h` | 38 (`fabric_{unicast,multicast}_noc_*` × `{plain,_set_state,_with_state}`) | plain / `_set_state` → forking encoders; `_with_state` never touches the route | no |
| `hw/inc/linear/api.h` | ~24 route-setting sites; **has 2D paths** under `#if defined(FABRIC_2D)` | forking encoders; multicast is one direction per slot, by connection tag | no — see below |
| `hw/inc/udm/tt_fabric_udm_impl.hpp` | `calculate_initial_direction` → `fabric_set_unicast_route` | forking encoder | express arm only (P7/P8) |
| `mesh/api.h` `fabric_multicast_source_inject_*` | 1 | the only public API calling an indexed encoder directly | unconditional (P6) |

**`linear/api.h` independently confirms decision Q1.** Its 2D multicast shim
(`linear/api.h:33-61`) switches on `slot.tag` and passes **exactly one nonzero extent per call** —
the per-direction contract, already hard-coded. That also localises where Q1's single-output-root
assert belongs: `linear/api.h` cannot express a multi-direction rectangle and so cannot trip it;
`mesh/api.h` takes all four extents together and is the layer the assert protects.

Two properties confirmed rather than assumed: `_set_state`/`_with_state` reuse is codec-agnostic (the
worker's template header is never mutated in transit, and indexed maps are immutable by
construction), and `SetRoute=false` has no callers outside `mesh/api.h`.

### 2.3 Device: producers

| # | Site | Fork | Target |
|---|---|---|---|
| P1 | `tt_fabric_api.h:288-295` — `fabric_set_unicast_route` (2D) | `#if FABRIC_EXPRESS_ENABLED` + `if constexpr (!called_from_router)` → indexed; **router path always legacy** | indexed unconditionally, both worker and router |
| P2 | `tt_fabric_api.h:205-227` — `fabric_set_mcast_route` | same shape | indexed unconditionally |
| P3 | `tt_fabric_api.h:48-68` — `fabric_set_single_hop_unicast_route*` | **no gate at all** — always legacy. `fabric_set_indexed_single_hop_*` (line 586) is **dead code** | route to the indexed helper; delete the legacy one |
| P4 | `tt_fabric_api.h:116` — `fabric_set_route` (direction + hop counts) | **legacy-only primitive, no indexed equivalent** (direction-plus-hop-count *is* the legacy model). Two external consumers: `cq_relay.hpp:90` and `test_fabric_set_unicast_route.cpp`, which uses it as its **reference** encoding | delete after P2, P5, and the §3.3 disposition |
| P5 | `cq_relay.hpp:89` — `#if GALAXY_CLUSTER && !FABRIC_EXPRESS_ENABLED` | picks `fabric_set_route` vs `fabric_set_unicast_route` | keep only the `fabric_set_unicast_route` arm |
| P6 | `mesh/api.h:1198` — `fabric_multicast_source_inject_noc_unicast_write` | whole function under `#if FABRIC_EXPRESS_ENABLED` | unconditional (the API is useful for any multi-output root, and non-express roots are single-output so it degenerates safely) |
| P7 | `udm/tt_fabric_udm_impl.hpp:107` — `calculate_initial_direction` | indexed vs `get_ns_hops()` | indexed only |
| P8 | `udm/tt_fabric_udm.hpp:482` — `select_relay_to_mux_connection` | indexed vs `get_ns_hops()` | indexed only |
| P9 | `routing_plane_connection_manager.hpp:21` — `TT_FABRIC_MAX_ROUTING_PLANE_CONNECTIONS` 6 vs 4 | `FABRIC_EXPRESS_ENABLED && ARCH_BLACKHOLE` | **no change at all** — this is G2 (Z-port presence), not G1, and the define keeps its name and condition. The only genuine express site left device-side. |

### 2.4 Device: kernel

| # | Site | Fork |
|---|---|---|
| K1 | `fabric_erisc_router.cpp:2149` | `if constexpr (express_enabled)` — indexed RX decode/admit vs `get_cmd_with_mesh_boundary_adjustment` + `can_forward_packet_completely` |
| K2 | `fabric_erisc_router.cpp:2252` | same split on the forward side |
| K3 | `fabric_erisc_router.cpp:1955` | `!UPDATE_PKT_HDR_ON_RX_CH && !express_enabled` suppresses `update_packet_header_before_eth_send` |
| K4 | `fabric_erisc_router.cpp:2550` | `if constexpr (express_enabled)` — caches `express_local_y/x` |
| K5 | `fabric_edge_node_router.hpp` (whole file) | `recompute_path` + `get_cmd_with_mesh_boundary_adjustment`; `NOOP` overload. Dead under express, sole path otherwise |
| K6 | `fabric_erisc_router.cpp` turn plumbing | `is_spine_direction`, `TURN_STATUS_ARRAY_SIZE`, `sender_channels_turn_status`, `IS_TURN`, `update_packet_header_before_eth_send` 2D body |
| K7 | `fabric_erisc_router_ct_args.hpp:432-447` | the `#if defined(FABRIC_EXPRESS_ENABLED)` block itself |
| K8 | WH + `FABRIC_2D_VC1_ACTIVE` | kernel §3.3 compass-only bit-test admission exception. **Retained** — but must be re-expressed as "no Z sender on this build," not "not express" |

### 2.5 Packet header fields to retire (codec §4.5.1)

| Field | Bytes | Blocking on |
|---|---|---|
| `LowLatencyMeshRoutingFields routing_fields` (`hop_index`, `branch_east_offset`, `branch_west_offset`) | 4 | K1-K6, P1-P5 all gone |
| `is_mcast_active` | 1 | written but never read for routing today |

Retiring both drops the 2D header base from **61 B → 56 B**. See §4.4.

---

## 3. Blockers

These are the things that make "just delete the `#if`" not work. Each is a real finding in the tree,
not a hypothetical.

### 3.1 Hard blockers

**B1 — `MAX_INDEXED_MESH_X = 4` excludes in-tree meshes.**
`fabric_common.h:299`. The indexed vector table is bounded at `[Y≤64, X≤4]`. But the tree ships
2D descriptors with X > 4:

| Shape | Descriptors | Y-table | X-table | Total | Packet Y+X |
|---|---|---|---|---|---|
| `[8,8]` | `dual_bh_galaxy_torus_xy`, `dual_galaxy` | 16 B | 16 B | 32 B | 16 |
| `[8,16]` | `quad_galaxy`, `quad_galaxy_torus_xy` | 16 B | 64 B | 80 B | 24 |
| `[16,8]` | `16x8_quad_bh_galaxy_torus_xy` | 64 B | 16 B | 80 B | 24 |
| `[1,16]` | `bh_lbx2_1x16` | 1 B | 64 B | 65 B | 17 |

None of these are anywhere near the 1024 B slot. The `X ≤ 4` bound is an artifact of sizing the
constant for the Galaxy `[64,4]` worst case, not a real constraint. Fix: bound on **total region
bytes** (`Y·⌈Y/4⌉ + X·⌈X/4⌉ ≤ 1024`), which admits everything up to `[32,32]` (512 B). Keep a
generous per-axis cap for the packed 6-bit tree descriptor (`≤ 64`).

**B2 — multicast reverse trees have no source on a LINE axis.**
`control_plane.cpp:1979-1985` builds trees from `ring_for_direction()`, which returns
`ExpressRingTopology`. For the Y axis that is `get_express_rings()` — **null on every non-express
mesh**. The generic alternative, `derive_ordinary_ring_topology`
(`express_ring_topology.cpp:549`), returns `nullopt` unless `axis_wraps(...)`. So:

```text
express mesh (Y has chords)   → express_rings          → trees ✓
non-express TORUS Y           → derive_ordinary_ring   → trees ✓ (needs B3 fixed)
non-express LINE Y            → nothing                → NO TREES  ✗
```

LINE-axis 2D meshes in tree: `single_bh_galaxy` `[8,4]`, `p150_x8` `[2,4]`, `dual_bh_lb` `[2,4]`,
`16x4_dual_bh_galaxy_2d` `[16,4]`, `quad_galaxy` `[8,16]`, `dual_galaxy` `[8,8]`, plus the small
`[2,2]`/`[1,2]` shapes. Without a fix, **unification silently removes multicast** on all of them.
Fix: add `derive_line_axis_topology(mesh_graph, mesh_id, axis)` — the degenerate arborescence where
`next_row(cur, dst)` is `cur ± 1` toward `dst`. Trivially an arborescence, so the gate always
passes.

**B3 — `x_rings_` is only derived for express meshes.**
`routing_table_generator.cpp:54-63`: the `derive_ordinary_ring_topology(..., axis=1)` call sits
*after* the `if (!rings.has_value()) continue;`. So even a non-express **torus** mesh gets no X
rings. One-line fix, but it must land or B2's torus case still fails.

**B4 — `fabric_set_single_hop_unicast_route*` is a live ABI mismatch on express meshes.**
`tt_fabric_api.h:48-68` has **no** express gate and writes `route_buffer[0] = <opposite dir bit>`
with `hop_index == 0`. The indexed kernel decodes `route_buffer[local_y]`. Callers:
`models/demos/deepseek_v3_b1/unified_kernels/{all_reduce,all_gather,broadcast,reduce_to_one_b1,sdpa_reduce_worker}.hpp`
— i.e. exactly the Galaxy workloads express targets. Meanwhile
`fabric_set_indexed_single_hop_unicast_route_from_direction` (line 586) exists and has **zero
callers**.

> This is a defect in the current tree, independent of unification. It should be fixed first, on its
> own, as Phase 0. See guide step 0.1.

**B5 — the router-side (`called_from_router`) 2D encode is still legacy even under express.**
`tt_fabric_api.h:288` guards the indexed call with `if constexpr (!called_from_router)`. Under
express the router-side callers (`recompute_path` in `fabric_edge_node_router.hpp`) are unreachable
because `get_cmd_with_mesh_boundary_adjustment` is only called from the non-express kernel arm — so
this is dead rather than wrong today. But it means the express path was never exercised for
router-originated re-encode, and the intermesh landing uses a *different* entry point
(`fabric_set_indexed_intermesh_landing_route`). Unification must confirm the landing encoder fully
covers what `recompute_path` used to do before `recompute_path` is deleted.

### 3.2 Soft blockers / sizing consequences

**B6 — Galaxy `[32,4]` route buffer is 36 B, one byte past the 35 B tier.**
See §4.4. Resolved by the header-field retirement, but the *ordering* matters: if the tier bump
lands before the field retirement, Galaxy silently moves to a 112 B header.

**B7 — WH+VC1 loses its compass-only admission exception entirely (K8).**
Not a re-key — the exception has nowhere to move to.

kernel §3.3/§6 says to retain the WH+VC1 runtime bit-test admission for code size. But it exists at
exactly **one** site, inside the *legacy* admit function:

```cpp
// fabric_erisc_router.cpp:730 — legacy can_forward_packet_completely
#if defined(ARCH_WORMHOLE) && defined(FABRIC_2D_VC1_ACTIVE)
    ... runtime E/W/N/S bit tests ...
#else
    ... 16-arm switch ...
#endif
```

`admit_express_dispatch` has **no** WH variant — all 16 arms go through `admit_express_combo<KEY>`.
That is harmless today because express is BH-only (`num_z_ports == 0` on WH), so WH never reaches the
indexed path. **After unification it does**, and Phase 4 deletes the legacy admit that holds the
exception.

Two halves, with different answers:

- **Z arms on WH — already handled, no define needed.** `express_arm_is_realizable<DOWNSTREAM_EDM_SIZE,
  DIRECTION>()` (`fabric_erisc_router.cpp:675`) compares the direction's compact index against the
  router's actual downstream array size. A WH router has no Z downstream, so the Z arm's
  `if constexpr` collapses to `ASSERT(false); ok = false;` at compile time. Nothing to gate.
- **The code-size exception — an open decision.** See §9 Q6: add a WH+VC1 bit-test variant of
  `admit_express_dispatch`, or accept the dense key on WH and measure the ERISC size cost.

Either way, this is not a Phase 1 re-key. Phase 1 keeps the static assertion that no unrealizable arm
is instantiated; the dispatch-shape decision lands with Phase 3/4 when WH first runs the indexed path.

**B8 — `intra_mesh_direction_table` is still the only source for `get_next_hop_router_direction`
on same-mesh destinations** (`tt_fabric_api.h:35-44`), which the single-hop helper and UDM depend
on. codec §2.11 calls it removable *after* cutover. Keep it through the whole project; remove it as
a separate cleanup (Phase 6) so that a single-hop regression cannot be confused with a codec
regression.

---

### 3.3 The route-encoding test uses a legacy primitive as its oracle

Found while auditing the API surface for step 2.5.

**C1.** `tests/.../kernels/test_fabric_set_unicast_route.cpp:57,74,78` builds a **reference** encoding
with `fabric_set_route` and diffs the real API against it. Phase 4.1 deletes `fabric_set_route`, so the
test stops compiling. Re-base its reference onto the indexed expectation, or retire it in favour of
the step-2.4 coverage — decided at 2.5, not discovered at 4.1.

This is a build break, so it cannot be missed. It is recorded only so the disposition is chosen
deliberately rather than under time pressure at 4.1.

---

## 4. Design decisions

### 4.1 D1 — take G1 off `express_routing_enabled`, and change nothing else

The only *requirement* is that **G1 stops reading `express_routing_enabled`** — it becomes
unconditional for 2D. G2-G5 keep calling it, unchanged, because for them it still means the right
thing: this mesh's route generation uses chords.

That needs **no new predicate and no new define.** `FABRIC_EXPRESS_ENABLED` survives, keyed on the
same `express_routing_enabled(mesh_id)` it reads today, for the one device-side site that is genuinely
about express: Z-port capacity in `routing_plane_connection_manager.hpp:21`. What changes is only
*who consumes it*:

| emission site | after |
|---|---|
| `fabric_context.cpp` → `get_express_kernel_defines` (worker kernels) | **keeps emitting `FABRIC_EXPRESS_ENABLED`** on the unchanged condition; the shape defines beside it become unconditional for 2D |
| `compute_mesh_router_builder.cpp:906-913` (router kernel) | **deleted** — the router does not include `routing_plane_connection_manager.hpp`, so its only consumers were the codec sites in `fabric_erisc_router_ct_args.hpp`, which move to `#if defined(FABRIC_2D)` |

Note the name gets *more* accurate, not less. Today it means "express topology **and** indexed
codec"; removing the codec consumers leaves it meaning exactly what it says and what its host
predicate is called.

Two earlier drafts of this decision were wrong and are recorded so they are not revived: four new
predicates (`mesh_has_z_ports`, `express_resource_shape_enabled`,
`fabric_optimizations_restricted`), which widened the diff into the VC-shape and optimization areas
this project must leave alone; and a `FABRIC_2D_HAS_Z_PORTS` rename, which invented a second spelling
for a fact whose derivation, value, and site are all unchanged.

What replaces it: a **test**, not a refactor. Risk R4 (accidentally re-enabling speedy or trim under
express, which deadlocks) is real but is guarded more cheaply by asserting the invariant than by
renaming the predicate. See guide step 1.2.

### 4.2 D2 — indexed ABI is unconditional for 2D, deployment-wide

Per codec §4.5.1 there is **no** mixed interpretation. Every 2D producer and consumer is indexed
after cutover. There is no runtime or per-mesh compatibility mode, no dual decode. A partially
migrated tree is a broken tree, which is why the cutover is one atomic step (Phase 3) rather than a
gradual migration.

Consequence: the flag day is real. Everything before Phase 3 is preparation that keeps both paths
green; Phase 3 flips; Phase 4 deletes.

### 4.3 D3 — Z stays in the ABI on every arch

Action bit 4 (`ACTION_Z`) is defined for all 2D builds. On a build with no Z sender:

- the host generator never emits `Y2_Z` (no Z edges exist to route over), so no packet ever carries
  it;
- `fwd_dirs<MY_DIR>()` still lists Z in the dense key, and the corresponding
  `express_arm_is_realizable<>` check is already `false`, so `admit_express_combo` asserts and fails
  closed (`fabric_erisc_router.cpp:1012-1050`). That is the correct behaviour and needs no change.

This is what makes the ABI uniform rather than arch-conditional. Do **not** compile Z out.

### 4.4 D4 — retire the legacy header fields to pay for the wider route buffer

This is the single most consequential sizing decision, so it is spelled out.

```text
today, 2D header base = 61 B
        = 44 B common
        +  4 B routing_fields   (hop_index | branch_east_offset | branch_west_offset)
        +  4 B dst_start_node_id
        +  8 B mcast_params_64
        +  1 B is_mcast_active

tiers:  61 + 19 =  80 B   (disabled: 8x4 Mesh perf regression)
        61 + 35 =  96 B   ← Galaxy today
        61 + 51 = 112 B
        61 + 67 = 128 B

Galaxy [32,4] indexed needs Y + X = 36 route bytes.
  36 > 35  →  next tier is 51  →  112 B header.   ✗  regression

after retiring routing_fields (4 B) and is_mcast_active (1 B):
base = 56 B
  56 + 40 =  96 B   ← a 40-byte route buffer fits the SAME 96 B header
```

So a 40-byte tier at 96 B covers, with room to spare:

| Shape | Y+X | Fits 96 B? |
|---|---|---|
| `[32,4]` Galaxy | 36 | ✓ |
| `[16,8]`, `[8,16]` | 24 | ✓ |
| `[8,8]` | 16 | ✓ |
| `[16,4]` | 20 | ✓ |
| `[1,16]` | 17 | ✓ |
| `[64,4]` (future) | 68 | needs 128 B tier (56+68=124) |

**Ordering requirement:** the field retirement (Phase 4) must land *before or with* the tier change,
or Galaxy takes a 112 B header for one phase. Prefer: do the tier bump as part of Phase 4, not
Phase 3. During Phase 3, accept 112 B on Galaxy for the duration of one phase, and measure it — that
measurement is itself the evidence for D4.

### 4.5 D5 — generalize the axis-topology source, don't special-case

Rather than adding `if (line) ... else if (torus) ... else if (express) ...` at each tree call site,
introduce one accessor that always answers:

```text
ControlPlane::axis_topology(mesh_id, axis) -> const AxisRouteTopology*
    express chords on this axis  → express ring decomposition   (existing)
    axis wraps                   → ordinary ring                (existing, needs B3 fix)
    otherwise                    → line                         (new, B2 fix)
```

`ExpressRingTopology` is already the right shape for all three — the tree builder uses only
`axis_len`, `axis_dim`, and `next_row(src, dst)` (`mcast_reverse_tree.cpp:29-93`). Rename it
`AxisRouteTopology` (or keep the name and note it) rather than introducing a parallel abstraction;
codec §11.1 and the memory-recorded constraint both say integrate, don't fork.

### 4.6 D6 — the arborescence gate becomes a hard requirement, not a warning

`control_plane.cpp:1994-2003` currently *warns* and zeroes the tree region when the gate fails, on
the theory that only multicast is lost. After unification, multicast has no fallback encoder at all
(codec §5.3, §5.7). A silent warning becomes a silently broken workload.

Change to: `TT_FATAL` for any mesh that will run 2D multicast. If a mesh legitimately cannot support
the gate, that must be a loud configuration error at fabric init, not a runtime hang.

---

## 5. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Non-express 2D multicast breaks on LINE-axis meshes (B2) | **High** — silent wrong results | B2 fix lands in Phase 1, with a host unit test per in-tree shape *before* Phase 3 |
| R2 | Galaxy header grows 96→112 B, costing bandwidth | Medium | D4; measure at Phase 3, fix in Phase 4; treat >2% BW loss as a Phase-4 blocker |
| R3 | Flag day (D2) means no bisect granularity across Phase 3 | Medium | Phase 3 is one commit; Phases 1-2 and 4-6 are individually bisectable. Keep Phase 3 minimal — *only* flip the gates, delete nothing |
| R4 | Unification accidentally re-enables speedy/trim under express (G5 leak) | **High** — deadlock | Guide step 1.2: a test asserting the *built router config* still has speedy/trim off and counter credits on for express meshes. It must survive Phase 3 unchanged. (The four-predicate rename that previously covered this was dropped — see §4.1.) |
| R5 | `X > 4` shapes were never exercised through the indexed encoder | Medium | B1 fix plus a host test sweeping every in-tree descriptor shape through `pack_indexed_route_vectors` + the arborescence gate |
| R6 | `recompute_path` deletion loses behaviour the landing encoder doesn't cover (B5) | Medium | Phase 2 writes a behavioural equivalence note for each `recompute_path` trigger before Phase 4 deletes it |
| R7 | WH+VC1 loses its compass-only exception or grows a Z arm (B7) | Medium | Re-key K8 explicitly in Phase 1; add a static_assert that no Z arm is instantiated when `num_z_ports == 0` |
| R8 | UDM header sizing recomputed independently of the 2D tier | Low | `get_udm_header_size` derives from `get_2d_header_size`; verify in Phase 4 |

---

## 6. Phasing

> **Sequencing change (owner call, 2026-08-21): all testing is deferred to a final Phase 7.**
> Implementation lands first. Each phase's exit gate below keeps only its *non-test* conditions.
>
> The tradeoff, stated once: Phase 3's gate was "full regression green on every 2D platform," which
> is what made a flag day survivable. Without tests it degrades to "it builds," and regressions from
> Phases 1-6 surface together at the end rather than at the phase that caused them. Per-phase log
> entries in the status doc are the bisect trail instead.

Each phase is independently landable and leaves the tree green. Phase 3 is the only flag day.

```text
Phase 0  Fix the live defect                       small, standalone, no dependencies
Phase 1  Split the gates + generalize topology     both paths still work
Phase 2  Close the indexed-path gaps               both paths still work
Phase 3  FLIP: indexed unconditional               flag day; delete nothing
Phase 4  Delete the legacy 2D codec                reclaim the 5 header bytes; tier bump
Phase 5  Kernel simplification                     dispatch/CT cleanup
Phase 6  L1 cleanup                                intra_mesh_direction_table removal
```

### Phase 0 — fix `fabric_set_single_hop_unicast_route` (B4)

Standalone bug fix. Route the legacy single-hop helpers to
`fabric_set_indexed_single_hop_*` under `FABRIC_EXPRESS_ENABLED`, exactly as `fabric_set_unicast_route`
already does. Ship this on its own — it is a correctness fix for shipping Galaxy workloads and
should not wait behind a refactor.

**Exit gate:** deepseek `all_gather` / `all_reduce` / `broadcast` pass on an express Galaxy config.

### Phase 1 — prepare: Z-port define, topology source, axis bound

- D1: audit and record which `FABRIC_EXPRESS_ENABLED` sites are codec vs genuinely express (no code
  change). G2-G5 untouched. G1 consumers keep reading `express_routing_enabled` *for now*.
- B1: replace `MAX_INDEXED_MESH_X = 4` with a total-bytes bound.
- B3: hoist X-ring derivation out of the express-only branch.
- B2: add the line-axis topology; add `ControlPlane::axis_topology(mesh_id, axis)` (D5).
- B7a: assert the Z-arm invariant (already true structurally). The WH admission *shape* (B7b) is
  Q6, decided at Gate 3 — not Phase 1 work.
- D6: promote the arborescence gate from warning to fatal — **but only for meshes that already take
  the indexed path**, so this phase stays behaviour-preserving.

**Exit gate:** no behaviour change anywhere. Full 1D + 2D + express regression green. New host unit
tests: every in-tree descriptor shape packs vectors and passes the arborescence gate on both axes.

### Phase 2 — close the indexed-path gaps

- P3/P6/P7/P8: make the indexed variants complete and reachable.
- B5: audit each `recompute_path` trigger against
  `fabric_set_indexed_intermesh_landing_route`; write the equivalence note; fill any gap.
- Add the missing device-side coverage: intermesh landing (intermediate and destination), multicast
  on a LINE-axis mesh, `X > 4` unicast.
- **Unify the 2D route API surface (guide 2.5)** so no consumer can reach a legacy encoder. Audit the
  client-facing layers first (§2.2) — they need no signature change, which is what makes this small. The
  ~200 `fabric_set_unicast_route` / `fabric_set_mcast_route` call sites need **no** change — they go
  through names that already fork internally, and the fork is what Phase 3 removes. The real work is
  narrow: `fabric_set_route`'s two consumers (P4), the `called_from_router` arm (B5), and one orphaned
  indexed variant.

**Exit gate:** every capability the legacy 2D path provides has a green indexed-path test, on both
express and non-express fixtures; the §3.3 test-oracle disposition is chosen.

### Phase 3 — flip (flag day)

Change **only** the four G1 gates (H1-H4) plus H5/H6 to unconditional. Do not delete the legacy
code; leave it dead. This keeps the diff small and the revert trivial.

- `FABRIC_EXPRESS_ENABLED` is still emitted, but for *every* 2D mesh. (Rename to
  the guard to `#if defined(FABRIC_2D)` and stops emitting the define — see §9 Q2.)

**Exit gate:** full regression on every 2D platform: WH Galaxy, BH Galaxy, BH LB, T3K-2D, dual/quad
galaxy. Measure Galaxy bandwidth and record the 112 B-header cost (R2).

### Phase 4 — delete the legacy 2D codec, reclaim the header

- Delete: `compressed_route_2d_t` fill path, `intra_mesh_routing_path_t<2,true>` from the union,
  `encode_2d_unicast`, `routing_encoding` 2D helpers, `fabric_set_route`, legacy
  `fabric_set_mcast_route` body, legacy single-hop helper, `fabric_edge_node_router.hpp` entirely,
  `RoutingFieldsConstants::Mesh` (or reduce it to whatever 1D still needs).
- Retire `routing_fields` (4 B) and `is_mcast_active` (1 B) from `HybridMeshPacketHeaderT`.
- Add the 40-byte / 96 B tier; update the four `static_assert`s at
  `fabric_edm_packet_header.hpp:1295-1298`.
- Update profiler/debug consumers that decode `hop_index` / `branch_*`.

⚠ Precondition: the §3.3 test-oracle disposition must be settled before this phase deletes
`fabric_set_route`.

**Exit gate:** Galaxy back to a 96 B header with bandwidth at or above the Phase-0 baseline. No
symbol named `hop_index`, `branch_east_offset`, `branch_west_offset`, or `turn_point` remains in a
2D path.

### Phase 5 — kernel simplification

- Delete K3-K7: `is_spine_direction`, `TURN_STATUS_ARRAY_SIZE`,
  `get_sender_channel_turn_statuses`, `sender_channels_turn_status`, `IS_TURN`, the 2D body of
  `update_packet_header_before_eth_send`, `UPDATE_PKT_HDR_ON_RX_CH` and its RX branches,
  the `express_enabled` constexpr itself.
- Rename `express_*` kernel symbols to `indexed_*` (`admit_express_combo` →
  `admit_indexed_combo`, `express_local_y` → `local_y`, etc.).

`port_direction_table` is *not* part of this phase — kernel §3.9 notes it is independent of the ABI
cutover, and it belongs to the deferred dead-code discussion (§1).

**Exit gate:** ERISC binary size at or below Phase-3, and no `express` identifier remains in the
kernel except where it genuinely means Z chords.

### Phase 6 — L1 cleanup

- Remove `intra_mesh_direction_table` (96 B); repoint `get_next_hop_router_direction` at the
  indexed vectors' first-hop peek (codec §2.11).
- Reclaim the freed 96 B; re-derive `routing_l1_info_t` offsets. This is what buys headroom for
  `[64,4]` (codec §6.2: 2608 B after reclamation vs 2704 B without).

**Exit gate:** `sizeof(routing_l1_info_t)` documented and asserted; `[64,4]` hybrid layout fits or
its exact shortfall is recorded.

---

## 7. Verification strategy

### 7.1 Host unit tests (cheap, run every phase)

| Test | Asserts |
|---|---|
| shape sweep | every in-tree descriptor shape: `pack_indexed_route_vectors` succeeds, region fits, `Y+X ≤ tier` |
| arborescence sweep | every shape × every root × both axes passes the gate |
| axis-topology coverage | `axis_topology(mesh, axis)` is non-null for every 2D mesh × axis (catches B2/B3 regressions) |
| gate separation | for an express mesh: G5 predicates are `false`; for a non-express mesh: G2 predicates are `false` (catches R4) |
| encoder equivalence | `encode_indexed_mcast_maps` == the codec §5.6 golden vector path-trace, for every root × every extent combination |
| exit-chip invariant | every intermesh carrier decodes to exactly `LOCAL_DELIVER` at its exit (codec §4.5) |

### 7.2 Device tests

| Test | Phase |
|---|---|
| 2D unicast all-pairs, express + non-express | 2, 3 |
| 2D multicast every extent, LINE and TORUS Y | 2, 3 |
| intermesh unicast + mcast, intermediate and destination landing | 2, 3 |
| single-hop helpers (deepseek kernels) | 0, 3 |
| UDM 2D | 2, 3 |
| `X > 4` shapes (`[8,8]`, `[8,16]`) | 3 |
| bandwidth: Galaxy `[32,4]` header size regression | 3, 4 |

### 7.3 Static assertions to add

- `Y + X ≤ sizeof(route_buffer)` for the built shape (already exists at `tt_fabric_api.h:162`,
  make unconditional).
- no Z arm instantiated when the build has no Z sender (B7).
- `vectors_region_bytes(Y,X) + mcast_tree_region_bytes(Y,X) ≤ slot` for the built shape.

---

## 8. Effort shape

Rough sizing, for sequencing rather than commitment:

| Phase | Files touched | Character |
|---|---|---|
| 0 | 1 | one-line fix + test |
| 1 | ~10 host | mechanical + one new topology derivation |
| 2 | ~6 device headers | fill gaps, mostly test writing |
| 3 | ~6 | small diff, large validation |
| 4 | ~15 | large deletion; header ABI change |
| 5 | ~4 kernel | large deletion |
| 6 | ~4 | L1 layout change |

The heavy items are **validation in Phase 3** and **deletion discipline in Phase 4**, not new code.
Phases 1 and 2 are where the real design work lives (B1, B2, D1, D5).

---

## 9. Open questions for the owner

1. **Multicast client contract.** Recommendation: adopt the per-direction contract that already
   exists (comparison §7), which makes the indexed encoder a drop-in with single-output roots on every
   chordless mesh and **zero client migrations**. Confirm, and confirm the guard
   `ASSERT(popcount(root_action & ETH_MASK) <= 1)` is acceptable on the shared path.
2. ~~**Rename `FABRIC_EXPRESS_ENABLED` → `FABRIC_2D_INDEXED_ABI`**~~ — **resolved: do not rename,
   delete.** The flag becomes fully redundant with `FABRIC_2D` at the flip: on the router,
   `defines["FABRIC_2D"]` (`compute_mesh_router_builder.cpp:869`) is set in the same `if` block as
   the express define 43 lines later; on the worker, `fabric.cpp:389` sets `FABRIC_2D` and the
   express defines are added inside that same `is_2D_routing_enabled()` block. Remove the express
   condition and the two are emitted under identical circumstances, so the guard becomes
   `#if defined(FABRIC_2D)` and no new name is introduced. Only the shape defines are renamed
   (`FABRIC_EXPRESS_MESH_{Y,X}_SIZE` → `FABRIC_2D_MESH_{Y,X}_SIZE`, kernel-side
   `EXPRESS_MESH_{Y,X}_SIZE` → `MESH_{Y,X}_SIZE`); `express_enabled` is deleted in Phase 5 rather
   than renamed. `FABRIC_EXPRESS_ENABLED` itself **survives under its own name** for the one genuine
   express site (Z-port capacity, §4.1) — it stops being a codec selector, not a define. H6 (§3.3 / guide 3.3) reduces to deleting the no-node
   overload's express fatal, whose premise -- a per-mesh 2D ABI -- is what the flip removes. The
   dispatch relays already resolve their own mesh for the shape defines.
3. **`derive_line_axis_topology` placement**: extend `express_ring_topology.{hpp,cpp}` (keeps all
   three derivations together, but the filename becomes wrong) or a new
   `axis_route_topology.{hpp,cpp}` that absorbs all three? Recommendation: rename the file and the
   struct in Phase 1 — one rename beats a permanent misnomer.
4. **D6 fatal-vs-warning** on the arborescence gate: any mesh you know of that should be allowed to
   ship without 2D multicast? If yes, the gate needs a per-mesh "multicast unsupported" declaration
   rather than a fatal.
5. **80 B tier**: it is disabled for an 8×4 Mesh perf regression. After Phase 4 the base is 56 B, so
   an 80 B tier would carry 24 route bytes — enough for `[8,8]`, `[16,8]`, `[8,16]`, `[16,4]`. Worth
   re-measuring, or leave disabled?
6. **WH+VC1 admission shape** (B7). Unification puts WH on the indexed path for the first time, and
   the compass-only bit-test exception kernel §3.3 wants retained lives only in the legacy admit that
   Phase 4 deletes. Add a WH+VC1 bit-test variant of `admit_express_dispatch`, or accept the dense
   16-arm key on WH and take the code-size hit? Recommendation: **measure first** — the Z arms already
   compile out on WH via `express_arm_is_realizable<>`, so the dense key may cost less than the
   contract assumed. Decide at Gate 3, when WH ERISC size is first observable.
