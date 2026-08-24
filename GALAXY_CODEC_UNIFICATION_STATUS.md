# Codec Unification — Working Status

Living tracker. Edit in place as work lands; this is the one file expected to churn.

| | |
|---|---|
| branch | `nnyamagoudar/subtorus-routing-kernel-unification` |
| base | `42204c541c3` (`agupta/subtorus-routing`) |
| last updated | 2026-08-21 |
| current phase | **Implementation complete; host tests green.** 29/29 pass (codec 23, line-axis 3, descriptor sweep 3). Remaining: device tests + platform regression |

Reference docs (stable; update those, not this one, when the *plan* changes):
`GALAXY_CODEC_UNIFICATION_COMPARISON.md` · `GALAXY_CODEC_UNIFICATION_PLAN.md` ·
`GALAXY_CODEC_UNIFICATION_IMPLEMENTATION_GUIDE.md`

**Status vocabulary:** `todo` · `wip` · `done` · `blocked` · `deferred` · `n/a`

> **Sequencing (owner call, 2026-08-21): all testing is deferred to Phase 7.** Implementation lands
> first, tests last. Every per-phase gate therefore keeps only its *non-test* conditions; the test
> conditions are collected in Phase 7 below.
>
> ⚠ Consequence worth holding onto: Gate 3 is the flag day, and its original condition was "full
> regression green on every 2D platform." Without tests that gate degrades to "it builds." Regressions
> from Phases 1-6 will surface together in Phase 7 rather than at the phase that caused them, so keep
> the per-phase log entries detailed enough to bisect against.

---

## 0. Step index — what every number means

Read this first. The rest of this file, the plan, and the implementation guide all refer to steps by
number; this table is the only place you need to look them up. **Names are the preferred way to refer
to a step in conversation** — the numbers exist for cross-referencing the guide, not for memorising.

Also used throughout: **codec §N** and **kernel §N** cite sections of
`GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md` and `GALAXY_DEVICE_ROUTER_KERNEL_CONTRACT.md`.

| # | Name | In one line | Status |
|---|---|---|---|
| 0.1 | **single-hop ABI fix** | route the single-hop helpers to the indexed encoder | `done` |
| 1.1 | **guard inventory** | classify the 9 `express_routing_enabled` sites: codec gate vs genuine express | `done` |
| 1.2 | **gate-separation test** | prove express meshes get speedy/trim off, counter credits on | `todo` (Phase 7) |
| 1.3 | **shape bound split** | one conflated `X ≤ 4` constant → `SLOT_SHAPE_*` + `MAX_INDEXED_MESH_AXIS` | `done` |
| 1.4 | **X-ring hoist** | derive ring topology outside the express-only branch | `done` (folded into line-axis topology) |
| 1.5 | **line-axis topology** | express → ring → **line** fallback per axis, + the `wraps` fix to `next_row` | `done` |
| 1.6 | **Z-arm invariant** | a WH router must not instantiate a Z dispatch arm | `done` (no code needed) |
| 1.7 | **one-feeder gate** | a missing multicast tree must fatal, not warn | `done` |
| 1.8 | **route-buffer sizing for all meshes** | size the route buffer from *every* 2D mesh, not just express ones | `done` (landed inside host gate flip) |
| 2.1 | **UDM producer audit** | L1 base agreement and Z tolerance in the mux path | `done` |
| 2.2 | **unconditional mcast inject** | make multicast source injection unconditional; assert single-output roots | `done` |
| 2.3 | **recompute_path audit** | router-side re-encode vs the indexed landing encoder | `done` |
| 2.4 | **device coverage** | the device tests the project owes | `todo` (Phase 7) |
| 2.5 | **2D API surface** | ensure every consumer reaches the indexed encoder | `done` (no signature changes needed) |
| 3.1 | **selector retirement** | stop using the express flag to choose a codec | `done` |
| 3.2 | **host gate flip** | the four host-side gates become unconditional for 2D | `done` |
| 3.3 | **no-node fatal removal** | drop the express `TT_FATAL` in the no-node overload | `done` |
| 3.4 | **device producer flip** | worker/UDM encoders always emit indexed | `done` |
| 3.5 | **define emission split** | shape defines unconditional; express define stays conditional | `done` |
| 3.6 | **kernel flip** | `if constexpr (indexed_2d)` → always true, arms still present | `done` |
| 4.1 | **legacy producer deletion** | delete `fabric_set_route` and the legacy encode paths | `done` |
| 4.2 | **edge-node router deletion** | delete `fabric_edge_node_router.hpp` | `done` |
| 4.3 | **legacy L1 table deletion** | delete `compressed_route_2d_t` and the 2D hop-program table | `done` |
| 4.4 | **header reclaim** | retire `is_mcast_active`; retier to 36/52/67 so Galaxy is 96 B again | `done` |
| 5.1 | **turn-set deletion** | delete the turn/header-mutation machinery the hop program needed | `done` |
| 5.2 | **port_direction_table** | **`deferred`** — dead-code question, out of this project. (Was briefly deleted during the stack-ceiling investigation, then **reverted** at owner request so this branch carries unification work only and a head-vs-branch stack comparison stays fair. `port_direction_table` is genuinely dead — never indexed anywhere in the tree — if anyone wants it later.) |
| 5.3 | **dispatch arm collapse** | remove the now-always-true constexpr and its dead arms | `done` |
| 5.4 | **kernel rename** | `express_*` → `indexed_*` in the router kernel | `done` |
| 6.1 | **direction-table removal** | remove `intra_mesh_direction_table` | **`will not do`** (1D needs it) |
| 6.2 | **L1 reclaim** | reclaim the 96 B that removal would free | **`dead`** (follows 6.1) |
| 6.3 | **axis validation lift** | flat `≤ 32` per axis → the real indexability predicate | `done` |

**Gates** are the exit conditions for each phase (Gate 0 … Gate 6), listed in the phase board below.
**B1–B8** are blockers, **D1–D9** decisions, **R1–R9** risks, **U1/U2** pre-existing UDM defects,
**H1–H6** host gate sites, **P1–P10** producer/API sites, **Q1–Q6** open questions — all tabled in
sections 2 and 3.

---

## 1. Phase board

Step numbers are 1:1 with the implementation guide. Do not renumber here — change the guide.

### Phase 0 — fix the live single-hop ABI mismatch *(standalone, ships alone)*

| step | what | status | notes |
|---|---|---|---|
| 0.1 **single-hop ABI fix** | ⚠ route single-hop helpers to the indexed encoder (`tt_fabric_api.h:48`) | **`done`** | code landed. Needed 2 guide corrections: hoist the shape-define block before first use; use `#if/#else` not an early return. Also added an explicit same-mesh `ASSERT` |

| gate | condition | status |
|---|---|---|
| **Gate 0** | deepseek `all_gather` / `all_reduce` / `broadcast` / `reduce_to_one_b1` pass on express Galaxy | `todo` |

### Phase 1 — prepare *(behaviour-preserving)*

| step | what | status | notes |
|---|---|---|---|
| 1.1 **guard inventory** | audit + record the express-guard inventory (9 sites: codec vs genuine express) | **`done`** | no code change. Inventory in plan §1.1; 8 codec + 1 genuine express (Z-port capacity). Client-API audit in plan §2.2 |
| 1.2 **gate-separation test** | ⚠ gate-separation test (built router config: speedy/trim off, counter credits on) | **`not testable — replaced by a diff guard`** | the R4 invariant is 5 inline `!express && ...` compositions in the builder, not pure functions; reaching them needs a ControlPlane + EDM builder. Replaced by a mechanical Gate 3 check that the Phase 3 diff touches none of those 5 lines |
| 1.3 **shape bound split** | remove the `X ≤ 4` bound → total-region bound (B1) | **`done`** | split `MAX_INDEXED_MESH_{Y,X}` into `SLOT_SHAPE_{Y,X}` (a budget) vs `MAX_INDEXED_MESH_AXIS` (a real limit, 64, fixed by the 6-bit tree descriptor). Added asserts pinning `[8,8]`/`[8,16]`/`[16,8]`/`[1,16]`/`[32,32]` |
| 1.4 **X-ring hoist** | hoist X-ring derivation out of the express-only branch (B3) | **`done` (via 1.5)** | not hoisted literally — `express_rings_`/`x_rings_` keep their exact population rules so nothing reading them changes. Instead a new `axis_topologies_[mesh][axis]` is populated for **every** mesh and **both** axes via `derive_axis_topology`, whose line fallback makes it safe |
| 1.5 **line-axis topology** | ⚠ line-axis topology + `axis_topology()` accessor (B2, D5) | **`done`** | `derive_line_axis_topology` + `derive_axis_topology`; `wraps` flag guarding `next_row`'s `step()`; `axis_topologies_` per mesh × axis; `RoutingTableGenerator::get_axis_topology` + `ControlPlane::axis_topology`. Q3 rename done (file + struct + `sources.cmake`) |
| 1.6 **Z-arm invariant** | assert the Z-arm invariant (B7a) | **`no change needed`** | invariant is already enforced structurally by the dispatch arms' `if constexpr`; any assert would be tautological, and `num_z_ports` is not visible to the kernel. Comment recorded at `express_arm_is_realizable`. WH admission *shape* remains Q6 |
| 1.7 **one-feeder gate** | one-feeder-per-row gate: warning → `TT_FATAL` (D6) | **`done`** | `control_plane.cpp`: `log_warning` → `TT_FATAL`, propagating the builder's row/both-feeders message. Also switched to `axis_topology()` + a fatal on a null axis topology |
| 1.8 **route-buffer sizing for all meshes** | un-skip non-express meshes in route-buffer sizing (H5) | **`done` (in 3.2)** | `get_max_2d_indexed_route_bytes_from_topology()` now sums Y+X for **every** 2D mesh unconditionally. The `[32,4]` tier bump it caused (96→112 B) was given back by 4.4 |

| gate | condition | status |
|---|---|---|
| **Gate 1** | no behaviour change anywhere; full 1D + 2D + express regression green | `todo` |
| | R4 diff guard recorded (the 5 lines Phase 3 must not touch — see §6 notes) | **`done`** |
| | tier bumps from 1.8 recorded per platform | `todo` |

### Phase 2 — close the indexed-path gaps *(behaviour-preserving)*

| step | what | status | notes |
|---|---|---|---|
| 2.1 **UDM producer audit** | verify indexed UDM producers (P7, P8) — L1 base agreement, Z-tolerant callers | **`done`** | ⚠ **not "no edit" — found two live defects on the express+UDM path.** (1) wrong L1 base: hardcoded `MEM_TENSIX_ROUTING_TABLE_BASE` → `ROUTING_TABLE_BASE`. (2) `direction_to_mux_index_map` is `[5][5]` but only 4×4 initialised, so a Z initial direction silently forwarded to mux 0 — now fail-loud. See U1/U2 |
| 2.2 **unconditional mcast inject** | make `fabric_multicast_source_inject_*` unconditional (P6); add the single-output-root assert | **`assert already present; unconditional half → 3.4`** | Q1's guard already exists verbatim (`ASSERT((root_outputs & (root_outputs-1)) == 0)`). Making the helper unconditional **cannot** happen in Phase 2: it references `FABRIC_EXPRESS_MESH_{Y,X}_SIZE`, whose fallback is itself guarded on `FABRIC_EXPRESS_ENABLED`, so a non-express build has no shape macros at all |
| 2.3 **recompute_path audit** | ⚠ audit `recompute_path` vs the indexed landing encoder (B5, R6) | **`done`** | equivalence table in guide 2.3: 2 triggers × 4 outcomes, all resolved. **4.2 is licensed.** Two divergences are improvements (no Z/NOOP overload; one-shot next-exit install). Two residual risks recorded |
| 2.4 **device coverage** | add the missing device coverage | **`deferred to Phase 7`** | target sets follow Q1's per-direction contract; also picks up the single-hop coverage skipped at Gate 0 |
| 2.5 **2D API surface** | unify the 2D route API surface; every consumer reaches the indexed encoder | **`done`** | API audit in plan §2.2 (no signature changes needed). Orphaned `fabric_set_indexed_single_hop_unicast_route` deleted — 0 callers, redundant after 0.1. All 5 remaining indexed variants reachable. `fabric_set_route`'s 2 consumers dispositioned: `cq_relay` → 3.4; test oracle → see C1 |

| gate | condition | status |
|---|---|---|
| | 2.3 equivalence table written, every row resolved | **`done`** |
| | 2.5: `fabric_set_route`'s 2 consumers each have a written disposition | **`done`** |
| | 2.5: `mesh/api.h`, `linear/api.h`, `udm/*` confirmed to reach only forking encoders | **`done`** |
| | 2.5: no orphaned indexed variants remain | **`done`** |

### Phase 3 — flip *(flag day; one commit; delete nothing)*

| step | what | status | notes |
|---|---|---|---|
| 3.1 **selector retirement** | retire the selector as a codec gate | **`done`** | 9 codec guards → `#if defined(FABRIC_2D)`; shape macros renamed `FABRIC_EXPRESS_MESH_*` → `FABRIC_2D_MESH_*` and kernel `EXPRESS_MESH_*` → `MESH_*`; `express_enabled` → `indexed_2d`. `FABRIC_EXPRESS_ENABLED` survives at **2** genuine sites: Z-port capacity and the UDM bar |
| 3.2 **host gate flip** | flip the four host gates (H1-H4) | **`done`** | `get_express_kernel_defines` → `get_2d_kernel_defines`, emission split (shape unconditional for 2D; express flag conditional). L1 embed always indexed — legacy `intra_mesh_routing_path_t<2,true>` write **deleted**. CT args on `is_2D_routing_enabled()`. Router-side express emission deleted. **1.8 landed here** |
| 3.3 **no-node fatal removal** | delete the no-node overload's express fatal (H6) | **`done`** | fatal removed; overload returns `API_TYPE_*` + `FABRIC_2D` |
| 3.4 **device producer flip** | flip the device producers | **`done`** | `tt_fabric_api.h` (unicast, mcast, single-hop, shape fallback, static_assert), `udm/*` ×2, `mesh/api.h` source-inject (**2.2's deferred half**), `cq_relay.hpp` legacy `fabric_set_route` arm deleted |
| 3.5 **define emission split** | ⚠ leave the connection-manager capacity alone; split the emission | **`done`** | `routing_plane_connection_manager.hpp` untouched, as designed. Emission split done in 3.2 |
| 3.6 **kernel flip** | flip the kernel (`if constexpr` → always-true; keep the dead arms) | **`done`** | ct_args block on `FABRIC_2D`; `express_enabled` → `indexed_2d` at all 4 sites. Legacy `else` arms deliberately left in place for Phase 5 |

| gate | condition | status |
|---|---|---|
| **Gate 3** | ⚠ regression conditions moved to Phase 7; this gate is now build + diff-guard only | `n/a` |
| | WH Galaxy `[8,4]`, `[32,4]` | `deferred to Phase 7` |
| | BH Galaxy `[8,4]`, `[32,4]`, `[4,4]` express | `deferred to Phase 7` |
| | BH LB `[2,4]` | `deferred to Phase 7` |
| | dual/quad galaxy `[8,8]`, `[8,16]`, `[16,8]` | `deferred to Phase 7` |
| | N300 2x2 / p150_x4 `[2,2]` | `deferred to Phase 7` |
| | **1D regression untouched** (any movement = a define leaked) | `deferred to Phase 7` |
| | Galaxy `[32,4]` header size + bandwidth **measured and recorded** (expect 112 B, R2) | `todo` |
| | ⚠ R4 diff guard: `git diff` for Phase 3 touches **none** of `erisc_datamover_builder.cpp:796,1082,1090,1129` or `fabric_builder_context.cpp:53` | **`PASS`** — only removed express line is the G1 CT-args gate; `fabric_builder_context.cpp` has zero changes |

### Phase 4 — delete the legacy 2D codec, reclaim the header

| step | what | status | notes |
|---|---|---|---|
| 4.1 **legacy producer deletion** | delete the legacy 2D producers | **`done`** | `single_hop_route_cmd_by_direction`, `fabric_set_route<mcast>`, legacy mcast spine/branch body, legacy single-hop arm, `called_from_router` arm (now a `static_assert(!called_from_router)`). ~12 KB removed from `tt_fabric_api.h` |
| 4.2 **edge-node router deletion** | delete `fabric_edge_node_router.hpp` | **`done`** | file removed (`git rm`) + include dropped, after 5.3 orphaned it. `recompute_path` and `get_cmd_with_mesh_boundary_adjustment` both at 0 external refs first |
| 4.3 **legacy L1 table deletion** | delete the legacy L1 2D table | **`done`** | `compressed_route_2d_t`, `intra_mesh_routing_path_t<2,true>` (+ decode + explicit instantiation), `encode_2d_unicast`, `routing_path_table_2d` union member, `MAX_CHIPS_LOWLAT_2D`, `SINGLE_ROUTE_SIZE_2D`, host 2D route generation. Template collapsed to 1D-only. `sizeof(routing_l1_info_t)` still **2576**. ⚠ `RoutingFieldsConstants::Mesh` **kept** — sole consumer is the profiler, out of scope |
| 4.4 **header reclaim** | retire `is_mcast_active`; retier 36/52/67 | **`done`** | **revised from the plan**: retiring `is_mcast_active` alone (zero readers) gives base 61→**60**, and 60+36=96 — so Galaxy is back on a **96 B** header without touching `routing_fields`, which the profiler reads. Tiers 35/51/67 → **36/52/67**; default 35→36. All 10 in-tree shapes verified on their pre-flip tier. ⚠ first build caught a 68-tier overreach — see log |

| gate | condition | status |
|---|---|---|
| **Gate 4** | Galaxy `[32,4]` back to a **96 B** header | **`met (by construction)`** — 60 + 36; needs the device measurement to confirm |
| | bandwidth **at or above the Phase-0 baseline** (not merely above Phase 3) | `todo` |
| | zero occurrences of `hop_index` / `branch_*_offset` / `turn_point` / `is_mcast_active` / `compressed_route_2d_t` in any 2D path | **`met, with one documented exception`** — `turn_point`, `is_mcast_active`, `compressed_route_2d_t` are gone tree-wide; the remaining `hop_index` hits are all **1D** (`LowLatencyFields` lambda params). The `routing_fields` struct declaration survives solely for the out-of-scope profiler |
| | R8: `get_udm_header_size` still derives from `get_2d_header_size` | `todo` |

### Phase 5 — kernel simplification

| step | what | status | notes |
|---|---|---|---|
| 5.1 **turn-set deletion** | delete the turn/header-mutation set (kernel §3.9) | **`done`** | `update_packet_header_before_eth_send`, `is_spine_direction`, `TURN_STATUS_ARRAY_SIZE`, `get_sender_channel_turn_statuses`, `sender_channels_turn_status` all gone. `UPDATE_PKT_HDR_ON_RX_CH` removed from **both** ends. ⚠ found a live mutation: the 2D `update_packet_header_for_next_hop` was still doing `routing_fields.value + 1` on the indexed path — now a no-op |
| 5.2 **port_direction_table** | `port_direction_table` | **`deferred`** — dead-code question, out of this project. (Was briefly deleted during the stack-ceiling investigation, then **reverted** at owner request so this branch carries unification work only and a head-vs-branch stack comparison stays fair. `port_direction_table` is genuinely dead — never indexed anywhere in the tree — if anyone wants it later.) |
| 5.3 **dispatch arm collapse** | collapse the `if constexpr (indexed_2d)` arms; delete the constexpr | **`done`** | all 3 arms collapsed, `indexed_2d` removed, legacy 2D admit (16-arm switch + WH/VC1 branch) and legacy 2D `receiver_forward_packet` (446 lines) deleted. Router 4151 → 3540 lines |
| 5.4 **kernel rename** | rename `express_*` → `indexed_*` in the kernel | **`done`** | 11 identifiers renamed (`admit/forward_express_combo`→`_indexed_combo`, `express_arm_is_realizable`→`dispatch_arm_is_realizable`, `express_local_{y,x}`→`local_{y,x}`, `express_egress`→`intermesh_egress`, etc). Kernel now has **zero** `express` occurrences |

| gate | condition | status |
|---|---|---|
| **Gate 5** | ERISC binary size at or below Phase 3 | `todo` — needs a device build |
| | no `express` identifier in the kernel that does not mean Z chords | **`met`** — `fabric_erisc_router.cpp` has **zero** occurrences of `express` |
| | 1D untouched | **`met structurally`** — no 1D-specific source appears in the diff; still needs the 1D regression to confirm behaviourally |

### Phase 6 — L1 cleanup

| step | what | status | notes |
|---|---|---|---|
| 6.1 **direction-table removal** | remove `intra_mesh_direction_table` | **`will not do`** | ⚠ the guide's premise is wrong. The table is **load-bearing for 1D**: `TEST_F(Fabric1DFixture, TestGetNextHopRouterDirection1D)` exercises the API on a 1D build, where no indexed vectors exist. It is redundant on 2D only, so the field and its L1 cost stay regardless |
| 6.2 **L1 reclaim** | reclaim the freed 96 B | **`dead`** | follows from 6.1 — nothing is freed, so there are no offsets to re-derive. `sizeof(routing_l1_info_t)` stays **2576** |
| 6.3 **axis validation lift** | lift the 32-per-axis validation | **`done`** | flat `<= 32` → `IndexedMeshRoutingFields::shape_is_indexable()`, the same predicate the packer enforces, so a shape that validates can never fail to pack. `[64,4]` now passes this bound *exactly* (1028 B = the whole slot) |

| gate | condition | status |
|---|---|---|
| **Gate 6** | `sizeof(routing_l1_info_t)` asserted and documented | **`met`** — **2576 B**, unchanged, `static_assert` intact; 6.2's reclaim is dead (see 6.1) |
| | `[64,4]` either fits or its exact shortfall is recorded | **`met`** — it does **not** fit. Indexable exactly (1028 B = the whole slot); fails the header bound by **one byte** (Y+X = 68 vs 67). Issue #32237 |

### Phase 7 — Testing *(all of it, deferred here by owner call)*

| test | origin | status |
|---|---|---|
| host: shape sweep — every 2D descriptor packs vectors, fits the region, `Y+X` fits a tier | Gate 1 | **`written`** — `test_indexed_route_codec.cpp` |
| host: axis-topology coverage — every mesh × axis derives and routes over declared edges only | Gate 1 | **`done, green`** — `test_axis_topology_sweep.cpp`, all 158 descriptors |
| host: arborescence sweep — every shape × every root × both axes | Gate 1 | **`already covered`** — `test_mcast_reverse_tree.cpp` runs `run_mcast_arborescence_gate` over every root on both axes (express_links_{8,16,24,32}x4). My table was stale |
| host: encoder equivalence — `encode_indexed_mcast_maps` vs the golden path-trace | §4 | **`already covered`** — `test_mcast_reverse_tree.cpp` check_encode() diffs against a forward-route golden. My table was stale |
| host: exit-chip invariant — every intermesh carrier decodes to exactly `LOCAL_DELIVER` at its exit | §4 | `todo` |
| host: tree-region validity — a zeroed region is rejected, not silently encoded | §4 | **`partly written`** — the zeroed *route buffer* decode + validity case is covered; the zeroed *tree region* host check still needs a mesh graph |
| host: **line-axis `next_row`** — a line never steps over the absent wrap edge (guards the 1.5 fix) | 1.5 | **`written`** — `test_express_ring_topology.cpp`, 3 cases incl. the ring counterpart |
| device: single-hop N/S/E/W + Z | Gate 0 | `todo` |
| device: 2D unicast all-pairs, express + non-express | 2.4 | `todo` |
| device: 2D multicast every extent, LINE **and** TORUS Y | 2.4 | `todo` |
| device: intermesh unicast, destination-mesh landing | 2.4 | `todo` |
| device: intermesh unicast, **intermediate**-mesh landing (needs ≥3-mesh fixture) | 2.4 | `todo` |
| device: intermesh multicast carrier + destination landing rebuild | 2.4 | `todo` |
| device: `X > 4` unicast (`[8,8]`, `[8,16]`) | 2.4 | **`cannot be tested on this cluster`** — galaxy cabling is 8x4 per galaxy extending along one axis (16x4, 32x4), so X is always 4. `[8,8]` and `[16,8]` descriptors exist but cannot be realised. `X > 4` is therefore covered **only** by the host unit tests (`IndexedRouteCodec` shape admissibility + pack/decode round-trip over `[8,8]`, `[8,16]`, `[16,8]`, `[1,16]`). Hardware validation of the shape bound split needs different cabling |
| device: UDM 2D | 2.4 | `todo` |
| device: bandwidth — Galaxy `[32,4]` header regression | Gate 3/4 | `todo` |
| platform regression: WH Galaxy, BH Galaxy, BH LB, dual/quad galaxy, N300 2x2 | Gate 3 | `todo` |
| platform regression: **1D untouched** | Gate 3/5 | `todo` |
| R4 diff guard (mechanical, not a test — can run any time) | 1.2 | `todo` |

**Not testable without a refactor** (recorded so it is not re-attempted): the R4
speedy/trim/credits invariant — see the §6 note.

---

## 2. Decisions

Plan §9. A `blocked` step above cannot start until its decision is `decided`.

| # | question | recommendation | state | blocks |
|---|---|---|---|---|
| Q1 | ~~Multicast client contract~~ | **adopt the per-direction contract + add the single-output-root assert.** Indexed multicast is a drop-in on chordless meshes; zero client migrations | `decided` 2026-08-20 | — |
| Q2 | ~~Rename `FABRIC_EXPRESS_ENABLED`~~ | **delete, don't rename** — redundant with `FABRIC_2D` after the flip | `decided` 2026-08-20 | — |
| Q3 | ~~`derive_line_axis_topology` placement~~ | **rename file *and* struct**: `express_ring_topology.{hpp,cpp}` → `axis_route_topology.{hpp,cpp}`, `ExpressRingTopology` → `AxisRouteTopology` | `decided` 2026-08-20 | — |
| Q4 | ~~Arborescence gate: fatal or opt-out?~~ | **hard `TT_FATAL`.** Structurally a no-op on chordless meshes (one feeder per row always), and express multicast works today so the check already passes there — a fatal cannot break anything functioning | `decided` 2026-08-20 | — |
| Q5 | **80 B tier** — superseded. The base is **60 B**, not the 56 B this question assumed (`routing_fields` stayed for the profiler), so an 80 B header carries **20** route bytes, covering `[8,8]` (16) and `[1,16]` (17) but **not** `[8,16]`/`[16,8]` (24) or `[16,4]` (20 — exactly at the limit). The tier is still commented out for the 8x4 benchmark instability noted in `fabric_context.hpp`. Re-measure before enabling | leave disabled unless measured | `deferred to Gate 4` | — |
| Q6 | ~~WH+VC1 admission shape~~ | **not a blocker — reframed.** The indexed path has exactly one admission implementation and never had a WH variant, so collapsing the legacy arms removes nothing from it. Kernel §3.3's compass-only recommendation was prescribing an indexed design that was never built. Whether to *add* a WH variant is a Phase 7 perf question | `decided` 2026-08-21 | — |

**Settled** (recorded so they are not relitigated):

| | decision | where |
|---|---|---|
| 1D scope | **out** — 1D keeps its own codec | plan §1.3 |
| dead code | **deferred** to a separate discussion | plan §1.2 |
| VC-shape axes, UDM-as-variant | **out of scope** | plan §1.2 |
| four-predicate gate split | **dropped** — no new predicate *and* no new define; one test instead | plan §4.1 |

---

## 3. Blockers and defects

Plan §3, comparison §9.

| # | item | fixed by | status |
|---|---|---|---|
| B1 | `MAX_INDEXED_MESH_X = 4` excludes `[8,8]`, `[8,16]`, `[16,8]`, `[1,16]` | 1.3 | **`closed`** — the conflated bound was split into `SLOT_SHAPE_{Y,X}` (physical slot) vs `MAX_INDEXED_MESH_AXIS` (addressable range); all four shapes now pass, pinned by `static_assert` |
| B2 | no reverse-tree source on a LINE axis → **silent multicast no-op** | 1.5 | **`closed`** — `derive_axis_topology()` now yields express → ring → **line** for every mesh × axis, and 1.7's `log_warning` became `TT_FATAL` so a missing tree cannot degrade silently |
| B3 | `x_rings_` derived only for express meshes | 1.4 | **`closed`** — folded into 1.5; `axis_topologies_[mesh][axis]` is populated for every 2D mesh, degenerate axes (< 2) left null by design |
| B4 | `fabric_set_single_hop_unicast_route*` writes the legacy hop byte with no express gate — **live defect** | 0.1 | **`fixed`** — pending Gate 0 |
| B5 | `called_from_router` 2D encode still legacy under express (dead, but unexercised) | 2.3 | **`closed`** — 4.1 deleted the arm and replaced it with `static_assert(!called_from_router)`, so the router-side re-encode is now a compile error rather than dead code |
| B6 | Galaxy `[32,4]` needs 36 route bytes, one past the 35 B tier | 4.4 | **`closed`** — retiring `is_mcast_active` (zero readers) moved the base 61 → 60, so the tier is 36 and 60+36 = 96 exactly |
| B7a | WH router must not instantiate a Z dispatch arm | 1.6 | **`closed — no code`**: any assert would be tautological. The dispatch arms' `if constexpr` already make a Z arm uninstantiable without Z ports |
| B7b | WH+VC1 compass-only admission lives only in the legacy admit Phase 4 deletes | Q6 | **`closed — false gate`**: the indexed path contains zero `ARCH_WORMHOLE` / `FABRIC_2D_VC1_ACTIVE` references, so collapsing the legacy admit removed nothing WH depended on. Kernel §3.3 was describing an unbuilt design |
| B8 | `intra_mesh_direction_table` is the only source for same-mesh `get_next_hop_router_direction` | 6.1 | **`closed — confirmed true, and it stays`**. On 2D the indexed peek could replace it; on 1D nothing can. Since one L1 struct serves both, the field is permanent |
| D9.1 | zeroed reverse-tree region → multicast silently delivers to nothing, no assert | 1.7 + validity check | **`host side closed`** (1.7 `TT_FATAL`); the device-side zeroed-region tripwire remains a Phase 7 test item |
| D9.3 | same-mesh mcast ignores the caller's anchor (`root_*` from `my_mesh_coord_*`) | 2.2 (add assert) | `open` |
| U1 | express UDM `calculate_initial_direction` read `MEM_TENSIX_ROUTING_TABLE_BASE` while the two other readers in the same file use core-type-dependent `ROUTING_TABLE_BASE`; coincide on Tensix, wrong region on dispatch-engine/idle-erisc | 2.1 | **`fixed`** |
| U2 | ⚠ `direction_to_mux_index_map[COUNT][COUNT]` (5×5) is initialised with only the cardinal 4×4, so the Z row/column read 0. `calculate_initial_direction`'s express arm can return Z → silent forward to mux 0 (wrong mux, no assert) | 2.1 | **`resolved`** — owner call: express stays out of UDM. `static_assert` in the UDM 2D arm of the header selection matrix, beside the existing UDM+1D one. Forward-path asserts kept as a backstop |
| C1 | `test_fabric_set_unicast_route.cpp` builds its **reference** encoding with `fabric_set_route`, which 4.1 deletes | 2.5 | **`partly resolved`** — the encoder is now `#ifdef FABRIC_2D`-guarded, which was **required**: it is unguarded upstream and was breaking the *live 1D* T3K tests. The 2D arm still needs a re-based reference before `DISABLED_TestSetUnicastRoute` can return |
| D9.5 | `routing_plane_connection_manager.hpp:21` keys Z-port capacity on the codec flag | — | **not a defect** — the flag stops being a codec selector, so the site becomes correct with no edit |

---

## 4. Test coverage

New tests the project owes. Keep this honest — an unchecked row is missing coverage, not deferred work.

### Host

| test | asserts | status |
|---|---|---|
| gate separation (1.2) | express meshes: speedy off, trim off, counter credits on; no Z sender without Z ports | `todo` |
| shape sweep | every 2D descriptor: vectors pack, region fits, `Y+X` fits a tier | `todo` |
| axis-topology coverage | `axis_topology(mesh, axis)` non-null for every 2D mesh × axis | `todo` |
| arborescence sweep | every shape × every root × both axes passes the gate | `todo` |
| encoder equivalence | `encode_indexed_mcast_maps` == the codec §5.6 golden path-trace, all roots × all extents | `todo` |
| exit-chip invariant | every intermesh carrier decodes to exactly `LOCAL_DELIVER`, no eth bits, at its exit | `todo` |
| tree-region validity | a zeroed region is rejected, not silently encoded | `todo` |

### Device

| test | phase | status |
|---|---|---|
| single-hop N/S/E/W + Z | 0 | `todo` |
| 2D unicast all-pairs, express + non-express | 2, 3 | `todo` |
| 2D multicast every extent, LINE **and** TORUS Y | 2, 3 | `todo` |
| intermesh unicast, destination-mesh landing | 2, 3 | `todo` |
| intermesh unicast, **intermediate**-mesh landing (needs ≥3-mesh fixture) | 2, 3 | `todo` |
| intermesh multicast carrier + destination landing rebuild | 2, 3 | `todo` |
| `X > 4` unicast (`[8,8]`, `[8,16]`) | 3 | `todo` |
| UDM 2D | 2, 3 | `todo` |
| bandwidth: Galaxy `[32,4]` header regression | 3, 4 | `todo` |

---

## 5. Measurements

Fill in as gates run. These numbers are the evidence for D4 and R2.

| what | expected | measured | when |
|---|---|---|---|
| Galaxy `[32,4]` header, Phase 0 baseline | 96 B | — | — |
| Galaxy `[32,4]` bandwidth, Phase 0 baseline | — | — | — |
| Galaxy `[32,4]` header, after Phase 3 | 112 B | — | — |
| Galaxy `[32,4]` bandwidth, after Phase 3 | regression vs baseline | — | — |
| Galaxy `[32,4]` header, after Phase 4 | **96 B** | — | — |
| Galaxy `[32,4]` bandwidth, after Phase 4 | ≥ baseline | — | — |
| ERISC binary size, Phase 3 | — | — | — |
| **WH** ERISC size, dense key vs legacy bit-test (Q6) | — | — | — |
| ERISC binary size, Phase 5 | ≤ Phase 3 | — | — |
| tier bumps from step 1.8, per platform | none expected | — | — |
| indexed mcast encode cost vs legacy | ~2-3× | — | — |
| `sizeof(routing_l1_info_t)` after Phase 6 | 2576 or 2480 | **2576** | unchanged — 6.2 is dead, see 6.1 |

---

## 6. Log

Newest first. One line per landed change or resolved decision; longer notes below the line.

| date | phase/step | what |
|---|---|---|
| 2026-08-24 | stack investigation | **Reverted, branch is unification-only.** Two stack-frame edits (`port_direction_table` removal, `static constexpr` rodata move) were made while chasing main's `-Werror=stack-usage=1912` ceiling, then reverted at owner request: both shrink `kernel_main()`'s frame, so leaving them in would flatter this branch in a head-vs-branch comparison. Verified `fabric_erisc_router_speedy_path.hpp` now diffs clean against HEAD and the router diff carries no stack-edit hunks. **No demonstrated saving was ever measured** — the numbers I reported earlier compared different logs and different router variants, not before/after. `port_direction_table` remains provably dead (never indexed anywhere) if anyone wants to take it separately. |
| 2026-08-24 | multi-mesh stack overflow | **NOT caused by this branch — owner confirmed.** `kernel_main()` reports 2016 B against the `-Werror=stack-usage=1912` link limit on the 4-mesh `8x4x4` config. The same overflow exists on **main**, on larger machines (TORUS_XY + multimesh); this branch is not the trigger. My initial entry called it a regression — that was wrong, drawn from reading "base passes" as "base passes this config", which was never run on base. Recorded here only so the failure is not re-diagnosed later. **Out of scope for the codec unification; no fix carried in this branch.** A `noinline` mitigation on `fabric_set_indexed_intermesh_landing_route()` was tried and **reverted** at owner request, to keep the branch strictly to the unification. Consequence: the 4-mesh `8x4x4` config cannot be link-tested until the stack issue is fixed on its own, so intermediate-mesh transit stays unverified here. If someone picks it up: `maps[]` is sized `Y + X` and grows with mesh shape, so out-of-lining only buys headroom — the structural fix is a volatile-aware encoder writing straight into the packet header, removing the stack array entirely. |
| 2026-08-24 | torus regression — **found** | **Ring multicast broken on non-express meshes; unicast was never at fault.** Owner's two experiments pinned it: `sync: false` passes (so torus unicast is fine), and express + sync passes (so the reverse-tree encoder itself is fine). That leaves the one path that differs: `express_mcast_outgoing_directions()` in `tt_fabric_test_device_setup.cpp` early-returned `{}` for any mesh without express routing, and the caller's fallback pushes exactly one outgoing direction. A full-ring multicast root needs **two** — the tree branches forward and backward around the ring — so one branch was never injected and the ring sync could never complete. Fixed: gate removed, and `ring_for_direction()` (null on a non-express Y axis) replaced with `axis_topology()`, which is never null for a 2D mesh. Also renamed `is_express_mcast` → `is_tree_mcast` since the shape is no longer express-specific. Audited the remaining `express_routing_enabled()` call sites in `tt_metal/`: all are genuine Z-chord facts (topology, BFC, sizing, optimization), none is codec selection. |
| 2026-08-24 | torus regression | **Open: 2D Torus hangs, 2D Mesh passes.** Owner confirms base passes, so this is a regression. All 4 endpoints stall at 0 packets. Torus is the first config to run reverse-tree multicast on a **non-express** mesh — on base, trees were embedded only for express meshes and a plain torus used the legacy hop-count multicast. **Excluded so far:** axis topology (added a per-mesh diagnostic; reports `Y=RING X=RING`, so the LINE-derivation theory is dead); indexed vector + tree generation (both guards now `TT_FATAL`, neither fired); multicast ring-extent encoding (`encode_indexed_mcast_maps` wraps correctly via `% y_size`); VC1 structurally excluding the admit (the two `#if !defined(FABRIC_2D_VC1_ACTIVE)` blocks are `FORCE_INLINE` selectors only, and `admit_indexed_dispatch` is guarded solely by `FABRIC_2D`); packet-header layout shift from retiring `is_mcast_active` (no `offsetof` into `HybridMeshPacketHeader` anywhere). **Next discriminator:** `sync: false` — sync is a full-ring multicast, so disabling it separates "ring multicast broken" from "torus unicast broken". |
| 2026-08-24 | hardening | Converted the indexed-table guards from `TT_ASSERT` to `TT_FATAL`. `TT_ASSERT` expands to `(void)(condition)` in Release, and `data` is `memset` to zero beforehand, so a failed pack silently embedded an **all-zero routing table** — every action decodes to 0, nothing forwards, senders stall at 0 packets. Same signature as the bug under investigation, and it would have hidden it. Also fixed the `dir.value()`-on-empty-optional probe guard. |
| 2026-08-22 | third device run | **Dispatch kernels now compile; the router did not.** 66 errors, one cause: `fabric_set_indexed_intermesh_landing_route` undeclared at `fabric_erisc_router.cpp:1478`. The declaration used to reach the router *transitively* through `fabric_edge_node_router.hpp` (which included `tt_fabric_api.h`); deleting that header in 4.2 removed the path. When I deleted it I verified its own two functions were unreferenced, but never checked what it re-exported. Restored the include directly, inside the empty `#ifdef FABRIC_2D` the removal had left behind — the call site's guard stack is exactly `#if defined(FABRIC_2D)`, so the gating matches. Verified the definition at `tt_fabric_api.h:302` is unguarded and that the two files share no other file-scope name, so the include introduces no collision. |
| 2026-08-22 | second device run | **Router kernel now compiles; the dispatch kernels did not.** 512 errors, one cause: `FABRIC_2D_MESH_{Y,X}_SIZE` undeclared in `tt_fabric_api.h` at lines 68/125/126/150, while building **`cq_dispatch` and `cq_prefetch`** — neither of which defines `FABRIC_2D`. The fallback `#define` was gated on `defined(FABRIC_2D)` while its users are ungated function bodies, and `cq_relay.hpp` reaches them through `if constexpr (FABRIC_2D)`, which does not preprocess the 2D arm away. Ungated the fallback. Verified by compiling a stub that includes `tt_fabric_api.h` with the *exact* non-2D flags taken from the failing `cq_dispatch` command line: zero `FABRIC_2D_MESH_*` errors (the one remaining diagnostic is an include-order artifact of the stub, not of the header). **This is the fourth time the same root cause has bitten** — scoping by which code *uses* a thing rather than which builds *compile* it. |
| 2026-08-22 | first device run | **Kernel JIT compile failed; one root cause, now fixed.** 624 errors, every one at `fabric_erisc_router.cpp:546`/`:553`: `map_downstream_direction_to_compact_index` undeclared. Cause was an over-deletion in turn-set deletion (5.1) — a text-anchored range cut that swept up `direction_to_compact_index_map` and both accessor overloads along with the turn machinery it was aimed at. Restored all three, definitions placed before their uses. Verified `hop_cmd_to_sender_channel_mask` (also in the swept range) was legitimately dead — only comment references remain. Swept the whole file for other file-scope symbols used-but-undefined: none. The log contains **no other error class**, so the rest of the kernel parsed. |
| 2026-08-21 | Phase 7 (host) | **All host tests green — 29/29.** The descriptor sweep passes over all 158 descriptors after both fixes: my `axis == 0` guard in `derive_axis_topology()`, and `[RING, RING]` → `[LINE, RING]` on all four meshes of `express_links_8x4x4`. `EveryRouteCrossesOnlyDeclaredEdges` runs 541 ms, so it is genuinely deriving and walking every route rather than short-circuiting. Also corrected two stale rows in this table: the **arborescence sweep** and **encoder equivalence** tests I had listed as owed already exist in `test_mcast_reverse_tree.cpp` — it runs the arborescence gate over every root on both axes and diffs the encoder against a forward-route golden. What it does *not* cover is non-express and wide-X shapes, which is precisely the gap the new sweep fills. |
| 2026-08-21 | Phase 7 (host) | **Both sweep findings resolved.** The `axis == 0` guard worked — every axis-1 fatal disappeared, leaving only the genuine dim-0 ones. Owner then changed **M0** of `express_links_8x4x4` to `dim_types: [LINE, RING]` and it derived cleanly, which confirmed the diagnosis outright: with dim 0 LINE, `fill_blocks()` drops the straddling `[6,9]` block, `declared_chords` falls 3 → 2, and it matches the 2 Z edges that physically exist. M1–M3 were identical 8x4 pieces of the same split still declaring `[RING, RING]`, so I applied the same change to them. Also fixed a flaw in my own test: `MinimumCoverage` was re-reporting derivation failures it does not own, turning one fault into two red tests — it now counts them and leaves the reporting to `EveryRouteCrossesOnlyDeclaredEdges`. |
| 2026-08-21 | Phase 7 (host) | **The descriptor sweep found two real bugs on its first run.** (1) **Mine, fixed:** `derive_axis_topology()` called `derive_express_ring_topology()` for *every* mesh and *every* axis. That function is not total — it `TT_FATAL`s on a dim-0 express configuration it cannot reconcile — so a dim-0 problem was being raised while the caller asked about **axis 1**, where express is irrelevant (express supports dimension 0 only). Before line-axis topology (1.5) this was unreachable: derivation ran only for meshes that had already passed the express gate. Now guarded with `if (axis == 0)`. (2) **Not mine, needs an owner decision:** `express_links_8x4x4_mesh_graph_descriptor.textproto` (untracked WIP) declares `dim_types: [RING, RING]` while its own header comment says `[LINE, RING]`. With dim 0 RING, `fill_blocks()` keeps the straddling block `[6,9]` as the wrap-around chord `(6,1)`, so `declared_chords` = 3 while only 2 Z edges exist — the descriptor's comment says that block "straddles the mesh boundary and is dropped", which is what `[LINE, RING]` would actually do. |
| 2026-08-21 | Phase 7 (host) | **26/26 pass** — `IndexedRouteCodec.*` (23) and `AxisRouteTopologyLine.*` (3), via `cmake --build build --target fabric_unit_tests`. Worth keeping in proportion: these cover the codec's *arithmetic* — packing, decode selection, bounds, action validity — which is the part least likely to have been wrong. The subsystems I changed most (router dispatch, multicast encoder, topology derivation over real descriptors) are still unexercised. |
| 2026-08-21 | Phase 7 (host) | **Wrote the machine-free unit tests.** New `fabric_router/test_indexed_route_codec.cpp` (registered in `sources.cmake`): shape admissibility across all 10 in-tree shapes + the `[64,4]` / oversize boundaries; `vectors_region_bytes` against hand arithmetic; header tier size-classes; pack/decode round-trip on a DOR oracle; pack rejects off-axis actions, oversize shapes, and Z-on-X; pack does not scribble past its region; and the decode rules — including the one that matters most, that a Y byte holding **`LOCAL_DELIVER` alone** must not fall through to X (the `!= 0` vs `& (N\|S\|Z)` trap). Plus action validity (self-facing bit, reserved bits) and `fwd_dirs`/`pack_fwd_key` slot-order round-trip. Appended 3 line-axis cases to `test_express_ring_topology.cpp` guarding the 1.5 `wraps` fix, with a ring counterpart so the flag cannot silently stop mattering. Both TUs pass `-fsyntax-only`; **not built or run** — owner is building. |
| 2026-08-21 | close-out | **Implementation done; ledger reconciled against the tree.** Closed B1, B2, B3, B5, B6, B7a, B7b and the host half of D9.1 — all had been resolved by the work but never marked. Verified the mechanically-checkable gates directly rather than asserting them: `turn_point`, `is_mcast_active` and `compressed_route_2d_t` are gone tree-wide; every surviving `hop_index` is **1D** (`LowLatencyFields` lambda parameters), and the lone 2D survivor is the `routing_fields` declaration the out-of-scope profiler decodes; `fabric_erisc_router.cpp` has zero `express` occurrences; no 1D-specific source appears in the diff. Gate 6 measured: `sizeof(routing_l1_info_t)` = **2576 B**, unchanged. Everything still open is either device testing (Phase 7) or explicitly deferred: 5.2 `port_direction_table` (dead-code discussion), the `routing_fields` reclaim (profiler), and `[64,4]` (issue #32237). |
| 2026-08-21 | regression found | **A green host build hid a real break, exactly where I said it would.** `fabric_set_route()` was declared *outside* any `FABRIC_2D` guard, and `tests/.../kernels/test_fabric_set_unicast_route.cpp` defines its reference encoder unguarded — so that encoder compiled into the **1D** build too, even though only the 2D arm of `kernel_main()` ever calls it. Deleting `fabric_set_route` in 4.1 therefore broke `TEST_F(Fabric1DFixture, TestSetUnicastRoute)` and `...IdleEth` — **live, non-disabled T3K tests** — at kernel JIT-compile time, for a function they never call. I had scoped C1 as touching only the two 2D tests I had disabled; that was wrong. Fixed by guarding the encoder with `#ifdef FABRIC_2D`. Then swept every symbol deleted from `tt_fabric_api.h` against the whole tree: only `fabric_set_route` and `fabric_set_indexed_single_hop_unicast_route` were genuinely removed, and the sole surviving caller of either is inside that now-guarded block. Also removed the two dead `MeshRoutingFields` aliases (`tt_fabric_api.h`, `fabric_erisc_router.cpp`); the profiler keeps its own. **Standing caveat: kernels are JIT-compiled at test time, so the host build validates none of them.** |
| 2026-08-21 | 6.1 direction-table removal, 6.2 L1 reclaim, 6.3 axis validation lift | **Phase 6 closed; two of its three steps turned out to rest on a false premise.** 6.1 assumed `intra_mesh_direction_table` is a redundant cache of the indexed first-hop peek. That is true on 2D and false on 1D: `TEST_F(Fabric1DFixture, TestGetNextHopRouterDirection1D)` exercises `get_next_hop_router_direction` on a 1D build, which has no indexed vectors, and the API has 14 external callers across ttnn CCL and deepseek. One `routing_l1_info_t` serves both modes, so the field is permanent — **B8 closes as confirmed-and-unfixable**, and 6.2's 96 B reclaim dies with it (`sizeof` stays 2576). I did not repoint the 2D arm to the peek either: the table stays regardless, so the change would buy nothing while risking the API's `INVALID_DIRECTION`-on-self/unreachable contract. 6.3 was real and is done: the flat `mesh_shape <= 32` per axis is replaced by `IndexedMeshRoutingFields::shape_is_indexable()`, the *same* predicate `pack_indexed_route_vectors()` enforces, so a validated shape can no longer fail to pack. Precise `[64,4]` verdict: it passes indexability exactly (1028 B, the entire slot) and fails **only** the header bound, by **one byte** (Y+X = 68 vs 67) — squarely issue #32237. |
| 2026-08-21 | first build | **Two compile errors, both from 4.4; both fixed. First real compiler feedback on ~2,900 lines of change.** (1) The new top tier `HybridMeshPacketHeaderT<68>` tripped a pre-existing `RouteBufferSize <= 67` static_assert — an L1 memory-map bound tied to `ROUTING_PATH_SIZE_2D = 1024` and tracked as issue #32237. My arithmetic was right that 60+68=128, but I had missed that the struct is `packed, aligned(16)`, so `<67>` already **is** a 128 B header (127 padded up). The 68-byte tier was never needed: top tier reverted to **67**, the bound left untouched — raising it is Phase 6 / #32237 work, not 4.4's. Cost is one padding byte in the largest tier and a hard ceiling of Y+X ≤ 67, which the existing host `TT_FATAL` at fabric_context.cpp:157 already reports clearly. (2) `-Werror,-Wunused-variable` on `topology` in erisc_datamover_builder.cpp, orphaned when the `UPDATE_PKT_HDR_ON_RX_CH` derivation was removed in 5.1. Removed. All 12 changed host TUs now pass `-fsyntax-only` with the real build flags. Kernel-side TUs remain unverified — they need the SFPI cross-compiler and a device build. |
| 2026-08-21 | 5.1 turn-set deletion, 5.4 kernel rename | **done — and 5.1 turned up a live mutation on the indexed hot path.** `forward_payload_to_downstream_edm()` is shared by 1D and indexed 2D, and its 2D `update_packet_header_for_next_hop` overload was still executing `routing_fields.value = cached + 1` whenever `UPDATE_PKT_HDR_ON_RX_CH` was set. Harmless in effect (nothing reads the field any more) but a real store per forwarded packet, and a direct violation of the immutable-maps invariant. Both 2D overloads are now no-ops — kept rather than deleted, because the shared call site resolves them by header type and removing them would break the 1D-shared path on a 2D build. `UPDATE_PKT_HDR_ON_RX_CH` then had no consumer and was removed from both the kernel declaration and the host emission, along with its host-side derivation. The turn plumbing (`is_spine_direction`, `TURN_STATUS_ARRAY_SIZE`, `get_sender_channel_turn_statuses`, `sender_channels_turn_status`) was by then entirely self-referential and deleted. 5.4: 11 identifiers renamed; the kernel now contains zero `express` occurrences. |
| 2026-08-21 | 4.4 header reclaim | **done, and the plan's D4 was wrong in a way that matters — for the better.** D4 said retire *both* `routing_fields` (4 B) and `is_mcast_active` (1 B) for a 56 B base and a 40-byte tier. But `routing_fields` cannot go: `tools/profiler/fabric_event_profiler.hpp` decodes `branch_east_offset` / `branch_west_offset`, and the profiler is out of scope by owner call. Retiring `is_mcast_active` **alone** turns out to be sufficient — it had zero readers (3 writes, 1 declaration), and 60 + 36 = 96 exactly. So Galaxy `[32,4]` is back to a **96 B** header with the profiler untouched. Tiers 35/51/67 → 36/52/68, `MAX_2D_ROUTE_BUFFER_SIZE` 67 → 68, default 35 → 36, and `get_2d_header_size`'s switch updated to match. Verified all 10 in-tree shapes land on their pre-flip tier; `[64,4]` still needs 128 B. Reclaiming `routing_fields`' 4 bytes stays blocked on the profiler decision. |
| 2026-08-21 | 4.3 legacy L1 table deletion | **done.** Deleted `compressed_route_2d_t`, the `intra_mesh_routing_path_t<2,true>` specialization plus its device decode and explicit instantiation, `encode_2d_unicast`, the `routing_path_table_2d` union member, `MAX_CHIPS_LOWLAT_2D`, `SINGLE_ROUTE_SIZE_2D`, and the host-side 2D route generation. Collapsed `intra_mesh_routing_path_t` to 1D-only (`static_assert(dim == 1)`), keeping the `<1, compressed>` spellings so existing uses still name it. `sizeof(routing_l1_info_t)` verified still **2576** — the union slot was already the 1028 B indexed table. Also removed the now-dead legacy `#else` arms in both UDM headers (UDM is 2D-only, so they were unreachable) — and broke `tt_fabric_udm.hpp`'s nesting doing it, caught again by the depth check. ⚠ `RoutingFieldsConstants::Mesh` **survives**: its only real consumer is the profiler, so it stays until that decision. |
| 2026-08-21 | 5.3, 4.2, 5.1 (partial) | **the legacy 2D kernel path is gone.** 5.3: collapsed all three `if constexpr (indexed_2d)` arms and removed the constant; deleted the legacy 2D admit (16-arm switch, including the `ARCH_WORMHOLE && FABRIC_2D_VC1_ACTIVE` branch — the WH exception's only home) and the legacy 2D `receiver_forward_packet` (446 lines). Router went 4151 → 3540 lines. ⚠ I broke the preprocessor nesting doing the block deletion (dropped the `#endif` for the 1D `#ifndef FABRIC_2D` arm) and caught it by comparing final `#if` depth against the pre-edit file — worth keeping that check on any further block surgery. 4.2: `fabric_edge_node_router.hpp` `git rm`'d and its include dropped, after confirming `recompute_path` and `get_cmd_with_mesh_boundary_adjustment` were both at 0 external refs. 5.1 partial: `update_packet_header_before_eth_send` (the `hop_index`/`branch_*` mutation) and its already-dead call site deleted; the builder-side turn plumbing (`is_spine_direction`, `TURN_STATUS_ARRAY_SIZE`, `get_sender_channel_turn_statuses`, `sender_channels_turn_status`) still has refs and is next. |
| 2026-08-21 | Q6 | **corrected: Q6 was a false gate, and I had already established that at B7.** I let it drift back into a blocker. The facts: the WH+VC1 compass-only admission exists at exactly one site (`fabric_erisc_router.cpp:738`) inside the *legacy* `can_forward_packet_completely`, and `admit_express_dispatch` / `admit_express_combo` contain **zero** `ARCH_WORMHOLE` or `FABRIC_2D_VC1_ACTIVE` references. The indexed path has one admission implementation, always. So deleting the legacy admit removes nothing from it — "preserve the exception vs lose it" was a false choice. Kernel §3.3's compass-only guidance was *prescribing* a design for the indexed path that was never implemented, not describing behaviour to protect. Whether to add a WH-specific variant is a Phase 7 perf question. 5.3 unblocked. |
| 2026-08-21 | 4.x | **plan ordering error found: Phase 4's deletions all depend on 5.3.** Phase 3 deliberately kept the router's legacy `else` arms so the flag day stayed revertible — but those arms are exactly what still reference everything Phase 4 deletes: `get_cmd_with_mesh_boundary_adjustment` (4.2), `MeshRoutingFields::` at 44 sites (4.3), and `routing_fields.hop_index` at 14 sites (4.4). So 5.3 (collapse the arms) has to run first, and the real order is 4.1 → 5.3 → 4.2 → 4.3 → 4.4 → 5.1/5.4. Worth noting the honest consequence: the "one-commit revert" property the flip was designed for ends at the first Phase 4 deletion, which was always true but the phase numbering hid it. |
| 2026-08-21 | 4.1 legacy producer deletion | **done.** Deleted from `tt_fabric_api.h`: the `single_hop_route_cmd_by_direction` opposite-direction table, `fabric_set_route<mcast>` (the raw hop-program primitive, no indexed equivalent by construction), the legacy spine/branch body of `fabric_set_mcast_route`, the legacy arm of the single-hop helper, and the `called_from_router` legacy encode. `fabric_set_unicast_route` keeps its template parameters so the ~149 call sites compile unchanged, but now carries `static_assert(!called_from_router)` — the only `<true>` caller was `fabric_edge_node_router.hpp`, which 4.2 deletes. ~12 KB removed. |
| 2026-08-21 | tests | 2D `TestSetUnicastRoute` / `...IdleEth` **disabled with a reason**: their kernel validates `fabric_set_unicast_route` by diffing it against a *reference* built from `fabric_set_route`, so Phase 3 already made the comparison meaningless (indexed maps vs hop program) — before any deletion. 1D variants unaffected. Replacing it needs an indexed oracle (assert widened maps against the L1 vectors, not against a second encoder); Phase 7 item. |
| 2026-08-21 | **Phase 3 — THE FLIP** | Indexed 2D is now unconditional. 9 codec guards moved from `FABRIC_EXPRESS_ENABLED` to `FABRIC_2D`; `FABRIC_EXPRESS_ENABLED` survives at exactly **2** genuine sites (Z-port capacity in `routing_plane_connection_manager.hpp`, and the UDM bar). Renames: `FABRIC_EXPRESS_MESH_*_SIZE` → `FABRIC_2D_MESH_*_SIZE`, kernel `EXPRESS_MESH_*` → `MESH_*`, `express_enabled` → `indexed_2d`, `get_express_kernel_defines` → `get_2d_kernel_defines`. The L1 embed no longer writes `intra_mesh_routing_path_t<2,true>` at all — that write is **deleted**, so the legacy 2D table is dead from this commit even though its type still exists until 4.3. Deferred halves of 1.8 (route-buffer sizing for every 2D mesh) and 2.2 (unconditional source-inject) landed here as planned. **R4 diff guard PASSES** — the only removed `express` line is the G1 CT-args gate, which is what this phase exists to flip; all four G4/G5 compositions are byte-identical and `fabric_builder_context.cpp` has zero changes. Legacy `else` arms in the router left in place deliberately so the flip is revertible; Phase 5 deletes them. |
| 2026-08-21 | 3.3 no-node fatal removal | **corrected — I overstated this.** I had flagged 3.3 as a hard precondition needing a decision between three options. Wrong on both counts. (a) The dispatch relays already resolve their own mesh and pull `get_express_kernel_defines`, with a comment saying why — the "every `FABRIC_2D` compile needs shape defines" risk was already closed. (b) The no-node overload's fatal exists because the 2D ABI varied per mesh, which is precisely what the flip removes, so 3.3 is just deleting it. `api_type` selects the Linear (1D) vs Mesh (2D) *API surface*; express is a flavour of mesh routing, not a third api_type, so there is no ABI choice at that layer. Guide 3.3, guide 3.1's note, plan §9 Q2 and this doc all corrected. |
| 2026-08-21 | 2.5 2D API surface | **done.** Deleted the orphaned `fabric_set_indexed_single_hop_unicast_route` — 0 callers, and redundant since 0.1 made the legacy-named wrapper resolve the direction and delegate to the forking `_from_direction`. Left a comment so it is not re-added. All 5 remaining indexed variants confirmed reachable. `fabric_set_route`'s two consumers dispositioned: `cq_relay.hpp` handled by 3.4; the test oracle is C1, still needing a re-base-or-retire call at 4.1. |
| 2026-08-21 | 2.3 recompute_path audit | **done — 4.2 is licensed.** Enumerated the legacy machinery (2 triggers × 4 outcomes incl. the mcast branch) and matched every one to an indexed counterpart. Trigger-coverage question resolved by construction: the `NOOP` that fires trigger A is planted at the *exit* into `route_buffer[1]`, so the router that trips on it is always the far-side landing, i.e. always an intermesh ingress — there is no path that plants a NOOP and trips on it without crossing a boundary. Two divergences are improvements, not gaps: the Z/`NOOP` overload disappears (legacy `set_forward` writes NOOP for a Z egress, which `recompute_path` then returns as a hop_cmd that admission stalls on), and an intermediate landing installs the whole next-exit route once instead of re-entering case 3 per hop. Residual risks recorded: the exit-chip `LOCAL_DELIVER` invariant loses its rebuild safety net (Phase 7 host test), and VC2 is excluded from the landing intercept by the builder's own documented decision. |
| 2026-08-21 | 2.2 unconditional mcast inject | **half already done, half must move to 3.4.** Q1's single-output-root guard already exists verbatim in the express mcast arm. Making `fabric_multicast_source_inject_*` unconditional cannot happen this phase: it references `FABRIC_EXPRESS_MESH_{Y,X}_SIZE`, and that fallback is itself guarded on `FABRIC_EXPRESS_ENABLED`, so a non-express build has no shape macros — it would not compile. Same ordering class as 1.4 and 1.8. |
| 2026-08-21 | U2 | **resolved by owner call: express stays out of UDM.** Established that the mux fabric is cardinal-only *by construction*, not by oversight — `NUM_DOWNSTREAM_MUX_CONNECTIONS` is a hardcoded 3 ("all directions except self") and the tensix builder loops `dir_idx < eth_chan_directions::Z` with the comment "Skip Z direction - it's for 3D routing". So there is no Z mux and no slot for one; extending `direction_to_mux_index_map` to a real 5×5 would index a connection that does not exist. Declared unsupported with a `static_assert` in the UDM 2D arm of `fabric_edm_packet_header.hpp`, beside the existing "UDM mode does not support 1D routing" assert — same shape, same file. Kept the 2.1 forward-path asserts as a backstop. **Consequence to protect:** this composes with Phase 3 only because Q2 kept `FABRIC_EXPRESS_ENABLED` express-scoped; had it been renamed to mean "indexed 2D", the assert would fire for every UDM build after the flip. Noted in guide 3.4, which also now flags that the UDM guards must move to `FABRIC_2D` or non-express UDM reads the indexed slot as a legacy table. U1's fix is now on an unreachable arm but kept, since it is correct if the combination is ever enabled. |
| 2026-08-21 | 2.1 UDM producer audit | **done, and it was not the no-op the guide predicted — two live defects on express+UDM.** (U1) `calculate_initial_direction`'s express arm hardcoded `MEM_TENSIX_ROUTING_TABLE_BASE`, while lines 181/252 of the same file use `ROUTING_TABLE_BASE`, which is core-type dependent (aerisc / ierisc / dispatch-tensix / tensix). They coincide on a plain Tensix worker, so it works there and reads the wrong region on a dispatch-engine or idle-erisc compile. Fixed to use the macro. (U2) `direction_to_mux_index_map` is declared `[COUNT][COUNT]` = 5×5 but initialised with only the cardinal 4×4, so its Z row and column are zero-filled — and the express arm of `calculate_initial_direction` **can return Z**. A chord first hop therefore indexed the zero column and forwarded to mux 0, the wrong mux, with no assert. Added fail-loud asserts; a real Z mapping needs the builder's mux allocation for the Z peer, which is not modelled here, so whether express+UDM must support a Z first hop is an owner decision. |
| 2026-08-21 | — | **sequencing: all testing deferred to a new Phase 7** (owner call). Per-phase gates keep only non-test conditions. Noted that Gate 3 thereby degrades from "full regression green" to "it builds", so the per-phase log is the bisect trail. |
| 2026-08-21 | 1.2 gate-separation test | **not written — cannot be tested without fabric.** The R4 invariant ("express keeps speedy/trim off, counter credits on") is not a pure function anywhere: it is five inline `!express_routing_enabled && ...` / `\|\| express_enabled` compositions in `erisc_datamover_builder.cpp` and `fabric_builder_context.cpp`. `vc0_speedy_path_enabled()` *is* pure but does not take express — the gate is applied at the call site. Reaching the composed decision needs a ControlPlane plus an EDM builder; making it pure would mean refactoring the G5 code this project leaves alone. Replaced with a mechanical Gate 3 diff guard naming the five lines. Honest limitation: that catches an accidental edit, not a semantic regression arriving by another route. |
| 2026-08-21 | 1.6 Z-arm invariant, 1.8 route-buffer sizing | **1.6 needs no code; 1.8 deferred to 3.2.** 1.6: the Z-arm invariant is already enforced structurally — every dispatch arm forms its array index only inside `if constexpr (express_arm_is_realizable<...>())` and fails closed otherwise, so WH compiles the Z arm away entirely. Any static_assert would restate that same condition, and `num_z_ports` is not visible to the kernel to cross-check. Wrote the reasoning as a comment instead of adding a fake check. 1.8: computed the tier impact per shape — only `[32,4]` moves (35→51 bytes, 96 B → 112 B header), which is a real bandwidth regression for a `[32,4]` mesh still running the legacy encoding, so it must land with the flip rather than a phase early. |
| 2026-08-21 | 1.7 one-feeder gate | **done.** `log_warning` → `TT_FATAL`, propagating `embed_mcast_reverse_trees`' own row/both-feeders message rather than summarising. Also switched from `ring_for_direction()` to `axis_topology()` and made a null axis topology fatal — `ring_for_direction` is null on a non-express Y or a non-closing X, and a null topology is exactly what leaves the tree region zeroed. Bonus correctness: `axis_topology` keys on the express ring's declared `axis_dim` instead of assuming N/S means express. Still inside the express-only branch, so non-express meshes are untouched this phase. |
| 2026-08-21 | 1.5 line-axis topology | **done, and it needed a correctness fix I had not planned for.** `next_row()`'s `step()` picks a direction by comparing forward vs backward distance **modulo the cycle length**, so on a line it would return the far end for a distant destination — a hop over the wrap edge a line does not have. Added a `wraps` flag (default `true`, so express and ordinary-ring behaviour is byte-identical) and guarded `step()` on it. Without this the line topology would have produced routes over nonexistent links. Also added `derive_line_axis_topology` (interior edges only; no wrap edge required) and `derive_axis_topology` (express-for-that-axis → ring → line), `axis_topologies_[mesh][axis]` populated for every mesh and both axes, `RoutingTableGenerator::get_axis_topology`, `ControlPlane::axis_topology`. Degenerate axes (len < 2) stay null rather than fabricating a one-element cycle. Confirmed the tree builder only needs `axis_len`/`axis_dim`/`next_row`, so it works on a line unchanged. Q3 rename applied: `express_ring_topology.{hpp,cpp}` → `axis_route_topology.{hpp,cpp}`, `ExpressRingTopology` → `AxisRouteTopology`, 13 files + `sources.cmake` (which would have broken the build if missed). `derive_express_ring_topology` keeps its name — it genuinely derives express rings. |
| 2026-08-21 | 1.4 X-ring hoist | **folded into 1.5, not done standalone.** Attempted the hoist, then reverted it: `derive_ordinary_ring_topology` (`express_ring_topology.cpp`) `TT_FATAL`s if any line on the axis is missing an ordinary edge, but `axis_wraps` only inspects line 0. Deriving X rings for every mesh could therefore abort init on an asymmetrically cabled mesh that boots today — a behaviour change Phase 1 is not allowed to make. It is safe only together with 1.5's line fallback, which turns that case into a line topology. A comment recording this is left at the call site. |
| 2026-08-21 | 1.3 shape bound split | **done.** Replaced the per-axis `X <= 4` cap with two clearly separated bounds: `SLOT_SHAPE_{Y,X}` (what the L1 slot is *sized* for, `[64,4]` = 1028 B) and `MAX_INDEXED_MESH_AXIS` (the real per-axis limit, 64, fixed by the packed reverse-tree descriptor's 6-bit row indices). Admissibility is now `both axes <= 64 AND packed tables fit the slot`. Added static asserts pinning the in-tree shapes the old cap excluded — `[8,8]` 60 B, `[8,16]`/`[16,8]` 124 B, `[1,16]` 98 B, `[32,32]` 636 B, all against a 1028 B slot. |
| 2026-08-21 | 1.1 guard inventory | **done** (audit, no code). Inventory in plan §1.1: 8 codec sites + 1 genuine express (Z-port capacity). Client-facing API audit in plan §2.2. |
| 2026-08-21 | 2.5 2D API surface | **audited the client-facing API layer** (owner caught that 2.5 was auditing encoders, not the APIs other teams call). Result: **no public 2D fabric API exposes a legacy-shaped signature** — all are `(dst_dev_id, dst_mesh_id)` or that plus `ranges`, never `direction + num_hops`. So no public signature changes and no consumer migrations. Also found `linear/api.h` **has 2D paths** under `FABRIC_2D`, and its multicast shim passes exactly one nonzero extent per slot by connection tag — independent confirmation of Q1, and it localises Q1's assert to `mesh/api.h`, the only layer that can express a multi-direction rectangle in one call. |
| 2026-08-20 | 2.5 2D API surface | **new step added** at owner request: unify the 2D route API surface so consumers can only reach the indexed encoders. Audit result: the ~200 `fabric_set_unicast_route` / `fabric_set_mcast_route` call sites need **no** change, because they go through names that already fork internally — the fork is what Phase 3 removes. Real work is narrow: `fabric_set_route`'s 2 consumers, the `called_from_router` arm (B5), and one orphaned indexed variant. |
| 2026-08-20 | 2.5 2D API surface | **retracted a false finding.** I reported the emulator as a third `fabric_set_route` consumer and added P10/C2/C3/R9 plus two gate conditions for it. It was a **grep substring artifact** — `__emule_fabric_set_route(` matched `fabric_set_route(`. The emulator does not call it and does not include `tt_fabric_api.h`. All emulator material removed from all three docs; no emulator code was ever modified. |
| 2026-08-20 | Gate 0 | device single-hop test **skipped** per owner; coverage folded into 2.4. Step 0.1 code stands. |
| 2026-08-20 | 0.1 single-hop ABI fix | **code landed** in `tt_fabric_api.h` (+53/-18). Express builds now delegate the single-hop helpers to `fabric_set_indexed_single_hop_unicast_route_from_direction`. Three things the guide had wrong or missing, all now folded back into step 0.1: the shape-define fallback had to be hoisted above first use or ERISC would not compile; `#if/#else` beats an early return because the legacy body then does not compile on express at all; and the same-mesh precondition needed an explicit `ASSERT` because the existing bounds check silently passes for an in-range foreign chip id. |
| 2026-08-20 | Q4 | **decided: hard `TT_FATAL`.** Confirmed with the owner that a row with two feeders is structurally impossible on a chordless 2D mesh, so the gate is a no-op there; and express multicast is working today, so the check already passes on shipping express topologies. A fatal therefore cannot reject a configuration that currently functions. |
| 2026-08-20 | docs | **corrected two errors of mine.** (1) Multicast targets are always a *contiguous* run from the anchor (`n_hops`/`s_hops` extents) — a non-adjacent target set like {3,5} is not expressible, and an example using one was wrong. (2) Codec §7.3.1's "express meshes ship unicast only" is **stale**: `fabric_multicast_source_inject_*` exists and express multicast is implemented, tested and working. Plan non-goals and comparison §7 corrected. |
| 2026-08-20 | Q1, Q3 | **decided.** Q1: adopt the per-direction multicast contract and add `ASSERT(popcount(root_action & ETH_MASK) <= 1)` on the shared path. Q3: rename `express_ring_topology.{hpp,cpp}` → `axis_route_topology.{hpp,cpp}` and `ExpressRingTopology` → `AxisRouteTopology` in Phase 1. Q5/Q6 reclassified `deferred` — both need measurements that do not exist until Gate 4 / Gate 3. |
| 2026-08-20 | 1.1 / 3.5 | **`FABRIC_EXPRESS_ENABLED` keeps its name.** Dropped the proposed `FABRIC_2D_HAS_Z_PORTS` rename — the emission condition (`express_routing_enabled`) never changes, so it was a second spelling for an unchanged fact. Stripping the codec consumers makes the existing name *more* accurate. `routing_plane_connection_manager.hpp` needs no edit at all; only the router-side emission (`compute_mesh_router_builder.cpp:906-913`) is deleted, since the router does not include that header. Step 1.1 becomes an audit. |
| 2026-08-20 | B7 | **corrected: B7 was mis-specified.** It was written as "re-key the WH+VC1 admission exception," but that exception exists only at `fabric_erisc_router.cpp:730`, inside the *legacy* admit that Phase 4 deletes; `admit_express_dispatch` has no WH variant. Split into B7a (Z-arm tripwire, step 1.6 — already true structurally) and B7b (WH admission shape → new Q6, decided at Gate 3). |
| 2026-08-20 | — | audited what still needs a genuine express guard device-side: **exactly one site**, `routing_plane_connection_manager.hpp:21`, because `TT_FABRIC_MAX_ROUTING_PLANE_CONNECTIONS` is an array bound in a type and cannot be a CT arg. Everything else express-specific is already a host-derived CT arg. Table in plan §1.1. |
| 2026-08-20 | Q2 | **decided: delete `FABRIC_EXPRESS_ENABLED`, don't rename it.** It is redundant with `FABRIC_2D` after the flip (both emitted from the same `if` block, `compute_mesh_router_builder.cpp:869` vs `:912`). Guard becomes `#if defined(FABRIC_2D)`. Side effect: makes step 3.3 a hard precondition — see notes. |
| 2026-08-20 | — | branch `nnyamagoudar/subtorus-routing-kernel-unification` created off `42204c541c3` |
| 2026-08-20 | — | comparison / plan / guide written; scope narrowed to the express↔prior-fabric divide |

### Notes

**R4 diff guard (replaces the untestable step 1.2).** The "express keeps speedy/trim off and counter
credits on" invariant is not reachable from a pure host test: it lives as five inline compositions in
the builder rather than in any pure function, so exercising it needs a ControlPlane plus an EDM
builder. Extracting predicates to make it testable would mean refactoring the G5 code this project
deliberately leaves alone. So the guard is mechanical instead — Phase 3 must not touch these lines:

```text
erisc_datamover_builder.cpp:796    if (channel_trimming_overrides.has_value() && !express_routing_enabled)
erisc_datamover_builder.cpp:1082   can_use_forwarding_capture_by_vc[vc] = !express_routing_enabled && ...
erisc_datamover_builder.cpp:1090   vc0_trim_fast_path_usable = !express_routing_enabled && ...
erisc_datamover_builder.cpp:1129   enable_speedy_vc0 = !express_routing_enabled && vc0_speedy_path_enabled(...)
fabric_builder_context.cpp:53      .vc1_uses_counters = multi_txq_enabled || express_enabled
```

Weaker than a test — it catches an accidental edit, not a semantic regression reached another way —
but it costs nothing and is checkable by `git diff`. Line numbers will drift; the predicates are
`!express_routing_enabled` in the trim/speedy decisions and `|| express_enabled` in the credit plan.

*(free-form: surprises, reverts, things the guide got wrong, findings that should move into the
reference docs)*

- **Shape defines**: keep the zero-fallback in `tt_fabric_api.h`, retargeted to `FABRIC_2D`, so a
  compile with no single mesh shape builds and fails at the runtime `ASSERT(y_size > 0 && x_size > 0)`.
  The dispatch relays need nothing special — `dispatch.cpp:585` / `prefetch.cpp:549` already resolve
  their own mesh and pull `get_express_kernel_defines`. (An earlier note here overstated this as a
  hard precondition on 3.3; it is not.)
- Guide line numbers are against `42204c541c3` and will drift once edits land. Each step names its
  enclosing symbol; prefer the symbol over the line number.
- If a step reveals that the plan or comparison is wrong, fix **that** document and note it here —
  do not let this tracker become the source of truth for design.

---

## 7. Deviations from the plan

Everything that was **not** in the plan but ended up needing to change, and why. Grouped by root
cause, because the causes repeat.

### A. The plan was wrong on the facts

| What changed | Why the plan got it wrong |
|---|---|
| **direction-table removal (6.1) abandoned** | The plan called `intra_mesh_direction_table` a redundant cache of the indexed first-hop peek. That is true on 2D and false on 1D — `TEST_F(Fabric1DFixture, TestGetNextHopRouterDirection1D)` exercises the API on a build with no indexed vectors, and the API has 14 external callers. One `routing_l1_info_t` serves both modes, so the field is permanent. |
| **L1 reclaim (6.2) is dead** | Follows directly from the above: nothing is freed. |
| **axis validation lift (6.3) premise was stale** | The plan said header reclaim (4.4) would ship a 72-byte tier covering `Y+X = 68`. It landed a 67 cap, so `[64,4]` is still blocked — by exactly one byte. |
| **header reclaim (4.4) retired one field, not two** | Plan decision D4 wanted both `routing_fields` and `is_mcast_active` gone (base 56 B, 40-byte tier). `routing_fields` is decoded by `tools/profiler/fabric_event_profiler.hpp`, which the owner ruled out of scope. Retiring `is_mcast_active` alone gives 60 + 36 = 96 exactly — a better outcome, but the plan's arithmetic assumed a change it was not allowed to make. |
| **Phase 4 had to interleave with Phase 5** | Real order is legacy producer deletion → **dispatch arm collapse** → edge-node router deletion → legacy L1 table deletion → header reclaim. Every Phase 4 deletion depends on the arm collapse. The phases were written by topic, not by dependency. |
| **X-ring hoist (1.4) folded into line-axis topology (1.5)** | The plan called the hoist behaviour-preserving. `derive_ordinary_ring_topology` fatals if *any* line lacks an edge, while `axis_wraps` checks only line 0 — so hoisting alone could newly `TT_FATAL`. |
| **Z-arm invariant (1.6) needed no code** | The plan specified an assert. The invariant is already enforced structurally by the dispatch arms' `if constexpr`; any assert would have been tautological. |
| **route-buffer sizing (1.8) deferred into the host gate flip** | The plan called it behaviour-preserving. It bumps `[32,4]` from 96 B to 112 B — a real performance change that had to land with the flip, not before it. |

### B. Found only by reading code the plan never opened

| What changed | Why it was missed |
|---|---|
| **`AxisRouteTopology::wraps` + the `next_row` guard** | The plan said "derive a line topology as a fallback" without noticing that `next_row`'s `% n` step is ring-only arithmetic. On a line it would route the "short way" over an edge the axis does not have. Found by reading `next_row`, not from the plan. |
| **`UPDATE_PKT_HDR_ON_RX_CH` removal + 2D no-op** | Turn-set deletion (5.1) listed the turn machinery but missed that the *shared* forward helper still executed `routing_fields.value + 1` on the indexed path — a live store per forwarded packet, violating the immutable-maps invariant. It sits one layer below the four sites the kernel flip touched. |
| **`fabric_edge_node_router.hpp` deleted** | The dispatch arm collapse orphaned it entirely; not anticipated. |
| **`MeshRoutingFields` aliases removed** | Dead-alias cleanup, unplanned. |
| **`shape_is_indexable()` extracted as a named predicate** | The plan said "add a bound check". Making it the *same* predicate the packer enforces means validation and packing cannot drift apart. |
| **`MAX_INDEXED_MESH_AXIS` split from `SLOT_SHAPE_*`** | The plan flagged blocker B1 but not the fix shape — one conflated constant had to become three. |

### C. Pre-existing bugs the plan did not know about

| What changed | How it surfaced |
|---|---|
| **U1: `MEM_TENSIX_ROUTING_TABLE_BASE` → `ROUTING_TABLE_BASE`** | UDM read the wrong L1 region on non-Tensix cores. Found while auditing UDM for the flip. |
| **U2: missing Z asserts in UDM** | Same audit. |
| **UDM + express compile-time bar** | `calculate_initial_direction` can legitimately return Z on an express mesh, and the mux fabric is cardinal-only with no Z mux. Owner call: keep express out of UDM for now. |

### D. My errors, caught by the compiler or by sweeping

| What changed | What I got wrong |
|---|---|
| **Top header tier 68 → 67** | I missed that the header is `packed, aligned(16)`, so a 67-byte buffer *already* yields a 128 B header; 68 tripped the pre-existing `RouteBufferSize <= 67` L1 bound (issue #32237). Caught by the first build. |
| **`-Wunused-variable` on `topology`** | Orphaned when the `UPDATE_PKT_HDR_ON_RX_CH` derivation was removed. Caught by the first build. |
| **Guarding the test kernel's reference encoder** | C1 was scoped as affecting only the two 2D tests. But `fabric_set_route` was declared *outside* any `FABRIC_2D` guard and the test kernel's encoder is unguarded, so deleting it broke **live 1D T3K tests** for a function they never call. Found by sweeping deleted symbols after the build went green. |
| **`sources.cmake` not updated by the file rename** | I scoped the rename as "file + struct" and forgot the build system references files by path. |
| **Preprocessor nesting broken twice** | Block deletions dropped an `#endif`. Both caught by an `#if`-depth check against the pre-edit file, not by review. |
| **Client-side multicast injection count left express-gated** | **The ring-multicast regression.** `express_mcast_outgoing_directions()` in the test infra computes how many directions a multicast root must inject into, and returned `{}` for any non-express mesh — whereupon the caller falls back to exactly **one** direction. That was right only while non-express 2D used the legacy hop-map codec, where a single packet carrying n/s/e/w hop counts fanned out inside the router. After the flip every 2D mesh encodes multicast as a reverse tree, so a plain **ring** root has the same multi-output shape an express root does: forward and backward around the ring. One injection into a two-branch tree means half the ring never receives, and a sync waiting on the whole ring hangs with every endpoint at zero packets. Invisible on a Mesh, where a directional extent on a line genuinely has one output. Fixed by removing the gate and switching from `ring_for_direction()` (null on a non-express Y axis) to `axis_topology()`. **Root cause of the miss: my Phase 1 guard inventory swept `tt_metal/` only, so a codec-shaped gate living in test infra was never classified.** |
| **Router lost its transitive include of `tt_fabric_api.h`** | Edge-node router deletion (4.2) removed `fabric_edge_node_router.hpp`, which included `tt_fabric_api.h` — the router's **only** path to that header. The router still calls `fabric_set_indexed_intermesh_landing_route()` when re-encoding an intermesh packet at its landing mesh, and with no declaration in scope GCC rejected it under two-phase lookup ("no arguments that depend on a template parameter"). I had checked that the deleted header's *own* functions (`recompute_path`, `get_cmd_with_mesh_boundary_adjustment`) were unused — and they are — but not what it *re-exported*. Fixed by including `tt_fabric_api.h` directly, 2D-gated to match the sole call site, in the empty `#ifdef FABRIC_2D` the removal left behind. |
| **Shape-define fallback gated on `defined(FABRIC_2D)`** | **My 1D-scoping mistake, fourth occurrence.** The `FABRIC_2D_MESH_{Y,X}_SIZE` fallback `#define` was guarded by `defined(FABRIC_2D)`, but its *users* are plain function bodies that every includer parses. `cq_relay.hpp` selects the 2D path with `if constexpr (FABRIC_2D)` — a constexpr branch, **not** a preprocessor one — so the discarded arm is still semantically checked, and the `cq_dispatch` / `cq_prefetch` kernels (compiled **without** `-DFABRIC_2D`) tried to parse a call referencing macros that did not exist. Harmless in HEAD because `fabric_set_unicast_route` still had a legacy path needing no shape; after legacy producer deletion (4.1) it always reaches the indexed encoder. Fixed by ungating the fallback: the requirement is "does this TU parse the 2D helpers" (always), not "is this a 2D build". |
| **Over-deleted `direction_to_compact_index_map` and both `map_downstream_direction_to_compact_index` overloads** | The worst of these. Turn-set deletion (5.1) cut a *range* anchored on `TURN_STATUS_ARRAY_SIZE` through the text `get_sender_channel_turn_statuses();` — but that string next occurs ~45 lines **later** than I assumed, at the `sender_channels_turn_status` initializer. Everything in between went with it, including the compact-index map and its two accessors, which have nothing to do with turns and are still used by `get_downstream_edm_interface_index()`. Invisible to every host check (the router kernel is JIT-compiled at test time) and caught only by the first device run: 624 errors, all at `fabric_erisc_router.cpp:546` and `:553`. **Lesson: never delete by text-anchored range — delete by explicit start/end and verify what fell between.** |

### E. Scope the owner cut, or that I withdrew

| Dropped | Why |
|---|---|
| Renaming `FABRIC_EXPRESS_ENABLED` | Owner rejected it twice; the name is accurate for what survives. |
| Four-predicate `express_routing_enabled` split | My scope creep. |
| Emulator changes | False positive — a grep matched `__emule_fabric_set_route(` on a substring. The emulator never calls it and does not include the header. |
| Profiler changes | Owner ruled it out; this is why header reclaim (4.4) landed as one field instead of two. |
| **2D API surface (2.5)** collapsed to a no-op | Re-auditing `mesh/api.h` (38 entry points), `linear/api.h` and `udm/*` showed no public API needs a signature change. |

### The pattern worth correcting

Groups **A** and **D** share one root cause: I repeatedly scoped by *which code uses a thing* rather
than *which builds compile it*. That is what made me miss 1D three separate times — direction-table
removal, C1, and the initial `get_next_hop_router_direction` caller audit. When a symbol is not behind
`#if defined(FABRIC_2D)`, assume a 1D build compiles it until proven otherwise.
