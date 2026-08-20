# Codec Unification — Working Status

Living tracker. Edit in place as work lands; this is the one file expected to churn.

| | |
|---|---|
| branch | `nnyamagoudar/subtorus-routing-kernel-unification` |
| base | `42204c541c3` (`agupta/subtorus-routing`) |
| last updated | 2026-08-20 — initial, nothing implemented yet |
| current phase | **Phase 0** (not started) |

Reference docs (stable; update those, not this one, when the *plan* changes):
`GALAXY_CODEC_UNIFICATION_COMPARISON.md` · `GALAXY_CODEC_UNIFICATION_PLAN.md` ·
`GALAXY_CODEC_UNIFICATION_IMPLEMENTATION_GUIDE.md`

**Status vocabulary:** `todo` · `wip` · `done` · `blocked` · `deferred` · `n/a`

---

## 1. Phase board

Step numbers are 1:1 with the implementation guide. Do not renumber here — change the guide.

### Phase 0 — fix the live single-hop ABI mismatch *(standalone, ships alone)*

| step | what | status | notes |
|---|---|---|---|
| 0.1 | ⚠ route single-hop helpers to the indexed encoder (`tt_fabric_api.h:48`) | `todo` | |

| gate | condition | status |
|---|---|---|
| **Gate 0** | deepseek `all_gather` / `all_reduce` / `broadcast` / `reduce_to_one_b1` pass on express Galaxy; new device test covers N/S/E/W + Z single-hop | `todo` |

### Phase 1 — prepare *(behaviour-preserving)*

| step | what | status | notes |
|---|---|---|---|
| 1.1 | audit + record the express-guard inventory (9 sites: codec vs genuine express) | `todo` | no code change; `FABRIC_EXPRESS_ENABLED` keeps its name |
| 1.2 | ⚠ gate-separation test (built router config: speedy/trim off, counter credits on) | `todo` | must survive Phase 3 **unchanged** |
| 1.3 | remove the `X ≤ 4` bound → total-region bound (B1) | `todo` | |
| 1.4 | hoist X-ring derivation out of the express-only branch (B3) | `todo` | one-line, but B2 depends on it |
| 1.5 | ⚠ line-axis topology + `axis_topology()` accessor (B2, D5) | `todo` | highest-value step in the phase. Q3: rename file **and** struct to `axis_route_topology` / `AxisRouteTopology` |
| 1.6 | assert the Z-arm invariant (B7 half 1) | `todo` | static_assert only — the WH admission *shape* is Q6, decided at Gate 3 |
| 1.7 | one-feeder-per-row gate: warning → `TT_FATAL` (D6) | `todo` | Q4 decided: hard fatal. Error must name the row and both feeders |
| 1.8 | un-skip non-express meshes in route-buffer sizing (H5) | `todo` | record any tier bump per platform |

| gate | condition | status |
|---|---|---|
| **Gate 1** | no behaviour change anywhere; full 1D + 2D + express regression green | `todo` |
| | new host test: shape sweep (every 2D descriptor packs + fits + `Y+X` fits a tier) | `todo` |
| | new host test: axis-topology coverage (non-null for every 2D mesh × axis) | `todo` |
| | new host test: arborescence sweep (every shape × root × axis) | `todo` |
| | tier bumps from 1.8 recorded per platform | `todo` |

### Phase 2 — close the indexed-path gaps *(behaviour-preserving)*

| step | what | status | notes |
|---|---|---|---|
| 2.1 | verify indexed UDM producers (P7, P8) — L1 base agreement, Z-tolerant callers | `todo` | verification, likely no edit |
| 2.2 | make `fabric_multicast_source_inject_*` unconditional (P6); add the single-output-root assert | `todo` | Q1 decided: per-direction contract + assert |
| 2.3 | ⚠ audit `recompute_path` vs the indexed landing encoder (B5, R6) | `todo` | **write the equivalence table**; gates Phase 4.2 |
| 2.4 | add the missing device coverage | `todo` | see §4. Q1 decided, so target sets follow the per-direction contract |

| gate | condition | status |
|---|---|---|
| **Gate 2** | every capability the legacy path provides has a green indexed test, on express **and** non-express fixtures | `todo` |
| | 2.3 equivalence table written, every row resolved | `todo` |

### Phase 3 — flip *(flag day; one commit; delete nothing)*

| step | what | status | notes |
|---|---|---|---|
| 3.1 | **retire** the selector *as a codec gate* — 8 codec sites → `#if defined(FABRIC_2D)`; delete the router-side emission; rename only the shape macros | `todo` | Q2: delete, don't rename. The define itself survives for the Z-port site (3.5) |
| 3.2 | flip the four host gates (H1-H4) | `todo` | |
| 3.3 | `get_fabric_kernel_defines()` node-aware requirement (H6) | `todo` | ⚠ **hard precondition** for 3.1/3.2 — guarding on `FABRIC_2D` means every `FABRIC_2D` compile needs the shape defines, dispatch kernels included |
| 3.4 | flip the device producers | `todo` | |
| 3.5 | ⚠ leave the connection-manager capacity alone; split `get_express_kernel_defines` emission | `todo` | `routing_plane_connection_manager.hpp` needs **no edit** |
| 3.6 | flip the kernel (`if constexpr` → always-true; keep the dead arms) | `todo` | |

| gate | condition | status |
|---|---|---|
| **Gate 3** | WH Galaxy `[8,4]`, `[32,4]` | `todo` |
| | BH Galaxy `[8,4]`, `[32,4]`, `[4,4]` express | `todo` |
| | BH LB `[2,4]` | `todo` |
| | dual/quad galaxy `[8,8]`, `[8,16]`, `[16,8]` | `todo` |
| | N300 2x2 / p150_x4 `[2,2]` | `todo` |
| | **1D regression untouched** (any movement = a define leaked) | `todo` |
| | Galaxy `[32,4]` header size + bandwidth **measured and recorded** (expect 112 B, R2) | `todo` |

### Phase 4 — delete the legacy 2D codec, reclaim the header

| step | what | status | notes |
|---|---|---|---|
| 4.1 | delete the legacy 2D producers | `todo` | |
| 4.2 | delete `fabric_edge_node_router.hpp` | `blocked` | precondition: 2.3 resolved |
| 4.3 | delete the legacy L1 2D table | `todo` | verify `sizeof(routing_l1_info_t)` stays 2576 |
| 4.4 | ⚠ retire `routing_fields` + `is_mcast_active`; add the 40 B / 96 B tier | `todo` | 61→56 B base; this is what buys Galaxy back its 96 B header |

| gate | condition | status |
|---|---|---|
| **Gate 4** | Galaxy `[32,4]` back to a **96 B** header | `todo` |
| | bandwidth **at or above the Phase-0 baseline** (not merely above Phase 3) | `todo` |
| | zero occurrences of `hop_index` / `branch_east_offset` / `branch_west_offset` / `turn_point` / `is_mcast_active` / `compressed_route_2d_t` in any 2D path | `todo` |
| | R8: `get_udm_header_size` still derives from `get_2d_header_size` | `todo` |

### Phase 5 — kernel simplification

| step | what | status | notes |
|---|---|---|---|
| 5.1 | delete the turn/header-mutation set (kernel §3.9) | `todo` | 1D overloads stay |
| 5.2 | `port_direction_table` | `deferred` | dead-code discussion, not this project |
| 5.3 | collapse the `if constexpr (express_enabled)` arms; delete the constexpr | `blocked` | needs Q6 — this is where the legacy admit (and WH's compass-only exception) dies |
| 5.4 | rename `express_*` → `indexed_*` in the kernel | `todo` | |

| gate | condition | status |
|---|---|---|
| **Gate 5** | ERISC binary size at or below Phase 3 | `todo` |
| | no `express` identifier in the kernel that does not mean Z chords | `todo` |
| | 1D untouched | `todo` |

### Phase 6 — L1 cleanup

| step | what | status | notes |
|---|---|---|---|
| 6.1 | remove `intra_mesh_direction_table`; repoint `get_next_hop_router_direction` (B8) | `todo` | share the peek with `calculate_initial_direction` |
| 6.2 | reclaim the freed 96 B; re-derive offsets | `todo` | decide: shrink to 2480 B or bank for `[64,4]` |
| 6.3 | lift the 32-per-axis validation (`control_plane.cpp:1966`) | `todo` | removes a `[64,4]` blocker, does not enable it |

| gate | condition | status |
|---|---|---|
| **Gate 6** | `sizeof(routing_l1_info_t)` asserted and documented | `todo` |
| | `[64,4]` either fits or its exact shortfall is recorded | `todo` |

---

## 2. Decisions

Plan §9. A `blocked` step above cannot start until its decision is `decided`.

| # | question | recommendation | state | blocks |
|---|---|---|---|---|
| Q1 | ~~Multicast client contract~~ | **adopt the per-direction contract + add the single-output-root assert.** Indexed multicast is a drop-in on chordless meshes; zero client migrations | `decided` 2026-08-20 | — |
| Q2 | ~~Rename `FABRIC_EXPRESS_ENABLED`~~ | **delete, don't rename** — redundant with `FABRIC_2D` after the flip | `decided` 2026-08-20 | — |
| Q3 | ~~`derive_line_axis_topology` placement~~ | **rename file *and* struct**: `express_ring_topology.{hpp,cpp}` → `axis_route_topology.{hpp,cpp}`, `ExpressRingTopology` → `AxisRouteTopology` | `decided` 2026-08-20 | — |
| Q4 | ~~Arborescence gate: fatal or opt-out?~~ | **hard `TT_FATAL`.** Structurally a no-op on chordless meshes (one feeder per row always), and express multicast works today so the check already passes there — a fatal cannot break anything functioning | `decided` 2026-08-20 | — |
| Q5 | **80 B tier** — after 4.4 the base is 56 B, so 80 B would carry 24 route bytes (covers `[8,8]`, `[16,8]`, `[8,16]`, `[16,4]`). Re-measure or leave disabled? | leave disabled unless measured | `deferred to Gate 4` | — |
| Q6 | **WH+VC1 admission shape** (B7b) — add a WH bit-test variant of `admit_express_dispatch`, or accept the dense 16-arm key on WH? | measure at Gate 3 first; the Z arms already compile out, so the dense key may cost less than kernel §3.3 assumed | `deferred to Gate 3` | 5.3 |

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
| B1 | `MAX_INDEXED_MESH_X = 4` excludes `[8,8]`, `[8,16]`, `[16,8]`, `[1,16]` | 1.3 | `open` |
| B2 | no reverse-tree source on a LINE axis → **silent multicast no-op** | 1.5 | `open` |
| B3 | `x_rings_` derived only for express meshes | 1.4 | `open` |
| B4 | `fabric_set_single_hop_unicast_route*` writes the legacy hop byte with no express gate — **live defect** | 0.1 | `open` |
| B5 | `called_from_router` 2D encode still legacy under express (dead, but unexercised) | 2.3 | `open` |
| B6 | Galaxy `[32,4]` needs 36 route bytes, one past the 35 B tier | 4.4 | `open` |
| B7a | WH router must not instantiate a Z dispatch arm | 1.6 | `open` — already true structurally; needs a tripwire |
| B7b | WH+VC1 compass-only admission lives only in the legacy admit Phase 4 deletes; WH first reaches the indexed path at Phase 3 | Q6 | `open` |
| B8 | `intra_mesh_direction_table` is the only source for same-mesh `get_next_hop_router_direction` | 6.1 | `open` |
| D9.1 | zeroed reverse-tree region → multicast silently delivers to nothing, no assert | 1.7 + validity check | `open` |
| D9.3 | same-mesh mcast ignores the caller's anchor (`root_*` from `my_mesh_coord_*`) | 2.2 (add assert) | `open` |
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
| `sizeof(routing_l1_info_t)` after Phase 6 | 2576 or 2480 | — | — |

---

## 6. Log

Newest first. One line per landed change or resolved decision; longer notes below the line.

| date | phase/step | what |
|---|---|---|
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

*(free-form: surprises, reverts, things the guide got wrong, findings that should move into the
reference docs)*

- **Shape defines must reach every `FABRIC_2D` compile** (from Q2). Once `#if defined(FABRIC_2D)`
  guards the read of `FABRIC_2D_MESH_*_SIZE`, the dispatch kernels are in scope too — they set
  `FABRIC_2D` at `dispatch.cpp:585` / `prefetch.cpp:549`, and `cq_relay.hpp` calls
  `fabric_set_unicast_route` after step 3.4. Either resolve 3.3 so the shape reaches them, or keep
  the zero-fallback at `tt_fabric_api.h:151-156` for compiles with no single mesh shape. Decide
  before 3.2.
- Guide line numbers are against `42204c541c3` and will drift once edits land. Each step names its
  enclosing symbol; prefer the symbol over the line number.
- If a step reveals that the plan or comparison is wrong, fix **that** document and note it here —
  do not let this tracker become the source of truth for design.
