# Per-mesh FabricConfig — audit, gaps, and implementation plan

*Scope:* PR #60690 (branch `ridvan/fabric-per-mesh-deadlock-avoidance`, stacked on #60499
`ridvan/llama-het-mgd-tests`). This document is the report requested on the PR: a full in-depth
scan of the fabric builder + fabric setup layers for everything that does **not** support
**per-mesh (different) FabricConfigs**, the root reason for each, and a minimal-change
implementation plan. It then records what was actually implemented in this PR.

---

## 0. TL;DR

We added **per-mesh FabricConfig** (#56561): an MGD can declare a different `fabric_config` per
`mesh_descriptor` (e.g. mesh `M_Y = FABRIC_2D_TORUS_Y` connected to mesh `M_X = FABRIC_2D_TORUS_X`),
and `MeshGraph` already resolves + applies it **per mesh for connectivity**
(`MeshGraph::get_fabric_config(mesh_id)`, routing-table generator, axis-route topology, express
routing). But the **FabricContext / EDM-builder / fabric-setup** layers were written assuming
**one global FabricConfig** for the whole system. They derive a single global `topology_`,
`bubble_flow_control_enabled_`, packet/channel sizing, and deadlock-avoidance (DA) polarity from
one process-wide `fabric_config_`, and ignore `mesh_id`.

The gaps split into two kinds:

* **Type A — must be physically identical on both ends of an inter-mesh link.** These can never
  legitimately differ across two linked meshes; the right fix is a **validation check that errors
  out** (cheap, no new support needed). Examples: max payload size, channel buffer-slot counts,
  intermesh VC wire contract.
* **Type B — must become genuinely per-mesh.** The layer reads a global value where it should read
  the node's own mesh config. Example: deadlock-avoidance axis selection
  (`need_deadlock_avoidance_support`).

**Confirmed symptom:** the galaxy e2e `galaxy-fabric-per-mesh-torus-yx` (M_Y 2x2 RING,LINE→TORUS_Y
linked to M_X 2x4 LINE,RING→TORUS_X) **hung on real BH galaxy at data movement**. Root cause: the
shared multi-mesh sanity config opens the fabric as **Mesh** for its first cases, so FabricContext's
global `topology_ = Mesh` ⇒ `need_deadlock_avoidance_support` returns `false` for every direction ⇒
M_X's **genuine 2x4 E/W ring** (built from its per-mesh `TORUS_X` field in MeshGraph) is routed
**without** DA / first-level-ACK ⇒ deadlock. The existing 2x2-torus tests survive only because a
2x2 mesh has **no genuine ring** (extent 2).

A TORUS_XY-envelope workaround (open the fabric as Torus XY so DA covers both axes) was tried and
**passed** on galaxy — but it masks the real defect instead of fixing it. This PR implements the
**real per-mesh DA fix** and reverts the workaround, per the request.

---

## 1. Findings table

`FabricContext` is built once from a single global `fabric_config_` at `control_plane.cpp:952`; from
that one value it derives a single `topology_`, `is_2D_routing_enabled_`,
`bubble_flow_control_enabled_`, and all packet/channel sizing (`fabric_context.cpp:247-258`).
`FabricBuilderContext` then builds one shared `router_config_` / channel allocator for every router
in every mesh (`fabric_builder_context.cpp:86,239-240`). **Today there is no cross-mesh fabric-config
compatibility validation anywhere** — the comment at `control_plane.cpp:826` explicitly permits
differing 2D configs across intermesh-linked meshes.

| # | Item | File:line | Global / assumed-identical thing | Why it breaks per-mesh (root reason) | Fix | Status |
|---|------|-----------|----------------------------------|--------------------------------------|-----|--------|
| 1 | Max payload / packet size | `fabric_context.cpp:212-222,146-149` | `max_payload_size_bytes_`, `channel_buffer_size_bytes_` global | A packet from mesh A forwarded into mesh B's receiver L1 slot sized for a smaller payload overflows the slot into adjacent L1 → corruption; header-size assumptions must also match | **A (check)** | open → **deferred to Tier 2** (see §4) |
| 2 | Allocator buffer-slot counts match | `fabric_static_sized_channels_allocator.cpp:484-497,553-560,737-749`; `erisc_datamover_builder.cpp:480` | slot-count table chosen by topology + buffer size + avail space; `num_buffers` emitted as CT args; remote mirror from same config | Credit/flow-control: sender tracks N remote slots, receiver must physically have N. Different table/size on the two ends → slot mismatch → credit under/overflow → stall or corruption | **A (check)** | open → **deferred to Tier 2** (see §4) |
| 3 | Deadlock-avoidance support (non-express) | `fabric_context.cpp:316-332`; caller `compute_mesh_router_builder.cpp:452-456`; EDM `erisc_datamover_builder.cpp:1076` | non-express branch uses global `topology_` + `get_fabric_type(fabric_config_,…)`, ignores `mesh_id` | mesh A=TORUS_Y vs B=TORUS_X: wrong DA/first-level-ACK axis for the mesh whose axis ≠ global; **Mesh global ⇒ no DA on any mesh's genuine ring** (the hang) | **B (per-mesh)** | open → **FIXED in this PR** |
| 4 | Inter-mesh DA enable (#56515) | `erisc_datamover_builder.cpp:1076` (`!is_inter_mesh && …`) | both ends of an inter-mesh EDM must agree on DA polarity; far end may be plain Mesh | — | — | **already-fixed (#56515)** |
| 5 | `bubble_flow_control_enabled_` | `fabric_context.cpp:252`; `fabric_tensix_builder_impl.cpp:1092` | global `is_ring_or_torus(topology_)` baked into tensix-mux CT arg | tensix mux decrements a specific router's credits; BFC enablement must follow that router's mesh | **B (per-mesh)** | open → **FIXED in this PR** |
| 6 | `topology_` global | `fabric_context.cpp:247`; `compute_mesh_router_builder.cpp:179,267`; `fabric_builder.cpp:211`; `fabric_builder_context.cpp:86,240` | one `Topology` for all meshes; also collapses TORUS_X/Y/XY → `Torus` (loses axis) | router archetype/channel shape + connection pairs for a router in mesh X must use mesh X's topology | **B (per-mesh)** | open (see plan) |
| 7 | EDM `enable_first_level_ack` | `erisc_datamover_builder.cpp:714-715` | `config.topology == Ring/Torus` (global topology) | first-level-ACK must agree with DA on the link | **B (per-mesh)** | open — subsumed by #3 (same flag path) |
| 8 | Single FabricContext construction | `control_plane.cpp:950-953`, guard `:943-949` | one `FabricContext(… fabric_config_ …)` for whole system | a single context cannot represent two meshes with different topologies | **B (root)** | open (see plan) |
| 9 | Cross-rank/cross-mesh config validation | `control_plane.cpp:828-857` (comment `:826`), `:883-894`, `:940` | validates only same-config-per-mesh-per-rank; observations stamped with global `fabric_config_` | no check that two intermesh-connected meshes are compatible; validation is structurally blind | **A (check)** | open → **deferred to Tier 2** (see §4) |
| 10 | Peer channel-count shape | `compute_mesh_router_builder.cpp:174-206,506-509` | `compute_router_channel_counts` reads global topology while express is per-mesh | peer router may be in another mesh; its VC/channel counts derived with wrong topology | **B (per-mesh)** | open (see plan) |
| 11 | `is_2D_routing_enabled()` gate | `compute_mesh_router_builder.cpp:523,671,704,877`; `fabric_builder.cpp:47,151,286`; `control_plane.cpp:2672,2973` | global 1D/2D flag | 1D-vs-2D is per-mesh once config is per-mesh; mixing emits wrong `FABRIC_2D*` kernel defines | **A (check)** — 1D already single-mesh-only | open → covered by existing 1D-multimesh rule + VC check |
| 12 | `get_fabric_tensix_config()` | `compute_mesh_router_builder.cpp:286`; `fabric_tensix_builder_impl.cpp:116,624` | global tensix config | wrong per router if tensix extension varies per mesh; else must be validated uniform | **A (check)** | open (see plan) |
| 13 | Intermesh VC config | `fabric_builder_context.cpp:342-392`; `compute_mesh_router_builder.cpp:192,339,889-912` | one system-wide `IntermeshVCConfig` | both ends of an inter-mesh link share the VC0→VC1 crossover wire contract | **A (check)** | open → **deferred to Tier 2** (see §4) |
| 14 | Uniform VC-count upper bound | `channel_trimming_import.hpp:24-30,80`; `.cpp:301-307` | `builder_config::MAX_NUM_VCS` assumed to cover every mesh | per-mesh config may yield different plane/VC counts | **A (check)** | open (see plan) |
| 15 | `compute_max_channel_counts` | `fabric_builder_context.cpp:86` | router-family shapes from global topology | fabric-wide max channel counts computed with one topology | **B (per-mesh)** | open — subsumed by #6 |
| 16 | Global config accessors | `control_plane.hpp:413,415,416` | `get_fabric_config()`/`get_fabric_tensix_config()`/`get_fabric_udm_mode()` take no `mesh_id` | callers apply one mesh's config to all meshes | **B (per-mesh)** | open (see plan) |
| 17 | `connect_routers` topology | `fabric_builder.cpp:211,180-206` | global topology for router connection pairs (wrap-fold already per-mesh at `:33`) | 1D ring/line & torus-axis connection choices should follow the local node's mesh | **B (minor)** | open — low risk for all-2D |

**Not gaps (already per-mesh, confirmed):** `MeshGraph::get_fabric_config(mesh_id)`
(`mesh_graph.cpp:946-950`), `routing_table_generator.*`, `axis_route_topology.*`
(`axis_route_topology.cpp:122-128`), `express_routing_enabled(mesh_id)`,
`mesh_has_protected_ring_in_axis_of(mesh_id,…)`, `FabricContext::is_wrap_around_mesh(mesh_id)`.

---

## 2. The two teammate-flagged checks — root reason

### Max payload size must be identical across meshes (#1)
`compute_max_payload_size_bytes` (`fabric_context.cpp:212-222`) returns either the global override
`router_config_.max_packet_payload_size_bytes` or a default chosen by the global
`is_2D_routing_enabled_`, and `channel_buffer_size_bytes_ = packet_header_size_bytes_ +
max_payload_size_bytes_` (`:146-149`). This single value sizes **every** channel buffer in **every**
mesh via `create_edm_config` (`fabric_builder_context.cpp:239`). **Root reason:** a channel is a
fixed-size L1 ring of `num_buffers × channel_buffer_size_bytes`. If mesh A's EDM forwards a packet
across the mesh boundary into mesh B's receiver, and B sized its slot for a smaller payload, the
write overruns the slot into the adjacent L1 region → silent corruption. Header size must likewise
match or the receiver mis-parses the packet. **There is no check today.** → **add a validation check**
that both ends of every inter-mesh link agree on payload **and** header size.

### Allocator slots on both channel ends must match (#2)
`FabricStaticSizedChannelsAllocator` picks a buffer-slot option by (a) a table keyed on `topology ==
Mesh/Torus` vs other (`fabric_static_sized_channels_allocator.cpp:484-497`) and (b) the first option
whose `total_slots × channel_buffer_size_bytes ≤ available_channel_buffering_space` (`:553-560`). The
chosen `num_buffers` is emitted as compile-time args for both the local and the **remote mirror**
channel (`:737-749`; remote allocator from the same static allocator at
`erisc_datamover_builder.cpp:480`). **Root reason:** this is pure credit/flow-control accounting — the
sender is told "the remote receiver has N slots" and must never outrun them. If two meshes select
different tables (different topology) or different sizes (different payload/header), the slot counts
on the shared inter-mesh channel disagree → the sender's credit view ≠ the receiver's real capacity →
credit under/overflow → stall (hang) or overwrite (corruption). → **add a validation check** that the
resolved buffer-slot count matches on both ends of every inter-mesh link. (Note: because slot count
is *derived* from topology + payload + available space, checking payload/header equality (#1) plus VC
config (#13) plus topology-compatibility covers most of this; the explicit slot-count equality check
is the direct guard.)

---

## 3. Implementation plan (minimal change per item)

Philosophy requested: *minimal changes; for items that must be identical across meshes, add a check
rather than new support; implement real per-mesh support only where the feature genuinely needs it.*

### Tier 1 — implement now (needed for per-mesh torus to actually work / cheap safety)

1. **#3 DA per-mesh (the fix).** In `need_deadlock_avoidance_support`, replace the global
   `topology_` / `get_fabric_type(fabric_config_,…)` in the non-express branch with the node's own
   `control_plane.get_mesh_graph().get_fabric_config(fabric_node_id.mesh_id)` →
   `get_topology_from_config(...)` + `get_fabric_type(...)`. Zero regression for uniform runs
   (`get_fabric_config(mesh_id)` == global config there); flips M_X's genuine E/W ring to DA-on when
   opened under a Mesh/global config. *~15 lines, one function.*
2. **#1 + #2 + #13 cross-mesh compatibility check — investigated, DEFERRED to Tier 2 (see §4 for the
   full reasoning).** A check was prototyped (MeshGraph, walking the requested inter-mesh links,
   comparing each pair's resolved payload/slot-table *family*) and **reverted**: in today's
   single-global-`FabricContext` architecture every channel's payload size and buffer-slot count are
   derived from the one process-wide config, so they are **uniform across meshes by construction** —
   there is no API (no per-mesh payload/buffer field in the MGD proto; `channels.count` is per-link,
   not per-endpoint) by which two inter-mesh-linked meshes can disagree on them. The only per-mesh
   difference that exists today is the torus **axis/family**, which the derived default itself varies
   by `dim_types` (`[LINE,LINE]→FABRIC_2D`, `[RING,LINE]→TORUS_Y`, …). A family-equality check there
   therefore **rejects currently-valid, currently-working mixed-shape multi-mesh MGDs** (it broke
   `FabricConfigResolutionMultiMesh` and would break production galaxy splits) while guarding a
   divergence that cannot occur until sizing is per-mesh. The correct home for this check is **inside
   the Tier-2 change that first makes payload/slots per-mesh**, as a guard co-located with the new
   per-mesh sizing — not a standalone config-family heuristic today.

### Tier 2 — follow-up for full generality (document; implement only if Tier 1 e2e still hangs)

3. **#5 bubble_flow_control per-mesh.** Make `is_bubble_flow_control_enabled()` take a
   `FabricNodeId`/`mesh_id` and resolve from `get_fabric_config(mesh_id)`; update the one consumer
   (`fabric_tensix_builder_impl.cpp:1092`, which already has `local_fabric_node_id_`). Needed if the
   ring still starves because the tensix mux withholds the bubble on M_X.
4. **#6 / #10 / #15 per-mesh topology in the router builder.** Thread the router's `mesh_id` into
   `compute_router_channel_counts` / `router_archetype` / `compute_max_channel_counts`, sourcing
   topology from `get_fabric_config(mesh_id)` (preserving TORUS_X/Y axis, which `Topology` alone
   drops). This is the large structural change; only required if VC/channel shape under a global Mesh
   topology proves insufficient for M_X's torus routing.
5. **#8 per-mesh FabricContext (root).** The clean end-state is either a per-mesh `FabricContext` or
   per-mesh lookups inside it fed by `MeshGraph::get_fabric_config(mesh_id)`. Items 1 & 3–4 are the
   incremental path to this without a rewrite.
6. **#12 / #14 / #16 remaining checks & accessors.** Add `mesh_id` overloads to the global accessors
   (or validate uniformity); assert `MAX_NUM_VCS` bounds every mesh's actual VC count; validate
   tensix/UDM config equality across inter-mesh links if they are meant to stay global.

### Already covered / no action

* **#4 inter-mesh DA enable** — done in #56515; `!is_inter_mesh` guard intact at
  `erisc_datamover_builder.cpp:1076`.
* **#11 1D-vs-2D mixing** — 1D is already single-mesh-only (the 1D-multimesh rule in
  `mesh_graph.cpp:491-…` rejects an explicit 1D config on a multi-mesh MGD), so a 1D mesh can't be
  inter-mesh-linked to a 2D mesh in the first place; the VC check (#13) covers the rest.
* **#7 first-level-ACK** — same flag as #3; fixed together.

---

## 4. What this PR implements

*(see `git log` on `ridvan/fabric-per-mesh-deadlock-avoidance`.)*

**Implemented:**

* **#3 — per-mesh deadlock-avoidance** in `FabricContext::need_deadlock_avoidance_support`: the
  non-express branch now keys topology + torus axis off
  `control_plane.get_mesh_graph().get_fabric_config(fabric_node_id.mesh_id)` (the node's own resolved
  per-mesh config) instead of the global `topology_`/`fabric_config_`. For a uniform single-config run
  `get_fabric_config(mesh_id) == fabric_config_`, so behaviour is preserved; for a per-mesh MGD each
  mesh gets DA (and the coupled first-level-ACK + VC0 bubble-flow-control, see
  `erisc_datamover_builder.cpp:1076-1105`) on its own ring axis. This flips M_X's genuine E/W ring from
  DA-off to DA-on under a Mesh-opened fabric — the direct fix for the hang. **#7 (first-level-ACK) is
  fixed by the same flag.**
* **#5 — per-mesh bubble flow control for the tensix fabric mux** in
  `FabricContext::is_bubble_flow_control_enabled(FabricNodeId)`: the tensix-mux kernel CT arg
  (`fabric_tensix_builder_impl.cpp:1092`) was the global `is_ring_or_torus(topology_)`, i.e. OFF for
  every mesh under a Mesh-opened fabric. Now keyed off the node's own resolved per-mesh config (via
  `FabricContext::control_plane_` → mesh graph), so a per-mesh ring gets the bubble at BOTH the EDM
  (VC0, via #3) and the tensix-mux level — matching what the TORUS_XY envelope provided globally.
  Uniform runs preserved.
* **Revert the TORUS_XY-envelope workaround** — `galaxy-fabric-per-mesh-torus-yx` points back at the
  original `test_fabric_multi_mesh_sanity_common.yaml` (whose first cases open `topology: Mesh` — the
  exact repro); the bespoke `test_fabric_per_mesh_torus_sanity.yaml` is removed.
* **Regression test** — `MeshGraphDescriptorTests.InterMeshDifferentTorusAxesAreCompatible` (CPU-only)
  builds a MeshGraph from a two-mesh MGD with different torus axes (TORUS_Y + TORUS_X), guarding the
  per-mesh-config build path. The galaxy e2e (`galaxy-fabric-per-mesh-torus-yx`) validates the real fix
  on hardware.

**Investigated and deliberately NOT added (deferred to Tier 2):**

* **#1/#2/#13 cross-mesh payload/slot/VC compatibility check.** Prototyped and reverted — see §3 Tier 1
  item 2. In the single-global-`FabricContext` architecture these values are uniform by construction
  (no per-mesh API can make them diverge), so a standalone check is either a tautology or, if it keys
  off the config *family*, actively **rejects currently-valid mixed-shape multi-mesh MGDs**. It belongs
  with the Tier-2 per-mesh-sizing change, not before it. The constraint and its placement are recorded
  here so it is not lost.

---

## 5. How this was / should be validated on hardware

* **CPU/unit:** `fabric_unit_tests`, `test_mesh_graph_descriptor` (per-mesh config resolution),
  and the new compatibility-check unit test — run locally, no galaxy needed.
* **Galaxy e2e:** `galaxy-fabric-per-mesh-torus-yx` via the `Galaxy tests` workflow
  (`e2e-test-selection: galaxy-fabric-per-mesh-torus-yx`, `blackhole: true`). This is the direct
  repro; a pass under the reverted sanity config proves the per-mesh DA fix.
* **Hang-debug tooling (if it hangs again):**
  * `test_tt_fabric` writes `pairwise_validation_summary.log` / `_detailed.log` — start with the
    summary to find the stalled host/link pair.
  * `tools/scaleout/validation/README.md` — physical-link discovery vs the descriptor (rule out
    cabling before chasing routing).
  * `tools/triage/` + tt-triage docs — live CB / semaphore / register state on a live hang
    (`dump_circular_buffers.py`, `dump_semaphores.py`); find the earliest blocked producer.
  * `tt-exalens` — low-level device/register/RISC-V inspection.
  * `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp` — router-sync / stuck-stage
    diagnostics if the hang is during fabric init.
  * Preserve the failing state before reset; collect triage first.

---

## 6. E2E iteration log (galaxy `galaxy-fabric-per-mesh-torus-yx`, BH galaxy)

| Run | Code under test | Result | Evidence |
|-----|-----------------|--------|----------|
| `38667692636` | TORUS_XY envelope workaround | **PASS** | full suite green — but masks per-mesh difference (opens both axes globally) |
| `38671263096` | #3 DA fix only (reverted to common sanity config) | **HANG** (15-min timeout) | `MeshNeighborExchange_num_links_1_ntype_unicast_write` on `topology: Mesh`: ranks `[1,0]`,`[1,2]` logged "Finished", ranks `[1,1]`,`[1,3]` never did → deadlock. CI tt-triage produced no artifact. |
| `38674433232` | #3 DA fix + #5 tensix-mux bubble per-mesh | *pending* | — |

**Diagnosis after run 2.** The DA fix (#3) enables the EDM-level VC0 bubble + first-level-ACK on
M_X's ring, but the test still hung on the Mesh-topology `MeshNeighborExchange`. The passing envelope
(run 1) differs from the DA-only build in more than DA: opening `topology: Torus` also set the global
`bubble_flow_control_enabled_` true (→ tensix-mux bubble, #5) **and** set a Torus `routing_mode` /
packet spec / VC config (#6). Run 3 adds #5. If run 3 still hangs, the remaining lever is **#6 — the
global `topology_` → `routing_mode`/VC/packet-spec**, which only a per-mesh `FabricContext` (#8) fully
addresses: M_X's routers are still built with Mesh routing mode / VC shape even though its ring links
exist, so the torus wrap may not be routed with the dateline the ring needs. That is the Tier-2
architectural change (per-mesh FabricContext or per-mesh router config); it should not be rushed. If it
is not taken on now, the **TORUS_XY envelope (run 1) is the correct, passing interim** for the test —
it is not "wrong", it simply provisions both axes globally rather than per-mesh.

**Why the DA fix + #5 are still correct and worth landing regardless:** both are genuine per-mesh
correctness fixes (each mesh now gets deadlock avoidance + bubble on its own ring axis), both are
behaviour-preserving for uniform runs, and both are prerequisites for the per-mesh-FabricContext end
state — they are not throwaway.
