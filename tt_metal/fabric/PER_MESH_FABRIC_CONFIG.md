# Per-mesh FabricConfig — audit, gaps, and implementation plan

*Scope:* PR #60690 (branch `ridvan/fabric-per-mesh-deadlock-avoidance`, stacked on #60499
`ridvan/llama-het-mgd-tests`). This document is the report requested on the PR: a full in-depth
scan of the fabric builder + fabric setup layers for everything that does **not** support
**per-mesh (different) FabricConfigs**, the root reason for each, and a minimal-change
implementation plan. It then records what was actually implemented in this PR.

---

## ⭐ RESOLUTION (read this first — supersedes the per-mesh-`FabricContext` / #8 framing below)

The deep audit below treated the fabric as one process that must represent several meshes at once,
which led to the conclusion that a per-mesh `FabricContext` (#8) was required. **That premise is wrong
for how fabric actually runs.** `tt-run` launches **one MPI rank (process) per mesh**, each with its own
`TT_MESH_ID` (`tools/scaleout/exabox/run_fabric_tests.sh:33`; rank-binding = `rank N → mesh_id N`), and
`ControlPlane::fabric_config_` is **per-process**. So **each fabric process owns exactly one mesh with one
config** — heterogeneity is *across processes*, which the architecture already supports. There is nothing
to make per-mesh *inside* a process.

The hang was simply that a rank built its `FabricContext` with the launch's process-wide override
(`FABRIC_2D`, from the test's `topology: Mesh`) instead of **its own mesh's** config: rank 1 owns the
`TORUS_X` mesh but built everything for `MESH`, so its genuine ring routed in `ROUTING_MODE_MESH` and
deadlocked.

**Fix (one method):** `ControlPlane::adopt_local_mesh_fabric_config()` sets this process's `fabric_config_`
to its local mesh's resolved config (`MeshGraph::get_fabric_config`: MGD-declared > override > default)
as soon as the local-mesh binding is known. Each process is then internally homogeneous, so the **entire
existing global-config pipeline is correct per rank with no per-mesh machinery** — `topology_`, the global
`ROUTING_MODE` kernel define, VC config, deadlock avoidance, bubble flow control and packet specs all
resolve to the rank's own mesh. No-op when the MGD declares no explicit per-mesh config (resolves back to
the override) or on the single-host multi-mesh *testing* path.

Consequently the earlier per-mesh lookups in `need_deadlock_avoidance_support` (#3) and
`is_bubble_flow_control_enabled` (#5) were **reverted** — they were compensating for the wrong global
config and are unnecessary once each rank adopts its own. The DA builder already keys off
`this->local_fabric_node_id` (this rank's node), so it already follows the per-process model.

The 17-gap audit below remains a valid catalogue of *global-config reads*, but under the per-rank model
the **type-B "make it per-mesh" items are not needed inside a process**; the only ones that still matter
are the **type-A cross-process constraints** at an inter-mesh link (payload size / slot count / VC
contract must match on both ends), which become a validation to add if/when per-mesh *sizing* is ever
introduced — today they are uniform because all meshes share packet geometry.

---

## ✅ RESULTS — fix, hardware validation, invariants, and robustness

### The fix that landed (minimal)
- **`ControlPlane::adopt_local_mesh_fabric_config`** (`control_plane.cpp`): each process (tt-run launches
  one MPI rank per mesh, each with its own `TT_MESH_ID`, so one process = one mesh) resolves its config
  from the MGD — `MeshGraph::get_fabric_config`: **MGD per-mesh `fabric_config` > SetFabricConfig override
  > dim_types default** — and **warns clearly if the config was set in two places** (MGD wins).
- **`MetalEnvImpl::get_fabric_config()` delegates to the control plane's resolved config**
  (`metal_env.cpp`): exactly **one fabric config per process**, pulled from one place from the start — no
  partial-init-with-the-override-then-overridden. Every consumer (firmware initializer, device setup,
  test harness) sees the same resolved config.
- Exploratory per-mesh lookups in `need_deadlock_avoidance_support` (#3) / `is_bubble_flow_control_enabled`
  (#5) and the global-topology upgrade (#6-interim) were **reverted** — unnecessary once each process
  builds for its own resolved config.

### Root cause of the original hang (and why the delegate fix)
The test opened `topology: Mesh` (→ `SetFabricConfig(FABRIC_2D)`); `adopt` switched the
control-plane/routers to the MGD's `TORUS_X`, but the **env/firmware kept `FABRIC_2D`** → the device
fabric was brought up Mesh-flavored under torus routers → M_X's genuine ring deadlocked. Proven on a
single mesh: **direct `topology: TorusX` passes, `topology: Mesh`+adopt hangs** — the only difference was
the env config value. The delegate fix makes the whole process consistent.

### Hardware validation (BH galaxy `bh-glx-110`, `test_tt_fabric` neighbor-exchange, `TT_METAL_OPERATION_TIMEOUT_SECONDS=0`)
| Config | Opened as | Result |
|--------|-----------|--------|
| 3-mesh **M_Y=TORUS_Y + M_X=TORUS_X + M_Y=TORUS_Y** | `topology: Torus` | ✅ intra+inter, 2 executed 0 failed (all ranks), exit 0 |
| 4-config **TORUS_XY + TORUS_X + TORUS_Y + FABRIC_2D** (incl. **mesh↔torus** boundaries) | `topology: Torus` | ✅ 2 executed 0 failed (all 4 ranks), exit 0 |
| **2×4 mesh split across 2 processes** (TORUS_X) + TORUS_Y mesh | `topology: Torus` | ✅ 2 executed 0 failed, exit 0 |
| single M_X 2×4 TORUS_X, **direct** `topology: TorusX` | TORUS_X | ✅ pass |
| single M_X 2×4 TORUS_X, `topology: Mesh`+adopt (pre-delegate-fix) | Mesh→adopt | ❌ hung (the bug) |

The warning fires as designed, e.g. *"Fabric config set in two places: SetFabricConfig requested
FABRIC_2D_TORUS_XY, but the MGD declares FABRIC_2D_TORUS_Y for mesh 2 … MGD takes precedence."*

### The `topology: Mesh` sanity cases are a TEST-HARNESS mismatch, not a fabric bug
`test_tt_fabric` computes `is_torus_2d_unicast` from the **yaml** `topology` (`tt_fabric_test_device_setup.cpp:482`),
so `topology: Mesh` routes packets **mesh-style into a torus fabric** → deadlock. A per-mesh-**torus** MGD
must be exercised with **torus** test cases (which pass and genuinely test per-mesh configs). This is why
the original `galaxy-fabric-per-mesh-torus-yx` on the common (Mesh-first) sanity config hung.

### Why heterogeneous 2D combos are robust — the boundary deep-dive
- **Buffer-slot counts** (`fabric_static_sized_channels_allocator.cpp:351+`): the slot-option tables are
  keyed by **(arch, VC-presence)**, NOT by mesh-vs-torus — all are generic-2D "mesh" tables. So
  **FABRIC_2D and torus get identical slot counts** given the same VC config / buffer size / arch → no
  slot mismatch across a mesh↔torus inter-mesh link. (This is why the teammate's #2 concern does not
  materialize for 2D.)
- **Max payload / packet-header size**: derived from `is_2D_routing_enabled` (true for *every* 2D config)
  + the process-global `max_packet_payload_size_bytes` override → **same for all 2D meshes** → matches
  across any 2D↔2D link by construction. (Differs only for 1D↔2D, which can't occur — see below.)
- **Inter-mesh EDMs disable deadlock avoidance** (#56515, `erisc_datamover_builder.cpp:1076`), decoupling
  cross-end credit coupling at the boundary.

### Invariants — what MUST be true for heterogeneous multi-mesh
1. **Inter-mesh-linked meshes must have the same routing dimensionality (2D).** All 2D configs
   (FABRIC_2D, TORUS_XY/X/Y) share payload/header/slot geometry → freely combinable across meshes.
2. **1D configs are single-mesh-only** — a 1D config has different packet geometry and cannot be
   inter-mesh-linked to a 2D one (and the existing 1D-multimesh rule in `mesh_graph.cpp` already forbids a
   1D config on an MGD with >1 compute mesh).
3. **Payload/header size and buffer-slot counts** must match across a link — satisfied for all 2D by
   construction (above).

### Implications — what can / cannot be done for multimesh
- ✅ **Any mix of the 2D configs** across meshes: FABRIC_2D, TORUS_XY, TORUS_X, TORUS_Y — proven.
- ✅ A **single mesh spanning multiple processes** (host_topology > 1×1) with per-mesh config differences
  (both ranks of the split mesh adopt the same config) — proven (2×4-split test).
- ❌ A **1D config in a multi-mesh fabric** — rejected.

### Validation added
- **`MeshGraph::validate_inter_mesh_fabric_config_compatibility()`** (`mesh_graph.cpp`): for every declared
  inter-mesh link, asserts the two meshes' resolved configs have the **same routing dimensionality**
  (errors out on a 1D↔2D link instead of corrupting; allows all 2D↔2D incl. mesh↔torus). CPU-level, runs
  at MGD build; no regression (74/74 MeshGraph tests pass).
- Plus the existing **1D-multimesh rule** and the new **"config set in two places" warning**.

### Inter-mesh mismatch audit — complete (no MGD-reachable mismatch)
A full audit of every per-process value that must agree across an inter-mesh link (18 values) concluded:
**no MGD per-mesh config combination can produce a divergent slot count, packet geometry, or credit
layout** across an inter-mesh link. The structural reasons:
- **Slot tables** (`fabric_static_sized_channels_allocator.cpp:484-497`) select the **same table for Mesh
  and Torus** (keyed only by arch + VC-presence), and whenever an inter-mesh link exists the **boundary
  router family (5 VC0 / 4 VC1) is enumerated into the channel maxima on both sides**
  (`fabric_builder_context.cpp:82-156`), pinning the slot counts identically regardless of
  express/Mesh/Torus. → **slots can't diverge**.
- **Packet header/payload** come from a **system-wide** action-map survey + `is_2D`/global override
  (`fabric_context.cpp:64-75,147-222`) → uniform.
- **IntermeshVCConfig** (`fabric_builder_context.cpp:342-393`) is a **system-wide** survey of the whole
  mesh graph → same on both ends (#13 is not per-mesh).
- **Inter-mesh EDMs force DA/first-level-ack off** (#56515, `erisc_datamover_builder.cpp:1075-1104,1174-1180`)
  → the cross-boundary credit accounting is **decoupled** (the deepest reason mismatches can't bite).
- **ROUTING_MODE** (#12) does differ per-mesh, but the on-wire header format is the uniform pre-baked 2D
  action map, so the decode is not sensitive to the TORUS/MESH bit; and under the per-process model each
  process owns one mesh, so this is correct (Tier-2 only if one process ever owns multiple topologies).

**The only way to force a mismatch is a non-uniform ENVIRONMENT across ranks (not reachable from the
MGD):** setting `enable_fabric_vc2`, `enable_fabric_mesh_pass_through`, or the `max_packet_payload_size`
override *differently per process* (these are read per-process from `rtoptions` and are **not**
cross-rank-validated the way `fabric_config` is). Each would diverge the VC/slot/buffer geometry → credit
stall. The launcher normally guarantees a uniform env, so this is a latent env-hygiene gap, not an MGD
risk.

**Env-parity guard — IMPLEMENTED (`control_plane.cpp`).** The cross-rank parity check at fabric bring-up
is now in place, folded into the existing `ControlPlane::validate_fabric_config_across_ranks` collective.
`MeshFabricConfigObservation` (the trivially-copyable struct already MPI-`all_gather`-ed per rank) carries
three extra process-wide knobs — `vc2_enabled` (`enable_fabric_vc2`), `mesh_pass_through_enabled`
(`enable_fabric_mesh_pass_through`), and `max_payload_override` (the `max_packet_payload_size` override,
`uint64::max` sentinel = default). `validate_fabric_configs` then `TT_FATAL`s unless all three are
**globally identical across every rank** (not just per-mesh — these are the knobs that select the per-VC
buffer-slot table and the channel buffer size, so any disagreement across an inter-mesh link diverges the
geometry and silently credit-stalls). This converts the only reachable exposure from a silent hang into a
clear startup error that throws **consistently on every rank** (so no partial-rank MPI deadlock). The
per-mesh *config* itself needs no further cross-mesh check beyond the routing-dimensionality guard above,
which is satisfied by all 2D combos.

**Hardware validation (BH galaxy, single-galaxy `single_galaxy_per_mesh_torus_yx` MGD, 3 processes):**
- **Non-uniform env** — `TT_METAL_ENABLE_FABRIC_VC2` set on even ranks only (rank 0 on, rank 1 off):
  all three ranks `TT_FATAL` at bring-up with the identical message
  *"Fabric channel-geometry settings must be identical on every rank, but rank 1 (mesh 1) reports
  (enable_fabric_vc2=false, …) while rank 0 (mesh 0) reports (true, …)"* — caught **before** any hang.
- **Uniform env** (control run, same MGD, env identical on all ranks): clean pass, `2 executed, 0 failed`
  on every rank, `tt-run` exit 0 — **no false positive**.
- Regression: full `fabric_unit_tests` mesh-graph/fabric-config suite green (77 passed; the 2 skips need
  hardware / ≥2 ranks).

A matching CPU/multi-rank gtest, `MultiHost.BHDualGalaxyNonUniformFabricEnvAcrossRanksFatal`
(`test_multi_host.cpp`), exercises the same path in CI alongside the existing
`BHDualGalaxyFabricConfigMismatchAcrossRanksFatal` (both skip below 2 ranks).

---

## 0. TL;DR (original framing — see RESOLUTION above for the correction)

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
| 6 | `topology_` / `routing_mode_` global | `fabric_context.cpp:247,255,365`; `compute_mesh_router_builder.cpp:179,267` | one `Topology`/`ROUTING_MODE` define for all meshes | a router in a per-mesh torus mesh runs `ROUTING_MODE_MESH` under a Mesh-opened fabric → no wrap/dateline routing → ring deadlocks | **B (per-mesh)** | open — interim torus-upgrade **tried & reverted** (insufficient, see §6); needs full #8 |
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

**Tried and reverted (did not work — kept here for the record):**

* **#6 interim — global topology→Torus upgrade when an MGD declares explicit per-mesh tori.** Upgraded
  `FabricContext::topology_` to the Torus superset (narrowed to explicit configs). Run 4 hung with the
  identical signature, so it was reverted. **Why it failed:** the passing envelope sets the *entire*
  global `fabric_config` to torus, which satisfies *every* per-mesh gap at once (routing mode **and**
  VC config #13, channel counts #10/#15, `is_2D` gates #11, peer-topology lookups, …). Flipping only
  `topology_`/`routing_mode_` fixes one of those, not all — so the Mesh-case still deadlocks. This is
  the evidence that the complete fix is the **full per-mesh FabricContext (#8)**, not any single flag.
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
| `38674433232` | #3 DA fix + #5 tensix-mux bubble per-mesh | **HANG** (15-min timeout) | same case/pattern as run 2: ranks `[1,1]`,`[1,3]` never finish. Confirms #6 (`ROUTING_MODE_MESH`) is the remaining blocker. |
| `38677621333` | #3 + #5 + #6-interim (global topology→Torus upgrade) | **HANG** (15-min timeout) | identical signature. Upgrading only `topology_`/`routing_mode_` is NOT enough — reverted. |
| run 5 (dispatched) | **per-rank fix**: each rank adopts its own mesh's config (#3/#5 reverted) | *pending* | rank 1 now builds its FabricContext for TORUS_X, not FABRIC_2D — see ⭐ RESOLUTION |

**Corrected conclusion.** The passing run-1 envelope differs from every failing run in two compounding
ways: (1) it opens the whole fabric with a **torus global config**, which satisfies *all* per-mesh gaps
at once (routing mode, VC config, channel counts, `is_2D` gates, peer-topology lookups), and (2) its
test cases are **all `topology: Torus`** — it **never runs a `topology: Mesh` case**. The hang is
specifically the common config's first case, `MeshNeighborExchange` on `topology: Mesh`, which asks the
fabric to route a plain-mesh neighbour-exchange over an MGD in which M_X is a genuine torus. That case
has **never worked** on this MGD (it is the original repro, pre-dating any of these fixes), and piecemeal
per-mesh fixes (#3 DA, #5 bubble, #6 routing-mode) each close one gap but not the set. **The complete
fix is the per-mesh `FabricContext` (#8): resolve topology / routing mode / VC config / channel counts /
packet specs per mesh from `MeshGraph::get_fabric_config(mesh_id)` rather than from one global config.**
That is a multi-file, kernel-touching change that needs hardware-in-the-loop debugging (tt-exalens on
the live hang to pin the exact stuck core/credit), which a 45-min CI cycle with empty triage and no local
galaxy cannot provide — so it is scoped, not attempted blind.

**What lands in this PR (correct, behaviour-preserving, worth keeping regardless of #8):** per-mesh
deadlock avoidance (#3) and per-mesh tensix-mux bubble flow control (#5). Both are genuine per-mesh
correctness fixes and prerequisites for #8.

**Options for finishing (your call):**
1. **Take on #8** (per-mesh FabricContext) with tt-exalens on the live galaxy hang to confirm the stuck
   point and drive the per-mesh plumbing. Cleanest, largest.
2. **Ship #3 + #5 now and test with a torus sanity config** (the run-1 envelope, or a dedicated
   per-mesh-torus config) — it exercises the per-mesh *torus* cases, which the fixes make correct, and
   is green today. The `topology: Mesh` case is simply not meaningful for an all-torus MGD.
3. **Hold the PR** until #8 is done.

**Diagnosis after run 2 (precise).** I walked what the global `topology_` actually drives in the
FabricContext constructor (`fabric_context.cpp:247-258`) and narrowed the Mesh-vs-TORUS_X delta to
exactly three things — two of which are now fixed:

* `is_2D_routing_enabled_` and `compute_packet_specifications` — **identical** for Mesh and TORUS_X
  (both are 2D). *Not a factor.*
* `bubble_flow_control_enabled_` → tensix-mux bubble — **fixed per-mesh (#5)**.
* The EDM VC0 bubble + first-level-ACK — **fixed per-mesh (#3)**.
* `compute_routing_mode()` → `routing_mode_`: `ROUTING_MODE_2D|ROUTING_MODE_MESH` under a Mesh-opened
  fabric vs `ROUTING_MODE_2D|ROUTING_MODE_TORUS` in the envelope. **This is the one remaining
  difference (#6).**

`routing_mode_` has a single consumer: it is emitted as the **global kernel define `ROUTING_MODE`**
(`fabric_context.cpp:365`, via `ControlPlane::get_fabric_kernel_defines()` applied to every fabric
kernel in `tt_metal.cpp:640,677`). So under a Mesh-opened fabric every router — including M_X's — runs
the kernel in `ROUTING_MODE_MESH`, which does not do the torus wrap/dateline handling M_X's genuine
ring needs, even though the wrap links exist in the routing tables and now carry DA + bubble. **Run 3 (DA + bubble) hung with
the same signature, confirming #6 as the completing fix.** Run 4 applies the #6 interim (upgrade the
global routing mode to the Torus superset when an MGD declares explicit per-mesh tori).

**On #6 and why a single routing-mode flip isn't the fix.** Converting the global `ROUTING_MODE` define
to a per-router value (per `local_fabric_node_id.mesh_id`) is itself a kernel/host-interface change, but
run 4 showed that even a *global* Mesh→Torus upgrade does not clear the Mesh-case hang — because routing
mode is only one of several values the global config drives (VC config #13, channel counts #10/#15,
`is_2D` gates #11, peer-topology lookups). The envelope passes by setting the whole global config to
torus, flipping all of them together. So #6 is not a standalone fix: the correct change is the per-mesh
`FabricContext` (#8) that resolves topology / routing mode / VC config / channel counts / packet specs
per mesh from `MeshGraph::get_fabric_config(mesh_id)`. That needs hardware-in-the-loop debugging
(tt-exalens on the live hang), which a 45-min CI cycle, empty CI triage, and no local galaxy cannot
provide — so it is scoped, not attempted blind.

**Why the DA fix + #5 are still correct and worth landing regardless:** both are genuine per-mesh
correctness fixes (each mesh now gets deadlock avoidance + bubble on its own ring axis), both are
behaviour-preserving for uniform runs, and both are prerequisites for the per-mesh-FabricContext end
state — they are not throwaway.
