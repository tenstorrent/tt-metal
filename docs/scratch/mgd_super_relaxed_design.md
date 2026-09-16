# MGD `SUPER_RELAXED` Mode — Design & Implementation Plan

Issue: https://github.com/tenstorrent/tt-metal/issues/56762
Status: draft for review (2026-09-16)

## 1. Problem

tt-blaze needs pipelines where a declared MGD connection is carried D2D **over the host
interconnect**, not over fabric ethernet. Today every declared `connections {}` edge is a hard
constraint: STRICT requires the exact channel count, RELAXED tolerates fewer channels but still
requires **at least one** physical link (`mesh_graph_descriptor.proto:482-485`), and a missing
edge fails the mapping outright (isomorphism gate in `topology_solver.tpp:3642-3680`, cardinality
gate returning `false` at `topology_mapper_utils.cpp:3151-3153`, control-plane throw at
`control_plane.cpp:3043-3052`).

We want a third policy:

```textproto
connections {
  nodes { mesh { mesh_descriptor: "S4x2" mesh_id: 0 } }
  nodes { mesh { mesh_descriptor: "S4x1" mesh_id: 1 } }
  channels { count: 8 policy: SUPER_RELAXED }
}
```

**Semantics.** A SUPER_RELAXED connection:
1. is **never** a feasibility constraint — mapping succeeds even with zero physical links
   between the placed endpoints;
2. **is** a placement preference — the mapper *tries* to place the endpoints close together and
   to claim physical links when available (costing, below);
3. is retained after mapping and queryable via a simple topology-mapper API, at the granularity
   it was written: device-level endpoints resolve to `FabricNodeId`s; mesh-level endpoints (no
   `device_id` in the `MeshRef`) stay mesh-level — never expanded per-device.

## 2. Design overview

**No proto types outside the descriptor.** `proto::Policy` is only forward-declared in
`mesh_graph_descriptor.hpp`; the topology mapper (and every other consumer) must query policy
semantics through the MeshGraphDescriptor API — `is_connection_super_relaxed(ConnectionId)`,
added in commit 1 — never by including the protobuf headers. Later commits follow the same rule
(add descriptor-level accessors as needed rather than leaking proto).

**Soft edges as a side channel, hard graph untouched.** SUPER_RELAXED connections are kept out
of the solver's hard target graph (`LogicalMultiMeshGraph::mesh_level_graph_` /
`mesh_exit_node_graphs_`) so every existing ≥1-link gate stays as-is, and are carried in a new
side container that flows: MGD parse → logical graph build → solver (as *preference* terms
only) → post-solve link resolution → query API.

```
ConnectionData (policy==SUPER_RELAXED)
   └─ build_logical_multi_mesh_adjacency_graph_impl
        └─ LogicalMultiMeshGraph::super_relaxed_edges_        (new)
             ├─ solver costing (heuristic score / SAT soft literals)   §4
             ├─ post-solve link resolution ("take links if possible")  §5
             └─ TopologyMapper::get_super_relaxed_connections()        §6
```

## 3. Data model

New types in `tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp`
(near `LogicalExitNode`, :239):

```cpp
// A logical-only (SUPER_RELAXED) connection request. Granularity is preserved from the MGD:
// fabric_node_id == nullopt on an endpoint means the connection was declared mesh-level.
struct SuperRelaxedEdge {
    LogicalExitNode src;
    LogicalExitNode dst;
    uint32_t requested_channel_count = 0;
    bool directional = false;
};

// Post-mapping result for one SUPER_RELAXED edge. resolved_links is what the mapper managed
// to claim opportunistically — possibly empty (the connection then rides the host interconnect).
struct SuperRelaxedConnectionResult {
    SuperRelaxedEdge edge;
    std::vector<std::pair<FabricNodeId, FabricNodeId>> resolved_links;  // per claimed channel
    bool placed_adjacent = false;  // endpoints landed on physically adjacent slots
};
```

`LogicalMultiMeshGraph` (:304-319) gains `std::vector<SuperRelaxedEdge> super_relaxed_edges_;`.
`TopologyMappingResult` (:123-130) gains
`std::vector<SuperRelaxedConnectionResult> super_relaxed_connections;`.

## 4. Costing ("place close together")

Both backends get the soft edges as **preference terms**, never constraints. Priority order is
strictly below the existing objectives so SUPER_RELAXED can never distort feasibility or host
packing: feasibility ≫ host-cap / host-minimization ≫ existing RELAXED channel preference ≫
**SUPER_RELAXED adjacency preference** ≫ tie-breaks.

Reward tiers per soft edge, evaluated on the candidate placement of its two endpoint meshes:
- **T0 — same host**: both placed slots on one host (host interconnect is intra-host; cheapest).
- **T1 — physically adjacent**: ≥1 eth link exists between the placed slots
  (edge present in `PhysicalMultiMeshGraph::mesh_level_graph_`).
- **T2 — anything else**: no reward, still valid.

Backend changes:
- **Heuristic** (`topology_solver.tpp`): extend the RELAXED scoring block (`:2273-2300` pattern)
  — when scoring a candidate assignment, add `w_super_relaxed * tier` for each soft edge whose
  endpoints are both assigned. No change to the pruning paths (`:2209-2243`, `:2501-2515`,
  `:2912-2918`); soft edges never appear there because they are not in the target graph.
- **SAT** (`topology_solver_sat.cpp`): **no engine changes** — the preference is expressed
  entirely in the encoding layer with existing primitives:
  - Per soft edge, build one T0 (same-host) and one T1 (adjacent) indicator with the existing
    `topology_sat_define_indicator_as_or_of_pairwise_and` (`:1301`) over the sparse set of
    close (src-slot, dst-slot) pairs (taken from physical adjacency lists, not slots²).
  - New `topology_sat_append_super_relaxed_preference_literals(...)` collects the indicators
    with weights T0=2 / T1=1 and maximizes the weighted hit count via the same bounded
    k-descent used by the preferred-hit pass (`topology_sat_append_preferred_hit_indicators`
    `:1249` + the bound helpers `:76-240`), using the weighted at-least-k overload from #55884
    (`topology_solver_sat_solver.hpp:114-120, :330-335`).
  - **Lexicographic ordering, not weight tuning**: the pass runs *after* the host-cap /
    host-minimization objective, with the host bound pinned via assumptions in the same
    incremental session — SUPER_RELAXED preference can never cost a host by construction.
  - Budget guard mirroring the relaxed-channel literal cap (`:1355` pattern): over budget,
    skip the preference pass entirely (feasibility unaffected).
  - `MultiMeshSolutionEnumerator` is exempt: enumeration semantics unchanged; preference
    applies to single-shot solves only (optional future: sort enumerated solutions by hit
    count).
  Note `MappingConstraints::preferred_global_indices` is *unary* (target → globals, known up
  front) and cannot express this pairwise/joint preference — hence indicators, not preferred
  lists.
- **Plumbing**: soft edges travel to the solvers via the existing config path
  (`TopologyMappingConfig`, `topology_mapper_utils.hpp:88-118`) as a new field rather than via
  the graph type, keeping `MappingValidator` blind to them.

Weight note: a single scalar `w_super_relaxed` (T0 = 2w, T1 = w) chosen at least an order of
magnitude below the smallest host-minimization weight in each backend, so N soft edges can never
outbid saving a host. Exact constants picked during implementation with a unit test asserting
the ordering (§8, C7).

## 5. Opportunistic link claiming ("take links if possible")

After placement, a resolution pass per soft edge (new helper in `topology_mapper_utils.cpp`,
next to the cardinality code at `:3086-3159`):
1. Enumerate physical links between the placed slots of src and dst (same lookup the RELAXED
   cardinality cap uses, `total_physical_links_toward_dst`, `:3119-3122`).
2. Claim `min(requested_channel_count, available_after_hard_edges)` links — **hard (STRICT /
   RELAXED) edges are budgeted first**; SUPER_RELAXED only takes leftovers, so it can never
   starve a real connection. Zero available ⇒ `resolved_links` stays empty, mapping still
   succeeds.
3. Claimed links are recorded in `SuperRelaxedConnectionResult::resolved_links` and (open
   question §9) optionally registered into the normal intermesh connectivity so the control
   plane routes over them like RELAXED links.

## 6. Query API

`TopologyMapper` (`topology_mapper.hpp`, query section near `get_inter_mesh_connectivity`,
:314-316):

```cpp
// Logical-only (SUPER_RELAXED) connections retained through mapping, with any
// opportunistically claimed physical links. Mesh-level connections keep
// fabric_node_id == nullopt on their endpoints.
const std::vector<SuperRelaxedConnectionResult>& get_super_relaxed_connections() const;
```

Free-function layer: populated on `TopologyMappingResult` by `map_multi_mesh_to_physical` /
`map_multi_mesh_to_physical_n` / the enumerator, so `generate_rank_bindings` and tt-run callers
see it without extra calls.

Deferred (issue open question): a `ControlPlane::get_super_relaxed_connections_between_meshes
(MeshId, MeshId)` convenience wrapper — trivial once the mapper API exists; blitz
`blitz_decode_pipeline.cpp:270` (`TT_FATAL` on empty candidates) migrates to it in a follow-up.

## 7. Touched-file inventory

| Area | File(s) | Change |
|---|---|---|
| Proto | `tt_metal/fabric/protobuf/mesh_graph_descriptor.proto:489` | `SUPER_RELAXED = 3` + doc comment |
| MGD parse/validate | `tt_metal/fabric/mesh_graph_descriptor.cpp:807-835` | mixing check ignores SUPER_RELAXED (mixable with either STRICT or RELAXED); no defaulting change (`:317-341` untouched — SUPER_RELAXED only ever explicit) |
| MeshGraph path | `tt_metal/fabric/mesh_graph.cpp:306-398` | skip SUPER_RELAXED when filling `RequestedIntermeshConnections/Ports` and the policy→bool flags (skip must precede the first-connection policy derivation at `:377`); store in new `requested_super_relaxed_connections_` (`mesh_graph.hpp` near `:190`) |
| Graph build | `tt_metal/fabric/topology_mapper_utils.cpp:395-509`, `:314-345` | route SUPER_RELAXED `ConnectionData` into `super_relaxed_edges_` instead of adjacency; preserves the mixed-policy TODO resolution (`:403-406`) |
| Validation modes | `topology_mapper.cpp:505-519`, `generate_rank_bindings.cpp:198-217`, `topology_mapper_utils.cpp:2706-2721` | SUPER_RELAXED connections excluded from mode derivation (a graph whose only inter-mesh edges are SUPER_RELAXED derives RELAXED-with-no-required-edges) |
| Costing: heuristic | `topology_solver.tpp` (scoring near `:2273-2300`) | tiered adjacency reward |
| Costing: SAT | `topology_solver_sat.cpp` (near `:1339-1460`, gate `:1650-1656`) | weighted soft preference literals |
| Config plumbing | `topology_mapper_utils.hpp:88-118` | soft-edge list into solver config |
| Link claiming | `topology_mapper_utils.cpp` near `:3086-3159` | post-solve resolution pass (leftover links only) |
| API | `topology_mapper.hpp/.cpp`, `topology_mapper_utils.hpp:123-130` | `get_super_relaxed_connections()`, result field |
| Control plane | `control_plane.cpp:2937-3075`, `:2753-2781` | verify zero-link edges never reach the validators (they shouldn't, given the MeshGraph skip; add assert + test) |
| Docs | `tt_metal/fabric/MGD_README.md` (policy table, `:683` guidance) | document SUPER_RELAXED |

Explicitly untouched: `MappingValidator` (`topology_solver.tpp:3452-3687`), search pruning,
SAT hard clauses, `ConnectionValidationMode` (stays two-valued — SUPER_RELAXED is not a
validation mode).

## 8. Checks / tests to add

MGD parse & validation (`test_mesh_graph_descriptor.cpp`):
- **C1** `SUPER_RELAXED` parses on a FABRIC connection; textproto round-trips via `Policy_Name`.
- **C2** Mixing: SUPER_RELAXED + STRICT in one graph OK; SUPER_RELAXED + RELAXED OK;
  STRICT + RELAXED still rejected (regression on `mesh_graph_descriptor.cpp:807-835`).
- **C3** Mesh-level vs device-level endpoints both accepted and granularity recorded.

Mapper (`test_topology_mapper_utils.cpp`, fixture at `:93`):
- **C4** *Zero-link success*: two meshes, no physical inter-mesh links, SUPER_RELAXED edge ⇒
  mapping succeeds; `super_relaxed_connections` has one entry, `resolved_links` empty. Identical
  MGD with RELAXED still fails (guards the `:3151` gate).
- **C5** *Granularity*: mesh-level edge returns `fabric_node_id == nullopt` endpoints;
  device-level edge returns concrete `FabricNodeId`s.
- **C6** *Costing works*: physical system with two candidate placements for mesh B — one
  adjacent to A, one not, otherwise symmetric ⇒ both backends pick the adjacent slot
  (`placed_adjacent == true`). Run for heuristic and SAT.
- **C7** *Costing never outbids host packing*: a placement using fewer hosts but breaking
  SUPER_RELAXED adjacency is preferred over more hosts with adjacency (weight-ordering assert).
- **C8** *Link claiming*: links available between placed slots ⇒ `resolved_links` has
  `min(requested, available)` pairs; with a competing RELAXED edge over the same slots, the
  hard edge gets its links first and SUPER_RELAXED only the remainder (or none).
- **C9** *No feasibility leakage*: solver solution count for an MGD with N hard edges is
  unchanged by adding a SUPER_RELAXED edge (enumerator, `MultiMeshSolutionEnumerator`).
- **C10** *Validation-mode derivation*: graph whose only inter-mesh edge is SUPER_RELAXED maps
  without inter-mesh validation errors in both `determine_inter_mesh_validation_mode` callers.

Control plane / integration:
- **C11** `validate_requested_intermesh_connections` does not throw for a SUPER_RELAXED-only
  MGD on a mock with zero inter-mesh links (mock-cluster tt-run test, CPU-only group).
- **C12** End-to-end CPU-only entry in `run_fabric_cpu_only_unit_tests.sh` (bh-misc or a new
  case): SUPER_RELAXED MGD on an existing single-host mock, gtest asserting the query API.

## 9. Open questions (for review)

1. **Registration of claimed links**: should `resolved_links` also be pushed into the normal
   intermesh connectivity tables (control plane routes over them like RELAXED), or stay
   report-only until the blaze consumer lands? Proposal: report-only in PR 1.
2. **ControlPlane query API now or later?** Proposal: later (issue lists it as an open
   question); the mapper API is sufficient for blaze's placement needs.
3. **`count` semantics under SUPER_RELAXED**: treated as a *claim cap* (take up to N leftovers).
   Alternative: ignore count entirely. Proposal: claim cap.
4. FSD fabric-reliability interaction: out of scope here, tracked on the issue.

## 10. Delivery plan

1. Branch `rsong/mgd-super-relaxed` off main; this doc moves to the PR description.
2. Commit 1: proto + MGD parse/validation + data model + graph-build routing (C1-C3).
3. Commit 2: mapper success path + API + MeshGraph/control-plane skips (C4-C5, C10-C11).
4. Commit 3: costing, both backends (C6-C7, C9).
5. Commit 4: link claiming (C8) + docs + CPU-only suite entry (C12).
Local validation: fabric unit tests + the `bh-*` CPU-only groups touched; then TM-Fabric
CPU-only CI on the branch.
