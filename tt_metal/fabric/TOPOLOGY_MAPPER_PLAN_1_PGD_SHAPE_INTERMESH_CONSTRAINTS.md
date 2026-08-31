# Plan 1 — PGD-shape-aware inter-mesh constraints

**Priority: 1 (ship first).** Self-contained; no solver-library changes.

Tracking issue: [#54623 — \[Auto-mapper\] Verify inter-mesh connectivity in heterogeneous placements via
SAT-based joint planning](https://github.com/tenstorrent/tt-metal/issues/54623)
Related: #40640 (SAT engine), #50510 (epic), #52016 (pipeline-stage adjacency in MGD).
Sibling plans: [Plan 2 — incremental inter-mesh solving](TOPOLOGY_MAPPER_PLAN_2_INCREMENTAL_INTERMESH_SOLVE.md),
[Plan 3 — connectivity-aware PGD placement](TOPOLOGY_MAPPER_PLAN_3_CONNECTIVITY_AWARE_PGD_PLACEMENT.md).

> **Goal.** Restrict each logical mesh's inter-mesh domain to physical meshes carved from the *same* PGD
> grouping / MGD mesh-descriptor name, so a shape-mismatched pair is never proposed.

---

## 1. How a heterogeneous MGD is mapped today

Two call paths reach the mesh-level solve. Both end in `map_multi_mesh_to_physical`.

### Path A — PGD-driven placement (Phase 1, no rank bindings)

`build_physical_multi_mesh_adjacency_graph`, `tt_metal/fabric/topology_mapper_utils.cpp:838`:

| Phase | Lines | What happens |
| --- | --- | --- |
| 2 | `882` | `get_valid_groupings_for_mgd` → `ValidGroupingsMap["MESH"][mesh_name] → vector<GroupingInfo>` |
| 3 | `907–951` | Per shape: `find_all_in_psd` → candidate `PsdPlacement`s; build a per-shape `PhysicalMultiMeshGraph`; precompute one chip bitmask per placement |
| 4 | `961–970` | Fast path: exactly one mesh shape → return that shape's graph, no solve |
| 5 | `993–1058` | One `MeshEnumState` per shape (own logical subgraph, own physical graph, own `TopologyMappingEnumerationSession`) |
| 6 | `1085–1458` | `DisjointPackingSearch`: DFS over one cached placement per shape, rejecting overlaps via bitmask |
| 7 | `1480–1515` | Re-key the winning placements under the **logical** MeshId the per-shape solver assigned, then `build_hierarchical_from_flat_graph` |

The critical detail is Phase 7. `combined_placements[logical_mesh_id]` is indexed by *logical* MeshId
(`topology_mapper_utils.cpp:1494–1512`), and `build_hierarchical_from_flat_graph` keys the resulting
`PhysicalMultiMeshGraph` by those same ids (`2420–2525`). **So in this path physical MeshId already
equals logical MeshId** — the correct assignment is fixed by construction before the inter-mesh solve
ever runs.

### Path B — rank-bound fast path (Phase 2)

`build_physical_multi_mesh_adjacency_graph(psd, asic_id_to_mesh_rank, ...)`,
`topology_mapper_utils.cpp:2358`, then `assign_pgd_pinnings_to_rank_bound_physical_graph` (`1539`)
attaches PGD pinnings without rediscovering footprints.

### The inter-mesh solve

`map_multi_mesh_to_physical`, `topology_mapper_utils.cpp:3379`:

```
3400  auto inter_mesh_constraints = build_inter_mesh_constraints(...)
3441  while (!success) {
3476      solver_result = solve_topology_mapping(mesh_logical_graph, mesh_physical_graph, ...)
3550      for each (logical, physical) pair: intra-mesh solve
              on failure → handle_forbidden_constraint(...)  → next while iteration
```

`build_inter_mesh_constraints` (`2680–2719`) contributes exactly three things:

- MGD pinnings → `add_required_constraint(mesh_id, physical_meshes)` (`2704`)
- rank-binding identity, **Phase 2 only** → `add_required_constraint(mesh_id, mesh_id)` (`2711`)
- host cover / cap / preferred bias → `add_inter_mesh_minimal_host_cover_from_hostname_map` (`2546`)

**Nothing tells the solver which physical mesh has which shape.** In Path A that information was known
and then thrown away.

## 2. The problem

For an unpinned Phase-1 MGD the inter-mesh SAT is free to map a logical 4×1 onto a physical region
carved for a 4×4. Nothing rejects it at the mesh level — mesh-level nodes are opaque `MeshId`s, and the
mesh-level adjacency of a 4×1 region and a 4×4 region can easily be isomorphic. The mistake only
surfaces one layer down, when the intra-mesh solve for that pair fails (`3550+`), which triggers
`handle_forbidden_constraint` (`3254`) and a **full re-solve from scratch** (`3476`). With `N` logical
and `M` physical meshes the loop is bounded by `N*M+1` attempts (`3438`); each attempt re-encodes the
whole CNF.

For the 69-mesh router pipeline in #54623 (`bh_glx_2branch_mesh_per_stage_router_pipeline.textproto`:
60× 4×1, 8× 4×2, 1× 4×4) that is up to 69·69 ≈ 4761 full re-solves, each one rediscovering a fact —
"a 4×1 does not fit a 4×4 region" — that the PGD already knew before the solve started.

This is the "intra-mesh solving errors" symptom. This plan removes the cause.

## 3. Why a required constraint is the right lever

The solver's required constraints are a **domain filter** applied in step 1 of the CNF encoding
(`topology_sat_build_initial_domains`, `tt_metal/fabric/topology_solver_sat.cpp:696–720`), before
assignment variables are created (step 3). Shrinking the domain therefore removes variables and clauses
rather than adding them: strictly cheaper CNF, strictly smaller search space, and no new soft objective
to trade off against the host cap.

The API already exists — the many-to-many overload:

```cpp
// tt_metal/api/tt-metalium/experimental/fabric/topology_solver.hpp:217
bool add_required_constraint(const std::set<TargetNode>& targets, const std::set<GlobalNode>& globals);
```

One call per shape class expresses "every logical mesh of shape S may only land on a physical mesh of
shape S". It returns `false` when `|targets| > |globals|`, which is exactly the "not enough regions of
this shape exist" precondition we want to fail loudly and early on.

**Prefer this over `set_same_rank_groups_constraint`.** Same-rank groups mean *co-location under one
global group label* (encoded as pairwise incompatibility clauses,
`topology_solver_sat.cpp:1010–1062`) and are already spoken for by host alignment
(`topology_mapper_utils.cpp:2605–2616`). Overloading them with shape identity would both conflict with
the host cover and be a weaker, more expensive encoding than a domain filter.

## 4. What has to be carried

The missing link is a per-physical-mesh shape label. Today `PhysicalMultiMeshGraph`
(`tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp:392–415`) carries
`mesh_adjacency_graphs_`, `mesh_level_graph_`, `mesh_exit_node_graphs_` and `mesh_pgd_pinnings_` — no
provenance.

Add one field, mirroring how `mesh_pgd_pinnings_` is already threaded through:

```cpp
struct PhysicalMultiMeshGraph {
    ...
    // Provenance of each physical mesh region: physical MeshId -> the MGD mesh/switch descriptor name
    // (the "MESH" key in ValidGroupingsMap) whose PGD grouping produced this region. Populated when the
    // graph was built from a PhysicalGroupingDescriptor; empty otherwise, in which case callers must not
    // constrain on shape.
    std::map<MeshId, std::string> mesh_shape_names_;
};
```

`MeshPhysicalLayout` (`topology_mapper_utils.hpp:421`) gains the same `std::string shape_name`, so the
label rides along the existing `PsdPlacement → MeshPhysicalLayout → build_hierarchical_from_flat_graph`
path rather than needing a parallel map.

## 5. Ownership and function passing

| Step | Function (file:line) | Change |
| --- | --- | --- |
| Produce | `find_all_in_psd` → `PsdPlacement` (`physical_grouping_descriptor.hpp:99`) | Add `std::string grouping_name` (or have the caller stamp it — the caller already knows `mesh_name` at `topology_mapper_utils.cpp:907`) |
| Carry | `mesh_physical_layouts_from_psd_placements` (`topology_mapper_utils.cpp:923`) | Copy `shape_name` into each `MeshPhysicalLayout` |
| Carry | Phase 7 re-key loop (`topology_mapper_utils.cpp:1494–1512`) | Stamp `combined_placements[logical].shape_name = mesh_order[j]` |
| Store | `build_hierarchical_from_flat_graph` (`topology_mapper_utils.cpp:2420`) | Copy into `mesh_shape_names_` next to the existing `mesh_pgd_pinnings_` copy at `2519–2523` |
| Store (Path B) | `assign_pgd_pinnings_to_rank_bound_physical_graph` (`topology_mapper_utils.cpp:1539`) | It already resolves MGD type name per mesh; write `mesh_shape_names_` in the same loop |
| Consume | `build_inter_mesh_constraints` (`topology_mapper_utils.cpp:2680`) | New helper below |

The logical side needs no new plumbing: `logical_mesh_id_to_mgd_instance_name`
(`topology_mapper_utils.cpp:1522`) already maps logical `MeshId → "M0"`-style descriptor name from the
MGD. It is file-local today, so either promote it to the header or pass the map into
`build_inter_mesh_constraints`. **Recommendation: pass the map in.** `build_inter_mesh_constraints`
currently takes no `MeshGraphDescriptor` and should not start depending on one; its callers
(`map_multi_mesh_to_physical:3400`, `MultiMeshSolutionEnumerator:3993`,
`enumerate_multi_mesh_placements:3908`) can build it once.

New helper, alongside `add_inter_mesh_minimal_host_cover_from_hostname_map`:

```cpp
// Restrict every logical mesh to physical regions carved from the same PGD grouping / MGD mesh
// descriptor. No-op when either side lacks shape labels, so non-PGD and hand-built graphs are unaffected.
void add_inter_mesh_shape_class_constraints(
    const PhysicalMultiMeshGraph& physical_graph,
    const std::unordered_map<MeshId, std::string>& logical_mesh_shape_names,
    ::tt::tt_fabric::MappingConstraints<MeshId, MeshId>& inter_mesh_constraints);
```

Body: bucket logical MeshIds by name, bucket physical MeshIds by `mesh_shape_names_`, then for each
name present on both sides call the many-to-many `add_required_constraint(logical_set, physical_set)`.

## 6. Behaviour matrix

| Situation | Behaviour |
| --- | --- |
| PGD path, labels on both sides | Hard domain filter per shape class |
| A logical shape with no physical regions | `add_required_constraint` returns `false` → fail early with "no PGD region for shape S", instead of `N*M` retries ending in a generic message |
| `mesh_shape_names_` empty (hand-built graph, unit tests) | Skip entirely — no behaviour change |
| Phase 2 rank-bound | Identity pin at `2711` already subsumes this; the shape filter is consistent with it and costs nothing |
| MGD pinnings present | Intersects with `2704`; intersection is the existing `MappingConstraints` semantics |

## 7. Interaction with the host cap

None. Shape classes constrain *which* physical mesh; the host cap constrains *how many host groups* are
occupied (`set_max_same_rank_groups_used`, `2631`). Both are hard, and the existing orchestration-level
relaxation at `3484–3492` (drop the cap, keep the soft minimize) still applies unchanged. Note the
relaxation must stay ordered so the shape filter is never the thing dropped.

## 8. Risk

Over-constraining if a PGD legitimately produces one physical region that can host two different MGD
shapes (e.g. a 4×2 region hosting a 4×1). Today that flexibility is accidental, not designed. Mitigation:
label with the **grouping** identity rather than the MGD instance name, and allow a region to carry a set
of compatible shape names rather than one. Start with the single-name form; widen only if a real MGD
needs it.

## 9. Validation

- `bh_glx_2branch_mesh_per_stage_router_pipeline.textproto` on the SC36 mock (the `bh-heterogeneous`
  group in `tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh`) — expect retry count to drop to 1.
- `llama_8b_4galaxy_unpinned_mesh_graph_descriptor.textproto` on the four SC4 mocks — the unpinned
  variant is the case with the most inter-mesh freedom, so it is the sharpest regression signal.
- Unit test in `tests/tt_metal/tt_fabric/fabric_router/test_topology_mapper_utils.cpp`: build a
  `PhysicalMultiMeshGraph` with two shape labels and assert the constraint rejects the cross pairing.
- Instrument the retry counter (`retry_attempt`, `3405`) and assert it stays at 1 for the heterogeneous
  MGDs — that is the real metric this plan moves.

## 10. Open questions

- Should a physical region carry one shape label or a set of compatible shapes? (See §8.) Needs a real
  MGD that wants a 4×2 region to host a 4×1.
- Should the label be the MGD instance name or the PGD grouping identity? The instance name is what
  `ValidGroupingsMap` is keyed by and is available on both paths; the grouping identity is the more
  honest description of the physical region. They coincide today because Phase 3 derives one from the
  other.
