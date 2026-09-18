# Plan 6 — Collapse intermesh SAT into incremental SAT placement

Moved from the earlier Cursor plan `collapse_intermesh_sat_9a729c25.plan.md`. Depends on
[Plan 5](TOPOLOGY_MAPPER_PLAN_5_PLACEMENT_AS_TOPOLOGY_SOLVER_API.md) for the session wrapper.
Do **not** implement until asked.

**Status: not started.** The MeshId permutation SAT (`MultiMeshSolutionEnumerator`) is still the
production consume path. Plan 2 made that SAT incremental; this plan deletes it for the unbound PGD
path.

> **Goal.** SAT joint placement already assigns each logical mesh instance to a footprint. The later
> MeshId→MeshId SAT is a leftover permutation that can pair the wrong mesh types. Incremental retry
> belongs on `IntermeshPlacementEnumerationSession`, with intra-mesh still run on an **identity**
> mapping.

---

## Checklist

| Item | Status |
| --- | --- |
| Keep `PlacedMesh.mesh_id` as `PhysicalMultiMeshGraph` keys (identity well-defined) | **Not done** |
| `IntermeshPlacementEnumerationSession` (pools + master CNF + `next`/`block` + growth re-encode) | **Not done** |
| `map(psd, pgd, mgd)`: next seating → internal graph → identity intra-mesh → retry placement | **Not done** |
| Demote PGD `build_physical_multi_mesh_*` from the public header | **Not done** |
| Two-graph `map` overload: skip MeshId SAT, identity + intra only | **Not done** |
| Delete unused `MultiPhysicalMultiMeshSolutionEnumerator` | **Not done** (still declared, no production callers) |
| Drop `MultiMeshSolutionEnumerator` as a MeshId retry engine | **Not done** |
| Retarget Phase-1 tests to the new map entry | **Not done** |

What this does **not** change (and must not delete):

- Intra-mesh chip solve (`TopologyMappingEnumerationSession` on fabric nodes → ASICs)
- TopologyMapper rank-bound path (already identity-pinned)
- Layer-1 geometry SAT (`enumerate_distinct_placements_for_grouping`)
- Single-mesh `map_mesh_to_physical`

---

## Why the second pass is leftover

`start_sat_placement` / `encode_master_problem` already place **per mesh instance**:

- exactly one seat per `GlobalMeshId`
- ASIC disjointness
- seam clauses from the MGD mesh-level graph
- host packing via `pack_active_lit`

`decode_master_model` still knows `mesh_id`. Then `psd_placements_from_assigned_meshes` **drops it**,
and `mesh_physical_layouts_from_psd_placements` re-keys `MeshId{i}` by vector index.
`map_multi_mesh_to_physical` treats those IDs as unlabeled slots and re-solves MeshId→MeshId.

That permutation is why `build_inter_mesh_constraints` still has to worry about pairing a logical mesh
with a **different-type** physical region. Placement already chose the type. Re-permuting undoes it.
[Plan 1](TOPOLOGY_MAPPER_PLAN_1_PGD_SHAPE_INTERMESH_CONSTRAINTS.md) exists only to paper over that;
it becomes irrelevant if this plan lands.

---

## Proposed Phase-1 loop

```
IntermeshPlacementEnumerationSession::next()
    → seating (logical MeshId → PsdPlacement, MeshIds kept)
    → internal PhysicalMultiMeshGraph keyed by those MeshIds
    → complete_intra_mesh_for_placement with identity mesh_mappings
    → on intra-mesh fail: add_forbidden / exclude_mapping, next() again
    → on master UNSAT + truncated lists: grow columns, new session
    → on master UNSAT + complete lists: exhausted
```

Same-shape swaps are different seat-literal models in the master SAT, not a second solver.

**Two-graph `map(logical, physical)`** (rank-bound / hand-built partitions): no enumerator. Identity
+ intra only. If that fails, fail. Ranks already fixed the partition.

**Public API.** Take PGD `build_physical_multi_mesh_adjacency_graph` / `_n` off the public header.
`generate_rank_bindings` calls the new map entry. Keep a narrow two-graph map for rank-bound and
hand-built tests.

---

## Risks

- **Host cover.** Inter-mesh has a hard at-most-k cap; placement has a softer “fill every used host”
  assumption. If a test depends on the hard cap, lift it into the master SAT rather than keeping MeshId
  SAT for it.
- **DFS fallback.** Incremental retry is SAT-native. Do not build incremental DFS.
- **`map_multi_mesh_to_physical_n` / `unique_shapes`.** After this, enumeration means distinct
  *seatings*, not MeshId permutations of a fixed physical graph.
