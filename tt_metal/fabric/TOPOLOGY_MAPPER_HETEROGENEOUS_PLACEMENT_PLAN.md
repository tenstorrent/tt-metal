# Heterogeneous placement in the auto-mapper — plan index

Tracking issue: [#54623 — \[Auto-mapper\] Verify inter-mesh connectivity in heterogeneous placements via
SAT-based joint planning](https://github.com/tenstorrent/tt-metal/issues/54623)

Related: #40640 (SAT engine), #50510 (epic: auto-mapper blockers for blaze scale-out),
#52016 (pipeline-stage adjacency in MGD).

| # | Plan | Status | One-line summary |
| --- | --- | --- | --- |
| 1 | [PGD-shape-aware inter-mesh constraints](TOPOLOGY_MAPPER_PLAN_1_PGD_SHAPE_INTERMESH_CONSTRAINTS.md) | **Not done.** Lower priority while MeshId SAT remains; **irrelevant** if Plan 6 lands | Domain-filter the MeshId SAT so a 4×1 cannot land on a 4×4 region |
| 2 | [Incremental inter-mesh session](TOPOLOGY_MAPPER_PLAN_2_INCREMENTAL_INTERMESH_SOLVE.md) | **Done** (session API + live forbid/required). Remaining items are optional | Encode MeshId SAT once; live units on intra-mesh reject; no silent `next()` restart |
| 3 | [Connectivity-aware PGD DFS](TOPOLOGY_MAPPER_PLAN_3_CONNECTIVITY_AWARE_PGD_PLACEMENT.md) | **Superseded as the default** by Plan 4. DFS is `TT_METAL_PLACEMENT_SOLVER=dfs` fallback (delete-soon) | Adjacency-guided mixed-shape DFS over a live candidate pool |
| 4 | [SAT joint placement](TOPOLOGY_MAPPER_PLAN_4_SAT_JOINT_PLACEMENT.md) | **Default path. Core done.** Follow-ups in Plans 5–6 | Layer-1 footprints + layer-2 master SAT (disjointness + seams) |
| 5 | [Placement as Topology Solver API](TOPOLOGY_MAPPER_PLAN_5_PLACEMENT_AS_TOPOLOGY_SOLVER_API.md) | **Session half done.** Resource / master-as-session **not done** | Same session API for master seating: `add_resource_constraint` + MeshId→SeatId |
| 6 | [Collapse intermesh SAT](TOPOLOGY_MAPPER_PLAN_6_COLLAPSE_INTERMESH_SAT.md) | **Not started.** Do not implement until asked | Drop MeshId permutation; retry on `IntermeshPlacementEnumerationSession` |

## What is done vs not done

### Done

- **Plan 4 core.** `start_sat_placement` is the default producer for `solve_adjacency_guided_placement`. Column generation, seam matrices, trait-free caps, multi-solution master enumerate (blocking models) are in matching.cpp.
- **Plan 2 session contract.** `TopologyMappingEnumerationSession` is ctor-only (no `start()`, no args on `next()`, immovable). `add_forbidden_constraint` / `add_required_constraint` update `MappingConstraints` (reject-before-mutate), rewrite `ConstraintIndexData` in place, then `refresh_constraints()` on SAT (units) and DFS (same pointer). `exclude_mapping` blocks now. Host-cap loosening still destroy-and-construct.
- **Inter-mesh retry uses that session.** `MultiMeshSolutionEnumerator` holds `unique_ptr` to the session, forces SAT, and live-forbids an intra-mesh failing pair without re-encoding. Intra-mesh reconstruct fallback is gone.
- **Plan 3 DFS exists** as the env-selected fallback when SAT fails without a trustworthy UNSAT (truncated lists).

### Not done

- **Plan 1** shape-class required constraints (`mesh_shape_names_` + `add_inter_mesh_shape_class_constraints`). Never landed.
- **Plan 2 leftovers:** shape-class forbidden generalization, capacity precheck, intra-mesh verdict cache, public `excluded()` / `already_enumerated()` lists.
- **Plan 4 leftovers:** rewire DFS onto the master candidate list + MRV (old Phases 2/3); `PGD_DFS_DEBUG` cleanup (Phase 7).
- **Plan 5:** `add_resource_constraint` (type densifies to `ResourceIndex`); rebuild `encode_master_problem` as `TopologyMappingEnumerationSession<MeshId, SeatId>`; `IntermeshPlacementEnumerationSession` wrapper.
- **Plan 6:** identity consume of placement MeshIds; delete MeshId SAT enumerators; hide PGD `build_physical_*`.

### Irrelevant / superseded

- **Plan 3 as the production placement search.** Plan 4 replaced it. Do not invest in DFS node-budget / pool-index work except as fallback maintenance.
- **Plan 1 if Plan 6 ships.** Placement already seats by mesh instance and type. Shape-class MeshId filters only matter while a second permutation SAT can pair the wrong types.
- **Silent `next(graphs, constraints, …)` restart** (old Plan 2 / old session). Deleted. Changing graphs/mode/engine is destroy-and-construct.
- **Name anything `SatPlacementSession`.** The wrapper name is `IntermeshPlacementEnumerationSession` (Plans 5–6).

## Sequencing from here

1. **Plan 5 remaining** — resource constraint + master-as-topology-session. This is the API that Plans 4 follow-ups and Plan 6 both assume.
2. **Plan 6** — only after Plan 5’s wrapper exists. Until then keep MeshId SAT (`MultiMeshSolutionEnumerator`).
3. **Plan 1** — only if Plan 6 is declined and wrong-type MeshId pairings still show up.
4. **Plan 2 leftovers / Plan 4 DFS rewire** — optional; do not block 5–6.

## How the plans relate

Plan 4 chooses the **region set** (and a witness labelling). Pass 2 (`map_multi_mesh_to_physical` / `MultiMeshSolutionEnumerator`) still **labels and embeds**. Plan 2 made that second SAT incremental. Plan 5 makes the first SAT speak the same session language. Plan 6 deletes the second SAT for the unbound PGD path.

Plan 3’s two-pass rule still holds: placement decides the region set; the chip solve decides labelling properties (exits, pinnings, intra-mesh fit). Plan 6 does **not** delete intra-mesh.

## Validation MGDs

Both live in `tests/tt_metal/tt_fabric/custom_mesh_descriptors/` and run from the `bh-heterogeneous`
group in `tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh`:

- `bh_glx_2branch_mesh_per_stage_router_pipeline.textproto` — 69 single-rank meshes (60× 4×1, 8× 4×2,
  1× 4×4) forming a two-branch FABRIC-return fork off a degree-4 router mesh, 352 chips on the SC36 mock.
- `llama_8b_4galaxy_unpinned_mesh_graph_descriptor.textproto` — the llama + audio 7-mesh ring with the
  tray-4 audio pinnings removed, on the four SC4 single-pod mocks.
