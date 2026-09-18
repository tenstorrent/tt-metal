# Plan 5 — Placement SAT as a Topology Solver API extension

Moved from the earlier Cursor plan `placement_as_topology_solver_api.plan.md`. Companion to
[Plan 6](TOPOLOGY_MAPPER_PLAN_6_COLLAPSE_INTERMESH_SAT.md) (where retry lives) and
[Plan 4](TOPOLOGY_MAPPER_PLAN_4_SAT_JOINT_PLACEMENT.md) (what the master SAT already does privately).

**Status: session half done. Resource / master-as-session not done.**

> **Goal.** Same public algorithm as `topology_solver.hpp` — graphs + `MappingConstraints` +
> enumeration session — for master seating. Layer 1 already uses that API. Layer 2 needs one new
> constraint family (resource-disjointness on overlapping footprints). Do not invent
> `SatPlacementSession`. The wrapper name is `IntermeshPlacementEnumerationSession`.

---

## Checklist

| Item | Status |
| --- | --- |
| Session ctor snapshots graphs/constraints/mode/engine/`unique_shapes` | **Done** |
| No `start()`. Restart = destroy and construct | **Done** |
| `next()` takes no arguments and does not silently restart | **Done** |
| Session immovable (DFS pointers into snapshots); hold in `unique_ptr` | **Done** |
| Live `add_forbidden` / `add_required` passthroughs | **Done** (applied immediately, not queued until `next()`) |
| `MappingConstraints` trial-copy, reject before mutate | **Done** |
| In-place `ConstraintIndexData` assign + no-arg `refresh_constraints()` | **Done** |
| SAT: units from the existing encode. DFS: re-read the same index | **Done** |
| `exclude_mapping` / `exclude_mappings` `block()` now | **Done** |
| `set_quiet_mode` live | **Done** on the session snapshot |
| `solve_topology_mapping` / `_n` / `_all` are thin ctor + `next()` wrappers | **Done** |
| Grouping / mapper callers construct then `next()` | **Done** |
| Public `excluded()` / `already_enumerated()` | **Not done** (do not add yet) |
| `add_resource_constraint<Resource>` → densify to `ResourceIndex` 0..R−1 | **Not done** |
| Disable bijection completeness when any resource constraint is set | **Not done** (required with the above) |
| Rebuild `encode_master_problem` as `TopologyMappingEnumerationSession<MeshId, SeatId>` | **Not done** |
| `IntermeshPlacementEnumerationSession` column-generation wrapper | **Not done** — Plan 6 |
| Delete private master SAT once the session matches its models | **Not done** — after the rebuild |

What the live session will **not** grow in place (still destroy-and-construct):

- target/global graph, validation mode, SAT↔DFS, `unique_shapes`
- domain expansion, un-forbid, un-exclude
- host-cap raise/drop, preferred, cardinality, same-rank partition replace, `merge`
- `add_resource_constraint` / new seats / column growth (new variables)

---

## Why master placement is not a drop-in

Layer 1 (`enumerate_distinct_placements_for_grouping`) is already
`TopologyMappingEnumerationSession<LogicalChipId, AsicID>`.

Layer 2 (`encode_master_problem`) assigns each `MeshId` a **seat** (an ASIC *set*). Two seats can share
chips, so injectivity on seat IDs is not disjointness. The missing primitive is **footprint
disjointness**: at-most-one chosen seat per chip. Do not call this occupancy — that word already means
host packing (`topology_sat_build_occupancy_indicators`).

If every seat is a node in `AdjacencyGraph<SeatId>` and each mesh’s domain is its pool
(`add_required_constraint(mesh, its_seats)`), adjacency support matches today’s seam clauses (edge iff
enough fabric links). Host packing stays `set_same_rank_groups_constraint` + cap/minimize, not
resources.

Column generation stays **outside** the encoder: grow pools, rebuild the seat graph, construct a new
session. Do not append variables to a live CNF.

---

## Proposed API (not done)

```cpp
// Resource exists only at this call. After return, the type is gone.
template <typename Resource>
bool add_resource_constraint(const std::map<Global, std::vector<Resource>>& global_to_resources);
```

1. Densify each distinct `Resource` to `ResourceIndex` (`uint32_t`, `0 .. R-1`).
2. Store only `std::map<Global, std::vector<uint32_t>>` and `R`. Drop the type.
3. Encode: invert `lits_per_resource[i]`, `topology_sat_add_at_most_one` if the bucket has size ≥ 2.

Do **not** put `Resource` on `MappingConstraints<Target, Global, Resource>` or the session. Chip
mapping keeps `MappingConstraints<LogicalChipId, AsicID>` and does not call this.

**Bijection completeness** (`n_target == n_global`) must be **off** when any resource constraint
exists. Placement often has more seats than meshes; forcing both overlapping seats is trivial UNSAT.

Then the master problem is:

```cpp
MappingConstraints<MeshId, SeatId> c;
// domain: each mesh required onto its pool
// resources: add_resource_constraint(seat → dense_asics_); stored as 0..R-1
TopologyMappingEnumerationSession<MeshId, SeatId> session(
    mesh_level_graph, seat_compatibility_graph, c, validation_mode);
session.next();
```

`IntermeshPlacementEnumerationSession` owns per-mesh pools, builds the seat graph + resource bags,
forwards enabled `MappingConstraints` passthroughs, and `start`s again when columns grow.

---

## What not to do

- Do not flatten the master problem to `MeshId → AsicID` (wrong arity).
- Do not keep `encode_master_problem` *and* a topology-session clone.
- Do not make incremental DFS for placement. SAT-native `next()` / `block` only.
- Do not name this `SatPlacementSession`.

## Risks

- Seat compatibility graph can be denser than today’s seam matrix; may need a compact view.
- SeatId must be append-only across growth or excluded mappings go stale.
- Placement’s “fill every used host” is not `max_same_rank_groups_used`. Do not overload the cap.
