# Plan 5 — Placement SAT as a Topology Solver API extension

Moved from the earlier Cursor plan `placement_as_topology_solver_api.plan.md`. Companion to
[Plan 6](TOPOLOGY_MAPPER_PLAN_6_COLLAPSE_INTERMESH_SAT.md) (where retry lives) and
[Plan 4](TOPOLOGY_MAPPER_PLAN_4_SAT_JOINT_PLACEMENT.md) (what the master SAT already does privately).

**Status: session half done. Resource API landed. Master-as-session uses the topology SAT
STRICT / RELAXED path on a seat graph with actual fabric-link multiplicity. Plan 6 wrapper
and deleting leftover private master SAT comments are still open.**

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
| `add_resource_constraint<Resource>` → densify to `ResourceIndex` 0..R−1 | **Done** (`uint32_t` identity-maps, `R = max+1`) |
| Disable bijection completeness when any resource constraint is set | **Done** |
| Rebuild `encode_master_problem` as `TopologyMappingEnumerationSession<MeshId, SeatId>` | **Done** |
| Do **not** add `ConnectionValidationMode::NONE` | **Done** — enum removed |
| Seat graph stores **actual** fabric-link multiplicity | **Done** |
| Session mode is `STRICT` then `RELAXED` (same graphs) | **Done** |
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
(`add_required_constraint(mesh, its_seats)`), the topology SAT’s existing STRICT / RELAXED
adjacency support is the seam check. Do **not** pre-filter seat edges by `need` and then skip
channel validation.

Host packing uses the existing same-rank group objective: seats that sit on one PSD host
form a global group, then `set_minimize_same_rank_groups_used` (soft; fall through if the
pack is infeasible). Do not add a parallel fill-used resource API. A hard
`max_same_rank_groups_used` cap would fail the session when the floor does not fit.

Column generation stays **outside** the encoder: grow pools, rebuild the seat graph, construct a new
session. Do not append variables to a live CNF.

### Demand vs supply (do not collapse these)

`collect_mesh_edges` is **logical demand**: which mesh pairs the MGD seamed, and required channel
count *N*. It walks `mesh_level_graph` (duplicate neighbors → *N*). Candidate footprints /
`boundary_link_dense_` / `fabric_links_to` / `AdjacencyMatrix::saturated_link_count` are **physical
supply**: how many fabric links one seating actually has into another seating’s chips. STRICT /
RELAXED compare supply to demand. Footprints cannot replace `collect_mesh_edges`.

---

## Master session: STRICT vs RELAXED (done)

Do not add `ConnectionValidationMode::NONE`. Bake-threshold-into-the-graph + `NONE` duplicates
what the engine already does on `target_conn_count` vs `global_conn_count`.

**Seat graph (build once per candidate generation).** For each `(m1, m2)` from
`collect_mesh_edges`, for each seat pair:

```text
links = adjacency.saturated_link_count(from, to)
if (links == 0) skip
add `links` parallel neighbor entries both ways
```

`GraphIndexData` turns those into `global_conn_count`. Required *N* stays on `mesh_level_graph`
(`target_conn_count`). Drop the `relaxed_tier` / `need` argument from the graph builder.

**Session mode (two attempts, same graphs).** SAT’s soft “prefer full channels” path is capped at
256 literals and will not run at gemma scale, so keep an explicit STRICT-first attempt:

| Policy | Attempt 1 | Attempt 2 |
| --- | --- | --- |
| Strict intermesh | `session(..., STRICT)` | none |
| Relaxed intermesh | `session(..., STRICT)` | if UNSAT: new session, **same graphs**, `RELAXED` |

```cpp
TopologyMappingEnumerationSession<GlobalMeshId, uint32_t> session(
    mesh_level_graph,
    seat_graph,       // actual link multiplicity
    constraints,      // domains + resource AMO + optional fill-used
    relaxed_tier ? ConnectionValidationMode::RELAXED
                 : ConnectionValidationMode::STRICT,
    /*quiet=*/true,
    TopologyMappingSolverEngine::Sat);
```

STRICT: support encoding keeps a partner only if `actual >= required`.
RELAXED: any adjacent pair (links ≥ 1) is hard-OK.

**Also in this pass:** delete `NONE` (enum, validator skip, tests). Resource overlap tests have no
seam edges — use `RELAXED`. Then PGD SAT unit tests and the llama / gemma SAT-log before/after.

**Not this pass:** one RELAXED session only (no STRICT-first); reintroduce private
`encode_master_problem` / seam indicators / `kStrictTierConflictBudget`; Plan 6 wrapper.

---

## Resource API (done)

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
- Do not add `ConnectionValidationMode::NONE`. Do not pre-filter seat edges by `need`.
- Do not replace `collect_mesh_edges` with Candidate footprints.

## Risks

- Seat compatibility graph can be denser than today’s seam matrix; may need a compact view.
- SeatId must be append-only across growth or excluded mappings go stale.
- Placement’s “fill every used host” is not `max_same_rank_groups_used`. Do not overload the cap.
