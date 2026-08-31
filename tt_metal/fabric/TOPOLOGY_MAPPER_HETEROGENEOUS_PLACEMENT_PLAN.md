# Heterogeneous placement in the auto-mapper — design plan

Tracking issue: [#54623 — \[Auto-mapper\] Verify inter-mesh connectivity in heterogeneous placements via
SAT-based joint planning](https://github.com/tenstorrent/tt-metal/issues/54623)

Related: #40640 (SAT engine), #50510 (epic: auto-mapper blockers for blaze scale-out),
#52016 (pipeline-stage adjacency in MGD).

This document plans three independent optimizations to the topology mapper. They are ordered by
priority, not by dependency:

1. **PGD-shape-aware inter-mesh constraints** — prune the inter-mesh SAT domain so shape-mismatched
   mesh pairs are unreachable, removing the dominant source of intra-mesh retry churn.
2. **Incremental inter-mesh solving** — reuse the SAT encoding across retries instead of re-solving
   from scratch, and strengthen the rejection constraints.
3. **Connectivity-aware PGD grouping placement** — feed the logical mesh-level adjacency graph into
   the per-shape placement search so the chosen groupings are seam-compatible by construction.

---

## 0. Background: how a heterogeneous MGD is mapped today

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
`topology_mapper_utils.cpp:2358`, then `assign_pgd_pinnings_to_rank_bound_physical_graph`
(`1539`) attaches PGD pinnings without rediscovering footprints.

### The inter-mesh solve

`map_multi_mesh_to_physical`, `topology_mapper_utils.cpp:3379`:

```
3400  auto inter_mesh_constraints = build_inter_mesh_constraints(...)
3433  TopologyMappingEnumerationSession<MeshId, MeshId> inter_mesh_session;   // declared, never used
3441  while (!success) {
3476      solver_result = solve_topology_mapping(mesh_logical_graph, mesh_physical_graph, ...)  // stateless
3484      if (!ok && max_same_rank_groups_used() > 0) { drop hard host cap; re-solve }
3550      for each (logical, physical) pair: intra-mesh solve
              on failure → handle_forbidden_constraint(...)  → next while iteration
```

`build_inter_mesh_constraints` (`2680–2719`) contributes exactly three things:

- MGD pinnings → `add_required_constraint(mesh_id, physical_meshes)` (`2704`)
- rank-binding identity, **Phase 2 only** → `add_required_constraint(mesh_id, mesh_id)` (`2711`)
- host cover / cap / preferred bias → `add_inter_mesh_minimal_host_cover_from_hostname_map` (`2546`)

**Nothing tells the solver which physical mesh has which shape.** In Path A that information was known
and then thrown away.

### Consequence

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

This is the "intra-mesh solving errors" symptom. Optimization 1 removes the cause.

---

## 1. PGD-shape-aware inter-mesh constraints  *(priority 1)*

### Goal

Restrict each logical mesh's inter-mesh domain to physical meshes carved from the *same* PGD grouping /
MGD mesh-descriptor name, so a shape-mismatched pair is never proposed.

### Why this is the right lever

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

### What has to be carried

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

### Ownership and function passing

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

### Behaviour matrix

| Situation | Behaviour |
| --- | --- |
| PGD path, labels on both sides | Hard domain filter per shape class |
| A logical shape with no physical regions | `add_required_constraint` returns `false` → fail early with "no PGD region for shape S", instead of `N*M` retries ending in a generic message |
| `mesh_shape_names_` empty (hand-built graph, unit tests) | Skip entirely — no behaviour change |
| Phase 2 rank-bound | Identity pin at `2711` already subsumes this; the shape filter is consistent with it and costs nothing |
| MGD pinnings present | Intersects with `2704`; intersection is the existing `MappingConstraints` semantics |

### Interaction with the host cap

None. Shape classes constrain *which* physical mesh; the host cap constrains *how many host groups* are
occupied (`set_max_same_rank_groups_used`, `2631`). Both are hard, and the existing orchestration-level
relaxation at `3484–3492` (drop the cap, keep the soft minimize) still applies unchanged. Note the
relaxation must stay ordered so the shape filter is never the thing dropped.

### Risk

Over-constraining if a PGD legitimately produces one physical region that can host two different MGD
shapes (e.g. a 4×2 region hosting a 4×1). Today that flexibility is accidental, not designed. Mitigation:
label with the **grouping** identity rather than the MGD instance name, and allow a region to carry a set
of compatible shape names rather than one. Start with the single-name form; widen only if a real MGD
needs it.

### Validation

- `bh_glx_2branch_mesh_per_stage_router_pipeline.textproto` on the SC36 mock (the `bh-heterogeneous`
  group in `tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh`) — expect retry count to drop to 1.
- `llama_8b_4galaxy_unpinned_mesh_graph_descriptor.textproto` on the four SC4 mocks — the unpinned
  variant is the case with the most inter-mesh freedom, so it is the sharpest regression signal.
- Unit test in `tests/tt_metal/tt_fabric/fabric_router/test_topology_mapper_utils.cpp`: build a
  `PhysicalMultiMeshGraph` with two shape labels and assert the constraint rejects the cross pairing.
- Instrument the retry counter (`retry_attempt`, `3405`) and assert it stays at 1 for the heterogeneous
  MGDs — that is the real metric this optimization moves.

---

## 2. Incremental inter-mesh solving + stronger rejection  *(priority 3, lower)*

### Goal

Replace the stateless `solve_topology_mapping` in the retry loop with the already-declared
`inter_mesh_session` (`topology_mapper_utils.cpp:3433`), so the hard CNF is encoded once and each retry
only appends clauses.

### Blocking correctness issue — must be fixed first

`TopologyMappingEnumerationSession::next` (`topology_solver.tpp:1280–1377`) only re-encodes the hard CNF
when the **context** changes (graphs, engine, validation mode, `unique_shapes`) or when the exclusion
list *shrinks* (`1372–1377`). On a context match it rebuilds `constraint_data_` (`1337`) but leaves
`sat_session_` — and therefore the baked-in domains — untouched.

The inter-mesh retry loop tightens `inter_mesh_constraints` between solves by adding **forbidden**
constraints (`handle_forbidden_constraint`, `3254`). Forbidden pairs are enforced as a domain filter at
encode time (`topology_solver_sat.cpp:706–707`), so a forbidden constraint added after the encode is
**not guaranteed to reach the CNF**. Naively swapping `solve_topology_mapping` for `session.next(...)`
would let the solver re-propose a pair the loop already rejected — an infinite retry, not a wrong answer,
but a hang in practice.

Two ways out, in preference order:

1. **Add an incremental-tightening API to the SAT bridge.** A forbidden pair is a unit clause
   `¬x_{t,g}`; CaDiCaL accepts clauses after `solve()` (`topology_solver_sat_solver.cpp:53–57`). Add
   `topology_sat_session_add_forbidden_pair(session, t_idx, g_idx)` next to the existing
   `topology_sat_session_add_blocking_clause` (`topology_solver_sat.cpp:1864–1897`), and have `next()`
   diff the constraint set against the previous call and emit unit clauses for newly forbidden pairs.
   This is the cheap, monotone case: constraints only ever tighten in this loop.
2. **Track a constraint generation counter** in `MappingConstraints`, and force a re-encode in `next()`
   when it changes. Correct but throws away the benefit whenever a pair is rejected — which is exactly
   when the loop is hot.

Option 1 is the real fix; option 2 is a safe stepping stone.

### Then: the loop change

```cpp
solver_result = inter_mesh_session.next(
    mesh_logical_graph, mesh_physical_graph, inter_mesh_constraints,
    /*excluded_mappings=*/{},          // rejection is expressed via forbidden pairs, not blocking clauses
    inter_mesh_validation_mode, quiet_mode,
    TopologyMappingSolverEngine::Sat, /*unique_shapes=*/false);
```

Note the host-cap relaxation at `3484–3492` mutates `inter_mesh_constraints`
(`set_max_same_rank_groups_used(0)`) — a *loosening*, which unit clauses cannot express. Keep that on the
stateless path, or hoist the decision before the session is created so the cap is decided once.

### "Additional forbidden constraints"

Once intra-mesh failure is understood as a **pair-local** property (the intra-mesh solve for
`logical L → physical P` depends only on that pair's two subgraphs, not on the rest of the assignment),
each observed failure can be generalized instead of recorded one pair at a time:

- **Shape-class generalization.** If `L → P` failed and `L' ` has the same MGD descriptor name as `L`,
  and `P'` the same PGD grouping as `P`, then `L' → P'` will fail too. Emit the whole cross product with
  the existing `add_forbidden_constraint(std::set targets, std::set globals)`
  (`topology_solver.hpp:301`). With optimization 1 in place most of these are already unreachable — which
  is why this is lower priority.
- **Capacity generalization.** If `L` has more fabric nodes than `P` has ASICs, forbid at constraint-build
  time, not after a failed solve. This is a cheap precheck in `build_inter_mesh_constraints`.
- **Cache the intra-mesh verdict** per `(logical, physical)` pair so repeated attempts inside one mapping
  never re-run the sub-solve.

### Validation

- `sat_hard_constraint_encode_calls()` / `sat_solve_calls()` are already exposed on the session
  (`topology_solver.hpp:1516–1517`) and already asserted in
  `tests/tt_metal/tt_fabric/fabric_router/test_topology_solver.cpp:5973+`. Assert encode calls == 1 across
  a multi-retry inter-mesh mapping.
- Regression: a case that *must* retry (force an intra-mesh failure) and assert it still converges to the
  same mapping as the stateless path.

---

## 3. Connectivity-aware PGD grouping placement  *(priority 2)*

This is the optimization that actually answers #54623. Optimizations 1 and 2 make the *inter-mesh solve*
cheap and correct; they do not stop Phase 6 from choosing a disjoint packing whose seams are unroutable.

### The gap

`DisjointPackingSearch` (`topology_mapper_utils.cpp:1085–1458`) accepts any combination of per-shape
placements that is **chip-disjoint** — the only test is `bitset_disjoint` (`1103`). Each shape's
placements were enumerated independently by its own session (`pull_next_solution`, `1151`), against its
own `logical_graph`/`physical_graph` pair built at `1037–1051`. That per-shape logical subgraph comes from
`build_mgd_mesh_level_subgraph_for_mesh_descriptor_name` (`1037`), which keeps **only the FABRIC edges
between meshes of the same shape** — every cross-shape edge is dropped on the floor.

So for the issue's worked example (a ring alternating shapes A and B, where *every* hop crosses shapes)
each shape's subgraph has no edges at all. The placer packs A's meshes adjacently and B's meshes
adjacently, produces a perfectly disjoint solution, and strands half the ring. This is the failure mode in
#54623 verbatim, and it gets *more* likely as shapes approach physical capacity, because packing slack —
which is what accidentally saves the seams today — vanishes.

### Design: cross-shape seam feasibility as a first-class filter

Keep the per-shape enumeration (it is what makes the search tractable) and add a **seam check** to the
combination step, plus a **seam-aware ordering** so good combinations surface early.

Three layers, cheapest first:

**(a) Seam check at the leaf of the packing DFS.** When a full combination is assembled, every logical
mesh has a physical region. Walk the MGD's cross-shape mesh-level edges and require each to land on a
physical region pair with at least one link — and, in `STRICT`, at least the requested channel count.
Reject the combination otherwise, exactly like the overlap rejection.

This needs a **region-adjacency oracle**: given two candidate placements (chip sets), how many links run
between them? Build it once from the PSD flat graph (`build_flat_adjacency_map_from_psd`, `737`) at the
same point the per-placement bitmasks are built (`927–946`): for each ordered pair of candidate placements
across shapes, count flat-graph edges crossing the two footprints. Cost is bounded by
`Σ_placements (chips × degree)`, i.e. one pass over the PSD edge list per placement pair bucket — the same
order as the bitmask precompute already accepted there.

**(b) Seam-aware ordering.** Sort each shape's cached placements by how many cross-shape seams they can
still satisfy, so the DFS meets a viable combination before exhausting the pathological ones. The existing
order is "largest embedding first" (`embedding_sizes`, `1189`); add seam degree as the tie-break.

**(c) Push seams into the SAT, per the issue's "joint planning".** The end state in #54623 is one solve
where disjointness *and* connectivity are both hard. That means abandoning per-shape independent sessions
for a single mesh-level problem over all shapes, with region-adjacency encoded as support clauses (the
mechanism already exists for intra-shape edges: `topology_sat_encode_adjacency_support`, step 6 of
`topology_sat_encode_hard_constraints`, `topology_solver_sat.cpp:1130–1180`). Layers (a) and (b) are worth
doing first regardless: they are small, they make the failure *loud* instead of silent, and they give a
correctness oracle to test (c) against.

### Ownership and function passing

| Concern | Owner | Notes |
| --- | --- | --- |
| Cross-shape logical edges | New `build_mgd_cross_shape_mesh_level_edges(mgd, mgd_intermesh_mesh_level)` next to `build_mgd_mesh_level_subgraph_for_mesh_descriptor_name` (`1037`) | Returns `vector<tuple<MeshId, MeshId, channels>>`; the exact complement of what the per-shape subgraph keeps |
| Region adjacency | New `PlacementAdjacencyOracle`, built in Phase 3 beside `group_bits_by_name` (`927`) | Owns `link_count(shape_a, idx_a, shape_b, idx_b)`; built from `flat_graph`, which Phase 3 already holds |
| Seam check | `DisjointPackingSearch` leaf (`1085–1458`) | Takes `const PlacementAdjacencyOracle&` and the cross-shape edge list by const ref; no ownership |
| Diagnostics | The existing `TT_THROW` at `1472` | Must distinguish "no disjoint packing" from "disjoint packing exists but seam L_i—L_j is unroutable" — the second is the actionable message and the one #54623 asks for |

`PlacementAdjacencyOracle` should be a plain struct in `topology_mapper_utils.cpp`'s anonymous namespace
until (c) needs it in the header. Keep it keyed by `(shape_name, placement_index)` — the same key space
`group_bits_by_name` and `placements_by_shape` already use — so no new index mapping is introduced.

### Interaction with optimization 1

Complementary, and they compose cleanly. Optimization 1 fixes *which physical region a logical mesh may
use*, given a set of regions. Optimization 3 fixes *which set of regions gets chosen*. Doing 3 without 1
still leaves the inter-mesh solve free to permute within a shape class; doing 1 without 3 leaves the seam
problem intact. Both are needed to close #54623.

### Validation

- The issue's 4-slot cycle example is small enough to encode directly as a unit test in
  `test_topology_mapper_utils.cpp`: 4 physical slots in a cycle, 4 logical meshes alternating two shapes,
  assert the mapper produces the interleaved placement and not the packed one.
- `bh_glx_2branch_mesh_per_stage_router_pipeline.textproto`: assert every declared MGD boundary resolves
  to ≥1 channel — the router mesh at fabric degree 4 is the sharpest case, since three of its four seams
  cross shapes.
- Extend the `bh-heterogeneous` CI group with a boundary-resolution assertion rather than only
  `TestGalaxyLayoutCheck`.

---

## Sequencing

1. **Optimization 1** — self-contained, no solver changes, immediate reduction in retry churn. Ship first.
2. **Optimization 3 layers (a) and (b)** — turns the silent unroutable placement into a loud, diagnosable
   failure, and fixes the common cases.
3. **Optimization 2** — needs the session-tightening fix in the solver bridge; its payoff shrinks once
   optimization 1 has removed most retries. Do it for the encode-once win, not for correctness.
4. **Optimization 3 layer (c)** — the full joint SAT from #54623, once (a) exists as an oracle to check
   it against.

## Open questions

- Should a physical region carry one shape label or a set of compatible shapes? (Optimization 1, risk
  section.) Needs a real MGD that wants a 4×2 region to host a 4×1.
- Is intra-mesh failure truly pair-local in all cases, including PGD-pinned meshes where
  `mesh_pgd_pinnings_` constrains the intra-mesh solve? If a pinning makes it context-dependent, the
  forbidden-constraint generalization in optimization 2 is unsound as stated.
- Should the seam check be hard, or hard-with-soft-fallback like the host cap? The issue argues hard
  ("fails loudly with an infeasibility core"); the host cap precedent argues for an orchestration-level
  relaxation. Recommend hard, since an unroutable seam is a wrong answer rather than a suboptimal one.
- Where does #52016 (pipeline-stage adjacency in MGD) intersect? If stage adjacency becomes explicit in
  the MGD, the cross-shape edge list in optimization 3 gets richer semantics (ordered pipeline hops rather
  than undirected seams).
