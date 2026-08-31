# Plan 3 — Connectivity-aware PGD grouping placement

**Priority: 2.** This is the plan that actually answers #54623.

Tracking issue: [#54623 — \[Auto-mapper\] Verify inter-mesh connectivity in heterogeneous placements via
SAT-based joint planning](https://github.com/tenstorrent/tt-metal/issues/54623)
Related: #40640 (SAT engine), #50510 (epic), #52016 (pipeline-stage adjacency in MGD).
Sibling plans: [Plan 1 — PGD-shape-aware inter-mesh constraints](TOPOLOGY_MAPPER_PLAN_1_PGD_SHAPE_INTERMESH_CONSTRAINTS.md),
[Plan 2 — incremental inter-mesh solving](TOPOLOGY_MAPPER_PLAN_2_INCREMENTAL_INTERMESH_SOLVE.md).

> **Goal.** Take the logical mesh-level adjacency graph into account when choosing PGD groupings, so the
> selected placements are seam-compatible by construction instead of accidentally so.

Plans 1 and 2 make the *inter-mesh solve* cheap and correct. Neither stops Phase 6 from choosing a
disjoint packing whose seams are unroutable — that is this plan.

"Seam" is used throughout and is not a codebase term; §3 defines it and says exactly what the seam check
does and does not test.

---

## 1. Where placements are chosen today

`build_physical_multi_mesh_adjacency_graph`, `tt_metal/fabric/topology_mapper_utils.cpp:838`:

| Phase | Lines | What happens |
| --- | --- | --- |
| 2 | `882` | `get_valid_groupings_for_mgd` → candidate groupings per mesh shape |
| 3 | `907–951` | Per shape: `find_all_in_psd` → candidate `PsdPlacement`s; precompute one chip bitmask per placement (`927–946`) |
| 5 | `993–1058` | One `MeshEnumState` per shape: its own logical subgraph (`1037`), its own physical graph, its own `TopologyMappingEnumerationSession` |
| 6 | `1085–1458` | `DisjointPackingSearch`: DFS over one cached placement per shape |
| 7 | `1480–1515` | Re-key winners under logical MeshId; build the combined graph |

## 2. The gap

`DisjointPackingSearch` accepts any combination of per-shape placements that is **chip-disjoint** — the
only test is `bitset_disjoint` (`1103`). Each shape's placements were enumerated independently by its
own session (`pull_next_solution`, `1151`) against the logical subgraph built at `1037–1051`. That
subgraph comes from `build_mgd_mesh_level_subgraph_for_mesh_descriptor_name` (`1037`), which keeps
**only the FABRIC edges between meshes of the same shape** — every cross-shape edge is dropped on the
floor.

So for the issue's worked example (a ring alternating shapes A and B, where *every* hop crosses shapes)
each shape's subgraph has no edges at all. The placer packs A's meshes adjacently and B's meshes
adjacently, produces a perfectly disjoint solution, and strands half the ring. This is the failure mode
in #54623 verbatim.

It gets *more* likely as shapes approach physical capacity: packing slack is what accidentally saves
the seams today, and slack vanishes exactly when the MGD is interesting.

## 3. Terminology: what a "seam" is, and what the check asks

"Seam" is not a term in the codebase. It is shorthand for **one declared inter-mesh boundary**: a single
edge of the MGD's mesh-level graph. In the textproto that is exactly one `connections` block between two
mesh instances:

```proto
connections {
  nodes { mesh { mesh_descriptor: "S4x1" mesh_id: 6 } }
  nodes { mesh { mesh_descriptor: "S4x2" mesh_id: 7 } }
  channels { count: 8 policy: RELAXED }
}  # A group 0 stage 4 -> A group 0 join
```

That is one seam: logical mesh 6 and logical mesh 7 have declared they must be able to talk, over 8
channels. Those edges are collected by `get_requested_intermesh_from_mgd` and land in
`LogicalMultiMeshGraph::mesh_level_graph_`
(`tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp:309`), built by
`build_logical_multi_mesh_adjacency_graph_impl` (`topology_mapper_utils.cpp:395–524`). Channel
multiplicity is represented as duplicate entries in the neighbour vector, so "8 channels" is 8 parallel
edges rather than a weight.

**Same-shape vs cross-shape seams.** A seam is *same-shape* when both endpoints use the same mesh
descriptor (mesh 2 → mesh 3, both `S4x1`) and *cross-shape* when they differ (mesh 6 `S4x1` → mesh 7
`S4x2`). This distinction is the whole problem: `build_mgd_mesh_level_subgraph_for_mesh_descriptor_name`
(`topology_mapper_utils.cpp:1037`) keeps only the same-shape seams for each per-shape solve, so the
cross-shape seams are the ones nobody reasons about. In the router-pipeline MGD every group boundary
(`4x1 → 4x2`, `4x2 → 4x1`, `4x2 → 4x4`, `4x4 → 4x2`) is cross-shape, and so is every one of the router
mesh's four edges.

**What the check asks.** After the packing DFS has assigned every logical mesh a physical region — a
chip set from a `PsdPlacement` — take each seam `(Li, Lj)`, look up the regions `Ri` and `Rj` they were
assigned, and ask the hardware one question:

> Is there at least one ethernet link from some chip in `Ri` to some chip in `Rj`?

The PSD flat graph answers it directly: `build_flat_adjacency_map_from_psd`
(`topology_mapper_utils.cpp:737`) already emits one edge per ethernet link, including the cross-host
links that form the inter-galaxy seams. So the check is a set-crossing count over an edge list, which is
what the region-adjacency oracle in §4 memoizes.

**What the check is not.** It is deliberately coarse, and three other things stay where they are:

| Not this | Where it actually happens |
| --- | --- |
| Which *chip* in `Ri` talks to which chip in `Rj` | Inter-mesh port assignment, later, from the exit-node graphs (`mesh_exit_node_graphs_`) |
| Whether the logical mesh's internal shape fits the region | The intra-mesh solve, `map_multi_mesh_to_physical:3550+` |
| Whether the region is the *right* region for that logical mesh | [Plan 1](TOPOLOGY_MAPPER_PLAN_1_PGD_SHAPE_INTERMESH_CONSTRAINTS.md) — shape-class domain filter |

The seam check only answers "could these two regions ever talk?" — which is precisely the question
nothing asks today, and precisely the one whose absence lets a perfectly disjoint packing strand half a
ring.

## 4. Design — cross-shape seam feasibility as a first-class filter

Keep the per-shape enumeration (it is what makes the search tractable) and add a **seam check** to the
combination step, plus a **seam-aware ordering** so good combinations surface early.

Three layers, cheapest first.

### (a) Seam check at the leaf of the packing DFS

When a full combination is assembled, every logical mesh has a physical region. Walk the MGD's
cross-shape mesh-level edges and require each to land on a physical region pair with at least one link.
Reject the combination otherwise, exactly like the overlap rejection.

> **TODO — defer channel counts and STRICT/RELAXED for the placement seam check.**
> Start at **existence only**: `link_count(Ri, Rj) >= 1`, regardless of the seam's declared
> `channels { count: N }`. Explicitly push "≥ the requested channel count" and any STRICT/RELAXED
> distinction to a follow-up, and only add it if a real MGD is shown to need it.
>
> Rationale: existence is the weakest predicate that still kills the #54623 failure mode (a stranded
> seam has *zero* links, not too few), and a weaker predicate is a strictly larger feasible set — the
> packing DFS reaches an accepted combination sooner and cannot be driven infeasible by a channel
> shortfall on an otherwise correct placement. Counting channels also makes the oracle below more
> expensive: existence can short-circuit on the first crossing edge found, whereas a count must walk
> every crossing edge for every region pair.
>
> Worth knowing before revisiting: the two solves on either side of this check already disagree about
> strictness. The per-shape placement enumeration runs **STRICT**
> (`pull_next_solution`, `topology_mapper_utils.cpp:1162`), while the inter-mesh solve defaults to
> **RELAXED** (`determine_inter_mesh_validation_mode`, `2722–2727`, overridable via
> `TopologyMappingConfig::inter_mesh_validation_mode`). So the natural default for the seam check is to
> follow the inter-mesh mode it is protecting — i.e. existence, matching RELAXED — and to consider
> honouring the requested count only when the caller has explicitly asked for STRICT inter-mesh
> validation.
>
> The cost of deferring, stated honestly: a placement can pass an existence-only seam check and still
> fail later under STRICT inter-mesh validation because the seam resolved to fewer channels than the MGD
> asked for. That trades early, precise failure for a faster and more permissive search. It is the right
> trade while the common bug is "zero channels", and the wrong one once the common bug becomes "not
> enough channels" — which is the signal to pick this back up.

This needs a **region-adjacency oracle**: given two candidate placements (chip sets), do any links run
between them? Build it once from the PSD flat graph (`build_flat_adjacency_map_from_psd`, `737`) at the
same point the per-placement bitmasks are built (`927–946`): for each ordered pair of candidate
placements across shapes, look for a flat-graph edge crossing the two footprints. Cost is bounded by
`Σ_placements (chips × degree)`, i.e. one pass over the PSD edge list per placement pair bucket — the
same order as the bitmask precompute already accepted there.

Cache a `bool` per pair for now. Keep the interface returning a count-shaped answer
(`std::size_t link_count(a, b)` with an early-out at 1) only if that costs nothing, so the deferred
channel-count work above is a change of threshold rather than a change of data structure.

### (b) Seam-aware ordering

Sort each shape's cached placements by how many cross-shape seams they can still satisfy, so the DFS
meets a viable combination before exhausting the pathological ones. The existing order is "largest
embedding first" (`embedding_sizes`, `1189`); add seam degree as the tie-break.

### (c) Push seams into the SAT — the issue's "joint planning"

The end state in #54623 is one solve where disjointness *and* connectivity are both hard. That means
abandoning per-shape independent sessions for a single mesh-level problem over all shapes, with
region-adjacency encoded as support clauses. The mechanism already exists for intra-shape edges:
`topology_sat_encode_adjacency_support`, step 6 of `topology_sat_encode_hard_constraints`
(`topology_solver_sat.cpp:1130–1180`).

Layers (a) and (b) are worth doing first regardless: they are small, they make the failure *loud*
instead of silent, and they give a correctness oracle to test (c) against.

## 5. Ownership and function passing

| Concern | Owner | Notes |
| --- | --- | --- |
| Cross-shape logical edges | New `build_mgd_cross_shape_mesh_level_edges(mgd, mgd_intermesh_mesh_level)` next to `build_mgd_mesh_level_subgraph_for_mesh_descriptor_name` (`1037`) | Returns `vector<tuple<MeshId, MeshId, channels>>`; the exact complement of what the per-shape subgraph keeps |
| Region adjacency | New `PlacementAdjacencyOracle`, built in Phase 3 beside `group_bits_by_name` (`927`) | Owns `link_count(shape_a, idx_a, shape_b, idx_b)`; built from `flat_graph`, which Phase 3 already holds |
| Seam check | `DisjointPackingSearch` leaf (`1085–1458`) | Takes `const PlacementAdjacencyOracle&` and the cross-shape edge list by const ref; no ownership |
| Placement ordering | `pull_next_solution` / the per-shape caches (`1151–1192`) | Seam degree computed from the oracle at insert time, stored next to `embedding_sizes` (`1189`) |
| Diagnostics | The existing `TT_THROW` at `1472` | Must distinguish "no disjoint packing" from "disjoint packing exists but seam L_i—L_j is unroutable" — the second is the actionable message and the one #54623 asks for |

`PlacementAdjacencyOracle` should be a plain struct in `topology_mapper_utils.cpp`'s anonymous namespace
until layer (c) needs it in the header. Key it by `(shape_name, placement_index)` — the same key space
`group_bits_by_name` and `placements_by_shape` already use — so no new index mapping is introduced.

## 6. Interaction with Plan 1

Complementary, and they compose cleanly. Plan 1 fixes *which physical region a logical mesh may use*,
given a set of regions. This plan fixes *which set of regions gets chosen*. Doing 3 without 1 still
leaves the inter-mesh solve free to permute within a shape class; doing 1 without 3 leaves the seam
problem intact. Both are needed to close #54623.

## 7. Validation

- The issue's 4-slot cycle example is small enough to encode directly as a unit test in
  `tests/tt_metal/tt_fabric/fabric_router/test_topology_mapper_utils.cpp`: 4 physical slots in a cycle,
  4 logical meshes alternating two shapes, assert the mapper produces the interleaved placement and not
  the packed one.
- `bh_glx_2branch_mesh_per_stage_router_pipeline.textproto`: assert every declared MGD boundary resolves
  to ≥1 channel — the router mesh at fabric degree 4 is the sharpest case, since three of its four seams
  cross shapes.
- Extend the `bh-heterogeneous` CI group in
  `tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh` with a boundary-resolution assertion
  rather than only `TestGalaxyLayoutCheck`.

## 8. Open questions

- Should the seam check be hard, or hard-with-soft-fallback like the host cap? The issue argues hard
  ("fails loudly with an infeasibility core"); the host-cap precedent argues for an orchestration-level
  relaxation. **Recommend hard**, since an unroutable seam is a wrong answer rather than a suboptimal
  one. Note this is orthogonal to the channel-count deferral in §4(a): "hard" is about whether a failed
  seam check can be relaxed away, "existence only" is about what the check tests in the first place.
- What is the trigger to pick the deferred channel-count check back up? Proposed: the first MGD whose
  seams resolve to a non-zero but insufficient channel count under STRICT inter-mesh validation. Until
  that exists, counting channels during placement is speculative work.
- Is the full pairwise oracle affordable at SC36 scale, or does it need to be built lazily per queried
  pair? The bitmask precompute is `O(placements)`; the oracle is `O(placements²)` in the worst case.
  Lazy-with-memo is the obvious fallback if the eager build shows up in profiles.
- Where does #52016 (pipeline-stage adjacency in MGD) intersect? If stage adjacency becomes explicit in
  the MGD, the cross-shape edge list gets richer semantics — ordered pipeline hops rather than
  undirected seams — and the seam check could enforce direction as well as existence.
- Does layer (c) subsume layers (a) and (b), or should the leaf check stay as a cheap assertion even
  after the joint SAT lands? Keeping it is a guard against encoding bugs, at the cost of one extra walk
  per accepted solution.
