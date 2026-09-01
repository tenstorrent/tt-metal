# Plan 3 — Connectivity-aware PGD grouping placement

**Adjacency-guided mixed-shape placement.** Replace the per-shape maximum-coverage tiling with a single
DFS that grows one mixed-shape placement along the MGD's own mesh graph.

**Priority: 2.** This is the plan that actually answers #54623.

Tracking issue: [#54623 — \[Auto-mapper\] Verify inter-mesh connectivity in heterogeneous placements via
SAT-based joint planning](https://github.com/tenstorrent/tt-metal/issues/54623)
Related: #40640 (SAT engine), #50510 (epic), #52016 (pipeline-stage adjacency in MGD).
Sibling plans: [Plan 1 — PGD-shape-aware inter-mesh constraints](TOPOLOGY_MAPPER_PLAN_1_PGD_SHAPE_INTERMESH_CONSTRAINTS.md),
[Plan 2 — incremental inter-mesh solving](TOPOLOGY_MAPPER_PLAN_2_INCREMENTAL_INTERMESH_SOLVE.md).

> **Goal.** Choose physical regions by walking the logical mesh-level adjacency graph, placing one mesh
> at a time next to a neighbour that is already placed, so every declared inter-mesh boundary is
> satisfied *by construction* rather than checked after the fact.

"Seam" is used throughout and is not a codebase term; §3 defines it and says exactly what it means.

**This does not replace the inter-mesh solve.** §4 states the architecture — two coupled passes — and
the rule that fixes what the DFS is allowed to decide: *the DFS decides properties of the region set;
the inter-mesh solve decides properties of the labelling.* §4(c) and §4(e) are the in/out lists.
§6 records optimizations under consideration and is explicitly undecided.

---

## 1. Where placements are chosen today

`build_physical_multi_mesh_adjacency_graph`, `tt_metal/fabric/topology_mapper_utils.cpp:838`:

| Phase | Lines | What happens |
| --- | --- | --- |
| 2 | `882` | `get_valid_groupings_for_mgd` → candidate groupings per mesh shape |
| 3 | `907–951` | Per shape: `find_all_in_psd` → `PsdPlacement`s; one chip bitmask per placement (`927–946`) |
| 4 | `953–970` | Fast path: single shape returns the pre-built graph directly |
| 5 | `993–1058` | One `MeshEnumState` per shape: own logical subgraph (`1037`), own physical graph, own `TopologyMappingEnumerationSession` |
| 6 | `1085–1458` | `DisjointPackingSearch`: DFS over one cached solution per shape, `bitset_disjoint` only |
| 7 | `1480–1515` | Re-key winners under logical MeshId; `build_hierarchical_from_flat_graph` |

The important part is inside Phase 3, one level down. `find_all_in_psd` →
`solve_for_many_groupings_to_psd_heterogeneous` (`physical_grouping_descriptor_matching.cpp:1718`) has
two phases of its own:

```
Phase A  enumerate_distinct_placements_for_grouping(grouping, …, kMaxPlacementsPerGrouping = 1024)
         → a rich pool of candidate placements, overlapping, deduped by image set   (1739–1789)

Phase B  solve_set_packing(candidates, universe_size, kSetPackingBudget = 5s)
         → ONE maximum-weight disjoint subset, weight = asic_slots.size()           (1800–1830)
```

Only Phase B's survivors are returned. So `placements_by_shape["S4x1"]` is not a menu of options — it is
a single frozen tiling of the machine chosen to cover as many chips as possible. The loop at
`topology_mapper_utils.cpp:907` runs this **once per shape, independently**, so each shape gets its own
maximum-coverage tiling of the same hardware, computed with no knowledge of the other shapes and no
knowledge of the MGD.

## 2. The gap

Two failures compound, and the second is the one that cannot be repaired downstream.

**The logical subgraph drops cross-shape edges.**
`build_mgd_mesh_level_subgraph_for_mesh_descriptor_name` (`1037`) keeps only FABRIC edges between meshes
of the *same* shape. `DisjointPackingSearch` then accepts any combination that is chip-disjoint —
`bitset_disjoint` (`1103`) is the only test. Nothing anywhere asks whether a cross-shape boundary landed
on physically connected regions.

**The tiling is frozen before any of that runs.** Phase B commits to tile boundaries while optimizing
for chip coverage, which is not what the MGD wants. Here is the smallest case that shows it — eight
chips in a line, an MGD chain of `S2 — S4 — S2`:

```
Physical:   c0 ── c1 ── c2 ── c3 ── c4 ── c5 ── c6 ── c7

Logical:    [S2_a] ──── [S4_a] ──── [S2_b]
```

Maximum-weight set packing has exactly one optimum per shape, and both are forced:

```
S4 tiling (weight 8):          S2 tiling (weight 8):
 c0 c1 c2 c3 c4 c5 c6 c7        c0 c1 c2 c3 c4 c5 c6 c7
 └─── T0 ──┘└─── T1 ──┘         └U0─┘└U1─┘└U2─┘└U3─┘
```

Pick `S4_a = T0`. The only `S2` tile both disjoint and adjacent is `U2`; two `S2` meshes, one candidate,
dead. `T1` is symmetric. **Over the frozen tilings this MGD is unsatisfiable**, though the hardware
plainly supports it:

```
 c0 c1 c2 c3 c4 c5 c6 c7
 └S2_a┘└── S4_a ──┘└S2_b┘
```

`S4_a = [c2..c5]` is in shape `S4`'s Phase A pool. It is never in `S4`'s Phase B tiling, because
covering all eight chips forces the `[c0..c3][c4..c7]` split. No seam check at the packing DFS, and no
constraint added to the inter-mesh SAT, can recover it: the tile boundary was destroyed two phases
earlier. This is #54623 in eight chips.

It gets worse as shapes approach capacity — packing slack is what accidentally saves seams today, and
slack vanishes exactly when the MGD is interesting.

## 3. Terminology: what a "seam" is

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
multiplicity is duplicate entries in the neighbour vector, so "8 channels" is 8 parallel edges rather
than a weight.

**Same-shape vs cross-shape.** A seam is *same-shape* when both endpoints use the same mesh descriptor
(mesh 2 → mesh 3, both `S4x1`) and *cross-shape* when they differ (mesh 6 `S4x1` → mesh 7 `S4x2`). In
the router-pipeline MGD every group boundary (`4x1 → 4x2`, `4x2 → 4x1`, `4x2 → 4x4`, `4x4 → 4x2`) is
cross-shape, and so is every one of the router mesh's four edges.

**The seam predicate.** Given two logical meshes assigned physical regions `Ri` and `Rj`, the question
is:

> Is there at least one ethernet link from some chip in `Ri` to some chip in `Rj`?

`build_flat_adjacency_map_from_psd` (`topology_mapper_utils.cpp:737`) already emits one edge per
ethernet link, cross-host links included, so this is a set-crossing test over an edge list.

**The change in this plan is where that predicate is applied.** Today it is applied nowhere. The earlier
draft of this plan applied it as a *filter* at the leaf of the packing DFS. This plan applies it as a
*generator*: candidates for the next mesh are drawn from the set of regions adjacent to the neighbours
already placed, so a violating placement is never constructed. Same predicate, far stronger pruning.

Three things stay where they are:

| Not this | Where it actually happens |
| --- | --- |
| Which *chip* in `Ri` talks to which chip in `Rj` | Inter-mesh port assignment, later, from `mesh_exit_node_graphs_` |
| Whether the logical mesh's internal shape fits the region | The intra-mesh solve, `map_multi_mesh_to_physical:3550+` |
| Whether the region is the *right* region for that logical mesh | [Plan 1](TOPOLOGY_MAPPER_PLAN_1_PGD_SHAPE_INTERMESH_CONSTRAINTS.md) — shape-class domain filter |

## 4. Architecture: two coupled passes, and what the DFS is allowed to decide

### (a) Verdict: stay two-pass, redraw the boundary

The obvious reading of this plan is that the DFS subsumes the inter-mesh solve — it assigns logical
meshes to physical regions, so why keep `map_multi_mesh_to_physical` at all? **Because the bug is
premature commitment, not the existence of stages.** What breaks today is Phase B discarding the
candidate pool (§1, §2); that is a lossy handoff, and the cheap fix is to stop losing information, not
to collapse the pipeline. Keeping the pool alive gets the whole benefit without touching the boundary.

Three things argue against merging the labelling into the DFS:

**Symmetry.** A DFS that assigns logical meshes as it places will, on failure, re-derive the same region
set repeatedly with permuted labels; it has no mechanism to recognise it has already tried this
equivalence class. SAT rejects the class at once via learned clauses. Nogoods recorded against an
*unlabelled region set* are also strictly stronger than nogoods against a labelled assignment.

**Intra-mesh failures are frequently not the placement's fault.** Channel-count validation, MGD
pinnings, and rank binding can all fail on a region set that is perfectly good. The cheap repair is to
permute labels; only when *every* labelling fails is the placement to blame. A single-pass DFS cannot
tell those apart and will re-place when it should have relabelled.

**Blast radius.** `map_multi_mesh_to_physical` has one production caller
(`tt_metal/fabric/topology_mapper.cpp:531`) but roughly thirty-five call sites in
`tests/tt_metal/tt_fabric/fabric_router/test_topology_mapper_utils.cpp`, and `map_multi_mesh_to_physical_n`
plus `TopologyMappingEnumerationSession` are documented public API sharing the same SAT path
(`topology_mapper_utils.hpp:591–637`). Deleting the stage means rewriting the mapper's test suite and
providing DFS equivalents for the enumeration entry points.

So: two passes, coupled as a coroutine rather than pipelined.

```
PASS 1  adjacency-guided DFS  (§5)
        → a chip-disjoint, seam-feasible set of physical regions
        → labels only where a pinning already forces one
                    │
                    ▼
PASS 2  map_multi_mesh_to_physical + intra-mesh solve
        → completes the labelling, embeds each mesh, checks exit nodes,
          channel counts, rank bindings, per-chip MGD pinnings
```

Pass 2 is near-identity work in the common case, because pass 1 already guaranteed the seams. It exists
to permute when intra-mesh rejects something, and to check the constraints the DFS deliberately cannot
see. §6(d) proposes — but does not adopt — a feedback edge back into pass 1.

### (b) The rule that decides scope

> The DFS decides properties of the **region set**. Pass 2 decides properties of the **labelling**.

Everything below follows from that one line, and it is the test to apply to anything proposed for the
DFS later. A property is a region-set property if it can be evaluated without knowing which logical mesh
sits on which region — footprint, disjointness, physical adjacency. A property is a labelling property
if changing which mesh sits where can change the answer — rank binding, exit-node identity, intra-mesh
fit.

### (c) In scope — the DFS determines these

| Decision | Why it belongs to the region set |
| --- | --- |
| Which PGD grouping variant (shape/topology) backs each mesh | Determines the footprint itself |
| Where each grouping lands on PSD chips | This *is* the region |
| Chip disjointness across all placed regions | Occupancy mask; native to a backtracking search |
| Seam existence between placed region pairs | Purely physical: does an ethernet link cross? (§3) |
| PGD↔MGD shape compatibility | Filters which candidates can serve which mesh |
| **MGD pinnings, mesh-level projection** | See (d) — a pinning is a *given*, not a search decision |
| Host locality | Soft only: value-ordering bias (§5(e)), never a constraint |

### (d) MGD pinnings: a given, not a decision

Pinnings are the one case where partial labelling is free, which is why they belong in the DFS despite
labelling being pass 2's job. A `PinningConstraint` names both a fabric node and an ASIC position, so it
answers two different questions at two different granularities:

| Granularity | Question answered | Consumer |
| --- | --- | --- |
| Mesh-level projection | *Which region must host logical mesh A?* — whichever region contains that ASIC position | **DFS**: seeds and pre-labels that mesh |
| Chip-level | *Which ASIC does fabric node (A, chip 3) occupy?* | Intra-mesh solve, `add_pinning_constraints` (`topology_mapper_utils.cpp:3616`) |

The DFS takes only the projection. It performs no search to obtain it — the pinning states it — so the
"DFS does not label" rule is intact. The payoff is large: a pinned mesh is a fixed anchor, which removes
the seed symmetry problem (§5(g)) and gives the search a most-constrained starting point for free.

The DFS output is therefore a region set with a *partial* labelling: pinned meshes bound, everything
else anonymous. Pass 2 completes it.

### (e) Out of scope — the DFS must not determine these

| Deferred to pass 2 | Rationale | Where it happens |
| --- | --- | --- |
| Labelling of unpinned meshes | Labelling property; SAT handles permutation and symmetry far better (§4(a)) | `map_multi_mesh_to_physical` |
| Exit node selection — *which chip* faces the neighbour | Labelling property, and orientation is a joint constraint across all of a mesh's neighbours | `add_exit_node_constraints` (`3023`) |
| MGD intra-mesh embedding, fabric node → ASIC | Independent of seam feasibility at region granularity | `solve_topology_mapping`, `3652` |
| Per-chip MGD pinnings | Chip granularity; see (d) | `add_pinning_constraints` (`3616`) |
| **Rank bindings** | A rank binding constrains a *label* — "mesh A lives on rank 3". The DFS has no labels for unpinned meshes, so it cannot evaluate it without doing pass 2's job | Rank-binding identity path in `build_inter_mesh_constraints` |
| Channel counts, STRICT vs RELAXED | Deferred by §5(j); existence-only is the weakest predicate that kills #54623 | Inter-mesh validation |
| PGD↔MGD node orientation | Frozen pre-placement and seam-blind; see (g) | `add_pgd_pinning_preferred_constraints` (`2957`), soft |

Rank bindings deserve the explicit note because they look like a placement property. They are not. The
most a region-set pass could do is a counting feasibility check — the multiset of host ranks covered
must be able to satisfy the demand — and that is a Hall-type bound, not a binding. It is not worth
building: rank-binding failures are exactly the case pass 2 repairs by relabelling, at no cost to the
DFS.

### (f) One seam predicate, two consumers

The seam predicate of §3 and the exit-node domain computed by `add_exit_node_constraints` are the same
function evaluated at different times. That function builds, for a region pair, the set of ASICs in one
region having links toward the other:

```3060:3062:tt_metal/fabric/topology_mapper_utils.cpp
            // Use the mapped physical mesh ID as the key (which is the same as dst_exit_node.mesh_id)
            valid_physical_exit_nodes_by_mesh[dst_exit_node.mesh_id].insert(src_exit_node.asic_id);
```

The DFS needs the same set — non-empty is exactly its seam test. **Extract it once and share it.** The
DFS uses non-emptiness as its generator (§5(c)); pass 2 uses the set itself as the required domain.

Two consequences, both good. The DFS structurally cannot hand pass 2 a region pair that will fail the
exit-node check, so the `"exit node constraints cannot be satisfied"` rejection at
`topology_mapper_utils.cpp:3611` becomes unreachable from this path. And the two stages stop carrying
two definitions of a usable seam that can drift apart.

### (g) Suppress the seam-blind PGD preference

The PGD↔MGD node correspondence is composed at *match* time, before placement:

```1113:1115:tt_metal/fabric/physical_grouping_descriptor_matching.cpp
                    committed.mesh_node_to_asic_position =
                        compose_mesh_node_to_asic_position_from_pgd_match(committed, match.mapping.target_to_global);
                    return committed;
```

At that moment nobody knows where the region will land or who its neighbours will be, so the orientation
it encodes is seam-blind by construction, and every placement of that grouping inherits it. It reaches
the intra-mesh solve as a soft preference, deliberately added after the hard constraints
(`topology_mapper_utils.cpp:3632–3645`), so it cannot cause a failure — but on a mesh that has seams it
biases toward a neighbour-ignorant orientation that the hard exit constraints must then undo.

Proposed: skip `add_pgd_pinning_preferred_constraints` for meshes with a non-empty exit-node graph, and
keep it for interior meshes where it is a genuine tie-break. Small, independently testable, and it does
not depend on the DFS landing.

**Do not promote this pinning to a hard constraint.** A 4x4 MESH has eight topologically equivalent
orientations; today the solver picks whichever satisfies the seams. Hard-pinning would pick one blind
and convert a cheap local repair into a whole-placement backtrack.

## 5. Design — adjacency-guided mixed-shape DFS

Keep Phase A untouched and delete Phase B from this path. Index the pool, then grow one placement over
all shapes at once, ordered by the MGD's own graph.

```
TODAY
  per shape S:
     enumerate_distinct_placements_for_grouping(S)  →  pool_S  (≤ 1024)
                                                         │
                                       solve_set_packing │  maximize Σ chips covered
                                                         ▼
                                                   frozen tiling_S      ◄── seam info dies here
     ────────────────────────────────────────────────────────────────
  per shape:      SAT embeds logical_S into tiling_S     (same-shape edges only)
  across shapes:  DFS picks 1 solution per shape, chip-disjoint   (no seam check at all)

PROPOSED
  per shape S:
     enumerate_distinct_placements_for_grouping(S)  →  pool_S  (unchanged, ≤ 1024)
                                                         │
                    keep the pool alive; index it by chip and by adjacency
                                                         ▼
  ONE adjacency-guided DFS over the whole MGD mesh graph, all shapes, all edges
     →  logical MeshId → PsdPlacement, disjoint and seam-feasible by construction
```

The intra-mesh solve is not fused into this. Seam feasibility at region granularity is answered from
`flat_graph` alone, and the answer does not depend on how chips inside a region get assigned.

### (a) Data structures

One global candidate index space across all shapes:

```cpp
struct Candidate {
    std::string shape;                    // MGD mesh descriptor name
    std::vector<std::uint64_t> chip_bits; // asic_word_count words, same encoding as group_bits_by_name
    ::tt::tt_fabric::PsdPlacement placement;
};

std::vector<Candidate>                                  pool;          // all shapes, one index space
std::unordered_map<std::string, std::vector<uint32_t>>  pool_by_shape;
std::vector<std::vector<uint32_t>>                      pool_by_chip;  // dense ASIC index → candidate ids
std::vector<std::vector<uint32_t>>                      touches;       // candidate → candidates with ≥1 crossing link
std::vector<boost::dynamic_bitset<>>                    touches_bits;  // same, for O(1) membership
```

`touches` must **not** be built by iterating candidate pairs. Drive it from the physical edge list:

```
for each candidate p:
  for each chip c in p.footprint:
    for each flat_graph neighbour d of c:
      for each candidate q in pool_by_chip[d]:
        touches[p] += q
```

Cost is `Σ_p |p| × degree × avg_candidates_per_chip`, not `O(P²)`. For the router-pipeline MGD (352
chips, three shapes, pool ≤ 1024 each) that is single-digit millions of increments. `touches_bits` at
3072 × 3072 bits is ~1.2 MB, which buys `O(1)` membership during the DFS.

### (b) Which logical mesh to place next

VF2-style: keep the frontier connected, and be most-constrained-first within it. The order is dynamic
because domain sizes change as chips get consumed.

**Seed.** If the MGD has pinnings, seed there — the pinning *is* the anchor and the symmetry problem
disappears. Otherwise pick maximum degree, tie-break by largest shape, then lowest MeshId for
determinism. For the router-pipeline MGD that selects the degree-4 `4x4` router mesh, which is correct:
it is the hardest mesh to place and three of its four seams cross shapes.

**Subsequent.** Among unplaced meshes with at least one placed neighbour, choose:

1. most placed neighbours (maximum constraint from the frontier),
2. then smallest current domain (fail-first),
3. then largest shape,
4. then lowest MeshId.

If the MGD mesh graph is disconnected, each component gets its own seed; components interact only
through the occupancy bitmask.

### (c) Generating adjacent candidates

This is the heart of it. For the next mesh `L`:

```
domain(L) = { c ∈ pool_by_shape[shape(L)]
              : chip_bits[c] ∩ occupied == 0
              ∧ ∀ placed neighbour N of L : touches_bits[assign(N)][c] }
```

Implementation order matters. Start from the *smallest* `touches[assign(N)]` list over `L`'s placed
neighbours, then filter that list by shape, by disjointness, and by membership in the remaining
neighbours' `touches_bits`. Iterating the smallest adjacency list rather than the shape pool is what
keeps this cheap: a placed region touches tens of candidates, while a shape pool holds up to 1024.

A newly seeded mesh has no placed neighbour, so its domain is the whole shape pool minus occupancy —
that is the expensive case, and it happens once per connected component.

```
MGD fragment:  L0(4x1) ─ L1(4x2) ─ L2(4x4) ─ L3(4x2) ─ L4(4x1)
                                     │
                                  L5(4x1)          ← router, degree 4

step 1: seed at L2 — largest shape, highest degree
        ┌────────┐
        │   L2   │
        └────────┘
step 2: frontier {L1, L3, L5}; each domain = pool_shape ∩ touches[L2] ∩ ¬occupied
        ┌────┐┌────────┐┌────┐
        │ L1 ││   L2   ││ L3 │
        └────┘└────────┘└────┘
                   │
                ┌────┐
                │ L5 │
                └────┘
step 3: extend outward along the pipeline: L0 from L1, L4 from L3
```

### (d) Forward checking

After tentatively assigning `L → c`, before recursing:

- `occupied |= chip_bits[c]`
- recompute `domain(M)` for every unplaced neighbour `M` of `L`; if any is empty, fail immediately
- **singleton conflict**: if two frontier meshes have the same single candidate, fail
- **union bound**: for the frontier set `F`, if `|⋃ domain(M)| < |F|`, fail

The last two are a cheap Hall approximation, both `O(|F| × domain)`. Full Hall matching is not worth it
until measurement says otherwise. In the 8-chip example above, the union bound alone prunes the two bad
seeds without ever making an assignment.

### (e) Which candidate to try first

Least-constraining-value, so the DFS keeps its options open:

1. prefer candidates leaving the most live candidates for `L`'s unplaced neighbours — approximate by
   counting members of `touches[c]` that are still disjoint from `occupied` and of a shape some unplaced
   neighbour needs,
2. then prefer fewer distinct hosts spanned (locality). `PackingCandidate::host_count` already computes
   this in the packer (`1780–1786`); `PsdPlacement` does not carry it, so either add the field or
   recompute from `get_host_name_for_asic`,
3. then lowest pool index, for determinism.

### (f) Backtracking, budget, fallback

Chronological backtracking to start. On budget expiry or exhaustion, **fall back to the existing Phase
5/6 path** rather than throwing — this plan should not be able to regress an MGD that maps today.

**Budget on search nodes expanded, not wall-clock**, with a generous wall-clock value kept only as a
coarse backstop against a pathology the node count fails to catch. The reason is reproducibility rather
than correctness: a wall-clock cutoff makes the search's outcome depend on machine load, so a CI job
that maps under load and fails when idle becomes flaky in a way that is painful to diagnose. A node
count makes a failure reproducible from the MGD and PSD alone. `kSetPackingBudget` (5s) is the existing
precedent for the wall-clock form and is why it is worth stating the departure explicitly.

**The budget can be generous.** The solve is not replicated across ranks. `TopologyMapper::build_mapping`
guards it with `if (generate_mapping_locally_ || my_rank == control_host_rank)`
(`tt_metal/fabric/topology_mapper.cpp:443`); rank 0 solves alone and broadcasts the result
(`571–580`), while every other rank waits in `receive_chip_info_from_host` (`583`). So search cost is
paid once per job on one host, not once per rank, and there is no cross-rank divergence hazard from a
non-deterministic cutoff — the constraint is purely about reproducible debugging.

Keep the deepest partial assignment reached. That is the diagnostic payload, and it is much better than
a SAT unsat core:

> placed 47 of 69 meshes; L48 (`S4x2`) needs a region adjacent to L47 at chips [...]; 0 of 812 `S4x2`
> candidates are both disjoint and adjacent

Conflict-directed backjumping and randomized restarts are the obvious escalations if chronological
backtracking thrashes. Do not build them up front.

### (g) Symmetry breaking

Galaxies are interchangeable, so an unpinned seed can rediscover the same dead end once per symmetric
image — a 4×–24× multiplier on every failure at SC36 scale.

Use symmetry only for **ordering**, never for pruning, so no solution can be lost: group seed candidates
by an invariant signature (sorted multiset of host/tray labels plus local degree profile), try one
representative from each signature class first, then the rest. If measurement later shows the classes
really are isomorphic, promoting this to a prune is a one-line change — but it needs evidence first,
because the machine is not guaranteed vertex-transitive under that signature.

### (h) Delete `kMaxPlacementsPerGrouping`

The cap is 1024 placements per grouping. It exists to stop Phase B's set packing from choking on a big
pool. Phase B is gone from this path, so the reason for the cap is gone too. Under this plan the pool
*is* the search space, and silently truncating it can turn a solvable MGD into a reported failure.

**Delete it, and pass `0` instead.** `solve_topology_mapping_n` treats `0` as "enumerate everything",
capped by its own backstop `kTopologyMappingEnumerateSolutionsHardCap = 500000`
(`topology_solver.tpp:44`, applied at `1160–1162`). There is also already an
`solve_topology_mapping_all` entry point that does exactly this.

**Real pools are tiny, so this costs nothing.** Every PGD grouping node is pinned to an exact
`(asic_location, tray_id)`:

```proto
{ id: 0 location { asic_location: ASIC_LOCATION_5 tray_id: TRAY_1 } },
{ id: 1 location { asic_location: ASIC_LOCATION_6 tray_id: TRAY_1 } },
```

`build_pgd_to_psd_constraints` turns those into `add_required_trait_constraint` domain restrictions
(`1467–1483`), so a grouping node can only land on a chip carrying that exact label — roughly one chip
per host. A `tray_1` grouping is therefore rigid: choose the host and the placement is decided.
Composite shapes are built from those same labelled trays. Pools land in the tens to low hundreds, far
below 1024, so the cap almost certainly never fires today.

Enumeration is also incremental AllSAT — `SatSearchEngine::search_n` keeps one solver, encodes once, and
appends a blocking clause per model — so the cost of enumerating a small set completely is small.

**No new bookkeeping.** Do not replace the cap with truncation tracking, a wall-clock guard, or a config
knob. The 500k backstop is orders of magnitude above any real pool, and if it ever fires the solver
already says so. Adding machinery for a case that cannot happen is the accretion this plan is trying to
remove.

### (i) Contingency: anchored placement, expressed with existing constraints

Only needed if some future PGD leaves slot labels unspecified, so domains do not collapse and the pool
is genuinely large. In that case, do not enumerate every placement of a shape across the whole machine.
Ask the solver for placements *next to where we already are*.

Both halves of that question are already expressible — no solver changes, no new primitives:

```cpp
MappingConstraints<uint32_t, AsicID> seed;

// 1. Stay off chips we have already used.
seed.add_forbidden_constraint(all_grouping_nodes, occupied_chips);

// 2. Touch each already-placed neighbour region R: at least one grouping node must land
//    on a chip that has an ethernet link into R.
for (const auto& R : placed_neighbour_regions) {
    seed.add_cardinality_constraint(all_grouping_nodes, open_boundary(R), /*min_count=*/1);
}

// 3. Hand it to the existing path, which already accepts initial constraints.
auto constraints = build_pgd_to_psd_constraints(grouping, physical_graph, psd, std::move(seed));
```

with `open_boundary(R) = { d : ∃ c ∈ R with (c,d) ∈ flat_graph, d ∉ R }`.

Constraint 2 *is* the seam predicate from §3, and `min_count = 1` is exactly the existence-only choice
made in §5(j) — the constraint form and the design decision agree for free.

Why this is cheap. Cardinality is real CNF, step 8 of `topology_sat_encode_hard_constraints`
(`topology_solver_sat.cpp:1179`), not a post-hoc check. Its literals are collected *after* domain
filtering and AC-3 (`1074–1091`), so the trait constraints have already shrunk each node's domain to
about one chip per host. A `4x4` grouping against a ~30-chip boundary produces tens of literals against
a cap of `kMaxCardinalityLiterals = 4096`, and overflow fails loudly rather than silently.

Why it is the contingency and not the default. `topology_sat_encode_hard_constraints` rebuilds domains,
runs AC-3, and emits CNF on every call. One anchored solve is small; thousands of them inside a
backtracking search is a different cost, and it is the aggregate that would need measuring. With the cap
deleted, enumerating the complete pool once and letting the DFS work on set operations avoids per-node
solver calls entirely.

Two honest limits. `min_count` counts grouping *nodes* landing in the boundary, not ethernet links
crossing it, so it is a lower-bound proxy if channel counts are ever wanted. And the seed is passed
before `build_pgd_to_psd_constraints` adds its trait constraints, so verify the pending-forbidden path
(`topology_solver.hpp:792`) behaves when `valid_mappings_` is not yet populated.

### (j) Deferred: channel counts and STRICT/RELAXED

> **TODO — defer channel counts for the placement-time seam predicate.**
> Start at **existence only**: `link_count(Ri, Rj) >= 1`, regardless of the seam's declared
> `channels { count: N }`. Push "≥ the requested channel count" and any STRICT/RELAXED distinction to a
> follow-up, and only add it if a real MGD is shown to need it.
>
> Rationale: existence is the weakest predicate that still kills the #54623 failure mode (a stranded
> seam has *zero* links, not too few), and a weaker predicate means larger domains — the DFS reaches a
> complete assignment sooner and cannot be driven infeasible by a channel shortfall on an otherwise
> correct placement. It is also cheaper to index: `touches` can be built with a short-circuit on the
> first crossing edge, whereas counting must walk every crossing edge for every pair.
>
> Worth knowing before revisiting: the surrounding solves already disagree about strictness. Per-shape
> placement enumeration runs **STRICT** (`pull_next_solution`, `topology_mapper_utils.cpp:1162`), while
> the inter-mesh solve defaults to **RELAXED** (`determine_inter_mesh_validation_mode`, `2722–2727`,
> overridable via `TopologyMappingConfig::inter_mesh_validation_mode`). The natural default is to follow
> the inter-mesh mode being protected — existence, matching RELAXED — and to honour the requested count
> only when the caller explicitly asked for STRICT.
>
> The cost, stated honestly: a placement can pass an existence-only predicate and still fail later under
> STRICT inter-mesh validation because the seam resolved to fewer channels than requested. That trades
> early precise failure for a faster, more permissive search. Right while the common bug is "zero
> channels"; wrong once it becomes "not enough channels", which is the signal to pick this back up.
>
> If it is picked up, `touches` becomes a count rather than a bool and the domain filter compares
> against the seam's declared count. Keep the accessor count-shaped (`std::size_t link_count(a, b)` with
> an early-out at 1) so that is a threshold change, not a data-structure change.

### (k) Worked trace on the 8-chip example

```
order: S4_a (largest shape, degree 2) → S2_a → S2_b

try S4_a = [c0..c3]
    S2_a domain = {[c4,c5]}      ← adjacent to c3, disjoint
    S2_b domain = {[c4,c5]}
    union bound: |F| = 2, |⋃ domain| = 1  → prune, no assignment made

try S4_a = [c1..c4]
    S2_a domain = {[c5,c6]} ; S2_b domain = {[c5,c6]}  → prune

try S4_a = [c2..c5]
    S2_a domain = {[c0,c1]} ; S2_b domain = {[c6,c7]}  → disjoint, assign, done
```

Three search nodes. The propagation does the work, not the enumeration.

## 6. DFS optimizations

### (a) Status

**Open — the option space below is recorded, not decided.** Nothing here is committed to the v1
implementation. The design in §5 is deliberately the lightest thing that can work: connected growth,
domain filtering, a union bound, chronological backtracking. Every entry below is a *response to a
measured symptom*, and the symptom has to show up first.

The guard rail: v1 ships with node-count, wall-clock, and deepest-depth counters. An optimization is
adopted when a validation MGD (§10) demonstrates the symptom it targets, and not before. This section
exists so that when a symptom appears the response is already thought through.

### (b) Candidates, grouped by the symptom they answer

| # | Optimization | Symptom it answers | Cost / risk |
| --- | --- | --- | --- |
| 1 | **State memoization** — hash on `(placed mesh set, occupied chip bitset)`, skip states already refuted | The same partial state is reached by different assignment orders | Likely dead weight; see (c) |
| 2 | **Capacity counting bounds** — free-chip popcount, and per-shape remaining-demand vs supply | Search continues below a state that provably has no room left | Near-free with incrementally maintained counts; strictly weaker than Hall |
| 3 | **Anchored enumeration** (§5(i)) — generate candidates next to the frontier instead of filtering a global pool | Pools large enough that global filtering dominates | Per-node solver calls; already documented as a contingency |
| 4 | **Conflict-directed backjumping + nogood learning on minimal conflict subsets** | Chronological backtracking thrashes on a long pipeline | One decision, not two — both need the same conflict-provenance tracking |
| 5 | **Full Hall matching** instead of the §5(d) union bound | Union bound admits states that die a few levels deeper | Bipartite matching per node; §5(d) says explicitly to wait for evidence |
| 6 | **Bidirectional growth** — seed both ends of a pipeline and meet in the middle | Long path MGDs walk far before finding a dead end | Two frontiers to merge; already raised in §12 |
| 7 | **Seed-class budget rotation** — give each signature class (§5(g)) a node budget, rotate on exhaustion | Symmetric dead ends rediscovered per image on unpinned MGDs | Completeness survives: a class is abandoned temporarily, not pruned |
| 8 | **Incremental domain maintenance** with an undo trail, instead of recomputing domains per node | Profiling shows constant factor, not search shape, is the cost | Standard CSP technique; trail bookkeeping |
| 9 | **Randomized restarts** | Heavy-tailed runtime; one bad early choice dominates | Non-reproducible failures unless the seed is fixed; see §5(f) |

**Not on this list, deliberately.** A shared seam cache is not an optimization — `touches` is a static
property of the pool and the PSD, independent of search state, so §5(a) already precomputes it in full.
The only lazily computed piece is the exit-domain *set* handed to pass 2 (§4(f)), needed for the region
pairs actually chosen and therefore O(seams). There is nothing left to cache.

### (c) Why state memoization is probably dead

Memoization pays when the same subproblem is reachable along many paths. It is not, here. The variable
order in §5(b) is a deterministic function of the current state, so any given partial assignment can be
built in exactly one order and the search is a tree rather than a DAG — the table would cost a bitset
hash per node and never hit.

The version that does pay is different in kind: record nogoods on a **minimal conflict subset** rather
than the full state. A subset like "these three regions together strand mesh L7" recurs under many
completions of everything else, so it prunes broadly. That is candidate 4, and it is why backjumping
and nogood learning are listed as one decision.

Recorded rather than deleted, because the argument depends on the variable order staying deterministic.
If ordering ever acquires a tie-break that is not a pure function of state, this needs revisiting.

### (d) Proposed, not adopted: pass-2 → pass-1 loopback

Recorded as a possibility. **Not part of the current design.**

Pass 1 cannot see the labelling properties in §4(e), so pass 2 can still fail on a region set the DFS
considers valid — a rank binding that no labelling satisfies, or an intra-mesh embedding that fails for
every permutation. Today that is terminal for the placement. The proposal is to make it recoverable:

```
pass 1 (DFS)  ──── region set ────►  pass 2 (label + intra-mesh)
     ▲                                        │
     └──── nogood on the REGION SET ◄─────────┘
              (only after ALL labellings fail)
```

The DFS becomes a resumable generator rather than a one-shot call, and resumes from its trail instead of
restarting. The nogood is recorded against the *unlabelled* set, which is what makes it strong — it
refutes every labelling at once, and §4(a) already argues that this granularity is the reason to keep
labelling in SAT.

**Why it is attractive.** It closes the last completeness gap without moving any decision into the DFS,
and it costs nothing on the common path, where pass 2 accepts the first region set.

**Why it is not adopted.** It converts a clean stage boundary into a control-flow cycle, which is the
most expensive kind of coupling to debug and the hardest to reason about when it misbehaves. It requires
pass 2 to distinguish "no labelling exists" from "this labelling failed", which is not a distinction the
current code makes. And it needs a global budget across the cycle, otherwise a pathological MGD can
alternate between passes indefinitely.

**The cheaper alternative, and the default.** Fail with the §5(f) diagnostic — deepest partial
assignment plus the specific mesh and seam that could not be satisfied — and let a human read it. If the
counters show pass-2 rejection of a DFS region set is common rather than rare, revisit.

**Decision gate.** Instrument first: count how often pass 2 rejects a DFS-produced region set for a
labelling-property reason. If that is ~0 across the MGD corpus, this stays unbuilt permanently. That
same counter is the falsification test for §4(a) — if pass 2 *never* permutes, the labelling stage is
dead weight and the argument for two passes weakens.

## 7. Ownership and function passing

| Concern | Owner | Notes |
| --- | --- | --- |
| Candidate pool (unpacked) | New `enumerate_all_candidates_in_psd(groupings, psd, flat_graph)` in `physical_grouping_descriptor_matching.cpp` | Extract Phase A (`1730–1789`) into its own function; `solve_for_many_groupings_to_psd_heterogeneous` calls it too, so the packer keeps working unchanged |
| Placement cap | `kMaxPlacementsPerGrouping` (`1663`) deleted; `enumerate_distinct_placements_for_grouping` passes `0` | Inherits the solver's own 500k backstop. No replacement bookkeeping (§5(h)) |
| Anchored placement | New `enumerate_placements_anchored` beside `enumerate_distinct_placements_for_grouping` (`1514`) | Seeds `initial_constraints` with a forbidden set plus one at-least-1 cardinality per placed neighbour; no solver change. Contingency only (§5(i)) |
| Pool indices | New `PlacementIndex` struct, anonymous namespace in `topology_mapper_utils.cpp`, built in Phase 3 beside `group_bits_by_name` (`927`) | Owns `pool`, `pool_by_shape`, `pool_by_chip`, `touches`, `touches_bits`; built from `flat_graph`, which Phase 3 already holds |
| Full logical mesh graph | `get_requested_intermesh_from_mgd` result used **unfiltered** (`1007`) | The existing `build_mgd_mesh_level_subgraph_for_mesh_descriptor_name` (`1037`) shape filter is simply not applied on this path |
| Shape of each logical mesh | `logical_mesh_id_to_mgd_instance_name` (`1522`) + `get_valid_groupings_for_mgd` keys | Already the key space `placements_by_shape` uses |
| The search | New `AdjacencyGuidedPlacementSearch` replacing `DisjointPackingSearch` (`1085–1458`) | Takes `const PlacementIndex&`, the logical graph, and the shape map by const ref; owns only `occupied`, `assign`, and the trail |
| Result assembly | Phase 7 (`1480–1515`), unchanged | Search emits `logical MeshId → PsdPlacement`; disjointness by construction keeps the one-ASIC-to-one-MeshId assumption in `build_hierarchical_from_flat_graph` (`2433–2437`) sound |
| Fallback | Existing Phases 5–6, kept | Entered on budget expiry or exhaustion |
| Diagnostics | Replaces the `TT_THROW` at `1472` | Reports the deepest partial assignment and the specific mesh and seam that could not be satisfied |

`PlacementIndex` and the search stay in the anonymous namespace in `topology_mapper_utils.cpp`. Nothing
here needs to appear in a public header.

## 8. What this displaces

| Phase | Fate |
| --- | --- |
| 3 — `find_all_in_psd` | Replaced on this path by the Phase A pool build. `find_all_in_psd` itself stays for its other callers |
| 4 — single-shape fast path | **Keep.** One shape means no cross-shape seams and no reason to search |
| 5 — per-shape `MeshEnumState` + SAT | Becomes fallback only |
| 6 — `DisjointPackingSearch` | Becomes fallback only |
| 7 — result assembly | Unchanged |

The per-shape SAT is subsumed rather than lost: it embedded the same-shape subgraph into a fixed tiling,
and the new search embeds the *whole* graph into the whole pool.

## 9. Interaction with Plans 1 and 2

Complementary, and the relationship gets stronger rather than weaker.

Phase 7 re-keys placements under the logical MeshId, so after this search the physical graph handed to
`map_multi_mesh_to_physical` is already keyed by the intended logical identity. The inter-mesh solve's
remaining freedom is exactly the permutation among same-shape regions — which is precisely what
[Plan 1](TOPOLOGY_MAPPER_PLAN_1_PGD_SHAPE_INTERMESH_CONSTRAINTS.md)'s shape-class domain filter removes.
Plans 1 and 3 together should make the inter-mesh solve close to deterministic, which in turn shrinks
the retry loop that [Plan 2](TOPOLOGY_MAPPER_PLAN_2_INCREMENTAL_INTERMESH_SOLVE.md) optimizes.

Plan 2's value therefore drops further if this lands. It remains an encode-once performance win, not a
correctness fix.

## 10. Validation

- **The 8-chip chain from §2**, as a direct unit test in
  `tests/tt_metal/tt_fabric/fabric_router/test_topology_mapper_utils.cpp`. It fails today and must pass
  after. This is the sharpest possible regression test because the frozen tilings make it provably
  unsatisfiable on the current path.
- **The issue's 4-slot cycle**: 4 physical slots in a cycle, 4 logical meshes alternating two shapes;
  assert the interleaved placement, not the packed one.
- **`bh_glx_2branch_mesh_per_stage_router_pipeline.textproto`**: assert every declared MGD boundary
  resolves to ≥1 channel. The degree-4 router mesh is the sharpest case, three of its four seams being
  cross-shape.
- **`llama_8b_4galaxy_unpinned_mesh_graph_descriptor.textproto`**: no pinnings, so it exercises the
  unpinned seed and symmetry-ordering path from §5(g).
- **Pool sizing**, per §5(h): log the per-shape pool size for the validation MGDs once the cap is
  deleted, confirming pools stay in the tens-to-low-hundreds range the trait constraints predict.
- Extend the `bh-heterogeneous` group in
  `tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh` with a boundary-resolution assertion
  rather than only `TestGalaxyLayoutCheck`.
- **Counters, shipped with v1** (§6(a)): search nodes expanded, wall-clock, deepest depth reached, and
  how often pass 2 rejects a DFS region set for a labelling-property reason. These are the evidence
  gates for every §6(b) optimization and for the §6(d) loopback; without them each is a guess.
- **Pinned vs unpinned pairing.** Run `llama_8b_4galaxy_unpinned_mesh_graph_descriptor.textproto`
  against its pinned sibling and compare node counts. The pinned MGD exercises the §4(d) seed path and
  should be dramatically cheaper; if it is not, the pinning projection is not being applied.

## 11. Risks

| Risk | Mitigation |
| --- | --- |
| DFS thrashes on a hard instance; no completeness guarantee within budget | Node-count budget plus fallback to the existing Phase 5/6 path (§5(f)) |
| A budget expiry is not reproducible, making a CI failure flaky | Budget on nodes expanded, not wall-clock, so the outcome depends only on the MGD and PSD (§5(f)) |
| Pool truncation makes a satisfiable MGD report failure | Delete the cap (§5(h)); trait constraints keep real pools far below it anyway |
| A future PGD leaves slot labels unspecified, so pools are large | Anchored placement via existing constraints (§5(i)) |
| Symmetry multiplies dead ends on unpinned MGDs | Signature-ordered seeds (§5(g)); pinned MGDs are unaffected |
| Loss of an explicit objective — the search returns *a* solution, not a good one | Host-locality as value-ordering bias (§5(e)). Max coverage was the wrong objective anyway |
| `touches` build or memory blows up at SC36 scale | Edge-list-driven construction, not pairwise (§5(a)); measure before assuming |
| Pass 2 rejects a region set the DFS considers valid, and there is no recovery | Accepted for v1: fail with the §5(f) diagnostic. §6(d) records the loopback as the escalation, gated on the §10 counter |
| Scope creep pulls labelling properties into the DFS one at a time | §4(b) is the single test to apply; §4(e) records each exclusion with its reason so the argument does not have to be re-litigated |

## 12. Open questions

- **What should the seed be for an unpinned MGD?** Most-constrained-mesh-first is right, but its
  *placement* choice is unconstrained, so symmetry ordering carries the whole load. Pinned MGDs get this
  for free. This is the weakest part of the design.
- **Is chronological backtracking enough for 69 meshes?** A pipeline is nearly a path, which is the
  friendly case, but it is also the case where you can walk a long way before discovering a dead end.
  Bidirectional growth from both ends of the pipeline is a cheap thing to try before backjumping.
- **Should the fallback ever be removed?** Keeping two placement paths is a maintenance cost. Proposed
  trigger for deleting Phases 5–6: the new search handles every MGD in the `bh-heterogeneous` group
  within budget for a full release cycle.
- **When does the deferred channel-count check get picked up?** Proposed: the first MGD whose seams
  resolve to a non-zero but insufficient channel count under STRICT inter-mesh validation (§5(j)).
- **Where does #52016 (pipeline-stage adjacency in MGD) intersect?** If stage adjacency becomes explicit,
  seams gain direction — ordered pipeline hops rather than undirected boundaries — and both the ordering
  heuristic in §5(b) and the predicate in §3 could use it.
- **Does pass 2 ever actually permute?** The whole two-pass argument in §4(a) rests on relabelling being
  a real repair. If the counter in §10 shows pass 2 always accepts the DFS labelling unchanged, the
  labelling stage is dead weight and collapsing the passes becomes defensible — with evidence rather
  than as a bet.
- **Is a rank-count feasibility bound worth adding to the DFS?** §4(e) says no: a Hall-type check on
  host-rank supply versus demand is the most a label-free pass can do, and pass 2 repairs those failures
  by relabelling anyway. Revisit only if rank-binding rejections turn out to dominate pass-2 failures.
- **Does the seam-blind preference suppression in §4(g) stand alone?** It needs neither the DFS nor
  Plan 1, so it could ship first as an independent change. Worth confirming it does not regress
  interior-mesh determinism, which is the one thing the preference currently buys.
- **Does the joint SAT from #54623 still have a role?** It becomes the escalation if the DFS proves
  inadequate, not the first move. The DFS is strictly less new machinery: encoding chip-disjointness in
  the SAT needs an at-most-one primitive that `MappingConstraints` does not have today
  (`add_cardinality_constraint` is at-*least*-N only), whereas disjointness is native to a search that
  carries an occupancy mask.
