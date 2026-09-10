# Plan 4 — Two-layer joint placement: precomputed candidates + SAT master problem

**Hoist the inner solve out of the search.** Enumerate each PGD grouping variant's placements once
against the full fabric, then choose one seat per mesh with a single SAT solve over those candidates.

**Status: §3–§5 implemented** (`start_sat_placement` in `physical_grouping_descriptor_matching.cpp`),
and the default path. `TT_METAL_PLACEMENT_SOLVER` selects `sat`, `dfs`, or `auto` (default: SAT first, the
Plan 3 adjacency-guided DFS only when SAT fails without a trustworthy UNSAT, i.e. with truncated candidate
lists). Phase 2/3 (rewiring the DFS onto the master list, MRV) are not done; the DFS is unchanged.

Implementation notes that differ from the sketches below:
- Candidates are deduplicated by footprint *across variants of one definition*, first variant wins
  (torus variants of one pinning share a footprint and would otherwise be pure symmetry for the solver).
- Seam link counts are computed once per unordered pair of *definitions* (`SeamLinkMatrices`) and
  reused by every mesh-level edge joining instances of those definitions.
- Trait-free variants (the MGD fallback, or an unpinned PGD grouping) are detected structurally
  (`variant_is_trait_free`) rather than by list position, and capped at 8192 candidates per definition.
- Under a RELAXED policy the strict-seam tier is solved with a conflict budget; the relaxed tier is not.

**Priority: 1 if Plan 3 cannot place Gemma; 3 otherwise.** §1 is the evidence.

Sibling plans: [Plan 3 — connectivity-aware PGD placement](TOPOLOGY_MAPPER_PLAN_3_CONNECTIVITY_AWARE_PGD_PLACEMENT.md).

> **Goal.** Split placement into a *geometry* layer that answers "where can this mesh sit, ignoring
> everyone else" and a *combinatorial* layer that answers "which seats can all be taken at once".
> The first is solved once per grouping variant and cached; the second is a SAT instance.

---

## 1. Why — what the current DFS does on Gemma

Measured on `gemma_specdecode_mesh_graph_descriptor.textproto` (70 meshes) against SC36 Rev C, 36 ranks,
instrumented build. After 850+ search nodes and ~20 minutes:

| Observation | Value |
| --- | --- |
| Deepest partial assignment | **19 of 70** meshes |
| Backtracks at depth 19 | **895** |
| Backtracks at depth 1 | **0** |
| `S4x4` placements ever attempted | **0** |
| Distinct depth-1 commits | **1** (`4x2_Mesh_horizontal_flat_torus_x`) |

Three separate faults, in increasing order of how much they matter.

**(a) The variable ordering never reaches the rigid shapes.** `select_next_mesh` picks the unplaced mesh
with the most already-placed neighbours, which grows a connected blob outward from mesh 0. Gemma's mesh
graph chains `S4x2 → S4x1 ×5-6 → S4x2 → …`, so the search threads flexible 4-chip strips through the
fabric while the rigid 16-chip `S4x4` blocks sit at the back of the queue. By the time it reaches them
the free space is shredded into strip-shaped fragments. It never got that far.

**(b) Chronological backtracking permutes the wrong decision.** `S4x2` mesh 19 returns 0 candidates. The
DFS responds by re-seating mesh 18 — 895 times — when the ASICs mesh 19 needs were consumed by a much
shallower decision.

**(c) Every candidate pool is a fresh CSP solve, discarded on backtrack.** `next_step_pool` runs
`enumerate_distinct_placements_for_grouping` once per grouping variant per search node, ~0.2–0.4 s, and
throws the result away when the branch fails. At ~850 nodes × 8–24 variants that is 7,000–20,000 inner
solves already burned.

Fault (c) is what makes (a) and (b) unfixable in place: minimum-remaining-values ordering and forward
checking both need domain *sizes* for every unplaced mesh at every node, which under the current
architecture means 70× the cost of the thing that is already the bottleneck.

There is also a soundness hazard. Pools are capped:

```cpp
// physical_grouping_descriptor_matching.cpp:2406
constexpr std::size_t kMaxPlacementsPerVariant = 10;
```

whose own comment says a truncated pool can hide the only seating that works and is "the first thing to
suspect" when a descriptor comes back unplaced. Gemma is coming back unplaced.

---

## 2. Architecture

**Layer 1 — geometry, per grouping variant, cached.** Enumerate placements of one `GroupingInfo` against
the **full** physical graph. Discharges everything positional: tray/ASIC-location pinning, host
alignment, torus wraps, the mesh's own internal adjacency. Output is a set of ASIC footprints.

**Layer 2 — combinatorics, once per solve.** Choose one footprint per mesh such that no two overlap and
every mesh-graph edge is realised by enough ethernet links. Three constraint families, no geometry.

The split works because a candidate's validity against the fabric does not depend on the assignment.
Only disjointness and seams do, and both are bitset operations.

**Why layer 1 is cheap here.** PGD groupings are fully location-pinned:

```
# wh_bh_rev_c_galaxy_physical_grouping_descriptor.textproto:274
  name: "generic_4x1"
  custom_type: "4x1"
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_1 tray_id: TRAY_1 } },
    { id: 1 location { asic_location: ASIC_LOCATION_5 tray_id: TRAY_1 } },
    { id: 2 location { asic_location: ASIC_LOCATION_5 tray_id: TRAY_2 } },
    { id: 3 location { asic_location: ASIC_LOCATION_1 tray_id: TRAY_2 } }
  ]
```

All four nodes carry an exact `(asic_location, tray_id)`. The pattern is determined; the only freedom is
which host it lands on. On SC36 that is ~36 placements per variant, and each solve is close to pure unit
propagation. The expensive case is the trait-free MGD fallback, which §5 defers until it is needed.

**Enumeration is per mesh *definition*, not per instance.** All 157 `S4x1` meshes resolve through the
same instance-name lookup and share one `std::vector<GroupingInfo>`:

```cpp
// physical_grouping_descriptor_matching.cpp:2764
const InstanceName grouping_key = merged_instance_key(mgd_index, mgd_count, name_it->second);
const auto groupings_it = mesh_groupings.find(grouping_key);
...
global_mesh_groupings.emplace(global_mesh_id, groupings_it->second);
```

So Gemma needs 24 + 8 + 18 = **50 enumerations total**, shared across all 70 meshes. Two orders of
magnitude less work than the DFS has already discarded.

### 2.1 What changes at the call site

Exactly one statement. `solve_adjacency_guided_placement` lines 2693–2789 build the inputs and lines
2805–2815 flatten the output; only the solver call in between is replaced, and the replacement takes the
same arguments and returns the same `AssignedMeshes`.

---

## 3. Layer 1 — candidate enumeration

### 3.1 Dense ASIC index and footprint bitsets

`AsicID` is `ttsl::StrongType<uint64_t, AsicIDTag>`, so it cannot index a bitset directly.

```cpp
// Dense 0..N-1 numbering of the fabric's ASICs. Built once from the physical graph, which is already
// ordered (AdjacencyMap is a std::map), so the numbering is deterministic across ranks and runs.
class AsicIndex {
public:
    explicit AsicIndex(const AdjacencyGraph<AsicID>& physical_graph) {
        const auto& nodes = physical_graph.get_nodes();
        dense_to_asic_.reserve(nodes.size());
        for (const AsicID& asic : nodes) {
            asic_to_dense_.emplace(asic, dense_to_asic_.size());
            dense_to_asic_.push_back(asic);
        }
    }

    std::size_t size() const { return dense_to_asic_.size(); }
    std::size_t dense(const AsicID& asic) const { return asic_to_dense_.at(asic); }
    AsicID asic(std::size_t dense) const { return dense_to_asic_[dense]; }

private:
    std::unordered_map<AsicID, std::size_t> asic_to_dense_;
    std::vector<AsicID> dense_to_asic_;
};

// Fixed-width bitset over the dense ASIC index. Word count is a runtime value (fabric size varies by
// system), so this is a vector rather than std::bitset, but it is never resized after construction.
class AsicBitset {
public:
    AsicBitset() = default;
    explicit AsicBitset(std::size_t bit_count) : words_((bit_count + 63) / 64, 0) {}

    void set(std::size_t bit) { words_[bit >> 6] |= (uint64_t{1} << (bit & 63)); }
    bool test(std::size_t bit) const { return (words_[bit >> 6] >> (bit & 63)) & 1; }

    bool intersects(const AsicBitset& other) const {
        for (std::size_t i = 0; i < words_.size(); ++i) {
            if (words_[i] & other.words_[i]) {
                return true;
            }
        }
        return false;
    }

    void or_with(const AsicBitset& other) {
        for (std::size_t i = 0; i < words_.size(); ++i) {
            words_[i] |= other.words_[i];
        }
    }

private:
    std::vector<uint64_t> words_;
};
```

### 3.2 Candidate representation

```cpp
// One legal seating of one grouping variant. Footprint-only: the per-node mapping from the solve is
// deliberately dropped, matching what next_step_pool already keeps today (matching.cpp:2380-2385) --
// downstream reconstructs positions from the variant's pinning map, not from the embedding.
struct MasterCandidate {
    AsicBitset footprint;
    std::vector<AsicID> asics;    // same content, for building PsdPlacement without a reverse lookup
    const GroupingInfo* variant;  // borrowed: name, type, mesh_node_to_asic_position
    uint16_t hosts_spanned = 1;   // precomputed, used for value ordering
};
```

`variant` is a borrowed pointer, not a copy. `mesh_node_to_asic_position` is identical across every
placement of a variant — it is a property of the grouping, not of the footprint — so copying it per
candidate the way line 2382 does would multiply a `std::map` by the candidate count for no information.
Lifetime is safe because `global_mesh_groupings` holds the `GroupingInfo` vectors by value and outlives
the solve; it must not be mutated once pointers are taken.

### 3.3 Resumable per-variant sources

This is what makes enumeration bounded: pull a batch, and only come back for more if layer 2 proves it
needs them.

```cpp
// One live enumeration per (definition, variant). TopologyMappingEnumerationSession holds a single
// TopologySearchEngine and appends blocking clauses between next() calls, so resuming costs one solve
// rather than a re-encode.
struct VariantSource {
    const GroupingInfo* variant = nullptr;
    TopologyMappingEnumerationSession<LogicalChipId, AsicID> session;
    MappingConstraints<LogicalChipId, AsicID> constraints;  // trait/pinning, encoded once
    std::vector<std::map<LogicalChipId, AsicID>> excluded;  // mappings already returned
    std::vector<MasterCandidate> found;
    bool exhausted = false;
    bool session_started = false;
};

// Pull up to `batch` more placements from one variant. Returns how many were added.
// Sets `exhausted` when the session reports no further distinct mapping, which is the only signal that
// this variant's list is COMPLETE -- and therefore the only condition under which a later UNSAT is
// trustworthy.
std::size_t grow_variant(
    VariantSource& source,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& psd,
    const AsicIndex& asic_index,
    ConnectionValidationMode validation_mode,
    std::size_t batch) {
    if (source.exhausted) {
        return 0;
    }
    if (!source.session_started) {
        // Same constraint construction enumerate_distinct_placements_for_grouping does at matching.cpp:1097.
        if (!add_pgd_to_psd_constraints(*source.variant, physical_graph, psd, source.constraints, nullptr)) {
            source.exhausted = true;
            return 0;
        }
        source.session_started = true;
    }

    std::size_t added = 0;
    while (added < batch) {
        MappingResult<LogicalChipId, AsicID> mapping = source.session.next(
            source.variant->adjacency_graph,
            physical_graph,
            source.constraints,
            source.excluded,
            validation_mode,
            /*quiet_mode=*/true,
            TopologyMappingSolverEngine::Auto,
            /*unique_shapes=*/true);
        if (!mapping.success) {
            source.exhausted = true;
            break;
        }

        MasterCandidate candidate;
        candidate.footprint = AsicBitset(asic_index.size());
        candidate.variant = source.variant;
        std::set<std::string> hosts;
        for (const auto& [node, asic] : mapping.target_to_global) {
            candidate.footprint.set(asic_index.dense(asic));
            candidate.asics.push_back(asic);
            hosts.insert(psd.get_host_name_for_asic(asic));
        }
        candidate.hosts_spanned = static_cast<uint16_t>(hosts.size());

        source.excluded.push_back(mapping.target_to_global);
        source.found.push_back(std::move(candidate));
        ++added;
    }
    return added;
}
```

`unique_shapes = true` is what the existing call already passes and is exactly right here: per the
header it *"counts solutions by the set of global nodes used (order-independent); permutations on the
same global set share one slot"*, and enforces this with CNF clauses so SAT skips whole automorphism
classes per model. Footprint-level dedup is lossless because the consumer only keeps the footprint.

### 3.4 Per-definition lists

```cpp
// Candidate lists keyed by mesh DEFINITION (the grouping-variant vector), not by mesh instance.
// All 157 S4x1 meshes share one entry.
class MasterCandidateLists {
public:
    // `definition` is the address of the GroupingInfo vector held in global_mesh_groupings, which is
    // shared by every instance of that mesh definition and is therefore a valid identity key.
    using DefinitionKey = const std::vector<GroupingInfo>*;

    void add_definition(DefinitionKey key, bool include_mgd_fallback);
    std::size_t grow(DefinitionKey key, std::size_t batch_per_variant);
    const std::vector<const MasterCandidate*>& candidates(DefinitionKey key) const;

    // True only when every variant of every definition reported exhaustion. UNSAT is meaningless
    // otherwise, so this is reported in PlacementSolveStats and logged with any UNSAT verdict.
    bool complete() const;

private:
    struct Definition {
        std::vector<VariantSource> pgd_variants;
        std::vector<VariantSource> mgd_fallback;   // enumerated only when §5 escalates
        std::vector<const MasterCandidate*> flat;  // rebuilt after each grow()
    };
    std::map<DefinitionKey, Definition> definitions_;
};
```

PGD and MGD-fallback variants are held separately because §5 solves with PGD only first. That is not an
optimization — it makes "prefer an all-PGD placement" a *global* property. The DFS can only get it
locally, which is why the trace shows `S4x1 (MESH)` fallbacks committed at depths 17–19 while PGD
options remained elsewhere. The ordering intent is already stated at matching.cpp:1455: *"PGD first, MGD
last."*

---

## 4. Seams

Two helpers. Both preserve the convention that parallel links are duplicate neighbour entries, which
`free_chips_bordering_region` (matching.cpp:2119) already relies on.

```cpp
// Fabric links crossing from footprint `a` into footprint `b`. Parallel links are duplicate neighbour
// entries, so this counts links and not chips.
std::size_t seam_link_count(
    const MasterCandidate& a, const MasterCandidate& b, const AdjacencyGraph<AsicID>& physical_graph,
    const AsicIndex& asic_index) {
    std::size_t links = 0;
    for (const AsicID& chip : a.asics) {
        for (const AsicID& neighbor : physical_graph.get_neighbors(chip)) {
            if (b.footprint.test(asic_index.dense(neighbor))) {
                ++links;
            }
        }
    }
    return links;
}

// The descriptor's channel count for this seam. mesh_level_graph carries channel multiplicity as
// duplicate neighbour entries, same as placed_neighbors_of (matching.cpp:2133) relies on.
std::size_t required_seam_links(
    const GlobalMeshId& m1, const GlobalMeshId& m2, const AdjacencyGraph<GlobalMeshId>& mesh_level_graph) {
    std::size_t links = 0;
    for (const GlobalMeshId& neighbor : mesh_level_graph.get_neighbors(m1)) {
        if (neighbor == m2) {
            ++links;
        }
    }
    return links;
}
```

The weighted cardinality constraint the current code builds per variant (matching.cpp:2340–2357) never
reaches SAT. Because `required_seam_links` is fixed per mesh edge, `seam_link_count >= need` is a
boolean evaluated in C++ while emitting clauses.

No pair table is materialised. Each pair's count is needed exactly once, when its clause is written. If
the RELAXED second tier re-thresholds, cache only the non-zero pairs — most footprint pairs are nowhere
near each other.

---

## 5. Layer 2 — the SAT encoding

### 5.1 Variables

```cpp
// One variable per (mesh INSTANCE, candidate). Instances cannot share variables even when they share
// footprints, because they take different seats.
struct SeatVars {
    std::map<GlobalMeshId, std::vector<int>> by_mesh;      // parallel to candidates(definition_of(mesh))
    std::vector<std::vector<int>> by_dense_asic;           // asic -> vars whose footprint covers it
};
```

### 5.2 At-most-one

Pairwise is not viable at these sizes: ~500 candidates per mesh is 125k clauses per mesh, and ~180
candidates touching a typical ASIC is 16k clauses per ASIC over 1152 ASICs. Both land in the millions.
Sequential (Sinz) is linear.

```cpp
// Sequential at-most-one over `lits`. Adds n-1 auxiliary variables and ~3n binary clauses, versus
// n^2/2 for pairwise. Below kPairwiseAtMostOneCutoff the aux variables cost more than they save.
constexpr std::size_t kPairwiseAtMostOneCutoff = 8;

void add_at_most_one(TopologySatSolver& sat, const std::vector<int>& lits) {
    if (lits.size() <= 1) {
        return;
    }
    if (lits.size() <= kPairwiseAtMostOneCutoff) {
        for (std::size_t i = 0; i < lits.size(); ++i) {
            for (std::size_t j = i + 1; j < lits.size(); ++j) {
                sat.add(-lits[i]);
                sat.add(-lits[j]);
                sat.add(0);
            }
        }
        return;
    }
    std::vector<int> chain(lits.size() - 1);
    for (int& s : chain) {
        s = sat.declare_one_more_variable();
    }
    sat.add(-lits[0]); sat.add(chain[0]); sat.add(0);
    for (std::size_t i = 1; i + 1 < lits.size(); ++i) {
        sat.add(-lits[i]);     sat.add(chain[i]);     sat.add(0);
        sat.add(-chain[i - 1]); sat.add(chain[i]);    sat.add(0);
        sat.add(-lits[i]);     sat.add(-chain[i - 1]); sat.add(0);
    }
    sat.add(-lits.back()); sat.add(-chain.back()); sat.add(0);
}
```

### 5.3 Encoding

```cpp
// Encodes the whole master problem into a fresh solver. Returns false if any mesh has an empty
// candidate list, which is UNSAT by construction and worth reporting directly rather than as a verdict.
bool encode_master_problem(
    TopologySatSolver& sat,
    SeatVars& vars,
    const std::map<GlobalMeshId, MasterCandidateLists::DefinitionKey>& definition_of,
    const MasterCandidateLists& lists,
    const AdjacencyGraph<GlobalMeshId>& mesh_level_graph,
    const AdjacencyGraph<AsicID>& physical_graph,
    const AsicIndex& asic_index,
    bool relaxed_tier) {

    vars.by_dense_asic.assign(asic_index.size(), {});

    // (1) Variables, at-least-one, at-most-one -- one seat per mesh.
    for (const auto& [mesh_id, definition] : definition_of) {
        const auto& candidates = lists.candidates(definition);
        if (candidates.empty()) {
            return false;
        }
        std::vector<int>& lits = vars.by_mesh[mesh_id];
        lits.reserve(candidates.size());
        for (const MasterCandidate* cand : candidates) {
            const int var = sat.declare_one_more_variable();
            lits.push_back(var);
            for (const AsicID& asic : cand->asics) {
                vars.by_dense_asic[asic_index.dense(asic)].push_back(var);
            }
        }
        for (const int lit : lits) {
            sat.add(lit);
        }
        sat.add(0);
        add_at_most_one(sat, lits);
    }

    // (2) Disjointness -- no ASIC serves two meshes.
    for (const std::vector<int>& users : vars.by_dense_asic) {
        add_at_most_one(sat, users);
    }

    // (3) Seams -- support clauses, both directions.
    //     Forward alone is logically sufficient given (1), but the reverse roughly doubles propagation
    //     strength and binary/short clauses are cheap.
    for (const auto& [m1, definition1] : definition_of) {
        for (const auto& [m2, definition2] : definition_of) {
            if (!(m1 < m2)) {
                continue;
            }
            const std::size_t k = required_seam_links(m1, m2, mesh_level_graph);
            if (k == 0) {
                continue;  // not a mesh-graph edge
            }
            const std::size_t need = relaxed_tier ? 1 : k;
            const auto& c1 = lists.candidates(definition1);
            const auto& c2 = lists.candidates(definition2);

            for (std::size_t i = 0; i < c1.size(); ++i) {
                sat.add(-vars.by_mesh.at(m1)[i]);
                for (std::size_t j = 0; j < c2.size(); ++j) {
                    if (seam_link_count(*c1[i], *c2[j], physical_graph, asic_index) >= need) {
                        sat.add(vars.by_mesh.at(m2)[j]);
                    }
                }
                sat.add(0);  // empty support degenerates to the unit clause -var, deleting the seat
            }
            for (std::size_t j = 0; j < c2.size(); ++j) {
                sat.add(-vars.by_mesh.at(m2)[j]);
                for (std::size_t i = 0; i < c1.size(); ++i) {
                    if (seam_link_count(*c1[i], *c2[j], physical_graph, asic_index) >= need) {
                        sat.add(vars.by_mesh.at(m1)[i]);
                    }
                }
                sat.add(0);
            }
        }
    }
    return true;
}
```

The degenerate case in (3) is the valuable one. A candidate with no compatible partner becomes a unit
clause and is deleted at encode time — that is the "mesh 19 has no candidates" discovery, found once
instead of 895 times at depth 19.

Expected size for Gemma: order 70k variables and 600k clauses. The inner solver already runs SAT on this
fabric.

### 5.4 Decode

```cpp
AssignedMeshes decode_model(
    const TopologySatSolver& sat,
    const SeatVars& vars,
    const std::map<GlobalMeshId, MasterCandidateLists::DefinitionKey>& definition_of,
    const MasterCandidateLists& lists) {
    AssignedMeshes assignment;
    assignment.reserve(definition_of.size());
    for (const auto& [mesh_id, definition] : definition_of) {
        const auto& candidates = lists.candidates(definition);
        const std::vector<int>& lits = vars.by_mesh.at(mesh_id);
        for (std::size_t i = 0; i < lits.size(); ++i) {
            if (sat.val(lits[i]) <= 0) {
                continue;
            }
            PsdPlacement placement;
            // Same reconstruction as matching.cpp:2380 -- the pinning map belongs to the variant and
            // cannot be recovered from the footprint.
            placement.mesh_node_to_asic_position = candidates[i]->variant->mesh_node_to_asic_position;
            placement.asics.insert(candidates[i]->asics.begin(), candidates[i]->asics.end());
            assignment.push_back(PlacedMesh{mesh_id, std::move(placement)});
            break;  // exactly-one guarantees no second true literal
        }
    }
    return assignment;
}
```

### 5.5 Column generation loop

```cpp
constexpr std::size_t kInitialBatchPerVariant = 32;
constexpr std::size_t kGrowthBatchPerVariant = 64;
constexpr int kMasterConflictBudget = 0;  // 0 = unbounded; set for an oracle-mode cap

AssignedMeshes start_sat_placement(
    const std::map<GlobalMeshId, std::vector<GroupingInfo>>& global_mesh_groupings,
    const AdjacencyGraph<GlobalMeshId>& mesh_level_graph,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& psd,
    bool relaxed_inter_mesh_policy,
    PlacementSolveStats* stats) {

    const AsicIndex asic_index(physical_graph);
    std::map<GlobalMeshId, MasterCandidateLists::DefinitionKey> definition_of;
    MasterCandidateLists lists;
    for (const auto& [mesh_id, variants] : global_mesh_groupings) {
        const auto key = &variants;
        definition_of.emplace(mesh_id, key);
        lists.add_definition(key, /*include_mgd_fallback=*/false);
    }

    for (const auto& [key, _] : definition_of) {
        lists.grow(key, kInitialBatchPerVariant);
    }

    // Tier 0: PGD variants only, strict seams. Tier 1: MGD fallback added. Under RELAXED each tier is
    // retried with the seam threshold dropped to 1 before escalating, mirroring next_step_pool's
    // fallback at matching.cpp:2351.
    for (int tier = 0; tier < 2; ++tier) {
        if (tier == 1) {
            for (auto& [key, _] : definition_of) {
                lists.enable_mgd_fallback(key);
                lists.grow(key, kInitialBatchPerVariant);
            }
        }
        for (;;) {
            for (const bool relaxed_tier : {false, true}) {
                if (relaxed_tier && !relaxed_inter_mesh_policy) {
                    continue;
                }
                // Re-encoded per attempt rather than extended in place. CNF clauses cannot gain
                // literals after the fact, so growing a mesh's candidate list would need extension
                // literals on every at-least-one and every support clause. Encoding is sub-second and
                // attempts are few; the EXPENSIVE state (the per-variant enumeration sessions) is what
                // persists across attempts, and it does.
                TopologySatSolver sat;
                SeatVars vars;
                if (encode_master_problem(
                        sat, vars, definition_of, lists, mesh_level_graph, physical_graph, asic_index,
                        relaxed_tier)) {
                    const int verdict =
                        kMasterConflictBudget == 0 ? sat.solve() : sat.solve_limited(kMasterConflictBudget);
                    if (verdict == TopologySatSolver::kSat) {
                        return decode_model(sat, vars, definition_of, lists);
                    }
                }
            }
            std::size_t grown = 0;
            for (const auto& [key, _] : definition_of) {
                grown += lists.grow(key, kGrowthBatchPerVariant);
            }
            if (grown == 0) {
                break;  // every session exhausted at this tier
            }
        }
    }

    if (stats != nullptr) {
        stats->candidate_lists_complete = lists.complete();  // new field, see §7
    }
    return {};
}
```

**Truncation degrades gracefully, and only in one direction.** Any model SAT returns is a real solution:
the footprints are real embeddings and the clauses are real constraints. Truncation can only cause a
spurious UNSAT. So the rule is *trust SAT unconditionally, trust UNSAT only when `lists.complete()`*, and
that flag must be logged with any failure.

---

## 6. Phasing

Each phase is independently shippable and validated against Plan 3's DFS on identical inputs.

### Phase 0 — Measure (gate)

Call `enumerate_distinct_placements_for_grouping` once per (definition, variant) against the full
`physical_graph` with `max_solutions = 256`; log count and elapsed per variant.

Confirms the §2 claim that trait-pinned variants return tens of placements and only the MGD fallback
runs long. **If PGD variants return hundreds and take seconds, §3.3 batch sizes and §5.3 clause
construction both need rework before Phase 3.** Nobody has ever called this path with `max_solutions`
above 10, so the number does not exist yet.

### Phase 1 — `AsicIndex`, `MasterCandidate`, `VariantSource`, `MasterCandidateLists`

§3 in full. No behavior change; nothing calls it yet.

*Validation:* for every footprint produced, assert it is a valid embedding of its variant against the
physical graph. The whole design rests on the full-graph list being a superset of what a free-graph solve
would return, so prove it once rather than assuming it.

### Phase 2 — Rewire the DFS onto the master list

Replace `next_step_pool`'s body with bitset filtering; carry `occupied` in the branch state instead of
recomputing it via `collect_occupied_asics`. Keep the DFS otherwise byte-identical.

```cpp
std::vector<PlacementCandidate> next_step_pool(
    const GlobalMeshId& mesh_id,
    const AssignedMeshes& assignment,
    const MasterCandidateLists& lists,
    MasterCandidateLists::DefinitionKey definition,
    const AdjacencyGraph<GlobalMeshId>& mesh_level_graph,
    const AdjacencyGraph<AsicID>& physical_graph,
    const AsicIndex& asic_index,
    const AsicBitset& occupied,
    bool relaxed_inter_mesh_policy) {
    std::vector<PlacementCandidate> pool;
    for (const MasterCandidate* cand : lists.candidates(definition)) {
        if (cand->footprint.intersects(occupied)) {
            continue;
        }
        bool reaches_every_seam = true;
        for (const PlacedMesh& placed : assignment) {
            const std::size_t k = required_seam_links(mesh_id, placed.mesh_id, mesh_level_graph);
            if (k == 0) {
                continue;
            }
            const std::size_t need = relaxed_inter_mesh_policy ? 1 : k;
            if (seam_link_count(*cand, *placed.candidate, physical_graph, asic_index) < need) {
                reaches_every_seam = false;
                break;
            }
        }
        if (reaches_every_seam) {
            pool.push_back(cand->as_placement_candidate());
        }
    }
    return pool;
}
```

`PlacedMesh` gains a `const MasterCandidate* candidate` so the seam check has the neighbour's footprint.
The early-out at matching.cpp:2321 disappears: a blocked neighbour now simply fails every candidate.

*This is the risk-reduction phase.* If existing PGD tests change behavior, the master list is wrong, and
that is discovered before SAT is anywhere near the picture. Expect identical placements at roughly four
orders of magnitude less cost per node.

### Phase 3 — MRV and forward checking

Domains are now cheap **and complete**, so fail-first ordering is finally trustworthy — under the cap of
10 the domain sizes MRV would compare are fictional.

```cpp
// Minimum remaining values, with the previous "most placed neighbours" heuristic as tiebreak. Fail-first
// on domain size is what surfaces the rigid shapes: an S4x4 has far fewer legal seatings than a 4x1
// strip, so it is chosen early, before flexible strips shred the space it needs. Under the old ordering
// no S4x4 was attempted at all in 850 search nodes.
std::optional<GlobalMeshId> select_next_mesh(...);

// After each commit, recompute domains for all unplaced meshes and fail on any wipeout. Combined with
// MRV this rejects a doomed prefix before descending into it, instead of rediscovering the conflict
// once per sibling permutation 895 times.
bool forward_check(...);
```

**Gemma may place here, with no SAT at all.** Try this before Phase 4.

### Phase 4 — `start_sat_placement`

§5.1–5.4 behind a flag, same signature and return type as `start_adjacency_guided_dfs`.

*Validation:* on small descriptors both paths place all meshes; diff the assignments. On a descriptor
known to be infeasible, SAT must report UNSAT with `complete() == true`.

### Phase 5 — Column generation

§5.5. Tier escalation, growth loop, `complete()` tracking.

*Validation:* an instance solvable only by the 40th candidate of some variant must still solve with
`kInitialBatchPerVariant = 32`, exercising at least one growth round.

### Phase 6 — RELAXED tier and value ordering

Two-tier seam threshold in both paths. For the DFS path, rank the surviving pool by seam width then
`hosts_spanned`, retiring the value-ordering TODO at matching.cpp:2501. This replaces
`add_relaxed_preferred_chip_constraints`, which has no equivalent once candidates are precomputed —
ranking actual candidates is strictly better information than hinting at chips.

### Phase 7 — Cleanup

Delete the `PGD_DFS_DEBUG` instrumentation (grep that tag; the removal checklist is at matching.cpp:2035)
and decide whether the DFS path stays as a fallback.

---

## 7. Interface changes

`PlacementSolveStats` gains:

```cpp
    // Layer 1 / layer 2 split
    std::size_t master_candidates_enumerated = 0;
    std::size_t master_growth_rounds = 0;
    bool candidate_lists_complete = false;  ///< false => an UNSAT verdict is NOT trustworthy
    std::size_t master_sat_vars = 0;
    std::size_t master_sat_clauses = 0;
    std::chrono::microseconds master_encode_elapsed{};
    std::chrono::microseconds master_solve_elapsed{};
```

`PlacedMesh` gains `const MasterCandidate* candidate` (Phase 2), used by the seam check and for
diagnostics. `PsdPlacement` and the public signature of `solve_adjacency_guided_placement` are unchanged.

---

## 8. Risks

| Risk | Mitigation |
| --- | --- |
| PGD variants enumerate to hundreds/thousands, not tens | Phase 0 gates on this. Fall back to per-variant caps with `complete() = false` and accept untrustworthy UNSAT |
| MGD fallback enumeration is unbounded | Never enumerated at tier 0; only reached if a full PGD solution does not exist |
| Full-graph enumeration is not a superset of free-graph enumeration | Phase 1 validation asserts every footprint embeds. STRICT is a "≥ required channels" check per the `ConnectionValidationMode` doc, so an induced subgraph cannot admit an embedding the full graph rejects — but this is asserted, not assumed |
| Seam clause construction is O(&#124;P₁&#124;·&#124;P₂&#124;) per edge | Acceptable at observed domain sizes. If Phase 0 says otherwise, bucket candidates by host and compare only host-adjacent pairs |
| RELAXED preference semantics change | Steering becomes ranking (Phase 6). Behavioural difference, documented, not a regression — but it means RELAXED results will not be bit-identical to today's |
| Re-encoding per growth round wastes learned clauses | Accepted. Rounds should be 1–3; if profiling shows otherwise, switch to extension literals on the at-least-one and support clauses |
