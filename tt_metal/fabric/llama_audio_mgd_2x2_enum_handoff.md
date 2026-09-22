# Handoff: Llama + Audio MGDs & the 2×2‑quadrant under‑enumeration bug

Context: tt‑blaze #3089 — map a Llama + audio enc/decoder pipeline onto three galaxies
(leaving 16 chips free). While testing the candidate MGDs on a quad galaxy, we found a
mapper bug: the 2×2 "audio" mesh under‑enumerates its physical placements. This doc hands
off both the MGD artifacts and the (deep) bug investigation.

All work below was done against a **single quad** carved from the SC36 revC subtorus aisle‑D
mock (`bh-glx-120-d01u02/d01u08/d02u02/d02u08` = SC36 ranks 0,1,16,17 → 128 ASICs).

---

## 1. Artifacts already copied into this repo

MGDs — `tests/tt_metal/tt_fabric/custom_mesh_descriptors/`:
| file | shape | maps on the quad? |
|---|---|---|
| `llama_audio_1x2_pod_relaxed_...` | 2 meshes, 4×16 + 4×16 (RELAXED, count 16) | ✅ yes (realizes 2 intermesh links) |
| `llama_audio_quad_5mesh_...` | M0[4,8] → M1[4,4] → M2[2,2]×2 → M3[4,16], **5‑ring** | ❌ inter‑mesh fails (ring can't close) |
| `llama_audio_quad_5mesh_pinned_...` | 5‑mesh + **all pins** (M2→tray4, M1→tray1/2) many‑to‑many | ❌ fails at **disjoint placement** (pins over‑constrain) |
| `llama_audio_quad_6mesh_split2x4_...` | 4×4 split into two **2×4** meshes, 6‑ring | ❌ fails (same M0/M3 leaves) |
| `llama_audio_quad_6mesh_split4x2_...` | two **4×2** RING,LINE, 6‑ring | ❌ fails (identical to 2×4) |
| `llama_audio_quad_5mesh_pinned_singlenode_...` | branch‑compat pin form (see §2) | (parse‑only helper) |

Mock mapping (inside the `tt-cluster-descriptors` submodule):
`.../superclusters/blackhole/SC36_32x4_revC_subtorus_aisleD/SC4_32x4_revC_subtorus_aisleD_mapping.yaml`
(untracked — 4 ranks = one quad).

`reproducer_test.patch` (repo root) — the isolated SAT unit tests from §5.

**Key mapping result (unpinned quads):** every cyclic inter‑mesh MGD fails because the two
**big** meshes (M0 32‑chip, M3 64‑chip) are corner‑pinned physical **leaves** — the ring's
M0↔M3 closing edge never exists. Restructuring the *middle* meshes (4×4 → 2×4 → 4×2) never
helps; the failure is at the ring endpoints. Only the acyclic 2‑mesh line maps.

---

## 2. ⚠️ Branch/feature gotcha — run on `main`

The many‑to‑many ("all‑to‑all") pinning MGDs need the **repeated** proto fields:
```proto
message AsicPinning {
  repeated LogicalFabricNodeId  logical_fabric_node_id  = 1;   // <-- repeated
  repeated PhysicalAsicPosition physical_asic_position  = 2;   // <-- repeated
}
```
`origin/main` HAS both repeated. Branch `riddy21/fatal-intermesh-routing-validation` has
**neither** (both non‑repeated) → the pinned MGDs fail to parse
(`Non-repeated field "logical_fabric_node_id"/"physical_asic_position" is specified multiple times`).
A rebase of that branch onto main is **not clean** (992 commits behind, immediate conflicts in
`control_plane.cpp`). Recommendation: **do the pinning runs on latest `main`** (or a fresh
worktree off main). The `_singlenode_` MGD variant only helps if `physical_asic_position` is
repeated, so it too needs main.

---

## 3. How to run (single‑quad, CPU‑only mock)

```bash
export PATH="$PWD/python_env/bin:$PATH"
MOCK=tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC36_32x4_revC_subtorus_aisleD/SC4_32x4_revC_subtorus_aisleD_mapping.yaml
MGD=tests/tt_metal/tt_fabric/custom_mesh_descriptors/llama_audio_quad_5mesh_pinned_mesh_graph_descriptor.textproto
timeout 150 env TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_LOGGER_LEVEL=Debug tt-run \
  --mesh-graph-descriptor "$MGD" --mock-cluster-rank-binding "$MOCK" \
  --mpi-args "--allow-run-as-root --oversubscribe" \
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter="MultiHost.TestLlama8b1x2PodControlPlaneInit"
```
Notes: `TestLlama8b1x2PodControlPlaneInit` just forces control‑plane init (the SAT mapping);
it asserts the wrong mesh count for these MGDs, but any per‑shape/mapping logging you care
about is emitted *before* the assertion. A mapping failure often teardown‑hangs in MPI, so
keep the `timeout`. To pick a different quad, edit the SC4 mapping (e.g. d03/d04 ranks
32,33,24,25 for a D3/D4 quad).

Existing (built‑in) debug worth grepping at `LOGGER_LEVEL=Debug`:
`Physical groupings: Mesh graph descriptor 'M2': N topology match(es)` (commit count),
`Heterogeneous solver: ... set-packing chose N ... total weight W` (per‑shape placements),
`... intermesh degree histogram {...}`, `[intermesh-solve] ... success=`, and the disjoint
throw `Topology mapper failed to find disjoint placements ... Solution counts per mesh: [...]`.

---

## 4. THE BUG — 2×2 mesh under‑enumeration (the real finding)

On the quad there are provably **32 valid 2×2 quadrants** (4 galaxies × 8 per galaxy; a galaxy
tray is a 2×4 = two 2×2 halves `{1,2,5,6}` and `{3,4,7,8}`). The mapper realizes only **15**.
The loss is entirely in software, in two stages that share one root:

1. **Commit stage** (`get_valid_groupings_for_mgd` → `find_any_in_psd`): only **6 of 8** quadrant
   templates get committed — `find_any_in_psd` returns *empty* for **tray‑1‑LEFT** and
   **tray‑2‑LEFT** `{1,2,5,6}`, i.e. a single‑solve SAT returning UNSAT on a satisfiable instance.
   (32 → 24.)
2. **Enumeration stage** (`find_all_in_psd` → `enumerate_distinct_placements_for_grouping`
   → `solve_topology_mapping_n(unique_shapes=true)`): finds only **2–3 of the 4** valid galaxies
   per committed template. (24 → 15.)

### Ruled OUT (each with a test, not a guess)
- **Physical / harvesting** — a from‑scratch 4‑cycle check proved all 32 quadrants valid, with
  **bidirectional, ≥2‑channel** edges (`min_ch 2` on all 32). Not connectivity.
- **PGD half‑tray definitions** — all 8 quadrant templates are defined & correctly oriented.
- **Host‑alignment preference** — `TT_TOPO_NO_HOST_PREF=1` (env gate added to
  `configure_pgd_psd_host_alignment_constraints`) gave byte‑identical 15. Its doc even says it
  "doesn't restrict valid mappings."
- **Minimal‑host permanent‑cap enum bug** — that path only fires for `n_target=5` (the inter‑mesh
  solve), not the `n_target=4` 2×2.
- **Trait/domain construction** — a domain dump showed the solver's input is correct: **4
  candidates per target** (`t4=4 t5=4 t6=4 t7=4`) yet it returns 2–3.
- **Non‑zero‑based node ids** — real flattened groupings use ids `{4,5,6,7}`/`{8..11}`/`{12..15}`;
  an isolated test with `{4,5,6,7}` still enumerates 4.
- **The bare solver** — isolated unit tests (clean 2×2 blocks; full 4×8 galaxies; offset ids) all
  correctly enumerate **4** (see `reproducer_test.patch`).

### Where it therefore is
The solver is correct on **synthetic** inputs but under‑enumerates on the **real
`physical_graph`** that `find_all_in_psd` builds/passes. The one input not yet replicated in a
unit test is the **connected 128‑node quad graph** — specifically the adjacency *among the 16
candidate chips* (4 candidates × 4 targets) in the connected quad (my synthetic graphs used
disjoint blocks). **Next step:** dump that induced 16‑chip subgraph (pairs + channel counts)
from the real run and replay it in the unit test — that should reproduce, then shrink.

---

## 5. Isolated reproducer unit tests

`reproducer_test.patch` adds `TopologySolverQuadReproducer.*` to
`tests/tt_metal/tt_fabric/fabric_router/test_topology_solver.cpp`
(4 disjoint 2×2 blocks / full 4×8 galaxies / offset ids; pin each target node to node‑i of any
block; assert `solve_topology_mapping_n(unique_shapes)` returns 4 image‑sets). **All PASS** —
they document that the bare solver is fine and pin down that the trigger is the real graph.
The patch was generated against the clause‑sharing branch; if it doesn't `git apply`, re‑add the
`run_quad_2x2_reproducer(target_base)` helper + two `TEST`s before the final
`}  // namespace tt::tt_fabric` (content is in the patch).

Build & run just these (no MPI, no mock):
```bash
cmake --build build --target fabric_unit_tests
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologySolverQuadReproducer.*"
```

---

## 6. Debug instrumentation to re‑add (was wiped by branch checkouts)

All gated on `log_debug(tt::LogFabric, ...)` (Debug level). Re‑apply in
`tt_metal/fabric/topology_mapper_utils.cpp` and `physical_grouping_descriptor_matching.cpp`.

**(a) Prove 32 valid quadrants — `topology_mapper_utils.cpp`, anon namespace before
`build_physical_multi_mesh_adjacency_graph`; call once after `flat_graph` is built:**
```cpp
// enumerate all 8 quadrants/galaxy (4 trays x halves {1,2,5,6},{3,4,7,8}); test each for a
// 4-cycle (2x2) with bidirectional edges; report min channel count. Assumption-free.
void diagnose_all_2x2_quadrants(const AdjacencyGraph<tt::tt_metal::AsicID>& flat_graph,
                                const tt::tt_metal::PhysicalSystemDescriptor& psd) {
  std::map<std::string,std::map<std::pair<uint32_t,uint32_t>,tt::tt_metal::AsicID>> pos;
  std::map<std::string,std::set<uint32_t>> trays;
  for (const auto& a : flat_graph.get_nodes()) {
    pos[psd.get_host_name_for_asic(a)][{*psd.get_tray_id(a),*psd.get_asic_location(a)}]=a;
    trays[psd.get_host_name_for_asic(a)].insert(*psd.get_tray_id(a));
  }
  auto ch=[&](auto a,auto b){const auto&n=flat_graph.get_neighbors(a);
    return (size_t)std::count(n.begin(),n.end(),b);};
  const std::vector<std::vector<uint32_t>> halves={{1,2,5,6},{3,4,7,8}};
  const int ord[3][4]={{0,1,2,3},{0,1,3,2},{0,2,1,3}};
  for (auto& [host,pm]:pos){ size_t ok=0,ch2=0,tot=0;
    for (uint32_t t:trays[host]) for (const auto& h:halves){ ++tot;
      std::vector<tt::tt_metal::AsicID> c;
      for (uint32_t l:h){auto it=pm.find({t,l}); if(it!=pm.end()) c.push_back(it->second);}
      if (c.size()!=4) continue;
      bool cyc=false; size_t mn=1000;
      for (auto& o:ord){bool g=true; size_t m=1000;
        for (int k=0;k<4;k++){size_t x=ch(c[o[k]],c[o[(k+1)%4]]),y=ch(c[o[(k+1)%4]],c[o[k]]);
          if(x<1||y<1){g=false;break;} m=std::min(m,std::min(x,y));}
        if(g){cyc=true;mn=m;break;}}
      if(cyc){++ok; if(mn>=2)++ch2;} }
    log_debug(tt::LogFabric,"[2x2-all] host {} : {}/{} valid (>=1ch); {} with >=2ch",host,ok,tot,ch2);}
}
```
Result on the quad: **32/32 valid, all min_ch 2.**

**(b) Domain sizes vs solutions — `physical_grouping_descriptor_matching.cpp`,
`enumerate_distinct_placements_for_grouping`, after the `solve_topology_mapping_n` call:**
```cpp
if (grouping_info.adjacency_graph.get_nodes().size()==4){
  const auto& vm=constraints_opt->get_valid_mappings(); std::string d;
  for (uint32_t n:grouping_info.adjacency_graph.get_nodes())
    d+=fmt::format("t{}={} ",n, vm.count(n)?vm.at(n).size():0);
  log_debug(tt::LogFabric,"[2x2-domains] '{}' domains: {}-> {} solutions",
            grouping_info.name,d,result.size());   // (capture solve result in `result`)
}
```
Result: `t*=4 ... -> 2/3 solutions` (correct input, under‑count output).

**(c) Pin count in the existing match log** — fold `applicable_pin_groups` /
`applicable_pinned_nodes` (computed from `applicable_pinnings`) into the
`"...N topology match(es)..."` `log_info`.

**(d) Host‑pref env gate** — top of `configure_pgd_psd_host_alignment_constraints`:
`if (std::getenv("TT_TOPO_NO_HOST_PREF")) return;`

**(e) Also useful (were present):** per‑shape placement‑count + per‑placement footprint dump
(host/tray/asic) in the `find_all_in_psd` loop; mesh‑level adjacency map labeled by shape
(`dump_multi_mesh_adjacency` from `log_{logical,physical}_multi_mesh_adjacency_histograms`);
per‑committed‑variant trait/connectivity check `diagnose_2x2_embedding`.

Other relevant env knobs (already in the code): `TT_METAL_RELAX_PGD_SLOT_CONSTRAINTS=1`
(drops the (tray,asic) trait constraints — pool explodes, confirms traits gate enumeration).

---

## 7. Open questions / next steps
1. Capture the real induced subgraph among the 16 candidate chips for one 2×2 variant and replay
   it in `TopologySolverQuadReproducer` — should finally reproduce, then shrink to the minimal
   trigger (§4 "Where it is").
2. Separately trace **why `find_any_in_psd` returns empty for tray‑1/2‑LEFT** (commit‑stage drop) —
   likely the same root as the enumeration under‑count.
3. Decide the MGD path for #3089: a 5/6‑mesh **ring** does not pack into one quad (M0/M3 leaves).
   Either break the closing edge into a **line**, or span more than one quad, or pin M0/M3 (not the
   2×2s) onto mutually‑adjacent galaxies.
