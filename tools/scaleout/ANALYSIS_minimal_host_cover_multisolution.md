# Analysis: minimal-host-cover is not applied when enumerating multiple solutions

Status: **analysis only** (no code changes). Investigating why
`generate_rank_bindings --all-solutions` produces solutions that use **more hosts
than the minimum**, even though the single-solution path minimizes host count.

Commit analyzed: `739d2db404e` (branch `ridvan/gen-rank-bindings-all-solutions`,
on top of latest `main`) plus uncommitted working-tree edits to
`generate_rank_bindings.cpp` / `generate_rank_bindings_helpers.hpp` /
`sweep_rank_binding_solutions.py` (the `--distinct-host-sets`-as-host-dedup +
`host_set` CSV changes). None of those working-tree edits touch the
single-solution path or the solver.

---

## TL;DR

- The minimal-host-cover objective **is added identically for both solve paths**
  (it lives on the shared `MappingConstraints`).
- The **single-solution** solve *optimizes* it → true minimum host count.
- The **multi-solution enumeration** (`solve_topology_mapping_n` → SAT
  `topology_sat_search_n`) encodes **hard constraints only** and never consults
  the host-budget / preferred objective → it returns feasible solutions in solver
  order, which typically use **more** hosts than the minimum.
- So `--all-solutions` on SC36 gives **25–26 hosts**; the normal single
  (`tt-run`) solve on SC20 gives **exactly 20**.

This is a real gap in the multi-solution path, **not** a regression from the
`--all-solutions` feature's mapper plumbing.

---

## What was observed

| Run | Path | Solver routine | Hosts | Wall time |
|-----|------|----------------|-------|-----------|
| supercluster_20 MGD on **SC20** mock (`tt-run`, single) | single-solution | `topology_sat_search` | **20** (true min) | ~156 s |
| supercluster_20 MGD on **SC36** mock (`--all-solutions`, 3 sols) | enumeration | `topology_sat_search_n` | **25, 26, 26** | ~7 s total |
| supercluster_20 MGD on **SC36** mock (`tt-run`, single) | single-solution | `topology_sat_search` | *(expected ~20; runs very slowly at this size — see below)* | ≫ 14 min |

MGD `blitz_decode_mesh_graph_descriptor_supercluster_20.textproto` = **80 meshes**
of 4×2 (8 chips) each, `host_topology {1,1}` = 640 chips. The physical grouping
exposes ~4 usable 8-chip sub-meshes per 128-chip galaxy, so the minimum host
cover is **ceil(80 / 4) = 20 galaxies**. SC20 has exactly 20 (forced to 20); SC36
has 36 available, so the minimum is still 20 — but enumeration spreads to 25–26.

---

## Root cause (code trace)

All refs are on the analyzed commit.

### 1. The objective is added to the constraints (shared by both paths)

`tt_metal/fabric/topology_mapper_utils.cpp`:
- `build_inter_mesh_constraints` (~L1730) **unconditionally** calls
  `add_inter_mesh_minimal_host_cover_from_hostname_map` (~L1761).
- That function (~L1623) adds, using `config.hostname_to_asics`:
  - `inter_mesh_constraints.set_minimize_same_rank_groups_used(true)` (~L1684) — the objective flag;
  - one `add_preferred_constraint(target, preferred_globals)` per logical mesh (~L1706), a **soft** bias toward the minimal-cover host set (`PhysicalGroupingDescriptor::find_minimum_coverage_group`, ~L1666);
  - registers per-host global groups (`set_same_rank_groups_constraint`, inert as a hard constraint unless a single host covers everything).

`tt-metalium/experimental/fabric/topology_solver.hpp`:
- `add_preferred_constraint` — documented as soft ("doesn't restrict valid mappings").
- `set_minimize_same_rank_groups_used` doc is explicit that it is **best-effort / opt-in**: SAT walks an at-most-K host budget up from `ceil(num_targets / max_group_size)`; DFS approximates via a host-affinity value-ordering bias; "off by default; intended for inter-mesh mapping."
- Both the flag and groups are carried into the indexed form (`ConstraintIndexData::minimize_same_rank_groups_used`) and the SAT view, so **both** search routines *receive* the objective. Whether they *use* it is the difference.

### 2. Single solve OPTIMIZES it

`tt_metal/fabric/topology_solver_sat.cpp` — `topology_sat_search` (single, ~L1386–1620):
- **Host-budget loop** (~L1484–1525): if `minimize_same_rank_groups_used`, walk `k` from the capacity lower bound upward, encoding an at-most-K host budget (`topology_sat_encode_host_group_budget`) and returning the first satisfiable `k` — the **true minimum host count**. Falls back to an unconstrained solve if no budget is satisfiable.
- **Preferred at-least-k pass** (~L1527–1584): appends preferred-hit indicator literals and forces a lower-bounded number of preferred hits.

DFS single (`SearchHeuristic::compute_candidate_cost`, `topology_solver.tpp` ~L2372–2412): greedy ordering with `HOST_AFFINITY_WEIGHT = 10000` ≫ `SOFT_WEIGHT = 1000`, so the first complete mapping is host-affinity-minimal (approx).

### 3. Enumeration DROPS it — this is the bug

`tt_metal/fabric/topology_solver_sat.cpp` — `topology_sat_search_n` (enumeration, ~L1624–1713):
- Encodes **hard constraints only** (`topology_sat_encode_hard_constraints`), then loops `solve()` + blocking-clause until `max_solutions`.
- **No** reference to `minimize_same_rank_groups_used`, **no** `topology_sat_encode_host_group_budget`, **no** `topology_sat_append_preferred_hit_indicators`. Those symbols appear only inside the single `topology_sat_search`.
- Result: CaDiCaL emits distinct feasible models in solver-internal order; high-host-count solutions are just as eligible (and often emitted first) as minimal ones.

DFS enumeration (`DFSSearchEngine::search_n`, `topology_solver.tpp` ~L3066–3262): keeps the soft candidate *ordering* bias (so discovery order is tilted) but backtracks unconditionally with **no budget/objective cutoff and no ranking** — every feasible mapping is recorded.

### 4. Stats exist but are unused

`MappingResult::ConstraintStats { preferred_satisfied, preferred_total }`
(`topology_solver.hpp` ~L553) is populated by the validator but **never used to
rank or filter** in either path. `solve_topology_mapping_n` returns results in raw
solver order with no sort by host count or preferred score. There is **no**
existing option to enumerate under the minimal-host-cover objective; `unique_shapes`
only dedups by global-node set.

---

## Why the SC36 *single* solve is slow (and SC20 is fine)

The SAT host-budget loop walks `k` from `k_min = ceil(80/4) = 20` upward, encoding
an at-most-K-of-N-host-groups budget at each step. With **N = 20** host groups
(SC20) the only feasible `k` is 20 and the search space is tiny → ~156 s. With
**N = 36** (SC36), `k = 20` must choose 20 of 36 galaxies — a far larger
combinatorial budget encoding, each solved under a conflict budget
(`kHostMinimizeConflictBudget`) and walked upward on failure → minutes to
(observed) ≫ 14 min. So on SC36 even the "correct" minimizing solve is expensive;
the enumeration path is fast precisely because it skips this.

---

## Implications for `--all-solutions` / the sweep

- Every enumerated solution is a *valid* placement, but host count is not minimized
  and varies (25–26 on SC36). `--distinct-host-sets` (host-set dedup, working-tree
  change) removes duplicates but does not make any solution use fewer hosts.
- The `20`-host expectation is met only by the single optimizing solve (e.g. the
  CPU-only fabric tests, which run supercluster_20 on the **SC20** mock via `tt-run`).

---

## Options to fix (not implemented; for discussion)

1. **Post-rank / filter enumerated solutions by host count** — sort
   `map_multi_mesh_to_physical_n` results ascending by distinct host count (and/or
   `preferred_satisfied`) before returning; optionally keep only those at the
   minimum. Cheap, low-risk. **Caveat:** only surfaces the best *among what
   enumeration produced* (e.g. 25), not the true 20, because enumeration never
   searches toward the minimum.
2. **Teach `topology_sat_search_n` to honor the objective** — encode the host-group
   budget (and/or preferred-hit floor) during enumeration, e.g. enumerate at the
   minimal `k` first, then relax. Most correct; a solver change; inherits the
   host-budget cost.
3. **Seed with one optimizing solve** — run `solve_topology_mapping` once to learn
   the minimum host count `K`, then enumerate with an at-most-`K` (or `K+δ`) host
   budget. Gives true-minimum-host solutions but pays the (expensive at SC36) single
   optimizing solve up front.

**Recommendation:** (1) as an immediate, low-risk improvement to what the sweep
surfaces, with (2)/(3) as the real follow-up if minimal-host solutions are required
in the multi-solution path. Given the SC36 optimizing-solve cost, (2)/(3) also need
a performance pass on the host-budget encoding for larger physical systems.

---

## Reproduction

```bash
export TT_METAL_HOME=/data/rsong/tt-metal2
export LD_LIBRARY_PATH=$TT_METAL_HOME/build_Debug/tt_metal:$TT_METAL_HOME/build_Debug/lib
export TT_METAL_SLOW_DISPATCH_MODE=1 ARCH_NAME=blackhole
MGD=models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_mesh_graph_descriptor_supercluster_20.textproto
SC20=tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC20_32x4_revC_subtorus_aisleC/SC20_32x4_revC_subtorus_aisleC_mapping.yaml

# Single (optimizing) solve on SC20 -> 20 hosts (~156 s):
tt-run --force-rediscovery --mesh-graph-descriptor "$MGD" \
  --mock-cluster-rank-binding "$SC20" \
  --mpi-args "--allow-run-as-root --oversubscribe" --skip-executable-check -- true
# then count distinct descriptors in generated/ttrun/<cache_id>/phase2_mock_mapping.yaml

# Multi-solution enumeration on SC36 -> 25-26 hosts (~seconds), NOT minimized:
mpirun ... generate_rank_bindings -m "$MGD" --all-solutions --max-solutions 3 -o <out>
# inspect <out>/solutions_index.yaml (num_hosts)
```
