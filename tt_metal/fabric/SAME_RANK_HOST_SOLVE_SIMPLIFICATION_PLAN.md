# Same-Rank-Host / Inter-Mesh Occupancy Solve — Simplification Plan

Goal: reduce the time and code complexity of the **inter-mesh "same-rank host" occupancy solve**
(the minimal-host packing step), and consolidate the overlapping solve paths. Grounded in a profiled
run of the biggest solve.

## Measured baseline (4x4 72-stage, full SC36 fill — the biggest solve)

Captured via `generate_rank_bindings` on the 72-stage MGD + 36-host mock, with the new
`[intermesh-solve]` timer and `[topo-sat-profile]` phase logs (build with `-DTT_METAL_ENABLE_LOGGING=ON`,
run with `TT_LOGGER_LEVEL=debug`; the inter-mesh solve's first attempt is now non-quiet).

```
[intermesh-solve] attempt 1 : 291.7 ms  (success, 72 logical -> 72 physical meshes)
  occupancy path: n_target=72 n_global=72 ring=true max_k=72 minimize=true
  encode hard constraints total ............ 23.3 ms
      encode.6_adjacency_support ........... 11.1 ms  (+10368 clauses)
      encode.4_exactly_one .................  5.0 ms  (+15336 clauses)
      encode.5_injectivity .................  4.3 ms  (+15264 clauses)
      encode.3_create_vars .................  1.2 ms  (+5184 vars)
      encode.7_same_rank_groups ............  0.0 ms  (+0 clauses)   <-- adds NOTHING here
      encode.8_cardinality .................  0.0 ms  (+0 clauses)
  minimize.group_reachable_mesh_sizes: 72 groups x 1 reachable
  minimize.warm_solve ...................... 258.4 ms (SAT, occupied=72)   <-- 88% of the solve
  minimize: solve_minimize_groups total .... 264.7 ms (best_k=72, hard_cap_k=72, hard_cap_met=false)
```

**Headline:** 88% of the inter-mesh solve is a single `warm_solve` inside `solve_minimize_groups`.
For a full fill the occupancy objective is **degenerate** (`k_min == n_target`, `best_k == n_target`,
same-rank encoding adds 0 clauses), so the entire minimal-host descent/hard-cap/permanent-cap wrapper
runs a full 258 ms SAT solve to achieve *nothing*.

## Where the code lives

| Concern | File / symbol |
|---|---|
| Build host-group partitions + set occupancy objective | `topology_mapper_utils.cpp` — `add_inter_mesh_minimal_host_cover_from_hostname_map` (~1600–1684) |
| Inter-mesh solve driver (retry loop) | `topology_mapper_utils.cpp` — `map_multi_mesh_to_physical` (~2400–2510), solve at 2494 |
| Same-rank-group constraint plumbing | `topology_solver.tpp` — `set_same_rank_groups_constraint` / `validate_same_rank_groups_feasible` (~696–786) |
| Occupancy objective encode/solve | `topology_solver_sat.cpp` — `encode_at_most_k_groups` (~751), `solve_minimize_groups` (~782–970: warm_solve → descent → hardlock → permanent_cap) |
| Same-rank encode in the hard CNF | `topology_solver_sat.cpp` — `encode_hard_constraints` step `7_same_rank_groups` |
| Three overlapping solve entry points | `topology_solver_sat.cpp` — `solve_topology_mapping` (single), `topology_sat_session_create_and_encode` (incremental/session), `topology_sat_search_n` (enumerate) |

## Simplification opportunities (ranked)

### 1. Short-circuit the degenerate occupancy objective  *(biggest win, lowest risk)*
When `k_min == n_target` (equivalently `max_group_capacity == 1`, or the same-rank encode adds 0 clauses),
the minimal-host objective is a no-op. Detect this in `add_inter_mesh_minimal_host_cover_from_hostname_map`
(or at the top of `solve_minimize_groups`) and **skip the occupancy objective entirely** — do a single plain
solve. Saves the descent/hardlock/permanent-cap scaffolding on every full/near-full fill and removes a whole
class of "why is minimize running when it can't minimize" confusion. Expected: 264 ms → ~1 plain warm solve
with no occupancy wrapper (same result, less code executed).

### 2. Collapse `solve_minimize_groups` from 4 stages to 2
Today it is: warm_solve → soft descent → optional hard-cap "lock" → permanent-cap unit clause. For the
common cases only warm_solve + permanent_cap fire. Proposal: make it **(a) one warm solve to get a feasible
k, (b) one hard-cap solve at k** and drop the incremental descent unless a knob requests it. The descent
exists to crack sparse packings; gate it behind the sparse case (see `topology-sparse-minhost-hang`) instead
of running it always.

### 3. Specialized bijection / assignment fast path
The 258 ms warm_solve for 72→72 is a bijection with adjacency support (a ring embedding). When
`n_target == n_global` and required-constraints already pin most nodes, this is an assignment/matching
problem, not general SAT. A dedicated bipartite-matching / AC-3 + greedy path (fall back to SAT on failure)
would likely cut the dominant cost by an order of magnitude. Bigger change — do after #1/#2.

### 4. Consolidate the three solve entry points
`solve_topology_mapping`, `topology_sat_session_create_and_encode`, and `topology_sat_search_n` each
re-implement: encode → (occupancy prime) → solve/enumerate. The occupancy priming block is duplicated in
`search_n` (~2400) and `create_and_encode` (~2540). Extract one `prime_minimal_host(solver, enc, constraints)`
helper and have all three call it. Reduces drift and makes #1/#2 land in one place.

### 5. Encode-side: adjacency_support is the encode hot spot (11 ms, 10k clauses)
Not "same-rank" per se, but it dominates encode. Worth checking whether adjacency support clauses can be
tightened (only over the reduced AC-3 domains) once #1 removes the occupancy wrapper noise.

## Suggested order
1. #1 short-circuit degenerate occupancy (fast, safe, immediately removes wasted work on full fills).
2. #4 consolidate the priming helper (enables #2/#3 in one spot).
3. #2 collapse descent stages behind the sparse-case knob.
4. #3 assignment fast path (largest payoff on the dominant warm_solve, largest effort).

## How to reproduce the profile
```
cmake -S . -B build -DTT_METAL_ENABLE_LOGGING=ON && cmake --build build --target generate_rank_bindings
# 36-rank producer on the 72-stage MGD + SC36 mock, TT_LOGGER_LEVEL=debug, first attempt non-quiet.
# Grep the log for "[intermesh-solve]" (total) and "[topo-sat-profile]" (phase breakdown).
```

## Instrumentation added for this analysis (uncommitted, profiling aids)
- `topology_mapper_utils.cpp`: always-on `[intermesh-solve]` timer around the 2494 solve; first inter-mesh
  attempt made non-quiet so the phase breakdown is emitted.
- `topology_solver_sat.cpp`: `search_n.*` and `preferred.*` phase timers (added earlier).

These are debug aids; fold in or drop as the refactor lands.
