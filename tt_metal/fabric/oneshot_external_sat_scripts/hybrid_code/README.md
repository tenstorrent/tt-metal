# Gimsatul hybrid + from-scratch enumeration — experiment code

The actual solve code behind the gimsatul-hybrid and incremental-vs-from-scratch experiments in this PR.
Provided here as reference source + a patch, because it **depends on the min-host solver**
(`topology_sat_solve_minimize_groups`) which lives on branch `ridvan/gen-rank-bindings-all-solutions`
(PR #49860), NOT on `main` — so it does not compile standalone on this branch.

## Contents
- `gimsatul_hybrid.patch` — unified diff against `ridvan/gen-rank-bindings-all-solutions` (apply with
  `git apply` from repo root on that branch).
- `topology_solver_sat.cpp` / `topology_solver_sat_solver.cpp` / `.hpp` — full modified sources (copies), for review.

## What the code adds (all env-gated; production unchanged unless a TT_TOPO_SAT_* var is set)
- `TopologySatSolver::write_dimacs()` — faithful DIMACS export via a clause "tee" (CaDiCaL's own write_dimacs drops
  post-solve incremental clauses).
- `TopologySatSolver::gimsatul_solve(threads, assumption_units)` — run gimsatul as a subprocess on the exported CNF
  (assumptions baked as hard units), parse the model back; `val()` answers from the gimsatul model.
- `TopologySatSolver::phase_hint_from_last_gimsatul_model()` — warm-start CaDiCaL from a gimsatul model.
- `delegated_solve()` in `solve_minimize_groups` — route the heavy solve to gimsatul (env `TT_TOPO_SAT_GIMSATUL`,
  `_THREADS`, `_BIN`) with CaDiCaL fallback; `TT_TOPO_SAT_GIM_FIRST` = gimsatul prime then native CaDiCaL descent.
- `TT_TOPO_SAT_ENUM_FROMSCRATCH` in `search_n` — from-scratch enumeration (fresh solver + replay blocks per solution)
  vs the default incremental; plus `TT_TOPO_SAT_SEED` applied to both enumeration modes and always-on path markers.

## Reproduce
Build gimsatul (`scripts .../build_gimsatul.sh`), then set e.g.
`TT_TOPO_SAT_GIMSATUL=1 TT_TOPO_SAT_GIMSATUL_BIN=<gimsatul> TT_TOPO_SAT_GIMSATUL_THREADS=32` on the producer.
