# Speeding up `minimize.warm_solve` — brainstorm + experiments

Target: the **258 ms** `minimize.warm_solve` = 88% of the 72-stage inter-mesh solve. It is a *cold*
`solver.solve()` at `topology_solver_sat.cpp:861` — CaDiCaL finds the first feasible 72→72 ring embedding
from scratch under 10 368 adjacency clauses + exactly-one + injectivity. No warm-start today.

Checkpoint before experiments: `cc0e88940e1`.

## Baseline facts
- Problem for the 72-stage: bijection (n_target = n_global = 72), ring adjacency, occupancy objective is
  **degenerate** (k_min = best_k = 72, same-rank encode = 0 clauses).
- The codebase already warm-starts a *different* solve (hard-cap path, `TT_TOPO_SAT_HARDCAP_WARMSTART`):
  solve a cheap encoding → `solver.phase(assign_lit ...)` → solve the hard one. Phase hints are branching
  preferences only, never constraints — always sound.

## Candidate approaches

### A. Greedy adjacency warm-start → phase hints (your idea, SAT-preserving)
Build a greedy feasible assignment guided by adjacency (walk the target graph from a seed, place each next
target on an unused global adjacent to its already-placed neighbours, within its AC-3 domain). Feed it as
`solver.phase()` hints, then run the warm `solve()`. CaDiCaL branches straight to the greedy embedding and
only repairs violations. **Expected: near-0 conflicts when the greedy is valid.** Lowest risk (correctness
unchanged), directly tests your hypothesis. **Run first.**

### B. Greedy-only bypass (skip SAT for the warm feasible model)
If the greedy produces a *fully valid* assignment (all adjacency + injectivity satisfied), decode it directly
and skip `solve()` entirely; the descent afterward still uses SAT. Fastest when greedy succeeds; needs a
validity check + fallback. Superset of A.

### C. Co-location contraction (your "whole group must land together")
When a same-rank host-group has capacity > 1 (several physical meshes on one host that must be co-assigned),
contract the group into one super-target/super-global before solving, cutting variables. Propagates "if one
rank of the group lands here, the rest do too" structurally instead of via SAT clauses. NOTE: for the 72-stage
groups are size 1 (no contraction), so this helps the *co-located* topologies (2 meshes/host etc.), not the
full-fill bijection. Complementary to A/B.

### D. CaDiCaL configuration for a first feasible model
The warm solve wants *any* model fast, not an optimized one. Try: `--target=2`/phase-saving tweaks, disable
expensive inprocessing for this first solve, or a rephasing schedule. Cheap to try (no new algorithm), but
smaller, less predictable payoff than A.

### E. Structural ring solver (no SAT for ring→ring)
`ring=true` bijection is a rotation/reflection embedding — solvable analytically (or with pure AC-3 + a single
walk) with no CDCL. Biggest potential win on the dominant case, but narrowest (only pure ring bijections).

## Experiment plan (measured on the 72-stage, warm_solve ms)
1. **A** — greedy adjacency hint, gated `TT_TOPO_SAT_WARM_GREEDY`. A/B off vs on.
2. **B** — extend A to bypass `solve()` when greedy is fully valid.
3. **D** — quick CaDiCaL option sweep as a cross-check.
Record warm_solve ms + conflicts for each; pick the best.
