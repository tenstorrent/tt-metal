# Plan: preferred-aware multi-solution enumeration (keep "preferred" soft)

Status: **design / plan** (no code written). Follow-up to
[`ANALYSIS_minimal_host_cover_multisolution.md`](ANALYSIS_minimal_host_cover_multisolution.md).

## Problem recap

The current attempt makes the preferred objective a **hard** constraint during
enumeration: `topology_sat_search_n` calls `topology_sat_append_soft_objective`,
which adds a **global at-least-k preferred-hit floor** (`k_lb` from a lower-bound
routine) once, up front. This is wrong for enumeration on three counts:

1. **It's hard, not soft.** Forcing "≥ k preferred hits" on *every* enumerated
   model eliminates every non-maximal model — the opposite of what enumeration is
   for. Combined with the accumulating **blocking clauses** (which *are* hard —
   they exclude already-found models), later models can't meet the fixed floor →
   the solve goes UNSAT and enumeration returns 0.
2. **A single global floor ignores the blocking state.** The right "most
   preferred" model at step *N* depends on which models are already blocked; a
   fixed `k_lb` computed once can't express that.
3. **Wrong scope.** `topology_sat_search_n` is also used by the **physical-grouping
   placement enumeration** (`build_physical_multi_mesh_adjacency_graph`, one call
   per 8-chip mesh shape, `n_target=8`). There the floor becomes `k=8` (all
   nodes), collapsing each shape's placement set from "all valid embeddings" to
   exactly one → the packing starves → inter-mesh mapping fails. This regressed
   **both** the single- and multi-solution paths (verified: `supercluster_20` on
   SC20/SC36 → "Inter-mesh mapping failed" / 0 solutions).

**The core insight (correct framing):** during enumeration, the blocking clauses
are the hard part; **"preferred" must stay a soft objective that is *re-optimized
at each step*, given the blocking clauses accumulated so far.** Each emitted
solution should be the *most-preferred model that hasn't been emitted yet* — not a
model forced to clear a fixed global floor.

## Proposed design: per-step incremental optimization (assumption-based)

Use the fact that CaDiCaL is **incremental** and supports **`assume(lit)`**
(assumptions apply to the next `solve()` only and are auto-retracted). This lets
each enumeration step optimize the soft objective *without* permanently
constraining the instance.

### One-time setup (per solver)
- Encode hard constraints once (`topology_sat_encode_hard_constraints`).
- Encode the **objective indicators once** as Tseitin literals with a
  **cardinality counter** whose output literals are *togglable via a single
  assumption*:
  - preferred: `topology_sat_append_preferred_hit_indicators` → per-target
    `pref_hit` literals, fed into a **totalizer/sequential counter** so that
    assuming output literal `geq[k]` means "≥ k preferred hits". (The existing
    `topology_sat_add_at_least_k_literals` adds *hard* clauses; for this we want
    the counter's outputs assumable, so encode the counter once and `assume` its
    outputs — a small addition, not a hard at-least-k.)
  - host-cover (optional, stronger): the existing
    `topology_sat_encode_host_group_budget` at-most-K, likewise exposed via an
    assumable budget literal so "≤ K hosts" can be assumed/retracted.

### Per enumeration step (replaces the bare `solver.solve()`)
```
best_model = optimize_soft(solver, enc, objective):
    # objective = maximize preferred hits (and/or minimize host count)
    # subject to the CURRENT hard state (hard constraints + all blocking clauses so far)
    lo, hi = feasible bound range
    # 1. plain solve with NO objective assumption -> proves a model still exists
    if solver.solve() != SAT: return NONE          # genuinely exhausted -> stop enumeration
    best = current model; best_val = objective_value(best)
    # 2. tighten via assumptions (retracted each solve), e.g. binary/linear search on k:
    while can_improve(best_val):
        assume(geq[best_val + 1])                   # "can we do strictly better?"
        s = solve_limited(budget)                   # cap expensive proofs
        if s == SAT:  best = model; best_val = objective_value(best)
        else:         break                         # UNSAT/unknown under assumption -> best_val is optimal (or give up)
    return best
commit best_model
add_blocking_clause(best_model)      # hard: exclude it from future steps
```

Properties:
- **Soft, not hard:** step 1 (unconstrained solve) guarantees we return the plain
  feasible model if optimization can't improve — so this **can never turn a
  feasible instance UNSAT**. The objective assumptions are retracted; nothing
  permanent forces a preferred floor.
- **Respects blocking state:** the optimization runs against the live solver, so
  "most preferred remaining" is computed *given* the already-emitted (blocked)
  models — exactly the user's requirement.
- **Best-first ordering:** solutions come out most-preferred (fewest hosts) first;
  `--max-solutions N` then keeps the N best.
- **Cheap:** one warm-started incremental solver; assumptions + `solve_limited`
  keep each step bounded, with graceful fallback on an intractable bound.

### Factor a shared primitive
Add `topology_sat_solve_optimizing(solver, enc, objective, budget) -> optional<model>`
and call it from **both**:
- **batch:** `topology_sat_search_n` — replace the inner `solver.solve()` in the
  `while (found < max_solutions)` loop (`topology_solver_sat.cpp` ~L1701).
- **incremental:** `TopologyMappingEnumerationSession::next()` — replace its inner
  SAT `solve()` (`topology_solver.tpp` ~L1280–1425). This is the session the
  multi-mesh packing and any future streaming caller use, so it must get the same
  treatment, not just the batch API.

Both already keep a persistent solver and add blocking clauses; only the "how we
pick the next model" changes.

## Scoping — do NOT optimize the placement enumeration (fixes the regression)

The preferred-optimization must be **opt-in**, enabled only for the **inter-mesh
host-cover** enumeration, never for the physical-grouping placement enumeration.

- Thread a flag (e.g. `bool optimize_preferred` on `solve_topology_mapping_n` /
  `TopologyMappingEnumerationSession::next`, or reuse
  `MappingConstraints::minimize_same_rank_groups_used` as the gate — it is already
  set only by `add_inter_mesh_minimal_host_cover_from_hostname_map`).
- `build_physical_multi_mesh_adjacency_graph`'s per-shape enumeration leaves it
  **off** → plain enumeration of all embeddings (restores the pre-change behavior
  that both paths depend on).

## DFS path

DFS enumeration (`DFSSearchEngine::search_n`, `topology_solver.tpp` ~L3066) already
biases *discovery order* via the value-ordering weights
(`HOST_AFFINITY_WEIGHT` ≫ `SOFT_WEIGHT`) and never adds a hard floor, so it already
approximates best-first without the SAT bug. To make DFS *exactly* best-first it
could sort its emitted models by objective before returning, but that's optional;
the required fix is on the SAT path.

## Simpler interim options (if full per-step MaxSAT is too much for v1)

1. **Enumerate plain, then rank.** Drop the floor entirely from `search_n`
   (restores correctness), enumerate feasible models, and **sort the returned
   vector by host count / preferred hits ascending** so callers/`--max-solutions`
   get the best first. Cheap and safe; downside: still enumerates non-optimal
   models before ranking (doesn't prune).
2. **Seed with the single optimizing solve.** Run the existing single
   `solve_topology_mapping` once to learn the optimal objective value `V` (min host
   count), then enumerate with a **hard bound at `V`** (at-most-`V` hosts). Safe
   from spurious UNSAT because `V` is *proven achievable* by the single solve, and
   it yields all solutions *at* the optimum. Costs one (possibly slow at SC36)
   optimizing solve up front. Must still be scoped to the inter-mesh solve only.

Recommendation: ship **interim option 1** immediately (removes the regression,
gives best-first ordering), and build the **per-step incremental optimization** as
the real solution, shared by `search_n` and `TopologyMappingEnumerationSession::next`.

## Correctness / test checklist

- `supercluster_20` on SC20 single **and** `--all-solutions` → back to a valid
  mapping (20 hosts), no "Inter-mesh mapping failed" / 0.
- `supercluster_20` on SC36 `--all-solutions --max-solutions N` → N solutions,
  ordered fewest-hosts-first, top one at (or near) the single-solve minimum.
- Physical-grouping placement enumeration unaffected (still finds all embeddings).
- Incremental `TopologyMappingEnumerationSession::next()` returns most-preferred
  remaining each call; repeated calls strictly decrease preference and never
  spuriously report "exhausted" while unblocked models remain.
- Never turns a hard-feasible instance UNSAT (the step-1 unconstrained solve is the
  floor).
