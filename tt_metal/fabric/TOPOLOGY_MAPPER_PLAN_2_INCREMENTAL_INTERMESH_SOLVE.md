# Plan 2 — Incremental inter-mesh solving + stronger rejection

**Priority: 3 (lowest of the three).** Needs a fix inside the solver's SAT session before it is safe.

Tracking issue: [#54623 — \[Auto-mapper\] Verify inter-mesh connectivity in heterogeneous placements via
SAT-based joint planning](https://github.com/tenstorrent/tt-metal/issues/54623)
Related: #40640 (SAT engine), #50510 (epic).
Sibling plans: [Plan 1 — PGD-shape-aware inter-mesh constraints](TOPOLOGY_MAPPER_PLAN_1_PGD_SHAPE_INTERMESH_CONSTRAINTS.md),
[Plan 3 — connectivity-aware PGD placement](TOPOLOGY_MAPPER_PLAN_3_CONNECTIVITY_AWARE_PGD_PLACEMENT.md).

> **Goal.** Replace the stateless `solve_topology_mapping` in the inter-mesh retry loop with the
> already-declared `inter_mesh_session` (`topology_mapper_utils.cpp:3433`), so the hard CNF is encoded
> once and each retry only appends clauses. Then generalize the rejections so each failure prunes more
> than one pair.

---

## 1. The loop today

`map_multi_mesh_to_physical`, `tt_metal/fabric/topology_mapper_utils.cpp:3379`:

```
3400  auto inter_mesh_constraints = build_inter_mesh_constraints(...)
3433  TopologyMappingEnumerationSession<MeshId, MeshId> inter_mesh_session;   // declared, never used
3438  const unsigned int max_retry_attempts = (logical_meshes.size() * physical_meshes.size()) + 1;
3441  while (!success) {
3476      solver_result = solve_topology_mapping(mesh_logical_graph, mesh_physical_graph, ...)  // stateless
3484      if (!ok && max_same_rank_groups_used() > 0) { drop hard host cap; re-solve }
3550      for each (logical, physical) pair: intra-mesh solve
              on failure → handle_forbidden_constraint(...) (3254) → next while iteration
```

Every retry throws away the entire CNF and rebuilds it. The session type that would avoid this is
already declared at `3433` and never used — the variable is dead code, presumably a planned migration.

## 2. Blocking correctness issue — must be fixed first

`TopologyMappingEnumerationSession::next`
(`tt_metal/api/tt-metalium/experimental/fabric/topology_solver.tpp:1280–1377`) only re-encodes the hard
CNF when the **context** changes (target graph, global graph, engine, validation mode, `unique_shapes`)
or when the exclusion list *shrinks* (`1372–1377`). On a context match it rebuilds `constraint_data_`
(`1337`) but leaves `sat_session_` — and therefore the baked-in domains — untouched.

The inter-mesh retry loop tightens `inter_mesh_constraints` between solves by adding **forbidden**
constraints (`handle_forbidden_constraint`, `3254`). Forbidden pairs are enforced as a domain filter at
encode time (`topology_solver_sat.cpp:706–707`), so a forbidden constraint added *after* the encode is
**not guaranteed to reach the CNF**. Naively swapping `solve_topology_mapping` for `session.next(...)`
would let the solver re-propose a pair the loop already rejected — an infinite retry. Not a wrong
answer, but a hang in practice.

Two ways out, in preference order:

1. **Add an incremental-tightening API to the SAT bridge.** A forbidden pair is a unit clause
   `¬x_{t,g}`; CaDiCaL accepts clauses after `solve()` (`topology_solver_sat_solver.cpp:53–57`). Add
   `topology_sat_session_add_forbidden_pair(session, t_idx, g_idx)` next to the existing
   `topology_sat_session_add_blocking_clause` (`topology_solver_sat.cpp:1864–1897`), and have `next()`
   diff the constraint set against the previous call and emit unit clauses for newly forbidden pairs.
   This is the cheap, monotone case: constraints only ever tighten in this loop.
2. **Track a constraint generation counter** in `MappingConstraints` and force a re-encode in `next()`
   when it changes. Correct, but throws away the benefit whenever a pair is rejected — which is exactly
   when the loop is hot.

Option 1 is the real fix; option 2 is a safe stepping stone that at least makes the swap sound.

## 3. Then: the loop change

```cpp
solver_result = inter_mesh_session.next(
    mesh_logical_graph, mesh_physical_graph, inter_mesh_constraints,
    /*excluded_mappings=*/{},          // rejection is expressed via forbidden pairs, not blocking clauses
    inter_mesh_validation_mode, quiet_mode,
    TopologyMappingSolverEngine::Sat, /*unique_shapes=*/false);
```

Note the host-cap relaxation at `3484–3492` mutates `inter_mesh_constraints`
(`set_max_same_rank_groups_used(0)`) — a *loosening*, which unit clauses cannot express. Keep that on
the stateless path, or hoist the decision before the session is created so the cap is decided once.

## 4. Ownership and function passing

Unlike Plans 1 and 3, almost nothing new is threaded through the mapper; the work lands inside the
solver and its SAT bridge.

| Concern | Owner | Notes |
| --- | --- | --- |
| Forbidden-pair unit clause | New `topology_sat_session_add_forbidden_pair(TopologySatSession&, int t_idx, int g_idx)` in `tt_metal/fabric/topology_solver_sat.cpp` (beside the bridge functions at `1864–1897`) | Mirrors `topology_sat_session_add_blocking_clause`; declared in `topology_solver.hpp:972–989` with the other bridge decls |
| Detecting newly forbidden pairs | `TopologyMappingEnumerationSession::next` (`topology_solver.tpp:1280`) | Keep the previous call's forbidden set in the session (it already snapshots graphs); diff against `constraint_data_` on a context match and emit one unit clause per new pair. Session owns the snapshot; `MappingConstraints` stays immutable to the session |
| Index translation | `GraphIndexData` already held by the session (`graph_data_`) | `target_to_idx` / `global_to_idx` are the same maps `to_index_mapping` uses at `topology_solver.tpp:1340–1350`; reuse, do not rebuild |
| Loop ownership | `map_multi_mesh_to_physical` (`topology_mapper_utils.cpp:3441`) | Owns the session (already declared at `3433`); passes `inter_mesh_constraints` by const ref exactly as it does to `solve_topology_mapping` today. No signature change to `build_inter_mesh_constraints` |
| Host-cap relaxation | `map_multi_mesh_to_physical` (`3484–3492`) | Must move *out* of the retry loop or stay stateless — a loosening cannot be expressed incrementally |
| Intra-mesh verdict cache | New file-local `std::map<std::pair<MeshId, MeshId>, bool>` in `map_multi_mesh_to_physical` | Owned by the mapping call, not the session; scope is one `map_multi_mesh_to_physical` invocation |

## 5. "Additional forbidden constraints"

Once intra-mesh failure is understood as a **pair-local** property (the intra-mesh solve for
`logical L → physical P` depends only on that pair's two subgraphs, not on the rest of the assignment),
each observed failure can be generalized instead of recorded one pair at a time:

- **Shape-class generalization.** If `L → P` failed, and `L'` has the same MGD descriptor name as `L`
  and `P'` the same PGD grouping as `P`, then `L' → P'` will fail too. Emit the whole cross product with
  the existing `add_forbidden_constraint(std::set targets, std::set globals)`
  (`topology_solver.hpp:301`). With Plan 1 in place most of these are already unreachable — which is
  precisely why this plan is the lowest priority of the three.
- **Capacity generalization.** If `L` has more fabric nodes than `P` has ASICs, forbid at
  constraint-build time rather than after a failed solve. Cheap precheck in
  `build_inter_mesh_constraints`.
- **Cache the intra-mesh verdict** per `(logical, physical)` pair so repeated attempts inside one
  mapping never re-run the sub-solve.

## 6. Validation

- `sat_hard_constraint_encode_calls()` / `sat_solve_calls()` are already exposed on the session
  (`topology_solver.hpp:1516–1517`) and already asserted in
  `tests/tt_metal/tt_fabric/fabric_router/test_topology_solver.cpp:5973+`. Assert encode calls == 1
  across a multi-retry inter-mesh mapping.
- Direct test for the §2 bug: drive a session to a solution, add a forbidden constraint for the pair it
  chose, call `next()` again, and assert the returned mapping does not contain that pair. This test
  should **fail before** the bridge fix and pass after — it is the regression guard for the whole plan.
- Regression: a case that *must* retry (force an intra-mesh failure) and assert it still converges to
  the same mapping as the stateless path.

## 7. Open questions

- Is intra-mesh failure truly pair-local in all cases, including PGD-pinned meshes where
  `mesh_pgd_pinnings_` constrains the intra-mesh solve? If a pinning makes the verdict
  context-dependent, the shape-class generalization in §5 is unsound as stated and must be restricted
  to the capacity case.
- Should the session expose an explicit `tighten(constraints)` entry point instead of diffing inside
  `next()`? Diffing is transparent to callers; an explicit call is cheaper and harder to misuse. The
  diff cost is `O(pairs)` per retry, which is small next to a re-encode but not free.
- Does anything else mutate `inter_mesh_constraints` mid-loop besides the host cap and the forbidden
  pairs? If a future caller adds a loosening, the incremental path silently goes stale; a generation
  counter (option 2 in §2) would at least turn that into a re-encode rather than a wrong domain.
