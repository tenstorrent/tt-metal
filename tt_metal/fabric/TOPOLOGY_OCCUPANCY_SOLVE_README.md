# Topology SAT — host-group occupancy solving (descent + full-packing lock)

How the topology mapper's SAT solver finds a **minimal-host** placement of a logical
mesh graph onto a physical cluster, and why it does so via an *incremental* warm
descent followed by a full-packing "lock" rather than a single cold solve.

- Code: `tt_metal/fabric/topology_solver_sat.cpp`
- Solver facade: `tt_metal/fabric/topology_solver_sat_solver.{hpp,cpp}` (thin wrapper over CaDiCaL)
- Constraint plumbing: `tt_metal/fabric/topology_mapper_utils.cpp`
  (`add_inter_mesh_minimal_host_cover_from_hostname_map` sets the cap)

The running example throughout is the **80-stage Blitz-decode ring on the 36-host
SC36 mock** (`blitz_decode_mesh_graph_descriptor_supercluster_20.textproto` on
`SC36_32x4_revC_subtorus_aisleD`): 80 logical meshes, 144 physical mesh slots, and a
minimal-host target of **k=20** hosts (80 meshes ÷ 4 meshes/host).

---

## 1. The problem

The mapper must embed a logical multi-mesh graph (a ring of 80 meshes) into a physical
multi-mesh graph (144 slots across 36 hosts) as an **injective, adjacency-preserving**
mapping. On top of correctness, we want to use **as few physical hosts as possible** so
a job doesn't scatter across the whole cluster.

Hosts are modeled as **same-rank groups**: each group is the set of physical mesh slots
that live on one host (4 slots/host here). Minimizing host usage = minimizing the number
of *occupied groups*.

There are two ways the caller can express this (both in `MappingConstraints`):

| Constraint | Meaning |
|---|---|
| `set_max_same_rank_groups_used(K)` | **HARD** cap: at most `K` groups may be occupied; the solver picks which `K`. |
| `set_minimize_same_rank_groups_used(true)` | **SOFT** objective: minimize occupied groups, best-effort. |

`add_inter_mesh_minimal_host_cover_from_hostname_map` sets **both**: the hard cap at the
capacity lower bound `k_min` (=20 here) *and* the soft minimize as a fallback:

```1685:1686:tt_metal/fabric/topology_mapper_utils.cpp
inter_mesh_constraints.set_max_same_rank_groups_used(k_min);      // HARD: fit within k_min hosts (any k_min)
inter_mesh_constraints.set_minimize_same_rank_groups_used(true);  // SOFT fallback: minimize if the cap is infeasible
```

---

## 2. Occupancy encoding building blocks

All three live in `topology_solver_sat.cpp` and operate on the base hard encoding
(`TopologySatHardEncoding enc`, which holds the per-target assignment literals
`enc.assign_lit[t][i]`).

### 2a. `topology_sat_build_group_occupancy(...)`  (occupancy indicators)

For each non-empty host group `g` it builds:

- a per-mesh-slot **used** indicator `used_m ⇔ OR(assign lits landing a target on slot m)`, and
- a per-group **occupied** indicator `occ_g ⇔ OR(used_m for m in g)`.

Optionally (`used_per_group_out`) it returns the raw `used_m` lists so a later stage can
add the full-packing tightening *in place* on the same solver.

When called with `all_or_nothing=true` it *also* emits, for every group,
`occ_g ⇒ used_m` for every slot `m` of the group — i.e. **"if a host is used at all, it
is completely full."**

### 2b. Full packing and why it's the fast path

Full packing is valid **only** when a used host must be completely filled: uniform group
capacity and an exact multiple, `n_target == K * max_cap` (80 == 20 × 4). Under that
condition the all-or-nothing clauses **alone** force the host count: an injective
placement of exactly `K*cap` meshes into all-or-nothing hosts of capacity `cap` occupies
exactly `K` hosts. So the fast path **skips the cardinality counter entirely**:

```751:761:tt_metal/fabric/topology_solver_sat.cpp
inline bool topology_sat_encode_at_most_k_groups(
    TopologySatSolver& solver,
    const TopologySatConstraintView& constraint_data,
    const TopologySatHardEncoding& enc,
    size_t k_hosts,
    bool full_packing) {
    std::vector<int> occ;
    topology_sat_build_group_occupancy(solver, constraint_data, enc, /*all_or_nothing=*/full_packing, occ);
    if (full_packing) {
        return true;  // all-or-nothing already forces exactly k_hosts occupied; no counter needed
    }
```

The all-or-nothing clauses are **local** (per host) and propagate strongly via unit
propagation. The alternative — a sequential-counter "at most K" over the `occ_g` — adds
aux variables whose propagation is weak and was the historical bottleneck.

### 2c. `topology_sat_add_all_or_nothing_tightening(...)`  (in-place upgrade)

The crucial primitive for the incremental strategy. It takes an occupancy encoding that
was built **without** all-or-nothing (partial packing) and adds the `occ_g ⇒ used_m`
clauses afterward, on a solver that has already been solving. This lets a **warm**
solver — full of learned clauses and saved variable phases from a partial-packing
descent — be tightened to full packing *in place*, then re-solved for the minimal-host
lock, reusing all of that accumulated work:

```705:715:tt_metal/fabric/topology_solver_sat.cpp
inline void topology_sat_add_all_or_nothing_tightening(
    TopologySatSolver& solver, const std::vector<int>& occ, const std::vector<std::vector<int>>& used_per_group) {
    const size_t n = std::min(occ.size(), used_per_group.size());
    for (size_t i = 0; i < n; ++i) {
        for (int um : used_per_group[i]) {
            solver.add(-occ[i]);
            solver.add(um);
            solver.add(0);
        }
    }
}
```

---

## 3. The incremental strategy (single-solution)

Implemented in `topology_sat_solve_minimize_groups(...)`
(`topology_solver_sat.cpp:782`). It runs on **one** `CaDiCaL` solver instance so every
stage inherits the previous stage's learned clauses and phase saving. Three stages:

### Stage 1 — warm feasible solve

Build **partial-packing** occupancy (no all-or-nothing) plus a single shared "at least j
groups UNoccupied" counter (so any bound `<= k occupied` is one assumption literal).
Then solve once, unconstrained. A plain ring embedding is easy, so this is fast (~5 ms)
and yields a feasible model with some occupied count (measured: **27**).

### Stage 2 — bounded descent (the warm-up)

Repeatedly assume `<= (current-1)` occupied and re-solve **under a small conflict
budget** (`solve_limited`). On SAT, the count may drop by more than one; keep the best
model. On UNSAT the current best is provably optimal; on **unknown** (budget exhausted)
we stop and keep the best proven model — *without* failing:

```862:896:tt_metal/fabric/topology_solver_sat.cpp
    while (best_k_out > floor) {
        const size_t target_k = best_k_out - 1;
        const int bound = atmost_lit(target_k);
        if (bound == 0) {
            break;
        }
        solver.assume(bound);
        auto t_iter = std::chrono::steady_clock::now();
        const int st = (conflict_cap > 0) ? solver.solve_limited(conflict_cap) : solver.solve();
        ++iter;
        if (st == TopologySatSolver::kSat) {
            best_k_out = count_occupied();  // may drop by more than one
            topology_sat_decode_hard_solution(solver, enc, best_mapping_out);
            ...
        } else {
            ...
            break;  // kUnsat: optimal reached.  kUnknown: too hard -> keep best proven.
        }
    }
```

The descent is deliberately **only a warm-up**. The deep steps (e.g. `target<=22`) are
combinatorially hard; the point is to cheaply drive the solver near `K` and pack its
clause database, *not* to reach `K` by descent. This is why the **descent budget is
small** (see §4).

### Stage 3 — full-packing hard lock

If a hard cap `K` was requested and the descent didn't already reach it, do one more
solve on the **same warm solver**: add the all-or-nothing tightening in place, assume
`<= K occupied`, and solve under a **large** budget. The all-or-nothing clauses give the
strong unit propagation the partial descent lacked, and the warm state (learned clauses +
phases pointing at a real ring embedding) is the strongest possible start — so this
cracks the exact minimal-host packing where cold solving stalls:

```904:918:tt_metal/fabric/topology_solver_sat.cpp
    if (hard_cap_k > 0 && !best_mapping_out.empty() && best_k_out > hard_cap_k && hard_cap_k < num_present) {
        topology_sat_add_all_or_nothing_tightening(solver, occ, used_per_group);
        const int bound = atmost_lit(hard_cap_k);
        auto t_lock = std::chrono::steady_clock::now();
        if (bound != 0) {
            solver.assume(bound);
        }
        const int st = (hard_conflict_cap > 0) ? solver.solve_limited(hard_conflict_cap) : solver.solve();
        if (st == TopologySatSolver::kSat) {
            best_k_out = count_occupied();
            topology_sat_decode_hard_solution(solver, enc, best_mapping_out);
            if (hard_cap_met_out != nullptr) {
                *hard_cap_met_out = (best_k_out <= hard_cap_k);
            }
        }
        ...
    }
```

**Soundness:** on UNSAT/unknown at any stage we keep the best descent model, so the lock
can never regress a feasible result — worst case it returns the soft best (k=23–24).

---

## 4. The two-budget rule (why ~90s, not 5+ min)

The descent and the lock get **separate** conflict budgets. This is the single most
important tuning decision, set in the occupancy path of `topology_sat_search`:

```2000:2003:tt_metal/fabric/topology_solver_sat.cpp
        const int kGroupObjectiveConflictBudget =
            static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_CONFLICT_BUDGET", 1'000'000));
        const int kGroupDescentConflictBudget =
            static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_DESCENT_BUDGET", 20'000));
```

- **Descent budget** — `TT_TOPO_SAT_DESCENT_BUDGET`, default **20 000**. Small on
  purpose: deep descent steps bail in well under a second and hand a warm solver to the
  lock.
- **Lock / cold budget** — `TT_TOPO_SAT_CONFLICT_BUDGET`, default **1 000 000**. The lock
  empirically needs ~250–270k conflicts to reach k=20, so 1M is a safe margin.

### Measured — single 1M budget for both (the regression)

```
minimize.warm_solve    :       4.5 ms (SAT, occupied=27)
minimize.descent[1]    :     312.4 ms (SAT, occupied=26)
minimize.descent[2]    :     552.4 ms (SAT, occupied=24)
minimize.descent[3]    :  21 998.3 ms (SAT, occupied=23)
minimize.descent[4]    : 318 803.3 ms (status=0 -> stop)   ← one deep step burns the whole 1M budget
```

The deep step grinds for **318 s** before the lock ever runs.

### Measured — two-budget (descent 20k + lock 1M), current default

```
minimize.warm_solve    :     4.2 ms (SAT, occupied=27)
minimize.descent[1]    :   315.3 ms (SAT, occupied=26)
minimize.descent[2]    :   530.3 ms (SAT, occupied=24)
minimize.descent[3]    :  5 322.7 ms (status=0 -> stop)     ← bails fast at 20k
solve_minimize_groups  : 88 452.3 ms (ok=true, best_k=20, hard_cap_met=true)
```

**~88 s end-to-end, k=20, hard cap met**, valid `rank_bindings.yaml` written.

---

## 5. Which code path uses the strategy today

`topology_sat_search` (single solution) routes the occupancy objective three ways:

| Constraints set | Path | Uses descent+lock? |
|---|---|---|
| `minimize` (± `max`) | `topology_sat_solve_minimize_groups` → **descent + lock** | ✅ yes |
| `max` only (no `minimize`) | cold `encode_at_most_k_groups` + phase-hint warm-start + one `solve_limited` (`topology_solver_sat.cpp:2028`) | ⚠️ cold, phase-hinted only |
| neither | plain solve | n/a |

The cold `max`-only path also warm-starts (solve the ring-only encoding, pin the model as
sticky CaDiCaL phase hints, *then* add the packing clauses) — but it does **not** do a
descent, so it is weaker than the `minimize` path and mostly relevant for callers that
set only the hard cap.

**The gap:** the other two entry points do **not** use the strategy at all —

- `topology_sat_search_n` (multi-solution `--all-solutions`) — `topology_solver_sat.cpp:2304`
- the incremental session (`.next`) — `topology_sat_session_create_and_encode` /
  `_solve_and_decode`, `topology_solver_sat.cpp:2421`

Both encode the **cold** cap (`encode_at_most_k_groups` with full packing) and then call
an **unbounded** `solver.solve()`. For the 80-ring/SC36 case that first solve is exactly
the combinatorial thrash the descent+lock avoids — measured **>6 min with zero solutions
produced** and no `solutions_index.yaml` written.

---

## 6. Plan — port the strategy into multi-solve (`topology_sat_search_n`)

**Goal:** the first enumerated solution should cost ~the single-solution ~90s, and each
subsequent solution should be cheap (incremental, cap already locked).

**Current code (to replace):**

```2350:2378:tt_metal/fabric/topology_solver_sat.cpp
    if (constraint_data.max_same_rank_groups_used > 0) {
        const size_t K = constraint_data.max_same_rank_groups_used;
        if (topology_sat_max_groups_cap_capacity_feasible(constraint_data, graph_data.n_target, K)) {
            ...
            const bool full_packing = (max_cap > 0 && graph_data.n_target == K * max_cap);
            topology_sat_encode_at_most_k_groups(solver, constraint_data, enc, K, full_packing);
        }
        ...
    }
    ...
    while (all_mappings_out.size() < max_solutions) {
        const int status = solver.solve();          // ← UNBOUNDED cold solve
        if (status != TopologySatSolver::kSat) break;
        ...
    }
```

### Step 1 — refactor the strategy into a reusable "prime the solver" helper

Extract stages 1–3 of `topology_sat_solve_minimize_groups` into a helper that leaves the
solver **primed and permanently capped**, returning the first model:

```cpp
// Prime `solver` for minimal-host enumeration: run the warm descent + full-packing lock,
// then make the cap PERMANENT (unit clause, not an assumption) so every later solve() is
// bounded by it. Returns the first (minimal-host) mapping, or std::nullopt on failure.
struct GroupOccupancyLock {
    std::vector<int> occ;                 // occupancy indicators (for count/debug)
    int atmost_bound_lit = 0;             // the "<= K occupied" literal, added as a unit clause
    size_t best_k = 0;
    bool hard_cap_met = false;
};
std::optional<std::vector<int>> topology_sat_prime_minimal_host_solver(
    TopologySatSolver& solver, const TopologySatHardEncoding& enc,
    const TopologySatConstraintView& constraint_data,
    int descent_budget, int lock_budget, size_t k_floor, size_t hard_cap_k,
    GroupOccupancyLock& lock_out);
```

Key difference from the single-solution version: the cap must be **permanent** across the
enumeration loop. `assume()` is one-shot (cleared after each `solve()`), so instead add
the bound as a **unit clause** once the lock proves it SAT:

```cpp
if (lock_status == kSat && best_k <= hard_cap_k) {
    solver.add(lock.atmost_bound_lit);  // permanent "<= K occupied"
    solver.add(0);
}
```

`topology_sat_solve_minimize_groups` then becomes a thin wrapper over the helper (assume
the bound instead of adding it as a unit clause), so the single-solution behavior is
unchanged.

### Step 2 — rewrite the `search_n` cap block to prime, then enumerate

```cpp
GroupOccupancyLock lock;
auto first = topology_sat_prime_minimal_host_solver(
    solver, enc, constraint_data,
    /*descent*/ kGroupDescentConflictBudget, /*lock*/ kGroupObjectiveConflictBudget,
    k_floor, /*hard_cap_k=*/K, lock);
if (first) {
    all_mappings_out.push_back(*first);
    topology_sat_add_blocking_clause_for_mapping(solver, enc, all_mappings_out.back(), unique_shapes);
}
// then the existing while-loop; the permanent unit clause keeps each solve() capped
while (all_mappings_out.size() < max_solutions) {
    const int status = solver.solve();   // now warm + permanently capped
    ...
}
```

### Step 3 — bound the enumeration loop solves too

Subsequent solutions differ only by blocking clauses, so they're usually fast — but to
guarantee CI never hangs, use `solve_limited(kGroupObjectiveConflictBudget)` in the loop
and treat `unknown` as "stop enumerating" (return what we have). This makes `search_n`
**always terminate**.

### Step 4 — fallback if the lock can't hit K

If priming fails to reach `K` (returns `hard_cap_met=false`), fall back to enumerating at
the **soft best** `best_k` (add `<= best_k` as the permanent unit clause instead of `K`),
so enumeration still produces solutions rather than none.

### Expected cost

- First solution: ~90s (same as single-solution).
- Each subsequent: seconds (warm, capped, only a blocking clause added).
- 5 solutions ≈ 90s + 4×(small) rather than 5×(cold grind).

---

## 7. Plan — port the strategy into incremental solve (session `.next`)

The session API (`topology_sat_session_*`, `topology_solver_sat.cpp:2421`) is the
streaming version of `search_n`: encode once, then `add_blocking_clause` +
`solve_and_decode` repeatedly. It has the **same** cold-cap defect.

### Step 1 — prime at session creation

In `topology_sat_session_create_and_encode`, after `encode_hard_constraints`, call the
same `topology_sat_prime_minimal_host_solver` helper (Step 1 of §6) instead of the cold
`encode_at_most_k_groups`. Stash the first primed model + the `GroupOccupancyLock` on the
session so the first `solve_and_decode` can return it without re-solving:

```cpp
struct TopologySatSession {
    TopologySatSolver solver;
    GroupOccupancyLock occ_lock;                 // NEW
    std::optional<std::vector<int>> primed_first; // NEW: first minimal-host model
};
```

Because the cap is added as a **permanent unit clause** during priming, every later
`session->solver.solve()` is automatically capped — no code change needed in
`_add_blocking_clause`.

### Step 2 — hand back the primed model first

```cpp
bool topology_sat_session_solve_and_decode(TopologySatSession* session, const TopologySatHardEncoding& enc,
                                           std::vector<int>& raw_out) {
    if (session->primed_first) {                       // first call: return the lock's model
        raw_out = std::move(*session->primed_first);
        session->primed_first.reset();
        return true;
    }
    if (session->solver.solve() != kSat) return false; // subsequent: warm + capped
    return topology_sat_decode_hard_solution(session->solver, enc, raw_out);
}
```

(If callers might not consume the first model before adding a blocking clause, guard the
blocking-clause path to also drain `primed_first` — or simply always route through the
solver after priming and skip stashing the model, at the cost of one extra warm solve.)

### Step 3 — budget + fallback

Same as §6 Steps 3–4: bound subsequent `.next` solves and fall back to the soft best if
the lock can't reach `K`.

### Shared consideration — CaDiCaL incrementality

The whole strategy depends on the solver **retaining learned clauses and phase saving
across solves**. Sessions already call `configure_for_blocking_clause_enumeration()`
(`topology_solver_sat.cpp:2427`), which puts CaDiCaL in the right incremental mode, so
priming + permanent-unit-clause capping + repeated `solve()` is exactly the supported
usage. The only correctness rule: the cap must be a **clause**, never a lingering
assumption, because assumptions are cleared after each `solve()`.

---

## 8. Testing plan

1. **Unit (fast, no MPI)** — in `tests/tt_metal/tt_fabric/fabric_router/test_topology_solver.cpp`:
   add a synthetic ring-into-larger-graph case with same-rank groups where `full_packing`
   holds, assert (a) single-solution reaches `k = k_min` with `hard_cap_met`, (b)
   `search_n` returns ≥2 distinct minimal-host solutions, each occupying exactly `k_min`
   groups, and (c) the session `.next` path returns the same first model as `search_n`.
2. **Budget regression guard** — assert a deep descent step never exceeds a wall-clock
   bound at the default budgets (catch a re-consolidation to a single budget).
3. **Mock end-to-end (CPU)** — once `search_n` is primed, add a **multisolution sweep**
   section to `tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh` (new group,
   e.g. `bh-ring-sweep`) that runs `tools/scaleout/sweep_rank_binding_solutions.py` on the
   SC36 mock + 80-stage MGD with a small `--max-solutions` and a `--per-solution-timeout`,
   asserting `summary.found >= 1` and one `sweep_run.log` per solution.

---

## 9. Environment variable reference

| Variable | Default | Effect |
|---|---|---|
| `TT_TOPO_SAT_CONFLICT_BUDGET` | `1000000` | Cold hard cap + final full-packing lock conflict budget. |
| `TT_TOPO_SAT_DESCENT_BUDGET` | `20000` | Per-step budget for the partial-packing descent (keep small). |
| `TT_TOPO_SAT_HARDCAP_WARMSTART` | `1` | Phase-hint warm start for the cold `max`-only path (A/B toggle). |
| `TT_TOPO_SAT_PROFILE` | `0` | Emit `[topo-sat-profile]` per-stage timing + CNF-size lines. |

Set `TT_TOPO_SAT_PROFILE=1` to see `occupancy path:`, `minimize.warm_solve`,
`minimize.descent[i]`, and `minimize.hardlock` lines — the exact traces quoted in §4.
