# Minimal-host inter-mesh mapping via same-rank-group occupancy constraints

Status: **implemented + validated**. Minimal host usage is expressed as a constraint over host-group
*occupancy*, driven entirely by `MappingConstraints` — the solver enforces it; there is no bespoke
host-minimization objective baked into the SAT search.

## Goal

When a multi-mesh MGD is placed on a larger physical system, use the **minimum number of hosts**
(`k_min = ceil(num_meshes / max host capacity)`, e.g. 64 × (4×2) meshes at 4/host ⇒ **16 hosts**), and let
the solver pick **which** hosts (any combination) rather than pinning to one specific cover.

## API (`MappingConstraints`)

Two opt-in knobs, both over the registered same-rank global groups (host partitions):

- **HARD** — `set_max_same_rank_groups_used(k)`: at most `k` host groups may be *occupied*. The solver
  chooses which `k`.
- **SOFT** — `set_minimize_same_rank_groups_used(true)`: best-effort minimize the number of occupied
  groups. Used as the **fallback** when the hard cap can't be met.

The inter-mesh setup (`add_inter_mesh_minimal_host_cover_from_hostname_map`) registers the host partitions
as same-rank global groups and sets **both**: `set_max_same_rank_groups_used(k_min)` +
`set_minimize_same_rank_groups_used(true)` → "fit in `k_min` hosts if possible, else minimize best-effort".

## Encoding (`topology_solver_sat.cpp`)

Per non-empty host group `g`, an **occupancy indicator** `occ_g ⇔ (some target maps into a global of g)`
(`topology_sat_build_group_occupancy`). Then:

- **`topology_sat_encode_at_most_k_groups(..., k, full_packing)`** (HARD): applied in `encode`-time for the
  single solve, the enumeration (`topology_sat_search_n`) and the incremental session, so *every* solution
  respects the cap.
  - **Fast path — `full_packing`** (uniform capacity and `n_target == k · capacity`, i.e. a used host must
    be completely full): add only the **all-or-nothing** clauses `occ_g ⇒ every mesh of g is used`. With an
    injective placement of exactly `k·capacity` meshes into all-or-nothing hosts, **exactly `k` hosts end up
    occupied — forced structurally, with no cardinality counter.** These are local per-host clauses with
    strong unit propagation.
  - **General path**: an explicit "at least `(num_groups − k)` groups unoccupied" via the sequential-counter
    at-least-k machinery.

- **`topology_sat_solve_minimize_groups(...)`** (SOFT): one warm feasible solve, then descend an assumable
  "at most (current − 1)" occupancy budget under a bounded conflict cap, keeping the best (fewest-group)
  model. Floors at `k_min` so it never probes below the achievable minimum. Never turns a feasible instance
  UNSAT.

Single-solve tries HARD first; on UNSAT/too-hard it backs down to SOFT. Enumeration/session apply the HARD
cap only (they rely on it being satisfiable, which it is for well-formed superpods).

## Why not the earlier attempts

- **SAT host-budget minimize walk** (`#45712`, reverted): walked `at-most-k` from `k_min` up under a 300k
  conflict cap; `k=16` hit the cap and it settled for 17. Removed from the SAT solver entirely.
- **Pin to the greedy cover** (`add_required_constraint` to `find_minimum_coverage_group`'s globals): the
  greedy cover is chosen by mesh **count**, ignoring connectivity, so pinning the 64-mesh ring into that
  specific 16-host set was intractable (>30 min / likely UNSAT). Constraining the **count** (this design)
  lets the solver choose a routable set.
- **at-most-K cardinality counter** (occupancy + sequential counter): correct but the counter's weak
  propagation made it ~17 min. The **all-or-nothing full-packing** insight removes the counter on the fast
  path.

## Validation (Release, SC16 blitz-superpod MGD on the SC24 `subtorus_virtu` mock)

| Path | Result | Time |
|---|---|---|
| Single solve | **16 hosts** (4 meshes/host) | **13 s** |
| `--all-solutions --max-solutions 3 --distinct-host-sets` | **3 distinct solutions, all 16 hosts** | **12 s** |
| Plain solve (no objective) | 20 hosts | 6 s |

The minimal-host solve is now essentially as fast as a plain feasible solve, versus ~13–17 min for the
counter/budget approaches.

## Notes / limits

- The `full_packing` fast path requires uniform host capacity and `n_target` a multiple of it (true for the
  blitz superpod MGDs). Otherwise the general at-most-k counter path is used (slower, still correct), with the
  soft-minimize fallback.
- DFS backend still approximates the objective via its host-affinity value ordering (unchanged).
