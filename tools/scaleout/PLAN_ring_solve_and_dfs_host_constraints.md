# Plan / handoff: host-count constraints across backends + fast ring solving

Status: **plan + situation writeup** (some pieces implemented, most not). Hand-off doc for continuing in Cursor.
Companion to [`PLAN_hard_host_count_constraint.md`](PLAN_hard_host_count_constraint.md) (the occupancy-constraint design) and the repo-root [`HANDOFF.md`](../../HANDOFF.md).

---

## 1. Background: the minimal-host objective (what exists today)

Minimal host usage for inter-mesh mapping is expressed as constraints over **same-rank-group occupancy** (host
partitions), on `MappingConstraints`:

- `set_max_same_rank_groups_used(k)` — **HARD** "at most `k` host groups occupied" (`k = ceil(num_meshes /
  max_host_capacity)`, e.g. 64 meshes @ 4/host ⇒ 16).
- `set_minimize_same_rank_groups_used(true)` — **SOFT** best-effort minimize (fallback if the hard cap can't be met).

The mapper (`add_inter_mesh_minimal_host_cover_from_hostname_map`, `topology_mapper_utils.cpp`) registers the host
partitions as same-rank global groups and sets both. Encoding lives in `topology_solver_sat.cpp`
(`topology_sat_build_group_occupancy`, `topology_sat_encode_at_most_k_groups` with a full-packing all-or-nothing
fast path, `topology_sat_solve_minimize_groups`). See `PLAN_hard_host_count_constraint.md` for the full design.

---

## 2. THE GAP: these constraints are SAT-only; DFS ignores them

There are two solver backends (`TopologyMappingSolverEngine::{Sat, Dfs, Auto}`). Both receive the flags (copied
into `ConstraintIndexData`, `topology_solver.tpp:2057-2058`), but **only SAT consumes them.**

| Constraint | SAT backend | DFS backend |
|---|---|---|
| `max_same_rank_groups_used` (HARD cap) | Enforced (occupancy at-most-k) | **NOT enforced** — DFS may return > k host groups |
| `minimize_same_rank_groups_used` (SOFT) | Honored (warm descent) | Flag **not read**; DFS instead applies an *unconditional* host-affinity packing bias |

**DFS's mechanism** (`SearchHeuristic::compute_candidate_cost`, `topology_solver.tpp:2375-2414`): a value-ordering
cost `-host_affinity_score * HOST_AFFINITY_WEIGHT(10000) - is_preferred*SOFT_WEIGHT(1000) - channel_match +
degree_gap`. `host_affinity_score` counts already-mapped globals on the candidate's host, so DFS greedily packs
onto already-used hosts (best-fit bin-packing). But it is:
- **greedy/soft** — the first feasible packing-biased mapping, not a guaranteed minimum, no hard bound;
- **unconditional** — active whenever same-rank groups are registered, regardless of the `minimize`/`max` flags.

**Practical impact:** with `Auto`/`Sat` (default, and what our runs use) both constraints work. If someone forces
`Dfs` with a hard cap set, **the cap is silently ignored**. In-code notes were added at
`topology_solver.tpp:2057` (flag copy) and `:2375` (host-affinity block) documenting this.

### Is closing the DFS gap easy? (assessment)

- **Hard cap on DFS — MODERATE-EASY.** DFS is backtracking that assigns targets one at a time. Track the set/count
  of currently-occupied same-rank groups in the search state; when generating candidates, **prune any candidate
  whose host group is new and would make the occupied count exceed `k`** (a hard filter in the
  candidate-generation step, alongside the existing degree/validity filters). Occupied groups only grow going
  down a branch and backtracking restores them, so rollback is natural. Main work: thread an
  `occupied_groups`/count through the DFS state and add the prune. Low risk (pure pruning; can't create UNSAT that
  wasn't already there — if k is infeasible, DFS should mirror the SAT fallback and relax/soft-minimize).
- **Soft minimize flag on DFS — TRIVIAL.** The packing bias already exists; just decide whether to gate it on
  `minimize_same_rank_groups_used` (currently always-on). One conditional. Note: making it a *guaranteed* minimum
  (not greedy) would need branch-and-bound / iterative-deepening on host count — that's more work, but the
  best-effort behavior is already there.
- **Full-packing all-or-nothing tightening** (the SAT fast path) has no direct DFS analog, but the hard-cap prune
  above is sufficient to enforce the bound; DFS's greedy packing tends to fill hosts anyway.

Recommended: implement the hard-cap prune + flag-gate, and add a **guardrail** — if `max_same_rank_groups_used > 0`
and the engine resolves to DFS without the prune, either force SAT or log a loud warning.

---

## 3. Performance situation: ring embedding is the real bottleneck

The SC36 ring-stress (`run_fabric_cpu_only_unit_tests.sh`, `bh-ring-stress`) now maps **full-host-count rings**
(80/96/112/128/144-stage = 20/24/28/32/36 hosts) onto the 36-host SC36 mock. These are **slow**:

- SC24/SC16 (64-mesh ring on 24 hosts): single-solve ~13 s.
- SC36 144-stage (144-mesh ring on 36 hosts): single-solve **≫ 4 min** (killed before finishing).

**Root cause:** embedding an N-node **ring** is a Hamiltonian-cycle-like search, made worse by **symmetry** — a ring
has **2N automorphisms** (N rotations × 2 reflections) and the physical torus has its own; CDCL re-derives
equivalent conflicts under each symmetric image and thrashes. Cost scales with ring length and, for sub-full
rings, with host-selection slack (C(36,20) ≈ 7.3 B for the 80-stage/20-host case). It is **not** a regression and
**not** the host-minimization per se — the ring embedding itself is the cost.

**The existing symmetry breaking is ineffective here** (`topology_sat_symmetry_assumption_lit`,
`topology_solver_sat.cpp:1549`):
1. **Not applied on the occupancy path** — only wired into `topology_sat_search`'s plain/preferred solve via
   `solve_with_symmetry_break`; the occupancy hard-cap solve, `search_n`, and the session have **no** symmetry
   breaking. Our rings go through the occupancy path.
2. **Gated on `n_target == n_global`** (exact bijection) — **do not use this**: many rings are *smaller* than the
   global graph (80-mesh ring on 144-mesh SC36), so equality skips exactly the cases we need.
3. **Rotation only** — fixes node 0's image; does not break the reflection (clockwise/CCW) twin.

---

## 4. Plan: fast ring solving

### 4A. Structural ring/snake detection (replace the size gate)

Detect on the **logical target graph** — cheap O(V+E), independent of global size:
- **Ring (cycle):** connected **and** every node degree == 2 **and** `edges == nodes`.
- **Snake (path):** connected **and** exactly two degree-1 endpoints, rest degree 2, `edges == nodes-1`.

**Remove** `topology_sat_symmetry_assumption_lit`'s `if (n_target != n_global) return 0;` and replace with this
detector. Fires for any ring/snake whether it fills the global graph or a subset.

### 4B. Ring-aware symmetry breaking, on ALL SAT paths

- **Rotation anchor:** fix node 0 → `assign_lit[0][0]` as an **assumption** (retracted on UNSAT ⇒ sound for any
  instance, including sub-full rings).
- **Reflection:** assert `img(v1) < img(v_{N-1})` (v0's two ring-neighbors) as a **hard lex clause** — sound for a
  *detected* ring/snake because the mirror solution always exists; removes the CCW twin (extra 2×).
- **Wire into the occupancy path** (single hard-cap solve, `topology_sat_search_n`, session) — the missing gap.

### 4C. Measure, then remove the old break

- Run **SC20** (supercluster_20 = 80-mesh ring on 80 physical — the old code's exact bijection case) old-vs-new,
  and an **SC36** subset ring (80-mesh on 144, which the old gate never helped). Remove the old size-gated path
  once new ≥ old everywhere.

### 4D. Snake / Hamiltonian shortcut (deeper, only if 4A-C insufficient)

Still enforcing all SAT constraints (channels, host groups, occupancy, reflection):
- **Low-risk:** "snake" build order — bias the search to assign ring nodes in cycle order, each preferring a
  physical neighbor of the previous node's image (phase/decision hints) so CDCL grows the ring contiguously.
- **Deeper:** a constructive constrained-cycle pre-pass (dedicated DFS walking physical adjacency to find a
  length-N cycle honoring channel/host/occupancy filters) that seeds/warm-starts the SAT to verify + complete the
  rest. Bigger change; only if snake-ordering doesn't close the gap.

---

## 5. Suggested sequencing (measure-gated)

1. Structural ring/snake detector (4A).
2. Rotation anchor + reflection lex, wired into the occupancy path (4B).
3. Measure SC20 + an SC36 subset ring (4C); remove the old size gate if new ≥ old.
4. If still too slow → snake ordering (4D low-risk); constructive pre-pass only if needed.
5. Separately: close the DFS host-constraint gap (§2) — hard-cap prune + flag-gate + guardrail.

---

## 6. Key file / function reference

| What | Location |
|---|---|
| Occupancy encoding (hard cap, soft minimize, full-packing) | `tt_metal/fabric/topology_solver_sat.cpp` — `topology_sat_build_group_occupancy`, `topology_sat_encode_at_most_k_groups`, `topology_sat_solve_minimize_groups` |
| Existing (size-gated, rotation-only) symmetry break | `topology_solver_sat.cpp:1549` `topology_sat_symmetry_assumption_lit` + `solve_with_symmetry_break` |
| Single / enumeration / session solve paths | `topology_sat_search`, `topology_sat_search_n`, `topology_sat_session_create_and_encode` (same file) |
| DFS host-affinity bias (SAT-only flags NOT read) | `tt_metal/api/.../topology_solver.tpp:2375` `SearchHeuristic::compute_candidate_cost`; flag copy + TODO at `:2057` |
| Mapper sets max + minimize | `tt_metal/fabric/topology_mapper_utils.cpp` `add_inter_mesh_minimal_host_cover_from_hostname_map` |
| Ring-stress test + MGDs | `tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh` (`bh-ring-stress`); `models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_ring_{96,112,128,144}stage…` + `…supercluster_20` (80) |
| Occupancy-cap unit tests | `tests/tt_metal/tt_fabric/fabric_router/test_topology_solver.cpp` (`*MaxSameRankGroups*`) |

## 7. How to build / measure (quick reference)

- Build: `/usr/local/bin/cmake --build build_Release --target generate_rank_bindings -j 16` (cmake 4.0.2 at
  `/usr/local/bin`; `/usr/bin/cmake` 3.22 is too old). Tests: `--target fabric_unit_tests` in `build_Debug` only.
- Single-solve timing on a mock = run `generate_rank_bindings` (no `--all-solutions`) under `mpirun` with per-rank
  `TT_METAL_MOCK_CLUSTER_DESC_PATH` from the SC## mapping yaml (see HANDOFF.md runner sketch). Count hosts:
  `grep -oaE 'bg-[a-z0-9-]+-r[0-9]+u[0-9]+-[0-9]+' <out>/phase2_mock_mapping.yaml | sort -u | wc -l`.
- SAT is conflict-bounded per solve (single: 5M cap → soft fallback); `--all-solutions`/`search_n` is UNBOUNDED, so
  don't uncapped-enumerate large rings.
