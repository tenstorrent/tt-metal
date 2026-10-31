# Blaze carve planner vs the footprint SAT solver

Blaze's ring carve planner (`plan_ring_carve` in tt-blaze `pipeline_builder/carve_planner.py`)
against the carve as a footprint mapping through tt-metal's topology solver (SAT, with resource
completeness), each followed by the resolver. The numbers are from `results/rerun_2026-10-23/REPORT.md`
section 5.3, with every row in `results/rerun_2026-10-27/carve_comparison.md`: 132 carve-requiring
cases, one 64-core machine, 60 s cap per case. The canvas `../carve-planner-vs-footprint-solver.canvas.tsx`
shows the same tables.

## Finding

Once a solve involves more than 32 host slices (any 8x8 or larger mesh, or more than one 4x8 galaxy),
the carve.py planner runs out of its 200k-expansion search budget on every case and places none of the
132 carve-requiring cases. The SAT-based footprint solver decided 127 of the 132 within 60 s, most in
under a second, the largest (8 galaxies, 128 stages over 256 slices) in 2.6 to 21 s. Where both finish,
on small rings of 32 slices or fewer, the SAT solver is about 9x to 18x faster. With more galaxies the
footprint solver itself slows down, and forks were tested on one small case only.

## By group

| Group | Cases | Planner + resolver: placed | Footprint + resolver: decided | Resolver after footprint, worst |
|---|---|---|---|---|
| original carve cases | 17 | 0 (17 budget exhausted) | 16 | 0.029 s |
| heterogeneous sweep, 60 to 95% fill | 30 | 0 (30 budget exhausted) | 30 | 0.001 s |
| single mesh, no 1x1 stages | 45 | 0 (stage-0 rule; 2 with it off) | 43 | 0.029 s |
| heterogeneous multi-mesh | 12 | 0 (stage-0 rule; 0 with it off) | 12 | 0.001 s |
| multi-mesh, no 1x1 stages | 28 | 0 (stage-0 rule; 0 with it off) | 26 | 0.036 s |
| **all** | **132** | **0** | **127** | **0.036 s** |

## Where both finish

Seconds as carve + resolver. Every case the planner completed has 32 host slices or fewer.

| Case | Slices | Planner + resolver | Footprint + resolver | Speedup |
|---|---|---|---|---|
| 4x8: 1x1, 2x4, 2x4, 2x2, 1x1 | 32 | 0.009 + 0.000 | 0.001 + 0.000 | ~9x |
| 4x8 ring of 32 x 1x1 | 32 | 0.038 + 0.001 | 0.004 + 0.001 | ~8x |
| 4x8 ring of 16 x 1x2 (planner's stage-0 rule off) | 32 | 0.036 + 0.000 | 0.002 + 0.000 | ~18x |
| 4x4: 1x1, 3x 2x2, 1x1 | 16 | 0.000 + 0.000 | 0.000 + 0.000 | both instant |

## Where the planner fails

| Case | Fabric | Planner | Footprint + resolver |
|---|---|---|---|
| 4x4: 1x1, 3x 2x2 (feasible) | 1 mesh | wrong reject: `no-cycle-of-length-13` | 0.000 + 0.000 |
| 4x6 and 4x8: 1x1, ring of 2x2, 1x1 (feasible) | 1 mesh | wrong reject: `no-alignment` | 0.001 + 0.000 |
| 8x8 fill: 1x1, 15x 2x2, 3x 1x1 | 1 mesh | budget exhausted | 0.026 + 0.000 |
| 16x16: 1x1, 62x 2x2, 1x1 | 1 mesh | budget exhausted | 0.13 + 0.004 |
| random 8x8, 30 stages | 1 mesh | budget exhausted | 0.60 + 0.001 |
| random 8x16, 37 stages | 1 mesh | budget exhausted | 5.6 + 0.001 |
| ring of 112 x 1x1, 1 cable per pair | 4 galaxies | budget exhausted | 9.9 + 0.020 |
| domino ring, 128 x 1x2, 4 cables per pair (2 seeds) | 8 galaxies | budget exhausted | 2.6 / 21 + 0.036 |
| mixed shapes, 56 / 62 stages, 4 cables per pair | 8 galaxies | budget exhausted | 36 / 24 + 0.004 |
| 7x7: 1x1, 10x 2x2 (infeasible) | 1 mesh | budget exhausted, no verdict | proved no, 2.1 |
| 5x5 ring of 25 x 1x1 (infeasible) | 1 mesh | budget exhausted, no verdict | proved no, 0.003 |

The planner does not run slowly on the large cases; it gives up. It stops at its budget in under
0.06 s. Grid-shaped meshes have no degree-2 chains for it to contract, so its cycle search becomes a
raw Hamiltonian-cycle search; its budget was sized for SC16, a ring of columns (about 900
expansions). Two rules also reject cases before the search: stage 0 must be one slice (85 of the 132
cases have no 1x1 stage), and first-fit grouping reports `no-alignment` on 16 feasible cases.

## The footprint solver as galaxies are added

The 40 carve-requiring multi-mesh rings (heterogeneous and no-1x1 groups), 4x8 galaxies, footprint +
completeness solve time:

| Galaxies | Stages | Decided | Median feasible | Slowest feasible | Infeasibility proofs |
|---|---|---|---|---|---|
| 2 | 12 to 32 | 18 of 18 | 0.023 s | 1.1 s | 0.10 to 8.1 s (3 cases) |
| 4 | 26 to 64 | 16 of 18 (2 open at the cap) | 0.78 s | 11 s | 51 s (1 case) |
| 8 | 56 to 128 | 4 of 4 | 23 s | 36 s | none in this set |

On 8 galaxies with 1 to 2 cables per pair, most single-slice rings run to the time cap for every
engine. Forks were tested on one designed 6x6 case only (footprint 0.008 s; the planner port handles
rings only).

## Method

- The planner column is `topo/carve_planner_port.hpp`, a C++ port of blaze's `plan_ring_carve` with
  blaze's 200k-expansion and 64-cycle budgets. It reproduces blaze's Python verdicts on the designed
  cases (`no-cycle-13`, both `no-alignment` rings, the mixed4 cablings); blaze's Python code itself was
  run only on those designed rings (`carve/carve_probe.py`).
- The footprint column is `solve_topology_mapping` under Auto (SAT on every case) with
  `topo/resource_completeness.patch` (not committed), conflict budget removed so a "no" is a proof.
- Every returned solution was re-checked independently (`BAD` never printed).
