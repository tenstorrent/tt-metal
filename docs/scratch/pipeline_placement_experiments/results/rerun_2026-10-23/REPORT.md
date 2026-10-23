# Rerun of the pipeline placement experiments, 2026-10-23

Every experiment in `run_experiments.sh` was rerun from the PR branch at `bfe2a840c77` on one
64-core machine (load average 3). The raw outputs sit next to this file. Two questions were put to
the results:

1. Should Tier B check anything, and how much does it save over leaving the work to Tier C?
2. Should Tier C search with SAT or DFS, and does that choice cover the carve as well?

Short answers: **Tier B should run the cheap checks and nothing more. Tier C should use DFS with
exact pre-checks, for the carve and for the resolver, and keep SAT for the seating only.** The
numbers behind those answers follow, then what was rerun and how it compares with the recorded
results.

## 1. Tier B

### What a check costs and what it catches

Random pipelines on every MGD in the repo with at most 8 meshes, 3 per fabric, seed 0, 405 cases
(`tier_b_quick_seed0_3per_fabric.txt`, `tier_b_timing.txt`). The recorded large run, 3,340 cases
on every fabric up to 144 meshes, is in `../tier_b_random_seed0_20per_fabric.txt`.

| Check | Rejects | Time per case (Python) |
|---|---|---|
| Simple checks: binding, capacity, connected group | 31.9% of cases (107 capacity, 18 connectivity, 4 capacity class) | p50 0.022 ms, max 0.6 ms, 0.02 s for all 405 |
| Full placement search, on top of the simple checks | a further 12.1% of cases, 17.8% of the simple passes (22.3% in the large run) | p50 0.10 ms, p99 13.8 ms, max 732 ms; 3 cases hit the 1 s cap |
| Degree check, on top of the simple checks | 16.7% of the gap (20 of 120, over 576 simple passes) | microseconds |

What the full search rejects that the simple checks do not: 43 of 49 cases (88%) are pipelines
whose crossings cannot ride the declared mesh-to-mesh connections, and 6 (12%) cannot be laid out
inside one mesh. The large run had the same split (86% / 14%).

Where the full search is slow: the three capped cases are the 3-mesh and 7-mesh subtorus fabrics
with 48 and 112 slices. In the large run all 115 capped cases were fabrics of 16 to 144 meshes,
every one of them a simple-check pass, so on the big machines the full search is exactly the check
that does not finish.

The designed ring of each MGD (one one-slice stage per slice, as blaze pads it), 111 fabrics with
at most 8 meshes (`tier_b_designed.txt`): 92 pass both checks, 5 fail both, 14 fail only the full
search. The 14 include `blitz_decode_single_galaxy`, the `pipeline_prefill_*galaxy` MGDs,
`sc4_mgd` and the nano exabox MGDs, whose declared connections form a line, so the ring's loopback
has no declared connection to ride.

### How much Tier B saves

A Tier B reject costs microseconds and saves one complete topology-mapper run: the SAT seating,
the intra-mesh solve, and in the design the carve and the resolver on each candidate seating until
the enumeration is exhausted. That run was not timed here. The rank-binding generator needs an
MPI launch against a physical system descriptor, and the binary in `build_Debug` predates the
current source (built 2026-08-28), so a timing from it would not describe today's mapper. What the
experiments do bound:

- The simple checks and the degree check are free at any fabric size and together catch about a
  third of random bad descriptors plus a sixth of the remaining gap.
- The full search catches the rest of the gap on small fabrics for under a millisecond, but on
  fabrics of 16 meshes and more it runs for seconds and gets capped. That is the same
  graph-embedding problem the SAT seating already decides, with real links instead of declared
  ones, so the work is not saved, it is done twice.
- 88% of what the full search finds is a connection problem. Tier C sees it on the first seating
  the solver proposes, because the seating itself has to put the crossings on seat pairs with
  links. The remaining 12% is a carve problem that a lattice or colour pre-check (section 2)
  settles in microseconds, also on the first seating.

**Recommendation.** Tier B keeps the simple checks and adds the degree check. No placement search
at descriptor time. The cases that only the full search catches are reported by Tier C on its
first candidate, at the cost of one seating solve instead of a search that does not scale.

## 2. SAT or DFS in Tier C

Tier C in the design has two searches per candidate seating: the carve (stage rectangles onto host
slices inside each mesh) and the resolver (stage to submesh, link and core-slot choice across the
whole pipeline). Both were benchmarked against a CaDiCaL encoding of the same problem.

### Carve, one mesh (`carve_quick.txt`, `carve_repo_quick_5s_cap.txt`)

| Case | Area | DFS | SAT | Pre-check |
|---|---|---|---|---|
| 4x6: 1x1, 4x 2x2, 1x1 | 18/24 | yes, 0.000 s | yes, 0.008 s | |
| 4x8: 1x1, 6x 2x2, 1x1 | 26/32 | yes, 0.001 s | yes, 0.005 s | |
| 8x8: 1x1, 14x 2x2, 1x1 | 58/64 | yes, 0.002 s | yes, 0.027 s | |
| 8x8: 1x1, 15x 2x2, 3x 1x1 | 64/64 | yes, 0.173 s | yes, 0.381 s | |
| 5x5: 1x1, 5x 2x2 (only 4 fit) | 21/25 | no, 0.008 s | no, 0.009 s | lattice, 28 us |
| 7x7: 1x1, 10x 2x2 (only 9 fit) | 41/49 | no, 2.2 s | no, 1.9 s | lattice, 32 us |
| 6x6: 1x1, 9x 2x2 | 37/36 | no, 0.001 s | no, 0.168 s | area, 31 us |
| 5x5 ring of 25 1x1 | 25/25 | no, 0.64 s | no, 0.015 s | colour parity |
| random 8x8 fills, 5 cases, 5 s cap | 63-64/64 | 2 capped | 3 capped | |

SAT is 3x to 10x slower whenever a carve exists, and no faster when none does except on the odd
grid, where a parity argument answers in constant time. The tight random fills are hard for both.

### Resolver (`resolver_feas_cables.txt`, `resolver_odd_big_60s_cap.txt`, `resolver_sat_quick_opt_cadical.txt`)

The resolver column is the search copied out of `tt_metal/fabric/pipeline_builder.cpp` at build
time. SAT times are with an optimized CaDiCaL (see the note below).

| Case | Layout exists | Resolver DFS | CaDiCaL |
|---|---|---|---|
| one 4x8 grid, 32 stages | yes | 0.7 ms | 4.5 ms |
| one 8x8 grid, 64 stages | yes | 4.3 ms | 35 ms |
| two 4x4 meshes, cables at (0,0),(0,1) | yes | 3.4 ms | not encoded |
| odd 5x5 grid, 25 stages | no | 2.4 s | 1.9 ms |
| odd 5x7 grid, 35 stages | no | over 60 s | 5.3 ms |
| two 4x4 meshes, both cables on same-colour slices | no | 3.3 s | 2.8 s |
| two 4x8 meshes, both cables on same-colour slices | no | over 60 s | over 400 s |

Same picture: DFS is 6x to 8x faster when a layout exists. SAT wins only on the odd grids, which
fail a one-line colouring argument. On the realistic failure, two cables that land on the wrong
slices (Case B in the design doc), SAT is as slow as DFS at two 4x4 meshes and slower at two 4x8.

### Does the choice include the carve

Yes. The carve and the resolver are the two searches Tier C runs per seating, and the comparison
comes out the same for both: DFS plus exact pre-checks. The one SAT in the loop stays the seating
solve, which already exists and is a different problem (which footprint each mesh takes).

**Recommendation.** DFS with a memo for the carve and the existing DFS for the resolver, each
behind the cheap exact pre-checks: area, lattice count per rectangle shape, colour parity for a
full grid and for a path between fixed cable slots. The pre-checks turn every slow "no" above into
microseconds. The open hard case for both methods is a tight carve that uses every slice and is
infeasible for a reason the pre-checks do not see; none of the repo MGD rings is one.

A second reason to avoid SAT here: it is 10x slower again when the CaDiCaL archive comes from a
Debug build, which is what `build_Debug` provides. The DFS has no such dependency.

## 3. Blaze planner and tt-blaze #5775

`carve_probe.py` against the PR branch of tt-blaze
[#5775](https://github.com/tenstorrent/tt-blaze/pull/5775) (`964be336f0`,
`planner_blaze_pr5775_964be336f0.txt`): the planner finds 2 of 5 rings and rejects 3 that exist
(`no-cycle-of-length-13`, `no-alignment` twice), as recorded. The PR's test file run with `ttnn`
stubbed, since this machine has no device build (`blaze_pr5775_test_carve_planner_rings.txt`):

```
7 passed, 3 xfailed
```

The three xfails are the three rings the planner misses. The C++ carve port in the design should
search stage rectangles, as `carve_bench.py` does, rather than slice cycles, or it inherits the
same rejects.

## 4. What was rerun and how it compares

| Experiment | Setting | Matches the recorded run |
|---|---|---|
| `tier_b` | quick: 3 per fabric, up to 8 meshes, 1 s cap | yes, same shape (gap 17.8% vs 22.3% of simple passes; the large run includes the big fabrics) |
| `tier_b_designed` | up to 8 meshes | yes (92 / 14 / 5) |
| `tier_b_timing` | as `tier_b` | new, not in the recorded set |
| `carve` | quick | yes, within 10% |
| `carve_repo` | quick, 5 s cap | yes, same capped cases |
| `planner` | tt-blaze #5775 | yes |
| `resolver` | feas, cables, odd, big; 60 s cap | yes; 5x7 and two 4x8 capped |
| `resolver_sat` | quick with optimized CaDiCaL | yes, within 10%; with the Debug archive every row is about 10x slower (`resolver_sat_quick_debug_cadical.txt`) |

`resolver_sat` needed two fixes to run: the CPM source cache for CaDiCaL was gone, and the
`build_Debug` archive is unoptimized. `run_experiments.sh` now takes `CADICAL_SRC` and
`CADICAL_LIB` and warns on a Debug build. The optimized archive was built from CaDiCaL
`rel-2.2.1` with `-O3 -DNBUILD`, the sources the tt-metal build uses.
