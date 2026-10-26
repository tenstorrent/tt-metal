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

### Carve, other SAT encodings (`carve_encodings_quick_60s_cap.txt`, `carve_encodings_open_600s_cap.txt`)

Follow-up question: does a slice-level encoding, where every stage is broken down to its slices
and grouping rules tie them to a rectangle, beat the position encoding above? `carve_bench_encodings.py`
runs four encodings against the DFS on larger and tighter cases, 60 s cap, solve time only (the
Python clause construction is excluded, since a C++ encoder would not pay it):

- `pos`: the position encoding above (one variable per stage per rectangle position).
- `pos+sym`: `pos` with stage 0 pinned to one quadrant of the grid (and below the diagonal when
  the grid and every stage are square).
- `cell`: slice-level. `y[k,b]` = slice b belongs to stage k, channelled to an anchor variable per
  rectangle position; at most one stage per slice over n literals; touching via "some slice of k
  has a neighbour in k+1".
- `cell+sym`: `cell` with the same pinning.

This run was on a busier 32-core box, so its absolute times are 10x to 20x above the table above;
only the columns within a row compare.

| Case | Slices | DFS | pos | pos+sym | cell | cell+sym |
|---|---|---|---|---|---|---|
| 8x8: 1x1, 14x 2x2, 1x1 | 58/64 | **0.10** | 0.39 | 0.19 | 1.19 | 0.91 |
| 8x8: 1x1, 15x 2x2, 3x 1x1 | 64/64 | **4.0** | 5.1 | 6.8 | 31.0 | 21.4 |
| 8x8 ring of 64 1x1 | 64/64 | **0.09** | 0.50 | 2.09 | 0.51 | 1.61 |
| 8x16: 1x1, 30x 2x2, 1x1 | 122/128 | **0.12** | cap | 1.61 | 0.42 | 2.08 |
| 8x16: 2x4 x14, 2x2 x3, 1x1 x3 | 127/128 | cap | cap | cap | **31.3** | cap |
| 8x16 ring of 128 1x1 | 128/128 | **0.10** | 0.48 | 0.62 | 1.00 | 1.19 |
| 16x16: 1x1, 62x 2x2, 1x1 | 250/256 | cap | **3.4** | 7.5 | cap | cap |
| random 8x8, 30 stages | 64/64 | cap | cap | cap | cap | cap |
| random 8x8, 29 stages | 64/64 | **47.4** | cap | cap | cap | cap |
| random 8x8, 31 stages | 63/64 | **27.5** | 40.1 | 28.4 | cap | cap |
| random 8x8, 34 stages | 64/64 | cap | cap | cap | cap | cap |
| random 8x8, 37 stages | 64/64 | **1.6** | 9.6 | 44.6 | cap | cap |
| random 8x16, 37 stages | 121/128 | cap | cap | cap | cap | cap |
| random 8x16, 32 stages | 127/128 | cap | cap | cap | cap | cap |
| random 8x16, 32 stages (second) | 122/128 | **1.5** | cap | cap | cap | cap |
| 7x7: 1x1, 10x 2x2, infeasible | 41/49 | 39.6 | 21.8 | 7.5 | 5.6 | **5.0** |
| 5x5 ring of 25 1x1, infeasible | 25/25 | 9.8 | **0.10** | 0.12 | cap | cap |

Of the 11 cases some solver decided feasible, the DFS was fastest in 9 and capped in 2 (the
16x16 grid of 2x2 stages, and the 8x16 fill of 2x4 stages). The two SAT wins came from different
encodings, so no single encoding covers the DFS's losses. The slice-level encoding is 3x to 8x
slower than positions on every 8x8 case, caps on every random 8x8 fill that positions solves, and
caps on the odd-grid parity case that positions proves in 0.1 s. Symmetry breaking changes the
solver's path but not predictably (rescues one `pos` timeout, makes another case 5x slower).

Where the new encodings do help is proving infeasibility: on the 7x7 case the DFS needs 40 s,
positions 22 s, the slice-level encodings 5 s, because a proof has to exhaust the space and
slice-level propagation plus symmetry breaking shrink it. The lattice pre-check still answers that
case in microseconds, so it is not a reason to carry SAT.

Four cases, the tight random fills at 64/64 and 121-127/128, were open for every solver at 60 s;
`carve_encodings_open_600s_cap.txt` reruns them with a 600 s cap: the two 8x8 fills are feasible
(DFS, 441 s and 542 s; positions capped), the 8x16 fill of 37 stages is feasible (positions 183 s,
DFS 530 s), and the 8x16 fill of 32 stages at 127/128 stayed open for both.

### Carve, a graph-to-graph encoding (`carve_encodings_sub_quick_60s_cap.txt`)

`sub` is the carve as a mapping between two graphs: one target node per slice of every stage,
with the stage's own grid edges as two-literal clauses that pin the rectangle (sub-node (i,j) on
slice (r,c) if and only if (i,j+1) is on (r,c+1)), one global node per host slice with the fabric
adjacency, `x[v,b]` = sub-node v sits on slice b, exactly one slice per sub-node, at most one
sub-node per slice, and for each pipeline edge an exit-port choice per stage whose slice must
neighbour a slice of the next stage. Single-slice stages reduce to "v on b implies u on a
neighbour of b", the plain subgraph embedding. One-way rigidity clauses were 12x slower than
two-way (33 s vs 2.8 s on the first row) and are not reported.

| Case | Slices | dfs2 | pos | sub | sub+sym |
|---|---|---|---|---|---|
| 8x8: 1x1, 14x 2x2, 1x1 | 58/64 | **0.004** | 0.30 | 3.1 | 13.0 |
| 8x8: 1x1, 15x 2x2, 3x 1x1 | 64/64 | **9.4** | 12.4 | 38.9 | cap |
| 8x8 ring of 64 1x1 | 64/64 | **0.10** | 0.79 | 0.51 | 2.5 |
| 8x16: 1x1, 30x 2x2, 1x1 | 122/128 | **0.10** | cap | cap | cap |
| 8x16: 2x4 x14, 2x2 x3, 1x1 x3 | 127/128 | cap | cap | cap | cap |
| 8x16 ring of 128 1x1 | 128/128 | **0.50** | 2.0 | 1.9 | 1.7 |
| 16x16: 1x1, 62x 2x2, 1x1 | 250/256 | cap | **11.8** | cap | cap |
| random 8x8, 30 stages | 64/64 | cap | cap | cap | cap |
| random 8x8, 29 stages | 64/64 | **0.6** | cap | cap | cap |
| random 8x8, 31 stages | 63/64 | **3.1** | cap | cap | cap |
| random 8x8, 34 stages | 64/64 | **4.5** | cap | 25.3 | cap |
| random 8x8, 37 stages | 64/64 | 43.5 | **16.6** | cap | cap |

The mapping encoding has 4 variables per stage per place where positions has 1, and the
exactly-one constraints over them are redundant with each other; CaDiCaL recovers some of that
through equivalence reasoning but not for free. It is behind positions on every row except the
single-slice rings, where the two are the same formulation, and the 34-stage fill. The variable
count is not what decides SAT time here; the ring closure is.

### Carve, an improved DFS (`carve_encodings_dfs2_quick_60s_cap.txt`, `carve_encodings_dfs2_open_600s_cap.txt`)

`carve_dfs2.py` adds to the DFS what the SAT encodings got for free: stage 0 pinned to one
quadrant (the same symmetry breaking), the lattice count re-checked on the remaining stages after
every placement, stages placed from both ends of the ring (0, 1, n-1, 2, n-2, ...) so a ring that
cannot close fails early, and candidate positions tried compact-first (fewest free slices around
the new rectangle). The two-ended reachability prune and the memo are kept.

| Case | Slices | DFS | dfs2 | pos |
|---|---|---|---|---|
| 8x8: 1x1, 14x 2x2, 1x1 | 58/64 | 0.20 | **0.10** | 0.50 |
| 8x8: 1x1, 15x 2x2, 3x 1x1 | 64/64 | **4.0** | 6.7 | 9.9 |
| 8x8 ring of 64 1x1 | 64/64 | 0.10 | 0.10 | 0.50 |
| 8x16: 1x1, 30x 2x2, 1x1 | 122/128 | 0.40 | **0.30** | cap |
| 8x16: 2x4 x14, 2x2 x3, 1x1 x3 | 127/128 | cap | cap | cap |
| 8x16 ring of 128 1x1 | 128/128 | 0.31 | **0.10** | 1.7 |
| 16x16: 1x1, 62x 2x2, 1x1 | 250/256 | cap | cap | **22.4** |
| random 8x8, 30 stages | 64/64 | cap | cap (600 s) | cap |
| random 8x8, 29 stages | 64/64 | cap | **0.7** | cap |
| random 8x8, 31 stages | 63/64 | 53.0 | **4.0** | cap |
| random 8x8, 34 stages | 64/64 | cap | **5.2** | cap |
| random 8x8, 37 stages | 64/64 | **1.5** | 44.6 | 15.8 |
| random 8x16, 37 stages | 121/128 | cap | cap | cap |
| random 8x16, 32 stages | 127/128 | cap | cap | cap |
| random 8x16, 32 stages (second) | 122/128 | **3.1** | cap (211 s at 600 s) | cap |
| 7x7: 1x1, 10x 2x2, infeasible | 41/49 | cap | **0.09** | 52.6 |
| 5x5 ring of 25 1x1, infeasible | 25/25 | 16.7 | 7.0 | **0.4** |

Three things follow. The dynamic lattice check turns every counting-infeasible case into one
search node. Two of the tight 8x8 fills that cost the original DFS 441 s and 542 s take dfs2
under 6 s, and it never loses to positions on an 8x8 grid. But the position ordering cuts both
ways: on two fills (37 stages on 8x8, the second 8x16) the original row-major order was the lucky
one and dfs2 capped. Neither ordering dominates, so the robust DFS is the two orderings racing
under one budget. Neither variant fixes the two big-grid losses (16x16 of 2x2 stages, the 2x4
fill of 8x16), where positions finishes in seconds; those need the SAT fallback or a
shape-aware ordering, and no repo MGD has a slice grid that large.

### Resolver (`resolver_feas_cables.txt`, `resolver_odd_big_60s_cap.txt`, `resolver_sat_quick_opt_cadical.txt`)

The resolver column is the search copied out of `tt_metal/fabric/pipeline_builder.cpp` at build
time: submeshes of 8 chips, 4 or 2 links per neighbouring pair, one cable chip pair per crossing,
core-slot reservation. The CaDiCaL column (`sat.cpp`) encodes a relaxation of that problem, stage
to submesh with consecutive stages on adjacent submeshes, and no link, chip or core-slot choice.
A SAT "infeasible" is therefore a true infeasibility, a SAT "feasible" is weaker than the
resolver's, and the SAT times are a lower bound on a full encoding. SAT times are with an
optimized CaDiCaL (see the note below).

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
microseconds. On the repo's slice grids (4x8, 8x4, 2x16, 8x8) the DFS was fastest on every case
it decided, including with the alternative encodings of the follow-up table. The DFS loses only
on 8x16 and 16x16 grids packed with many identical wide stages, where it caps and the position
encoding finishes in seconds; if those grids become real, the cheap fix is a portfolio (DFS with
a short budget, then the position encoding), not a different encoding. The open hard case for
every method is a tight carve that uses every slice and is infeasible for a reason the pre-checks
do not see; none of the repo MGD rings is one.

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

## 3b. The same instances through the real C++ code (`topo/`)

Everything in section 2 is Python, with CaDiCaL driven through python-sat; none of it is the
repo's code. To compare the two C++ engines that exist, `topo/resolver_same_cases.cpp` runs the
resolver search copied out of `pipeline_builder.cpp` and `topo/topo_solver_bench.cpp` runs
`solve_topology_mapping` with its SAT and DFS engines, on identical instances: one host slice per
node (a 4x2-chip submesh for the resolver, with 4 links per horizontal neighbour pair, 2 per
vertical, 1 per cable), one node per slice of every stage, grid adjacency plus cables.

Multi-slice stages are not in this comparison. A node-to-node mapping engine has to draw the
pipeline edge k -> k+1 between two fixed sub-nodes, and with any fixed choice (bottom-right of k
to top-left of k+1) every step moves right or down, so a ring can never close: the first attempt
had the resolver cap on "8x8: 1x1, 14x 2x2, 1x1" for that reason, not a carve reason. "Some
slice of k touches some slice of k+1" has no node-to-node encoding in either engine, so the
exact comparison is rings of single-slice stages, on one mesh and across cables, which is also
what both engines exist for. The multi-slice carve stays a separate step (the design's C1).

Same loaded 32-core box as the tables above; 60 s cap per solve.

`topo_solver_bench` could not be linked against `libtt_metal`: every build tree predates the
solver's last change (2026-10-15), the CPM cache for fmt/spdlog/tt-logger/enchantum/cadical is
empty, and a reconfigure hung twice in cmake's `git status` over NFS. `topo/make_topo_bench.sh`
instead compiles the solver's three source files straight into the benchmark (the solver itself
is header-only in `topology_solver.tpp`) with the header-only dependencies fetched at their pinned
versions and the optimized CaDiCaL archive from `build_RelWithDebInfo`; the parts of
`topology_solver.cpp` that touch `FabricNodeId`, `MeshGraph` and the PSD are never called and are
dropped by `--gc-sections`. `STRICT` validation, `quiet_mode`, default `MappingConstraints`.

| Case | Resolver (`pipeline_builder.cpp`) | topology solver, SAT | topology solver, DFS |
|---|---|---|---|
| 4x8 ring of 32 | 0.002 | 0.009 | **0.001** |
| 8x8 ring of 64 | 0.20 | 0.21 | **0.09** |
| 8x16 ring of 128 | 1.1 | 1.6 | **0.21** |
| 16x16 ring of 256 | 8.8 | 3.4 | **0.90** |
| 8x8 ring of 60, 4 spare | 0.10 | 0.50 | **0.003** |
| 5x5 ring, infeasible (odd) | cap | **0.007** | 29.8 |
| 7x7 ring, infeasible (odd) | cap | **0.29** | cap |
| two 4x4 meshes, cables (0,0),(0,1), ring of 32 | 0.011 | 0.19 | **0.005** |
| two 4x4 meshes, same-colour cables, infeasible | cap | **38.6** | cap |
| two 8x8 meshes, cables (0,0),(0,1), ring of 128 | cap | **1.3** | cap |
| two 8x8 meshes, same-colour cables, infeasible | cap | cap | cap |
| two 4x8 meshes, same-colour cables, infeasible | cap | cap | cap |

Reading. On single-mesh feasible rings all three finish and the two DFS engines lead, the
topology solver's DFS by 10x over the resolver on the 16x16 ring (the resolver also chooses a
link and core slots per edge). On everything that needs a proof or a crossing decision only the
SAT engine finishes: both odd grids in well under a second, the feasible two-mesh ring of 128 in
1.3 s where both DFS searches cap, the infeasible two-4x4 case in 38.6 s. The two larger
same-colour cases defeat SAT too within 60 s, so the colour pre-check from the design stays
necessary in front of whichever engine runs; SAT does not make it optional.

### Galaxy-shaped multi-mesh rings (`topo_resolver_multi2_60s_cap.txt`, `topo_solver_multi2_60s_cap.txt`)

`topo/multi2_cases.hpp`: 2, 4 and 8 meshes of 4x8 slices, in a ring (mesh i cabled to i+1 and
the last to the first) or a line, 1 or 4 east-west cables per adjacent pair with rows chosen by a
seeded generator, so each seed is one seating; a ring or chain of single-slice stages over every
slice, plus two rings that leave 8 of 32 slices per mesh idle. The `multi2` mode of both
harnesses; `TOPO_SKIP=n` resumes after n cases.

| Case | Resolver | topo SAT | topo DFS |
|---|---|---|---|
| 2 galaxies ring, 1 cable/pair, seed 0 | cap | cap | cap |
| 2 galaxies ring, 1 cable/pair, seed 1 | 9.8 | **2.3** | cap |
| 2 galaxies ring, 4 cables/pair, seed 0 | 1.1 | 0.50 | **0.40** |
| 2 galaxies ring, 4 cables/pair, seed 1 | **0.20** | 0.40 | 2.5 |
| 4 galaxies ring, 1 cable/pair, seed 0 | cap | cap | cap |
| 4 galaxies ring, 1 cable/pair, seed 1 | cap | cap | cap |
| 4 galaxies ring, 4 cables/pair, seed 0 | cap | **3.0** | cap |
| 4 galaxies ring, 4 cables/pair, seed 1 | cap | **22.0** | cap |
| 8 meshes ring, 1 cable/pair, seeds 0 and 1 | cap | cap | cap |
| 8 meshes ring, 4 cables/pair, seeds 0 and 1 | cap | cap | cap |
| 2 meshes line, 1 cable/pair | **0.20** | 3.1 | cap |
| 2 meshes line, 4 cables/pair | **0.10** | 0.30 | 5.4 |
| 4 meshes line, 1 cable/pair | cap | cap | cap |
| 4 meshes line, 4 cables/pair | cap | **7.4** | cap |
| 2 meshes ring, 24 of 32 per mesh, 2 cables/pair | 0.10 | 0.30 | **0.001** |
| 4 meshes ring, 24 of 32 per mesh, 2 cables/pair | cap | **0.80** | cap |

("galaxies" in the raw files and the first draft of this table means these 32-rank meshes of
4x8 slices, 256 chips in the resolver harness, not one Galaxy.)

With one cable per pair the crossing rows are forced and the case is decided by the colour
condition on each mesh; the capped single-cable rows are most likely infeasible seatings that
none of the three engines can prove in 60 s. With four cables per pair the solver has a choice:
the resolver finishes only on two meshes, the SAT engine finishes at four meshes (3 s, 22 s,
7.4 s for the chain, 0.8 s with spare slices), and at eight meshes, rings of 256 stages, all
three cap within the budget. Spare slices make the DFS engines instant on two meshes and do not
save them on four.

### Stages bigger than one slice: pre-carved slots (`topo_resolver_slots_60s_cap.txt`, `topo_solver_slots_60s_cap.txt`)

`topo/slot_cases.hpp`: the mesh is carved first into slots of 2x2 or 2x4 slices, uniform or a
mixed tiling (a 4x8 mesh as two 2x4 slots over four 2x2 slots), and every stage is ONE node
whose shape must match its slot's. For the resolver a slot is one submesh of 32 or 64 chips with
links wherever slices are adjacent across slots; for the topology solver a slot is one global
node and a required trait constraint matches stage shape to slot shape. This is the level the
design's Tier C hands to the resolver after C1, and the only level at which either engine solves
multi-slice stages exactly.

| Case | Stages / slots | Resolver | topo SAT | topo DFS |
|---|---|---|---|---|
| 8x8 mesh tiled 2x2: ring of 16 x 2x2 | 16 / 16 | 0.000 | 0.19 | 0.001 |
| 8x8 tiled 2x2: ring of 15 x 2x2, 1 spare (infeasible: odd cycle) | 15 / 16 | 1.1 | 0.09 | 0.30 |
| 8x8 mesh tiled 2x4: ring of 8 x 2x4 | 8 / 8 | 0.000 | 0.001 | 0.000 |
| 16x16 mesh tiled 2x2: ring of 64 x 2x2 | 64 / 64 | 0.005 | 0.10 | 0.002 |
| 4x8 mixed: ring 2x4, 2x2, 2x2, 2x2, 2x2, 2x4 | 6 / 6 | 0.000 | 0.09 | 0.000 |
| 4x8 mixed: ring 2x4, 2x2, 2x2, 2x4, 2x2, 2x2 (infeasible) | 6 / 6 | 0.08 | 0.001 | 0.000 |
| 2 meshes tiled 2x2, 4 cables/pair: ring of 16 x 2x2 | 16 / 16 | 0.001 | 0.004 | 0.000 |
| 4 meshes tiled 2x2, 4 cables/pair: ring of 32 x 2x2 | 32 / 32 | 0.001 | 0.009 | 0.000 |
| 2 meshes mixed, 4 cables/pair: ring of 12 heterogeneous | 12 / 12 | 0.000 | 0.001 | 0.000 |
| 4 meshes tiled 2x2, 4 cables/pair: chain of 32 x 2x2 | 32 / 32 | 0.001 | 0.08 | 0.001 |

Every row is decided by every engine in about a second or less, the heterogeneous rings and the
two infeasible ones included. Once the carve has happened, stage-to-slot is a small problem: 16
to 64 nodes instead of 64 to 256, and the slot graph has far fewer symmetric alternatives. The
15-of-16 case is the odd-cycle argument again (a 4x4 slot grid minus one slot is bipartite 8+7),
and all three prove it quickly at this size. This is the strongest argument for the design's
order of operations: carve in C1, then resolve, rather than asking any engine to do both at slice
granularity.

### Integration: resolver or topology solver

| | Resolver (`detail::resolve_graph_layout_with_connections`) | Topology solver (`solve_topology_mapping`) |
|---|---|---|
| Exists today for | stage -> submesh, link per edge, core slots, entry/exit chips, host I/O | node -> node embedding with edge support, pinnings, trait constraints, enumeration with blocking clauses, soft edges |
| Granularity it needs | pre-carved submeshes (one node per stage, chip count must match) | any graph; one node per slice works |
| Carve of multi-slice stages | not modelled (needs C1 first) | not expressible (fixed-port problem above) |
| Capacity, core slots, chip-level links | yes | no |
| Multi-mesh with cables | yes, via `DirectLinks` | yes, as global-graph edges |
| Infeasibility proofs | exponential (needs the colour pre-check) | SAT engine learns; see table |
| Failure reporting | one message string (design 3.1 asks for a structured reason) | `failure_stage`, warnings, stats |
| Shared with runtime | yes, Tier D calls the same function | no |
| Integration into Tier C | call with `DirectLinks` from the candidate seating; already planned (design 4.2) | would need a new target/global graph builder plus a translation of its node mapping back to stages and chips, then the resolver anyway for links and slots |

Both engines need the carve done first. The question "resolver or topology solver" is therefore
only about the stage-to-slot step. At that level (the slots table) every engine is fast, so the
resolver, which already models links, core slots and host I/O and is shared with the runtime, is
the one to call, as the design says. The case for the topology solver's SAT engine is the
slice-granularity problem: full rings of single-rank stages across several meshes, where it is
the only engine that finishes (two 8x8 meshes 1.3 s, four 32-rank meshes 3 to 22 s) and the
only one that proves the odd-grid cases. If Tier C ever has to decide such a ring without a
carve (one-slice stages tiling every rank, the blitz decode shape, on a seating with several
candidate cables), run the SAT engine on the slice graph first and hand its crossing choice to
the resolver; and keep the colour pre-check in front of both, since none of the three proves the
same-colour cable cases on 8x8 or 4x8 meshes within 60 s.

### Plugging the resolver into C2 as it is

The benchmark harness is also the answer to "how much of the resolver has to change for C2".
It changed nothing. `topo/resolver_same_cases.cpp` calls `detail::resolve_pipeline_placement`
with the signature the public header already declares
(`tt-metalium/experimental/fabric/pipeline_builder.hpp`, namespace `detail`), on a copy of the
search taken out of `pipeline_builder.cpp` at build time. Every number in the three tables above
is the unmodified search: `PlacementSearch` (stage-to-submesh backtracking with the chip-count,
core-capacity, neighbour-count, connectivity and lookahead checks), `LinkSearch` (one link per
edge, core slots per chip, the failed-state memo, the shared-chip fallback) and the host
endpoint placement.

What the harness bypassed is the public wrapper, `detail::resolve_graph_layout_with_connections`.
That wrapper validates the inputs, runs Kahn's sort on the non-loopback edges, converts
`ChipTuple` to the internal chip type, and then calls the same search. Its only control-plane
use is `discover_submesh_links`, and that runs only when `direct_links` is null. C2 passes its
own links, so the call never reaches the control plane. The harness skipped the wrapper only
because the stub build had no `FabricNodeId`; in the tree, C2 should call the wrapper, not the
search, and get the validation and the sort for free.

| Harness | Real call from C2 | Same? |
|---|---|---|
| `resolve_pipeline_placement(order, edges, links, counts, chips, {})` | `resolve_graph_layout_with_connections(nodes, edges, submesh_chips, node_chip_counts, node_pipeline_core_counts, nullopt, &links)` | same search; the wrapper adds validation and the topological sort |
| stage order given by the caller | Kahn's sort over the non-loopback edges | equivalent for a DAG plus loopbacks |
| chips as `InternalChip{FabricNodeId, row, col}` | chips as `ChipTuple(mesh_id, chip_id, row, col)`, converted by the wrapper | same |
| `DirectLinks` built by hand, both directions per physical link | `DirectLinks` built by C1 from the candidate cables and the routed intra-mesh links | same type; C1 must insert both directions (see below) |
| core-count overrides empty: 1 slot per chip on submeshes of 8 or more chips, 2 on smaller | blaze's `_pipeline_core_defaults` per node, or empty for the same defaults | an input, not a resolver change |
| cap by `fork` + `alarm` outside the search | none exists inside the search | the one thing to add if open question 13 keeps a cap |

**Both library and header are already in place.** `pipeline_builder.cpp` is in
`tt_metal/fabric/sources.cmake`, the same library as `topology_mapper_utils.cpp` where
`MultiMeshSolutionEnumerator::next()` lives, and the header includes only `fabric_types.hpp`
and `mesh_graph.hpp`. The P8 hook can include the header and call the wrapper directly.

**The input contract C1 has to meet.** These are the things the harness had to get right for
the search to accept anything, so they are what C1's slot builder has to produce:

- *Slot-local coordinates.* Each slot's chips carry `(row, col)` relative to the slot's own
  origin, and every `LinkPair` uses those coordinates. The slot harness recomputes them from the
  slice's position inside the slot (`local()` in `timed_slots`), and so must C1 when a slot
  spans several host slices.
- *Links are directed per key.* `place_stages` looks up `{src_slot, dst_slot}` for the edge's
  direction, and `search_links` reads `submesh_links.at({src, dst})`. A physical cable that can
  carry traffic both ways has to appear under both keys, each with its own exit and entry
  chips. `discover_submesh_links` does this because it iterates ordered pairs; a hand-built
  table must too.
- *Chip counts pin a stage to a slot shape.* `node_chip_counts[stage] = rows x cols x chips per
  slice` makes the shape filter exact: a 2x4-slice stage can only land on a 2x4 slot. That is
  how the mixed tilings in the slots table were decided in milliseconds. Leaving a stage out of
  the map lets it bind to any slot.
- *Stage 0 needs room for host I/O.* Stage 0 asks for its edge links plus 2 slots for H2D and
  D2H, on distinct chips in the first pass and shared chips only as a fallback for other
  stages. A single-chip stage 0 (a 1x1 host slice of 1 chip) has 2 slots and needs 4, so it
  fails on capacity before any search. The existing test skips all-1x1 rings for this reason.
  C1 should never produce a 1-chip slot for stage 0; with 4x2-chip slices (8 chips, 1 slot
  each) stage 0 has 8 slots and needs 4, which is what every row above used.
- *One loopback per cycle, marked.* The closing edge carries `is_loopback = true`; the sort
  ignores it and the link search still places it. Open chains pass no loopback.

**What the result already gives C2 and C3.** `GraphLayoutResult.node_to_submesh` is the stage
to slot map, `resolved_edges` carries each edge's exit and entry chip in slot-local
coordinates, and the H2D and D2H chips are separate fields. `R` for P9 is read straight off
`resolved_edges`: for every edge whose two slots are on different meshes, map the exit and
entry coordinates back through C1's slot table to the two `FabricNodeId`s, and that pair is the
cable the layout crosses on. The demand check is a count per mesh pair over that list. Neither
needs a resolver change.

**What does have to change, and when.**

| Change | Needed by | Where | Size |
|---|---|---|---|
| none | P8 first version: accept or reject a seating, `exclude_mapping` on reject | | 0 lines |
| structured failure (stage, edge, capacity conflict) next to the message string | P15's exit-node retry, which needs to know which crossing failed; P8 only needs the string for `error_message` | `resolve_pipeline_placement` already records the first capacity conflict in `search.failure`; add the failing stage index and edge from `place_stages` and `search_links` | P4's first step; small |
| a deadline the search checks in `place_stages` | only if open question 13 keeps the hard-fail cap; the harness capped from outside with `alarm` | one optional argument, one check per recursion | about 10 lines |
| the colour test in `remaining_stages_can_connect` when a mesh's unused slots equal its remaining stages | the per-rank tiling path at full occupancy on several meshes (the multi2 rows the resolver caps on) | the lookahead already walks components; add the parity test on the open path ends | design 4.2, small |
| a per-pair link budget inside `LinkSearch` | only if the demand check fires in practice | `search_links`, skip links whose pair is at budget | later, design 4.2 |

P4's second step, splitting `discover_submesh_links` behind a link-query callback, is not
needed by C2 at all: C2 never discovers links, it supplies them. It is a Tier D refactor and
can be dropped from P4 or deferred.

**Where plugging it in straight is enough, and where it is not.** The slots table is the
planned carve path and the single-mesh tiling path: slots already pinned to one stage shape,
cables as single links. There the unmodified resolver answers every case, feasible or not, in
at most 1.1 s, including the heterogeneous tilings and the 4-mesh chains. That path can call
the resolver as it is. The multi2 table is the per-rank tiling path at full occupancy: every
rank its own one-slice stage, the pipeline a Hamiltonian cycle through every slot of every
mesh. There the resolver finishes the 2-mesh rings with 4 cables per pair and the 2-mesh chains,
and caps on 4-mesh rings and on 1-cable 2-mesh rings. That path needs the pre-check in front of
the resolver (the colour and segment test from design 4.2), or the topology solver's SAT engine
on the slice graph to choose the crossings first, and the resolver after it for links and
slots. Neither changes what the resolver is; both are callers.

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
