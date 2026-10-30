# Pipeline placement experiments

CPU-only experiments behind the MGD pipeline design (`docs/scratch/mgd-pipeline-graph-design.md`).
They answer four questions:

1. **Tier B.** Is a descriptor-time search for stage placement worth having, or do three simple
   checks catch enough? (`tier_b/`)
2. **Carve.** Inside one mesh, is a DFS or a SAT solver better at laying a ring of stage
   rectangles onto host slices? (`carve/carve_bench*.py`)
3. **Blaze planner.** Does blaze's carve planner find carves that exist? (`carve/carve_probe.py`)
4. **Resolver.** How does the existing resolver search in `pipeline_builder.cpp` behave, and does
   SAT do better? (`resolver/`)

Nothing here touches hardware or tt-metal code. The results below are from one machine; times are
indicative.

## How to run

```bash
cd docs/scratch/pipeline_placement_experiments
./run_experiments.sh setup        # once: venv (protobuf 3.20.3, python-sat), generated proto module, MGD list
./run_experiments.sh quick        # every experiment at its small size, a few minutes in total
./run_experiments.sh tier_b       # or one experiment by name
```

| Name | What it runs | Default size and time |
|---|---|---|
| `tier_b` | `tier_b/run.py`: random pipelines, simple checks vs full search | 3 pipelines per fabric, fabrics of up to 8 meshes, 1 s cap per case; about 2 s |
| `tier_b_designed` | `tier_b/run2.py`: the ring each MGD is built for, plus a degree check | fabrics of up to 8 meshes; under a minute |
| `tier_b_timing` | `tier_b/run3.py`: time of each check, split by outcome | same limits as `tier_b` |
| `carve` | `carve/carve_bench.py`: ring of rectangles on one grid, DFS vs CaDiCaL | about 10 s |
| `carve_repo` | `carve/carve_bench_repo.py`: the same on repo-shaped grids | each solve killed after `CARVE_TIMEOUT` (default 30 s); about 35 s with `CARVE_TIMEOUT=5` |
| `carve_encodings` | `carve/carve_bench_encodings.py`: four SAT encodings (positions, slice-level grouping, each with and without symmetry breaking) vs the DFS on 8x8 to 16x16 grids | 17 cases x 5 solvers, many hit the cap: about 25 min with `CARVE_TIMEOUT=60`; `open` reruns the undecided cases, `sizes` prints clause counts only |
| `planner` | `carve/carve_probe.py`: blaze's `plan_ring_carve` vs brute force | needs `TT_BLAZE=/path/to/tt-blaze`; seconds |
| `resolver` | `resolver/bench.cpp`: the resolver search, groups `feas` and `cables` | about 6 s |
| `resolver_sat` | `resolver/sat.cpp`: a CaDiCaL encoding of the same placement | needs a tt-metal build for CaDiCaL; capped at 60 s |
| `topo` | `topo/`: the same instances through the resolver and through `solve_topology_mapping` (SAT and DFS engines): single-slice rings (`all`), rings and chains over 2 to 8 meshes of 32 ranks (`multi2`), and pre-carved slots with 2x2, 2x4 and mixed stages (`slots`) | the solver half is built by `topo/make_topo_bench.sh` from the solver's own sources plus `libcadical.a` from a tt-metal build (`TOPO_BUILD_DIR`), with the header-only deps from the CPM cache or `TOPO_DEPS`; 60 s cap per solve, `all` about 25 min, `multi2` about 40 min, `slots` seconds |
| `carve_footprint` | `topo/carve_footprint_bench.cpp`: the carve as a footprint mapping through `solve_topology_mapping` (Auto, SAT and DFS engines), no solver change; groups `quick` (the rings blaze rejects, infeasible counting and area cases, a fork), `multi` (`mixed4` cablings, two 4x4 meshes), `repo` (repo slice grids from `carve_repo`), `all`, `multi2` (the same 32-rank multi-mesh instances as `topo multi2`), `file` (the Python `carve_bench_encodings` cases from `topo/carve_encoding_cases.txt`, so the Python tables and this one are the same instances) | built like `topo` (`make_topo_bench.sh <build> <bin> carve_footprint_bench.cpp`); no venv needed; `TOPO_TIMEOUT` per solve (default 20 s); `CARVE_ENGINES=sat,dfs` picks engines; `CARVE_CONFLICT_CAP=2000000000` removes the SAT conflict budget so a "no" is a proof (see section 5); `quick` and `multi` about a second each, `file` and `multi2` up to 40 min at a 60 s cap |

The folder is self-contained and can be copied anywhere. Outside a tt-metal checkout, set
`TT_METAL_HOME` to one (it supplies the MGD files and `mesh_graph_descriptor.proto`). Needs
`python3`, `protoc`, `g++` and `git`.

Knobs, as environment variables: `PER_MGD` (pipelines per fabric, default 3), `MAX_MESHES`
(largest fabric to include, default 8; the blitz rings go to 144), `CASE_TIMEOUT` (seconds per
case, default 1), `SEED`, `OUT` (output directory, default `./.out`), `BUILD_DIR` (tt-metal build
used for CaDiCaL, default `build_Debug`), `TT_BLAZE`, `CARVE_TIMEOUT` (seconds per carve solve;
each solve runs in a child process that is killed, since an alarm cannot interrupt CaDiCaL),
`CADICAL_SRC` and `CADICAL_LIB` (see below).

**Keep runs small on shared machines.** The large settings used for the results below
(`PER_MGD=20 MAX_MESHES=1000 CASE_TIMEOUT=3`) took about 16 minutes, mostly in cases that hit the
cap. The slow cases are opt-in with `big`: `./run_experiments.sh resolver big` (groups `odd` and
`big`, each over two minutes), `resolver_sat big` (over 230 s), `carve big` and `carve_repo big`
(one 9x9 case had not finished after 15 minutes). Run those on a spare machine.

Recorded outputs are in `results/`: the full Tier B run, and the quick runs of `resolver`,
`resolver_sat`, `carve`, `carve_repo` (with `CARVE_TIMEOUT=5`) and `planner` (tt-blaze
`45d3be26c4`) from 2026-10-22. A full rerun from 2026-10-23, with the answers to "should Tier B
check anything" and "SAT or DFS in Tier C", is in
[`results/rerun_2026-10-23/REPORT.md`](results/rerun_2026-10-23/REPORT.md).

`resolver_sat` takes its CaDiCaL from the tt-metal build. If the CPM source cache is gone, point
`CADICAL_SRC` at a CaDiCaL `rel-2.2.1` checkout; if the build is Debug, its archive is unoptimized
and every SAT time is about 10x slower, so point `CADICAL_LIB` at an `-O3` build.

## 1. Tier B: simple checks vs full search

**What is checked.** Each MGD is reduced to what Tier B can see: meshes, each mesh's host slices,
and the declared channel count per mesh pair. 229 files give 167 distinct fabrics. Pipelines are
generated for each fabric: rings, chains and forks, stages of 1, 2 or 4 slices, total size from a
quarter of the system to a little over it. Two checks run on each:

- **Simple checks** (`slim_check`): every stage fits some mesh; the total slices fit (and per
  binding set); a pipeline that fits on no single mesh has a connected group of meshes big enough.
  No placement.
- **Full search** (`full_check`): assigns every stage to a mesh so that slices fit and every
  pipeline edge between two meshes has a declared channel left; then, per mesh, places each stage
  as a rectangle of slices, with no overlap and with stages joined by an edge touching. It stops at
  the first success. Symmetry between identical meshes, look-ahead, memo and a node budget, as in
  the design doc's section 2.2.

This is placement on the declared fabric only: no chips, no PSD, no SAT seating, no cables, no
resolver.

**Results, 3,340 random cases (`results/tier_b_random_seed0_20per_fabric.txt`):**

| | Result |
|---|---|
| simple checks reject | 29.5%, mostly capacity |
| full search also rejects, after the simple checks pass | 22% of the simple passes |
| why | 86% the pipeline cannot be laid onto the declared connections; 14% the layout inside a mesh |
| of those, caught by a cheap degree check | 16% |
| full search time, fabrics of up to 8 meshes (Python) | median 0.08 ms, p99 270 ms, worst 2 s |
| fabrics of 16 to 144 meshes | 115 cases hit the 3 s cap; all passed the simple checks |

**The ring each MGD is built for** (one one-slice stage per slice): both pass on 121 fabrics, both
reject 5, and the full search alone rejects 17. Among the 17 are `blitz_decode_single_galaxy`,
`blitz_decode_*_dual_pod` and `pipeline_prefill_*galaxy`, whose declared connections form a line.
Their ring's loopback has no declared connection, so it presumably runs over host today
(unconfirmed against the blaze configs). A STRICT-only first version would reject them in Tier C
as well.

**Reading.** The gap is real but is a graph-embedding problem, the same one Tier C and the SAT
seating already solve with real links. On large fabrics the search is expensive. Keeping Tier B to
the simple checks loses early errors for some impossible descriptors and nothing else.

## 2. Carve inside one mesh: DFS vs SAT

**What is checked.** One grid of slices; stage k takes a rectangle of its shape; rectangles are
disjoint; stage k touches stage k+1 and the last touches the first. `dfs_solve` places stages in
ring order with a memo on `(stage, used, last, first)`, area pruning and a reachability check.
`sat_solve` is a CaDiCaL encoding: one variable per (stage, position), exactly one position per
stage, at most one stage per slice, and a touching clause per consecutive pair. `lattice_check` is
an exact counting pre-check: each h x w rectangle holds exactly one cell with `i % h == h-1` and
`j % w == w-1`.

| Case | Area | DFS | SAT |
|---|---|---|---|
| 4x6: 1x1, 4x 2x2, 1x1 | 18/24 | yes, 0.000 s | yes, 0.002 s |
| 4x8: 1x1, 6x 2x2, 1x1 | 26/32 | yes, 0.000 s | yes, 0.003 s |
| 8x8: 1x1, 14x 2x2, 1x1 | 58/64 | yes, 0.002 s | yes, 0.025 s |
| 8x8: 1x1, 15x 2x2, 3x 1x1 | 64/64 | yes, 0.183 s | yes, 0.396 s |
| 5x5: 1x1, 5x 2x2 (only 4 fit) | 21/25 | no, 0.008 s | no, 0.008 s |
| 7x7: 1x1, 10x 2x2 (only 9 fit) | 41/49 | no, 2.4 s | no, 2.0 s |
| 6x6: 1x1, 9x 2x2 | 37/36 | no, 0.001 s | no, 0.25 s |

The lattice check rejects every "no" row above in about 20 microseconds.

On repo-shaped grids (`carve_repo`), random 8x8 mixes that fill all 64 slices are hard for both:
with a 5 s cap, the DFS timed out on 2 of 5 and SAT on 3 of 5. Tight carves are the open case for
fail-fast, as they are for the resolver in section 4.

**Reading.** SAT is no faster when a carve exists and no better when it does not. The slow "no"
cases fail on a counting argument that a pre-check settles instantly.

A follow-up (`carve_encodings`, table in the 2026-10-23 report) tried a slice-level encoding, where
each stage is broken down to its slices and grouping rules tie them to one rectangle, with and
without symmetry breaking. It is slower than the position encoding on every 8x8 case and caps where
positions finish; the DFS stayed fastest on 9 of the 11 decided cases and lost only on 8x16 and
16x16 grids packed with identical wide stages. Slice-level propagation and symmetry breaking do
speed up infeasibility proofs (7x7: 5 s vs 40 s for the DFS), which the lattice pre-check already
answers in microseconds.

## 3. Blaze's carve planner

**What is checked.** `plan_ring_carve` from `tt-blaze/pipeline_builder/carve_planner.py`, run on
grids of 1x1 slices, against a brute-force search for a ring of disjoint, touching rectangles.

| Grid | Stage shapes | Planner | A ring that works |
|---|---|---|---|
| 4x4 | 1x1, 2x2, 2x2, 2x2 | `no-cycle-of-length-13` | yes |
| 4x4 | 1x1, 2x2, 2x2, 1x1 | plan found | yes |
| 4x6 | 1x1, four 2x2, 1x1 | `no-alignment` after 64 cycles | yes |
| 4x8 | 1x1, six 2x2, 1x1 | `no-alignment` after 64 cycles | yes |
| 4x8 | 1x1, 2x4, 2x4, 2x2, 1x1 | plan found | yes |

**Why.** The planner looks for a cycle of slices and cuts it into stages, so a stage of k slices
must be k consecutive slices on the cycle. The real requirement is only that consecutive stage
rectangles touch. Three effects follow: a ring whose slices add up to an odd number is always
rejected (grid cycles have even length), each multi-slice stage filters the cycles further, and the
64-cycle cap hides the cycles that would align.

**Reading.** The planner wrongly rejects carves that exist once several stages span several
slices. A carve that searches stage rectangles directly (as in section 2) does not have this
problem.

## 4. The resolver in `pipeline_builder.cpp`

**What is checked.** `resolve_pipeline_placement`'s search, copied line for line out of
`tt_metal/fabric/pipeline_builder.cpp` when `run_experiments.sh resolver` builds it (the anonymous
namespace in `tt::tt_fabric::detail`), so it always measures the current code; the results below
are from `088a531eb93`. `bench.cpp` runs it on
synthetic fabrics of 4x2-chip submeshes, each case a ring of single-submesh stages. `sat.cpp` is a
CaDiCaL encoding of the same placement: one variable per (stage, submesh), consecutive stages on
adjacent submeshes.

| Case | Layout exists | Resolver | CaDiCaL |
|---|---|---|---|
| one 4x8 grid, 32 stages | yes | 0.7 ms | 5 ms |
| one 8x8 grid, 64 stages | yes | 4.4 ms | 35 ms |
| odd 5x5 grid, 25 stages | no | 2.5 s | 2 ms |
| odd 5x7 grid, 35 stages | no | over 120 s | 5 ms |
| two 4x4 meshes, both cables on same-colour slices | no | 3.4 s | 3.0 s |
| two 4x8 meshes, both cables on same-colour slices | no | over 120 s | over 230 s |

**Reading.** The resolver is fast whenever a layout exists and exponential when none does. SAT
only wins on the odd grids, which fail a one-line colouring argument. The realistic failure,
cables landing on the wrong slices, is just as hard for SAT. On a full rectangular grid, whether a
path through every slot between two given slots exists has a constant-time answer (Itai,
Papadimitriou and Szwarcfiter, 1982): mostly a colour check, with exceptions for grids one to
three wide.

## 5. The carve as a footprint mapping (`carve_footprint`)

**What is checked.** Whether the carve can run on tt-metal's own topology solver unchanged, the
way the mesh seating does: one **global node per position** (a place one stage shape fits inside
one mesh, in host slices), an edge between two positions that do not overlap and border each
other (grid inside a mesh, or a cable between meshes), and the **pipeline as the target graph**
(ring, chain or fork). `add_required_constraint` keeps each stage on positions of its own shape;
`add_resource_constraint` (position to covered slices) keeps chosen positions disjoint. A mapping
is then exactly a carve. Every returned carve is re-checked independently (own shape, disjoint,
every pipeline edge touching), and on cases with at most 8 stages the yes/no is compared with a
brute-force search; a disagreement prints `MISMATCH`.

**How it differs from `topo`.** `topo_solver_bench.cpp` maps one node per slice of every stage
with fixed ports, which is exact only for single-slice stages. This encoding is exact for
multi-slice stages, forks and several meshes. Both stay runnable for comparison:

```bash
./run_experiments.sh topo slots            # the old encoding, pre-carved slots
./run_experiments.sh carve_footprint all   # this one; repo is the comparison with carve_repo
```

**First result** (`results/rerun_2026-10-23/carve_footprint_quick_multi_10s_cap.txt`, 10 s cap):
every case decided by all three engines in under 0.1 s and in agreement with brute force,
including the three rings blaze's planner rejects (4x4 13-slice, 4x6, 4x8), the `mixed4`
cablings from the design doc ((P,R) accept, (P,Q) and Case B reject), a 6x6 fork and two 4x4
meshes joined by cables. `Auto` picked SAT on every case: `topology_mapping_should_use_sat_engine`
(`topology_solver.tpp`) returns SAT whenever a resource constraint is set, unless DFS is forced,
so for this encoding `Auto` means SAT. The `dfs` column is the forced DFS engine.

**Against the Python encodings and the C++ engines** (60 s cap, files
`carve_footprint_encodings_cases_*`, `carve_footprint_repo_*`, `carve_footprint_multi2_*`; the
tables are in the report, section 3b "The carve as a footprint mapping"). The footprint SAT is
the fastest carve solver measured: 20x faster than the best Python column on the 8x8 fill, 25x
on the 16x16 ring of 62 2x2 stages, first to solve the 8x16 2x4 fill (2.8 s) and the 8x16
random 37-stage case (5.6 s) under 60 s, and 3x to 40x faster than the per-slice encoding on
the multi-mesh rings it solves. It loses to `dfs2` on two random 8x8 fills and caps on two
multi-mesh cases the per-slice SAT finished. Nothing finishes the 1-cable 4-mesh rings or the
8-mesh rings.

**A SAT "no" is undecided unless the budget is removed.** The solver runs every SAT solve under
a conflict budget (`kDefaultConflictCap`, 300k in `topology_solver_sat.cpp`) and reports a
budget-exhausted solve as "no mapping". The first 60 s run returned "no" on three cases the
Python DFS had carved. `CARVE_CONFLICT_CAP=2000000000` runs the solve through
`TopologyMappingEnumerationSession`, the only API that takes the budget, and those cases become
"cap" (undecided). The `*_sat_unlimited_budget.txt` files are the ones to read for proofs; any
integration must treat budget exhaustion as unknown, which today needs the session API or a new
flag on `MappingResult`.

**Which is best: blaze's carve planner + resolver against footprint + resolver** (the
2026-10-27 rerun, report section 5.3, `results/rerun_2026-10-27/carve_comparison.md`; 132
carve-requiring cases on one machine, 60 s cap). The planner is blaze's `plan_ring_carve` ported
to C++ (`topo/carve_planner_port.hpp`, blaze's 200k-expansion and 64-cycle budgets); the
footprint runs with resource completeness (report 5.4).

| Group | Cases | Planner + resolver: placed | Footprint + resolver: decided | Resolver after footprint, worst |
|---|---|---|---|---|
| original carve cases | 17 | 0 (17 budget-exhausted) | 16 | 0.029 s |
| heterogeneous sweep | 30 | 0 (30 budget-exhausted) | 30 | 0.001 s |
| single mesh, no 1x1 stages | 45 | 0 (stage-0 rule; off: 2) | 43 | 0.029 s |
| heterogeneous multi-mesh | 12 | 0 | 12 | 0.001 s |
| multi-mesh, no 1x1 stages | 28 | 0 | 26 | 0.036 s |
| **all** | **132** | **0** | **127** | **0.036 s** |

Where the planner does complete (small rings of columns) both take milliseconds, for example
4x8 ring of 32 one-slice stages: 0.038 + 0.001 s against 0.004 + 0.001 s. Use footprint +
resolver; the per-case table is in the design doc, 4.2 "Speed".

## What the experiments suggest

- **Tier B:** binding, capacity and a connectivity walk; no search. Placement stays in Tier C.
- **Carve:** search stage rectangles directly rather than slice cycles, as a footprint mapping
  through tt-metal's topology solver with the SAT engine (section 5), with resource
  completeness and a deadline that fails the seating rather than passing it. Run the cheap exact
  pre-checks (area, lattice, colour) before the solve. The resolver then checks the carve and
  records the exact channel of every crossing.
- **Resolver:** keep its DFS; add the colour pre-check for fully used meshes between fixed cable
  slots.
- **Blaze:** its planner rejects carves that exist; tt-blaze
  [#5775](https://github.com/tenstorrent/tt-blaze/pull/5775) adds CPU tests for this (xfail today).
- **STRICT-first:** line-connected blitz and prefill MGDs need their loopback marked as host-only,
  so RELAXED edges may be needed in the first version.

## Handing this to another agent

Point it at this README and `run_experiments.sh`, and ask it to keep to the default sizes unless
told otherwise. Useful prompts:

- "Run `./run_experiments.sh quick` and compare the output with the README's tables."
- "Add a case to `carve/carve_bench.py` for <grid and stage shapes>, run `./run_experiments.sh carve`."
- "Check whether the 17 designed-ring rejects in section 1 run over host in blaze today."

## Caveats

- The random pipelines are not real workloads, so the 22% shows the size of the gap, not how often
  it would happen.
- Connections declared between sub-graphs are spread over every mesh pair, which is optimistic.
- Meshes with more than two dimensions are not carved (`Carver` passes them).
- Python times are slower than C++ would be; the resolver numbers are C++.
