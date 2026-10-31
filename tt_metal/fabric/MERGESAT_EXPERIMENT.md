# MergeSat experiment — deterministic parallel incremental SAT as a gimsatul alternative

**Branch:** `rsong/exp-mergesat` · **Date:** 2026-09-16 · **Status:** benchmarked, not integrated.

## Motivation
gimsatul is fast but (a) one-shot (no incremental API) and (b) NONDETERMINISTIC (thread-race decides the
winner). We wanted a solver that is **parallel + incremental + deterministic** — MergeSat
(conp-solutions/mergesat, MiniSat/Glucose lineage) fits on paper: `-cores=N` deterministic parallel
(`-untouch-thread0`, `-diversify-rank`) + full IPASIR incremental (`ipasir_add/assume/solve/set_learn/
solve_final`). Determinism would remove the run-to-run variance that plagued the gimsatul experiments.

## Install (works)
```
git clone --depth 1 https://github.com/conp-solutions/mergesat
cd mergesat && make r lr -j 16        # build/release/bin/mergesat + build/release/lib/libmergesat.a
```
Builds clean. IPASIR symbols present in the static lib. Deterministic parallel via `-cores`.

## Standalone benchmark — SINGLE solve, 96-stage min-host CNF (42077 vars, 126610 clauses)
CNF dumped via `TT_TOPO_SAT_DUMP_DIMACS`. Same formula fed to both, 16-way parallel:

| solver | config | result | wall |
|---|---|---|---|
| gimsatul | 16 threads | SAT | **30.3s** |
| MergeSat | `-cores=16` (deterministic) | SAT | **255.2s** |

**gimsatul is ~8.4× faster than MergeSat on our formula.**

## Three-way comparison (same 96-stage CNF, single solve)
| solver | mode | determinism | wall |
|---|---|---|---|
| gimsatul | 16 threads | NO (thread-race) | 30s |
| MergeSat | -cores=16 | YES (deterministic parallel) | 255s |
| CaDiCaL | 1 thread | YES | >600s (unsolved, timed out) |
| gimsatul | 1 thread | YES | 369s (from earlier probe) |

So among DETERMINISTIC options MergeSat is the BEST (255s beats CaDiCaL >600s and single-thread gimsatul
369s) — its parallelism genuinely helps. It only loses to NONDETERMINISTIC gimsatul (30s), by ~8×.

## Full size sweep (single solve, same dumped CNF per size, 32 cores for parallel solvers)
| stage | vars | gimsatul 32t (nondet) | CaDiCaL 1t (det) | MergeSat 32c (det) |
|---:|---:|---|---|---|
| 32 | 14.5k | 0.18s | 1.6s | 4.0s |
| 64 | 28.3k | 8.0s | 63s | 81s |
| 80 | 35.2k | 23s | 777s | 143s |
| 96 | 42.1k | 32s | 1085s | 737s |
gimsatul dominates everywhere. CaDiCaL beats MergeSat on EASY instances (32/64 — MergeSat's 32-worker
overhead not worth it); MergeSat beats CaDiCaL on HARD ones (80: 5.4x, 96: 1.5x). MergeSat's determinism is
only a speed win over CaDiCaL past ~80-stage difficulty, and never competitive with gimsatul.
(32-stage MGD hand-generated to mirror gen_pipeline_sweep_mgds.py; CaDiCaL benchmarked via a tiny harness
linking build_Debug/third_party/libcadical.a.)

## Verdict + the determinism/speed tradeoff (fundamental)
MergeSat is not an engine replacement for the SPEED path (~8× slower than gimsatul). BUT it IS the fastest
DETERMINISTIC option we have. The ~8× is not a MergeSat weakness — it's inherent:
- gimsatul is fast BECAUSE it's nondeterministic (lock-free async racing). Determinism needs sync barriers.
- Det-vs-nondet gap WITHIN a lineage is now small (ManyGlucose/DPS closed it, ~1%). But all det-parallel
  solvers (ManySAT, ManyGlucose, MergeSat, DPS) are Glucose/MiniSat-lineage — the slower CORE. That LINEAGE
  gap vs gimsatul/Kissat is the real 8×. No fast (Kissat-lineage) solver has a deterministic-parallel mode.

## Recommended alternative: OUTPUT determinism, not solve determinism
A production mapper needs the SAME MAPPING each run (reproducibility), not the same search path. Keep fast
nondeterministic gimsatul (30s) and CANONICALIZE the output (deterministic symmetry reduction to a canonical
representative). Reproducible output at gimsatul speed — decouples solve-speed from output-determinism, and
reuses the symmetry-breaking machinery. This is the better path than adopting a slower deterministic engine.
Other det-parallel libs to know: [[ManyGlucose/DPS]] (gap-closed Glucose), Treengeling (Biere cube&conquer).

## Integration effort (if it had been competitive)
- Full CaDiCaL replacement: significant (pervasive `connect_learner`/`fixed`/`phase`/`limit`/`write_dimacs`).
- Core-only via IPASIR (`add/assume/solve/val`): feasible — the CaDiCaL-specific features we'd drop are the
  pool/warm-start machinery already shown to be dead ends.
- NOT pursued given the 8× deficit.

## Not yet tested (deprioritized by the 8× gap)
The IPASIR **incremental** path (solve → add blocking → solve, reusing state) — MergeSat's actual strength.
Only worth it if the 8× single-solve gap can be closed, which is unlikely. Alternative parallel-incremental
candidate with a modern backend: **MallobSat** (clause-sharing, IPASIR-emulated incremental) — a better bet
than MergeSat if we pursue parallel-incremental, but heavier (distributed framework).
