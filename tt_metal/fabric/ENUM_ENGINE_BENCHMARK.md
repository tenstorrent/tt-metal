# Enumeration engine benchmark — repeated-cold gimsatul vs warm incremental CaDiCaL

**Question.** For enumerating *N* solutions of the inter-mesh SAT problem, what is fastest and most stable: solving
each solution **cold** with gimsatul (a purpose-built parallel clause-sharing solver, no incremental API), or keeping
one/many **warm** incremental CaDiCaL solvers and adding blocking clauses? And do the hybrid / multithreaded variants
help?

**Method (apples-to-apples, isolates the SAT engine).** Dumped the real **2×4-128 base-embedding CNF** onto SC36 via
`TT_TOPO_SAT_DUMP_DIMACS` (55,024 vars, 156,936 clauses). Every approach enumerates successive solutions with the same
**full-model blocking clause** (¬ the entire assignment) so the comparison is purely "how fast does this engine find
the next satisfying assignment". Solvers: **gimsatul v1.1.3** (built from source, `--threads=8`), **CaDiCaL** (the
version tt-metal links, via `libcadical.a`). Harnesses in `enum_engine_benchmark_scripts/`. All numbers on base 128;
the capped/min-host regime is **not yet tested** (see Open).

## Approaches benchmarked

| approach | what it is | harness |
|---|---|---|
| **cold gimsatul** | solve → parse model → append blocking clause → re-solve cold, repeat | `cold_enum.sh` |
| **single warm CaDiCaL** | one incremental solver: solve → block → solve … (learned clauses + phases persist) | `warm_enum.cpp` |
| **hybrid** | gimsatul finds #1 cold, then hand off to a warm CaDiCaL for the rest | `hybrid_enum.sh` |
| **MT warm** | 16 incremental CaDiCaL workers sharing learned + blocking clauses (native clause-sharing) | `mt_warm_enum.cpp` |

## Headline results — 10 solutions, base 128

| approach | time (10 sols) | stability (6 runs) | shape |
|---|---:|---|---|
| **cold gimsatul** | **~13 s** | **12.8–15.4 s (tight)** | steady ~1–2 s every solution |
| MT warm (16-way share) | ~40 s median | **16–90 s (very wide)** | #1 dominates; #2 = free *or* 60–65 s thrash; tail free |
| hybrid (gimsatul #1 → 1 CaDiCaL) | ~33–40 s | ~stable | 0.2 s + one ~32 s solve, then free |
| single warm CaDiCaL | 70–148 s | **wild seed gamble** | #1 = 42–100 s; #2 = 0.03 s *or* 91 s thrash |

**Winner on base 128: repeated cold gimsatul — fastest and by far the most stable.**

## The per-solution breakdown (where the time goes)

**cold gimsatul** — cost is *evenly spread*, first solution is cheap:
```
sol:   1     2     3     4     5     6     7     8     9    10
delta 0.22  1.13  3.91  2.15  0.33  2.95  2.00  2.86  0.38  2.80   (s)   -> linear, #1 ~2%
```
**warm (all CaDiCaL variants)** — cost is *front-loaded on #1*, then the tail is ~free (full-model blocking makes
near-duplicates trivial), but **#2 is a coin flip**: either ~0.05 s or a **60–90 s phase-thrash**.

## Why warm CaDiCaL thrashes (the key mechanism)

After finding solution #1, CaDiCaL's **saved phases point at exactly #1's assignment**. Searching for a *different*
solution, it keeps reconstructing #1, slams into the (huge) blocking clause, flips one variable, and phase-saving drags
it right back — thrashing for up to ~90 s before escaping #1's basin. Once escaped, near-neighbours are free. Whether it
thrashes is **seed-dependent** (single-thread) or **nondeterministic** (MT) — hence the wild variance.

## What the variants do about it

- **Hybrid** *avoids* the thrash: gimsatul finds #1, so CaDiCaL **never phase-pins to #1** and finds #2 in a clean
  ~32 s solve (no thrash), then free. 20 sols: **32.6 s** (gimsatul 0.19 s + CaDiCaL 32 s + free tail). Robust.
- **MT warm (16-way)** *reduces but does not kill* the thrash: with 16 diverse seeds sharing clauses, usually one worker
  escapes #1 fast — but not always (2 of 6 runs still thrashed 65–90 s). Wide distribution (16–90 s). Native, no
  external dep, but doesn't catch gimsatul and isn't stable.
- **Single-thread warm** is a pure seed gamble (70–148 s).

## 20-solution results (base 128)

| approach | 20 sols |
|---|---:|
| cold gimsatul | 27 s |
| hybrid | 33 s |
| single warm | 110 s (#1+#2 then free) |
| MT warm | 86–137 s (tail grows as the easy near-#1 region exhausts) |

## Why gimsatul wins (base 128)

gimsatul is a **purpose-built parallel clause-sharing solver**: it does the 16-way sharing *natively and lock-free*,
and it has **no phase memory to poison itself**. Every cold solve is a clean ~1 s parallel search. MT-warm reimplements
sharing on top of *incremental* CaDiCaL and inherits (a) CaDiCaL's phase-saving pathology and (b) a coarse lock-based
pool — so gimsatul does the same trick better and skips the failure mode.

## Caveats

- **Full-model blocking** makes the near-tail "free" (near-duplicate assignments differing in one auxiliary var). Real
  distinct-*mapping* (shape-dedup) enumeration would not get those for free; the per-solution costs would rise for all
  approaches. This benchmark measures *engine speed at producing the next assignment*, which is the right comparison
  for the cold-vs-warm question, but absolute solution counts are not distinct mappings.
- **Base (easy) regime only.** gimsatul solves base 128 in ~1 s, so repeated-cold is cheap. On the **capped/min-host**
  regime gimsatul is ~73 s *per solve*, so repeated-cold would be ~73 s × N and the amortizing warm/hybrid approaches
  should win. **Untested — the decisive next measurement.**

## Candidate directions to implement (for later decision)

1. **Repeated cold gimsatul** — best + simplest on base; external dep; loses on capped (73 s × N).
2. **Hybrid (gimsatul #1 → warm CaDiCaL)** — robust across regimes: pay the hard solve once, warm cheap tail, and it
   *structurally avoids the phase-thrash*. Predicted winner on capped. Needs the gimsatul round-trip.
3. **MT warm clause-sharing (native CaDiCaL)** — no external dep, beats single-thread, but unstable and doesn't catch
   gimsatul on base. Would benefit from phase-reset to attack the thrash.
4. **Adaptive portfolio** — race cold + warm + hybrid on one shared frontier, keep whoever produces (see
   `ADAPTIVE_PORTFOLIO_ENUM_PLAN.md`).

## Reproduce

Build gimsatul (`git clone https://github.com/arminbiere/gimsatul; ./configure && make`). Dump the CNF with
`TT_TOPO_SAT_DUMP_DIMACS` + `TT_TOPO_SAT_NO_MINHOST=1` on the 2×4-128 pipeline MGD onto SC36. Then:
```
# warm single-thread:      warm_enum <cnf> <N> [seed]        (link libcadical.a)
# cold gimsatul repeated:  cold_enum.sh <cnf> <N> <threads>
# hybrid:                  hybrid_enum.sh <cnf> <N> <threads> [cadical-seed]
# MT warm 16-way:          mt_warm_enum <cnf> <N> <workers> <hint:0|1> [seedbase]
```
