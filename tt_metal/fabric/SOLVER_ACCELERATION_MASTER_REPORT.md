# Inter-mesh minimal-host solve — acceleration experiments, master report

Consolidated findings from the gimsatul / one-shot / incremental / enumeration experiments. Companion to
`ONESHOT_EXTERNAL_SAT_EXPERIMENT.md` (raw tables) and the scripts in `oneshot_external_sat_scripts/`.
All numbers: 2x4/4x4 pipeline sweep onto SC36 (36 galaxies) mock, host cap on unless noted. Branch context:
solver-min code lives on `ridvan/gen-rank-bindings-all-solutions`; this doc/scripts on `ridvan/exp-oneshot-external-sat`.

## 0. The problem shape (why any of this is hard)
- Two independent difficulties: (A) **base embedding** — find any valid ring embedding; (B) **host minimization** —
  pack into the fewest hosts (a cap/occupancy objective).
- Base embedding is trivially easy up to 112 stages (<100ms). **All the cost is the 128->144 fill cliff** and the
  host-cap objective. Near full fill the solution space is tiny and rigid.
- The full problem is ONE hard CNF: embedding + injectivity + pinnings + preferred(DFS->hard at-least-k) +
  relaxed-channel + occupancy/cap. No soft/MaxSAT anywhere -> any SAT solver reproduces every feature.

## 1. Single min-host solve — gimsatul HYBRID is a big win
Our CaDiCaL encodes+drives; each heavy solve delegated to gimsatul (CLI subprocess, DIMACS round-trip via a
faithful clause "tee"; assumptions baked as hard units). Host cap on, one solve.

| 2x4 stages | pure CaDiCaL cold cap | best gimsatul hybrid | speedup |
|---:|--:|--:|--:|
| 64  | 11.4s | 5.6s (t8) | 2.0x |
| 96  | 32.6s | 52.9s (t32) | CaDiCaL wins |
| 112 | 988.5s | 125.7s (t16) | 7.9x |
| 128 | 1345.6s | **72.9s (t32)** | **18.5x** |
| 144 | **TIMEOUT >25min** | **~105s (t8/t32)** | solvable where CaDiCaL can't |

Findings: hybrid wins big on the hard cases (128, 144); **instance-dependent** (loses on 96); **best thread count is
non-monotonic** (64->t8, 96->t32, 112->t16, 128->t32, 144->t8). 4x4 is trivial (cap non-binding, <0.2s all sizes).

## 2. Why gimsatul is faster, and the incremental "tax"
- Same-version standalone CaDiCaL, one-shot on the same CNF: 128 = 43.7s (vs 205s our incremental base) -> **incremental
  mode is a ~5x tax** (restricts bounded variable elimination). But our MINIMIZE path already eliminates ~35% of vars
  ("35% pinned" in logs) -- it is NOT fully starved of BVE.
- gimsatul beyond that: ~45x over one-shot CaDiCaL on 128 via a stronger engine + 32-thread clause sharing.
- gimsatul is a **CLI subprocess, one-shot, no incremental API, no fastsat option** (only `-O` simplification level).

## 3. CaDiCaL ablation (host cap on, single solve)
- **Cold cap (mode3) is the most robust** min strategy.
- **Warm+descent (mode0) is the WORST** and times out at 128 -- the one-host-at-a-time descent re-solves tighter
  problems and grinds; the descent is intractable at the cliff on any engine.
- **fastsat erratic** on single solve (hurts 64/96, helps 128 855s vs 1345s). **seed high-variance** (times out at 128).

## 4. Enumeration / multi-solution (search_n, `-n N`) -- the incremental workload
Path: PRIME with a min-host solve (where gimsatul delegation lives), then enumerate `.next` on incremental CaDiCaL
with blocking clauses. gimsatul CANNOT do `.next` (no incremental) -- it only accelerates the prime.

| 2x4, to 5 solutions | wall |
|---|--:|
| CaDiCaL-only (enum_hardcap) | 64: 233s / 96: 292s |
| + fastsat | 64: 220s / 96: 281s (~5% faster, mostly the tail) |
| gimsatul-prime hybrid | 64: 366s (slower) / 96: TIMEOUT |

- **CaDiCaL-only iterative BEATS the gimsatul hybrid for enumeration.** An external gimsatul prime leaves the CaDiCaL
  enumerator COLD (no learned clauses to reuse), so every `.next` is slow.
- **Enumeration is SUPER-LINEAR** (64: 3 sols=49s, 5 sols=233s) -- later distinct solutions get much costlier.
- **fastsat helps the incremental path ~5%** (concentrated on the tail); the ONLY optimization that helped enumeration.

## 5. gimsatul-prime UNLOCKS the production-dead cases (key result)
Production baseline finds **0 solutions** on 2x4 96/112/128/144 (can't find even 1 in the 15-min budget). gimsatul-prime
enumeration (MIN_MODE=3 + GIMSATUL, `-n 20`, 40-min cap):

| 2x4 stages | time to 1st solution | solutions in 40 min | production baseline |
|---:|--:|--:|--:|
| 96  | **6s** | 4 | **0** |
| 112 | 390s | 3 | **0** |
| 128 | **101s** | 1 | **0** |
| 144 | **63s** | 1 (done in 64s, rc=0) | **0** |

- gimsatul **breaks through the "can't find solution 1" wall** that gives production 0 -> 1-4 > 0. Real win.
- But the **super-linear tail** means it stalls after a few: 128 got #1 in 101s then no #2 in the remaining ~38 min.
  So it does NOT reach 20 on the hard cases -- it converts "0 solutions" into "a few solutions".
- **144 is special:** full fill uses ALL 36 hosts in every solution, so under distinct-host-set dedup there is
  effectively ONE distinct solution -> enumeration finished cleanly in 64s (rc=0), first solution at 63s. The 144
  "0 in production" was purely the can't-find-#1 wall; gimsatul-prime fully solves it (1 distinct solution, 64s).

## 6. Partitioned-parallel (pin stage-0) -- NEGATIVE for single hard solve
Split by pinning stage-0 placement (vars 1..144, the first exactly-one group), solve buckets in parallel with plain
CaDiCaL. Result on 128 hardcap: **17/18 buckets TIMEOUT >400s, 1 SAT@390s**. Pinning ONE of 128 stages barely reduces
difficulty -> residual is ~as hard as the original. Partition-by-one-variable does NOT accelerate a single hard solve.
(Caveat: 400s cap < the ~1345s a full solve needs, so this shows "doesn't rescue within 400s", not a clean loss.)

## 7. Warm state vs gimsatul -- what incremental has that gimsatul can't
gimsatul is stateless/one-shot (immutable shared clauses = its speed trick = forbids incremental mutation). It cannot
carry: (a) learned clauses, (b) saved phases (warm-start), (c) VSIDS activity.
- Saved phases only help when the next solution is NEAR the last -> useless/harmful at the cliff (bias points at the
  just-blocked region).
- Learned clauses + VSIDS are GLOBAL and DO carry useful structural knowledge even for far solutions -- BUT: CaDiCaL
  **deletes most learned clauses** (LBD-based reduction keeps only an elite core), learned clauses are **densest where
  derived** (region of #1, not the far #2), and VSIDS re-tunes/decays. So the carryover is real but partial.
- Empirical clincher: the warm carryover is a **single-thread modest edge**; it loses to gimsatul's **32-thread raw
  power** on these instances. Warm single-thread CaDiCaL < cold 32-thread gimsatul at the cliff.

## 8. Recommendations / where this lands
- **Single min-host solve on the hard cases (112-144):** use the gimsatul hybrid at a high thread count; race a couple
  thread counts (+ pure CaDiCaL cold cap) and take first-to-finish (covers the instance-dependence + non-monotonicity).
- **Enumeration on easy/mid cases:** stay on CaDiCaL-only + fastsat (hybrid hurts).
- **Enumeration on the production-DEAD cases (96-144):** gimsatul-prime to unlock solution #1 (1-4 > 0), accepting it
  won't reach 20 due to the super-linear tail.
- **Drop the warm-descent path** for large solves (worst performer, times out).
- **Open / higher-effort ideas not yet built:** repeated-cold-gimsatul enumeration loop (each solution = fresh cold
  gimsatul + blocking clause -- promising for the hard tail where incremental stalls); cube-and-conquer (march_cu +
  parallel CDCL -- the proper parallelization; my pin-stage-0 was a degenerate version); Painless framework; Mallob
  (only true parallel-incremental, heavy MPI); dedicated AllSAT solvers (bc_minisat_all) -- fast enumeration but lose
  features. freeze-then-eliminate to keep BVE during incremental enumeration.

## Reproduce / artifacts
Scripts: `oneshot_external_sat_scripts/{full_sweep,hybrid_sweep,enum_sweep,partition_test,gimfirst_enum,build_gimsatul,
build_cadical,dump_cnf}.sh`. Raw results: `RESULTS_*.txt` in the same dir. Env knobs: `TT_TOPO_SAT_MIN_MODE`
(0 warm+descent / 1 warm+lock / 3 hardcap), `TT_TOPO_SAT_GIMSATUL[_BIN|_THREADS]`, `TT_TOPO_SAT_GIM_FIRST`,
`TT_TOPO_SAT_FASTSAT`, `TT_TOPO_SAT_SEED`, `TT_TOPO_SAT_DUMP_DIMACS`.

---

# Story 3 result — incremental vs from-scratch enumeration (seed-controlled)

Question: how much does incremental state reuse (learned clauses + phases + VSIDS) actually help when enumerating
distinct (unique-shape / distinct-host-set) solutions? Setup: CaDiCaL-only, MIN_MODE=3 hardcap (no warm descent),
`-n 5`, 15-min cap. **A** = incremental (one solver, blocking clauses); **B** = from-scratch (fresh solver +
re-encode + replay all prior blocks each solution). Both paths verified via an always-on marker; 3 seeds each
(1/7/42) to rule out seed variance. Env: `TT_TOPO_SAT_ENUM_FROMSCRATCH=1`, `TT_TOPO_SAT_SEED=N`.

| stages | incremental (5 sols) | from-scratch | winner (all 3 seeds) |
|---:|--:|--:|:--|
| 64 | 216 / 310 / 218 s → 5 | **135 / 252 / 147 s → 5** | from-scratch (~1.5×) |
| 80 | 902 s → **1** (all seeds) | 902 s → **3-4** (all seeds) | from-scratch (3-4× more under budget) |
| 96 | **310 / 347 / 314 s → 5** | 902 s → **1** (all seeds) | incremental (big) |

**Finding: the winner is STRUCTURAL (per stage-count), not seed-dependent.** Seed spread is tight (96-incr:
310/314/347s) — far smaller than the mode gap. There is **no universal winner**:
- 64, 80 → from-scratch wins; 96 → incremental wins, every seed.
- Consistent with the warm-state theory: incremental reuse is a big win when consecutive distinct solutions are
  *structurally close* (96: nails 5 in ~320s), and a liability when they're *far apart* (80: warm phases mislead ->
  stalls at 1, while from-scratch's fresh fast hardcap solves keep producing 3-4).

Implication: a robust production enumerator should **run both modes and take whichever is winning** (or detect the
regime) rather than committing to one. Raw data: oneshot_external_sat_scripts/RESULTS_story3_*.txt.
