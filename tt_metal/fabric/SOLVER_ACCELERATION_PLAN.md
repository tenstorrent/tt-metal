# Inter-mesh min-host solver acceleration — design & implementation plan

**Area:** fabric / scaleout auto-mapper · **Status:** design (this PR) · **Prototype + full data:** branch
`rsong/exp-gimsatul-pool-rebased` · **Prior experiment PRs:** #51533 (one-shot/external SAT), #52849
(clause-sharing portfolio).

This document consolidates what a long sequence of experiments established and lays out the plan to
productionize the winner. It is the plan of record; the code it recommends is prototyped and benchmarked
on the experiment branch and referenced throughout.

---

## 1. Problem

The inter-mesh minimal-host solve — place an N-stage pipeline ring onto physical hosts using the fewest
hosts — is intractable at scale on the single-threaded incremental CaDiCaL path. Symptoms (2x4 pipeline
sweep → SC36, 36 galaxies):
- **Single min-host solve:** 2x4-128 ~22 min; 2x4-144 never finishes (>25 min).
- **Multi-solution enumeration (`-n 20`):** 2x4 96/112/128/144 return **0 solutions** in the 15-min budget
  (can't find even solution #1). This is why the 96–144-stage rings are **disabled** in the SC24 CPU sweep
  (issue #51629 / PR #54682).

The full problem is ONE hard CNF (embedding + injectivity + pinnings + preferred + relaxed-channel +
occupancy/host-cap). No MaxSAT — every feature is hard-clause encoded, so any SAT engine reproduces every
feature and nothing is lost by swapping engines.

## 2. What the experiments established (evidence, not opinion)

Two independent difficulties: **(A) base embedding** (find any ring embedding — easy to 112, <100ms) and
**(B) host minimization + enumeration tail** (the cost; the 128→144 "fill cliff" where the solution space
is tiny and rigid).

**Findings (see §5 for the full benchmark table):**
1. **gimsatul (parallel, lock-free clause sharing) crushes the single hard solve** — 128: 1346s→73s
   (18.5×); 144 solvable (~105s) where CaDiCaL times out. Instance-dependent; best thread count non-monotonic.
2. **gimsatul is one-shot** (immutable shared clause DB = its speed = no incremental API). It cannot drive
   descent/enumeration by itself.
3. **The transferable half of incremental state can be baked into gimsatul as clauses.** Learned clauses and
   root-fixed units are *formula-entailed*, and our workload is add-only, so injecting them into each
   gimsatul run is sound. This is the **distilled clause pool**. The non-transferable half (saved phases,
   VSIDS) was shown *harmful* at the cliff, so dropping it is a feature.
4. **The clause-share portfolio (#52849)** reproduces gimsatul's clause-sharing *natively in CaDiCaL* (stays
   fully incremental) — but its published wins were base-embedding only; **on the actual min-host objective
   it times out** on 128/144 and can't enumerate. Its real value is as the pool's *export machinery*.
5. **gimsatul can be driven in-process as a linked library** — validated end-to-end (§4): no subprocess, no
   DIMACS file; the pool feeds straight in as clauses.
6. **The 112/128 cliff is run-variance dominated for every mode** — the answer there is a best-of-N race,
   not a single engine.

## 3. Decision

**Ship gimsatul driven in-process (P6), wrapped in a best-of-N race (P2); do NOT ship the distilled pool as
a default — demote it to an experiment gated behind P3.** Rationale from the *8-seed, contention-controlled*
data (§5a — this replaces earlier overstated reads): with constant thread usage the pool is **not
meaningfully better than cold gimsatul** — marginal at 112 (4/8 vs 3/8), a wash at 128, and a **net negative
at 144** (~3× slower; injection overhead with nothing to prune), because in pure gimsatul-every-step mode it
barely fills. So the load-bearing wins are: (1) **in-process gimsatul** (removes the subprocess/DIMACS cost —
done, P6), and (2) the **best-of-N race** (the real answer to gimsatul's large run-to-run nondeterminism —
144 walls spanned 66–973s). The pool waits on **P3** (a live CaDiCaL searcher feeding it real learned
clauses) to justify itself, and is off by default until then. The clause-share portfolio is kept only as the
pool's producer; plain CaDiCaL is the race's warm arm + fallback.

## 4. Architecture

```
CaDiCaL driver — owns the session: encode, host-cap, preferred, pinnings, symmetry, blocking, decode, dedup
  │   each CaDiCaL solve attaches a Learner ──► DistilledClausePool
  │       (short learned clauses + root-fixed units; deduped; cap-tagged; size-bounded; per-process, in-memory)
  │
  ├── each heavy solve (prime or enumeration step):
  │      RACE {  gimsatul(threads=t1) ,  gimsatul(threads=t2) ,  CaDiCaL cold-cap (warm/incremental)  }
  │        every gimsatul arm gets:  base CNF  +  permanent host-cap units  +  blocking clauses  +  POOL
  │        first to finish wins; cancel the rest; parse model back to driver vars
  │      the CaDiCaL arm's Learner keeps the pool filling every step (even when a gimsatul arm wins the solve)
  │
  └── add blocking clause → next step (pool now richer) → repeat until -n N or exhaustion
```

**In-process gimsatul (validated).** gimsatul's core is library-shaped:
`initialize_options → new_ruler(vars,threads) → add clauses → simplify_ruler → clone_rings → solve_rings →
winner->status (10 SAT/20 UNSAT) → extend_witness → read model[var]`. A ~230-line shim
(`gim_new/gim_add_clause/gim_solve/gim_val/gim_free`) over a vendored static lib exposes this. Literal
encoding is gimsatul's `2*var+sign`; the pool + blocking clauses feed through `gim_add_clause` directly. No
gimsatul source changes. **ABI footgun:** the shim TU must compile `-DNDEBUG` (matching gimsatul's objects —
`struct ruler` has an `#ifndef NDEBUG` field) — baked into the CMake target.

## 5. Evidence — full benchmark (latest solver base, 90 cells, seeds 0/7/13/42)

> **⚠️ Read these numbers as distributions, not point comparisons — gimsatul is nondeterministic
> run-to-run, even at a fixed seed.** The seed controls only the *CaDiCaL driver's* randomization; the
> actual solve is delegated to gimsatul's 32-thread parallel portfolio (lock-free clause sharing, threads
> racing), and which thread finds the model first depends on wall-clock scheduling/timing, not the seed.
> So a B cell and the A cell at the same size/seed are **two independent nondeterministic computations** —
> re-running either one gives a different wall time (e.g. 144 measured 252s in one run and 925s in
> another for the same mode). A single-seed A-vs-B gap is therefore mostly this variance, **not** the pool
> helping or hurting. Only the per-size *distribution* across many seeds (median wall + completion rate) is
> a valid A-vs-B signal; a multi-seed paired sweep at 112/128/144 is in progress to firm this up. Where the
> pool has few clauses to contribute (144: a single rigid host-set; and pure-gimsatul mode fills the pool
> only with fixed units), B and A are expected to be a wash within that variance.

Enumeration `-n 5`, solutions found (`@cap` = hit the 20-min budget):

| size | **B: gimsatul+pool** | **A: gimsatul cold** | **D: clause-share portfolio** |
|---:|---|---|---|
| 64  | 5/5/5 · 29–49s | 5/5/5 · 29–53s | 5/5/5 · 187–265s |
| 80  | 5/5/5 · 116–202s | 5/5/5 · 107–177s | 5/5/5 · 497–1048s |
| 96  | **5/5/5/5 · 174–348s** | 5/5/5 · 173–395s | 5, 2@cap, 2@cap · 945s |
| 112 | 5/5, 2@cap · 944–1108s | 5/5/3 · 357–737s | 0, 0 @cap |
| 128 | **5**, 4@cap, 3@cap, 3@cap · 837s | 2, 2, 1 @cap | 0/0/0 @cap |
| 144 | 1 (exhaustive) · 252s | 1 (exhaustive) · 132s | 0 @cap |

`-n 10` deep tail (seeds 0/7): 96 → B 10·493s / A 10·1159s / D 10·958s; 128 → B 3–4@cap / A 7@cap / D 0@cap.
Primes: B solves all sizes/seeds with **no timeouts**; D times out on the 128 prime.

### 5a. Contention-controlled re-measurement (the table above was contaminated)

The 90-cell table above used **32 gimsatul threads under a 36-rank mpirun on a 64-core box** — where the 35
non-solving ranks busy-wait at ~89% CPU each (~31 cores), so 32 threads + ~31 spinning cores ≈ 63, right at
the oversubscription edge, and gimsatul's threads were preempted by a *fluctuating* amount → timing variance
that is partly a harness artifact. A re-run at **16 threads (16 + 35 spin ≈ 51 ≪ 64, guaranteed headroom)**,
paired A-vs-B on the same seeds {1,5,99,123}, isolates the pool from CPU contention:

Final **8-seed** result (seeds {1,5,7,42,99,123,777,2024}), every cell verified at `load ≈ 47–52` on the
idle 64-core box (constant thread usage confirmed per-cell):

| size | **B: gimsatul+pool** | **A: gimsatul cold** | honest verdict |
|---:|---|---|---|
| 112 | completes **4/8**; faster when both finish (623 vs 907s; 626 vs 1197s) | completes **3/8** | **marginal B edge** |
| 128 | **0/8** complete; median 2 sols @cap | **1/8** complete; median 2 sols | **wash** (A took the sole completion at 16t) |
| 144 | 8/8 · median **~507s** | 8/8 · median **~157s** | **A ~3× faster — pool is a net negative here** |

**Final conclusion (8 seeds, contention-controlled — supersedes all earlier framing):**
- The original "pool wins broadly" was an artifact of **CPU oversubscription + too few seeds**. With
  constant thread usage and 8 seeds, **gimsatul+pool is *not* meaningfully better than cold gimsatul.**
- 112: only a **marginal** edge (4/8 vs 3/8 completions). 128: **wash** at 16 threads (the 32-thread pool
  lead did not survive a fair core budget). 144: the pool is a **net negative** (~3× slower) — its injected
  clauses add propagation overhead where a single rigid solution leaves nothing to prune.
- Root cause: in **pure gimsatul-every-step mode the pool barely fills** (fixed units only, no learned
  clauses), so it can't help and sometimes hurts. **The pool only earns its keep with P3** — a live CaDiCaL
  searcher feeding it real learned clauses — which this benchmark does *not* exercise.
- **Revised ship recommendation:** the load-bearing wins are **in-process gimsatul (P6)** and the
  **best-of-N race (P2)** — which directly answers gimsatul's large run-to-run nondeterminism (144 walls
  spanned 66–973s across seeds/modes). The **distilled pool is demoted to an experiment gated behind P3**;
  do not ship it as a default until P3 shows it accumulates useful clauses. Cold-gimsatul-in-a-race is the
  safe default.
- **Methodology (mandatory for future runs):** cap `gimsatul_threads + mpi_ranks ≤ cores` (the 36-rank
  harness busy-waits ~31 cores regardless of yield flags), log `load@start` per cell, and average ≥8 seeds —
  single-seed A-vs-B gaps are pure nondeterminism.

D remains not viable alone on min-host (times out on 128 prime + all 128/144 enums) in either measurement.

## 6. Implementation stories

- **P6 — In-process libgimsatul (DONE, prototyped on the branch).** Vendored gimsatul + shim + CMake static
  lib (`TT_METAL_FABRIC_GIMSATUL` option) + call-site rewrite (subprocess/DIMACS removed). Verified: 96
  prime 59s in-process vs ~83s subprocess; `-n 3` re-entrancy proven. **Effort: done.**
- **P1 — Harden pool + gimsatul path [M].** DistilledClausePool (Learner export, fixed-unit harvest,
  cap-tag soundness, size cap), collapse env knobs into one `FabricSolverConfig`. (Prototyped; needs config
  consolidation + tests.)
- **P2 — Best-of-N race [M, highest cliff value].** Launch {gimsatul t8, gimsatul t32, CaDiCaL cold-cap} per
  heavy solve, first-to-finish, cancel the rest. Covers instance-dependence + 112/128 run-variance.
- **P3 — Pool enrichment [S].** The race's CaDiCaL arm keeps the pool filling even when a gimsatul arm wins
  (in pure-gimsatul mode the pool only gained fixed units). Optional DRAT-mining of gimsatul's own short
  clauses as a hard-tail escalation (gated off; heavy).
- **P4 — Correctness guards [S, required].** Cap-soundness unit test (clause tagged ≤K only injected when
  run-cap ≤K); >5000-var family gate (intra- vs inter-mesh pools never mix); model-equivalence test (pool
  changes speed, not the answer set); fallback test (lib absent → CaDiCaL).
- **P5 — Wire in + close #51629 [S].** Route `generate_rank_bindings`/the sweep through
  `FabricSolverConfig{engine=auto}` (race+pool default; `engine=cadical` escape hatch). This is the lever to
  **re-enable the 96–144-stage rings disabled in the SC24 sweep (PR #54682)**.

**Sequencing:** P6 (done) → P1+P2+P4 (one focused PR — the ship) → P5 (default + #51629) → P3 (follow-up).

## 7. Acceptance metrics (all met by B in the benchmark except the last)
- 2x4-128 single solve < 5 min. ✅ (73s–353s)
- 2x4-144 produces a valid solution (was: never). ✅ (132–415s)
- 2x4 96/112/128 enumeration returns > 0. ✅ (was 0)
- 96 `-n 5`/`-n 10` completes reliably. ✅ (B every seed)
- 2x4-128 enumeration to ~20 distinct solutions. ⚠️ partial — cliff is intrinsically rigid; the race raises
  the floor but `-n 20`-at-128 remains the open research edge (cube-and-conquer / Mallob if it becomes a hard
  requirement).

## 8. Explicitly NOT doing
- Clause-share portfolio as a standalone min-host mode (loses; kept only as the pool's export path).
- Making gimsatul itself incremental (architecturally impossible — the pool IS the workaround).
- Cube-and-conquer / Mallob now (heavier; only if `-n 20`-at-128 survives P2 as a hard requirement).

## 9. Risks
- gimsatul's best thread count is instance-specific and its parallel search is nondeterministic run-to-run
  → the race (P2) is the mitigation, not a single-config bet.
- Vendored third-party C in the build: pinned source, `-w` on the target, `-DNDEBUG` correctness — all
  handled in P6's CMake target.
- The super-linear enumeration tail near the fill cliff may cap achievable large-`-n` at 128 regardless of
  engine (physics of a tiny rigid solution space), independent of this work.
