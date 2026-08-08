# One-shot / external-parallel SAT for the inter-mesh minimal-host solve — experiment log

**Branch:** `ridvan/exp-oneshot-external-sat` (off `origin/main`).
**Scope:** ONLY a CNF-export hook (`write_dimacs`) + a benchmark harness. **No solver-minimization behavior is
changed** — the production solve path is untouched unless `TT_TOPO_SAT_DUMP_DIMACS` is set. This branch is a place to
track the numbers and the implementation plan; it does not itself change how we solve.

## TL;DR

The expensive part of the inter-mesh solve (the base embedding, e.g. 2x4-128 = 205s on our path) is a **single hard
CNF**. Two independent, stackable wins:

1. **Solve the first solution *one-shot* instead of incremental** — ~5–8× for free, no new dependency. Our production
   solver runs in *incremental* mode (`ilb=2`, tuned for blocking-clause enumeration), which **restricts CaDiCaL's
   bounded variable elimination**. The same CaDiCaL (same version) run one-shot eliminates ~34% of the variables up
   front and is far faster on the same CNF.
2. **Use a stronger / clause-sharing external solver (gimsatul) one-shot** — a further large win on top, **instance
   dependent**: huge on 128 (~230×), modest+threaded on the harder 144.

Neither regresses preferred constraints or pinnings: those are **DFS-bounded and encoded as hard `at-least-k`
clauses** (not soft/MaxSAT), so the whole problem is one hard CNF a plain SAT solver handles verbatim.

## Method (apples-to-apples)

- **CNF capture:** added `TopologySatSolver::write_dimacs()` (thin CaDiCaL passthrough) and a dump hook in
  `solve_with_symmetry_break` — the funnel **both** the no-preferred (`solve_hard_only`) and preferred paths hit, so
  the dumped CNF is feature-complete (embedding + injectivity + required pinnings + preferred-at-least-k + relaxed
  channel literals). Gated `>5000 vars` to skip the tiny per-mesh intra solves. Env: `TT_TOPO_SAT_DUMP_DIMACS=<path>`.
- **Instances:** 2x4 pipeline sweep, base embedding (`TT_TOPO_SAT_NO_MINHOST=1`), stages 64/80/96/112/128/144 mapped
  onto SC36 (36 physical galaxies). Var/clause counts scale linearly with stages.
- **Solvers, all on the identical DIMACS file:**
  - *Our production path* — CaDiCaL 2.2.1 **incremental** (numbers from this session's profiling).
  - *One-shot standalone CaDiCaL 2.2.1* — the **same version we link**, built from `.cpmcache`, run one-shot.
  - *gimsatul v1.1.3* — native multi-threaded clause-sharing solver, `--threads=N`.
- Wall clock via `date`; each solver verified to emit `s SATISFIABLE` + a model. 420s cap per run.

## Results — raw wall-clock to first solution (base embedding)

| stages | vars | our incremental CaDiCaL | 1-shot CaDiCaL (same ver) | gimsatul t1 | gimsatul t8 | gimsatul t16 | gimsatul t32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64  | 27,440 | ≪1s (easy) | 38ms  | 47ms  | 70ms  | 84ms  | 87ms  |
| 80  | 34,336 | ≪1s (easy) | 43ms  | 60ms  | 96ms  | 99ms  | 108ms |
| 96  | 41,232 | ≪1s (easy) | 52ms  | 73ms  | 115ms | 116ms | 142ms |
| 112 | 48,128 | ~1s (easy) | 64ms  | 94ms  | 150ms | 148ms | 172ms |
| 128 | 55,024 | **205s**   | **43.7s** | **0.93s** | **0.19s** | 0.65s | 0.26s |
| 144 | 61,920 | **910s** (no-min) | **113s** | **>420s** | >420s | **69s** | 280s |

Our other measured options on 128, for reference: **fastsat (config only) = 63s**; **16-way incremental seed
portfolio + fastsat = 13.7s**.

### Speedups on the two hard cases (vs our incremental production path)

| case | 1-shot CaDiCaL | gimsatul (best threads) |
|---|---|---|
| 128 | 205s → 43.7s (**4.7×**) | 205s → 0.19s (**~1000×**) |
| 144 | 910s → 113s (**8.1×**) | 910s → 69s (**13×**) |

## Analysis — what actually causes the win

- **Base embedding is trivially easy up to 112** (<100ms one-shot, either solver). The entire cost is the 128→144
  fill cliff. Any optimization only matters there.
- **Incremental mode is a real ~5–8× tax.** Same solver, same CNF: incremental 205s vs one-shot 43.7s on 128 (910s vs
  113s on 144). One-shot eliminated 18,751 vars (128) / 31,110 vars (144) — elimination that incremental mode
  suppresses because an eliminated variable can't be reintroduced for the next incremental clause. This is the
  **cheapest** win and needs **no new dependency**: just run the *first* solve one-shot.
- **gimsatul is a genuinely stronger engine — but not uniformly.** Both it and one-shot CaDiCaL eliminate the same
  ~18.7k vars on 128, yet gimsatul's post-preprocessing search is ~45× faster there. On 144, however, **single-thread
  gimsatul times out (>420s) while one-shot CaDiCaL finishes in 113s** — gimsatul only wins on 144 *with* its 16-way
  clause sharing (69s), and even then only ~1.6× over one-shot CaDiCaL. So the external-solver win is large but
  instance-dependent; it is not a blanket replacement.
- **Multithreading is non-monotonic.** Easy sizes: 1 thread is fastest (threads are pure overhead). 128: 8 threads
  best. 144: 16 threads best (1/8 time out), 32 degrades. Sweet spot ≈ 8–16.
- **Contrast with the incremental seed portfolio (13.7s on 128, already built as `TT_TOPO_SAT_PORTFOLIO`).** That
  races 16 *incremental* CaDiCaL seeds and beats a single incremental solve, but it is still incremental (no full
  BVE), so it loses to one-shot gimsatul (0.19s) and even to a one-shot approach on 128.

## Feature preservation — does this regress preferred / pinnings / host-min?

Read of `topology_solver_sat.cpp` (main) confirms **no soft/MaxSAT anywhere** on the solve path:

- **Preferred constraints:** `append_preferred_hit_indicators` → DFS/greedy lower bound `k_lb`
  (`topology_sat_preferred_exact_lower_bound`) → `topology_sat_add_at_least_k_literals(...)` encodes "≥k preferred
  hits" as a **hard cardinality clause** → single `solve_with_symmetry_break`. It is one hard CNF.
- **Required pinnings:** hard clauses in `encode_hard_constraints`.
- **Relaxed channel thresholds:** hard indicator literals, appended before the same solve.

⇒ A one-shot SAT solver (our CaDiCaL one-shot, or gimsatul) solving the dumped CNF reproduces **all** of these
exactly — no feature regression for finding a feasible solution. The genuinely *incremental* pieces are only:

- **Multi-solution enumeration** (`topology_sat_search_n`): one solver + blocking clauses in a loop.
- **Goal-2 host minimization**: `solve_limited` conflict-capped descent over tightening host budgets.

## Implementation plan (staged; each independently shippable, no regression)

**Tier 0 — one-shot first solution in our own CaDiCaL (cheapest, no new dependency).**
- For the *first* feasible solution, use a non-incremental CaDiCaL (don't set `ilb`; let full preprocessing run), then
  hand the found model to the existing incremental solver as the seed of enumeration.
- Wins ~5–8× on the hard cases (205→43.7s, 910→113s) with zero external code. Lowest risk; do this first.

**Tier 1 — multithreaded one-shot (portfolio of one-shot CaDiCaL, in-process).**
- Reuse the already-built Design-B thread portfolio (`TT_TOPO_SAT_PORTFOLIO`), but each worker runs **one-shot**
  (full BVE) instead of incremental. Race N seeds; first-to-SAT wins; the rest are cancelled via the existing
  cancel-flag terminator. Uses the idle MPI cores (`OMPI_MCA_mpi_yield_when_idle=1`). Multithreaded **and** keeps all
  features (each worker solves the identical full hard CNF). Expected to beat 13.7s on 128 substantially.

**Tier 2 — external clause-sharing solver (gimsatul) via DIMACS round-trip (biggest single win on 128).**
- Encode → `write_dimacs` → run `gimsatul --threads=8..16` as a subprocess → parse the `v` model back → decode with
  the existing `topology_sat_decode_hard_solution` (variable numbering already round-trips). Little code, no library
  linkage. Gate behind an env flag; fall back to Tier 0/1 if the binary is absent or on the instances where gimsatul
  loses (e.g. single-thread 144).
- Caveat: gimsatul is **one-shot only** (no IPASIR incremental). So it can produce the *first* solution but cannot run
  the enumeration/host-min loops itself.

**How incremental + multi-solution + host-min still work with all tiers.**
- **First solution:** one-shot (Tier 0/1/2) — fast, feature-complete.
- **Enumeration (more solutions):** keep on incremental CaDiCaL. Load the CNF, add the first solution's model as a
  blocking clause, and continue `solve()`ing — exactly `topology_sat_search_n` today, just seeded from the one-shot
  model instead of finding #1 itself.
- **Host minimization (Goal 2):** each tightening step is its own hard CNF (host budget = k). Solve each step
  one-shot (re-encode is ~0.1s) or keep the conflict-capped incremental descent; independent of Tier 0–2.
- **Preferred / pinnings:** already in the dumped CNF (proven above) — nothing extra to do.

## Risks / open items

- gimsatul dependency: extra binary to build/ship; one-shot only; instance-dependent win. Recommend Tier 0 first,
  Tier 2 as an opt-in accelerator.
- Verify decode correctness of an externally-produced model end-to-end (round-trip a gimsatul `v`-line model through
  `topology_sat_decode_hard_solution` and validate the mapping) before trusting Tier 2 in production.
- Preferred-inclusive CNF was validated by code reading; still worth an empirical dump on a *pinned* MGD (the 2x4
  pipeline sweep has no pinnings) to confirm the at-least-k clauses appear in the DIMACS.
- All numbers are the SC36 mock on this host; re-confirm on target hardware.

## Reproduce

```bash
# 1. build with the dump hook
cmake --build build --target generate_rank_bindings
cp -f build/tt_metal/libtt_metal.so build/lib/libtt_metal.so

# 2. dump a base-embedding CNF (per-rank mock cluster descs; NO_MINHOST = base embedding)
#    TT_TOPO_SAT_DUMP_DIMACS=<path> encodes then writes the CNF and aborts the solve.
#    (see scripts/dump_cnf.sh for the 36-rank mpirun invocation)

# 3. one-shot standalone CaDiCaL (same version we link), built from .cpmcache:
#    scripts/build_cadical.sh ; build/cadical <cnf>

# 4. gimsatul (native clause-sharing), built from github v1.1.3:
#    scripts/build_gimsatul.sh ; gimsatul <cnf> --threads=8

# 5. sweeps: scripts/gim_sweep.sh , scripts/cad_sweep.sh
```

---

# UPDATE — the host-cap reality check (this is the important part)

Everything above is the BASE embedding (host minimization OFF, `NO_MINHOST`). That is the *easy* problem. The
production solve applies a **host cap** (minimize/limit hosts used). Re-running the whole comparison with the cap on
reverses the conclusion.

## Engine map (what runs what — avoid conflating these)
| measurement | engine | notes |
|---|---|---|
| the ablation table below | **our CaDiCaL**, in-solver, incremental | env-knob configs of our own solver; cap ON |
| base + with-cap gimsatul rows | **gimsatul** (external) | standalone binary on an *exported static CNF* — NO infrastructure, NO descent |
| one-shot CaDiCaL rows | **standalone CaDiCaL** (external) | vendored binary, one-shot, on exported CNF |

## CNF-capture bug + fix (prerequisite for any with-cap export)
`CaDiCaL::write_dimacs` silently **drops clauses added incrementally after a `solve()`** — so the exported hardcap
CNF came out byte-identical to the base CNF (the occupancy/cap clauses were missing). Fixed with a **clause tee** in
the wrapper (record every `add()` literal when `TT_TOPO_SAT_DUMP_DIMACS` is set; emit DIMACS from the tape). Also:
setting `target`/`phase` (fastsat) **after** encode aborts CaDiCaL (SIGABRT) — must be set pre-encode. (Code on
branch `ridvan/gen-rank-bindings-all-solutions`.)

## 2x4-128 WITH host cap (occupied ≤ 32; 178,426 clauses) — the reversal
| solver | time |
|---|---|
| one-shot standalone CaDiCaL | **> 400s (timeout)** |
| gimsatul t8 | **> 900s (timeout)** |
| gimsatul t16 | 530s |
| **gimsatul t32** | **67s** ✅ |
| incremental warm-descent (our prod) | ~22 min |

The base-embedding win does **not** transfer cleanly: cold one-shot / low-thread solvers *fail* on the capped
problem. gimsatul only wins at **t32** (67s, ~20× vs incremental), and it is fragile / non-monotonic. The
incremental warm-descent does real work cold solving can't replicate at low thread counts.

## In-solver optimization ablation (our CaDiCaL; host cap ON; ONE solve each)
Each column toggles one optimization vs the lean cold-cap baseline. Time = inter-mesh solve seconds. (`mode3` = cold
single capped solve; `mode1` = warm feasible + cap lock; `mode0` = warm + soft descent + cap lock, the prod default;
`+fastsat` = CaDiCaL target=2,phase=1; `+seed7` = fixed seed, a portfolio-spread proxy.) Sweep in progress — cells
fill over hours; `…` = still running.

| shape·stages | mode3 cold cap | mode1 warm+lock | mode0 warm+**descent**+lock | mode3+fastsat | mode3+seed7 |
|---|--:|--:|--:|--:|--:|
| 2x4·64  | **11.9s** | 17.2s | 240s | 44.5s | 17.3s |
| 2x4·96  | **32.6s** | … | … | 176s | … |
| 2x4·112 | … | … | … | … | … |
| 2x4·128 | … | … | … | … | … |
| 4x4·32  | … | … | … | … | … |
| 4x4·48  | … | … | … | … | … |
| 4x4·64  | … | … | … | … | … |

How to read: **mode0−mode1** = the soft descent's cost/benefit; **mode1−mode3** = the warm solve's; **+fastsat/+seed
− mode3** = those knobs. Early signal (2x4·64, ·96): **cold cap (mode3) is the clear winner**, the descent is a big
overhead on dense packings (240s vs 11.9s on 64), and **fastsat HURTS** the capped solve (44.5s / 176s vs 11.9s /
32.6s). All `occupied` land on the minimal host count (64→16, 96→24), i.e. correct.

Not columns here (by design): **preferred at-least-k** (pipeline MGDs have no pinnings — measured separately later
as the cost of adding it); **multithread portfolio & variable-elimination** (not wired into the cap path — that's
the hybrid).

## Next: hybrid integration (in progress)
Gimsatul standalone can't keep the infrastructure (no incremental, no descent, no enumeration, no assumptions). The
hybrid keeps our CaDiCaL driving (encode / occupancy / cap / descent / decode / preferred / symmetry / enumeration)
and delegates each heavy SAT `solve()` to gimsatul (export tape + current assumption units → subprocess → parse
model → import). Goal: measure *integrated* speed with warm solve + all features on. Being built on
`ridvan/gen-rank-bindings-all-solutions`.

---

# COMPREHENSIVE sweep — CaDiCaL ablation vs gimsatul HYBRID (integrated, host cap ON)

Full sweep 2026-07-30. All runs go through the real solve path with the **host cap applied**, ONE solve each, time =
inter-mesh solve seconds, `occupied` = minimal host count reached (all correct). The gimsatul rows are the **HYBRID**:
our CaDiCaL drives encode/occupancy/cap/descent/decode and delegates each heavy SAT solve to gimsatul (export tape +
assumption units → subprocess → import model). So these are *integrated* numbers with all features intact, not
standalone-on-a-flat-CNF. `gim_first` = gimsatul does only the warm feasible solve, then CaDiCaL runs the incremental
descent warm-started from gimsatul's model (phase hints). Per-cell cap 1500s (TIMEOUT = did not finish in 25 min).

## Table A — CaDiCaL ablation (2x4, seconds; host cap on)
| stages | mode3 cold cap | mode1 warm+lock | mode0 warm+descent | mode3+fastsat | mode3+seed7 |
|---:|--:|--:|--:|--:|--:|
| 64  | 11.4 | 17.5 | 224.2 | 48.3 | 19.2 |
| 96  | 32.6 | 190.9 | 839.7 | 186.0 | 630.1 |
| 112 | 988.5 | 345.2 | 726.0 | 519.7 | 1459.6 |
| 128 | 1345.6 | TIMEOUT | TIMEOUT | 855.8 | TIMEOUT |

## Table B — gimsatul HYBRID (2x4, seconds; host cap on; thread-count sweep)
| stages | gim t8 | gim t16 | gim t32 | gim_first t32 | best hybrid | best CaDiCaL | hybrid speedup |
|---:|--:|--:|--:|--:|--:|--:|--:|
| 64  | 5.6 | 9.6 | 12.6 | 39.8 | **5.6** (t8) | 11.4 | 2.0× |
| 96  | 225.2 | 114.0 | 52.9 | 444.1 | 52.9 (t32) | **32.6** | 0.6× (CaDiCaL wins) |
| 112 | 606.7 | 125.7 | 310.8 | 209.9 | **125.7** (t16) | 345.2 | 2.7× (vs best CaDiCaL) / 7.9× vs cad_mode3 |
| 128 | 1283.9 | 515.1 | **72.9** | TIMEOUT | **72.9** (t32) | 855.8 | 11.7× (vs best CaDiCaL) / 18.5× vs cad_mode3 |

## Table C — 4x4 (all configs ≈ 0.0–0.2 s)
Every 4x4 case (32/48/64) solved in <0.2 s with `occupied == stages` (32→32, 48→48, 64→64). The cap is **not
binding** — a 4x4 mesh fills a whole host, so there is no packing to do (trivial exact-fit). 4x4 is therefore not a
useful stress case for host minimization; the interesting regime is 2x4 partial fills.

## Findings
1. **The integrated hybrid wins big on the HARD cases** (where it matters): 112 → 125.7s (gim t16) vs 988.5s
   (cad_mode3), **7.9×**; 128 → **72.9s (gim t32) vs 1345.6s (cad_mode3), 18.5×** — and 128 is the case that
   motivated the whole exercise. Target "2x4-128 with host cap well under 30 min" is met at **~1.2 min**, integrated,
   all features intact (occupied=32 correct).
2. **gimsatul is instance-dependent and NOT a blanket win.** On 96, pure CaDiCaL cold cap (32.6s) beats every hybrid
   config. gimsatul only pays off once the capped solve gets genuinely hard (112, 128).
3. **Best thread count is non-monotonic and instance-specific:** 64→t8, 96→t32, 112→t16, 128→t32. No single count
   dominates — a production integration would need to race a couple counts or pick adaptively.
4. **`gim_first` (gimsatul warm + CaDiCaL descent) is not the answer for the hard cases** — it TIMED OUT on 128,
   because the *descent itself* is intractable there (cad_mode0/mode1 also timed out). Warm-starting CaDiCaL from
   gimsatul's model doesn't rescue an intractable descent. It only helps mid-range (112: 209.9s).
5. **Among CaDiCaL configs, cold cap (mode3) is the most robust**; the warm descent (mode0) is consistently the worst
   and times out at 128. **fastsat is erratic** — it *hurts* at 64/96 but *helps* at 128 (855.8s vs 1345.6s). **seed
   choice is high-variance** (seed7 times out at 128 while default solves in 1345.6s).
6. **Bottom line:** the best single strategy for the hard host-cap solve is **cold-cap delegated to gimsatul at a
   high thread count** (gim_mode3_t32) — 18× on 128 — but it must fall back to / race pure CaDiCaL cold cap on the
   mid stages where gimsatul loses. The hybrid keeps every feature (preferred, symmetry, enumeration all remain on
   CaDiCaL; only the heavy solve is delegated).

Raw data: `oneshot_external_sat_scripts/RESULTS_full_sweep.txt`. Hybrid code (gimsatul_solve / delegated_solve /
phase_hint_from_last_gimsatul_model, env TT_TOPO_SAT_GIMSATUL[_BIN|_THREADS] / TT_TOPO_SAT_GIM_FIRST) is on branch
`ridvan/gen-rank-bindings-all-solutions` (depends on the min-host solver, which is not on main).

---

# ENUMERATION (search_n / -n N) experiments — the incremental / multi-solution workload

The single-solve results above are for finding ONE min-host mapping. The real workload also enumerates N distinct
solutions (`generate_rank_bindings -n N`, implies `--all-solutions`). This path **primes** with a min-host solve
(`solve_minimize_groups` — where the gimsatul delegation lives), then enumerates `.next` on incremental CaDiCaL with
blocking clauses. gimsatul CANNOT do the `.next` steps (no incremental API), so it only accelerates the prime.

## Enumeration to 5 solutions (2x4, host cap on, wall seconds)
| stages | enum_hardcap (CaDiCaL-only) | +fastsat | gim_prime_t32 (hybrid) | warmdescent |
|---:|--:|--:|--:|--:|
| 64 | 233 | 220 | 366 (slower) | 374 |
| 96 | 292 | 281 | TIMEOUT@800s | TIMEOUT@800s |

## Per-solution curve (64, time from enum start) — enumeration is SUPER-LINEAR
| config | to 3 sols | to 5 sols | tail (sols 4-5) |
|---:|--:|--:|--:|
| hardcap | 49s | 233s | +184s |
| hardcap+fastsat | 55s | 220s | fastsat helps the tail |
| gim_prime | 280s | 366s | (cold enumerator) |

### Findings (enumeration)
- **CaDiCaL-only iterative BEATS the gimsatul hybrid for enumeration.** On 64: 233s vs 366s to 5 sols; on 96 the
  hybrid TIMES OUT while CaDiCaL-only finishes in 292s. Reason: an externally-solved (gimsatul) prime leaves the
  CaDiCaL enumerator COLD (no learned clauses to reuse for `.next`), so every subsequent solution is slow. A CaDiCaL
  prime warms the enumerator.
- **Enumeration cost is super-linear** (64: 3 sols=49s, 5 sols=233s) — later distinct solutions get much more
  expensive as blocking clauses accumulate and genuinely-new mappings get rarer.
- **fastsat helps the incremental path ~5%**, concentrated on the expensive tail. It applies across every
  incremental solve (set pre-encode; persists). It is the ONLY optimization that helped enumeration here.
- **Our incremental solve already eliminates ~35% of vars** ("35.0% pinned" in logs) — incremental is NOT starved of
  BVE on this path, so external pre-elimination buys less than the one-shot-vs-incremental gap implied. External BVE
  did reduce the 128 hardcap CNF by 34.5% (cadical -P20 -c0 -o), but to stay incremental you must FREEZE the assign
  vars (blocking clauses reference them), limiting BVE to auxiliaries during enumeration.

## Partitioned-parallel (pin stage-0) proof-of-concept — NEGATIVE for single hard solve
Hypothesis: split a hard solve into disjoint sub-spaces by pinning stage-0's placement (vars 1..144, the first
exactly-one group), solve buckets in parallel with plain CaDiCaL → rival gimsatul while staying pure-CaDiCaL.
Result on 128 hardcap (18 of 144 buckets sampled, plain cadical, parallel): **17/18 TIMEOUT >400s, 1 SAT@390s**.
- **Pinning ONE stage of 128 barely reduces difficulty** — the residual packing problem is ~as hard as the original.
  So partition-by-pinning-one-variable does NOT make a single hard solve fast. gimsatul remains the single-solve win.
- (Caveat: the 400s per-bucket cap was below the ~1345s the full in-solver solve needs, so this shows "pinning
  doesn't rescue it within 400s", not a clean "slower than gimsatul".)
- Partitioning is a DIFFERENT mechanism for ENUMERATION (splitting the work of finding MANY solutions), which this
  PoC does not test. Risks there: skew across buckets, and partition-axis vs distinct-host-set dedup mismatch
  (cross-thread duplicate solutions). Still open.

## gim-first enumeration on the PRODUCTION-DEAD cases (RUNNING)
Production baseline finds **0 solutions** on 2x4 96/112/128/144 (can't find even 1 in the 15-min search budget).
gimsatul CAN find the first solution (128 ~73s, 144 ~105s). Test: MIN_MODE=3 + GIMSATUL=1 + `-n 20` — gimsatul
primes, CaDiCaL enumerates — to see if it unlocks these dead cases. Results pending (see
oneshot_external_sat_scripts/RESULTS_gimfirst_enum.txt).

## Bottom line (two workloads, two engines)
- **Single min-host solve:** gimsatul hybrid wins big (128: 18.5×; 144: solvable where CaDiCaL times out).
- **Enumeration / iterative:** stay on CaDiCaL-only + fastsat; the gimsatul prime leaves the enumerator cold and
  loses. The gimsatul value for enumeration is narrow: only to get PAST the "can't find solution 1" wall on the
  hardest cases (96-144) where production currently returns 0 — under test now.
