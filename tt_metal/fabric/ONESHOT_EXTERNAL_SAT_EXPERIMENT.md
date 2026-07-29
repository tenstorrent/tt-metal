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
