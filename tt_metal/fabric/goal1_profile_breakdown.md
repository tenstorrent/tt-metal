# Goal-1 base-embedding A/B — profile breakdown (2x4-128, TT_TOPO_SAT_NO_MINHOST=1)

Total inter-mesh solve split into encode / warm-hint / SAT-solve. baseline = plain base embedding (~55k vars,
~157k clauses; n_target=128 n_global=144). CAVEAT: earlier variants ran under CPU contention with a concurrent
144 test — a clean re-run of the winners is needed before final claims.

| variant | total (intermesh) | encode | warm-hint (hints) | SAT solve | hosts | speedup vs baseline |
|---|---|---|---|---|---|---|
| baseline | 205.3s | 113ms | - | 205.2s | 36 | 1.0x |
| seed7 | 140.9s | 100ms | - | 140.8s | 36 | 1.5x |
| seed42 | 32.0s | 99ms | - | 31.9s | 36 | 6.4x |
| fastsat | 62.8s | 98ms | - | 62.7s | 36 | 3.3x |
| warmhint | 130.2s | 99ms | 0ms (120) | 130.1s | 36 | 1.6x |
| warmhint+fastsat | 167.7s | 100ms | 0ms (120) | 167.6s | 36 | 1.2x |
| warmhint+seed7 | 36.7s | 58ms | 0ms (120) | 36.6s | 36 | 5.6x |

## Bottleneck
- Encode is negligible (~0.1s of the whole solve); adjacency_support is the biggest encode piece (~65ms) but still tiny.
- **The SAT solve dominates (>99%)** — that's the only thing the knobs move.
- Seed variance is large (see seed7 vs seed42) => a seed **portfolio** (run k seeds, take first) is a cheap win.

Raw per-variant profiles: scratchpad/G1_<variant>/log ([topo-sat-profile] lines).

## Seed variance + portfolio (2x4-128 base embedding, clean/sequential)
| seed | solve | seed | solve |
|---|---|---|---|
| 3 | 33s | 10 | 68s |
| 21 | 35s | 2 | 72s |
| 5 | 42s | 99 | 64s |
| 42 | 32s | 1 | 91s |
| 7 | 141s | 0 | 128s |
| default(baseline) | 205s | | |

min=33s median=66s max=128s (plain seeds). seed99+fastsat=21s (best; fastsat compounds on a good seed).
=> NOT a fluke: nearly every seed beats the 205s default. **Seed portfolio** (run k seeds, take first) has
expected wall ~= min ~= 33s (6x), ~21s with fastsat. This is the recommended early win.

## Multithreading: parallel seed portfolio — optimal K (multi-PROCESS)
Wall-to-first-solution vs number of concurrent 36-rank producers (2x4-128 base embedding, fixed fast seed set):
| K (producers) | total ranks | wall-to-first |
|---|---|---|
| 1 | 36 | 38s |
| 2 | 72 | 66s |
| 3 | 108 | 75s |
| 4 | 144 | 82s |
| 6 | 216 | 141s |

Optimal K = 1. It gets WORSE with K, not better:
(a) each producer needs 36 ranks (one per SC36 host) and the 35 idle ranks BUSY-POLL the post-solve barrier,
    so K*36 > 64 cores oversubscribes and starves the actual solvers (K=6 -> load 151);
(b) so the practical multi-process optimum is K=1 (no hedging benefit once you already include a fast seed);
(c) to scale the portfolio properly, eliminate the idle-rank overhead — single-rank producers (needs a combined
    mock descriptor) OR an IN-PROCESS thread portfolio (one MPI job, N CaDiCaL solver threads, distinct seeds,
    first-to-SAT cancels the rest via terminate()).

RESOLVED: the in-process thread portfolio (Design B) was implemented and beats all of the above —
16-way + fastsat = 13.7s on 2x4-128 (vs 205s), reusing the single producer's free cores (36 ranks + N threads
<= 64, no oversubscription). See PARALLEL_PORTFOLIO_DESIGN.md.

## Portfolio width sweep (N) — how big a portfolio helps (2x4-128, Design B + fastsat)
Idle ranks quieted via env `OMPI_MCA_mpi_yield_when_idle=1` (this env form works; the `--mca` form errored), so
peak load << 36+N — i.e. NOT core-bound.
| N workers | ranks | winner solve | wall | peak load |
|---|---|---|---|---|
| 8  | 44 | 24.4s | 39s | 18 |
| 16 | 52 | **13.1s** | 27s | 25 |
| 24 | 60 | 14.3s | 28s | 33 |
| 32 | 68 | 18.8s | 33s | 42 |
| 48 | 84 | 20.3s | 36s | 51 |

Inverted-U: best at N~16 (~13s), then DEGRADES. Not compute-bound (load 51 < 64 at N=48). Two ceilings:
1. **Seed floor** — the portfolio only picks the luckiest of N seeds, so min-over-N bottoms out at the
   fastest-possible seed (~13s with fastsat); more workers can't go below it.
2. **Memory bandwidth** — many concurrent CaDiCaL solvers thrash shared bandwidth/cache, slowing EACH solve
   (winner 13s@16 -> 20s@48). This is the real ceiling past ~16, independent of core count.

Conclusion: max practical speedup ~15x (205s -> ~13s) at N~16 + fastsat. More cores/workers do NOT help; the
bottleneck is memory bandwidth + the seed floor, not compute. Design A (36 solver ranks) would add slots but
hits the same ceilings; not worth it for this instance. Recommended default: ~12-16 workers + fastsat.
