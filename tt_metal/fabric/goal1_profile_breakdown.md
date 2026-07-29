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
