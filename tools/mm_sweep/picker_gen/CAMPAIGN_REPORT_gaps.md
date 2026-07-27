# Generator-driven campaign: optimum gaps + pruning audit

141 shapes measured (results_v2). Per-config wall = median over ok samples (initial + rerun).

## Production picker gap vs measured optimum

| subset | n | geomean | median | worst | >3% | >5% |
|---|---|---|---|---|---|---|
| ALL | 141 | 7.8% | 4.1% | 53.5% | 75 | 63 |
| train | 83 | 9.7% | 5.8% | 53.5% | 47 | 44 |
| val | 29 | 7.6% | 4.4% | 50.4% | 17 | 13 |
| holdout | 29 | 3.1% | 1.4% | 17.2% | 11 | 6 |
| fluxltx | 42 | 1.0% | 0.0% | 9.9% | 6 | 3 |

### Largest production gaps

| shape | split | Mt | gap | prod cfg | opt cfg |
|---|---|---|---|---|---|
| 224x8192x1536 | train | 7 | 53.5% | [3, 4, 1, 2, 1] | [1, 11, 1, 3, 2] |
| 128x7168x1280 | train | 4 | 51.5% | [1, 10, 1, 1, 1] | [1, 7, 1, 4, 2] |
| 192x8192x1536 | val | 6 | 50.4% | [3, 4, 1, 2, 1] | [1, 8, 1, 2, 3] |
| 256x8192x1536 | train | 8 | 46.5% | [3, 4, 1, 2, 1] | [1, 11, 1, 3, 1] |
| 160x8192x1536 | train | 5 | 44.6% | [3, 4, 1, 2, 1] | [1, 8, 1, 2, 3] |
| 32x7168x1280 | train | 1 | 39.6% | [1, 10, 1, 1, 2] | [1, 7, 1, 4, 1] |
| 256x4608x3072 | train | 8 | 35.6% | [4, 3, 1, 2, 1] | [2, 3, 2, 2, 6] |
| 128x8192x1536 | train | 4 | 33.1% | [3, 4, 1, 2, 1] | [1, 8, 1, 2, 3] |
| 256x7168x1280 | train | 8 | 32.8% | [1, 10, 1, 1, 1] | [1, 4, 3, 7, 5] |
| 256x7680x4608 | train | 8 | 31.6% | [3, 4, 1, 2, 1] | [1, 5, 2, 2, 6] |
| 96x8192x1536 | train | 3 | 26.4% | [3, 4, 1, 2, 1] | [1, 8, 1, 2, 2] |
| 128x4608x3072 | train | 4 | 24.3% | [4, 3, 1, 2, 1] | [1, 9, 1, 2, 2] |

## Pruning audit

audit-beats-structured: **0/141**; optimum-is-audit: **0/141**.

Winner rerun spread: median 0.73%, max 4.00%. Winner PCC min 0.99998.
