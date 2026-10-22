# Timestral optimization candidates

Headline: `predict_block`, paper shape (1024 x 2048 -> 64), accuracy precision, median of 5 steady runs.
Guards: `long_context` and `multivariate` (accuracy) and paper with the `performance()` precision.

| round | candidate | paper wall ms | paper device ms | vs best | guards | PCC (norm) | decision | evidence |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline (default 1D matmul configs, unfused activations) | 1445.3 | 1307.2 | — | long 463.4, multi 1797.5, perf 1353.0 | 0.99995 | baseline | iter_00/summary.md |
| 1 | P1: 2D mcast configs (activation fused), `minimal_matmul` for N = 768 without activation | 836.4 | 702.0 | −42.1 % | long 346.0 (−25.3 %), multi 1209.1 (−32.7 %), perf 816.9 (−39.6 %) | 0.99993 | **accept** | iter_01/summary.md, iter_01/matmul_sweep_*.md |
