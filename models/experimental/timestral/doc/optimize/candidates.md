# Timestral optimization candidates

Headline: `predict_block`, paper shape (1024 x 2048 -> 64), accuracy precision, median of 5 steady runs.
Guards: `long_context` and `multivariate` (accuracy) and paper with the `performance()` precision.

| round | candidate | paper wall ms | paper device ms | vs best | guards | PCC (norm) | decision | evidence |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline (default 1D matmul configs, unfused activations) | 1445.3 | 1307.2 | — | long 463.4, multi 1797.5, perf 1353.0 | 0.99995 | baseline | iter_00/summary.md |
