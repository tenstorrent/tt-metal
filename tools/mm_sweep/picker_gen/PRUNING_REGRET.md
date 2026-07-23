# Pruning-regret vs aborted exhaustive shapes

budget=96 audit=8; 11 exhaustive shapes.

| shape | split | Mt | measured | selected | overlap | opt retained | regret | within-2% kept |
|---|---|---|---|---|---|---|---|---|
| 96x6144x1536 | val | 3 | 811 | 96 | 79 | N | +2.36% | 0/3 |
| 64x2048x2048 | train | 2 | 916 | 96 | 83 | N | +1.99% | 1/3 |
| 32x6144x1536 | train | 1 | 464 | 96 | 76 | N | +1.31% | 1/4 |
| 256x2048x1024 | train | 8 | 683 | 96 | 83 | Y | +0.00% | 1/1 |
| 256x2048x512 | train | 8 | 312 | 96 | 81 | Y | +0.00% | 1/1 |
| 256x15360x1536 | train | 8 | 324 | 96 | 77 | Y | +0.00% | 1/1 |
| 192x6144x1536 | train | 6 | 910 | 96 | 75 | Y | +0.00% | 1/1 |
| 64x6144x1536 | train | 2 | 680 | 96 | 82 | Y | +0.00% | 1/4 |
| 224x6144x512 | train | 7 | 268 | 96 | 73 | Y | +0.00% | 1/1 |
| 128x2048x2048 | val | 4 | 1220 | 96 | 82 | Y | +0.00% | 1/1 |
| 128x15360x768 | val | 4 | 333 | 96 | 75 | Y | +0.00% | 1/1 |

**optimum retained 8/11; all-within-2% retained 7/11**; median regret +0.00%, worst +2.36%.
