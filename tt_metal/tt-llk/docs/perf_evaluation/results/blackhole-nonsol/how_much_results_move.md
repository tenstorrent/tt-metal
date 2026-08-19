# How much do results move between identical runs?

Same code, same machine, 5 runs. Blackhole, L1_TO_L1, non speed-of-light.
`move` = (max - min) / median across the 5 runs.

| marker | numbers measured | moved >0.5% | >1% | >2% | >5% | worst move | worst move in cycles |
|---|--:|--:|--:|--:|--:|--:|--:|
| INIT | 35,576 | 2,184 | 1,390 | 464 | 14 | 7.91% | 25 |
| KERNEL | 35,576 | 28 | 7 | 0 | 0 | 1.87% | 5108 |
| TILE_LOOP | 35,576 | 26 | 7 | 0 | 0 | 1.88% | 5110 |
| UNINIT | 1,649 | 91 | 35 | 10 | 0 | 3.57% | 9 |
| **all** | 108,377 | 2,329 | 1,439 | 474 | 14 | 7.91% | 5110 |
