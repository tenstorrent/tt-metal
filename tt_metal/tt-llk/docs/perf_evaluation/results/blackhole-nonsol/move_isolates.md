# How much do results move between identical runs? -- Blackhole, isolates, non-SoL

Same code, same machine, 5 runs.
`move` = (largest of the 5 runs - smallest) / median.

| marker | numbers measured | moved >0.5% | >1% | >2% | >5% | worst move | worst move in cycles |
|---|--:|--:|--:|--:|--:|--:|--:|
| INIT | 102,675 | 4,038 | 2,364 | 861 | 1 | 5.15% | 21 |
| KERNEL | 102,675 | 26 | 5 | 1 | 1 | 14.12% | 349 |
| TILE_LOOP | 102,675 | 21 | 4 | 1 | 1 | 19.75% | 353 |
| UNINIT | 3,327 | 222 | 222 | 222 | 2 | 14.29% | 6 |
| all | 311,352 | 4,307 | 2,595 | 1,085 | 5 | 19.75% | 353 |
