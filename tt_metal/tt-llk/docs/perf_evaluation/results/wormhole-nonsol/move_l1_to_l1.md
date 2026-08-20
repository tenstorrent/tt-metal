# How much do results move between identical runs? -- Wormhole, L1_TO_L1, non-SoL

Same code, same machine, 5 runs.
`move` = (largest of the 5 runs - smallest) / median.

| marker | measured | >0.5% | >1% | >2% | >5% | worst move | worst cycles |
|---|--:|--:|--:|--:|--:|--:|--:|
| INIT | 33,657 | 8,776 | 3,882 | 742 | 2 | 5.65% | 27 |
| KERNEL | 33,657 | 176 | 61 | 26 | 0 | 4.60% | 9165 |
| TILE_LOOP | 33,657 | 186 | 64 | 27 | 0 | 4.62% | 9165 |
| all | 100,971 | 9,138 | 4,007 | 795 | 2 | 5.65% | 9165 |

## Rule check: >2% AND >30 cycles

Fires on **53** of 100,971 measurements of unchanged code.

| test | marker | run type | median | min | max | move | cycles |
|---|---|---|--:|--:|--:|--:|--:|
| perf_math_matmul | KERNEL | L1_TO_L1 | 124027 | 120490 | 124028 | 2.85% | 3538 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 120262 | 120261 | 123064 | 2.33% | 2803 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 119663 | 119663 | 123099 | 2.87% | 3436 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 124027 | 120490 | 124028 | 2.85% | 3538 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 123063 | 120258 | 123065 | 2.28% | 2807 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 119663 | 119661 | 123099 | 2.87% | 3438 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 124154 | 121492 | 124155 | 2.14% | 2663 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 166619 | 160663 | 166619 | 3.57% | 5956 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 166619 | 161280 | 166641 | 3.22% | 5361 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 119119 | 119119 | 123184 | 3.41% | 4065 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 154632 | 154629 | 159461 | 3.12% | 4832 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 154377 | 154377 | 159446 | 3.28% | 5069 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 154632 | 154632 | 159463 | 3.12% | 4831 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 197400 | 191862 | 197417 | 2.81% | 5555 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 185125 | 185101 | 191178 | 3.28% | 6077 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 194730 | 185774 | 194733 | 4.60% | 8959 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 191319 | 191319 | 195212 | 2.03% | 3893 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 212231 | 211075 | 218430 | 3.47% | 7355 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 211346 | 211079 | 216194 | 2.42% | 5115 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 220117 | 217338 | 222307 | 2.26% | 4969 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 215789 | 211088 | 220253 | 4.25% | 9165 |
| perf_math_matmul | KERNEL | L1_TO_L1 | 212925 | 211075 | 219674 | 4.04% | 8599 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 123266 | 119730 | 123267 | 2.87% | 3537 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 119494 | 119493 | 122298 | 2.35% | 2805 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 118915 | 118915 | 122356 | 2.89% | 3441 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 123266 | 119730 | 123267 | 2.87% | 3537 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 122297 | 119490 | 122299 | 2.30% | 2809 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 118915 | 118915 | 122356 | 2.89% | 3441 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 123386 | 120724 | 123387 | 2.16% | 2663 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 118355 | 118355 | 120738 | 2.01% | 2383 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 165874 | 159918 | 165874 | 3.59% | 5956 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 165874 | 160535 | 165896 | 3.23% | 5361 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 118371 | 118371 | 122436 | 3.43% | 4065 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 153863 | 153863 | 158698 | 3.14% | 4835 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 153629 | 153629 | 158704 | 3.30% | 5075 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 153863 | 153863 | 158698 | 3.14% | 4835 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 196643 | 191104 | 196660 | 2.83% | 5556 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 184371 | 184348 | 190424 | 3.30% | 6076 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 193975 | 185022 | 193978 | 4.62% | 8956 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 190568 | 190568 | 194464 | 2.04% | 3896 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 211498 | 210344 | 217697 | 3.48% | 7353 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 210607 | 210339 | 215455 | 2.43% | 5116 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 219382 | 216603 | 221572 | 2.26% | 4969 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 215056 | 210355 | 219520 | 4.26% | 9165 |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 212186 | 210336 | 218935 | 4.05% | 8599 |
| perf_matmul | KERNEL | L1_TO_L1 | 23522 | 23014 | 23523 | 2.16% | 509 |
| perf_matmul | KERNEL | L1_TO_L1 | 16747 | 16747 | 17179 | 2.58% | 432 |
| perf_matmul | KERNEL | L1_TO_L1 | 44843 | 44150 | 45186 | 2.31% | 1036 |
| perf_matmul | KERNEL | L1_TO_L1 | 206669 | 203233 | 207586 | 2.11% | 4353 |
| perf_matmul | TILE_LOOP | L1_TO_L1 | 22902 | 22406 | 22903 | 2.17% | 497 |
| perf_matmul | TILE_LOOP | L1_TO_L1 | 16141 | 16141 | 16586 | 2.76% | 445 |
| perf_matmul | TILE_LOOP | L1_TO_L1 | 44257 | 43563 | 44607 | 2.36% | 1044 |
| perf_matmul | TILE_LOOP | L1_TO_L1 | 206063 | 202627 | 206992 | 2.12% | 4365 |
