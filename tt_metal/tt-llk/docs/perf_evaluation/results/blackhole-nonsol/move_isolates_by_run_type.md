# Isolates, split per run type -- Blackhole, non speed-of-light

Each run type is a separate ELF and a separate device run.
`move` = (largest of the 5 runs - smallest) / median.

| run type | marker | measured | >0.5% | >1% | >2% | worst move | worst cycles |
|---|---|--:|--:|--:|--:|--:|--:|
| MATH_ISOLATE | INIT | 33,177 | 1,656 | 979 | 555 | 5.15% | 11 |
| MATH_ISOLATE | KERNEL | 33,177 | 7 | 0 | 0 | 0.93% | 17 |
| MATH_ISOLATE | TILE_LOOP | 33,177 | 14 | 3 | 0 | 1.47% | 12 |
| MATH_ISOLATE | UNINIT | 1,649 | 114 | 114 | 114 | 14.29% | 6 |
| PACK_ISOLATE | INIT | 35,573 | 1,873 | 1,216 | 287 | 4.73% | 21 |
| PACK_ISOLATE | KERNEL | 35,573 | 12 | 3 | 1 | 14.12% | 349 |
| PACK_ISOLATE | TILE_LOOP | 35,573 | 2 | 1 | 1 | 19.75% | 353 |
| PACK_ISOLATE | UNINIT | 29 | 0 | 0 | 0 | 0.00% | 0 |
| UNPACK_ISOLATE | INIT | 33,925 | 509 | 169 | 19 | 2.61% | 8 |
| UNPACK_ISOLATE | KERNEL | 33,925 | 7 | 2 | 0 | 1.21% | 98 |
| UNPACK_ISOLATE | TILE_LOOP | 33,925 | 5 | 0 | 0 | 0.88% | 98 |
| UNPACK_ISOLATE | UNINIT | 1,649 | 108 | 108 | 108 | 3.85% | 1 |
