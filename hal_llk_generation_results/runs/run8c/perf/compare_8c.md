# Perf cycles: original vs 8c

original runs: p8c_orig_r1 p8c_orig_r2 p8c_orig_r3; 8c runs: 8c_r1 8c_r2 8c_r3
Values = median over runs of the framework's mean(); TILE_LOOP = cycles per tile, KERNEL/INIT = cycles per zone. ratio = new / original (> 1 = slower).

## TILE_LOOP

| run type | orig median over variants | new median over variants | ratio median | ratio min | ratio max | orig rep spread max % | new rep spread max % |
|---|---|---|---|---|---|---|---|
| L1_TO_L1 | 32.80 | 46.18 | 1.418 | 1.379 | 1.534 | 0.0 | 0.0 |
| UNPACK_ISOLATE | 27.43 | 27.46 | 1.001 | 1.000 | 1.002 | 0.0 | 0.1 |
| MATH_ISOLATE | 16.64 | 16.42 | 0.988 | 0.976 | 0.990 | 0.0 | 0.0 |
| PACK_ISOLATE | 25.85 | 38.97 | 1.504 | 1.459 | 1.620 | 0.2 | 0.2 |
| L1_CONGESTION[UNPACK] | 27.42 | 27.46 | 1.001 | 1.000 | 1.003 | 0.1 | 0.1 |
| L1_CONGESTION[PACK] | 25.89 | 39.89 | 1.537 | 1.474 | 1.663 | 0.2 | 0.0 |

## KERNEL

| run type | orig median over variants | new median over variants | ratio median | ratio min | ratio max | orig rep spread max % | new rep spread max % |
|---|---|---|---|---|---|---|---|
| L1_TO_L1 | 4739.50 | 6454.00 | 1.376 | 1.342 | 1.480 | 0.0 | 0.0 |
| UNPACK_ISOLATE | 4050.00 | 4084.00 | 1.007 | 0.998 | 1.014 | 0.1 | 0.1 |
| MATH_ISOLATE | 2698.00 | 2683.00 | 0.993 | 0.979 | 1.001 | 0.0 | 0.0 |
| PACK_ISOLATE | 3857.50 | 5560.00 | 1.442 | 1.398 | 1.540 | 0.3 | 0.1 |
| L1_CONGESTION[UNPACK] | 4048.50 | 4086.00 | 1.008 | 0.999 | 1.013 | 0.1 | 0.1 |
| L1_CONGESTION[PACK] | 3868.00 | 5670.50 | 1.466 | 1.414 | 1.581 | 0.3 | 0.0 |

## INIT

| run type | orig median over variants | new median over variants | ratio median | ratio min | ratio max | orig rep spread max % | new rep spread max % |
|---|---|---|---|---|---|---|---|
| L1_TO_L1 | 287.00 | 228.00 | 0.797 | 0.728 | 0.854 | 0.0 | 0.0 |
| UNPACK_ISOLATE | 272.00 | 320.00 | 1.231 | 1.176 | 1.232 | 0.0 | 1.4 |
| MATH_ISOLATE | 189.00 | 148.00 | 0.779 | 0.749 | 0.794 | 0.0 | 0.0 |
| PACK_ISOLATE | 296.00 | 252.00 | 0.865 | 0.750 | 0.918 | 1.8 | 3.4 |
| L1_CONGESTION[UNPACK] | 272.00 | 319.00 | 1.228 | 1.173 | 1.232 | 0.0 | 0.9 |
| L1_CONGESTION[PACK] | 296.00 | 242.00 | 0.824 | 0.735 | 0.882 | 1.6 | 0.5 |

## Per variant, TILE_LOOP L1_TO_L1 (cycles per tile)

| variant | orig | new | ratio |
|---|---|---|---|
| Bfp8_b | Bfp8_b | Bfp8_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 31.12 | 46.98 | 1.510 |
| Bfp8_b | Bfp8_b | Bfp8_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 33.73 | 49.02 | 1.453 |
| Bfp8_b | Bfp8_b | Float16 | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.79 | 46.18 | 1.408 |
| Bfp8_b | Bfp8_b | Float16_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 43.72 | 62.43 | 1.428 |
| Bfp8_b | Bfp8_b | Float16_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 46.18 | 1.408 |
| Float16 | Float16 | Bfp8_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 31.09 | 47.69 | 1.534 |
| Float16 | Float16 | Bfp8_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 33.72 | 49.73 | 1.475 |
| Float16 | Float16 | Float16 | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 30.58 | 42.96 | 1.405 |
| Float16 | Float16 | Float16 | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 45.23 | 1.379 |
| Float16 | Float16 | Float16_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 30.81 | 44.95 | 1.459 |
| Float16 | Float16 | Float16_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 45.23 | 1.379 |
| Float16_b | Float16_b | Bfp8_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 31.09 | 47.69 | 1.534 |
| Float16_b | Float16_b | Bfp8_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 33.72 | 49.73 | 1.475 |
| Float16_b | Float16_b | Float16 | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 45.23 | 1.379 |
| Float16_b | Float16_b | Float16_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 30.57 | 42.96 | 1.405 |
| Float16_b | Float16_b | Float16_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 45.23 | 1.379 |
