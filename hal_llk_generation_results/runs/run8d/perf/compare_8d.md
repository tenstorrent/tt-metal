# Perf cycles: original vs 8d

original runs: p8d_orig_r1 p8d_orig_r2 p8d_orig_r3; 8d runs: 8d_r1 8d_r2 8d_r3
Values = median over runs of the framework's mean(); TILE_LOOP = cycles per tile, KERNEL/INIT = cycles per zone. ratio = new / original (> 1 = slower).

## TILE_LOOP

| run type | orig median over variants | new median over variants | ratio median | ratio min | ratio max | orig rep spread max % | new rep spread max % |
|---|---|---|---|---|---|---|---|
| L1_TO_L1 | 32.80 | 41.92 | 1.320 | 1.247 | 1.438 | 0.1 | 0.1 |
| UNPACK_ISOLATE | 27.43 | 26.93 | 0.982 | 0.980 | 0.984 | 0.0 | 0.0 |
| MATH_ISOLATE | 16.64 | 16.44 | 0.988 | 0.977 | 0.990 | 0.0 | 0.0 |
| PACK_ISOLATE | 25.85 | 37.88 | 1.459 | 1.417 | 1.572 | 0.1 | 0.1 |
| L1_CONGESTION[UNPACK] | 27.42 | 26.93 | 0.982 | 0.981 | 0.984 | 0.0 | 0.0 |
| L1_CONGESTION[PACK] | 25.89 | 39.09 | 1.506 | 1.433 | 1.620 | 0.0 | 0.1 |

## KERNEL

| run type | orig median over variants | new median over variants | ratio median | ratio min | ratio max | orig rep spread max % | new rep spread max % |
|---|---|---|---|---|---|---|---|
| L1_TO_L1 | 4739.50 | 5927.50 | 1.286 | 1.228 | 1.395 | 0.1 | 0.1 |
| UNPACK_ISOLATE | 4050.00 | 4020.00 | 0.991 | 0.979 | 0.998 | 0.0 | 0.0 |
| MATH_ISOLATE | 2698.00 | 2685.00 | 0.994 | 0.977 | 1.003 | 0.1 | 0.2 |
| PACK_ISOLATE | 3857.50 | 5427.00 | 1.405 | 1.358 | 1.503 | 0.3 | 0.1 |
| L1_CONGESTION[UNPACK] | 4048.50 | 4018.00 | 0.992 | 0.980 | 0.995 | 0.0 | 0.0 |
| L1_CONGESTION[PACK] | 3868.00 | 5563.50 | 1.438 | 1.378 | 1.543 | 0.0 | 0.1 |

## INIT

| run type | orig median over variants | new median over variants | ratio median | ratio min | ratio max | orig rep spread max % | new rep spread max % |
|---|---|---|---|---|---|---|---|
| L1_TO_L1 | 287.00 | 241.00 | 0.824 | 0.787 | 0.880 | 1.7 | 0.4 |
| UNPACK_ISOLATE | 272.00 | 315.00 | 1.167 | 1.145 | 1.188 | 0.0 | 0.0 |
| MATH_ISOLATE | 189.00 | 122.00 | 0.646 | 0.636 | 0.647 | 0.0 | 2.5 |
| PACK_ISOLATE | 296.00 | 255.00 | 0.847 | 0.789 | 0.891 | 2.3 | 0.0 |
| L1_CONGESTION[UNPACK] | 272.00 | 315.00 | 1.162 | 1.145 | 1.176 | 0.0 | 0.0 |
| L1_CONGESTION[PACK] | 296.00 | 253.00 | 0.851 | 0.829 | 0.902 | 0.0 | 0.0 |

## Per variant, TILE_LOOP L1_TO_L1 (cycles per tile)

| variant | orig | new | ratio |
|---|---|---|---|
| Bfp8_b | Bfp8_b | Bfp8_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 31.12 | 44.17 | 1.420 |
| Bfp8_b | Bfp8_b | Bfp8_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 33.73 | 44.41 | 1.316 |
| Bfp8_b | Bfp8_b | Float16 | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.79 | 41.65 | 1.270 |
| Bfp8_b | Bfp8_b | Float16_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 43.72 | 59.70 | 1.365 |
| Bfp8_b | Bfp8_b | Float16_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 41.65 | 1.270 |
| Float16 | Float16 | Bfp8_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 31.09 | 44.70 | 1.438 |
| Float16 | Float16 | Bfp8_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 33.72 | 45.69 | 1.355 |
| Float16 | Float16 | Float16 | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 30.58 | 40.34 | 1.319 |
| Float16 | Float16 | Float16 | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 40.91 | 1.247 |
| Float16 | Float16 | Float16_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 30.81 | 42.19 | 1.369 |
| Float16 | Float16 | Float16_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 40.91 | 1.247 |
| Float16_b | Float16_b | Bfp8_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 31.09 | 44.70 | 1.438 |
| Float16_b | Float16_b | Bfp8_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 33.72 | 45.69 | 1.355 |
| Float16_b | Float16_b | Float16 | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 40.91 | 1.247 |
| Float16_b | Float16_b | Float16_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 30.57 | 40.34 | 1.320 |
| Float16_b | Float16_b | Float16_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 40.91 | 1.247 |
