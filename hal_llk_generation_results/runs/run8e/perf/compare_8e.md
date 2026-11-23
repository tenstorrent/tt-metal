# Perf cycles: original vs 8e

original runs: p8e_orig_r1 p8e_orig_r2 p8e_orig_r3; 8e runs: 8e_r1 8e_r2 8e_r3
Values = median over runs of the framework's mean(); TILE_LOOP = cycles per tile, KERNEL/INIT = cycles per zone. ratio = new / original (> 1 = slower).

## TILE_LOOP

| run type | orig median over variants | new median over variants | ratio median | ratio min | ratio max | orig rep spread max % | new rep spread max % |
|---|---|---|---|---|---|---|---|
| L1_TO_L1 | 32.80 | 47.70 | 1.449 | 1.407 | 1.586 | 0.0 | 0.0 |
| UNPACK_ISOLATE | 27.43 | 45.88 | 1.672 | 1.671 | 1.674 | 0.0 | 0.0 |
| MATH_ISOLATE | 16.64 | 16.44 | 0.988 | 0.977 | 0.992 | 0.0 | 0.3 |
| PACK_ISOLATE | 25.85 | 37.88 | 1.458 | 1.417 | 1.571 | 0.0 | 0.0 |
| L1_CONGESTION[UNPACK] | 27.42 | 45.93 | 1.675 | 1.670 | 1.676 | 0.0 | 0.1 |
| L1_CONGESTION[PACK] | 25.89 | 38.95 | 1.501 | 1.427 | 1.669 | 0.0 | 0.1 |

## KERNEL

| run type | orig median over variants | new median over variants | ratio median | ratio min | ratio max | orig rep spread max % | new rep spread max % |
|---|---|---|---|---|---|---|---|
| L1_TO_L1 | 4739.50 | 6642.00 | 1.399 | 1.362 | 1.519 | 0.1 | 0.0 |
| UNPACK_ISOLATE | 4050.00 | 6429.00 | 1.584 | 1.569 | 1.591 | 0.0 | 0.0 |
| MATH_ISOLATE | 2698.00 | 2670.00 | 0.990 | 0.972 | 0.997 | 0.0 | 0.0 |
| PACK_ISOLATE | 3857.50 | 5421.00 | 1.402 | 1.354 | 1.501 | 0.0 | 0.1 |
| L1_CONGESTION[UNPACK] | 4048.50 | 6434.00 | 1.587 | 1.573 | 1.592 | 0.0 | 0.2 |
| L1_CONGESTION[PACK] | 3868.00 | 5533.00 | 1.430 | 1.371 | 1.571 | 0.0 | 0.2 |

## INIT

| run type | orig median over variants | new median over variants | ratio median | ratio min | ratio max | orig rep spread max % | new rep spread max % |
|---|---|---|---|---|---|---|---|
| L1_TO_L1 | 287.00 | 212.00 | 0.716 | 0.669 | 0.756 | 0.3 | 0.0 |
| UNPACK_ISOLATE | 272.00 | 304.00 | 1.118 | 1.094 | 1.129 | 0.0 | 0.0 |
| MATH_ISOLATE | 189.00 | 131.00 | 0.690 | 0.689 | 0.693 | 0.0 | 0.8 |
| PACK_ISOLATE | 296.00 | 218.00 | 0.731 | 0.664 | 0.789 | 0.0 | 2.3 |
| L1_CONGESTION[UNPACK] | 272.00 | 303.00 | 1.114 | 1.090 | 1.129 | 0.0 | 1.0 |
| L1_CONGESTION[PACK] | 296.00 | 220.00 | 0.738 | 0.691 | 0.783 | 0.0 | 2.7 |

## Per variant, TILE_LOOP L1_TO_L1 (cycles per tile)

| variant | orig | new | ratio |
|---|---|---|---|
| Bfp8_b | Bfp8_b | Bfp8_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 31.12 | 48.75 | 1.567 |
| Bfp8_b | Bfp8_b | Bfp8_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 33.73 | 47.64 | 1.412 |
| Bfp8_b | Bfp8_b | Float16 | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.79 | 47.70 | 1.455 |
| Bfp8_b | Bfp8_b | Float16_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 43.72 | 61.69 | 1.411 |
| Bfp8_b | Bfp8_b | Float16_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 47.70 | 1.455 |
| Float16 | Float16 | Bfp8_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 31.09 | 48.80 | 1.570 |
| Float16 | Float16 | Bfp8_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 33.72 | 47.45 | 1.407 |
| Float16 | Float16 | Float16 | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 30.58 | 48.49 | 1.586 |
| Float16 | Float16 | Float16 | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 47.38 | 1.444 |
| Float16 | Float16 | Float16_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 30.81 | 48.64 | 1.579 |
| Float16 | Float16 | Float16_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 47.38 | 1.444 |
| Float16_b | Float16_b | Bfp8_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 31.09 | 48.79 | 1.569 |
| Float16_b | Float16_b | Bfp8_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 33.72 | 47.45 | 1.407 |
| Float16_b | Float16_b | Float16 | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 47.38 | 1.444 |
| Float16_b | Float16_b | Float16_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 30.57 | 48.49 | 1.586 |
| Float16_b | Float16_b | Float16_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 47.38 | 1.444 |
