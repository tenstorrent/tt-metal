# Perf cycles: original vs HAL run 8a (asserts off)

original runs: orig_na_r1 orig_na_r2 orig_na_r3; HAL run 8a (asserts off) runs: hal8a_na_r1 hal8a_na_r2 hal8a_na_r3
Values = median over runs of the framework's mean(); TILE_LOOP = cycles per tile, KERNEL/INIT = cycles per zone. ratio = new / original (> 1 = slower).

## TILE_LOOP

| run type | orig median over variants | new median over variants | ratio median | ratio min | ratio max | orig rep spread max % | new rep spread max % |
|---|---|---|---|---|---|---|---|
| L1_TO_L1 | 32.80 | 41.98 | 1.324 | 1.226 | 1.449 | 0.0 | 0.1 |
| UNPACK_ISOLATE | 27.43 | 32.45 | 1.183 | 1.182 | 1.184 | 0.0 | 0.1 |
| MATH_ISOLATE | 16.64 | 16.46 | 0.990 | 0.978 | 0.992 | 0.0 | 0.0 |
| PACK_ISOLATE | 25.85 | 37.88 | 1.458 | 1.417 | 1.572 | 0.0 | 0.0 |
| L1_CONGESTION[UNPACK] | 27.42 | 32.51 | 1.185 | 1.182 | 1.188 | 0.1 | 0.0 |
| L1_CONGESTION[PACK] | 25.89 | 38.88 | 1.498 | 1.427 | 1.619 | 0.2 | 0.0 |

## KERNEL

| run type | orig median over variants | new median over variants | ratio median | ratio min | ratio max | orig rep spread max % | new rep spread max % |
|---|---|---|---|---|---|---|---|
| L1_TO_L1 | 4739.50 | 5971.50 | 1.297 | 1.217 | 1.412 | 0.1 | 0.0 |
| UNPACK_ISOLATE | 4050.00 | 4764.00 | 1.174 | 1.163 | 1.180 | 0.1 | 0.1 |
| MATH_ISOLATE | 2698.00 | 2722.00 | 1.006 | 0.990 | 1.018 | 0.0 | 0.0 |
| PACK_ISOLATE | 3857.50 | 5473.00 | 1.416 | 1.368 | 1.512 | 0.0 | 0.0 |
| L1_CONGESTION[UNPACK] | 4048.50 | 4769.00 | 1.177 | 1.163 | 1.181 | 0.0 | 0.0 |
| L1_CONGESTION[PACK] | 3868.00 | 5579.50 | 1.442 | 1.384 | 1.553 | 0.2 | 0.0 |

## INIT

| run type | orig median over variants | new median over variants | ratio median | ratio min | ratio max | orig rep spread max % | new rep spread max % |
|---|---|---|---|---|---|---|---|
| L1_TO_L1 | 287.00 | 227.00 | 0.802 | 0.721 | 0.920 | 0.3 | 1.6 |
| UNPACK_ISOLATE | 272.00 | 353.00 | 1.316 | 1.298 | 1.329 | 0.0 | 0.6 |
| MATH_ISOLATE | 189.00 | 146.00 | 0.768 | 0.706 | 0.772 | 0.0 | 0.0 |
| PACK_ISOLATE | 296.00 | 235.00 | 0.802 | 0.714 | 0.955 | 0.0 | 1.7 |
| L1_CONGESTION[UNPACK] | 272.00 | 353.00 | 1.327 | 1.298 | 1.342 | 0.0 | 0.0 |
| L1_CONGESTION[PACK] | 296.00 | 244.00 | 0.824 | 0.755 | 0.931 | 0.0 | 0.0 |

## Per variant, TILE_LOOP L1_TO_L1 (cycles per tile)

| variant | orig | new | ratio |
|---|---|---|---|
| Bfp8_b | Bfp8_b | Bfp8_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 31.12 | 44.51 | 1.430 |
| Bfp8_b | Bfp8_b | Bfp8_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 33.73 | 44.53 | 1.320 |
| Bfp8_b | Bfp8_b | Float16 | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.79 | 41.67 | 1.271 |
| Bfp8_b | Bfp8_b | Float16_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 43.72 | 59.41 | 1.359 |
| Bfp8_b | Bfp8_b | Float16_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 41.67 | 1.271 |
| Float16 | Float16 | Bfp8_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 31.09 | 45.04 | 1.449 |
| Float16 | Float16 | Bfp8_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 33.72 | 45.34 | 1.345 |
| Float16 | Float16 | Float16 | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 30.58 | 40.48 | 1.324 |
| Float16 | Float16 | Float16 | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 40.21 | 1.226 |
| Float16 | Float16 | Float16_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 30.81 | 42.30 | 1.373 |
| Float16 | Float16 | Float16_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 40.21 | 1.226 |
| Float16_b | Float16_b | Bfp8_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 31.09 | 45.05 | 1.449 |
| Float16_b | Float16_b | Bfp8_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 33.72 | 45.34 | 1.345 |
| Float16_b | Float16_b | Float16 | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 40.21 | 1.226 |
| Float16_b | Float16_b | Float16_b | DestAccumulation.No | MathOperation.Elwadd | MathFidelity.LoFi | 30.57 | 40.48 | 1.324 |
| Float16_b | Float16_b | Float16_b | DestAccumulation.Yes | MathOperation.Elwadd | MathFidelity.LoFi | 32.80 | 40.21 | 1.226 |
