## Layer 4 traced decode step on one chip (wall, 20 replays; `GLM53_PERF_B`)

| users B | composite ms/step | fast ms/step | delta |
|---|---|---|---|
| 1 | 8.04 | 6.57 | -1.47 ms (-18 %) |
| 4 | 10.47 | 8.97 | -1.50 ms (-14 %) |
| 32 | 22.32 | 20.41 | -1.91 ms (-9 %) |

## One wrap, traced (`tests/test_mhc_perf.py`, 20 replays)

| T | composite ms | fast ms |
|---|---|---|
| 32 | 1.12 | 0.216 |
| 1024 | 5.611 | 5.155 |

## Per-block device time, layer 4 traced decode (Tracy, one replay, labelled eager pass)

### B=4, wrap (`doc/mhc/layer4_B4_wrap/summary.md`)

traced decode window: 466 ops, 10.630 ms device time

| module (innermost label) | ops | device us | % |
|---|---|---|---|
| ATTN_HC | 110 | 935 | 8.8 % |
| NORM | 2 | 93 | 0.9 % |
| ATTN | 75 | 2382 | 22.4 % |
| FFN_HC | 110 | 936 | 8.8 % |
| ROUTER | 139 | 700 | 6.6 % |
| EXPERTS | 21 | 5450 | 51.3 % |
| SHARED | 8 | 132 | 1.2 % |
| MOE | 1 | 3 | 0.0 % |

## Hyper-connection wraps (their own ops, sublayers excluded)

| wrap | ops | device us |
|---|---|---|
| ATTN_HC | 110 | 935 |
| FFN_HC | 110 | 936 |
| both | 220 | 1871 |

### Wrap ops by kind (both wraps)

| op | count | us total | us each |
|---|---|---|---|
| TernaryDeviceOperation | 38 | 608 | 16.0 |
| MatmulDeviceOperation | 2 | 263 | 131.6 |
| BinaryNgDeviceOperation | 18 | 233 | 12.9 |
| ReduceDeviceOperation | 2 | 229 | 114.5 |
| SliceDeviceOperation | 64 | 139 | 2.2 |
| TilizeWithValPaddingDeviceOperation | 42 | 133 | 3.2 |
| MhcSplitSinkhornDeviceOperation | 2 | 104 | 52.2 |
| UntilizeWithUnpaddingDeviceOperation | 42 | 82 | 2.0 |
| FillPadDeviceOperation | 2 | 30 | 15.2 |
| ConcatDeviceOperation | 2 | 30 | 15.1 |
| TypecastDeviceOperation | 4 | 13 | 3.2 |
| UnaryDeviceOperation | 2 | 6 | 3.2 |

### ATTN_HC ops in order

| # | op | cores | us |
|---|---|---|---|
| 0 | MatmulDeviceOperation 32 x 16384 x 32 | 1 | 131 |
| 1 | BinaryNgDeviceOperation | 120 | 24 |
| 2 | FillPadDeviceOperation | 120 | 15 |
| 3 | ReduceDeviceOperation | 1 | 115 |
| 4 | BinaryNgDeviceOperation | 120 | 5 |
| 5 | BinaryNgDeviceOperation | 120 | 5 |
| 6 | UnaryDeviceOperation | 120 | 3 |
| 7 | BinaryNgDeviceOperation | 120 | 7 |
| 8 | MhcSplitSinkhornDeviceOperation | 1 | 52 |
| 9 | SliceDeviceOperation | 120 | 3 |
| 10 | SliceDeviceOperation | 120 | 3 |
| 11 | SliceDeviceOperation | 120 | 3 |
| 12 | SliceDeviceOperation | 120 | 3 |
| 13 | SliceDeviceOperation | 1 | 1 |
| 14 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 15 | SliceDeviceOperation | 4 | 2 |
| 16 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 17 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 18 | SliceDeviceOperation | 4 | 2 |
| 19 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 20 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 21 | SliceDeviceOperation | 4 | 2 |
| 22 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 23 | BinaryNgDeviceOperation | 120 | 16 |
| 24 | TernaryDeviceOperation | 120 | 16 |
| 25 | TernaryDeviceOperation | 120 | 16 |
| 26 | TernaryDeviceOperation | 120 | 16 |
| 27 | TypecastDeviceOperation | 120 | 3 |
| 28 | TypecastDeviceOperation | 120 | 3 |
| 29 | SliceDeviceOperation | 120 | 3 |
| 30 | SliceDeviceOperation | 120 | 3 |
| 31 | SliceDeviceOperation | 120 | 3 |
| 32 | SliceDeviceOperation | 120 | 3 |
| 33 | SliceDeviceOperation | 1 | 1 |
| 34 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 35 | SliceDeviceOperation | 4 | 2 |
| 36 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 37 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 38 | SliceDeviceOperation | 4 | 2 |
| 39 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 40 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 41 | SliceDeviceOperation | 4 | 2 |
| 42 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 43 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 44 | SliceDeviceOperation | 4 | 2 |
| 45 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 46 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 47 | SliceDeviceOperation | 4 | 2 |
| 48 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 49 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 50 | SliceDeviceOperation | 4 | 2 |
| 51 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 52 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 53 | SliceDeviceOperation | 4 | 2 |
| 54 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 55 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 56 | SliceDeviceOperation | 4 | 2 |
| 57 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 58 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 59 | SliceDeviceOperation | 4 | 2 |
| 60 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 61 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 62 | SliceDeviceOperation | 4 | 2 |
| 63 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 64 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 65 | SliceDeviceOperation | 4 | 2 |
| 66 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 67 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 68 | SliceDeviceOperation | 4 | 2 |
| 69 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 70 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 71 | SliceDeviceOperation | 4 | 2 |
| 72 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 73 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 74 | SliceDeviceOperation | 4 | 2 |
| 75 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 76 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 77 | SliceDeviceOperation | 4 | 2 |
| 78 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 79 | SliceDeviceOperation | 1 | 1 |
| 80 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 81 | SliceDeviceOperation | 4 | 2 |
| 82 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 83 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 84 | SliceDeviceOperation | 4 | 2 |
| 85 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 86 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 87 | SliceDeviceOperation | 4 | 2 |
| 88 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 89 | BinaryNgDeviceOperation | 120 | 15 |
| 90 | BinaryNgDeviceOperation | 120 | 15 |
| 91 | BinaryNgDeviceOperation | 120 | 15 |
| 92 | BinaryNgDeviceOperation | 120 | 15 |
| 93 | TernaryDeviceOperation | 120 | 16 |
| 94 | TernaryDeviceOperation | 120 | 16 |
| 95 | TernaryDeviceOperation | 120 | 17 |
| 96 | TernaryDeviceOperation | 120 | 16 |
| 97 | TernaryDeviceOperation | 120 | 16 |
| 98 | TernaryDeviceOperation | 120 | 16 |
| 99 | TernaryDeviceOperation | 120 | 17 |
| 100 | TernaryDeviceOperation | 120 | 16 |
| 101 | TernaryDeviceOperation | 120 | 17 |
| 102 | TernaryDeviceOperation | 120 | 16 |
| 103 | TernaryDeviceOperation | 120 | 17 |
| 104 | TernaryDeviceOperation | 120 | 16 |
| 105 | TernaryDeviceOperation | 120 | 16 |
| 106 | TernaryDeviceOperation | 120 | 16 |
| 107 | TernaryDeviceOperation | 120 | 17 |
| 108 | TernaryDeviceOperation | 120 | 16 |
| 109 | ConcatDeviceOperation | 120 | 15 |

### B=4, fast (`doc/mhc/layer4_B4_fast/summary.md`)

traced decode window: 288 ops, 9.046 ms device time

| module (innermost label) | ops | device us | % |
|---|---|---|---|
| ATTN_HC | 22 | 228 | 2.5 % |
| ATTN | 75 | 2314 | 25.6 % |
| FFN_HC | 22 | 227 | 2.5 % |
| ROUTER | 139 | 698 | 7.7 % |
| EXPERTS | 21 | 5444 | 60.2 % |
| SHARED | 8 | 132 | 1.5 % |
| MOE | 1 | 3 | 0.0 % |

## Hyper-connection wraps (their own ops, sublayers excluded)

| wrap | ops | device us |
|---|---|---|
| ATTN_HC | 22 | 228 |
| FFN_HC | 22 | 227 |
| both | 44 | 455 |

### Wrap ops by kind (both wraps)

| op | count | us total | us each |
|---|---|---|---|
| BinaryNgDeviceOperation | 16 | 137 | 8.6 |
| MhcSplitSinkhornDeviceOperation | 2 | 104 | 52.0 |
| TernaryDeviceOperation | 2 | 60 | 30.1 |
| MatmulDeviceOperation | 6 | 58 | 9.6 |
| ReduceDeviceOperation | 6 | 30 | 4.9 |
| MatmulDeviceOperation b={32} x | 2 | 27 | 13.3 |
| FastReduceNCDeviceOperation | 4 | 16 | 4.1 |
| LayerNormDeviceOperation | 2 | 14 | 7.0 |
| TypecastDeviceOperation | 2 | 6 | 3.0 |
| ShardedToInterleavedDeviceOperation | 2 | 3 | 1.6 |

### ATTN_HC ops in order

| # | op | cores | us |
|---|---|---|---|
| 0 | MatmulDeviceOperation b={32} x 32 x 512 x 32 | 32 | 13 |
| 1 | BinaryNgDeviceOperation | 120 | 24 |
| 2 | ReduceDeviceOperation | 32 | 10 |
| 3 | FastReduceNCDeviceOperation | 1 | 4 |
| 4 | FastReduceNCDeviceOperation | 1 | 4 |
| 5 | BinaryNgDeviceOperation | 120 | 5 |
| 6 | BinaryNgDeviceOperation | 120 | 7 |
| 7 | MhcSplitSinkhornDeviceOperation | 1 | 52 |
| 8 | MatmulDeviceOperation 32 x 32 x 128 | 4 | 3 |
| 9 | BinaryNgDeviceOperation | 120 | 3 |
| 10 | MatmulDeviceOperation 32 x 128 x 4096 | 32 | 8 |
| 11 | LayerNormDeviceOperation | 32 | 7 |
| 12 | ShardedToInterleavedDeviceOperation | 32 | 2 |
| 13 | TypecastDeviceOperation | 120 | 3 |
| 14 | BinaryNgDeviceOperation | 120 | 15 |
| 15 | BinaryNgDeviceOperation | 120 | 4 |
| 16 | ReduceDeviceOperation | 16 | 2 |
| 17 | BinaryNgDeviceOperation | 120 | 3 |
| 18 | ReduceDeviceOperation | 4 | 2 |
| 19 | BinaryNgDeviceOperation | 120 | 7 |
| 20 | MatmulDeviceOperation 128 x 128 x 4096 | 64 | 19 |
| 21 | TernaryDeviceOperation | 120 | 30 |

- B=32 wrap: no CSV
### B=32, fast (`doc/mhc/layer4_B32_fast/summary.md`)

traced decode window: 251 ops, 20.550 ms device time

| module (innermost label) | ops | device us | % |
|---|---|---|---|
| ATTN_HC | 21 | 212 | 1.0 % |
| ATTN | 75 | 11424 | 55.6 % |
| FFN_HC | 21 | 213 | 1.0 % |
| ROUTER | 107 | 535 | 2.6 % |
| EXPERTS | 18 | 8030 | 39.1 % |
| SHARED | 8 | 132 | 0.6 % |
| MOE | 1 | 3 | 0.0 % |

## Hyper-connection wraps (their own ops, sublayers excluded)

| wrap | ops | device us |
|---|---|---|
| ATTN_HC | 21 | 212 |
| FFN_HC | 21 | 213 |
| both | 42 | 426 |

### Wrap ops by kind (both wraps)

| op | count | us total | us each |
|---|---|---|---|
| BinaryNgDeviceOperation | 14 | 108 | 7.7 |
| MhcSplitSinkhornDeviceOperation | 2 | 104 | 52.1 |
| TernaryDeviceOperation | 2 | 60 | 30.1 |
| MatmulDeviceOperation | 6 | 58 | 9.7 |
| ReduceDeviceOperation | 6 | 29 | 4.9 |
| MatmulDeviceOperation b={32} x | 2 | 26 | 13.1 |
| FastReduceNCDeviceOperation | 4 | 16 | 4.1 |
| LayerNormDeviceOperation | 2 | 14 | 7.0 |
| TypecastDeviceOperation | 2 | 6 | 2.9 |
| ShardedToInterleavedDeviceOperation | 2 | 3 | 1.7 |

### ATTN_HC ops in order

| # | op | cores | us |
|---|---|---|---|
| 0 | MatmulDeviceOperation b={32} x 32 x 512 x 32 | 32 | 13 |
| 1 | BinaryNgDeviceOperation | 120 | 24 |
| 2 | ReduceDeviceOperation | 32 | 10 |
| 3 | FastReduceNCDeviceOperation | 1 | 4 |
| 4 | FastReduceNCDeviceOperation | 1 | 4 |
| 5 | BinaryNgDeviceOperation | 120 | 5 |
| 6 | BinaryNgDeviceOperation | 120 | 7 |
| 7 | MhcSplitSinkhornDeviceOperation | 1 | 52 |
| 8 | MatmulDeviceOperation 32 x 32 x 128 | 4 | 3 |
| 9 | BinaryNgDeviceOperation | 120 | 3 |
| 10 | MatmulDeviceOperation 32 x 128 x 4096 | 32 | 7 |
| 11 | LayerNormDeviceOperation | 32 | 7 |
| 12 | ShardedToInterleavedDeviceOperation | 32 | 2 |
| 13 | TypecastDeviceOperation | 120 | 3 |
| 14 | BinaryNgDeviceOperation | 120 | 4 |
| 15 | ReduceDeviceOperation | 16 | 2 |
| 16 | BinaryNgDeviceOperation | 120 | 3 |
| 17 | ReduceDeviceOperation | 4 | 2 |
| 18 | BinaryNgDeviceOperation | 120 | 7 |
| 19 | MatmulDeviceOperation 128 x 128 x 4096 | 64 | 19 |
| 20 | TernaryDeviceOperation | 120 | 30 |

## Regression tests

- `optmhc_regress_decoder_layer.log`: 5 passed, 2 warnings in 206.04s (0:03:26)
    - layer 0 (KDA+dense) prefill S=256: pcc 0.999844
    - decode step 0 (pos 256): pcc 0.999886
    - decode step 1 (pos 257): pcc 0.999921
    - decode step 2 (pos 258): pcc 0.999929
    - decode step 3 (pos 259): pcc 0.999901
    - layer 0 prefill S=200 (non-aligned): pcc 0.999799
    - decode step 0 (pos 200): pcc 0.999876
    - decode step 1 (pos 201): pcc 0.999860
    - layer 3 (DSA+MoE) prefill S=256: pcc 0.999920
    - decode step 0 (pos 256): pcc 0.999959
    - decode step 1 (pos 257): pcc 0.999975
    - decode step 2 (pos 258): pcc 0.999959
    - decode step 3 (pos 259): pcc 0.999950
    - layer 3 prefill S=200 (non-aligned): pcc 0.999930
    - decode step 0 (pos 200): pcc 0.999976
    - decode step 1 (pos 201): pcc 0.999983
    - layer 4 (KDA+MoE) prefill S=256: pcc 0.999966
    - decode step 0 (pos 256): pcc 0.999996
    - decode step 1 (pos 257): pcc 0.999988
    - decode step 2 (pos 258): pcc 0.999994
    - decode step 3 (pos 259): pcc 0.999995
- `optmhc_regress_decode_trace.log`: 5 passed, 2 warnings in 188.89s (0:03:08)
    - layer 0 prefill S=256: pcc 0.999844
    - traced decode step 0 (pos 256): pcc 0.999886
    - traced decode step 1 (pos 257): pcc 0.999921
    - traced decode step 2 (pos 258): pcc 0.999929
    - traced decode step 3 (pos 259): pcc 0.999901
    - layer 0 user 0 prefill S=100: pcc 0.999784
    - layer 0 user 1 prefill S=37: pcc 0.999779
    - layer 3 prefill S=256: pcc 0.999920
    - traced decode step 0 (pos 256): pcc 0.999959
    - traced decode step 1 (pos 257): pcc 0.999975
    - traced decode step 2 (pos 258): pcc 0.999959
    - traced decode step 3 (pos 259): pcc 0.999950
    - layer 3 user 0 prefill S=100: pcc 0.999957
    - layer 3 user 1 prefill S=37: pcc 0.999946
    - layer 4 prefill S=256: pcc 0.999966
    - traced decode step 0 (pos 256): pcc 0.999996
    - traced decode step 1 (pos 257): pcc 0.999988
    - traced decode step 2 (pos 258): pcc 0.999994
    - traced decode step 3 (pos 259): pcc 0.999995

## Prefill (layer 4, S=1024, warmed wall)

- wrap: 285.2 ms
- fast: 282.7 ms
