# layer 4, B=32, fast exact=3

traced decode window: 182 ops, 20.550 ms device time

| module (innermost label) | ops | device us | % |
|---|---|---|---|
| ATTN_HC | 37 | 379 | 1.8 % |
| ATTN | 75 | 11286 | 54.9 % |
| FFN_HC | 37 | 378 | 1.8 % |
| ROUTER | 15 | 159 | 0.8 % |
| EXPERTS | 12 | 8267 | 40.2 % |
| SHARED | 5 | 79 | 0.4 % |
| MOE | 1 | 3 | 0.0 % |

## Hyper-connection wraps (their own ops, sublayers excluded)

| wrap | ops | device us |
|---|---|---|
| ATTN_HC | 37 | 379 |
| FFN_HC | 37 | 378 |
| both | 74 | 756 |

### Wrap ops by kind (both wraps)

| op | count | us total | us each |
|---|---|---|---|
| BinaryNgDeviceOperation | 22 | 215 | 9.8 |
| TypecastDeviceOperation | 22 | 144 | 6.5 |
| MhcSplitSinkhornDeviceOperation | 2 | 104 | 51.9 |
| MatmulDeviceOperation | 6 | 75 | 12.4 |
| ConcatDeviceOperation | 4 | 69 | 17.3 |
| TernaryDeviceOperation | 2 | 60 | 30.2 |
| ReduceDeviceOperation | 6 | 30 | 5.0 |
| MatmulDeviceOperation b={32} x | 2 | 26 | 12.9 |
| FastReduceNCDeviceOperation | 4 | 16 | 4.1 |
| LayerNormDeviceOperation | 2 | 14 | 7.0 |
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
| 14 | BinaryNgDeviceOperation | 120 | 4 |
| 15 | ReduceDeviceOperation | 16 | 2 |
| 16 | BinaryNgDeviceOperation | 120 | 3 |
| 17 | ReduceDeviceOperation | 4 | 2 |
| 18 | BinaryNgDeviceOperation | 120 | 7 |
| 19 | TypecastDeviceOperation | 16 | 2 |
| 20 | TypecastDeviceOperation | 16 | 2 |
| 21 | BinaryNgDeviceOperation | 120 | 3 |
| 22 | TypecastDeviceOperation | 16 | 2 |
| 23 | TypecastDeviceOperation | 16 | 2 |
| 24 | BinaryNgDeviceOperation | 120 | 3 |
| 25 | TypecastDeviceOperation | 16 | 2 |
| 26 | TypecastDeviceOperation | 120 | 13 |
| 27 | TypecastDeviceOperation | 120 | 11 |
| 28 | BinaryNgDeviceOperation | 120 | 24 |
| 29 | TypecastDeviceOperation | 120 | 13 |
| 30 | TypecastDeviceOperation | 120 | 11 |
| 31 | BinaryNgDeviceOperation | 120 | 24 |
| 32 | TypecastDeviceOperation | 120 | 13 |
| 33 | ConcatDeviceOperation | 96 | 2 |
| 34 | ConcatDeviceOperation | 120 | 32 |
| 35 | MatmulDeviceOperation 128 x 768 x 4096 | 64 | 27 |
| 36 | TernaryDeviceOperation | 120 | 30 |
