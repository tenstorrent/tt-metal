# layer 4, B=4, fast exact=3

traced decode window: 189 ops, 8.293 ms device time

| module (innermost label) | ops | device us | % |
|---|---|---|---|
| ATTN_HC | 38 | 394 | 4.8 % |
| ATTN | 75 | 2175 | 26.2 % |
| FFN_HC | 38 | 394 | 4.7 % |
| ROUTER | 17 | 167 | 2.0 % |
| EXPERTS | 15 | 5081 | 61.3 % |
| SHARED | 5 | 79 | 1.0 % |
| MOE | 1 | 3 | 0.0 % |

## Hyper-connection wraps (their own ops, sublayers excluded)

| wrap | ops | device us |
|---|---|---|
| ATTN_HC | 38 | 394 |
| FFN_HC | 38 | 394 |
| both | 76 | 788 |

### Wrap ops by kind (both wraps)

| op | count | us total | us each |
|---|---|---|---|
| BinaryNgDeviceOperation | 24 | 244 | 10.2 |
| TypecastDeviceOperation | 22 | 145 | 6.6 |
| MhcSplitSinkhornDeviceOperation | 2 | 104 | 52.1 |
| MatmulDeviceOperation | 6 | 75 | 12.5 |
| ConcatDeviceOperation | 4 | 70 | 17.5 |
| TernaryDeviceOperation | 2 | 60 | 30.0 |
| ReduceDeviceOperation | 6 | 29 | 4.8 |
| MatmulDeviceOperation b={32} x | 2 | 27 | 13.3 |
| FastReduceNCDeviceOperation | 4 | 16 | 4.1 |
| LayerNormDeviceOperation | 2 | 14 | 7.0 |
| ShardedToInterleavedDeviceOperation | 2 | 3 | 1.6 |

### ATTN_HC ops in order

| # | op | cores | us |
|---|---|---|---|
| 0 | MatmulDeviceOperation b={32} x 32 x 512 x 32 | 32 | 13 |
| 1 | BinaryNgDeviceOperation | 120 | 23 |
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
| 20 | TypecastDeviceOperation | 16 | 2 |
| 21 | TypecastDeviceOperation | 16 | 2 |
| 22 | BinaryNgDeviceOperation | 120 | 3 |
| 23 | TypecastDeviceOperation | 16 | 2 |
| 24 | TypecastDeviceOperation | 16 | 2 |
| 25 | BinaryNgDeviceOperation | 120 | 3 |
| 26 | TypecastDeviceOperation | 16 | 2 |
| 27 | TypecastDeviceOperation | 120 | 13 |
| 28 | TypecastDeviceOperation | 120 | 11 |
| 29 | BinaryNgDeviceOperation | 120 | 24 |
| 30 | TypecastDeviceOperation | 120 | 13 |
| 31 | TypecastDeviceOperation | 120 | 11 |
| 32 | BinaryNgDeviceOperation | 120 | 24 |
| 33 | TypecastDeviceOperation | 120 | 12 |
| 34 | ConcatDeviceOperation | 96 | 2 |
| 35 | ConcatDeviceOperation | 120 | 34 |
| 36 | MatmulDeviceOperation 128 x 768 x 4096 | 64 | 27 |
| 37 | TernaryDeviceOperation | 120 | 30 |
