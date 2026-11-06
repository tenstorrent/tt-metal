# layer 4, B=32, fast

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
