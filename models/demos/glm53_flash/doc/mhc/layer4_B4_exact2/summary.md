# layer 4, B=4, fast exact=2

traced decode window: 177 ops, 8.175 ms device time

| module (innermost label) | ops | device us | % |
|---|---|---|---|
| ATTN_HC | 32 | 321 | 3.9 % |
| ATTN | 75 | 2199 | 26.9 % |
| FFN_HC | 32 | 316 | 3.9 % |
| ROUTER | 17 | 167 | 2.0 % |
| EXPERTS | 15 | 5090 | 62.3 % |
| SHARED | 5 | 79 | 1.0 % |
| MOE | 1 | 3 | 0.0 % |

## Hyper-connection wraps (their own ops, sublayers excluded)

| wrap | ops | device us |
|---|---|---|
| ATTN_HC | 32 | 321 |
| FFN_HC | 32 | 316 |
| both | 64 | 637 |

### Wrap ops by kind (both wraps)

| op | count | us total | us each |
|---|---|---|---|
| BinaryNgDeviceOperation | 20 | 192 | 9.6 |
| MhcSplitSinkhornDeviceOperation | 2 | 104 | 52.0 |
| TypecastDeviceOperation | 14 | 90 | 6.4 |
| MatmulDeviceOperation | 6 | 62 | 10.4 |
| TernaryDeviceOperation | 2 | 60 | 30.0 |
| ConcatDeviceOperation | 4 | 39 | 9.8 |
| ReduceDeviceOperation | 6 | 29 | 4.9 |
| MatmulDeviceOperation b={32} x | 2 | 26 | 13.0 |
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
| 24 | TypecastDeviceOperation | 120 | 13 |
| 25 | TypecastDeviceOperation | 120 | 11 |
| 26 | BinaryNgDeviceOperation | 120 | 24 |
| 27 | TypecastDeviceOperation | 120 | 13 |
| 28 | ConcatDeviceOperation | 48 | 2 |
| 29 | ConcatDeviceOperation | 120 | 19 |
| 30 | MatmulDeviceOperation 128 x 384 x 4096 | 64 | 21 |
| 31 | TernaryDeviceOperation | 120 | 30 |
