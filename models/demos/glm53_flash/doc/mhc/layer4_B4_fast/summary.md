# layer 4, B=4, fast

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
