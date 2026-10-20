# layer 4, B=4, fast + mhc_post_mix kernel

traced decode window: 143 ops, 7.886 ms device time

| module (innermost label) | ops | device us | % |
|---|---|---|---|
| ATTN_HC | 15 | 192 | 2.4 % |
| ATTN | 75 | 2173 | 27.6 % |
| FFN_HC | 15 | 193 | 2.5 % |
| ROUTER | 17 | 167 | 2.1 % |
| EXPERTS | 15 | 5078 | 64.4 % |
| SHARED | 5 | 79 | 1.0 % |
| MOE | 1 | 3 | 0.0 % |

## Hyper-connection wraps (their own ops, sublayers excluded)

| wrap | ops | device us |
|---|---|---|
| ATTN_HC | 15 | 192 |
| FFN_HC | 15 | 193 |
| both | 30 | 386 |

### Wrap ops by kind (both wraps)

| op | count | us total | us each |
|---|---|---|---|
| MhcSplitSinkhornDeviceOperation | 2 | 104 | 52.0 |
| MhcPostMixDeviceOperation | 2 | 94 | 47.1 |
| BinaryNgDeviceOperation | 8 | 79 | 9.9 |
| MatmulDeviceOperation b={32} x | 2 | 27 | 13.5 |
| MatmulDeviceOperation | 4 | 21 | 5.2 |
| ReduceDeviceOperation | 2 | 21 | 10.4 |
| FastReduceNCDeviceOperation | 4 | 16 | 4.1 |
| LayerNormDeviceOperation | 2 | 14 | 7.0 |
| TypecastDeviceOperation | 2 | 6 | 3.1 |
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
| 14 | MhcPostMixDeviceOperation | 120 | 47 |
