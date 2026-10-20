# layer 4, B=4, fast + mhc_post_mix custom op (cop2)

traced decode window: 141 ops, 7.855 ms device time

| module (innermost label) | ops | device us | % |
|---|---|---|---|
| ATTN_HC | 14 | 173 | 2.2 % |
| ATTN | 75 | 2173 | 27.7 % |
| FFN_HC | 14 | 174 | 2.2 % |
| ROUTER | 17 | 167 | 2.1 % |
| EXPERTS | 15 | 5087 | 64.8 % |
| SHARED | 5 | 79 | 1.0 % |
| MOE | 1 | 3 | 0.0 % |

## Hyper-connection wraps (their own ops, sublayers excluded)

| wrap | ops | device us |
|---|---|---|
| ATTN_HC | 14 | 173 |
| FFN_HC | 14 | 174 |
| both | 28 | 347 |

### Wrap ops by kind (both wraps)

| op | count | us total | us each |
|---|---|---|---|
| MhcSplitSinkhornDeviceOperation | 2 | 104 | 52.0 |
| BinaryNgDeviceOperation | 8 | 78 | 9.8 |
| GenericOpDeviceOperation | 2 | 64 | 31.9 |
| MatmulDeviceOperation b={32} x | 2 | 27 | 13.3 |
| MatmulDeviceOperation | 4 | 21 | 5.2 |
| ReduceDeviceOperation | 2 | 20 | 10.1 |
| FastReduceNCDeviceOperation | 4 | 16 | 4.1 |
| LayerNormDeviceOperation | 2 | 14 | 7.0 |
| ShardedToInterleavedDeviceOperation | 2 | 3 | 1.5 |

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
| 12 | ShardedToInterleavedDeviceOperation | 32 | 1 |
| 13 | GenericOpDeviceOperation | 120 | 32 |
