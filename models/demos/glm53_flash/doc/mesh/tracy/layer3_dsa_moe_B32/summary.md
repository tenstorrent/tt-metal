# layer 3 (DSA+MoE, experts bfp8) on the (2,4) block, B=32/row (64 users), position 1024 (seed-only cache), one traced replay (2026_10_20_02_33_35)

## Device time per chip (traced decode window, one replay)

| chip | (r,c) | ops | device ms/step | CCL kernel ms | CCL composite ms |
|---|---|---|---|---|---|
| 1 | (0,0) | 443 | 9.940 | 0.333 | 0.333 |
| 0 | (0,1) | 443 | 9.935 | 0.127 | 0.127 |
| 2 | (0,2) | 443 | 9.935 | 0.124 | 0.124 |
| 3 | (0,3) | 443 | 9.939 | 0.530 | 0.530 |
| 5 | (1,0) | 443 | 9.932 | 0.319 | 0.319 |
| 4 | (1,1) | 443 | 9.931 | 0.124 | 0.124 |
| 6 | (1,2) | 443 | 9.933 | 0.122 | 0.122 |
| 7 | (1,3) | 443 | 9.929 | 0.521 | 0.521 |

## Chip 1 (0,0): shares of the 9.940 ms step

| group | device us | % of step | ops |
|---|---|---|---|
| CCL ops (AllGather/AllBroadcast/ReduceScatter/AllReduce kernels) | 333 | 3.3 % | 7 |
| CCL composites (all_reduce/all_gather/reduce_scatter incl. local reduce ops) | 333 | 3.3 % | 7 |
| sparse_matmul (expert matmuls) | 2106 | 21.2 % | 4 |
| EXPERTS local (router-weight select, sparse_matmul, slicing; no CCL) | 2965 | 29.8 % | 45 |
| ROUTER (exact fp32 composite top-8) | 3382 | 34.0 % | 123 |
| SHARED expert | 88 | 0.9 % | 8 |
| MoE total (router + experts + shared + CCL + add) | 6732 | 67.7 % | 182 |
| ATTN composite incl. its all_reduce (KDA recurrence / DSA flash-MLA + index) | 908 | 9.1 % | 41 |
| ATTN local only (no CCL) | 869 | 8.7 % | 39 |
| NORM (RMSNorm x2) | 95 | 1.0 % | 2 |
| MHC (hyper-connections x2, everything outside the sublayers) | 2205 | 22.2 % | 218 |

### Chip 1 (0,0): top-10 ops

| ID | op | module | device us | % of step |
|---|---|---|---|---|
| 1687 | SparseMatmulDeviceOperation active=?/72 x 32 x 4096 x 2048 | MOE/EXPERTS | 789 | 7.9 % |
| 1709 | SparseMatmulDeviceOperation active=?/72 x 32 x 4096 x 2048 | MOE/EXPERTS | 789 | 7.9 % |
| 1587 | AccumulationDeviceOperation | MOE/ROUTER | 331 | 3.3 % |
| 1657 | AccumulationDeviceOperation | MOE/ROUTER | 331 | 3.3 % |
| 1615 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 3.3 % |
| 1643 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 3.3 % |
| 1560 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 3.3 % |
| 1601 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 3.3 % |
| 1629 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 3.3 % |
| 1573 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 3.3 % |

### Chip 1 (0,0): by op class

| op class | device us | % of step | count |
|---|---|---|---|
| AccumulationDeviceOperation | 2644 | 26.6 % | 8 |
| SparseMatmulDeviceOperation | 2106 | 21.2 % | 4 |
| MatmulDeviceOperation | 902 | 9.1 % | 15 |
| BinaryNgDeviceOperation | 662 | 6.7 % | 94 |
| TernaryDeviceOperation | 607 | 6.1 % | 38 |
| UnaryDeviceOperation | 474 | 4.8 % | 17 |
| UntilizeWithUnpaddingDeviceOperation | 402 | 4.0 % | 44 |
| ReduceDeviceOperation | 322 | 3.2 % | 30 |
| SliceDeviceOperation | 261 | 2.6 % | 76 |
| ReduceScatterDeviceOperation | 257 | 2.6 % | 3 |
| PermuteDeviceOperation | 245 | 2.5 % | 16 |
| TilizeWithValPaddingDeviceOperation | 176 | 1.8 % | 42 |
| LayerNormDeviceOperation | 128 | 1.3 % | 5 |
| ReshapeViewDeviceOperation | 121 | 1.2 % | 7 |

### Chip 1 (0,0): CCL ops in order

| ID | op | module | device us |
|---|---|---|---|
| 1442 | ReduceScatterDeviceOperation | ATTN/CCL_ALL_REDUCE | 24 |
| 1443 | AllGatherDeviceOperation | ATTN/CCL_ALL_REDUCE | 16 |
| 1677 | AllGatherDeviceOperation | MOE/EXPERTS/CCL_ALL_GATHER | 32 |
| 1678 | AllGatherDeviceOperation | MOE/EXPERTS/CCL_ALL_GATHER | 7 |
| 1724 | ReduceScatterDeviceOperation | MOE/EXPERTS/CCL_REDUCE_SCATTER | 19 |
| 1734 | ReduceScatterDeviceOperation | MOE/CCL_ALL_REDUCE | 215 |
| 1735 | AllGatherDeviceOperation | MOE/CCL_ALL_REDUCE | 20 |

## Chip 5 (1,0): shares of the 9.932 ms step

| group | device us | % of step | ops |
|---|---|---|---|
| CCL ops (AllGather/AllBroadcast/ReduceScatter/AllReduce kernels) | 319 | 3.2 % | 7 |
| CCL composites (all_reduce/all_gather/reduce_scatter incl. local reduce ops) | 319 | 3.2 % | 7 |
| sparse_matmul (expert matmuls) | 2110 | 21.2 % | 4 |
| EXPERTS local (router-weight select, sparse_matmul, slicing; no CCL) | 2967 | 29.9 % | 45 |
| ROUTER (exact fp32 composite top-8) | 3380 | 34.0 % | 123 |
| SHARED expert | 89 | 0.9 % | 8 |
| MoE total (router + experts + shared + CCL + add) | 6719 | 67.7 % | 182 |
| ATTN composite incl. its all_reduce (KDA recurrence / DSA flash-MLA + index) | 908 | 9.1 % | 41 |
| ATTN local only (no CCL) | 868 | 8.7 % | 39 |
| NORM (RMSNorm x2) | 95 | 1.0 % | 2 |
| MHC (hyper-connections x2, everything outside the sublayers) | 2209 | 22.2 % | 218 |

### Chip 5 (1,0): top-10 ops

| ID | op | module | device us | % of step |
|---|---|---|---|---|
| 1687 | SparseMatmulDeviceOperation active=?/72 x 32 x 4096 x 2048 | MOE/EXPERTS | 789 | 7.9 % |
| 1709 | SparseMatmulDeviceOperation active=?/72 x 32 x 4096 x 2048 | MOE/EXPERTS | 789 | 7.9 % |
| 1657 | AccumulationDeviceOperation | MOE/ROUTER | 331 | 3.3 % |
| 1643 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 3.3 % |
| 1560 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 3.3 % |
| 1629 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 3.3 % |
| 1601 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 3.3 % |
| 1615 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 3.3 % |
| 1587 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 3.3 % |
| 1573 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 3.3 % |

### Chip 5 (1,0): by op class

| op class | device us | % of step | count |
|---|---|---|---|
| AccumulationDeviceOperation | 2643 | 26.6 % | 8 |
| SparseMatmulDeviceOperation | 2110 | 21.2 % | 4 |
| MatmulDeviceOperation | 901 | 9.1 % | 15 |
| BinaryNgDeviceOperation | 661 | 6.7 % | 94 |
| TernaryDeviceOperation | 607 | 6.1 % | 38 |
| UnaryDeviceOperation | 472 | 4.8 % | 17 |
| UntilizeWithUnpaddingDeviceOperation | 402 | 4.0 % | 44 |
| ReduceDeviceOperation | 322 | 3.2 % | 30 |
| SliceDeviceOperation | 266 | 2.7 % | 76 |
| ReduceScatterDeviceOperation | 259 | 2.6 % | 3 |
| PermuteDeviceOperation | 245 | 2.5 % | 16 |
| TilizeWithValPaddingDeviceOperation | 178 | 1.8 % | 42 |
| LayerNormDeviceOperation | 128 | 1.3 % | 5 |
| ReshapeViewDeviceOperation | 121 | 1.2 % | 7 |

### Chip 5 (1,0): CCL ops in order

| ID | op | module | device us |
|---|---|---|---|
| 1442 | ReduceScatterDeviceOperation | ATTN/CCL_ALL_REDUCE | 24 |
| 1443 | AllGatherDeviceOperation | ATTN/CCL_ALL_REDUCE | 15 |
| 1677 | AllGatherDeviceOperation | MOE/EXPERTS/CCL_ALL_GATHER | 16 |
| 1678 | AllGatherDeviceOperation | MOE/EXPERTS/CCL_ALL_GATHER | 7 |
| 1724 | ReduceScatterDeviceOperation | MOE/EXPERTS/CCL_REDUCE_SCATTER | 21 |
| 1734 | ReduceScatterDeviceOperation | MOE/CCL_ALL_REDUCE | 214 |
| 1735 | AllGatherDeviceOperation | MOE/CCL_ALL_REDUCE | 22 |
