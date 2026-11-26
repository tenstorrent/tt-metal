# layer 4 (KDA+MoE, experts bfp8) on the (2,4) block, B=32/row (64 users), position 1024, one traced replay (2026_10_20_02_13_31)

## Device time per chip (traced decode window, one replay)

| chip | (r,c) | ops | device ms/step | CCL kernel ms | CCL composite ms |
|---|---|---|---|---|---|
| 1 | (0,0) | 479 | 12.448 | 0.238 | 0.238 |
| 0 | (0,1) | 479 | 12.453 | 0.155 | 0.155 |
| 2 | (0,2) | 479 | 12.443 | 0.400 | 0.400 |
| 3 | (0,3) | 479 | 12.450 | 0.713 | 0.713 |
| 5 | (1,0) | 479 | 12.450 | 0.252 | 0.252 |
| 4 | (1,1) | 479 | 12.441 | 0.160 | 0.160 |
| 6 | (1,2) | 479 | 12.435 | 0.366 | 0.366 |
| 7 | (1,3) | 479 | 12.440 | 0.717 | 0.717 |

## Chip 1 (0,0): shares of the 12.448 ms step

| group | device us | % of step | ops |
|---|---|---|---|
| CCL ops (AllGather/AllBroadcast/ReduceScatter/AllReduce kernels) | 238 | 1.9 % | 7 |
| CCL composites (all_reduce/all_gather/reduce_scatter incl. local reduce ops) | 238 | 1.9 % | 7 |
| sparse_matmul (expert matmuls) | 2353 | 18.9 % | 4 |
| EXPERTS local (router-weight select, sparse_matmul, slicing; no CCL) | 3215 | 25.8 % | 45 |
| ROUTER (exact fp32 composite top-8) | 3380 | 27.2 % | 123 |
| SHARED expert | 90 | 0.7 % | 8 |
| MoE total (router + experts + shared + CCL + add) | 6862 | 55.1 % | 182 |
| ATTN composite incl. its all_reduce (KDA recurrence / DSA flash-MLA + index) | 3269 | 26.3 % | 77 |
| ATTN local only (no CCL) | 3204 | 25.7 % | 75 |
| NORM (RMSNorm x2) | 93 | 0.7 % | 2 |
| MHC (hyper-connections x2, everything outside the sublayers) | 2224 | 17.9 % | 218 |

### Chip 1 (0,0): top-10 ops

| ID | op | module | device us | % of step |
|---|---|---|---|---|
| 1829 | SparseMatmulDeviceOperation active=?/72 x 32 x 4096 x 2048 | MOE/EXPERTS | 908 | 7.3 % |
| 1851 | SparseMatmulDeviceOperation active=?/72 x 32 x 4096 x 2048 | MOE/EXPERTS | 868 | 7.0 % |
| 1702 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 2.7 % |
| 1757 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 2.7 % |
| 1729 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 2.7 % |
| 1715 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 2.7 % |
| 1799 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 2.7 % |
| 1785 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 2.7 % |
| 1771 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 2.7 % |
| 1743 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 2.7 % |

### Chip 1 (0,0): by op class

| op class | device us | % of step | count |
|---|---|---|---|
| AccumulationDeviceOperation | 2642 | 21.2 % | 8 |
| SparseMatmulDeviceOperation | 2353 | 18.9 % | 4 |
| BinaryNgDeviceOperation | 1984 | 15.9 % | 112 |
| MatmulDeviceOperation | 686 | 5.5 % | 14 |
| TernaryDeviceOperation | 622 | 5.0 % | 38 |
| ReshapeViewDeviceOperation | 586 | 4.7 % | 16 |
| ReduceDeviceOperation | 523 | 4.2 % | 31 |
| UnaryDeviceOperation | 499 | 4.0 % | 22 |
| UntilizeWithUnpaddingDeviceOperation | 475 | 3.8 % | 46 |
| TilizeWithValPaddingDeviceOperation | 470 | 3.8 % | 45 |
| SliceDeviceOperation | 338 | 2.7 % | 81 |
| PermuteDeviceOperation | 244 | 2.0 % | 16 |
| ReduceScatterDeviceOperation | 182 | 1.5 % | 3 |
| CopyDeviceOperation | 179 | 1.4 % | 2 |

### Chip 1 (0,0): CCL ops in order

| ID | op | module | device us |
|---|---|---|---|
| 1582 | ReduceScatterDeviceOperation | ATTN/CCL_ALL_REDUCE | 51 |
| 1583 | AllGatherDeviceOperation | ATTN/CCL_ALL_REDUCE | 14 |
| 1819 | AllGatherDeviceOperation | MOE/EXPERTS/CCL_ALL_GATHER | 16 |
| 1820 | AllGatherDeviceOperation | MOE/EXPERTS/CCL_ALL_GATHER | 7 |
| 1866 | ReduceScatterDeviceOperation | MOE/EXPERTS/CCL_REDUCE_SCATTER | 19 |
| 1876 | ReduceScatterDeviceOperation | MOE/CCL_ALL_REDUCE | 112 |
| 1877 | AllGatherDeviceOperation | MOE/CCL_ALL_REDUCE | 19 |

## Chip 5 (1,0): shares of the 12.450 ms step

| group | device us | % of step | ops |
|---|---|---|---|
| CCL ops (AllGather/AllBroadcast/ReduceScatter/AllReduce kernels) | 252 | 2.0 % | 7 |
| CCL composites (all_reduce/all_gather/reduce_scatter incl. local reduce ops) | 252 | 2.0 % | 7 |
| sparse_matmul (expert matmuls) | 2353 | 18.9 % | 4 |
| EXPERTS local (router-weight select, sparse_matmul, slicing; no CCL) | 3210 | 25.8 % | 45 |
| ROUTER (exact fp32 composite top-8) | 3380 | 27.2 % | 123 |
| SHARED expert | 89 | 0.7 % | 8 |
| MoE total (router + experts + shared + CCL + add) | 6867 | 55.2 % | 182 |
| ATTN composite incl. its all_reduce (KDA recurrence / DSA flash-MLA + index) | 3259 | 26.2 % | 77 |
| ATTN local only (no CCL) | 3192 | 25.6 % | 75 |
| NORM (RMSNorm x2) | 93 | 0.7 % | 2 |
| MHC (hyper-connections x2, everything outside the sublayers) | 2231 | 17.9 % | 218 |

### Chip 5 (1,0): top-10 ops

| ID | op | module | device us | % of step |
|---|---|---|---|---|
| 1829 | SparseMatmulDeviceOperation active=?/72 x 32 x 4096 x 2048 | MOE/EXPERTS | 907 | 7.3 % |
| 1851 | SparseMatmulDeviceOperation active=?/72 x 32 x 4096 x 2048 | MOE/EXPERTS | 868 | 7.0 % |
| 1715 | AccumulationDeviceOperation | MOE/ROUTER | 331 | 2.7 % |
| 1785 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 2.7 % |
| 1757 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 2.7 % |
| 1702 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 2.7 % |
| 1771 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 2.7 % |
| 1799 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 2.7 % |
| 1729 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 2.7 % |
| 1743 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 2.7 % |

### Chip 5 (1,0): by op class

| op class | device us | % of step | count |
|---|---|---|---|
| AccumulationDeviceOperation | 2643 | 21.2 % | 8 |
| SparseMatmulDeviceOperation | 2353 | 18.9 % | 4 |
| BinaryNgDeviceOperation | 1964 | 15.8 % | 112 |
| MatmulDeviceOperation | 686 | 5.5 % | 14 |
| TernaryDeviceOperation | 624 | 5.0 % | 38 |
| ReshapeViewDeviceOperation | 587 | 4.7 % | 16 |
| ReduceDeviceOperation | 522 | 4.2 % | 31 |
| UnaryDeviceOperation | 495 | 4.0 % | 22 |
| UntilizeWithUnpaddingDeviceOperation | 475 | 3.8 % | 46 |
| TilizeWithValPaddingDeviceOperation | 472 | 3.8 % | 45 |
| SliceDeviceOperation | 346 | 2.8 % | 81 |
| PermuteDeviceOperation | 246 | 2.0 % | 16 |
| ReduceScatterDeviceOperation | 189 | 1.5 % | 3 |
| CopyDeviceOperation | 176 | 1.4 % | 2 |

### Chip 5 (1,0): CCL ops in order

| ID | op | module | device us |
|---|---|---|---|
| 1582 | ReduceScatterDeviceOperation | ATTN/CCL_ALL_REDUCE | 51 |
| 1583 | AllGatherDeviceOperation | ATTN/CCL_ALL_REDUCE | 16 |
| 1819 | AllGatherDeviceOperation | MOE/EXPERTS/CCL_ALL_GATHER | 20 |
| 1820 | AllGatherDeviceOperation | MOE/EXPERTS/CCL_ALL_GATHER | 7 |
| 1866 | ReduceScatterDeviceOperation | MOE/EXPERTS/CCL_REDUCE_SCATTER | 25 |
| 1876 | ReduceScatterDeviceOperation | MOE/CCL_ALL_REDUCE | 113 |
| 1877 | AllGatherDeviceOperation | MOE/CCL_ALL_REDUCE | 20 |
