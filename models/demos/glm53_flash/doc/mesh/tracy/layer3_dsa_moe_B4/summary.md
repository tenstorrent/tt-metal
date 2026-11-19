# layer 3 (DSA+MoE, experts bfp8) on the (2,4) block, B=4/row (8 users), position 1024 (seed-only cache), one traced replay (2026_10_20_02_21_14)

## Device time per chip (traced decode window, one replay)

| chip | (r,c) | ops | device ms/step | CCL kernel ms | CCL composite ms |
|---|---|---|---|---|---|
| 1 | (0,0) | 462 | 7.785 | 0.230 | 0.309 |
| 0 | (0,1) | 462 | 7.783 | 0.131 | 0.210 |
| 2 | (0,2) | 462 | 7.781 | 0.123 | 0.202 |
| 3 | (0,3) | 462 | 7.782 | 0.323 | 0.402 |
| 5 | (1,0) | 462 | 7.780 | 0.221 | 0.299 |
| 4 | (1,1) | 462 | 7.775 | 0.123 | 0.202 |
| 6 | (1,2) | 462 | 7.771 | 0.113 | 0.192 |
| 7 | (1,3) | 462 | 7.773 | 0.315 | 0.393 |

## Chip 1 (0,0): shares of the 7.785 ms step

| group | device us | % of step | ops |
|---|---|---|---|
| CCL ops (AllGather/AllBroadcast/ReduceScatter/AllReduce kernels) | 230 | 3.0 % | 7 |
| CCL composites (all_reduce/all_gather/reduce_scatter incl. local reduce ops) | 309 | 4.0 % | 23 |
| sparse_matmul (expert matmuls) | 802 | 10.3 % | 2 |
| EXPERTS local (router-weight select, sparse_matmul, slicing; no CCL) | 1326 | 17.0 % | 22 |
| ROUTER (exact fp32 composite top-8) | 3396 | 43.6 % | 147 |
| SHARED expert | 89 | 1.1 % | 8 |
| MoE total (router + experts + shared + CCL + add) | 5084 | 65.3 % | 199 |
| ATTN composite incl. its all_reduce (KDA recurrence / DSA flash-MLA + index) | 738 | 9.5 % | 41 |
| ATTN local only (no CCL) | 699 | 9.0 % | 39 |
| NORM (RMSNorm x2) | 93 | 1.2 % | 2 |
| MHC (hyper-connections x2, everything outside the sublayers) | 1869 | 24.0 % | 220 |

### Chip 1 (0,0): top-10 ops

| ID | op | module | device us | % of step |
|---|---|---|---|---|
| 1748 | SparseMatmulDeviceOperation active=?/72 x 32 x 4096 x 2048 | MOE/EXPERTS | 591 | 7.6 % |
| 1710 | AccumulationDeviceOperation | MOE/ROUTER | 331 | 4.3 % |
| 1625 | AccumulationDeviceOperation | MOE/ROUTER | 331 | 4.3 % |
| 1693 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.2 % |
| 1676 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.2 % |
| 1592 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.2 % |
| 1659 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.2 % |
| 1642 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.2 % |
| 1608 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.2 % |
| 1760 | SparseMatmulDeviceOperation active=?/72 x 32 x 1024 x 4096 | MOE/EXPERTS | 211 | 2.7 % |

### Chip 1 (0,0): by op class

| op class | device us | % of step | count |
|---|---|---|---|
| AccumulationDeviceOperation | 2644 | 34.0 % | 8 |
| MatmulDeviceOperation | 894 | 11.5 % | 14 |
| SparseMatmulDeviceOperation | 802 | 10.3 % | 2 |
| TernaryDeviceOperation | 610 | 7.8 % | 38 |
| BinaryNgDeviceOperation | 589 | 7.6 % | 92 |
| ReduceDeviceOperation | 320 | 4.1 % | 29 |
| FillPadDeviceOperation | 284 | 3.6 % | 32 |
| UnaryDeviceOperation | 246 | 3.2 % | 12 |
| SliceDeviceOperation | 205 | 2.6 % | 74 |
| ReduceScatterDeviceOperation | 168 | 2.2 % | 3 |
| TilizeWithValPaddingDeviceOperation | 163 | 2.1 % | 45 |
| PermuteDeviceOperation | 138 | 1.8 % | 16 |
| LayerNormDeviceOperation | 126 | 1.6 % | 5 |
| UntilizeWithUnpaddingDeviceOperation | 108 | 1.4 % | 48 |

### Chip 1 (0,0): CCL ops in order

| ID | op | module | device us |
|---|---|---|---|
| 1472 | ReduceScatterDeviceOperation | ATTN/CCL_ALL_REDUCE | 25 |
| 1473 | AllGatherDeviceOperation | ATTN/CCL_ALL_REDUCE | 14 |
| 1732 | AllBroadcastDeviceOperation | MOE/EXPERTS/CCL_ALL_GATHER | 27 |
| 1737 | AllBroadcastDeviceOperation | MOE/EXPERTS/CCL_ALL_GATHER | 6 |
| 1771 | ReduceScatterDeviceOperation | MOE/EXPERTS/CCL_REDUCE_SCATTER | 20 |
| 1782 | ReduceScatterDeviceOperation | MOE/CCL_ALL_REDUCE | 122 |
| 1783 | AllGatherDeviceOperation | MOE/CCL_ALL_REDUCE | 15 |

## Chip 5 (1,0): shares of the 7.780 ms step

| group | device us | % of step | ops |
|---|---|---|---|
| CCL ops (AllGather/AllBroadcast/ReduceScatter/AllReduce kernels) | 221 | 2.8 % | 7 |
| CCL composites (all_reduce/all_gather/reduce_scatter incl. local reduce ops) | 299 | 3.8 % | 23 |
| sparse_matmul (expert matmuls) | 803 | 10.3 % | 2 |
| EXPERTS local (router-weight select, sparse_matmul, slicing; no CCL) | 1328 | 17.1 % | 22 |
| ROUTER (exact fp32 composite top-8) | 3396 | 43.6 % | 147 |
| SHARED expert | 89 | 1.1 % | 8 |
| MoE total (router + experts + shared + CCL + add) | 5075 | 65.2 % | 199 |
| ATTN composite incl. its all_reduce (KDA recurrence / DSA flash-MLA + index) | 739 | 9.5 % | 41 |
| ATTN local only (no CCL) | 699 | 9.0 % | 39 |
| NORM (RMSNorm x2) | 94 | 1.2 % | 2 |
| MHC (hyper-connections x2, everything outside the sublayers) | 1873 | 24.1 % | 220 |

### Chip 5 (1,0): top-10 ops

| ID | op | module | device us | % of step |
|---|---|---|---|---|
| 1748 | SparseMatmulDeviceOperation active=?/72 x 32 x 4096 x 2048 | MOE/EXPERTS | 591 | 7.6 % |
| 1625 | AccumulationDeviceOperation | MOE/ROUTER | 331 | 4.3 % |
| 1710 | AccumulationDeviceOperation | MOE/ROUTER | 331 | 4.3 % |
| 1676 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.2 % |
| 1693 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.2 % |
| 1642 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.2 % |
| 1608 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.2 % |
| 1592 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.2 % |
| 1659 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.2 % |
| 1760 | SparseMatmulDeviceOperation active=?/72 x 32 x 1024 x 4096 | MOE/EXPERTS | 212 | 2.7 % |

### Chip 5 (1,0): by op class

| op class | device us | % of step | count |
|---|---|---|---|
| AccumulationDeviceOperation | 2644 | 34.0 % | 8 |
| MatmulDeviceOperation | 894 | 11.5 % | 14 |
| SparseMatmulDeviceOperation | 803 | 10.3 % | 2 |
| TernaryDeviceOperation | 611 | 7.9 % | 38 |
| BinaryNgDeviceOperation | 590 | 7.6 % | 92 |
| ReduceDeviceOperation | 320 | 4.1 % | 29 |
| FillPadDeviceOperation | 284 | 3.6 % | 32 |
| UnaryDeviceOperation | 247 | 3.2 % | 12 |
| SliceDeviceOperation | 206 | 2.6 % | 74 |
| ReduceScatterDeviceOperation | 166 | 2.1 % | 3 |
| TilizeWithValPaddingDeviceOperation | 163 | 2.1 % | 45 |
| PermuteDeviceOperation | 137 | 1.8 % | 16 |
| LayerNormDeviceOperation | 126 | 1.6 % | 5 |
| UntilizeWithUnpaddingDeviceOperation | 108 | 1.4 % | 48 |

### Chip 5 (1,0): CCL ops in order

| ID | op | module | device us |
|---|---|---|---|
| 1472 | ReduceScatterDeviceOperation | ATTN/CCL_ALL_REDUCE | 25 |
| 1473 | AllGatherDeviceOperation | ATTN/CCL_ALL_REDUCE | 15 |
| 1732 | AllBroadcastDeviceOperation | MOE/EXPERTS/CCL_ALL_GATHER | 17 |
| 1737 | AllBroadcastDeviceOperation | MOE/EXPERTS/CCL_ALL_GATHER | 6 |
| 1771 | ReduceScatterDeviceOperation | MOE/EXPERTS/CCL_REDUCE_SCATTER | 20 |
| 1782 | ReduceScatterDeviceOperation | MOE/CCL_ALL_REDUCE | 121 |
| 1783 | AllGatherDeviceOperation | MOE/CCL_ALL_REDUCE | 16 |
