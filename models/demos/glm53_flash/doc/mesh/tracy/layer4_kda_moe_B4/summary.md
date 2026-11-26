# layer 4 (KDA+MoE, experts bfp8) on the (2,4) block, B=4/row (8 users), position 1024, one traced replay (2026_10_20_02_06_14)

## Device time per chip (traced decode window, one replay)

| chip | (r,c) | ops | device ms/step | CCL kernel ms | CCL composite ms |
|---|---|---|---|---|---|
| 1 | (0,0) | 498 | 7.985 | 0.135 | 0.214 |
| 0 | (0,1) | 498 | 7.984 | 0.122 | 0.201 |
| 2 | (0,2) | 498 | 7.983 | 0.129 | 0.208 |
| 3 | (0,3) | 498 | 7.984 | 0.375 | 0.454 |
| 5 | (1,0) | 498 | 7.974 | 0.123 | 0.202 |
| 4 | (1,1) | 498 | 7.977 | 0.116 | 0.196 |
| 6 | (1,2) | 498 | 7.978 | 0.128 | 0.206 |
| 7 | (1,3) | 498 | 7.975 | 0.367 | 0.445 |

## Chip 1 (0,0): shares of the 7.985 ms step

| group | device us | % of step | ops |
|---|---|---|---|
| CCL ops (AllGather/AllBroadcast/ReduceScatter/AllReduce kernels) | 135 | 1.7 % | 7 |
| CCL composites (all_reduce/all_gather/reduce_scatter incl. local reduce ops) | 214 | 2.7 % | 23 |
| sparse_matmul (expert matmuls) | 899 | 11.3 % | 2 |
| EXPERTS local (router-weight select, sparse_matmul, slicing; no CCL) | 1424 | 17.8 % | 22 |
| ROUTER (exact fp32 composite top-8) | 3391 | 42.5 % | 147 |
| SHARED expert | 89 | 1.1 % | 8 |
| MoE total (router + experts + shared + CCL + add) | 5080 | 63.6 % | 199 |
| ATTN composite incl. its all_reduce (KDA recurrence / DSA flash-MLA + index) | 928 | 11.6 % | 77 |
| ATTN local only (no CCL) | 886 | 11.1 % | 75 |
| NORM (RMSNorm x2) | 93 | 1.2 % | 2 |
| MHC (hyper-connections x2, everything outside the sublayers) | 1885 | 23.6 % | 220 |

### Chip 1 (0,0): top-10 ops

| ID | op | module | device us | % of step |
|---|---|---|---|---|
| 1911 | SparseMatmulDeviceOperation active=?/72 x 32 x 4096 x 2048 | MOE/EXPERTS | 670 | 8.4 % |
| 1788 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.1 % |
| 1839 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.1 % |
| 1771 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.1 % |
| 1755 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.1 % |
| 1873 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.1 % |
| 1856 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.1 % |
| 1805 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.1 % |
| 1822 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.1 % |
| 1923 | SparseMatmulDeviceOperation active=?/72 x 32 x 1024 x 4096 | MOE/EXPERTS | 229 | 2.9 % |

### Chip 1 (0,0): by op class

| op class | device us | % of step | count |
|---|---|---|---|
| AccumulationDeviceOperation | 2642 | 33.1 % | 8 |
| SparseMatmulDeviceOperation | 899 | 11.3 % | 2 |
| BinaryNgDeviceOperation | 845 | 10.6 % | 110 |
| MatmulDeviceOperation | 676 | 8.5 % | 13 |
| TernaryDeviceOperation | 622 | 7.8 % | 38 |
| ReduceDeviceOperation | 350 | 4.4 % | 30 |
| FillPadDeviceOperation | 277 | 3.5 % | 31 |
| UnaryDeviceOperation | 267 | 3.3 % | 17 |
| SliceDeviceOperation | 221 | 2.8 % | 79 |
| TilizeWithValPaddingDeviceOperation | 208 | 2.6 % | 48 |
| ReshapeViewDeviceOperation | 139 | 1.7 % | 15 |
| PermuteDeviceOperation | 136 | 1.7 % | 16 |
| UntilizeWithUnpaddingDeviceOperation | 124 | 1.5 % | 51 |
| LayerNormDeviceOperation | 106 | 1.3 % | 5 |

### Chip 1 (0,0): CCL ops in order

| ID | op | module | device us |
|---|---|---|---|
| 1633 | ReduceScatterDeviceOperation | ATTN/CCL_ALL_REDUCE | 27 |
| 1634 | AllGatherDeviceOperation | ATTN/CCL_ALL_REDUCE | 14 |
| 1895 | AllBroadcastDeviceOperation | MOE/EXPERTS/CCL_ALL_GATHER | 24 |
| 1900 | AllBroadcastDeviceOperation | MOE/EXPERTS/CCL_ALL_GATHER | 6 |
| 1934 | ReduceScatterDeviceOperation | MOE/EXPERTS/CCL_REDUCE_SCATTER | 19 |
| 1945 | ReduceScatterDeviceOperation | MOE/CCL_ALL_REDUCE | 23 |
| 1946 | AllGatherDeviceOperation | MOE/CCL_ALL_REDUCE | 20 |

## Chip 5 (1,0): shares of the 7.974 ms step

| group | device us | % of step | ops |
|---|---|---|---|
| CCL ops (AllGather/AllBroadcast/ReduceScatter/AllReduce kernels) | 123 | 1.5 % | 7 |
| CCL composites (all_reduce/all_gather/reduce_scatter incl. local reduce ops) | 202 | 2.5 % | 23 |
| sparse_matmul (expert matmuls) | 901 | 11.3 % | 2 |
| EXPERTS local (router-weight select, sparse_matmul, slicing; no CCL) | 1424 | 17.9 % | 22 |
| ROUTER (exact fp32 composite top-8) | 3392 | 42.5 % | 147 |
| SHARED expert | 89 | 1.1 % | 8 |
| MoE total (router + experts + shared + CCL + add) | 5073 | 63.6 % | 199 |
| ATTN composite incl. its all_reduce (KDA recurrence / DSA flash-MLA + index) | 924 | 11.6 % | 77 |
| ATTN local only (no CCL) | 887 | 11.1 % | 75 |
| NORM (RMSNorm x2) | 93 | 1.2 % | 2 |
| MHC (hyper-connections x2, everything outside the sublayers) | 1884 | 23.6 % | 220 |

### Chip 5 (1,0): top-10 ops

| ID | op | module | device us | % of step |
|---|---|---|---|---|
| 1910 | SparseMatmulDeviceOperation active=?/72 x 32 x 4096 x 2048 | MOE/EXPERTS | 670 | 8.4 % |
| 1787 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.1 % |
| 1838 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.1 % |
| 1770 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.1 % |
| 1804 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.1 % |
| 1754 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.1 % |
| 1872 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.1 % |
| 1855 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.1 % |
| 1821 | AccumulationDeviceOperation | MOE/ROUTER | 330 | 4.1 % |
| 1922 | SparseMatmulDeviceOperation active=?/72 x 32 x 1024 x 4096 | MOE/EXPERTS | 230 | 2.9 % |

### Chip 5 (1,0): by op class

| op class | device us | % of step | count |
|---|---|---|---|
| AccumulationDeviceOperation | 2642 | 33.1 % | 8 |
| SparseMatmulDeviceOperation | 901 | 11.3 % | 2 |
| BinaryNgDeviceOperation | 846 | 10.6 % | 110 |
| MatmulDeviceOperation | 677 | 8.5 % | 13 |
| TernaryDeviceOperation | 620 | 7.8 % | 38 |
| ReduceDeviceOperation | 352 | 4.4 % | 30 |
| FillPadDeviceOperation | 278 | 3.5 % | 31 |
| UnaryDeviceOperation | 264 | 3.3 % | 17 |
| SliceDeviceOperation | 222 | 2.8 % | 79 |
| TilizeWithValPaddingDeviceOperation | 208 | 2.6 % | 48 |
| ReshapeViewDeviceOperation | 137 | 1.7 % | 15 |
| PermuteDeviceOperation | 136 | 1.7 % | 16 |
| UntilizeWithUnpaddingDeviceOperation | 124 | 1.6 % | 51 |
| LayerNormDeviceOperation | 106 | 1.3 % | 5 |

### Chip 5 (1,0): CCL ops in order

| ID | op | module | device us |
|---|---|---|---|
| 1632 | ReduceScatterDeviceOperation | ATTN/CCL_ALL_REDUCE | 21 |
| 1633 | AllGatherDeviceOperation | ATTN/CCL_ALL_REDUCE | 16 |
| 1894 | AllBroadcastDeviceOperation | MOE/EXPERTS/CCL_ALL_GATHER | 17 |
| 1899 | AllBroadcastDeviceOperation | MOE/EXPERTS/CCL_ALL_GATHER | 6 |
| 1933 | ReduceScatterDeviceOperation | MOE/EXPERTS/CCL_REDUCE_SCATTER | 21 |
| 1944 | ReduceScatterDeviceOperation | MOE/CCL_ALL_REDUCE | 23 |
| 1945 | AllGatherDeviceOperation | MOE/CCL_ALL_REDUCE | 20 |
