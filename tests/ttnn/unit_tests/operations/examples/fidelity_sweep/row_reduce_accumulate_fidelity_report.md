# row_reduce_accumulate — math-fidelity x DEST sweep (single core, distribution=signal)

box=bgd-lab-t3003-special-mstaletovic-for-reservation-40918  arch=WH_B0  cores=1  N=5 (median)  kernel-iters=200
input=bf16, output=fp32. DEST: fp32 = fp32_dest_acc_en ON (precision bf16-fp32), bf16 = OFF (bf16-bf16).
note: l1_accum forces fp32 DEST (packer L1-acc is fp32-DEST-only); its 'bf16 DEST' column rounds only the L1 accumulator CB.

## Perf — median ns per row-mean (rows: method x fidelity; one block per DEST)

### DEST=fp32

| method.fidelity | 1t | 8t | 32t |
|---|---|---|---|
| reduce_fold.LoFi | 366 | 835 | 2398 |
| reduce_fold.HiFi2 | 375 | 840 | 2406 |
| reduce_fold.HiFi3 | 419 | 1183 | 3780 |
| reduce_fold.HiFi4 | 470 | 1569 | 5306 |
| l1_accum.LoFi | 758 | 1755 | 5096 |
| l1_accum.HiFi2 | 768 | 1782 | 5105 |
| l1_accum.HiFi3 | 783 | 1780 | 5115 |
| l1_accum.HiFi4 | 816 | 1811 | 5142 |
| dest_accum_pairs.LoFi | 771 | 971 | 1745 |
| dest_accum_pairs.HiFi2 | 779 | 987 | 1764 |
| dest_accum_pairs.HiFi3 | 795 | 995 | 1772 |
| dest_accum_pairs.HiFi4 | 823 | 1025 | 1806 |
| dest_accum_pairs_sfpu.LoFi | 898 | 1074 | 1855 |
| dest_accum_pairs_sfpu.HiFi2 | 906 | 1074 | 1857 |
| dest_accum_pairs_sfpu.HiFi3 | 906 | 1075 | 1860 |
| dest_accum_pairs_sfpu.HiFi4 | 909 | 1078 | 1858 |

### DEST=bf16

| method.fidelity | 1t | 8t | 32t |
|---|---|---|---|
| reduce_fold.LoFi | 348 | 819 | 2380 |
| reduce_fold.HiFi2 | 361 | 827 | 2393 |
| reduce_fold.HiFi3 | 406 | 1170 | 3764 |
| reduce_fold.HiFi4 | 450 | 1554 | 5302 |
| l1_accum.LoFi | 638 | 1316 | 3363 |
| l1_accum.HiFi2 | 648 | 1308 | 3364 |
| l1_accum.HiFi3 | 690 | 1319 | 3367 |
| l1_accum.HiFi4 | 740 | 1346 | 3401 |
| dest_accum_pairs.LoFi | 651 | 832 | 1613 |
| dest_accum_pairs.HiFi2 | 660 | 845 | 1623 |
| dest_accum_pairs.HiFi3 | 686 | 887 | 1666 |
| dest_accum_pairs.HiFi4 | 726 | 935 | 1716 |
| dest_accum_pairs_sfpu.LoFi | 888 | 1058 | 1840 |
| dest_accum_pairs_sfpu.HiFi2 | 896 | 1055 | 1839 |
| dest_accum_pairs_sfpu.HiFi3 | 890 | 1061 | 1841 |
| dest_accum_pairs_sfpu.HiFi4 | 894 | 1057 | 1843 |

## Accuracy — error vs fp64 mean (cell = max_abs | max ULP_bf16); rows: method x fidelity

### DEST=fp32

| method.fidelity | 1t | 8t | 32t |
|---|---|---|---|
| reduce_fold.LoFi | 1.2e-02 \| 0.9u | 1.3e-02 \| 0.7u | 1.2e-02 \| 0.8u |
| reduce_fold.HiFi2 | 1.2e-02 \| 0.9u | 1.3e-02 \| 0.7u | 1.2e-02 \| 0.8u |
| reduce_fold.HiFi3 | 1.5e-03 \| 0.2u | 7.4e-04 \| 0.0u | 2.6e-04 \| 0.0u |
| reduce_fold.HiFi4 | 1.5e-03 \| 0.2u | 7.4e-04 \| 0.0u | 2.6e-04 \| 0.0u |
| l1_accum.LoFi | 1.2e-02 \| 0.9u | 2.0e-02 \| 1.2u | 1.7e-02 \| 1.4u |
| l1_accum.HiFi2 | 1.2e-02 \| 0.9u | 2.0e-02 \| 1.2u | 1.7e-02 \| 1.4u |
| l1_accum.HiFi3 | 1.5e-03 \| 0.2u | 1.6e-03 \| 0.1u | 1.4e-03 \| 0.1u |
| l1_accum.HiFi4 | 1.5e-03 \| 0.2u | 1.6e-03 \| 0.1u | 1.4e-03 \| 0.1u |
| dest_accum_pairs.LoFi | 1.2e-02 \| 0.9u | 2.0e-02 \| 1.2u | 1.7e-02 \| 1.4u |
| dest_accum_pairs.HiFi2 | 1.2e-02 \| 0.9u | 2.0e-02 \| 1.2u | 1.7e-02 \| 1.4u |
| dest_accum_pairs.HiFi3 | 1.5e-03 \| 0.2u | 1.6e-03 \| 0.1u | 1.4e-03 \| 0.1u |
| dest_accum_pairs.HiFi4 | 1.5e-03 \| 0.2u | 1.6e-03 \| 0.1u | 1.4e-03 \| 0.1u |
| dest_accum_pairs_sfpu.LoFi | 1.5e-03 \| 0.2u | 7.4e-04 \| 0.0u | 2.6e-04 \| 0.0u |
| dest_accum_pairs_sfpu.HiFi2 | 1.5e-03 \| 0.2u | 7.4e-04 \| 0.0u | 2.6e-04 \| 0.0u |
| dest_accum_pairs_sfpu.HiFi3 | 1.5e-03 \| 0.2u | 7.4e-04 \| 0.0u | 2.6e-04 \| 0.0u |
| dest_accum_pairs_sfpu.HiFi4 | 1.5e-03 \| 0.2u | 7.4e-04 \| 0.0u | 2.6e-04 \| 0.0u |

### DEST=bf16

| method.fidelity | 1t | 8t | 32t |
|---|---|---|---|
| reduce_fold.LoFi | 1.3e-02 \| 1.4u | 2.9e-02 \| 1.8u | 2.1e-01 \| 13.3u |
| reduce_fold.HiFi2 | 1.3e-02 \| 1.4u | 2.9e-02 \| 1.8u | 2.1e-01 \| 13.3u |
| reduce_fold.HiFi3 | 1.1e-02 \| 1.4u | 2.9e-02 \| 1.8u | 2.1e-01 \| 13.3u |
| reduce_fold.HiFi4 | 1.1e-02 \| 1.4u | 2.9e-02 \| 1.8u | 2.1e-01 \| 13.3u |
| l1_accum.LoFi | 1.2e-02 \| 0.9u | 1.3e-02 \| 0.6u | 1.3e-02 \| 0.8u |
| l1_accum.HiFi2 | 1.2e-02 \| 0.9u | 1.3e-02 \| 0.6u | 1.3e-02 \| 0.8u |
| l1_accum.HiFi3 | 1.5e-03 \| 0.2u | 4.5e-03 \| 0.3u | 3.7e-03 \| 0.2u |
| l1_accum.HiFi4 | 1.5e-03 \| 0.2u | 4.5e-03 \| 0.3u | 3.7e-03 \| 0.2u |
| dest_accum_pairs.LoFi | 1.3e-02 \| 1.4u | 2.0e-02 \| 0.7u | 1.3e-02 \| 1.0u |
| dest_accum_pairs.HiFi2 | 1.3e-02 \| 1.4u | 2.0e-02 \| 0.7u | 1.3e-02 \| 1.0u |
| dest_accum_pairs.HiFi3 | 1.1e-02 \| 1.4u | 2.7e-02 \| 0.9u | 2.1e-02 \| 1.4u |
| dest_accum_pairs.HiFi4 | 1.1e-02 \| 1.4u | 2.7e-02 \| 0.9u | 2.1e-02 \| 1.4u |
| dest_accum_pairs_sfpu.LoFi | 2.8e-02 \| 0.9u | 1.2e-02 \| 0.8u | 8.2e-03 \| 1.0u |
| dest_accum_pairs_sfpu.HiFi2 | 2.8e-02 \| 0.9u | 1.2e-02 \| 0.8u | 8.2e-03 \| 1.0u |
| dest_accum_pairs_sfpu.HiFi3 | 2.8e-02 \| 0.9u | 1.2e-02 \| 0.8u | 8.2e-03 \| 1.0u |
| dest_accum_pairs_sfpu.HiFi4 | 2.8e-02 \| 0.9u | 1.2e-02 \| 0.8u | 8.2e-03 \| 1.0u |
