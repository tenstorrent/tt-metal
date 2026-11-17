# genify_matmul precision matrix results

- Last run: 2026-11-17 (Refinement 1), Blackhole p300c. 540/540 pass.
- Test: `test_genify_matmul_precision_matrix.py`. It covers interleaved DRAM with epilogue none, and the output dtype is A's dtype.
- Gates:
  - HiFi4: `helpers.TOLERANCES[(out dtype, fp32_dest_acc_en)]` on PCC and relative RMS.
  - Below HiFi4: PCC ≥ `eval.metrics.LOWER_FIDELITY_PCC` only.
- Skipped / excluded: float32 × `fp32_dest_acc_en=False` is the op's permanent EXCLUSION. The test asserts that it raises `ExcludedCell`.
- "rand" = zero-centred uniform (`torch.rand - 0.5`). Relative RMS = RMS error / std(reference).
- With 16-bit DEST the K chain inside DEST is capped at `DEST16_BLOCK_K_MAX = 8` tiles. See changelog Refinement 1.

| A x B | A (=out) | B | fp32 DEST | fidelity | dist | PCC | rel-RMS | max abs | got/true median |
|---|---|---|---|---|---|---|---|---|---|
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT16 | True | HiFi4 | randn | 0.999998 | 0.00174 | 0.0078 | 0.9999 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT16 | True | HiFi4 | rand | 0.999999 | 0.00164 | 0.0008 | 1.0000 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT16 | True | HiFi2 | randn | 0.999994 | 0.00455 | 0.0179 | 0.9969 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT16 | True | HiFi2 | rand | 0.999995 | 0.00404 | 0.0013 | 0.9972 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT16 | True | LoFi | randn | 0.999906 | 0.02551 | 0.1076 | 0.9788 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT16 | True | LoFi | rand | 0.999892 | 0.02671 | 0.0082 | 0.9770 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT16 | False | HiFi4 | randn | 0.999992 | 0.00421 | 0.0216 | 1.0014 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT16 | False | HiFi4 | rand | 0.999993 | 0.00393 | 0.0016 | 1.0016 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT16 | False | HiFi2 | randn | 0.999991 | 0.00470 | 0.0216 | 0.9979 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT16 | False | HiFi2 | rand | 0.999992 | 0.00423 | 0.0016 | 0.9983 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT16 | False | LoFi | randn | 0.999906 | 0.02518 | 0.1076 | 0.9793 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT16 | False | LoFi | rand | 0.999893 | 0.02647 | 0.0082 | 0.9775 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT8_B | True | HiFi4 | randn | 0.999998 | 0.00177 | 0.0083 | 1.0000 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT8_B | True | HiFi4 | rand | 0.999999 | 0.00169 | 0.0010 | 1.0000 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT8_B | True | HiFi2 | randn | 0.999994 | 0.00456 | 0.0227 | 0.9970 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT8_B | True | HiFi2 | rand | 0.999995 | 0.00400 | 0.0015 | 0.9973 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT8_B | True | LoFi | randn | 0.999910 | 0.02139 | 0.0807 | 0.9846 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT8_B | True | LoFi | rand | 0.999907 | 0.02261 | 0.0079 | 0.9820 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT8_B | False | HiFi4 | randn | 0.999992 | 0.00414 | 0.0225 | 1.0014 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT8_B | False | HiFi4 | rand | 0.999992 | 0.00414 | 0.0021 | 1.0013 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT8_B | False | HiFi2 | randn | 0.999989 | 0.00509 | 0.0240 | 0.9976 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT8_B | False | HiFi2 | rand | 0.999991 | 0.00451 | 0.0018 | 0.9981 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT8_B | False | LoFi | randn | 0.999909 | 0.02121 | 0.0807 | 0.9848 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT8_B | False | LoFi | rand | 0.999905 | 0.02264 | 0.0076 | 0.9823 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT4_B | True | HiFi4 | randn | 0.999999 | 0.00168 | 0.0076 | 1.0001 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT4_B | True | HiFi4 | rand | 0.999999 | 0.00159 | 0.0008 | 1.0000 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT4_B | True | HiFi2 | randn | 0.999994 | 0.00463 | 0.0211 | 0.9968 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT4_B | True | HiFi2 | rand | 0.999995 | 0.00403 | 0.0015 | 0.9975 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT4_B | True | LoFi | randn | 0.999994 | 0.00463 | 0.0211 | 0.9968 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT4_B | True | LoFi | rand | 0.999995 | 0.00403 | 0.0015 | 0.9975 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT4_B | False | HiFi4 | randn | 0.999995 | 0.00316 | 0.0160 | 1.0008 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT4_B | False | HiFi4 | rand | 0.999995 | 0.00319 | 0.0019 | 1.0009 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT4_B | False | HiFi2 | randn | 0.999994 | 0.00448 | 0.0211 | 0.9971 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT4_B | False | HiFi2 | rand | 0.999994 | 0.00411 | 0.0019 | 0.9977 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT4_B | False | LoFi | randn | 0.999994 | 0.00448 | 0.0211 | 0.9971 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT16 | BFLOAT4_B | False | LoFi | rand | 0.999994 | 0.00411 | 0.0019 | 0.9977 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT16 | True | HiFi4 | randn | 0.999966 | 0.00857 | 0.0235 | 1.0035 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT16 | True | HiFi4 | rand | 0.999971 | 0.00813 | 0.0028 | 1.0036 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT16 | True | HiFi2 | randn | 0.999966 | 0.00857 | 0.0235 | 1.0035 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT16 | True | HiFi2 | rand | 0.999971 | 0.00813 | 0.0028 | 1.0036 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT16 | True | LoFi | randn | 0.999883 | 0.02204 | 0.0727 | 0.9852 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT16 | True | LoFi | rand | 0.999869 | 0.02381 | 0.0076 | 0.9831 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT16 | False | HiFi4 | randn | 0.999964 | 0.01083 | 0.0333 | 1.0079 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT16 | False | HiFi4 | rand | 0.999970 | 0.01027 | 0.0038 | 1.0074 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT16 | False | HiFi2 | randn | 0.999964 | 0.01083 | 0.0333 | 1.0079 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT16 | False | HiFi2 | rand | 0.999970 | 0.01027 | 0.0038 | 1.0074 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT16 | False | LoFi | randn | 0.999880 | 0.02007 | 0.0727 | 0.9880 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT16 | False | LoFi | rand | 0.999869 | 0.02167 | 0.0069 | 0.9862 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT8_B | True | HiFi4 | randn | 0.999969 | 0.00834 | 0.0237 | 1.0031 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT8_B | True | HiFi4 | rand | 0.999973 | 0.00796 | 0.0029 | 1.0033 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT8_B | True | HiFi2 | randn | 0.999969 | 0.00834 | 0.0237 | 1.0031 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT8_B | True | HiFi2 | rand | 0.999973 | 0.00796 | 0.0029 | 1.0033 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT8_B | True | LoFi | randn | 0.999885 | 0.01912 | 0.0708 | 0.9891 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT8_B | True | LoFi | rand | 0.999882 | 0.02038 | 0.0070 | 0.9878 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT8_B | False | HiFi4 | randn | 0.999965 | 0.01041 | 0.0363 | 1.0066 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT8_B | False | HiFi4 | rand | 0.999969 | 0.01015 | 0.0034 | 1.0072 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT8_B | False | HiFi2 | randn | 0.999965 | 0.01041 | 0.0363 | 1.0066 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT8_B | False | HiFi2 | rand | 0.999969 | 0.01015 | 0.0034 | 1.0072 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT8_B | False | LoFi | randn | 0.999885 | 0.01719 | 0.0708 | 0.9934 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT8_B | False | LoFi | rand | 0.999886 | 0.01824 | 0.0062 | 0.9908 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT4_B | True | HiFi4 | randn | 0.999971 | 0.00817 | 0.0234 | 1.0037 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT4_B | True | HiFi4 | rand | 0.999971 | 0.00815 | 0.0029 | 1.0039 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT4_B | True | HiFi2 | randn | 0.999971 | 0.00817 | 0.0234 | 1.0037 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT4_B | True | HiFi2 | rand | 0.999971 | 0.00815 | 0.0029 | 1.0039 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT4_B | True | LoFi | randn | 0.999971 | 0.00817 | 0.0234 | 1.0037 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT4_B | True | LoFi | rand | 0.999971 | 0.00815 | 0.0029 | 1.0039 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT4_B | False | HiFi4 | randn | 0.999969 | 0.00994 | 0.0312 | 1.0066 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT4_B | False | HiFi4 | rand | 0.999972 | 0.00957 | 0.0032 | 1.0063 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT4_B | False | HiFi2 | randn | 0.999969 | 0.00994 | 0.0312 | 1.0066 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT4_B | False | HiFi2 | rand | 0.999972 | 0.00957 | 0.0032 | 1.0063 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT4_B | False | LoFi | randn | 0.999969 | 0.00994 | 0.0312 | 1.0066 |
| (1, 1, 32, 32)x(32, 32) | BFLOAT8_B | BFLOAT4_B | False | LoFi | rand | 0.999972 | 0.00957 | 0.0032 | 1.0063 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT16 | True | HiFi4 | randn | 1.000000 | 0.00049 | 0.0019 | 0.9997 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT16 | True | HiFi4 | rand | 1.000000 | 0.00040 | 0.0001 | 0.9997 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT16 | True | HiFi2 | randn | 0.999995 | 0.00618 | 0.0251 | 0.9946 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT16 | True | HiFi2 | rand | 0.999996 | 0.00564 | 0.0017 | 0.9953 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT16 | True | LoFi | randn | 0.999904 | 0.02739 | 0.1048 | 0.9767 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT16 | True | LoFi | rand | 0.999895 | 0.02855 | 0.0084 | 0.9749 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT8_B | True | HiFi4 | randn | 1.000000 | 0.00049 | 0.0021 | 0.9997 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT8_B | True | HiFi4 | rand | 1.000000 | 0.00040 | 0.0001 | 0.9997 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT8_B | True | HiFi2 | randn | 0.999995 | 0.00618 | 0.0253 | 0.9946 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT8_B | True | HiFi2 | rand | 0.999996 | 0.00563 | 0.0016 | 0.9953 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT8_B | True | LoFi | randn | 0.999909 | 0.02314 | 0.0869 | 0.9822 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT8_B | True | LoFi | rand | 0.999910 | 0.02444 | 0.0078 | 0.9797 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT4_B | True | HiFi4 | randn | 1.000000 | 0.00045 | 0.0020 | 0.9997 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT4_B | True | HiFi4 | rand | 1.000000 | 0.00038 | 0.0001 | 0.9997 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT4_B | True | HiFi2 | randn | 0.999995 | 0.00618 | 0.0274 | 0.9947 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT4_B | True | HiFi2 | rand | 0.999996 | 0.00568 | 0.0016 | 0.9952 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT4_B | True | LoFi | randn | 0.999995 | 0.00618 | 0.0274 | 0.9947 |
| (1, 1, 32, 32)x(32, 32) | FLOAT32 | BFLOAT4_B | True | LoFi | rand | 0.999996 | 0.00568 | 0.0016 | 0.9952 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT16 | True | HiFi4 | randn | 0.999999 | 0.00168 | 0.0079 | 1.0000 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT16 | True | HiFi4 | rand | 0.999999 | 0.00166 | 0.0009 | 1.0000 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT16 | True | HiFi2 | randn | 0.999994 | 0.00451 | 0.0198 | 0.9970 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT16 | True | HiFi2 | rand | 0.999995 | 0.00384 | 0.0012 | 0.9978 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT16 | True | LoFi | randn | 0.999900 | 0.02644 | 0.0862 | 0.9781 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT16 | True | LoFi | rand | 0.999900 | 0.02705 | 0.0087 | 0.9773 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT16 | False | HiFi4 | randn | 0.999919 | 0.01683 | 0.0882 | 1.0123 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT16 | False | HiFi4 | rand | 0.999919 | 0.01638 | 0.0081 | 1.0113 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT16 | False | HiFi2 | randn | 0.999923 | 0.01546 | 0.0882 | 1.0103 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT16 | False | HiFi2 | rand | 0.999922 | 0.01553 | 0.0081 | 1.0104 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT16 | False | LoFi | randn | 0.999861 | 0.02207 | 0.0763 | 0.9855 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT16 | False | LoFi | rand | 0.999859 | 0.02281 | 0.0089 | 0.9842 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT8_B | True | HiFi4 | randn | 0.999999 | 0.00172 | 0.0080 | 1.0001 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT8_B | True | HiFi4 | rand | 0.999999 | 0.00163 | 0.0008 | 1.0000 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT8_B | True | HiFi2 | randn | 0.999994 | 0.00445 | 0.0172 | 0.9971 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT8_B | True | HiFi2 | rand | 0.999995 | 0.00397 | 0.0016 | 0.9977 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT8_B | True | LoFi | randn | 0.999897 | 0.02193 | 0.0761 | 0.9835 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT8_B | True | LoFi | rand | 0.999902 | 0.02296 | 0.0074 | 0.9819 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT8_B | False | HiFi4 | randn | 0.999926 | 0.01579 | 0.0559 | 1.0107 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT8_B | False | HiFi4 | rand | 0.999933 | 0.01552 | 0.0051 | 1.0114 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT8_B | False | HiFi2 | randn | 0.999929 | 0.01460 | 0.0557 | 1.0091 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT8_B | False | HiFi2 | rand | 0.999932 | 0.01477 | 0.0048 | 1.0100 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT8_B | False | LoFi | randn | 0.999856 | 0.01912 | 0.0813 | 0.9914 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT8_B | False | LoFi | rand | 0.999870 | 0.01914 | 0.0077 | 0.9892 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT4_B | True | HiFi4 | randn | 0.999999 | 0.00168 | 0.0078 | 0.9999 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT4_B | True | HiFi4 | rand | 0.999999 | 0.00164 | 0.0007 | 0.9999 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT4_B | True | HiFi2 | randn | 0.999994 | 0.00463 | 0.0193 | 0.9969 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT4_B | True | HiFi2 | rand | 0.999996 | 0.00380 | 0.0015 | 0.9977 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT4_B | True | LoFi | randn | 0.999994 | 0.00463 | 0.0193 | 0.9969 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT4_B | True | LoFi | rand | 0.999996 | 0.00380 | 0.0015 | 0.9977 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT4_B | False | HiFi4 | randn | 0.999954 | 0.01207 | 0.0652 | 1.0064 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT4_B | False | HiFi4 | rand | 0.999958 | 0.01163 | 0.0073 | 1.0061 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT4_B | False | HiFi2 | randn | 0.999955 | 0.01105 | 0.0496 | 1.0048 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT4_B | False | HiFi2 | rand | 0.999957 | 0.01095 | 0.0054 | 1.0048 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT4_B | False | LoFi | randn | 0.999955 | 0.01105 | 0.0496 | 1.0048 |
| (1, 2048)x(2048, 1536) | BFLOAT16 | BFLOAT4_B | False | LoFi | rand | 0.999957 | 0.01095 | 0.0054 | 1.0048 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT16 | True | HiFi4 | randn | 0.999969 | 0.00838 | 0.0235 | 1.0034 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT16 | True | HiFi4 | rand | 0.999972 | 0.00785 | 0.0024 | 1.0032 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT16 | True | HiFi2 | randn | 0.999969 | 0.00838 | 0.0235 | 1.0034 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT16 | True | HiFi2 | rand | 0.999972 | 0.00785 | 0.0024 | 1.0032 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT16 | True | LoFi | randn | 0.999870 | 0.02351 | 0.0830 | 0.9841 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT16 | True | LoFi | rand | 0.999874 | 0.02403 | 0.0082 | 0.9831 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT16 | False | HiFi4 | randn | 0.999900 | 0.02318 | 0.0896 | 1.0192 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT16 | False | HiFi4 | rand | 0.999889 | 0.02244 | 0.0096 | 1.0195 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT16 | False | HiFi2 | randn | 0.999900 | 0.02318 | 0.0896 | 1.0192 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT16 | False | HiFi2 | rand | 0.999889 | 0.02244 | 0.0096 | 1.0195 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT16 | False | LoFi | randn | 0.999826 | 0.01971 | 0.0801 | 0.9944 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT16 | False | LoFi | rand | 0.999839 | 0.01966 | 0.0080 | 0.9935 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT8_B | True | HiFi4 | randn | 0.999969 | 0.00842 | 0.0236 | 1.0033 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT8_B | True | HiFi4 | rand | 0.999972 | 0.00792 | 0.0028 | 1.0032 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT8_B | True | HiFi2 | randn | 0.999969 | 0.00842 | 0.0236 | 1.0033 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT8_B | True | HiFi2 | rand | 0.999972 | 0.00792 | 0.0028 | 1.0032 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT8_B | True | LoFi | randn | 0.999878 | 0.01942 | 0.0681 | 0.9888 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT8_B | True | LoFi | rand | 0.999879 | 0.02046 | 0.0064 | 0.9876 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT8_B | False | HiFi4 | randn | 0.999895 | 0.02238 | 0.0848 | 1.0178 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT8_B | False | HiFi4 | rand | 0.999905 | 0.02161 | 0.0059 | 1.0202 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT8_B | False | HiFi2 | randn | 0.999895 | 0.02238 | 0.0848 | 1.0178 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT8_B | False | HiFi2 | rand | 0.999905 | 0.02161 | 0.0059 | 1.0202 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT8_B | False | LoFi | randn | 0.999831 | 0.01864 | 0.0690 | 0.9999 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT8_B | False | LoFi | rand | 0.999846 | 0.01782 | 0.0052 | 0.9982 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT4_B | True | HiFi4 | randn | 0.999969 | 0.00839 | 0.0231 | 1.0034 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT4_B | True | HiFi4 | rand | 0.999971 | 0.00810 | 0.0027 | 1.0038 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT4_B | True | HiFi2 | randn | 0.999969 | 0.00839 | 0.0231 | 1.0034 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT4_B | True | HiFi2 | rand | 0.999971 | 0.00810 | 0.0027 | 1.0038 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT4_B | True | LoFi | randn | 0.999969 | 0.00839 | 0.0231 | 1.0034 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT4_B | True | LoFi | rand | 0.999971 | 0.00810 | 0.0027 | 1.0038 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT4_B | False | HiFi4 | randn | 0.999925 | 0.01891 | 0.0836 | 1.0147 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT4_B | False | HiFi4 | rand | 0.999938 | 0.01751 | 0.0069 | 1.0130 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT4_B | False | HiFi2 | randn | 0.999925 | 0.01891 | 0.0836 | 1.0147 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT4_B | False | HiFi2 | rand | 0.999938 | 0.01751 | 0.0069 | 1.0130 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT4_B | False | LoFi | randn | 0.999925 | 0.01891 | 0.0836 | 1.0147 |
| (1, 2048)x(2048, 1536) | BFLOAT8_B | BFLOAT4_B | False | LoFi | rand | 0.999938 | 0.01751 | 0.0069 | 1.0130 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT16 | True | HiFi4 | randn | 1.000000 | 0.00049 | 0.0019 | 0.9997 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT16 | True | HiFi4 | rand | 1.000000 | 0.00039 | 0.0001 | 0.9997 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT16 | True | HiFi2 | randn | 0.999994 | 0.00630 | 0.0192 | 0.9947 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT16 | True | HiFi2 | rand | 0.999996 | 0.00569 | 0.0022 | 0.9951 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT16 | True | LoFi | randn | 0.999899 | 0.02844 | 0.0947 | 0.9762 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT16 | True | LoFi | rand | 0.999901 | 0.02898 | 0.0093 | 0.9744 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT8_B | True | HiFi4 | randn | 1.000000 | 0.00049 | 0.0018 | 0.9997 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT8_B | True | HiFi4 | rand | 1.000000 | 0.00040 | 0.0001 | 0.9997 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT8_B | True | HiFi2 | randn | 0.999994 | 0.00631 | 0.0192 | 0.9948 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT8_B | True | HiFi2 | rand | 0.999996 | 0.00569 | 0.0022 | 0.9950 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT8_B | True | LoFi | randn | 0.999897 | 0.02367 | 0.0794 | 0.9815 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT8_B | True | LoFi | rand | 0.999903 | 0.02485 | 0.0078 | 0.9795 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT4_B | True | HiFi4 | randn | 1.000000 | 0.00044 | 0.0014 | 0.9997 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT4_B | True | HiFi4 | rand | 1.000000 | 0.00038 | 0.0001 | 0.9997 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT4_B | True | HiFi2 | randn | 0.999994 | 0.00631 | 0.0194 | 0.9947 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT4_B | True | HiFi2 | rand | 0.999996 | 0.00570 | 0.0022 | 0.9951 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT4_B | True | LoFi | randn | 0.999994 | 0.00631 | 0.0194 | 0.9947 |
| (1, 2048)x(2048, 1536) | FLOAT32 | BFLOAT4_B | True | LoFi | rand | 0.999996 | 0.00570 | 0.0022 | 0.9951 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT16 | True | HiFi4 | randn | 0.999999 | 0.00169 | 0.0156 | 1.0000 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT16 | True | HiFi4 | rand | 0.999999 | 0.00167 | 0.0010 | 1.0000 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT16 | True | HiFi2 | randn | 0.999994 | 0.00438 | 0.0266 | 0.9972 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT16 | True | HiFi2 | rand | 0.999995 | 0.00396 | 0.0021 | 0.9975 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT16 | True | LoFi | randn | 0.999900 | 0.02610 | 0.1268 | 0.9779 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT16 | True | LoFi | rand | 0.999902 | 0.02599 | 0.0105 | 0.9779 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT16 | False | HiFi4 | randn | 0.999921 | 0.01624 | 0.1288 | 1.0111 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT16 | False | HiFi4 | rand | 0.999921 | 0.01628 | 0.0111 | 1.0113 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT16 | False | HiFi2 | randn | 0.999923 | 0.01529 | 0.1132 | 1.0095 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT16 | False | HiFi2 | rand | 0.999922 | 0.01560 | 0.0103 | 1.0101 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT16 | False | LoFi | randn | 0.999864 | 0.02219 | 0.1378 | 0.9849 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT16 | False | LoFi | rand | 0.999866 | 0.02204 | 0.0103 | 0.9850 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT8_B | True | HiFi4 | randn | 0.999999 | 0.00169 | 0.0156 | 1.0000 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT8_B | True | HiFi4 | rand | 0.999999 | 0.00167 | 0.0010 | 1.0000 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT8_B | True | HiFi2 | randn | 0.999994 | 0.00439 | 0.0263 | 0.9972 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT8_B | True | HiFi2 | rand | 0.999995 | 0.00396 | 0.0022 | 0.9975 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT8_B | True | LoFi | randn | 0.999904 | 0.02120 | 0.1001 | 0.9838 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT8_B | True | LoFi | rand | 0.999905 | 0.02236 | 0.0089 | 0.9822 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT8_B | False | HiFi4 | randn | 0.999926 | 0.01531 | 0.1076 | 1.0100 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT8_B | False | HiFi4 | rand | 0.999923 | 0.01571 | 0.0104 | 1.0107 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT8_B | False | HiFi2 | randn | 0.999928 | 0.01436 | 0.1076 | 1.0083 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT8_B | False | HiFi2 | rand | 0.999925 | 0.01504 | 0.0095 | 1.0095 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT8_B | False | LoFi | randn | 0.999868 | 0.01857 | 0.1035 | 0.9908 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT8_B | False | LoFi | rand | 0.999869 | 0.01930 | 0.0148 | 0.9892 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT4_B | True | HiFi4 | randn | 0.999999 | 0.00167 | 0.0146 | 1.0000 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT4_B | True | HiFi4 | rand | 0.999999 | 0.00166 | 0.0010 | 1.0000 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT4_B | True | HiFi2 | randn | 0.999994 | 0.00440 | 0.0297 | 0.9971 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT4_B | True | HiFi2 | rand | 0.999995 | 0.00397 | 0.0021 | 0.9975 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT4_B | True | LoFi | randn | 0.999994 | 0.00440 | 0.0297 | 0.9971 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT4_B | True | LoFi | rand | 0.999995 | 0.00397 | 0.0021 | 0.9975 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT4_B | False | HiFi4 | randn | 0.999957 | 0.01149 | 0.1307 | 1.0058 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT4_B | False | HiFi4 | rand | 0.999958 | 0.01124 | 0.0111 | 1.0057 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT4_B | False | HiFi2 | randn | 0.999958 | 0.01054 | 0.1006 | 1.0041 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT4_B | False | HiFi2 | rand | 0.999959 | 0.01053 | 0.0111 | 1.0044 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT4_B | False | LoFi | randn | 0.999958 | 0.01054 | 0.1006 | 1.0041 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT16 | BFLOAT4_B | False | LoFi | rand | 0.999959 | 0.01053 | 0.0111 | 1.0044 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT16 | True | HiFi4 | randn | 0.999970 | 0.00819 | 0.0453 | 1.0031 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT16 | True | HiFi4 | rand | 0.999972 | 0.00785 | 0.0029 | 1.0034 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT16 | True | HiFi2 | randn | 0.999970 | 0.00819 | 0.0453 | 1.0031 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT16 | True | HiFi2 | rand | 0.999972 | 0.00785 | 0.0029 | 1.0034 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT16 | True | LoFi | randn | 0.999876 | 0.02297 | 0.1235 | 0.9838 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT16 | True | LoFi | rand | 0.999879 | 0.02307 | 0.0104 | 0.9838 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT16 | False | HiFi4 | randn | 0.999894 | 0.02248 | 0.1558 | 1.0191 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT16 | False | HiFi4 | rand | 0.999895 | 0.02243 | 0.0130 | 1.0196 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT16 | False | HiFi2 | randn | 0.999894 | 0.02248 | 0.1558 | 1.0191 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT16 | False | HiFi2 | rand | 0.999895 | 0.02243 | 0.0130 | 1.0196 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT16 | False | LoFi | randn | 0.999839 | 0.01921 | 0.1066 | 0.9940 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT16 | False | LoFi | rand | 0.999843 | 0.01910 | 0.0099 | 0.9939 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT8_B | True | HiFi4 | randn | 0.999970 | 0.00819 | 0.0454 | 1.0031 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT8_B | True | HiFi4 | rand | 0.999972 | 0.00786 | 0.0030 | 1.0033 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT8_B | True | HiFi2 | randn | 0.999970 | 0.00819 | 0.0454 | 1.0031 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT8_B | True | HiFi2 | rand | 0.999972 | 0.00786 | 0.0030 | 1.0033 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT8_B | True | LoFi | randn | 0.999879 | 0.01888 | 0.0931 | 0.9899 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT8_B | True | LoFi | rand | 0.999882 | 0.01994 | 0.0105 | 0.9880 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT8_B | False | HiFi4 | randn | 0.999901 | 0.02145 | 0.1248 | 1.0179 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT8_B | False | HiFi4 | rand | 0.999897 | 0.02183 | 0.0121 | 1.0189 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT8_B | False | HiFi2 | randn | 0.999901 | 0.02145 | 0.1248 | 1.0179 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT8_B | False | HiFi2 | rand | 0.999897 | 0.02183 | 0.0121 | 1.0189 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT8_B | False | LoFi | randn | 0.999843 | 0.01784 | 0.1083 | 1.0001 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT8_B | False | LoFi | rand | 0.999846 | 0.01781 | 0.0094 | 0.9981 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT4_B | True | HiFi4 | randn | 0.999970 | 0.00820 | 0.0461 | 1.0031 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT4_B | True | HiFi4 | rand | 0.999972 | 0.00785 | 0.0029 | 1.0034 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT4_B | True | HiFi2 | randn | 0.999970 | 0.00820 | 0.0461 | 1.0031 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT4_B | True | HiFi2 | rand | 0.999972 | 0.00785 | 0.0029 | 1.0034 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT4_B | True | LoFi | randn | 0.999970 | 0.00820 | 0.0461 | 1.0031 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT4_B | True | LoFi | rand | 0.999972 | 0.00785 | 0.0029 | 1.0034 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT4_B | False | HiFi4 | randn | 0.999931 | 0.01815 | 0.1214 | 1.0141 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT4_B | False | HiFi4 | rand | 0.999936 | 0.01766 | 0.0136 | 1.0139 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT4_B | False | HiFi2 | randn | 0.999931 | 0.01815 | 0.1214 | 1.0141 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT4_B | False | HiFi2 | rand | 0.999936 | 0.01766 | 0.0136 | 1.0139 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT4_B | False | LoFi | randn | 0.999931 | 0.01815 | 0.1214 | 1.0141 |
| (1, 384, 1536)x(1536, 1024) | BFLOAT8_B | BFLOAT4_B | False | LoFi | rand | 0.999936 | 0.01766 | 0.0136 | 1.0139 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT16 | True | HiFi4 | randn | 1.000000 | 0.00066 | 0.0038 | 0.9995 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT16 | True | HiFi4 | rand | 1.000000 | 0.00058 | 0.0003 | 0.9996 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT16 | True | HiFi2 | randn | 0.999994 | 0.00674 | 0.0320 | 0.9942 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT16 | True | HiFi2 | rand | 0.999995 | 0.00602 | 0.0022 | 0.9948 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT16 | True | LoFi | randn | 0.999900 | 0.02860 | 0.1310 | 0.9749 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT16 | True | LoFi | rand | 0.999902 | 0.02824 | 0.0107 | 0.9753 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT8_B | True | HiFi4 | randn | 1.000000 | 0.00049 | 0.0026 | 0.9997 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT8_B | True | HiFi4 | rand | 1.000000 | 0.00039 | 0.0002 | 0.9997 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT8_B | True | HiFi2 | randn | 0.999994 | 0.00659 | 0.0312 | 0.9944 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT8_B | True | HiFi2 | rand | 0.999995 | 0.00587 | 0.0022 | 0.9950 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT8_B | True | LoFi | randn | 0.999903 | 0.02336 | 0.1057 | 0.9810 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT8_B | True | LoFi | rand | 0.999905 | 0.02436 | 0.0098 | 0.9797 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT4_B | True | HiFi4 | randn | 1.000000 | 0.00045 | 0.0022 | 0.9997 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT4_B | True | HiFi4 | rand | 1.000000 | 0.00038 | 0.0001 | 0.9997 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT4_B | True | HiFi2 | randn | 0.999994 | 0.00662 | 0.0304 | 0.9944 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT4_B | True | HiFi2 | rand | 0.999995 | 0.00590 | 0.0022 | 0.9950 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT4_B | True | LoFi | randn | 0.999994 | 0.00662 | 0.0304 | 0.9944 |
| (1, 384, 1536)x(1536, 1024) | FLOAT32 | BFLOAT4_B | True | LoFi | rand | 0.999995 | 0.00590 | 0.0022 | 0.9950 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT16 | True | HiFi4 | randn | 0.999999 | 0.00172 | 0.0118 | 0.9999 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT16 | True | HiFi4 | rand | 0.999999 | 0.00169 | 0.0011 | 0.9998 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT16 | True | HiFi2 | randn | 0.999994 | 0.00448 | 0.0285 | 0.9970 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT16 | True | HiFi2 | rand | 0.999995 | 0.00408 | 0.0020 | 0.9973 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT16 | True | LoFi | randn | 0.999899 | 0.02640 | 0.1177 | 0.9775 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT16 | True | LoFi | rand | 0.999919 | 0.02391 | 0.0088 | 0.9795 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT16 | False | HiFi4 | randn | 0.999920 | 0.01647 | 0.1011 | 1.0110 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT16 | False | HiFi4 | rand | 0.999922 | 0.01602 | 0.0094 | 1.0107 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT16 | False | HiFi2 | randn | 0.999921 | 0.01553 | 0.1011 | 1.0093 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT16 | False | HiFi2 | rand | 0.999922 | 0.01535 | 0.0094 | 1.0096 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT16 | False | LoFi | randn | 0.999854 | 0.02248 | 0.1210 | 0.9854 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT16 | False | LoFi | rand | 0.999876 | 0.02013 | 0.0111 | 0.9874 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT8_B | True | HiFi4 | randn | 0.999999 | 0.00170 | 0.0152 | 1.0000 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT8_B | True | HiFi4 | rand | 0.999999 | 0.00166 | 0.0010 | 1.0000 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT8_B | True | HiFi2 | randn | 0.999994 | 0.00439 | 0.0221 | 0.9972 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT8_B | True | HiFi2 | rand | 0.999995 | 0.00395 | 0.0021 | 0.9975 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT8_B | True | LoFi | randn | 0.999903 | 0.02096 | 0.1031 | 0.9841 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT8_B | True | LoFi | rand | 0.999925 | 0.02096 | 0.0079 | 0.9828 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT8_B | False | HiFi4 | randn | 0.999925 | 0.01572 | 0.1041 | 1.0100 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT8_B | False | HiFi4 | rand | 0.999923 | 0.01570 | 0.0097 | 1.0101 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT8_B | False | HiFi2 | randn | 0.999926 | 0.01478 | 0.1030 | 1.0083 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT8_B | False | HiFi2 | rand | 0.999924 | 0.01505 | 0.0097 | 1.0090 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT8_B | False | LoFi | randn | 0.999858 | 0.01883 | 0.0997 | 0.9918 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT8_B | False | LoFi | rand | 0.999881 | 0.01812 | 0.0082 | 0.9905 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT4_B | True | HiFi4 | randn | 0.999999 | 0.00167 | 0.0103 | 1.0000 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT4_B | True | HiFi4 | rand | 0.999999 | 0.00166 | 0.0010 | 1.0000 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT4_B | True | HiFi2 | randn | 0.999994 | 0.00440 | 0.0232 | 0.9972 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT4_B | True | HiFi2 | rand | 0.999995 | 0.00397 | 0.0019 | 0.9975 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT4_B | True | LoFi | randn | 0.999994 | 0.00440 | 0.0232 | 0.9972 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT4_B | True | LoFi | rand | 0.999995 | 0.00397 | 0.0019 | 0.9975 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT4_B | False | HiFi4 | randn | 0.999949 | 0.01309 | 0.1062 | 1.0065 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT4_B | False | HiFi4 | rand | 0.999950 | 0.01254 | 0.0091 | 1.0063 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT4_B | False | HiFi2 | randn | 0.999949 | 0.01200 | 0.0906 | 1.0048 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT4_B | False | HiFi2 | rand | 0.999951 | 0.01173 | 0.0091 | 1.0052 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT4_B | False | LoFi | randn | 0.999949 | 0.01200 | 0.0906 | 1.0048 |
| (2, 64, 4096)x(4096, 512) | BFLOAT16 | BFLOAT4_B | False | LoFi | rand | 0.999951 | 0.01173 | 0.0091 | 1.0052 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT16 | True | HiFi4 | randn | 0.999970 | 0.00819 | 0.0398 | 1.0031 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT16 | True | HiFi4 | rand | 0.999972 | 0.00784 | 0.0029 | 1.0032 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT16 | True | HiFi2 | randn | 0.999970 | 0.00819 | 0.0398 | 1.0031 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT16 | True | HiFi2 | rand | 0.999972 | 0.00784 | 0.0029 | 1.0032 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT16 | True | LoFi | randn | 0.999875 | 0.02309 | 0.1082 | 0.9838 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT16 | True | LoFi | rand | 0.999897 | 0.02095 | 0.0089 | 0.9855 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT16 | False | HiFi4 | randn | 0.999894 | 0.02254 | 0.1163 | 1.0191 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT16 | False | HiFi4 | rand | 0.999898 | 0.02180 | 0.0121 | 1.0187 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT16 | False | HiFi2 | randn | 0.999894 | 0.02254 | 0.1163 | 1.0191 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT16 | False | HiFi2 | rand | 0.999898 | 0.02180 | 0.0121 | 1.0187 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT16 | False | LoFi | randn | 0.999828 | 0.01980 | 0.1165 | 0.9945 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT16 | False | LoFi | rand | 0.999852 | 0.01796 | 0.0102 | 0.9962 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT8_B | True | HiFi4 | randn | 0.999970 | 0.00822 | 0.0411 | 1.0032 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT8_B | True | HiFi4 | rand | 0.999972 | 0.00786 | 0.0029 | 1.0033 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT8_B | True | HiFi2 | randn | 0.999970 | 0.00822 | 0.0411 | 1.0032 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT8_B | True | HiFi2 | rand | 0.999972 | 0.00786 | 0.0029 | 1.0033 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT8_B | True | LoFi | randn | 0.999879 | 0.01871 | 0.0861 | 0.9903 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT8_B | True | LoFi | rand | 0.999902 | 0.01851 | 0.0076 | 0.9887 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT8_B | False | HiFi4 | randn | 0.999899 | 0.02186 | 0.1262 | 1.0178 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT8_B | False | HiFi4 | rand | 0.999898 | 0.02167 | 0.0129 | 1.0184 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT8_B | False | HiFi2 | randn | 0.999899 | 0.02186 | 0.1262 | 1.0178 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT8_B | False | HiFi2 | rand | 0.999898 | 0.02167 | 0.0129 | 1.0184 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT8_B | False | LoFi | randn | 0.999835 | 0.01848 | 0.0991 | 1.0010 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT8_B | False | LoFi | rand | 0.999857 | 0.01712 | 0.0078 | 0.9996 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT4_B | True | HiFi4 | randn | 0.999970 | 0.00819 | 0.0344 | 1.0032 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT4_B | True | HiFi4 | rand | 0.999972 | 0.00786 | 0.0029 | 1.0033 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT4_B | True | HiFi2 | randn | 0.999970 | 0.00819 | 0.0344 | 1.0032 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT4_B | True | HiFi2 | rand | 0.999972 | 0.00786 | 0.0029 | 1.0033 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT4_B | True | LoFi | randn | 0.999970 | 0.00819 | 0.0344 | 1.0032 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT4_B | True | LoFi | rand | 0.999972 | 0.00786 | 0.0029 | 1.0033 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT4_B | False | HiFi4 | randn | 0.999924 | 0.01939 | 0.1156 | 1.0147 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT4_B | False | HiFi4 | rand | 0.999927 | 0.01911 | 0.0092 | 1.0146 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT4_B | False | HiFi2 | randn | 0.999924 | 0.01939 | 0.1156 | 1.0147 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT4_B | False | HiFi2 | rand | 0.999927 | 0.01911 | 0.0092 | 1.0146 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT4_B | False | LoFi | randn | 0.999924 | 0.01939 | 0.1156 | 1.0147 |
| (2, 64, 4096)x(4096, 512) | BFLOAT8_B | BFLOAT4_B | False | LoFi | rand | 0.999927 | 0.01911 | 0.0092 | 1.0146 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT16 | True | HiFi4 | randn | 1.000000 | 0.00066 | 0.0037 | 0.9995 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT16 | True | HiFi4 | rand | 1.000000 | 0.00058 | 0.0003 | 0.9996 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT16 | True | HiFi2 | randn | 0.999994 | 0.00674 | 0.0301 | 0.9942 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT16 | True | HiFi2 | rand | 0.999995 | 0.00604 | 0.0024 | 0.9948 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT16 | True | LoFi | randn | 0.999899 | 0.02876 | 0.1246 | 0.9748 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT16 | True | LoFi | rand | 0.999919 | 0.02603 | 0.0096 | 0.9771 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT8_B | True | HiFi4 | randn | 1.000000 | 0.00066 | 0.0035 | 0.9995 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT8_B | True | HiFi4 | rand | 1.000000 | 0.00058 | 0.0003 | 0.9996 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT8_B | True | HiFi2 | randn | 0.999994 | 0.00674 | 0.0297 | 0.9942 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT8_B | True | HiFi2 | rand | 0.999995 | 0.00604 | 0.0024 | 0.9948 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT8_B | True | LoFi | randn | 0.999903 | 0.02325 | 0.0981 | 0.9812 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT8_B | True | LoFi | rand | 0.999925 | 0.02316 | 0.0082 | 0.9803 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT4_B | True | HiFi4 | randn | 1.000000 | 0.00045 | 0.0020 | 0.9996 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT4_B | True | HiFi4 | rand | 1.000000 | 0.00037 | 0.0001 | 0.9997 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT4_B | True | HiFi2 | randn | 0.999994 | 0.00661 | 0.0286 | 0.9944 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT4_B | True | HiFi2 | rand | 0.999995 | 0.00590 | 0.0023 | 0.9950 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT4_B | True | LoFi | randn | 0.999994 | 0.00661 | 0.0286 | 0.9944 |
| (2, 64, 4096)x(4096, 512) | FLOAT32 | BFLOAT4_B | True | LoFi | rand | 0.999995 | 0.00590 | 0.0023 | 0.9950 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT16 | True | HiFi4 | randn | 0.999999 | 0.00170 | 0.0099 | 0.9998 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT16 | True | HiFi4 | rand | 0.999999 | 0.00169 | 0.0010 | 0.9998 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT16 | True | HiFi2 | randn | 0.999994 | 0.00456 | 0.0222 | 0.9970 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT16 | True | HiFi2 | rand | 0.999995 | 0.00412 | 0.0018 | 0.9973 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT16 | True | LoFi | randn | 0.999902 | 0.02623 | 0.1062 | 0.9773 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT16 | True | LoFi | rand | 0.999921 | 0.02372 | 0.0086 | 0.9797 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT16 | False | HiFi4 | randn | 0.999896 | 0.01856 | 0.1203 | 1.0099 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT16 | False | HiFi4 | rand | 0.999895 | 0.01820 | 0.0122 | 1.0100 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT16 | False | HiFi2 | randn | 0.999896 | 0.01774 | 0.1102 | 1.0085 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT16 | False | HiFi2 | rand | 0.999894 | 0.01764 | 0.0111 | 1.0087 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT16 | False | LoFi | randn | 0.999825 | 0.02421 | 0.1349 | 0.9854 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT16 | False | LoFi | rand | 0.999841 | 0.02189 | 0.0098 | 0.9877 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT8_B | True | HiFi4 | randn | 0.999999 | 0.00170 | 0.0089 | 0.9998 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT8_B | True | HiFi4 | rand | 0.999999 | 0.00171 | 0.0010 | 0.9998 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT8_B | True | HiFi2 | randn | 0.999994 | 0.00452 | 0.0272 | 0.9970 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT8_B | True | HiFi2 | rand | 0.999995 | 0.00411 | 0.0019 | 0.9973 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT8_B | True | LoFi | randn | 0.999907 | 0.02106 | 0.0895 | 0.9838 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT8_B | True | LoFi | rand | 0.999926 | 0.02107 | 0.0072 | 0.9825 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT8_B | False | HiFi4 | randn | 0.999897 | 0.01872 | 0.1368 | 1.0094 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT8_B | False | HiFi4 | rand | 0.999894 | 0.01829 | 0.0102 | 1.0095 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT8_B | False | HiFi2 | randn | 0.999898 | 0.01778 | 0.1211 | 1.0077 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT8_B | False | HiFi2 | rand | 0.999896 | 0.01752 | 0.0090 | 1.0087 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT8_B | False | LoFi | randn | 0.999827 | 0.02110 | 0.1087 | 0.9923 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT8_B | False | LoFi | rand | 0.999849 | 0.02001 | 0.0084 | 0.9910 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT4_B | True | HiFi4 | randn | 0.999999 | 0.00172 | 0.0091 | 0.9998 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT4_B | True | HiFi4 | rand | 0.999999 | 0.00167 | 0.0010 | 0.9998 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT4_B | True | HiFi2 | randn | 0.999994 | 0.00451 | 0.0236 | 0.9970 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT4_B | True | HiFi2 | rand | 0.999995 | 0.00407 | 0.0020 | 0.9974 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT4_B | True | LoFi | randn | 0.999994 | 0.00451 | 0.0236 | 0.9970 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT4_B | True | LoFi | rand | 0.999995 | 0.00407 | 0.0020 | 0.9974 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT4_B | False | HiFi4 | randn | 0.999914 | 0.01796 | 0.1177 | 1.0069 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT4_B | False | HiFi4 | rand | 0.999916 | 0.01658 | 0.0078 | 1.0068 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT4_B | False | HiFi2 | randn | 0.999915 | 0.01608 | 0.1188 | 1.0051 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT4_B | False | HiFi2 | rand | 0.999914 | 0.01542 | 0.0093 | 1.0054 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT4_B | False | LoFi | randn | 0.999915 | 0.01608 | 0.1188 | 1.0051 |
| (1, 64, 16384)x(16384, 256) | BFLOAT16 | BFLOAT4_B | False | LoFi | rand | 0.999914 | 0.01542 | 0.0093 | 1.0054 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT16 | True | HiFi4 | randn | 0.999970 | 0.00815 | 0.0343 | 1.0030 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT16 | True | HiFi4 | rand | 0.999972 | 0.00774 | 0.0029 | 1.0030 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT16 | True | HiFi2 | randn | 0.999970 | 0.00815 | 0.0343 | 1.0030 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT16 | True | HiFi2 | rand | 0.999972 | 0.00774 | 0.0029 | 1.0030 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT16 | True | LoFi | randn | 0.999876 | 0.02311 | 0.1299 | 0.9837 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT16 | True | LoFi | rand | 0.999897 | 0.02086 | 0.0079 | 0.9857 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT16 | False | HiFi4 | randn | 0.999872 | 0.02370 | 0.1201 | 1.0178 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT16 | False | HiFi4 | rand | 0.999871 | 0.02354 | 0.0135 | 1.0181 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT16 | False | HiFi2 | randn | 0.999872 | 0.02370 | 0.1201 | 1.0178 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT16 | False | HiFi2 | rand | 0.999871 | 0.02354 | 0.0135 | 1.0181 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT16 | False | LoFi | randn | 0.999800 | 0.02182 | 0.1096 | 0.9946 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT16 | False | LoFi | rand | 0.999821 | 0.01989 | 0.0098 | 0.9972 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT8_B | True | HiFi4 | randn | 0.999969 | 0.00820 | 0.0344 | 1.0029 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT8_B | True | HiFi4 | rand | 0.999972 | 0.00776 | 0.0029 | 1.0029 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT8_B | True | HiFi2 | randn | 0.999969 | 0.00820 | 0.0344 | 1.0029 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT8_B | True | HiFi2 | rand | 0.999972 | 0.00776 | 0.0029 | 1.0029 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT8_B | True | LoFi | randn | 0.999882 | 0.01867 | 0.0817 | 0.9899 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT8_B | True | LoFi | rand | 0.999904 | 0.01852 | 0.0072 | 0.9884 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT8_B | False | HiFi4 | randn | 0.999869 | 0.02466 | 0.1335 | 1.0170 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT8_B | False | HiFi4 | rand | 0.999874 | 0.02352 | 0.0123 | 1.0177 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT8_B | False | HiFi2 | randn | 0.999869 | 0.02466 | 0.1335 | 1.0170 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT8_B | False | HiFi2 | rand | 0.999874 | 0.02352 | 0.0123 | 1.0177 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT8_B | False | LoFi | randn | 0.999801 | 0.02096 | 0.1067 | 1.0012 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT8_B | False | LoFi | rand | 0.999822 | 0.01946 | 0.0080 | 0.9997 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT4_B | True | HiFi4 | randn | 0.999969 | 0.00817 | 0.0429 | 1.0027 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT4_B | True | HiFi4 | rand | 0.999971 | 0.00791 | 0.0029 | 1.0031 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT4_B | True | HiFi2 | randn | 0.999969 | 0.00817 | 0.0429 | 1.0027 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT4_B | True | HiFi2 | rand | 0.999971 | 0.00791 | 0.0029 | 1.0031 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT4_B | True | LoFi | randn | 0.999969 | 0.00817 | 0.0429 | 1.0027 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT4_B | True | LoFi | rand | 0.999971 | 0.00791 | 0.0029 | 1.0031 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT4_B | False | HiFi4 | randn | 0.999888 | 0.02322 | 0.1370 | 1.0152 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT4_B | False | HiFi4 | rand | 0.999891 | 0.02366 | 0.0108 | 1.0148 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT4_B | False | HiFi2 | randn | 0.999888 | 0.02322 | 0.1370 | 1.0152 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT4_B | False | HiFi2 | rand | 0.999891 | 0.02366 | 0.0108 | 1.0148 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT4_B | False | LoFi | randn | 0.999888 | 0.02322 | 0.1370 | 1.0152 |
| (1, 64, 16384)x(16384, 256) | BFLOAT8_B | BFLOAT4_B | False | LoFi | rand | 0.999891 | 0.02366 | 0.0108 | 1.0148 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT16 | True | HiFi4 | randn | 1.000000 | 0.00073 | 0.0036 | 0.9994 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT16 | True | HiFi4 | rand | 1.000000 | 0.00067 | 0.0003 | 0.9994 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT16 | True | HiFi2 | randn | 0.999994 | 0.00685 | 0.0297 | 0.9941 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT16 | True | HiFi2 | rand | 0.999995 | 0.00608 | 0.0024 | 0.9948 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT16 | True | LoFi | randn | 0.999902 | 0.02861 | 0.1222 | 0.9746 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT16 | True | LoFi | rand | 0.999921 | 0.02582 | 0.0092 | 0.9772 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT8_B | True | HiFi4 | randn | 1.000000 | 0.00073 | 0.0037 | 0.9994 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT8_B | True | HiFi4 | rand | 1.000000 | 0.00065 | 0.0003 | 0.9995 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT8_B | True | HiFi2 | randn | 0.999994 | 0.00683 | 0.0283 | 0.9941 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT8_B | True | HiFi2 | rand | 0.999995 | 0.00606 | 0.0023 | 0.9948 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT8_B | True | LoFi | randn | 0.999906 | 0.02325 | 0.0902 | 0.9810 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT8_B | True | LoFi | rand | 0.999927 | 0.02312 | 0.0078 | 0.9800 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT4_B | True | HiFi4 | randn | 1.000000 | 0.00071 | 0.0032 | 0.9994 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT4_B | True | HiFi4 | rand | 1.000000 | 0.00066 | 0.0002 | 0.9994 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT4_B | True | HiFi2 | randn | 0.999994 | 0.00686 | 0.0295 | 0.9941 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT4_B | True | HiFi2 | rand | 0.999995 | 0.00609 | 0.0023 | 0.9947 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT4_B | True | LoFi | randn | 0.999994 | 0.00686 | 0.0295 | 0.9941 |
| (1, 64, 16384)x(16384, 256) | FLOAT32 | BFLOAT4_B | True | LoFi | rand | 0.999995 | 0.00609 | 0.0023 | 0.9947 |
