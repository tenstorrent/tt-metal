# Precision sweep (fp32 accumulation, fidelity, performance policy)

Device ms: median of 3 forward_device calls after a 0.3 s idle. Metrics: `golden_io.head_metrics` / `final_metrics`; `spread_ratio` per active row of the golden.

## A0 hifi2, no fp32 acc (round 0)

policy: `{"weights": "bf16", "bf16_roles": [], "matmul": "hifi2, fp32_acc=False", "sdpa": "hifi2, fp32_acc=False", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1370.7 | 0.99987 | 0.99963 | 0.99307 | 0.94917 | 0.39713 | 0.89133 | 1.04659 | 1.10951 | 0.99965 |  | spread_ratio_min |
| long_context | 180.9 | 0.99991 | 0.99972 | 0.99872 | 0.99861 | 1.00850 | 1.00964 | 1.03811 | 1.03867 | 0.99970 |  | ok |
| multivariate | 4998.9 | 0.99988 | 0.99968 | 0.99535 | 0.98631 | 0.86202 | 0.96302 | 1.04178 | 1.10181 | 0.99967 |  | ok |
| full16_2048_h64 |  | 0.99985 | 0.99976 | 0.99965 | 0.99965 | 0.98788 | 0.98802 | 1.02217 | 1.02223 | 0.99936 | 0.99873 | ok |
| full16_2048_h128 |  | 0.99988 | 0.99975 | 0.99963 | 0.99962 | 0.99879 | 0.99916 | 1.03228 | 1.03282 | 0.99955 | 0.99872 | ok |
| mv4x8_2048_h64 |  | 0.99982 | 0.99976 | 0.99956 | 0.99952 | 0.98141 | 0.98484 | 1.02676 | 1.02703 | 0.99975 | 0.99823 | ok |
| gift64_h48 |  | 0.99986 | 0.99961 | 0.99873 | 0.99848 | 0.97742 | 0.97990 | 1.05154 | 1.06986 | 0.99960 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.397 / 0.9771 / 0.0451; 483: 0.480 / 0.9806 / 0.0402; 806: 0.592 / 0.9851 / 0.0204; 226: 0.672 / 0.9953 / 0.0243

## A1 hifi2, fp32 acc (matmul + sdpa)

policy: `{"weights": "bf16", "bf16_roles": [], "matmul": "hifi2, fp32_acc=True", "sdpa": "hifi2, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1372.4 | 0.99993 | 0.99988 | 0.99849 | 0.99261 | 0.68929 | 0.93006 | 1.03286 | 1.11478 | 0.99993 |  | spread_ratio_min |
| long_context | 233.2 | 0.99996 | 0.99992 | 0.99950 | 0.99950 | 0.98969 | 0.98971 | 1.01605 | 1.01646 | 0.99995 |  | ok |
| multivariate | 5113.1 | 0.99994 | 0.99992 | 0.99922 | 0.99707 | 0.83966 | 0.97792 | 1.03043 | 1.05123 | 0.99992 |  | ok |
| full16_2048_h64 |  | 0.99994 | 0.99995 | 0.99991 | 0.99991 | 0.98647 | 0.98782 | 1.01948 | 1.01986 | 0.99989 | 0.99954 | ok |
| full16_2048_h128 |  | 0.99994 | 0.99992 | 0.99989 | 0.99989 | 0.99621 | 0.99675 | 1.01956 | 1.01995 | 0.99991 | 0.99954 | ok |
| mv4x8_2048_h64 |  | 0.99992 | 0.99994 | 0.99983 | 0.99982 | 0.99199 | 0.99340 | 1.02373 | 1.02398 | 0.99993 | 0.99941 | ok |
| gift64_h48 |  | 0.99994 | 0.99993 | 0.99978 | 0.99971 | 0.96966 | 0.98392 | 1.03019 | 1.03402 | 0.99995 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.879 / 0.9986 / 0.0095; 483: 0.689 / 0.9939 / 0.0240; 806: 0.782 / 0.9713 / 0.0145; 226: 0.900 / 0.9989 / 0.0078

## A2 hifi4, fp32 acc (matmul + sdpa)

policy: `{"weights": "bf16", "bf16_roles": [], "matmul": "hifi4, fp32_acc=True", "sdpa": "hifi4, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1474.4 | 0.99993 | 0.99988 | 0.99845 | 0.98928 | 0.53673 | 0.88595 | 1.03449 | 1.05434 | 0.99993 |  | spread_ratio_min |
| long_context | 244.0 | 0.99997 | 0.99992 | 0.99949 | 0.99943 | 0.98649 | 0.98806 | 1.02128 | 1.02133 | 0.99996 |  | ok |
| multivariate | 5252.5 | 0.99994 | 0.99993 | 0.99914 | 0.99759 | 0.83600 | 0.98238 | 1.03296 | 1.04870 | 0.99994 |  | ok |
| full16_2048_h64 |  | 0.99996 | 0.99996 | 0.99993 | 0.99993 | 1.00376 | 1.00394 | 1.02309 | 1.02324 | 0.99990 | 0.99960 | ok |
| full16_2048_h128 |  | 0.99995 | 0.99992 | 0.99993 | 0.99993 | 1.00047 | 1.00059 | 1.02148 | 1.02178 | 0.99993 | 0.99960 | ok |
| mv4x8_2048_h64 |  | 0.99993 | 0.99995 | 0.99988 | 0.99986 | 1.00189 | 1.00285 | 1.02445 | 1.02457 | 0.99994 | 0.99950 | ok |
| gift64_h48 |  | 0.99996 | 0.99993 | 0.99982 | 0.99976 | 0.97401 | 0.98370 | 1.03416 | 1.03645 | 0.99995 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.859 / 0.9983 / 0.0109; 483: 0.537 / 0.9893 / 0.0354; 806: 0.629 / 0.9885 / 0.0186; 226: 0.896 / 0.9991 / 0.0079

## A3 hifi4, no fp32 acc

policy: `{"weights": "bf16", "bf16_roles": [], "matmul": "hifi4, fp32_acc=False", "sdpa": "hifi4, fp32_acc=False", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1463.1 | 0.99987 | 0.99961 | 0.99368 | 0.95490 | 0.40989 | 0.89659 | 1.05897 | 1.13806 | 0.99969 |  | spread_ratio_min |
| long_context | 206.1 | 0.99991 | 0.99971 | 0.99884 | 0.99875 | 1.01606 | 1.01802 | 1.04480 | 1.04494 | 0.99970 |  | ok |
| multivariate | 5138.8 | 0.99988 | 0.99968 | 0.99589 | 0.98581 | 0.86081 | 0.98395 | 1.04966 | 1.11487 | 0.99969 |  | ok |
| full16_2048_h64 |  | 0.99984 | 0.99970 | 0.99925 | 0.99916 | 0.99458 | 0.99592 | 1.03792 | 1.03837 | 0.99944 | 0.99876 | ok |
| full16_2048_h128 |  | 0.99983 | 0.99964 | 0.99940 | 0.99936 | 1.00225 | 1.00410 | 1.04924 | 1.05173 | 0.99962 | 0.99875 | ok |
| mv4x8_2048_h64 |  | 0.99983 | 0.99977 | 0.99968 | 0.99967 | 0.99390 | 0.99468 | 1.02967 | 1.02999 | 0.99978 | 0.99822 | ok |
| gift64_h48 |  | 0.99988 | 0.99963 | 0.99894 | 0.99848 | 0.98254 | 0.99400 | 1.10215 | 1.15491 | 0.99948 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.410 / 0.9787 / 0.0442; 483: 0.415 / 0.9805 / 0.0450; 806: 0.472 / 0.9805 / 0.0258; 226: 0.636 / 0.9935 / 0.0269

## A4 hifi2, fp32 acc matmul only

policy: `{"weights": "bf16", "bf16_roles": [], "matmul": "hifi2, fp32_acc=True", "sdpa": "hifi2, fp32_acc=False", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1368.9 | 0.99992 | 0.99986 | 0.99818 | 0.98944 | 0.55678 | 0.92225 | 1.03564 | 1.06437 | 0.99992 |  | spread_ratio_min |
| long_context | 214.0 | 0.99996 | 0.99991 | 0.99948 | 0.99947 | 0.98054 | 0.98144 | 1.01564 | 1.01594 | 0.99995 |  | ok |
| multivariate | 5106.2 | 0.99993 | 0.99991 | 0.99907 | 0.99797 | 0.83706 | 0.96913 | 1.02855 | 1.05669 | 0.99990 |  | ok |
| full16_2048_h64 |  | 0.99992 | 0.99994 | 0.99988 | 0.99988 | 0.97923 | 0.98109 | 1.02477 | 1.02576 | 0.99987 | 0.99944 | ok |
| full16_2048_h128 |  | 0.99992 | 0.99993 | 0.99988 | 0.99988 | 0.98663 | 0.98809 | 1.02064 | 1.02162 | 0.99993 | 0.99944 | ok |
| mv4x8_2048_h64 |  | 0.99990 | 0.99994 | 0.99979 | 0.99977 | 0.99133 | 0.99176 | 1.01948 | 1.01958 | 0.99992 | 0.99923 | ok |
| gift64_h48 |  | 0.99992 | 0.99990 | 0.99966 | 0.99966 | 0.96431 | 0.97959 | 1.03744 | 1.04396 | 0.99991 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.736 / 0.9977 / 0.0198; 483: 0.557 / 0.9894 / 0.0340; 806: 0.592 / 0.9783 / 0.0209; 226: 0.940 / 0.9978 / 0.0063

## A5 hifi2, fp32 acc sdpa only

policy: `{"weights": "bf16", "bf16_roles": [], "matmul": "hifi2, fp32_acc=False", "sdpa": "hifi2, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1373.7 | 0.99988 | 0.99966 | 0.99373 | 0.95759 | 0.63838 | 0.89850 | 1.04973 | 1.08984 | 0.99967 |  | spread_ratio_min |
| long_context | 198.0 | 0.99991 | 0.99972 | 0.99881 | 0.99871 | 1.01178 | 1.01328 | 1.03907 | 1.03940 | 0.99971 |  | ok |
| multivariate | 5002.3 | 0.99988 | 0.99968 | 0.99566 | 0.98307 | 0.84839 | 0.97181 | 1.05064 | 1.11089 | 0.99968 |  | ok |
| full16_2048_h64 |  | 0.99991 | 0.99978 | 0.99966 | 0.99964 | 0.99023 | 0.99097 | 1.02958 | 1.03076 | 0.99936 | 0.99884 | ok |
| full16_2048_h128 |  | 0.99989 | 0.99977 | 0.99966 | 0.99965 | 1.00242 | 1.00296 | 1.02641 | 1.02676 | 0.99956 | 0.99883 | ok |
| mv4x8_2048_h64 |  | 0.99987 | 0.99977 | 0.99969 | 0.99969 | 0.99868 | 0.99975 | 1.03180 | 1.03449 | 0.99976 | 0.99842 | ok |
| gift64_h48 |  | 0.99989 | 0.99964 | 0.99894 | 0.99876 | 0.97610 | 0.97894 | 1.06283 | 1.08128 | 0.99966 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.731 / 0.9934 / 0.0208; 483: 0.790 / 0.9924 / 0.0175; 806: 0.742 / 0.9641 / 0.0164; 226: 0.680 / 0.9928 / 0.0240

## P0 bf8 + lofi (round 0)

policy: `{"weights": "bfloat8_b", "bf16_roles": [], "matmul": "lofi, fp32_acc=False", "sdpa": "lofi, fp32_acc=False", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1371.0 | 0.99788 | 0.99381 | 0.53116 | -0.03421 | 0.51715 | 0.63496 | 0.97447 | 1.19470 | 0.99333 |  | pcc_row_p01, pcc_row_min, pcc_row_p01_active, pcc_row_min_active, spread_ratio_min |
| multivariate | 5005.0 | 0.99805 | 0.99386 | 0.79057 | 0.30843 | 0.56662 | 0.65476 | 0.96492 | 1.12840 | 0.99384 |  | pcc_row_min, pcc_row_p01_active, pcc_row_min_active, spread_ratio_min |

paper named rows (spread ratio / pcc / err std): 458: 0.756 / 0.2968 / 0.0860; 483: 0.650 / -0.0342 / 0.1025; 806: 1.013 / 0.0440 / 0.0801; 226: 0.759 / 0.3151 / 0.0835

## P1 bf16 + hifi2

policy: `{"weights": "bf16", "bf16_roles": [], "matmul": "hifi2, fp32_acc=False", "sdpa": "hifi2, fp32_acc=False", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1372.2 | 0.99987 | 0.99963 | 0.99307 | 0.94917 | 0.39713 | 0.89133 | 1.04659 | 1.10951 | 0.99965 |  | spread_ratio_min |
| multivariate | 4999.0 | 0.99988 | 0.99968 | 0.99535 | 0.98631 | 0.86202 | 0.96302 | 1.04178 | 1.10181 | 0.99967 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.397 / 0.9771 / 0.0451; 483: 0.480 / 0.9806 / 0.0402; 806: 0.592 / 0.9851 / 0.0204; 226: 0.672 / 0.9953 / 0.0243

## P2 bf8 except ffn + lofi

policy: `{"weights": "bfloat8_b", "bf16_roles": ["ffn"], "matmul": "lofi, fp32_acc=False", "sdpa": "lofi, fp32_acc=False", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1371.1 | 0.99966 | 0.99883 | 0.97292 | 0.88883 | 0.39907 | 0.83565 | 1.07212 | 1.23280 | 0.99896 |  | pcc_row_p01, spread_ratio_min |
| multivariate | 5003.2 | 0.99971 | 0.99903 | 0.98629 | 0.96893 | 0.71344 | 0.93505 | 1.03646 | 1.24795 | 0.99900 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.533 / 0.9280 / 0.0391; 483: 0.399 / 0.8960 / 0.0491; 806: 0.951 / 0.9270 / 0.0181; 226: 0.839 / 0.9582 / 0.0221

## P3 bf16 + lofi

policy: `{"weights": "bf16", "bf16_roles": [], "matmul": "lofi, fp32_acc=False", "sdpa": "lofi, fp32_acc=False", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1365.5 | 0.99971 | 0.99911 | 0.98658 | 0.85749 | 0.27081 | 0.80733 | 1.02070 | 1.07875 | 0.99924 |  | pcc_row_p01, spread_ratio_min |
| multivariate | 4995.0 | 0.99975 | 0.99920 | 0.98761 | 0.96749 | 0.68792 | 0.92424 | 1.01215 | 1.12946 | 0.99919 |  | spread_ratio_min |

paper named rows (spread ratio / pcc / err std): 458: 0.401 / 0.9457 / 0.0461; 483: 0.271 / 0.9153 / 0.0565; 806: 0.960 / 0.9877 / 0.0076; 226: 0.806 / 0.9923 / 0.0162

## P4 bf8 + hifi2

policy: `{"weights": "bfloat8_b", "bf16_roles": [], "matmul": "hifi2, fp32_acc=False", "sdpa": "hifi2, fp32_acc=False", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1371.4 | 0.99807 | 0.99445 | 0.54427 | -0.02240 | 0.55355 | 0.63447 | 1.00704 | 1.16974 | 0.99399 |  | pcc_row_p01, pcc_row_min, pcc_row_p01_active, pcc_row_min_active, spread_ratio_min |
| multivariate | 4997.7 | 0.99828 | 0.99470 | 0.79381 | 0.37413 | 0.63267 | 0.67817 | 0.98242 | 1.11064 | 0.99482 |  | pcc_row_min, pcc_row_p01_active, pcc_row_min_active, spread_ratio_min |

paper named rows (spread ratio / pcc / err std): 458: 0.760 / 0.3280 / 0.0845; 483: 0.644 / -0.0224 / 0.1014; 806: 1.088 / 0.0949 / 0.0802; 226: 0.780 / 0.3516 / 0.0819

## P0a bf8 + lofi, fp32 acc

policy: `{"weights": "bfloat8_b", "bf16_roles": [], "matmul": "lofi, fp32_acc=True", "sdpa": "lofi, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1114.9 | 0.99795 | 0.99412 | 0.47806 | -0.06146 | 0.41520 | 0.63237 | 0.97052 | 1.16088 | 0.99361 |  | pcc_row_p01, pcc_row_min, pcc_row_p01_active, pcc_row_min_active, spread_ratio_min |
| multivariate | 4610.0 | 0.99816 | 0.99435 | 0.79448 | 0.31105 | 0.57729 | 0.65140 | 0.95254 | 1.20738 | 0.99465 |  | pcc_row_min, pcc_row_p01_active, pcc_row_min_active, spread_ratio_min |

paper named rows (spread ratio / pcc / err std): 458: 0.792 / 0.3903 / 0.0811; 483: 0.660 / -0.0615 / 0.1041; 806: 1.015 / 0.0399 / 0.0801; 226: 0.813 / 0.4046 / 0.0795

## P2a bf8 except ffn + lofi, fp32 acc

policy: `{"weights": "bfloat8_b", "bf16_roles": ["ffn"], "matmul": "lofi, fp32_acc=True", "sdpa": "lofi, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1114.4 | 0.99971 | 0.99916 | 0.97681 | 0.82318 | 0.29691 | 0.89008 | 1.06456 | 1.11497 | 0.99922 |  | pcc_row_p01, spread_ratio_min |
| multivariate | 4606.9 | 0.99975 | 0.99923 | 0.98918 | 0.97066 | 0.84369 | 0.92991 | 1.03279 | 1.14337 | 0.99920 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.977 / 0.9800 / 0.0146; 483: 0.297 / 0.8232 / 0.0571; 806: 0.767 / 0.9116 / 0.0205; 226: 0.543 / 0.9257 / 0.0381

## P3a bf16 + lofi, fp32 acc

policy: `{"weights": "bf16", "bf16_roles": [], "matmul": "lofi, fp32_acc=True", "sdpa": "lofi, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1381.0 | 0.99977 | 0.99934 | 0.98653 | 0.89816 | 0.40537 | 0.84090 | 1.02879 | 1.10040 | 0.99946 |  | pcc_row_p01, spread_ratio_min |
| multivariate | 5139.7 | 0.99979 | 0.99937 | 0.99175 | 0.97813 | 0.78390 | 0.92511 | 1.00840 | 1.06878 | 0.99932 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.668 / 0.9856 / 0.0261; 483: 0.405 / 0.9640 / 0.0460; 806: 1.165 / 0.9930 / 0.0101; 226: 0.895 / 0.9922 / 0.0114

## P4a bf8 + hifi2, fp32 acc

policy: `{"weights": "bfloat8_b", "bf16_roles": [], "matmul": "hifi2, fp32_acc=True", "sdpa": "hifi2, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1129.8 | 0.99814 | 0.99478 | 0.53247 | -0.08319 | 0.59441 | 0.63291 | 0.99258 | 1.16981 | 0.99431 |  | pcc_row_p01, pcc_row_min, pcc_row_p01_active, pcc_row_min_active, spread_ratio_min |
| multivariate | 4628.8 | 0.99834 | 0.99499 | 0.80394 | 0.31444 | 0.62814 | 0.66560 | 0.97566 | 1.09174 | 0.99509 |  | pcc_row_min, pcc_row_p01_active, pcc_row_min_active, spread_ratio_min |

paper named rows (spread ratio / pcc / err std): 458: 0.891 / 0.5133 / 0.0757; 483: 0.646 / -0.0832 / 0.1044; 806: 1.078 / 0.1270 / 0.0786; 226: 0.876 / 0.4793 / 0.0763
