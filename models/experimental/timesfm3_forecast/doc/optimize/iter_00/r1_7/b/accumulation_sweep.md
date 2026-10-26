# Precision sweep (fp32 accumulation, fidelity, performance policy)

Device ms: median of 3 forward_device calls after a 0.3 s idle. Metrics: `golden_io.head_metrics` / `final_metrics`; `spread_ratio` per active row of the golden.

## P6a bf16, lofi except ffn (hifi2), sdpa hifi2

policy: `{"weights": "bf16", "bf16_roles": ["ffn"], "matmul": "lofi, fp32_acc=True", "bf16_role_matmul": "hifi2", "sdpa": "hifi2, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 575.4 | 0.99983 | 0.99948 | 0.99168 | 0.92583 | 0.70174 | 0.89567 | 1.02213 | 1.14289 | 0.99952 |  | spread_ratio_p01 |
| multivariate | 2669.3 | 0.99985 | 0.99951 | 0.99357 | 0.97930 | 0.81692 | 0.93035 | 1.00152 | 1.04879 | 0.99950 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.932 / 0.9933 / 0.0095; 483: 0.812 / 0.9885 / 0.0171; 806: 0.809 / 0.9855 / 0.0116; 226: 0.918 / 0.9964 / 0.0082

## P6b bf8 qkv only, hifi2

policy: `{"weights": "bfloat8_b", "bf16_roles": ["tokenizer", "seq_wo", "var_wo", "w_ov", "ffn", "head"], "matmul": "hifi2, fp32_acc=True", "bf16_role_matmul": "as matmul", "sdpa": "hifi2, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 579.8 | 0.99993 | 0.99989 | 0.99807 | 0.99318 | 0.75264 | 0.91049 | 1.03076 | 1.05562 | 0.99992 |  | ok |
| multivariate | 2670.4 | 0.99993 | 0.99991 | 0.99871 | 0.99593 | 0.83233 | 0.97112 | 1.03153 | 1.05761 | 0.99991 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.766 / 0.9943 / 0.0185; 483: 0.774 / 0.9963 / 0.0176; 806: 0.729 / 0.9890 / 0.0140; 226: 0.970 / 0.9985 / 0.0044

## P6c bf8 qkv + wo, hifi2

policy: `{"weights": "bfloat8_b", "bf16_roles": ["tokenizer", "ffn", "head"], "matmul": "hifi2, fp32_acc=True", "bf16_role_matmul": "as matmul", "sdpa": "hifi2, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 571.8 | 0.99988 | 0.99973 | 0.98741 | 0.96076 | 0.76681 | 0.92850 | 1.04519 | 1.14236 | 0.99975 |  | pcc_row_p01, pcc_row_min |
| multivariate | 2662.4 | 0.99989 | 0.99981 | 0.99539 | 0.97981 | 0.91914 | 0.98565 | 1.04060 | 1.06812 | 0.99980 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 1.014 / 0.9624 / 0.0204; 483: 0.767 / 0.9789 / 0.0217; 806: 0.797 / 0.9166 / 0.0197; 226: 0.813 / 0.9686 / 0.0209

## P6d bf16, hifi2, sdpa lofi

policy: `{"weights": "bf16", "bf16_roles": ["tokenizer", "seq_wo", "var_wo", "w_ov", "ffn", "head"], "matmul": "hifi2, fp32_acc=True", "bf16_role_matmul": "as matmul", "sdpa": "lofi, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 598.4 | 0.99989 | 0.99977 | 0.99722 | 0.97909 | 0.53697 | 0.89789 | 1.03505 | 1.08887 | 0.99990 |  | spread_ratio_p01 |
| multivariate | 2704.4 | 0.99990 | 0.99984 | 0.99796 | 0.98905 | 0.88704 | 0.96152 | 1.04048 | 1.07981 | 0.99980 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.889 / 0.9960 / 0.0101; 483: 0.537 / 0.9905 / 0.0353; 806: 0.537 / 0.9891 / 0.0225; 226: 1.089 / 0.9978 / 0.0080

