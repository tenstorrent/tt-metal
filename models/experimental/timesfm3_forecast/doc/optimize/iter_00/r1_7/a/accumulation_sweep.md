# Precision sweep (fp32 accumulation, fidelity, performance policy)

Device ms: median of 3 forward_device calls after a 0.3 s idle. Metrics: `golden_io.head_metrics` / `final_metrics`; `spread_ratio` per active row of the golden.

## P2a bf8 except ffn + lofi, fp32 acc (interim)

policy: `{"weights": "bfloat8_b", "bf16_roles": ["ffn"], "matmul": "hifi2, fp32_acc=True", "bf16_role_matmul": "as matmul", "sdpa": "hifi2, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 579.8 | 0.99986 | 0.99966 | 0.98800 | 0.84308 | 0.32006 | 0.93063 | 1.05550 | 1.13849 | 0.99969 |  | pcc_row_p01, pcc_row_min, pcc_row_min_active, spread_ratio_min |
| multivariate | 2770.9 | 0.99988 | 0.99975 | 0.99454 | 0.97576 | 0.82011 | 0.98288 | 1.04910 | 1.09120 | 0.99972 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 1.015 / 0.9734 / 0.0172; 483: 0.320 / 0.8431 / 0.0557; 806: 0.749 / 0.8946 / 0.0222; 226: 0.849 / 0.9679 / 0.0198

## P5a bf8 except ffn + hifi2, fp32 acc

policy: `{"weights": "bfloat8_b", "bf16_roles": ["ffn"], "matmul": "hifi2, fp32_acc=True", "bf16_role_matmul": "as matmul", "sdpa": "hifi2, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 579.6 | 0.99986 | 0.99966 | 0.98800 | 0.84308 | 0.32006 | 0.93063 | 1.05550 | 1.13849 | 0.99969 |  | pcc_row_p01, pcc_row_min, pcc_row_min_active, spread_ratio_min |
| multivariate | 2772.1 | 0.99988 | 0.99975 | 0.99454 | 0.97576 | 0.82011 | 0.98288 | 1.04910 | 1.09120 | 0.99972 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 1.015 / 0.9734 / 0.0172; 483: 0.320 / 0.8431 / 0.0557; 806: 0.749 / 0.8946 / 0.0222; 226: 0.849 / 0.9679 / 0.0198

## P5b bf8 except ffn, lofi on bf8, hifi2 on ffn + sdpa

policy: `{"weights": "bfloat8_b", "bf16_roles": ["ffn"], "matmul": "lofi, fp32_acc=True", "bf16_role_matmul": "hifi2", "sdpa": "hifi2, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 575.0 | 0.99976 | 0.99927 | 0.98284 | 0.93243 | 0.55181 | 0.88640 | 1.06331 | 1.19813 | 0.99930 |  | pcc_row_p01, pcc_row_min, spread_ratio_p01 |
| multivariate | 2761.9 | 0.99979 | 0.99934 | 0.99074 | 0.97319 | 0.85340 | 0.93892 | 1.02280 | 1.13464 | 0.99933 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.907 / 0.9746 / 0.0171; 483: 0.552 / 0.9503 / 0.0374; 806: 0.848 / 0.9390 / 0.0168; 226: 0.781 / 0.9696 / 0.0219

## P5c bf8 except ffn, lofi on bf8 + sdpa, hifi2 on ffn

policy: `{"weights": "bfloat8_b", "bf16_roles": ["ffn"], "matmul": "lofi, fp32_acc=True", "bf16_role_matmul": "hifi2", "sdpa": "lofi, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 574.1 | 0.99972 | 0.99916 | 0.98096 | 0.86559 | 0.36118 | 0.87673 | 1.06108 | 1.15503 | 0.99930 |  | pcc_row_p01, pcc_row_min, spread_ratio_p01, spread_ratio_min |
| multivariate | 2761.8 | 0.99976 | 0.99926 | 0.98949 | 0.96012 | 0.83493 | 0.93676 | 1.04313 | 1.18766 | 0.99922 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.704 / 0.9569 / 0.0278; 483: 0.361 / 0.8706 / 0.0525; 806: 0.909 / 0.9498 / 0.0150; 226: 0.555 / 0.8656 / 0.0419

## P5d bf8 except ffn + head, lofi on bf8, hifi2 on ffn + head + sdpa

policy: `{"weights": "bfloat8_b", "bf16_roles": ["ffn", "head"], "matmul": "lofi, fp32_acc=True", "bf16_role_matmul": "hifi2", "sdpa": "hifi2, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 574.8 | 0.99982 | 0.99957 | 0.98872 | 0.95951 | 0.54742 | 0.90312 | 1.07737 | 1.21911 | 0.99967 |  | pcc_row_p01, pcc_row_min |
| multivariate | 2761.3 | 0.99985 | 0.99970 | 0.99471 | 0.97511 | 0.87163 | 0.95544 | 1.03825 | 1.15507 | 0.99970 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.913 / 0.9766 / 0.0163; 483: 0.547 / 0.9595 / 0.0369; 806: 0.849 / 0.9446 / 0.0161; 226: 0.787 / 0.9735 / 0.0208

