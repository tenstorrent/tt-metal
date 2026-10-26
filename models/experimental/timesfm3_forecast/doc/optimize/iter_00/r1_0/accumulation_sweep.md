# Precision sweep (fp32 accumulation, fidelity, performance policy)

Device ms: median of 3 forward_device calls after a 0.3 s idle. Metrics: `golden_io.head_metrics` / `final_metrics`; `spread_ratio` per active row of the golden.

## P2a bf8 except ffn + lofi, fp32 acc (interim)

policy: `{"weights": "bfloat8_b", "bf16_roles": ["ffn"], "matmul": "lofi, fp32_acc=True", "bf16_role_matmul": "as matmul", "sdpa": "lofi, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1114.1 | 0.99971 | 0.99916 | 0.97681 | 0.82318 | 0.29691 | 0.89008 | 1.06456 | 1.11497 | 0.99922 |  | pcc_row_p01, pcc_row_min, pcc_row_min_active, spread_ratio_p01, spread_ratio_min |
| multivariate | 4603.4 | 0.99975 | 0.99923 | 0.98918 | 0.97066 | 0.84369 | 0.92991 | 1.03279 | 1.14337 | 0.99920 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.977 / 0.9800 / 0.0146; 483: 0.297 / 0.8232 / 0.0571; 806: 0.767 / 0.9116 / 0.0205; 226: 0.543 / 0.9257 / 0.0381

## P5a bf8 except ffn + hifi2, fp32 acc

policy: `{"weights": "bfloat8_b", "bf16_roles": ["ffn"], "matmul": "hifi2, fp32_acc=True", "bf16_role_matmul": "as matmul", "sdpa": "hifi2, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1129.6 | 0.99986 | 0.99968 | 0.98772 | 0.84195 | 0.29531 | 0.91055 | 1.05942 | 1.14329 | 0.99970 |  | pcc_row_p01, pcc_row_min, pcc_row_min_active, spread_ratio_min |
| multivariate | 4623.5 | 0.99988 | 0.99976 | 0.99500 | 0.98127 | 0.91011 | 0.98074 | 1.04673 | 1.06884 | 0.99974 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 1.073 / 0.9757 / 0.0177; 483: 0.295 / 0.8420 / 0.0569; 806: 0.902 / 0.9000 / 0.0208; 226: 0.717 / 0.9490 / 0.0278

## P5b bf8 except ffn, lofi on bf8, hifi2 on ffn + sdpa

policy: `{"weights": "bfloat8_b", "bf16_roles": ["ffn"], "matmul": "lofi, fp32_acc=True", "bf16_role_matmul": "hifi2", "sdpa": "hifi2, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1119.0 | 0.99976 | 0.99926 | 0.98403 | 0.91757 | 0.41342 | 0.89577 | 1.05663 | 1.15149 | 0.99930 |  | pcc_row_p01, pcc_row_min, spread_ratio_p01, spread_ratio_min |
| multivariate | 4606.4 | 0.99979 | 0.99934 | 0.99056 | 0.95693 | 0.80120 | 0.93992 | 1.03212 | 1.15542 | 0.99930 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 1.014 / 0.9778 / 0.0157; 483: 0.413 / 0.9176 / 0.0475; 806: 1.102 / 0.9682 / 0.0139; 226: 0.633 / 0.9569 / 0.0308

## P5c bf8 except ffn, lofi on bf8 + sdpa, hifi2 on ffn

policy: `{"weights": "bfloat8_b", "bf16_roles": ["ffn"], "matmul": "lofi, fp32_acc=True", "bf16_role_matmul": "hifi2", "sdpa": "lofi, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1118.7 | 0.99973 | 0.99917 | 0.97982 | 0.88463 | 0.35973 | 0.86793 | 1.06005 | 1.14086 | 0.99928 |  | pcc_row_p01, pcc_row_min, spread_ratio_p01, spread_ratio_min |
| multivariate | 4606.8 | 0.99976 | 0.99927 | 0.99014 | 0.95234 | 0.80035 | 0.93745 | 1.03701 | 1.22599 | 0.99925 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.759 / 0.9635 / 0.0244; 483: 0.360 / 0.8846 / 0.0520; 806: 0.815 / 0.9414 / 0.0170; 226: 0.608 / 0.9418 / 0.0337

## P5d bf8 except ffn + head, lofi on bf8, hifi2 on ffn + head + sdpa

policy: `{"weights": "bfloat8_b", "bf16_roles": ["ffn", "head"], "matmul": "lofi, fp32_acc=True", "bf16_role_matmul": "hifi2", "sdpa": "hifi2, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1118.9 | 0.99981 | 0.99956 | 0.98814 | 0.93289 | 0.40593 | 0.91270 | 1.06899 | 1.15866 | 0.99967 |  | pcc_row_p01, pcc_row_min, spread_ratio_min |
| multivariate | 4607.9 | 0.99985 | 0.99970 | 0.99389 | 0.95935 | 0.81831 | 0.95620 | 1.04222 | 1.18359 | 0.99968 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 1.023 / 0.9791 / 0.0154; 483: 0.406 / 0.9329 / 0.0472; 806: 1.109 / 0.9721 / 0.0133; 226: 0.636 / 0.9625 / 0.0300

## P6a bf16, lofi except ffn (hifi2), sdpa hifi2

policy: `{"weights": "bf16", "bf16_roles": ["ffn"], "matmul": "lofi, fp32_acc=True", "bf16_role_matmul": "hifi2", "sdpa": "hifi2, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1385.0 | 0.99983 | 0.99949 | 0.99151 | 0.92232 | 0.69748 | 0.88173 | 1.01993 | 1.14722 | 0.99951 |  | spread_ratio_p01 |
| multivariate | 5145.6 | 0.99985 | 0.99952 | 0.99342 | 0.97959 | 0.85300 | 0.93262 | 1.00285 | 1.05652 | 0.99951 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 1.021 / 0.9934 / 0.0088; 483: 0.697 / 0.9853 / 0.0246; 806: 1.029 / 0.9939 / 0.0056; 226: 0.926 / 0.9963 / 0.0080

## P6b bf8 qkv only, hifi2

policy: `{"weights": "bfloat8_b", "bf16_roles": ["tokenizer", "seq_wo", "var_wo", "w_ov", "ffn", "head"], "matmul": "hifi2, fp32_acc=True", "bf16_role_matmul": "as matmul", "sdpa": "hifi2, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1132.1 | 0.99993 | 0.99988 | 0.99779 | 0.99233 | 0.69450 | 0.90004 | 1.03135 | 1.04808 | 0.99993 |  | ok |
| multivariate | 4631.9 | 0.99993 | 0.99991 | 0.99886 | 0.99451 | 0.86387 | 0.97132 | 1.03193 | 1.07321 | 0.99991 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.766 / 0.9937 / 0.0186; 483: 0.694 / 0.9949 / 0.0235; 806: 0.757 / 0.9911 / 0.0128; 226: 0.788 / 0.9947 / 0.0164

## P6c bf8 qkv + wo, hifi2

policy: `{"weights": "bfloat8_b", "bf16_roles": ["tokenizer", "ffn", "head"], "matmul": "hifi2, fp32_acc=True", "bf16_role_matmul": "as matmul", "sdpa": "hifi2, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1130.6 | 0.99988 | 0.99973 | 0.98805 | 0.95647 | 0.79872 | 0.94547 | 1.04528 | 1.12814 | 0.99975 |  | pcc_row_p01, pcc_row_min |
| multivariate | 4628.6 | 0.99989 | 0.99981 | 0.99497 | 0.98315 | 0.89434 | 0.98461 | 1.04600 | 1.09927 | 0.99980 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.851 / 0.9680 / 0.0201; 483: 0.834 / 0.9753 / 0.0195; 806: 0.799 / 0.9328 / 0.0181; 226: 0.799 / 0.9565 / 0.0236

## P6d bf16, hifi2, sdpa lofi

policy: `{"weights": "bf16", "bf16_roles": ["ffn"], "matmul": "hifi2, fp32_acc=True", "bf16_role_matmul": "as matmul", "sdpa": "lofi, fp32_acc=True", "embed": "as matmul", "norms_rope": "hifi4, fp32_acc=True"}`

| case | device ms | pcc_normalized | pcc_spread | pcc_row_p01_active | pcc_row_min_active | spread_ratio_min | spread_ratio_p01 | spread_ratio_p99 | spread_ratio_max | pcc_final_spread | layer19 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 1372.6 | 0.99989 | 0.99976 | 0.99710 | 0.95958 | 0.42633 | 0.89529 | 1.03344 | 1.11965 | 0.99990 |  | spread_ratio_p01, spread_ratio_min |
| multivariate | 5109.8 | 0.99989 | 0.99986 | 0.99814 | 0.98452 | 0.78245 | 0.97026 | 1.04619 | 1.10522 | 0.99985 |  | ok |

paper named rows (spread ratio / pcc / err std): 458: 0.816 / 0.9926 / 0.0155; 483: 0.426 / 0.9819 / 0.0439; 806: 0.736 / 0.9862 / 0.0144; 226: 1.027 / 0.9971 / 0.0058

