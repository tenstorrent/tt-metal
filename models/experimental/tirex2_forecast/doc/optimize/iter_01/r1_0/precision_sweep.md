# Precision sweep, paper (1024 rows, realistic mix), traced device time back to back on one chip

| policy | device ms | pcc_normalized | pcc_spread | pcc_row_min_active | spread_ratio_p01 | spread_ratio_p99 | pcc_final_spread | median_err | paper64 row_min_active | paper64 median_err |
|---|---|---|---|---|---|---|---|---|---|---|
| performance | 3792.2 | 0.9995 | 0.9991 | 0.8459 | 0.8667 | 1.2333 | 0.9992 | 0.437 | 0.9862 | 0.175 |
| performance+hifi2=all | 3798.6 | 0.9999 | 0.9998 | 0.9369 | 0.9354 | 1.1829 | 0.9998 | 0.207 | 0.9880 | 0.103 |
| performance+hifi2=embed | 3791.5 | 0.9998 | 0.9994 | 0.8932 | 0.8901 | 1.2210 | 0.9993 | 0.500 | 0.9822 | 0.136 |
| performance+hifi2=mlstm | 3792.7 | 0.9994 | 0.9986 | 0.8334 | 0.8714 | 1.2498 | 0.9991 | 0.619 | 0.9866 | 0.173 |
| performance+hifi2=w_in | 3792.2 | 0.9996 | 0.9992 | 0.9312 | 0.8745 | 1.2097 | 0.9992 | 0.405 | 0.9856 | 0.171 |
| performance+hifi2=ffn | 3795.9 | 0.9996 | 0.9992 | 0.8377 | 0.8505 | 1.2788 | 0.9992 | 0.373 | 0.9870 | 0.159 |
| performance+hifi2=head | 3792.0 | 0.9996 | 0.9993 | 0.8491 | 0.8864 | 1.1990 | 0.9996 | 0.334 | 0.9935 | 0.157 |
| performance+hifi4=w_in+bf16=w_in | 3798.6 | 0.9996 | 0.9992 | 0.9573 | 0.8549 | 1.2235 | 0.9993 | 0.406 | 0.9877 | 0.171 |
| performance+hifi4=w_in+bf16=w_in+hifi2=all | 3799.7 | 0.9999 | 0.9998 | 0.9431 | 0.9175 | 1.1878 | 0.9998 | 0.165 | 0.9903 | 0.111 |
| performance+hifi4=w_in+bf16=w_in+hifi2=embed,mlstm,head | 3806.7 | 0.9998 | 0.9997 | 0.9649 | 0.9381 | 1.1659 | 0.9997 | 0.211 | 0.9940 | 0.131 |
| accuracy | 3875.8 | 1.0000 | 0.9999 | 0.9953 | 0.9673 | 1.0403 | 0.9999 | 0.118 | 0.9980 | 0.041 |
