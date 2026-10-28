# Profile summary: paper/accuracy, 512 rows

- reduced forward kernel time 234.87 ms over 170 ops; op-to-op gaps 151.14 ms (eager, not in the traced bench)
- full-depth estimate 690.48 ms; measured 691.29 ms (-0.1 %)

| section | profiled | mean ms | x full | full ms |
|---|---|---|---|---|
| embed | 1 | 4.306 | 1 | 4.31 |
| enc_block | 2 | 47.570 | 6 | 285.42 |
| enc_norm | 1 | 0.259 | 1 | 0.26 |
| dec_block | 2 | 66.333 | 6 | 398.00 |
| dec_norm | 1 | 0.091 | 1 | 0.09 |
| head | 1 | 2.407 | 1 | 2.41 |

| op | ms (reduced) | share |
|---|---|---|
| MatmulDeviceOperation | 176.092 | 75.0 % |
| ReshapeViewDeviceOperation | 23.548 | 10.0 % |
| BinaryNgDeviceOperation | 14.330 | 6.1 % |
| TransposeDeviceOperation | 6.736 | 2.9 % |
| UnaryDeviceOperation | 3.805 | 1.6 % |
| SoftmaxDeviceOperation | 2.902 | 1.2 % |
| LayerNormDeviceOperation | 1.922 | 0.8 % |
| NlpCreateHeadsDeviceOperation | 1.916 | 0.8 % |
| TypecastDeviceOperation | 1.456 | 0.6 % |
| SliceDeviceOperation | 1.077 | 0.5 % |
| NLPConcatHeadsDeviceOperation | 0.820 | 0.3 % |
| ConcatDeviceOperation | 0.266 | 0.1 % |

Top 15 ops:

| section | op | ms | cores | fidelity | in0 shape |
|---|---|---|---|---|---|
| enc_block | MatmulDeviceOperation | 9.584 | 6 | HiFi2 | 96[96] x 96[96] |
| enc_block | MatmulDeviceOperation | 9.581 | 6 | HiFi2 | 96[96] x 96[96] |
| dec_block | MatmulDeviceOperation | 8.526 | 2 | HiFi2 | 32[32] x 96[96] |
| dec_block | MatmulDeviceOperation | 8.526 | 2 | HiFi2 | 32[32] x 96[96] |
| enc_block | MatmulDeviceOperation | 7.609 | 9 | HiFi4 | 96[96] x 64[64] |
| enc_block | MatmulDeviceOperation | 7.608 | 9 | HiFi4 | 96[96] x 64[64] |
| dec_block | MatmulDeviceOperation | 7.578 | 6 | HiFi4 | 96[96] x 64[64] |
| dec_block | MatmulDeviceOperation | 7.578 | 6 | HiFi4 | 96[96] x 64[64] |
| enc_block | MatmulDeviceOperation | 7.577 | 6 | HiFi4 | 96[96] x 64[64] |
| enc_block | MatmulDeviceOperation | 7.577 | 6 | HiFi4 | 96[96] x 64[64] |
| enc_block | MatmulDeviceOperation | 7.574 | 6 | HiFi4 | 96[96] x 64[64] |
| enc_block | MatmulDeviceOperation | 7.573 | 6 | HiFi4 | 96[96] x 64[64] |
| dec_block | ReshapeViewDeviceOperation | 5.052 | 110 |  | 96[96] x 512[512] |
| dec_block | ReshapeViewDeviceOperation | 5.041 | 110 |  | 96[96] x 512[512] |
| dec_block | ReshapeViewDeviceOperation | 5.039 | 110 |  | 96[96] x 512[512] |
