# Profile summary: paper/accuracy, 512 rows

- reduced forward kernel time 235.09 ms over 170 ops; op-to-op gaps 208.93 ms (eager, not in the traced bench)
- full-depth estimate 691.21 ms; measured 691.08 ms (+0.0 %)

| section | profiled | mean ms | x full | full ms |
|---|---|---|---|---|
| embed | 1 | 4.279 | 1 | 4.28 |
| enc_block | 2 | 47.676 | 6 | 286.06 |
| enc_norm | 1 | 0.261 | 1 | 0.26 |
| dec_block | 2 | 66.353 | 6 | 398.12 |
| dec_norm | 1 | 0.090 | 1 | 0.09 |
| head | 1 | 2.402 | 1 | 2.40 |

| op | ms (reduced) | share |
|---|---|---|
| MatmulDeviceOperation | 176.292 | 75.0 % |
| ReshapeViewDeviceOperation | 23.541 | 10.0 % |
| BinaryNgDeviceOperation | 14.332 | 6.1 % |
| TransposeDeviceOperation | 6.745 | 2.9 % |
| UnaryDeviceOperation | 3.815 | 1.6 % |
| SoftmaxDeviceOperation | 2.891 | 1.2 % |
| LayerNormDeviceOperation | 1.936 | 0.8 % |
| NlpCreateHeadsDeviceOperation | 1.916 | 0.8 % |
| TypecastDeviceOperation | 1.458 | 0.6 % |
| SliceDeviceOperation | 1.081 | 0.5 % |
| NLPConcatHeadsDeviceOperation | 0.822 | 0.3 % |
| ConcatDeviceOperation | 0.263 | 0.1 % |

Top 15 ops:

| section | op | ms | cores | fidelity | in0 shape |
|---|---|---|---|---|---|
| enc_block | MatmulDeviceOperation | 9.585 | 6 | HiFi2 | 96[96] x 96[96] |
| enc_block | MatmulDeviceOperation | 9.585 | 6 | HiFi2 | 96[96] x 96[96] |
| dec_block | MatmulDeviceOperation | 8.526 | 2 | HiFi2 | 32[32] x 96[96] |
| dec_block | MatmulDeviceOperation | 8.526 | 2 | HiFi2 | 32[32] x 96[96] |
| enc_block | MatmulDeviceOperation | 7.614 | 9 | HiFi4 | 96[96] x 64[64] |
| enc_block | MatmulDeviceOperation | 7.608 | 9 | HiFi4 | 96[96] x 64[64] |
| enc_block | MatmulDeviceOperation | 7.578 | 6 | HiFi4 | 96[96] x 64[64] |
| enc_block | MatmulDeviceOperation | 7.578 | 6 | HiFi4 | 96[96] x 64[64] |
| dec_block | MatmulDeviceOperation | 7.578 | 6 | HiFi4 | 96[96] x 64[64] |
| enc_block | MatmulDeviceOperation | 7.578 | 6 | HiFi4 | 96[96] x 64[64] |
| enc_block | MatmulDeviceOperation | 7.578 | 6 | HiFi4 | 96[96] x 64[64] |
| dec_block | MatmulDeviceOperation | 7.577 | 6 | HiFi4 | 96[96] x 64[64] |
| dec_block | ReshapeViewDeviceOperation | 5.047 | 110 |  | 96[96] x 512[512] |
| dec_block | ReshapeViewDeviceOperation | 5.041 | 110 |  | 96[96] x 512[512] |
| dec_block | ReshapeViewDeviceOperation | 5.041 | 110 |  | 96[96] x 512[512] |
