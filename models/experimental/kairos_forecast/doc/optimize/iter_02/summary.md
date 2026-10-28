# Profile summary: paper/accuracy, 512 rows

- reduced forward kernel time 50.25 ms over 142 ops; op-to-op gaps 123.51 ms (eager, not in the traced bench)
- full-depth estimate 136.88 ms

| section | profiled | mean ms | x full | full ms |
|---|---|---|---|---|
| embed | 1 | 4.861 | 1 | 4.86 |
| enc_block | 2 | 12.494 | 6 | 74.97 |
| enc_norm | 1 | 0.250 | 1 | 0.25 |
| dec_block | 2 | 9.162 | 6 | 54.97 |
| dec_norm | 1 | 0.091 | 1 | 0.09 |
| head | 1 | 1.740 | 1 | 1.74 |

| op | ms (reduced) | share |
|---|---|---|
| MatmulDeviceOperation | 22.751 | 45.3 % |
| BinaryNgDeviceOperation | 14.385 | 28.6 % |
| NlpCreateHeadsDeviceOperation | 5.745 | 11.4 % |
| SoftmaxDeviceOperation | 2.906 | 5.8 % |
| LayerNormDeviceOperation | 1.917 | 3.8 % |
| TypecastDeviceOperation | 1.459 | 2.9 % |
| NLPConcatHeadsDeviceOperation | 0.827 | 1.6 % |
| ConcatDeviceOperation | 0.265 | 0.5 % |

Top 15 ops:

| section | op | ms | cores | fidelity | in0 shape |
|---|---|---|---|---|---|
| embed | MatmulDeviceOperation | 1.936 | 110 | HiFi4 | 32768[32768] x 800[800] |
| embed | MatmulDeviceOperation | 1.592 | 80 | HiFi2 | 32768[32768] x 2048[2048] |
| enc_block | MatmulDeviceOperation | 1.009 | 110 | HiFi2 | 49152[49152] x 512[512] |
| enc_block | MatmulDeviceOperation | 1.008 | 110 | HiFi2 | 49152[49152] x 512[512] |
| enc_block | SoftmaxDeviceOperation | 1.003 | 110 | HiFi4 | 96[96] x 96[96] |
| enc_block | SoftmaxDeviceOperation | 1.002 | 110 | HiFi4 | 96[96] x 96[96] |
| enc_block | BinaryNgDeviceOperation | 1.001 | 110 | HiFi4 | 96[96] x 96[96] |
| enc_block | BinaryNgDeviceOperation | 0.995 | 110 | HiFi4 | 96[96] x 96[96] |
| enc_block | MatmulDeviceOperation | 0.984 | 80 | HiFi2 | 49152[49152] x 2048[2048] |
| enc_block | MatmulDeviceOperation | 0.983 | 80 | HiFi2 | 49152[49152] x 2048[2048] |
| enc_block | MatmulDeviceOperation | 0.849 | 100 | HiFi2 | 49152[49152] x 512[512] |
| enc_block | MatmulDeviceOperation | 0.848 | 100 | HiFi2 | 49152[49152] x 512[512] |
| dec_block | MatmulDeviceOperation | 0.847 | 100 | HiFi2 | 49152[49152] x 512[512] |
| embed | MatmulDeviceOperation | 0.846 | 80 | HiFi4 | 32768[32768] x 800[800] |
| enc_block | MatmulDeviceOperation | 0.846 | 100 | HiFi2 | 49152[49152] x 512[512] |
