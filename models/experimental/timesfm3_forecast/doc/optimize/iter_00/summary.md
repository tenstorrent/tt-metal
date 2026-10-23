# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 11.08 | 5 | 1 | 11.1 |
| layer | 67.86 | 22 | 20 | 1357.2 |
| head | 2.15 | 5 | 1 | 2.1 |

Estimated full-model device time: **1370.4 ms** over 450 ops.
Measured: device 1370.5 ms (estimate off by -0.0%), steady wall 1988.8 ms, pcc_normalized 0.99987.

Category share of the reduced window: Compute 89.8%, TM 8.8%, Other 1.4%; matmul weighted FLOPs utilization 22.4%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 53.2 | 43.13 | 9 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 12.7 | 10.30 | 8 |
| RotaryEmbeddingLlamaDeviceOperation (in0:dram_interleaved) | 9.2 | 7.49 | 2 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 8.2 | 6.62 | 4 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 5.9 | 4.76 | 1 |
| SDPAOperation (in0:dram_interleaved) | 3.3 | 2.66 | 1 |
| UnaryDeviceOperation (in0:dram_interleaved) | 3.2 | 2.62 | 2 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 1.7 | 1.40 | 1 |
| UntilizeCodegenDeviceOperation (in0:dram_interleaved) | 1.4 | 1.14 | 1 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 0.6 | 0.48 | 1 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 0.5 | 0.43 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.1 | 0.05 | 1 |
