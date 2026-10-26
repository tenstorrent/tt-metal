# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 49.43 | 5 | 1 | 49.4 |
| layer_time | 170.60 | 18 | 46 | 7847.7 |
| last_time | 83.52 | 22 | 1 | 83.5 |
| layer_variate | 2.12 | 13 | 1 | 2.1 |
| head | 1.25 | 6 | 1 | 1.2 |

Estimated full-model device time: **7984.0 ms** over 874 ops.

Category share of the reduced window: Compute 94.0%, TM 5.4%, Other 0.6%; matmul weighted FLOPs utilization 31.3%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 70.0 | 214.96 | 21 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 8.2 | 25.27 | 17 |
| RotaryEmbeddingLlamaDeviceOperation (in0:dram_interleaved) | 5.6 | 17.30 | 4 |
| UnaryDeviceOperation (in0:dram_interleaved) | 5.3 | 16.15 | 5 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 3.7 | 11.48 | 2 |
| SDPAOperation (in0:dram_interleaved) | 2.8 | 8.68 | 2 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 1.9 | 5.98 | 7 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 1.3 | 3.91 | 2 |
| UntilizeCodegenDeviceOperation (in0:dram_interleaved) | 0.6 | 1.89 | 1 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 0.2 | 0.75 | 1 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 0.1 | 0.46 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.0 | 0.09 | 1 |

Measured (job 554): 8209 ms device; estimate -2.7 % (op-to-op gaps not in the per-section window; see proposals.md Reconciliation).
