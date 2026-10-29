### b1_s65536: 637 ops, device kernel sum 1420.502 ms, eager op-to-op gaps 0.528 ms, roofline floor 872.248 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 4 | 787913.2 | 55.5% | 120 | H4 S65536 keys65536 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 424194.3 | 24117.2 | 424194.3 | 1.9 | compute |
| SDPA |  | 20 | 88285.4 | 6.2% | 120 | H4 S65536 keys1025 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 16586.3 | 110100.5 | 110100.5 | 0.8 | DRAM |
| MinimalMatmul | GELU_TANH | 24 | 64578.0 | 4.5% | 120 | 65536x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 9942.1 | 15780.9 | 15780.9 | 4.1 | DRAM (far above floor) |
| MinimalMatmul |  | 44 | 59896.4 | 4.2% | 120 | 65536x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 18227.1 | 28931.6 | 28931.6 | 2.1 | DRAM |
| MinimalMatmul | GELU_TANH | 24 | 52950.1 | 3.7% | 120 | 65536x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2485.5 | 6304.5 | 6304.5 | 8.4 | DRAM (far above floor) |
| BinaryNg | MUL | 48 | 51028.5 | 3.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 75497.5 | 75497.5 | 0.7 | DRAM |
| LayerNorm | RMSNORM | 149 | 49284.0 | 3.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 39059.8 | 39059.8 | 1.3 | DRAM |
| BinaryNg | ADD | 72 | 30528.5 | 2.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 28311.6 | 28311.6 | 1.1 | DRAM |
| RotaryEmbedding |  | 20 | 29053.6 | 2.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 13107.2 | 13107.2 | 2.2 | DRAM |
| MinimalMatmul |  | 28 | 28719.5 | 2.0% | 120 | 65536x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 11599.1 | 18411.0 | 18411.0 | 1.6 | DRAM |
| BinaryNg | ADD | 1 | 24132.3 | 1.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 4735.0 | 4735.0 | 5.1 | DRAM (far above floor) |
| NlpCreateHeads |  | 20 | 22803.8 | 1.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 20971.5 | 20971.5 | 1.1 | DRAM |
| BinaryNg | ADD | 1 | 20927.4 | 1.5% | 120 |  | BFLOAT4_B/BFLOAT4_B/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 14155.8 | 14155.8 | 1.5 | DRAM |
| MinimalMatmul |  | 48 | 20360.3 | 1.4% | 120 | 65536x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 4971.0 | 12609.0 | 12609.0 | 1.6 | DRAM |
| RotaryEmbedding |  | 20 | 14701.2 | 1.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 7864.3 | 7864.3 | 1.9 | DRAM |
| LayerNorm | RMSNORM | 20 | 13416.4 | 0.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 10485.8 | 10485.8 | 1.3 | DRAM |
| RotaryEmbedding |  | 4 | 11689.5 | 0.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 5242.9 | 5242.9 | 2.2 | DRAM |
| MinimalMatmul |  | 20 | 10471.1 | 0.7% | 120 | 65536x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 4142.5 | 7886.1 | 7886.1 | 1.3 | DRAM |
| MinimalMatmul |  | 4 | 7934.2 | 0.6% | 120 | 65536x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2485.5 | 3683.1 | 3683.1 | 2.2 | DRAM |
| NlpCreateHeads |  | 4 | 6846.8 | 0.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 6291.5 | 6291.5 | 1.1 | DRAM |
| LayerNorm | RMSNORM | 20 | 6814.6 | 0.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 5242.9 | 5242.9 | 1.3 | DRAM |
| LayerNorm | RMSNORM | 20 | 6702.1 | 0.5% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 5242.9 | 5242.9 | 1.3 | DRAM |
| LayerNorm | RMSNORM | 4 | 5305.9 | 0.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 4194.3 | 4194.3 | 1.3 | DRAM |
| RotaryEmbedding |  | 4 | 3021.7 | 0.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 2097.2 | 2097.2 | 1.4 | DRAM |
| LayerNorm | RMSNORM | 4 | 1279.0 | 0.1% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 1048.6 | 1048.6 | 1.2 | DRAM |
| Matmul |  | 1 | 904.4 | 0.1% | 24 | 32x65536x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 19.4 | 205.0 | 205.0 | 4.4 | DRAM (far above floor) |
| MinimalMatmul |  | 1 | 584.7 | 0.0% | 120 | 65536x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 155.3 | 328.5 | 328.5 | 1.8 | DRAM |
| Embeddings |  | 1 | 340.9 | 0.0% | 120 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 262.7 | 262.7 | 1.3 | DRAM |
| Reduce |  | 1 | 6.9 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 34.3 | DRAM (far above floor) |
| BinaryNg | DIV | 1 | 6.9 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 17.5 | DRAM (far above floor) |
| FillPad |  | 1 | 5.1 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 13.2 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.9 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 5.0 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.7 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 167.5 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 1 | 2.3 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 144.4 | DRAM (far above floor) |
| Clone |  | 1 | 1.7 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.4 | DRAM (far above floor) |
