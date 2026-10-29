### b1_s32768: 637 ops, device kernel sum 512.119 ms, eager op-to-op gaps 0.513 ms, roofline floor 301.907 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 4 | 196330.5 | 38.3% | 120 | H4 S32768 keys32768 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 106048.6 | 7340.0 | 106048.6 | 1.9 | compute |
| SDPA |  | 20 | 45694.4 | 8.9% | 120 | H4 S32768 keys1025 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 8293.1 | 31457.3 | 31457.3 | 1.5 | DRAM |
| MinimalMatmul | GELU_TANH | 24 | 32876.5 | 6.4% | 120 | 32768x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 4971.0 | 7916.5 | 7916.5 | 4.2 | DRAM (far above floor) |
| MinimalMatmul |  | 44 | 29951.5 | 5.8% | 120 | 32768x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 9113.5 | 14513.7 | 14513.7 | 2.1 | DRAM |
| BinaryNg | MUL | 48 | 26858.9 | 5.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 37748.7 | 37748.7 | 0.7 | DRAM |
| MinimalMatmul | GELU_TANH | 24 | 26518.4 | 5.2% | 120 | 32768x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1242.8 | 3158.8 | 3158.8 | 8.4 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 149 | 25671.1 | 5.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 19530.0 | 19530.0 | 1.3 | DRAM |
| BinaryNg | ADD | 72 | 16548.5 | 3.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 14155.8 | 14155.8 | 1.2 | DRAM |
| RotaryEmbedding |  | 20 | 15137.6 | 3.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 6553.6 | 6553.6 | 2.3 | DRAM |
| MinimalMatmul |  | 28 | 14612.2 | 2.9% | 120 | 32768x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 5799.5 | 9236.0 | 9236.0 | 1.6 | DRAM |
| NlpCreateHeads |  | 20 | 11898.9 | 2.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 10485.8 | 10485.8 | 1.1 | DRAM |
| MinimalMatmul |  | 48 | 10832.6 | 2.1% | 120 | 32768x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2485.5 | 6317.6 | 6317.6 | 1.7 | DRAM |
| RotaryEmbedding |  | 20 | 7811.1 | 1.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 3932.2 | 3932.2 | 2.0 | DRAM |
| LayerNorm | RMSNORM | 20 | 6923.6 | 1.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 5242.9 | 5242.9 | 1.3 | DRAM |
| BinaryNg | ADD | 1 | 6513.6 | 1.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1187.8 | 1187.8 | 5.5 | DRAM (far above floor) |
| RotaryEmbedding |  | 4 | 6053.4 | 1.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 2621.4 | 2621.4 | 2.3 | DRAM |
| MinimalMatmul |  | 20 | 5655.9 | 1.1% | 120 | 32768x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2071.3 | 3953.9 | 3953.9 | 1.4 | DRAM |
| BinaryNg | ADD | 1 | 5579.2 | 1.1% | 120 |  | BFLOAT4_B/BFLOAT4_B/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 3538.9 | 3538.9 | 1.6 | DRAM |
| MinimalMatmul |  | 4 | 4013.2 | 0.8% | 120 | 32768x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1242.8 | 1848.1 | 1848.1 | 2.2 | DRAM |
| LayerNorm | RMSNORM | 20 | 3561.9 | 0.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2621.5 | 2621.5 | 1.4 | DRAM |
| NlpCreateHeads |  | 4 | 3556.3 | 0.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 3145.7 | 3145.7 | 1.1 | DRAM |
| LayerNorm | RMSNORM | 20 | 3483.4 | 0.7% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 2621.4 | 2621.4 | 1.3 | DRAM |
| LayerNorm | RMSNORM | 4 | 2754.3 | 0.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2097.2 | 2097.2 | 1.3 | DRAM |
| RotaryEmbedding |  | 4 | 1657.2 | 0.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1048.6 | 1048.6 | 1.6 | DRAM |
| LayerNorm | RMSNORM | 4 | 659.8 | 0.1% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 524.3 | 524.3 | 1.3 | DRAM |
| Matmul |  | 1 | 450.0 | 0.1% | 24 | 32x32768x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 9.7 | 102.6 | 102.6 | 4.4 | DRAM (far above floor) |
| MinimalMatmul |  | 1 | 301.9 | 0.1% | 120 | 32768x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 77.7 | 164.7 | 164.7 | 1.8 | DRAM |
| Embeddings |  | 1 | 184.4 | 0.0% | 120 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 131.3 | 131.3 | 1.4 | DRAM |
| BinaryNg | DIV | 1 | 7.1 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 18.1 | DRAM (far above floor) |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.8 | DRAM (far above floor) |
| FillPad |  | 1 | 5.0 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 13.1 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.9 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 5.0 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.7 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 167.7 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 1 | 2.3 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 143.8 | DRAM (far above floor) |
| Clone |  | 1 | 1.7 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.4 | DRAM (far above floor) |
