### b1_s32768: 646 ops, device kernel sum 561.992 ms, eager op-to-op gaps 0.532 ms, roofline floor 302.440 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 4 | 196402.0 | 34.9% | 120 | H4 S32768 keys32768 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 106048.6 | 7340.0 | 106048.6 | 1.9 | compute |
| BinaryNg | ADD | 1 | 51877.8 | 9.2% | 120 |  | BFLOAT4_B/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2363.4 | 2363.4 | 22.0 | DRAM (far above floor) |
| SDPA |  | 20 | 45742.9 | 8.1% | 120 | H4 S32768 keys1025 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 8293.1 | 31457.3 | 31457.3 | 1.5 | DRAM |
| MinimalMatmul | GELU_TANH | 24 | 32909.7 | 5.9% | 120 | 32768x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 4971.0 | 7916.5 | 7916.5 | 4.2 | DRAM (far above floor) |
| MinimalMatmul |  | 44 | 30157.1 | 5.4% | 120 | 32768x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 9113.5 | 14513.7 | 14513.7 | 2.1 | DRAM |
| BinaryNg | MUL | 48 | 26912.4 | 4.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 37748.7 | 37748.7 | 0.7 | DRAM |
| MinimalMatmul | GELU_TANH | 24 | 26509.5 | 4.7% | 120 | 32768x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1242.8 | 3158.8 | 3158.8 | 8.4 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 149 | 25732.0 | 4.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 19530.0 | 19530.0 | 1.3 | DRAM |
| BinaryNg | ADD | 72 | 16573.1 | 2.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 14155.8 | 14155.8 | 1.2 | DRAM |
| RotaryEmbedding |  | 20 | 15186.1 | 2.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 6553.6 | 6553.6 | 2.3 | DRAM |
| MinimalMatmul |  | 28 | 14667.6 | 2.6% | 120 | 32768x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 5799.5 | 9236.0 | 9236.0 | 1.6 | DRAM |
| NlpCreateHeads |  | 20 | 11928.8 | 2.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 10485.8 | 10485.8 | 1.1 | DRAM |
| RotaryEmbedding |  | 20 | 7833.2 | 1.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 3932.2 | 3932.2 | 2.0 | DRAM |
| LayerNorm | RMSNORM | 20 | 6938.5 | 1.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 5242.9 | 5242.9 | 1.3 | DRAM |
| BinaryNg | ADD | 1 | 6456.4 | 1.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1187.8 | 1187.8 | 5.4 | DRAM (far above floor) |
| RotaryEmbedding |  | 4 | 6071.1 | 1.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 2621.4 | 2621.4 | 2.3 | DRAM |
| MinimalMatmul |  | 20 | 5682.7 | 1.0% | 120 | 32768x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2071.3 | 3953.9 | 3953.9 | 1.4 | DRAM |
| MinimalMatmul |  | 24 | 5426.8 | 1.0% | 120 | 32768x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1242.8 | 3158.8 | 3158.8 | 1.7 | DRAM |
| Slice |  | 24 | 4195.4 | 0.7% | 120 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i | 0.0 | 3145.7 | 3145.7 | 1.3 | DRAM |
| MinimalMatmul |  | 4 | 3985.9 | 0.7% | 120 | 32768x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1242.8 | 1848.1 | 1848.1 | 2.2 | DRAM |
| MinimalMatmul |  | 1 | 3804.1 | 0.7% | 120 | 32768x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1242.8 | 1651.5 | 1651.5 | 2.3 | DRAM |
| LayerNorm | RMSNORM | 20 | 3572.3 | 0.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2621.5 | 2621.5 | 1.4 | DRAM |
| NlpCreateHeads |  | 4 | 3563.3 | 0.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 3145.7 | 3145.7 | 1.1 | DRAM |
| LayerNorm | RMSNORM | 20 | 3487.3 | 0.6% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 2621.4 | 2621.4 | 1.3 | DRAM |
| LayerNorm | RMSNORM | 4 | 2730.5 | 0.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2097.2 | 2097.2 | 1.3 | DRAM |
| RotaryEmbedding |  | 4 | 1666.1 | 0.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1048.6 | 1048.6 | 1.6 | DRAM |
| LayerNorm | RMSNORM | 4 | 670.6 | 0.1% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 524.3 | 524.3 | 1.3 | DRAM |
| Matmul |  | 1 | 450.6 | 0.1% | 24 | 32x32768x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 9.7 | 102.6 | 102.6 | 4.4 | DRAM (far above floor) |
| MinimalMatmul |  | 1 | 306.5 | 0.1% | 120 | 32768x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 77.7 | 164.7 | 164.7 | 1.9 | DRAM |
| Reduce |  | 1 | 225.4 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 8.2 | 8.2 | 27.5 | DRAM (far above floor) |
| Embeddings |  | 1 | 184.3 | 0.0% | 120 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 131.3 | 131.3 | 1.4 | DRAM |
| Clone |  | 1 | 31.1 | 0.0% | 120 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 16.4 | 16.4 | 1.9 | DRAM |
| FillPad |  | 1 | 26.2 | 0.0% | 120 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 16.4 | 16.4 | 1.6 | DRAM |
| Typecast |  | 1 | 19.7 | 0.0% | 120 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 12.3 | 12.3 | 1.6 | DRAM |
| BinaryNg | MUL | 1 | 14.3 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 8.2 | 8.2 | 1.7 | DRAM |
| BinaryNg | DIV | 2 | 14.1 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 18.0 | DRAM (far above floor) |
| BinaryNg | SUB | 1 | 13.5 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 8.2 | 8.2 | 1.6 | DRAM |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 34.1 | DRAM (far above floor) |
| FillPad |  | 1 | 4.9 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 12.8 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 2 | 4.6 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 142.6 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.7 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 4.7 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.7 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 166.4 | DRAM (far above floor) |
| Clone |  | 1 | 1.6 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.2 | DRAM (far above floor) |
