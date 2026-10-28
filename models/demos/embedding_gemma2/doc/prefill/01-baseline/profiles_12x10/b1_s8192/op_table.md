### b1_s8192: 646 ops, device kernel sum 103.377 ms, eager op-to-op gaps 0.525 ms, roofline floor 50.854 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 4 | 16486.5 | 15.9% | 120 | H4 S8192 keys8192 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 6628.0 | 950.3 | 6628.0 | 2.5 | compute |
| SDPA |  | 20 | 14949.5 | 14.5% | 120 | H4 S8192 keys1025 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 2073.3 | 3440.6 | 3440.6 | 4.3 | DRAM (far above floor) |
| MinimalMatmul | GELU_TANH | 24 | 9987.8 | 9.7% | 120 | 8192x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1242.8 | 2018.3 | 2018.3 | 4.9 | DRAM (far above floor) |
| MinimalMatmul |  | 44 | 8259.5 | 8.0% | 120 | 8192x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2278.4 | 3700.2 | 3700.2 | 2.2 | DRAM |
| LayerNorm | RMSNORM | 149 | 7131.6 | 6.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 4882.7 | 4882.7 | 1.5 | DRAM |
| BinaryNg | MUL | 48 | 6915.4 | 6.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 9437.2 | 9437.2 | 0.7 | DRAM |
| RotaryEmbedding |  | 20 | 4304.4 | 4.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1638.4 | 1638.4 | 2.6 | DRAM |
| BinaryNg | ADD | 72 | 4151.5 | 4.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 3538.9 | 3538.9 | 1.2 | DRAM |
| MinimalMatmul |  | 28 | 4038.7 | 3.9% | 120 | 8192x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1449.9 | 2354.7 | 2354.7 | 1.7 | DRAM |
| Matmul | GELU_TANH | 24 | 3389.2 | 3.3% | 80 | 8192x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 310.7 | 799.5 | 799.5 | 4.2 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 3246.5 | 3.1% | 120 |  | BFLOAT4_B/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 148.5 | 148.5 | 21.9 | DRAM (far above floor) |
| NlpCreateHeads |  | 20 | 3011.7 | 2.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 2621.4 | 2621.4 | 1.1 | DRAM |
| RotaryEmbedding |  | 20 | 2559.0 | 2.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 983.0 | 983.0 | 2.6 | DRAM |
| LayerNorm | RMSNORM | 20 | 1878.3 | 1.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1310.7 | 1310.7 | 1.4 | DRAM |
| MinimalMatmul |  | 20 | 1747.5 | 1.7% | 120 | 8192x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 517.8 | 1004.8 | 1004.8 | 1.7 | DRAM |
| RotaryEmbedding |  | 4 | 1656.9 | 1.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 655.4 | 655.4 | 2.5 | DRAM |
| MinimalMatmul |  | 24 | 1519.0 | 1.5% | 120 | 8192x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 310.7 | 799.5 | 799.5 | 1.9 | DRAM |
| MinimalMatmul |  | 4 | 1080.8 | 1.0% | 120 | 8192x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 310.7 | 471.8 | 471.8 | 2.3 | DRAM |
| MinimalMatmul |  | 1 | 1070.3 | 1.0% | 120 | 8192x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 310.7 | 422.7 | 422.7 | 2.5 | DRAM |
| Slice |  | 24 | 1055.8 | 1.0% | 120 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i | 0.0 | 786.4 | 786.4 | 1.3 | DRAM |
| LayerNorm | RMSNORM | 20 | 929.2 | 0.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 655.4 | 655.4 | 1.4 | DRAM |
| NlpCreateHeads |  | 4 | 898.9 | 0.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 786.4 | 786.4 | 1.1 | DRAM |
| LayerNorm | RMSNORM | 20 | 894.3 | 0.9% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 655.4 | 655.4 | 1.4 | DRAM |
| LayerNorm | RMSNORM | 4 | 690.7 | 0.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 524.3 | 524.3 | 1.3 | DRAM |
| RotaryEmbedding |  | 4 | 575.6 | 0.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 262.1 | 262.1 | 2.2 | DRAM |
| BinaryNg | ADD | 1 | 414.6 | 0.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 75.8 | 75.8 | 5.5 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 4 | 165.7 | 0.2% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 131.1 | 131.1 | 1.3 | DRAM |
| Matmul |  | 1 | 113.8 | 0.1% | 24 | 32x8192x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 2.4 | 25.8 | 25.8 | 4.4 | DRAM (far above floor) |
| MinimalMatmul |  | 1 | 84.0 | 0.1% | 120 | 8192x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 19.4 | 41.8 | 41.8 | 2.0 | DRAM |
| Reduce |  | 1 | 57.4 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 2.1 | 2.1 | 27.9 | DRAM (far above floor) |
| Embeddings |  | 1 | 47.2 | 0.0% | 120 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 32.8 | 32.8 | 1.4 | DRAM |
| BinaryNg | DIV | 2 | 14.1 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 18.0 | DRAM (far above floor) |
| FillPad |  | 1 | 7.9 | 0.0% | 120 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 4.1 | 4.1 | 1.9 | DRAM |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 34.1 | DRAM (far above floor) |
| Clone |  | 1 | 6.2 | 0.0% | 120 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 4.1 | 4.1 | 1.5 | DRAM |
| Typecast |  | 1 | 5.1 | 0.0% | 120 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 3.1 | 3.1 | 1.6 | DRAM |
| FillPad |  | 1 | 5.0 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 13.0 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 4.6 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 2.0 | 2.0 | 2.2 | DRAM |
| Unary | CLAMP_TSS | 2 | 4.5 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 142.1 | DRAM (far above floor) |
| BinaryNg | SUB | 1 | 4.2 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 2.0 | 2.0 | 2.1 | DRAM |
| BinaryNg | MUL | 1 | 2.7 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 4.7 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.6 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 164.7 | DRAM (far above floor) |
| Clone |  | 1 | 1.7 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.3 | DRAM (far above floor) |
