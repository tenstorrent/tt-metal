### b1_s8192: 637 ops, device kernel sum 99.983 ms, eager op-to-op gaps 0.501 ms, roofline floor 50.499 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 4 | 16485.0 | 16.5% | 120 | H4 S8192 keys8192 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 6628.0 | 950.3 | 6628.0 | 2.5 | compute |
| SDPA |  | 20 | 14993.2 | 15.0% | 120 | H4 S8192 keys1025 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 2073.3 | 3440.6 | 3440.6 | 4.4 | DRAM (far above floor) |
| MinimalMatmul | GELU_TANH | 24 | 9988.2 | 10.0% | 120 | 8192x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1242.8 | 2018.3 | 2018.3 | 4.9 | DRAM (far above floor) |
| MinimalMatmul |  | 44 | 8255.4 | 8.3% | 120 | 8192x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2278.4 | 3700.2 | 3700.2 | 2.2 | DRAM |
| LayerNorm | RMSNORM | 149 | 7151.0 | 7.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 4882.7 | 4882.7 | 1.5 | DRAM |
| BinaryNg | MUL | 48 | 6968.4 | 7.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 9437.2 | 9437.2 | 0.7 | DRAM |
| RotaryEmbedding |  | 20 | 4305.5 | 4.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1638.4 | 1638.4 | 2.6 | DRAM |
| BinaryNg | ADD | 72 | 4165.3 | 4.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 3538.9 | 3538.9 | 1.2 | DRAM |
| MinimalMatmul |  | 28 | 4070.1 | 4.1% | 120 | 8192x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1449.9 | 2354.7 | 2354.7 | 1.7 | DRAM |
| Matmul | GELU_TANH | 24 | 3390.0 | 3.4% | 80 | 8192x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 310.7 | 799.5 | 799.5 | 4.2 | DRAM (far above floor) |
| MinimalMatmul |  | 48 | 3048.7 | 3.0% | 120 | 8192x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 621.4 | 1599.0 | 1599.0 | 1.9 | DRAM |
| NlpCreateHeads |  | 20 | 3010.7 | 3.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 2621.4 | 2621.4 | 1.1 | DRAM |
| RotaryEmbedding |  | 20 | 2561.9 | 2.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 983.0 | 983.0 | 2.6 | DRAM |
| LayerNorm | RMSNORM | 20 | 1869.1 | 1.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1310.7 | 1310.7 | 1.4 | DRAM |
| MinimalMatmul |  | 20 | 1748.3 | 1.7% | 120 | 8192x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 517.8 | 1004.8 | 1004.8 | 1.7 | DRAM |
| RotaryEmbedding |  | 4 | 1661.3 | 1.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 655.4 | 655.4 | 2.5 | DRAM |
| MinimalMatmul |  | 4 | 1091.1 | 1.1% | 120 | 8192x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 310.7 | 471.8 | 471.8 | 2.3 | DRAM |
| LayerNorm | RMSNORM | 20 | 935.1 | 0.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 655.4 | 655.4 | 1.4 | DRAM |
| LayerNorm | RMSNORM | 20 | 904.5 | 0.9% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 655.4 | 655.4 | 1.4 | DRAM |
| NlpCreateHeads |  | 4 | 898.7 | 0.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 786.4 | 786.4 | 1.1 | DRAM |
| LayerNorm | RMSNORM | 4 | 690.3 | 0.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 524.3 | 524.3 | 1.3 | DRAM |
| RotaryEmbedding |  | 4 | 572.7 | 0.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 262.1 | 262.1 | 2.2 | DRAM |
| BinaryNg | ADD | 1 | 420.2 | 0.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 75.8 | 75.8 | 5.5 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 352.9 | 0.4% | 120 |  | BFLOAT4_B/BFLOAT4_B/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 221.2 | 221.2 | 1.6 | DRAM |
| LayerNorm | RMSNORM | 4 | 169.0 | 0.2% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 131.1 | 131.1 | 1.3 | DRAM |
| Matmul |  | 1 | 113.7 | 0.1% | 24 | 32x8192x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 2.4 | 25.8 | 25.8 | 4.4 | DRAM (far above floor) |
| MinimalMatmul |  | 1 | 85.5 | 0.1% | 120 | 8192x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 19.4 | 41.8 | 41.8 | 2.0 | DRAM |
| Embeddings |  | 1 | 48.7 | 0.0% | 120 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 32.8 | 32.8 | 1.5 | DRAM |
| BinaryNg | DIV | 1 | 7.0 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 17.8 | DRAM (far above floor) |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.8 | DRAM (far above floor) |
| FillPad |  | 1 | 5.0 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 12.9 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.9 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 5.1 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.7 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 166.1 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 1 | 2.3 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 142.0 | DRAM (far above floor) |
| Clone |  | 1 | 1.7 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.4 | DRAM (far above floor) |
