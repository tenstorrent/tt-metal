### b1_s8192: 637 ops, device kernel sum 94.426 ms, eager op-to-op gaps 0.471 ms, roofline floor 50.499 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 4 | 16533.2 | 17.5% | 120 | H4 S8192 keys8192 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 6628.0 | 950.3 | 6628.0 | 2.5 | compute |
| SDPA |  | 20 | 14977.2 | 15.9% | 120 | H4 S8192 keys1025 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 2073.3 | 3440.6 | 3440.6 | 4.4 | DRAM (far above floor) |
| BinaryNg | MUL | 48 | 9430.0 | 10.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 9437.2 | 9437.2 | 1.0 | DRAM |
| MinimalMatmul |  | 44 | 8230.0 | 8.7% | 120 | 8192x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2278.4 | 3700.2 | 3700.2 | 2.2 | DRAM |
| LayerNorm | RMSNORM | 149 | 7168.5 | 7.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 4882.7 | 4882.7 | 1.5 | DRAM |
| MinimalMatmul |  | 72 | 4469.4 | 4.7% | 120 | 8192x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 932.1 | 2398.5 | 2398.5 | 1.9 | DRAM |
| RotaryEmbedding |  | 20 | 4307.3 | 4.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1638.4 | 1638.4 | 2.6 | DRAM |
| Matmul |  | 24 | 4280.1 | 4.5% | 110 | 2x4096x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1242.8 | 2018.3 | 2018.3 | 2.1 | DRAM |
| BinaryNg | ADD | 72 | 4171.1 | 4.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 3538.9 | 3538.9 | 1.2 | DRAM |
| MinimalMatmul |  | 28 | 3863.5 | 4.1% | 120 | 8192x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1449.9 | 2354.7 | 2354.7 | 1.6 | DRAM |
| NlpCreateHeads |  | 20 | 3011.4 | 3.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 2621.4 | 2621.4 | 1.1 | DRAM |
| RotaryEmbedding |  | 20 | 2559.4 | 2.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 983.0 | 983.0 | 2.6 | DRAM |
| LayerNorm | RMSNORM | 20 | 1871.9 | 2.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1310.7 | 1310.7 | 1.4 | DRAM |
| RotaryEmbedding |  | 4 | 1648.8 | 1.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 655.4 | 655.4 | 2.5 | DRAM |
| MinimalMatmul |  | 20 | 1573.9 | 1.7% | 120 | 8192x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 517.8 | 1004.8 | 1004.8 | 1.6 | DRAM |
| MinimalMatmul |  | 4 | 1109.4 | 1.2% | 120 | 8192x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 310.7 | 471.8 | 471.8 | 2.4 | DRAM |
| LayerNorm | RMSNORM | 20 | 935.8 | 1.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 655.4 | 655.4 | 1.4 | DRAM |
| LayerNorm | RMSNORM | 20 | 902.9 | 1.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 655.4 | 655.4 | 1.4 | DRAM |
| NlpCreateHeads |  | 4 | 902.4 | 1.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 786.4 | 786.4 | 1.1 | DRAM |
| LayerNorm | RMSNORM | 4 | 692.3 | 0.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 524.3 | 524.3 | 1.3 | DRAM |
| RotaryEmbedding |  | 4 | 566.5 | 0.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 262.1 | 262.1 | 2.2 | DRAM |
| BinaryNg | ADD | 1 | 424.5 | 0.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 75.8 | 75.8 | 5.6 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 354.5 | 0.4% | 120 |  | BFLOAT4_B/BFLOAT4_B/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 221.2 | 221.2 | 1.6 | DRAM |
| LayerNorm | RMSNORM | 4 | 168.5 | 0.2% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 131.1 | 131.1 | 1.3 | DRAM |
| Matmul |  | 1 | 114.0 | 0.1% | 24 | 32x8192x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 2.4 | 25.8 | 25.8 | 4.4 | DRAM (far above floor) |
| MinimalMatmul |  | 1 | 83.4 | 0.1% | 120 | 8192x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 19.4 | 41.8 | 41.8 | 2.0 | DRAM |
| Embeddings |  | 1 | 47.8 | 0.1% | 120 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 32.8 | 32.8 | 1.5 | DRAM |
| BinaryNg | DIV | 1 | 7.0 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 17.9 | DRAM (far above floor) |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.9 | DRAM (far above floor) |
| FillPad |  | 1 | 5.0 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 12.9 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.9 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 5.1 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.7 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 167.4 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 1 | 2.3 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 142.0 | DRAM (far above floor) |
| Clone |  | 1 | 1.7 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.3 | DRAM (far above floor) |
