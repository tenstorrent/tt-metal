### b1_s2048: 638 ops, device kernel sum 23.573 ms, eager op-to-op gaps 0.477 ms, roofline floor 5.790 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 20 | 6385.1 | 27.1% | 120 | H4 S2048 keys1025 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 518.3 | 116.4 | 518.3 | 12.3 | compute (far above floor) |
| Matmul |  | 68 | 2917.1 | 12.4% | 110 | 2048x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 880.3 | 1275.8 | 1275.8 | 2.3 | DRAM |
| SDPA |  | 4 | 2267.6 | 9.6% | 120 | H4 S2048 keys2048 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 414.3 | 182.3 | 414.3 | 5.5 | compute (far above floor) |
| BinaryNg | MUL | 24 | 1958.3 | 8.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1179.6 | 1179.6 | 1.7 | DRAM |
| LayerNorm | RMSNORM | 144 | 1528.5 | 6.5% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 58.5 | 58.5 | 26.1 | L1/NoC (far above floor) |
| MinimalMatmul |  | 28 | 1304.5 | 5.5% | 120 | 2048x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/L1-i | 362.5 | 525.3 | 525.3 | 2.5 | DRAM |
| RotaryEmbedding |  | 20 | 1026.7 | 4.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 98.1 | 98.1 | 10.5 | L1/NoC (far above floor) |
| Matmul |  | 48 | 681.4 | 2.9% | 80 | 2048x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 155.3 | 45.5 | 155.3 | 4.4 | compute (far above floor) |
| RotaryEmbedding |  | 20 | 643.2 | 2.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 90.0 | 90.0 | 7.1 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 559.7 | 2.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 163.8 | 163.8 | 3.4 | DRAM (far above floor) |
| BinaryNg | MUL | 24 | 534.8 | 2.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 14.6 | 14.6 | 36.7 | L1/NoC (far above floor) |
| BinaryNg | ADD | 71 | 466.6 | 2.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 43.1 | 43.1 | 10.8 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 20 | 444.9 | 1.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i/L1-i/L1-i | 0.0 | 343.9 | 343.9 | 1.3 | DRAM |
| Matmul |  | 20 | 430.3 | 1.8% | 80 | 2048x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 129.5 | 33.9 | 129.5 | 3.3 | compute (far above floor) |
| LayerNorm | RMSNORM | 20 | 328.1 | 1.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 16.2 | 16.2 | 20.2 | L1/NoC (far above floor) |
| MinimalMatmul |  | 1 | 279.3 | 1.2% | 120 | 2048x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 77.7 | 115.5 | 115.5 | 2.4 | DRAM |
| Matmul |  | 4 | 247.1 | 1.0% | 120 | 2048x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 77.7 | 112.2 | 112.2 | 2.2 | DRAM |
| NlpCreateHeads |  | 4 | 233.2 | 1.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 196.6 | 196.6 | 1.2 | DRAM |
| Slice |  | 24 | 227.1 | 1.0% | 120 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i | 0.0 | 9.7 | 9.7 | 23.4 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 225.4 | 1.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 8.1 | 8.1 | 27.8 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 208.2 | 0.9% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 65.5 | 65.5 | 3.2 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 4 | 191.3 | 0.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 131.1 | 131.1 | 1.5 | DRAM |
| LayerNorm | RMSNORM | 20 | 182.0 | 0.8% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 8.1 | 8.1 | 22.5 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 64.4 | 0.3% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 32.8 | 32.8 | 2.0 | DRAM |
| LayerNorm | RMSNORM | 4 | 58.6 | 0.2% | 64 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 32.8 | 32.8 | 1.8 | DRAM |
| BinaryNg | ADD | 1 | 33.0 | 0.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 5.1 | 5.1 | 6.4 | DRAM (far above floor) |
| Matmul |  | 1 | 29.7 | 0.1% | 24 | 32x2048x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.6 | 6.6 | 6.6 | 4.5 | DRAM (far above floor) |
| MinimalMatmul |  | 1 | 28.4 | 0.1% | 120 | 2048x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 4.9 | 7.2 | 7.2 | 4.0 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 22.5 | 0.1% | 120 |  | BFLOAT4_B/BFLOAT4_B/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 13.8 | 13.8 | 1.6 | DRAM |
| Embeddings |  | 1 | 14.3 | 0.1% | 64 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 8.2 | 8.2 | 1.7 | DRAM |
| LayerNorm | RMSNORM | 1 | 13.7 | 0.1% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 4.3 | 4.3 | 3.2 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 9.6 | 0.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 4.5 | 4.5 | 2.1 | L1/NoC |
| BinaryNg | DIV | 1 | 7.0 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 17.9 | DRAM (far above floor) |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 34.0 | DRAM (far above floor) |
| FillPad |  | 1 | 4.9 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 12.9 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 3.0 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 5.1 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.7 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 168.2 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 1 | 2.3 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 141.9 | DRAM (far above floor) |
| Clone |  | 1 | 1.6 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.3 | DRAM (far above floor) |
