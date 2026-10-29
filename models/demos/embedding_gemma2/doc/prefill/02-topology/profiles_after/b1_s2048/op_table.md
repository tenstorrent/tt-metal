### b1_s2048: 638 ops, device kernel sum 24.875 ms, eager op-to-op gaps 0.471 ms, roofline floor 5.790 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 20 | 6385.3 | 25.7% | 120 | H4 S2048 keys1025 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 518.3 | 116.4 | 518.3 | 12.3 | compute (far above floor) |
| SDPA |  | 4 | 2268.6 | 9.1% | 120 | H4 S2048 keys2048 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 414.3 | 182.3 | 414.3 | 5.5 | compute (far above floor) |
| Matmul |  | 44 | 2141.2 | 8.6% | 110 | 2048x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 569.6 | 825.5 | 825.5 | 2.6 | DRAM |
| Matmul | GELU_TANH | 24 | 1853.0 | 7.4% | 110 | 2048x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 310.7 | 450.3 | 450.3 | 4.1 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 144 | 1507.6 | 6.1% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 58.5 | 58.5 | 25.8 | L1/NoC (far above floor) |
| MinimalMatmul |  | 28 | 1414.6 | 5.7% | 120 | 2048x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/L1-i | 362.5 | 525.3 | 525.3 | 2.7 | DRAM |
| BinaryNg | MUL | 24 | 1412.7 | 5.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1179.6 | 1179.6 | 1.2 | DRAM |
| RotaryEmbedding |  | 20 | 1030.7 | 4.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 98.1 | 98.1 | 10.5 | L1/NoC (far above floor) |
| MinimalMatmul |  | 20 | 732.5 | 2.9% | 120 | 2048x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 129.5 | 33.9 | 129.5 | 5.7 | compute (far above floor) |
| Matmul | GELU_TANH | 24 | 729.6 | 2.9% | 80 | 2048x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 77.7 | 22.8 | 77.7 | 9.4 | compute (far above floor) |
| RotaryEmbedding |  | 20 | 637.0 | 2.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 90.0 | 90.0 | 7.1 | L1/NoC (far above floor) |
| MinimalMatmul |  | 24 | 622.2 | 2.5% | 120 | 2048x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 77.7 | 22.8 | 77.7 | 8.0 | compute (far above floor) |
| RotaryEmbedding |  | 4 | 558.2 | 2.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 163.8 | 163.8 | 3.4 | DRAM (far above floor) |
| BinaryNg | ADD | 71 | 468.9 | 1.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 43.1 | 43.1 | 10.9 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 20 | 446.5 | 1.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i/L1-i/L1-i | 0.0 | 343.9 | 343.9 | 1.3 | DRAM |
| LayerNorm | RMSNORM | 20 | 327.3 | 1.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 16.2 | 16.2 | 20.2 | L1/NoC (far above floor) |
| MinimalMatmul |  | 1 | 296.3 | 1.2% | 120 | 2048x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 77.7 | 115.5 | 115.5 | 2.6 | DRAM |
| Matmul |  | 4 | 277.9 | 1.1% | 120 | 2048x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 77.7 | 112.2 | 112.2 | 2.5 | DRAM |
| NlpCreateHeads |  | 4 | 234.2 | 0.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 196.6 | 196.6 | 1.2 | DRAM |
| Slice |  | 24 | 227.5 | 0.9% | 120 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i | 0.0 | 9.7 | 9.7 | 23.4 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 226.1 | 0.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 8.1 | 8.1 | 27.9 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 208.4 | 0.8% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 65.5 | 65.5 | 3.2 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 4 | 193.6 | 0.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 131.1 | 131.1 | 1.5 | DRAM |
| BinaryNg | MUL | 24 | 190.1 | 0.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 14.6 | 14.6 | 13.1 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 181.4 | 0.7% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 8.1 | 8.1 | 22.4 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 64.7 | 0.3% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 32.8 | 32.8 | 2.0 | DRAM |
| LayerNorm | RMSNORM | 4 | 58.2 | 0.2% | 64 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 32.8 | 32.8 | 1.8 | DRAM |
| BinaryNg | ADD | 1 | 33.3 | 0.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 5.1 | 5.1 | 6.5 | DRAM (far above floor) |
| Matmul |  | 1 | 29.7 | 0.1% | 24 | 32x2048x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.6 | 6.6 | 6.6 | 4.5 | DRAM (far above floor) |
| MinimalMatmul |  | 1 | 28.6 | 0.1% | 120 | 2048x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 4.9 | 7.2 | 7.2 | 4.0 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 22.7 | 0.1% | 120 |  | BFLOAT4_B/BFLOAT4_B/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 13.8 | 13.8 | 1.6 | DRAM |
| LayerNorm | RMSNORM | 1 | 14.0 | 0.1% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 4.3 | 4.3 | 3.3 | DRAM (far above floor) |
| Embeddings |  | 1 | 13.6 | 0.1% | 64 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 8.2 | 8.2 | 1.7 | DRAM |
| BinaryNg | ADD | 1 | 9.8 | 0.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 4.5 | 4.5 | 2.2 | L1/NoC |
| BinaryNg | DIV | 1 | 7.2 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 18.3 | DRAM (far above floor) |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.9 | DRAM (far above floor) |
| FillPad |  | 1 | 5.0 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 12.9 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 3.0 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 5.2 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.7 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 167.9 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 1 | 2.3 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 142.7 | DRAM (far above floor) |
| Clone |  | 1 | 1.7 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.4 | DRAM (far above floor) |
