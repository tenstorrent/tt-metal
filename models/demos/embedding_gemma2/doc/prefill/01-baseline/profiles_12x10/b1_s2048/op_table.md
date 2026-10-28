### b1_s2048: 646 ops, device kernel sum 25.095 ms, eager op-to-op gaps 0.448 ms, roofline floor 5.791 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 20 | 6390.3 | 25.5% | 120 | H4 S2048 keys1025 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 518.3 | 116.4 | 518.3 | 12.3 | compute (far above floor) |
| SDPA |  | 4 | 2268.9 | 9.0% | 120 | H4 S2048 keys2048 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 414.3 | 182.3 | 414.3 | 5.5 | compute (far above floor) |
| Matmul |  | 44 | 2136.8 | 8.5% | 110 | 2048x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 569.6 | 825.5 | 825.5 | 2.6 | DRAM |
| Matmul | GELU_TANH | 24 | 1854.5 | 7.4% | 110 | 2048x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 310.7 | 450.3 | 450.3 | 4.1 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 144 | 1510.2 | 6.0% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 58.5 | 58.5 | 25.8 | L1/NoC (far above floor) |
| MinimalMatmul |  | 28 | 1417.4 | 5.6% | 120 | 2048x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/L1-i | 362.5 | 525.3 | 525.3 | 2.7 | DRAM |
| BinaryNg | MUL | 24 | 1411.5 | 5.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1179.6 | 1179.6 | 1.2 | DRAM |
| RotaryEmbedding |  | 20 | 1026.4 | 4.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 98.1 | 98.1 | 10.5 | L1/NoC (far above floor) |
| Matmul | GELU_TANH | 24 | 727.0 | 2.9% | 80 | 2048x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 77.7 | 22.8 | 77.7 | 9.4 | compute (far above floor) |
| MinimalMatmul |  | 20 | 722.0 | 2.9% | 120 | 2048x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 129.5 | 33.9 | 129.5 | 5.6 | compute (far above floor) |
| RotaryEmbedding |  | 20 | 637.5 | 2.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 90.0 | 90.0 | 7.1 | L1/NoC (far above floor) |
| MinimalMatmul |  | 24 | 625.2 | 2.5% | 120 | 2048x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 77.7 | 22.8 | 77.7 | 8.0 | compute (far above floor) |
| RotaryEmbedding |  | 4 | 556.1 | 2.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 163.8 | 163.8 | 3.4 | DRAM (far above floor) |
| BinaryNg | ADD | 71 | 477.7 | 1.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 43.1 | 43.1 | 11.1 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 20 | 448.1 | 1.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i/L1-i/L1-i | 0.0 | 343.9 | 343.9 | 1.3 | DRAM |
| LayerNorm | RMSNORM | 20 | 327.9 | 1.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 16.2 | 16.2 | 20.2 | L1/NoC (far above floor) |
| MinimalMatmul |  | 1 | 287.4 | 1.1% | 120 | 2048x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 77.7 | 115.5 | 115.5 | 2.5 | DRAM |
| Matmul |  | 4 | 278.4 | 1.1% | 120 | 2048x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 77.7 | 112.2 | 112.2 | 2.5 | DRAM |
| NlpCreateHeads |  | 4 | 236.0 | 0.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 196.6 | 196.6 | 1.2 | DRAM |
| Slice |  | 24 | 229.6 | 0.9% | 120 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i | 0.0 | 9.7 | 9.7 | 23.6 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 225.5 | 0.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 8.1 | 8.1 | 27.8 | L1/NoC (far above floor) |
| BinaryNg | ADD | 1 | 210.4 | 0.8% | 120 |  | BFLOAT4_B/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 9.5 | 9.5 | 22.2 | DRAM (far above floor) |
| RotaryEmbedding |  | 4 | 207.7 | 0.8% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 65.5 | 65.5 | 3.2 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 4 | 191.7 | 0.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 131.1 | 131.1 | 1.5 | DRAM |
| BinaryNg | MUL | 24 | 188.7 | 0.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 14.6 | 14.6 | 13.0 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 180.6 | 0.7% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 8.1 | 8.1 | 22.3 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 65.5 | 0.3% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 32.8 | 32.8 | 2.0 | DRAM |
| LayerNorm | RMSNORM | 4 | 59.2 | 0.2% | 64 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 32.8 | 32.8 | 1.8 | DRAM |
| BinaryNg | ADD | 1 | 33.5 | 0.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 5.1 | 5.1 | 6.5 | DRAM (far above floor) |
| Matmul |  | 1 | 29.8 | 0.1% | 24 | 32x2048x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.6 | 6.6 | 6.6 | 4.5 | DRAM (far above floor) |
| MinimalMatmul |  | 1 | 29.1 | 0.1% | 120 | 2048x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 4.9 | 7.2 | 7.2 | 4.1 | DRAM (far above floor) |
| Reduce |  | 1 | 15.5 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.5 | 0.5 | 29.9 | DRAM (far above floor) |
| BinaryNg | DIV | 2 | 14.1 | 0.1% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 17.9 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 1 | 13.9 | 0.1% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 4.3 | 4.3 | 3.2 | DRAM (far above floor) |
| Embeddings |  | 1 | 13.8 | 0.1% | 64 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 8.2 | 8.2 | 1.7 | DRAM |
| BinaryNg | ADD | 1 | 9.9 | 0.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 4.5 | 4.5 | 2.2 | L1/NoC |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.9 | DRAM (far above floor) |
| FillPad |  | 1 | 5.1 | 0.0% | 64 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 1.0 | 1.0 | 5.0 | DRAM (far above floor) |
| FillPad |  | 1 | 4.9 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 12.9 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 2 | 4.5 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 141.9 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.8 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.5 | 0.5 | 5.4 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.7 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 4.7 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.6 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 165.5 | DRAM (far above floor) |
| BinaryNg | SUB | 1 | 2.2 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.5 | 0.5 | 4.3 | DRAM (far above floor) |
| Typecast |  | 1 | 1.9 | 0.0% | 64 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 2.5 | DRAM |
| Clone |  | 1 | 1.9 | 0.0% | 64 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 1.0 | 1.0 | 1.8 | DRAM |
| Clone |  | 1 | 1.6 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.2 | DRAM (far above floor) |
