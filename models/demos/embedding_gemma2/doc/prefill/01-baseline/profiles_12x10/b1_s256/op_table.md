### b1_s256: 645 ops, device kernel sum 5.584 ms, eager op-to-op gaps 11.854 ms, roofline floor 0.388 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LayerNorm | RMSNORM | 148 | 1368.9 | 24.5% | 8 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 7.8 | 7.8 | 176.0 | L1/NoC (far above floor) |
| Matmul |  | 68 | 629.4 | 11.3% | 88 | 256x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 110.0 | 156.6 | 156.6 | 4.0 | L1/NoC (far above floor) |
| SDPA |  | 20 | 620.3 | 11.1% | 64 | H4 S256 keys256 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 16.2 | 4.5 | 16.2 | 38.3 | compute (far above floor) |
| BinaryNg | MUL | 48 | 450.4 | 8.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 14.6 | 14.6 | 30.9 | L1/NoC (far above floor) |
| Matmul |  | 28 | 355.9 | 6.4% | 64 | 256x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 45.3 | 64.5 | 64.5 | 5.5 | L1/NoC (far above floor) |
| SDPA |  | 4 | 328.8 | 5.9% | 64 | H4 S256 keys256 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 6.5 | 1.3 | 6.5 | 50.8 | compute (far above floor) |
| RotaryEmbedding |  | 20 | 225.6 | 4.0% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 1.5 | 1.5 | 148.7 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 20 | 219.2 | 3.9% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 2.5 | 2.5 | 86.7 | L1/NoC (far above floor) |
| Matmul |  | 48 | 218.3 | 3.9% | 64 | 256x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 19.4 | 28.5 | 28.5 | 7.6 | L1/NoC (far above floor) |
| BinaryNg | ADD | 71 | 145.6 | 2.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 5.4 | 5.4 | 27.0 | L1/NoC (far above floor) |
| Matmul |  | 20 | 144.3 | 2.6% | 64 | 256x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 16.2 | 23.3 | 23.3 | 6.2 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 114.6 | 2.1% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 2.0 | 2.0 | 56.1 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 109.6 | 2.0% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 1.0 | 1.0 | 106.3 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 94.9 | 1.7% | 16 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 1.0 | 1.0 | 93.8 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 81.9 | 1.5% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 1.0 | 1.0 | 81.0 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 80.8 | 1.4% | 8 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.4 | 0.4 | 199.8 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 20 | 77.9 | 1.4% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 4.0 | 4.0 | 19.3 | L1/NoC (far above floor) |
| Matmul |  | 1 | 54.3 | 1.0% | 96 | 256x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 9.7 | 25.9 | 25.9 | 2.1 | DRAM |
| Matmul |  | 4 | 47.0 | 0.8% | 96 | 256x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 9.7 | 13.8 | 13.8 | 3.4 | L1/NoC (far above floor) |
| Slice |  | 24 | 44.8 | 0.8% | 120 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i | 0.0 | 1.2 | 1.2 | 36.9 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 38.7 | 0.7% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 0.8 | 0.8 | 47.3 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 26.3 | 0.5% | 8 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 0.2 | 0.2 | 129.9 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 4 | 22.4 | 0.4% | 48 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 1.2 | 1.2 | 18.5 | L1/NoC (far above floor) |
| BinaryNg | DIV | 2 | 14.7 | 0.3% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 18.7 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 1 | 9.6 | 0.2% | 8 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 0.5 | 0.5 | 17.8 | DRAM (far above floor) |
| Reduce |  | 1 | 6.7 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.7 | DRAM (far above floor) |
| Embeddings |  | 1 | 6.7 | 0.1% | 8 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1.0 | 1.0 | 6.5 | DRAM (far above floor) |
| Matmul |  | 1 | 6.0 | 0.1% | 96 | 256x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 0.6 | 1.6 | 1.6 | 3.7 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 5.9 | 0.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 43.3 | DRAM (far above floor) |
| Matmul |  | 1 | 5.5 | 0.1% | 24 | 32x256x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.1 | 1.0 | 1.0 | 5.5 | DRAM (far above floor) |
| FillPad |  | 1 | 4.6 | 0.1% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 11.9 | DRAM (far above floor) |
| FillPad |  | 1 | 4.3 | 0.1% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 33.8 | DRAM (far above floor) |
| Reduce |  | 1 | 3.2 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 44.7 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 2 | 2.8 | 0.1% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 88.6 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.5 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 39.8 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.2 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 3.8 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 2.1 | 0.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 0.6 | 0.6 | 3.8 | L1/NoC (far above floor) |
| BinaryNg | SUB | 1 | 2.0 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 30.9 | DRAM (far above floor) |
| Unary | SQRT | 1 | 1.8 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 110.6 | DRAM (far above floor) |
| Clone |  | 1 | 1.4 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 3.5 | DRAM (far above floor) |
| Typecast |  | 1 | 1.1 | 0.0% | 8 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 11.6 | DRAM (far above floor) |
| Clone |  | 1 | 1.0 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 7.9 | DRAM (far above floor) |
