### b1_s256: 645 ops, device kernel sum 5.120 ms, eager op-to-op gaps 10.744 ms, roofline floor 0.362 ms (clock 1350 MHz, 130 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LayerNorm | RMSNORM | 144 | 1333.5 | 26.0% | 8 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 7.0 | 7.0 | 190.2 | L1/NoC (far above floor) |
| SDPA |  | 20 | 577.2 | 11.3% | 64 | H4 S256 keys256 D256 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 14.9 | 2.9 | 14.9 | 38.6 | compute (far above floor) |
| Matmul |  | 68 | 554.8 | 10.8% | 104 | 256x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | L1-i/DRAM-i/L1-i | 101.6 | 152.9 | 152.9 | 3.6 | DRAM (far above floor) |
| Matmul |  | 28 | 356.4 | 7.0% | 64 | 256x2048x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 41.8 | 63.0 | 63.0 | 5.7 | DRAM (far above floor) |
| SDPA |  | 4 | 229.7 | 4.5% | 64 | H4 S256 keys256 D512 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 6.0 | 0.8 | 6.0 | 38.4 | compute (far above floor) |
| Matmul |  | 48 | 219.9 | 4.3% | 64 | 256x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 17.9 | 28.4 | 28.4 | 7.8 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 20 | 218.8 | 4.3% | 16 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 1.0 | 1.0 | 227.3 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 20 | 212.5 | 4.2% | 32 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 1.5 | 1.5 | 145.7 | L1/NoC (far above floor) |
| BinaryNg | MUL | 24 | 201.7 | 3.9% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 3.6 | 3.6 | 56.5 | L1/NoC (far above floor) |
| Matmul |  | 20 | 145.6 | 2.8% | 64 | 256x1024x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 14.9 | 22.7 | 22.7 | 6.4 | DRAM (far above floor) |
| BinaryNg | ADD | 71 | 125.2 | 2.4% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 5.0 | 5.0 | 25.2 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 111.0 | 2.2% | 32 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 1.0 | 1.0 | 109.7 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 108.1 | 2.1% | 16 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 0.5 | 0.5 | 209.5 | L1/NoC (far above floor) |
| BinaryNg | MUL | 24 | 95.4 | 1.9% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 1.7 | 1.7 | 56.8 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 90.5 | 1.8% | 16 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | L1-i/L1-i | 0.0 | 0.5 | 0.5 | 182.4 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 84.1 | 1.6% | 8 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.3 | 0.3 | 294.1 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 81.0 | 1.6% | 32 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.6 | 0.6 | 138.7 | L1/NoC (far above floor) |
| Matmul |  | 1 | 49.1 | 1.0% | 104 | 256x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 9.0 | 25.9 | 25.9 | 1.9 | DRAM |
| NlpCreateHeads |  | 20 | 46.9 | 0.9% | 64 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 2.0 | 2.0 | 23.6 | L1/NoC (far above floor) |
| Matmul |  | 4 | 45.0 | 0.9% | 96 | 256x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | L1-i/DRAM-i/L1-i | 9.0 | 13.4 | 13.4 | 3.3 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 4 | 37.7 | 0.7% | 32 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 0.4 | 0.4 | 93.1 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 37.3 | 0.7% | 8 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 0.1 | 0.1 | 347.7 | L1/NoC (far above floor) |
| Slice |  | 24 | 33.9 | 0.7% | 128 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i | 0.0 | 1.1 | 1.1 | 30.3 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 26.3 | 0.5% | 8 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | L1-i/L1-i | 0.0 | 0.1 | 0.1 | 265.3 | L1/NoC (far above floor) |
| BinaryNg | DIV | 2 | 14.6 | 0.3% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 18.6 | DRAM (far above floor) |
| NlpCreateHeads |  | 4 | 14.0 | 0.3% | 48 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.6 | 0.6 | 23.6 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 1 | 9.6 | 0.2% | 8 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 0.5 | 0.5 | 17.8 | DRAM (far above floor) |
| Embeddings |  | 1 | 6.8 | 0.1% | 8 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1.0 | 1.0 | 6.7 | DRAM (far above floor) |
| Reduce |  | 1 | 6.8 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.8 | DRAM (far above floor) |
| Matmul |  | 1 | 6.1 | 0.1% | 96 | 256x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 0.6 | 1.6 | 1.6 | 3.8 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 5.9 | 0.1% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 43.5 | DRAM (far above floor) |
| Matmul |  | 1 | 5.5 | 0.1% | 26 | 32x256x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.1 | 1.0 | 1.0 | 5.5 | DRAM (far above floor) |
| FillPad |  | 1 | 4.5 | 0.1% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 11.7 | DRAM (far above floor) |
| FillPad |  | 1 | 4.4 | 0.1% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 34.1 | DRAM (far above floor) |
| Reduce |  | 1 | 3.3 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 45.6 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 2 | 2.8 | 0.1% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 88.8 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.7 | 0.1% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 42.1 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.1 | 0.0% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 3.6 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 2.0 | 0.0% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 0.6 | 0.6 | 3.6 | L1/NoC (far above floor) |
| BinaryNg | SUB | 1 | 2.0 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 31.3 | DRAM (far above floor) |
| Unary | SQRT | 1 | 1.8 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 111.1 | DRAM (far above floor) |
| Clone |  | 1 | 1.3 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 3.4 | DRAM (far above floor) |
| Typecast |  | 1 | 1.1 | 0.0% | 8 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 11.8 | DRAM (far above floor) |
| Clone |  | 1 | 1.0 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 7.8 | DRAM (far above floor) |
