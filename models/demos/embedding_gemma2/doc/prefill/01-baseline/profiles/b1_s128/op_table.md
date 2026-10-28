### b1_s128: 645 ops, device kernel sum 4.454 ms, eager op-to-op gaps 12.995 ms, roofline floor 0.319 ms (clock 1350 MHz, 130 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LayerNorm | RMSNORM | 144 | 1317.6 | 29.6% | 4 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 3.6 | 3.6 | 361.1 | L1/NoC (far above floor) |
| Matmul |  | 68 | 543.5 | 12.2% | 52 | 128x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | L1-i/DRAM-i/L1-i | 50.8 | 150.4 | 150.4 | 3.6 | DRAM (far above floor) |
| Matmul |  | 28 | 343.3 | 7.7% | 32 | 128x2048x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 20.9 | 61.9 | 61.9 | 5.5 | DRAM (far above floor) |
| SDPA |  | 20 | 239.4 | 5.4% | 64 | H4 S128 keys128 D256 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 3.7 | 1.1 | 3.7 | 64.1 | compute (far above floor) |
| RotaryEmbedding |  | 20 | 219.1 | 4.9% | 8 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.5 | 0.5 | 455.3 | L1/NoC (far above floor) |
| Matmul |  | 48 | 211.6 | 4.8% | 32 | 128x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 9.0 | 27.2 | 27.2 | 7.8 | DRAM (far above floor) |
| RotaryEmbedding |  | 20 | 208.6 | 4.7% | 16 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.7 | 0.7 | 286.0 | L1/NoC (far above floor) |
| Matmul |  | 20 | 139.2 | 3.1% | 32 | 128x1024x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 7.5 | 22.2 | 22.2 | 6.3 | DRAM (far above floor) |
| BinaryNg | MUL | 24 | 115.9 | 2.6% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 1.8 | 1.8 | 64.9 | L1/NoC (far above floor) |
| BinaryNg | ADD | 71 | 110.7 | 2.5% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 2.5 | 2.5 | 44.5 | L1/NoC (far above floor) |
| SDPA |  | 4 | 109.2 | 2.5% | 64 | H4 S128 keys128 D512 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 1.5 | 0.3 | 1.5 | 73.1 | compute (far above floor) |
| LayerNorm | RMSNORM | 20 | 106.3 | 2.4% | 8 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 0.3 | 0.3 | 396.6 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 106.2 | 2.4% | 16 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 0.5 | 0.5 | 205.8 | L1/NoC (far above floor) |
| BinaryNg | MUL | 24 | 89.6 | 2.0% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 0.8 | 0.8 | 106.6 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 89.4 | 2.0% | 8 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | L1-i/L1-i | 0.0 | 0.2 | 0.2 | 360.5 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 82.4 | 1.9% | 4 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.1 | 0.1 | 576.4 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 79.0 | 1.8% | 16 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.3 | 0.3 | 270.9 | L1/NoC (far above floor) |
| Matmul |  | 4 | 44.6 | 1.0% | 48 | 128x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | L1-i/DRAM-i/L1-i | 4.5 | 13.3 | 13.3 | 3.4 | DRAM (far above floor) |
| NlpCreateHeads |  | 20 | 42.7 | 1.0% | 32 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 1.0 | 1.0 | 43.0 | L1/NoC (far above floor) |
| Matmul |  | 1 | 40.3 | 0.9% | 52 | 128x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 4.5 | 19.5 | 19.5 | 2.1 | DRAM |
| LayerNorm | RMSNORM | 4 | 36.4 | 0.8% | 16 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 0.2 | 0.2 | 176.2 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 36.1 | 0.8% | 4 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 0.1 | 0.1 | 626.0 | L1/NoC (far above floor) |
| Slice |  | 24 | 28.5 | 0.6% | 64 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i | 0.0 | 0.6 | 0.6 | 50.8 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 25.2 | 0.6% | 4 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | L1-i/L1-i | 0.0 | 0.0 | 0.0 | 509.0 | L1/NoC (far above floor) |
| BinaryNg | DIV | 2 | 14.3 | 0.3% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 18.2 | DRAM (far above floor) |
| NlpCreateHeads |  | 4 | 12.3 | 0.3% | 24 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.3 | 0.3 | 41.2 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 1 | 9.5 | 0.2% | 4 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 0.3 | 0.3 | 35.1 | DRAM (far above floor) |
| Reduce |  | 1 | 6.8 | 0.2% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.8 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 5.6 | 0.1% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 111.9 | DRAM (far above floor) |
| Matmul |  | 1 | 5.1 | 0.1% | 48 | 128x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 0.3 | 1.2 | 1.2 | 4.2 | DRAM (far above floor) |
| Embeddings |  | 1 | 4.6 | 0.1% | 4 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.5 | 0.5 | 9.0 | DRAM (far above floor) |
| FillPad |  | 1 | 4.6 | 0.1% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 11.9 | DRAM (far above floor) |
| FillPad |  | 1 | 4.2 | 0.1% | 4 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 65.1 | DRAM (far above floor) |
| Matmul |  | 1 | 3.7 | 0.1% | 26 | 32x128x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 6.3 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 2 | 3.1 | 0.1% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 95.7 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.5 | 0.1% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 78.4 | DRAM (far above floor) |
| Reduce |  | 1 | 2.3 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 58.5 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.1 | 0.0% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 3.7 | DRAM (far above floor) |
| BinaryNg | SUB | 1 | 2.0 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 61.4 | DRAM (far above floor) |
| Unary | SQRT | 1 | 1.8 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 111.6 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 1.6 | 0.0% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 0.3 | 0.3 | 5.9 | L1/NoC (far above floor) |
| Clone |  | 1 | 1.3 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 3.5 | DRAM (far above floor) |
| Typecast |  | 1 | 1.0 | 0.0% | 4 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 21.0 | DRAM (far above floor) |
| Clone |  | 1 | 0.9 | 0.0% | 4 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 14.4 | DRAM (far above floor) |
