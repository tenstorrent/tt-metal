### b1_s128: 637 ops, device kernel sum 4.518 ms, eager op-to-op gaps 13.495 ms, roofline floor 0.331 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LayerNorm | RMSNORM | 148 | 1348.6 | 29.8% | 4 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 4.0 | 4.0 | 334.0 | L1/NoC (far above floor) |
| Matmul |  | 68 | 601.1 | 13.3% | 44-64 | 128x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 55.0 | 152.3 | 152.3 | 3.9 | DRAM (far above floor) |
| Matmul |  | 28 | 343.2 | 7.6% | 32 | 128x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 22.7 | 62.7 | 62.7 | 5.5 | DRAM (far above floor) |
| BinaryNg | MUL | 48 | 287.8 | 6.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 7.3 | 7.3 | 39.5 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 20 | 224.5 | 5.0% | 8 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.8 | 0.8 | 295.9 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 20 | 213.9 | 4.7% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 1.3 | 1.3 | 169.2 | L1/NoC (far above floor) |
| Matmul |  | 48 | 200.5 | 4.4% | 32 | 128x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 9.7 | 27.3 | 27.3 | 7.3 | DRAM (far above floor) |
| SDPA |  | 20 | 182.6 | 4.0% | 64 | H4 S128 keys128 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 4.0 | 1.9 | 4.0 | 45.1 | compute (far above floor) |
| Matmul |  | 20 | 140.2 | 3.1% | 32 | 128x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 8.1 | 22.5 | 22.5 | 6.2 | DRAM (far above floor) |
| BinaryNg | ADD | 71 | 113.1 | 2.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 2.7 | 2.7 | 42.0 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 109.7 | 2.4% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 1.0 | 1.0 | 106.4 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 107.7 | 2.4% | 8 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 0.5 | 0.5 | 204.9 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 92.0 | 2.0% | 8 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 0.5 | 0.5 | 181.9 | L1/NoC (far above floor) |
| SDPA |  | 4 | 83.0 | 1.8% | 64 | H4 S128 keys128 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 1.6 | 0.6 | 1.6 | 51.3 | compute (far above floor) |
| RotaryEmbedding |  | 4 | 81.8 | 1.8% | 4 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.2 | 0.2 | 404.5 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 79.9 | 1.8% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.5 | 0.5 | 158.0 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 20 | 69.3 | 1.5% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 2.0 | 2.0 | 34.2 | L1/NoC (far above floor) |
| Matmul |  | 4 | 37.8 | 0.8% | 96 | 128x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 4.9 | 13.4 | 13.4 | 2.8 | DRAM |
| LayerNorm | RMSNORM | 4 | 37.1 | 0.8% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 0.4 | 0.4 | 89.9 | L1/NoC (far above floor) |
| Matmul |  | 1 | 34.2 | 0.8% | 96 | 128x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 4.9 | 19.5 | 19.5 | 1.8 | DRAM |
| Slice |  | 24 | 28.1 | 0.6% | 64 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i | 0.0 | 0.6 | 0.6 | 46.2 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 25.8 | 0.6% | 4 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 0.1 | 0.1 | 255.2 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 4 | 20.6 | 0.5% | 24 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.6 | 0.6 | 34.0 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 1 | 9.5 | 0.2% | 4 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 0.3 | 0.3 | 35.0 | DRAM (far above floor) |
| BinaryNg | DIV | 1 | 7.2 | 0.2% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 18.4 | DRAM (far above floor) |
| Reduce |  | 1 | 6.7 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.7 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 6.1 | 0.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 121.0 | DRAM (far above floor) |
| Matmul |  | 1 | 5.0 | 0.1% | 48 | 128x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 0.3 | 1.2 | 1.2 | 4.2 | DRAM (far above floor) |
| Embeddings |  | 1 | 4.6 | 0.1% | 4 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.5 | 0.5 | 8.9 | DRAM (far above floor) |
| FillPad |  | 1 | 4.5 | 0.1% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 11.6 | DRAM (far above floor) |
| Matmul |  | 1 | 3.8 | 0.1% | 24 | 32x128x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 6.4 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.2 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 3.9 | DRAM (far above floor) |
| Unary | SQRT | 1 | 1.8 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 110.9 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 1.7 | 0.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 0.3 | 0.3 | 6.0 | L1/NoC (far above floor) |
| Unary | CLAMP_TSS | 1 | 1.4 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 89.4 | DRAM (far above floor) |
| Clone |  | 1 | 1.3 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 3.5 | DRAM (far above floor) |
