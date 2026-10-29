### b1_s1024: 638 ops, device kernel sum 11.446 ms, eager op-to-op gaps 5.977 ms, roofline floor 1.539 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 20 | 2093.8 | 18.3% | 64 | H4 S1024 keys1024 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 258.9 | 35.2 | 258.9 | 8.1 | compute (far above floor) |
| LayerNorm | RMSNORM | 148 | 1481.1 | 12.9% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 30.2 | 30.2 | 49.0 | L1/NoC (far above floor) |
| BinaryNg | MUL | 48 | 1320.9 | 11.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 58.3 | 58.3 | 22.7 | L1/NoC (far above floor) |
| Matmul |  | 68 | 1204.1 | 10.5% | 88 | 1024x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 440.1 | 182.4 | 440.1 | 2.7 | compute |
| SDPA |  | 4 | 944.5 | 8.3% | 64 | H4 S1024 keys1024 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 103.6 | 8.7 | 103.6 | 9.1 | compute (far above floor) |
| Matmul |  | 28 | 690.2 | 6.0% | 64 | 1024x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 181.2 | 75.1 | 181.2 | 3.8 | compute (far above floor) |
| RotaryEmbedding |  | 20 | 646.0 | 5.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 49.1 | 49.1 | 13.2 | L1/NoC (far above floor) |
| Matmul |  | 48 | 438.7 | 3.8% | 64 | 1024x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 77.7 | 35.8 | 77.7 | 5.6 | compute (far above floor) |
| RotaryEmbedding |  | 20 | 419.7 | 3.7% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 45.0 | 45.0 | 9.3 | L1/NoC (far above floor) |
| BinaryNg | ADD | 71 | 286.7 | 2.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 21.5 | 21.5 | 13.3 | L1/NoC (far above floor) |
| Matmul |  | 20 | 283.3 | 2.5% | 64 | 1024x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 64.7 | 27.8 | 64.7 | 4.4 | compute (far above floor) |
| LayerNorm | RMSNORM | 20 | 223.9 | 2.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 8.1 | 8.1 | 27.6 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 223.0 | 1.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 19.6 | 19.6 | 11.4 | L1/NoC (far above floor) |
| Matmul |  | 1 | 187.9 | 1.6% | 96 | 1024x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 38.8 | 64.3 | 64.3 | 2.9 | DRAM |
| NlpCreateHeads |  | 20 | 167.1 | 1.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 16.2 | 16.2 | 10.3 | L1/NoC (far above floor) |
| Slice |  | 24 | 132.1 | 1.2% | 120 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i | 0.0 | 4.9 | 4.9 | 27.2 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 129.4 | 1.1% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 4.1 | 4.1 | 31.8 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 116.1 | 1.0% | 64 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 4.0 | 4.0 | 28.7 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 106.8 | 0.9% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 17.2 | 17.2 | 6.2 | DRAM (far above floor) |
| Matmul |  | 4 | 90.1 | 0.8% | 96 | 1024x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 38.8 | 15.9 | 38.8 | 2.3 | compute |
| LayerNorm | RMSNORM | 4 | 74.7 | 0.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 3.2 | 3.2 | 23.0 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 4 | 47.9 | 0.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 4.9 | 4.9 | 9.9 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 30.5 | 0.3% | 32 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 0.8 | 0.8 | 37.7 | L1/NoC (far above floor) |
| BinaryNg | ADD | 1 | 17.6 | 0.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1.4 | 1.4 | 12.5 | DRAM (far above floor) |
| Matmul |  | 1 | 15.8 | 0.1% | 24 | 32x1024x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.3 | 3.4 | 3.4 | 4.7 | DRAM (far above floor) |
| Matmul |  | 1 | 14.0 | 0.1% | 96 | 1024x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 2.4 | 4.0 | 4.0 | 3.5 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 1 | 11.0 | 0.1% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 2.2 | 2.2 | 5.1 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 9.2 | 0.1% | 120 |  | BFLOAT4_B/BFLOAT4_B/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 3.5 | 3.5 | 2.7 | DRAM |
| Embeddings |  | 1 | 8.2 | 0.1% | 32 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 4.1 | 4.1 | 2.0 | DRAM |
| BinaryNg | DIV | 1 | 7.1 | 0.1% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 18.2 | DRAM (far above floor) |
| Reduce |  | 1 | 6.7 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.7 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 6.3 | 0.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 2.3 | 2.3 | 2.8 | L1/NoC |
| FillPad |  | 1 | 4.5 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 11.6 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.2 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 3.8 | DRAM (far above floor) |
| Unary | SQRT | 1 | 1.8 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 112.6 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 1 | 1.4 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 87.7 | DRAM (far above floor) |
| Clone |  | 1 | 1.3 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 3.4 | DRAM (far above floor) |
