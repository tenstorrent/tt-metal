### b1_s512: 645 ops, device kernel sum 6.519 ms, eager op-to-op gaps 9.508 ms, roofline floor 0.558 ms (clock 1350 MHz, 130 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LayerNorm | RMSNORM | 144 | 1382.7 | 21.2% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 13.7 | 13.7 | 100.7 | L1/NoC (far above floor) |
| SDPA |  | 20 | 1144.0 | 17.5% | 64 | H4 S512 keys512 D256 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 59.7 | 8.7 | 59.7 | 19.1 | compute (far above floor) |
| Matmul |  | 68 | 667.0 | 10.2% | 104 | 512x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | L1-i/DRAM-i/L1-i | 203.1 | 157.9 | 203.1 | 3.3 | compute (far above floor) |
| SDPA |  | 4 | 395.0 | 6.1% | 64 | H4 S512 keys512 D512 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 23.9 | 2.1 | 23.9 | 16.5 | compute (far above floor) |
| Matmul |  | 28 | 384.3 | 5.9% | 64 | 512x2048x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 83.6 | 65.0 | 83.6 | 4.6 | compute (far above floor) |
| BinaryNg | MUL | 24 | 368.3 | 5.6% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 7.1 | 7.1 | 51.6 | L1/NoC (far above floor) |
| Matmul |  | 48 | 283.0 | 4.3% | 64 | 512x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 35.8 | 30.6 | 35.8 | 7.9 | compute (far above floor) |
| RotaryEmbedding |  | 20 | 222.0 | 3.4% | 32 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 1.9 | 1.9 | 115.3 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 20 | 218.7 | 3.4% | 64 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 2.9 | 2.9 | 75.0 | L1/NoC (far above floor) |
| BinaryNg | ADD | 71 | 170.5 | 2.6% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 9.9 | 9.9 | 17.1 | L1/NoC (far above floor) |
| Matmul |  | 20 | 163.9 | 2.5% | 64 | 512x1024x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 29.9 | 23.7 | 29.9 | 5.5 | compute (far above floor) |
| BinaryNg | MUL | 24 | 150.9 | 2.3% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 3.4 | 3.4 | 44.9 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 125.1 | 1.9% | 64 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 2.0 | 2.0 | 62.4 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 114.6 | 1.8% | 32 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 1.0 | 1.0 | 113.3 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 92.9 | 1.4% | 32 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | L1-i/L1-i | 0.0 | 1.0 | 1.0 | 93.6 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 84.1 | 1.3% | 16 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.6 | 0.6 | 147.1 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 83.2 | 1.3% | 64 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 1.2 | 1.2 | 71.3 | L1/NoC (far above floor) |
| Matmul |  | 1 | 75.7 | 1.2% | 104 | 512x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 17.9 | 38.7 | 38.7 | 2.0 | DRAM |
| Slice |  | 24 | 63.6 | 1.0% | 130 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i | 0.0 | 2.2 | 2.2 | 28.4 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 20 | 56.9 | 0.9% | 128 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 4.0 | 4.0 | 14.3 | L1/NoC (far above floor) |
| Matmul |  | 4 | 55.8 | 0.9% | 96 | 512x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | L1-i/DRAM-i/L1-i | 17.9 | 13.8 | 17.9 | 3.1 | compute (far above floor) |
| LayerNorm | RMSNORM | 4 | 40.5 | 0.6% | 64 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 0.8 | 0.8 | 50.5 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 38.4 | 0.6% | 16 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 0.2 | 0.2 | 186.3 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 26.3 | 0.4% | 16 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | L1-i/L1-i | 0.0 | 0.2 | 0.2 | 132.6 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 4 | 15.5 | 0.2% | 96 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 1.2 | 1.2 | 13.1 | L1/NoC (far above floor) |
| BinaryNg | DIV | 2 | 14.6 | 0.2% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 18.6 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 1 | 9.8 | 0.2% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 1.1 | 1.1 | 9.1 | DRAM (far above floor) |
| Matmul |  | 1 | 8.9 | 0.1% | 26 | 32x512x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.1 | 1.8 | 1.8 | 4.9 | DRAM (far above floor) |
| Matmul |  | 1 | 8.5 | 0.1% | 96 | 512x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 1.1 | 2.4 | 2.4 | 3.6 | DRAM (far above floor) |
| Embeddings |  | 1 | 7.9 | 0.1% | 16 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2.1 | 2.1 | 3.9 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 7.3 | 0.1% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 17.5 | DRAM (far above floor) |
| Reduce |  | 1 | 6.7 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.6 | DRAM (far above floor) |
| Reduce |  | 1 | 5.0 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 37.0 | DRAM (far above floor) |
| FillPad |  | 1 | 4.4 | 0.1% | 16 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.3 | 0.3 | 17.4 | DRAM (far above floor) |
| FillPad |  | 1 | 4.4 | 0.1% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 11.6 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 3.4 | 0.1% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 1.1 | 1.1 | 3.0 | L1/NoC (far above floor) |
| Unary | CLAMP_TSS | 2 | 2.8 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 88.3 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.5 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 19.6 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.1 | 0.0% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 3.7 | DRAM (far above floor) |
| BinaryNg | SUB | 1 | 2.0 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 16.0 | DRAM (far above floor) |
| Unary | SQRT | 1 | 1.8 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 111.3 | DRAM (far above floor) |
| Clone |  | 1 | 1.4 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 3.5 | DRAM (far above floor) |
| Typecast |  | 1 | 1.3 | 0.0% | 16 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 6.7 | DRAM (far above floor) |
| Clone |  | 1 | 1.2 | 0.0% | 16 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.3 | 0.3 | 4.6 | DRAM (far above floor) |
