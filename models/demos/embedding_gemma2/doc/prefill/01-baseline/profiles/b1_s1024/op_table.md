### b1_s1024: 646 ops, device kernel sum 10.536 ms, eager op-to-op gaps 5.530 ms, roofline floor 1.380 ms (clock 1350 MHz, 130 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 20 | 2221.3 | 21.1% | 130 | H4 S1024 keys1024 D256 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 239.0 | 29.0 | 239.0 | 9.3 | compute (far above floor) |
| LayerNorm | RMSNORM | 144 | 1441.1 | 13.7% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 27.2 | 27.2 | 53.0 | L1/NoC (far above floor) |
| Matmul |  | 68 | 1025.7 | 9.7% | 104 | 1024x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | L1-i/DRAM-i/L1-i | 406.3 | 167.8 | 406.3 | 2.5 | compute |
| SDPA |  | 4 | 887.1 | 8.4% | 130 | H4 S1024 keys1024 D512 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 95.6 | 6.6 | 95.6 | 9.3 | compute (far above floor) |
| BinaryNg | MUL | 24 | 711.3 | 6.8% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 14.3 | 14.3 | 49.8 | L1/NoC (far above floor) |
| Matmul |  | 28 | 615.3 | 5.8% | 64 | 1024x2048x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 167.3 | 69.1 | 167.3 | 3.7 | compute (far above floor) |
| RotaryEmbedding |  | 20 | 558.2 | 5.3% | 128 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 44.9 | 44.9 | 12.4 | L1/NoC (far above floor) |
| Matmul |  | 48 | 446.4 | 4.2% | 64 | 1024x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 71.7 | 35.1 | 71.7 | 6.2 | compute (far above floor) |
| RotaryEmbedding |  | 20 | 399.0 | 3.8% | 64 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 42.9 | 42.9 | 9.3 | L1/NoC (far above floor) |
| BinaryNg | ADD | 71 | 265.7 | 2.5% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 19.9 | 19.9 | 13.4 | L1/NoC (far above floor) |
| BinaryNg | MUL | 24 | 262.5 | 2.5% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 6.7 | 6.7 | 39.0 | L1/NoC (far above floor) |
| Matmul |  | 20 | 255.9 | 2.4% | 64 | 1024x1024x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 59.7 | 25.6 | 59.7 | 4.3 | compute (far above floor) |
| RotaryEmbedding |  | 4 | 161.8 | 1.5% | 128 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 18.0 | 18.0 | 9.0 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 155.7 | 1.5% | 128 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 4.0 | 4.0 | 39.1 | L1/NoC (far above floor) |
| Matmul |  | 1 | 152.2 | 1.4% | 104 | 1024x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 35.8 | 64.3 | 64.3 | 2.4 | DRAM |
| Slice |  | 24 | 147.8 | 1.4% | 130 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i | 0.0 | 4.5 | 4.5 | 33.0 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 125.2 | 1.2% | 64 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 2.0 | 2.0 | 62.5 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 104.1 | 1.0% | 32 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 16.8 | 16.8 | 6.2 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 20 | 96.6 | 0.9% | 64 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | L1-i/L1-i | 0.0 | 2.0 | 2.0 | 48.7 | L1/NoC (far above floor) |
| Matmul |  | 4 | 88.3 | 0.8% | 96 | 1024x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | L1-i/DRAM-i/L1-i | 35.8 | 14.6 | 35.8 | 2.5 | compute |
| NlpCreateHeads |  | 20 | 86.2 | 0.8% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 7.9 | 7.9 | 10.9 | L1/NoC (far above floor) |
| BinaryNg | ADD | 1 | 53.8 | 0.5% | 130 |  | BFLOAT4_B/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2.4 | 2.4 | 22.1 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 4 | 47.4 | 0.5% | 128 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 1.6 | 1.6 | 29.7 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 40.0 | 0.4% | 32 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 0.4 | 0.4 | 98.8 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 4 | 29.6 | 0.3% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 2.4 | 2.4 | 12.4 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 27.7 | 0.3% | 32 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | L1-i/L1-i | 0.0 | 0.4 | 0.4 | 69.7 | L1/NoC (far above floor) |
| BinaryNg | ADD | 1 | 18.1 | 0.2% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1.4 | 1.4 | 12.8 | DRAM (far above floor) |
| Matmul |  | 1 | 16.0 | 0.2% | 26 | 32x1024x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.3 | 3.4 | 3.4 | 4.7 | DRAM (far above floor) |
| BinaryNg | DIV | 2 | 14.3 | 0.1% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 18.2 | DRAM (far above floor) |
| Matmul |  | 1 | 14.1 | 0.1% | 96 | 1024x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 2.2 | 4.0 | 4.0 | 3.5 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 1 | 10.9 | 0.1% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 2.1 | 2.1 | 5.1 | DRAM (far above floor) |
| Embeddings |  | 1 | 8.8 | 0.1% | 32 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 4.1 | 4.1 | 2.1 | DRAM |
| Reduce |  | 1 | 8.5 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.3 | 0.3 | 32.1 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 7.2 | 0.1% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 2.2 | 2.2 | 3.2 | L1/NoC (far above floor) |
| Reduce |  | 1 | 6.8 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.9 | DRAM (far above floor) |
| FillPad |  | 1 | 4.6 | 0.0% | 32 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.5 | 0.5 | 9.0 | DRAM (far above floor) |
| FillPad |  | 1 | 4.5 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 11.6 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 2 | 2.8 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 87.0 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.6 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.3 | 0.3 | 10.2 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.1 | 0.0% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 3.6 | DRAM (far above floor) |
| BinaryNg | SUB | 1 | 2.1 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.3 | 0.3 | 8.1 | DRAM (far above floor) |
| Unary | SQRT | 1 | 1.8 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 112.1 | DRAM (far above floor) |
| Typecast |  | 1 | 1.8 | 0.0% | 32 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.6 | DRAM (far above floor) |
| Clone |  | 1 | 1.4 | 0.0% | 32 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.5 | 0.5 | 2.7 | DRAM |
| Clone |  | 1 | 1.4 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 3.6 | DRAM (far above floor) |
