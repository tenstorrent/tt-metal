### b1_s1024: 646 ops, device kernel sum 11.554 ms, eager op-to-op gaps 6.165 ms, roofline floor 1.541 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 20 | 2103.3 | 18.2% | 64 | H4 S1024 keys1024 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 258.9 | 35.2 | 258.9 | 8.1 | compute (far above floor) |
| LayerNorm | RMSNORM | 148 | 1477.6 | 12.8% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 30.2 | 30.2 | 48.9 | L1/NoC (far above floor) |
| BinaryNg | MUL | 48 | 1321.2 | 11.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 58.3 | 58.3 | 22.7 | L1/NoC (far above floor) |
| Matmul |  | 68 | 1208.4 | 10.5% | 88 | 1024x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 440.1 | 182.4 | 440.1 | 2.7 | compute |
| SDPA |  | 4 | 943.3 | 8.2% | 64 | H4 S1024 keys1024 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 103.6 | 8.7 | 103.6 | 9.1 | compute (far above floor) |
| Matmul |  | 28 | 690.0 | 6.0% | 64 | 1024x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 181.2 | 75.1 | 181.2 | 3.8 | compute (far above floor) |
| RotaryEmbedding |  | 20 | 650.8 | 5.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 49.1 | 49.1 | 13.3 | L1/NoC (far above floor) |
| Matmul |  | 48 | 444.5 | 3.8% | 64 | 1024x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 77.7 | 35.8 | 77.7 | 5.7 | compute (far above floor) |
| RotaryEmbedding |  | 20 | 418.7 | 3.6% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 45.0 | 45.0 | 9.3 | L1/NoC (far above floor) |
| BinaryNg | ADD | 71 | 290.4 | 2.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 21.5 | 21.5 | 13.5 | L1/NoC (far above floor) |
| Matmul |  | 20 | 285.9 | 2.5% | 64 | 1024x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 64.7 | 27.8 | 64.7 | 4.4 | compute (far above floor) |
| LayerNorm | RMSNORM | 20 | 226.1 | 2.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 8.1 | 8.1 | 27.9 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 223.6 | 1.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 19.6 | 19.6 | 11.4 | L1/NoC (far above floor) |
| Matmul |  | 1 | 189.5 | 1.6% | 96 | 1024x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 38.8 | 64.3 | 64.3 | 2.9 | DRAM |
| NlpCreateHeads |  | 20 | 167.5 | 1.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 16.2 | 16.2 | 10.4 | L1/NoC (far above floor) |
| Slice |  | 24 | 133.2 | 1.2% | 120 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i | 0.0 | 4.9 | 4.9 | 27.4 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 129.0 | 1.1% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 4.1 | 4.1 | 31.7 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 115.8 | 1.0% | 64 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 4.0 | 4.0 | 28.6 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 107.1 | 0.9% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 17.2 | 17.2 | 6.2 | DRAM (far above floor) |
| Matmul |  | 4 | 89.9 | 0.8% | 96 | 1024x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 38.8 | 15.9 | 38.8 | 2.3 | compute |
| LayerNorm | RMSNORM | 4 | 72.9 | 0.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 3.2 | 3.2 | 22.5 | L1/NoC (far above floor) |
| BinaryNg | ADD | 1 | 57.7 | 0.5% | 120 |  | BFLOAT4_B/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2.4 | 2.4 | 23.7 | DRAM (far above floor) |
| NlpCreateHeads |  | 4 | 48.3 | 0.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 4.9 | 4.9 | 9.9 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 30.4 | 0.3% | 32 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 0.8 | 0.8 | 37.6 | L1/NoC (far above floor) |
| BinaryNg | ADD | 1 | 18.0 | 0.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1.4 | 1.4 | 12.8 | DRAM (far above floor) |
| Matmul |  | 1 | 15.9 | 0.1% | 24 | 32x1024x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.3 | 3.4 | 3.4 | 4.7 | DRAM (far above floor) |
| BinaryNg | DIV | 2 | 14.4 | 0.1% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 18.4 | DRAM (far above floor) |
| Matmul |  | 1 | 14.2 | 0.1% | 96 | 1024x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 2.4 | 4.0 | 4.0 | 3.6 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 1 | 10.9 | 0.1% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 2.2 | 2.2 | 5.1 | DRAM (far above floor) |
| Embeddings |  | 1 | 8.8 | 0.1% | 32 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 4.1 | 4.1 | 2.1 | DRAM |
| Reduce |  | 1 | 8.5 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.3 | 0.3 | 32.1 | DRAM (far above floor) |
| Reduce |  | 1 | 6.7 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.6 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 6.1 | 0.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 2.3 | 2.3 | 2.7 | L1/NoC |
| FillPad |  | 1 | 4.7 | 0.0% | 32 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.5 | 0.5 | 9.1 | DRAM (far above floor) |
| FillPad |  | 1 | 4.6 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 11.9 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 2 | 2.8 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 87.2 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.6 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.3 | 0.3 | 10.2 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.2 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 3.8 | DRAM (far above floor) |
| BinaryNg | SUB | 1 | 2.1 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.3 | 0.3 | 8.1 | DRAM (far above floor) |
| Unary | SQRT | 1 | 1.8 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 111.6 | DRAM (far above floor) |
| Typecast |  | 1 | 1.8 | 0.0% | 32 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.6 | DRAM (far above floor) |
| Clone |  | 1 | 1.5 | 0.0% | 32 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.5 | 0.5 | 2.9 | DRAM |
| Clone |  | 1 | 1.3 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 3.4 | DRAM (far above floor) |
