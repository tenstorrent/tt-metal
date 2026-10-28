### b1_s128: 645 ops, device kernel sum 4.573 ms, eager op-to-op gaps 12.456 ms, roofline floor 0.331 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LayerNorm | RMSNORM | 148 | 1351.3 | 29.6% | 4 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 4.0 | 4.0 | 334.6 | L1/NoC (far above floor) |
| Matmul |  | 68 | 610.3 | 13.3% | 44 | 128x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 55.0 | 152.3 | 152.3 | 4.0 | DRAM (far above floor) |
| Matmul |  | 28 | 343.4 | 7.5% | 32 | 128x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 22.7 | 62.7 | 62.7 | 5.5 | DRAM (far above floor) |
| BinaryNg | MUL | 48 | 287.0 | 6.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 7.3 | 7.3 | 39.4 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 20 | 222.8 | 4.9% | 8 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.8 | 0.8 | 293.8 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 20 | 212.9 | 4.7% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 1.3 | 1.3 | 168.4 | L1/NoC (far above floor) |
| Matmul |  | 48 | 212.4 | 4.6% | 32 | 128x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 9.7 | 27.3 | 27.3 | 7.8 | DRAM (far above floor) |
| SDPA |  | 20 | 181.7 | 4.0% | 64 | H4 S128 keys128 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 4.0 | 1.9 | 4.0 | 44.9 | compute (far above floor) |
| Matmul |  | 20 | 140.0 | 3.1% | 32 | 128x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 8.1 | 22.5 | 22.5 | 6.2 | DRAM (far above floor) |
| BinaryNg | ADD | 71 | 111.3 | 2.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 2.7 | 2.7 | 41.3 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 109.0 | 2.4% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 1.0 | 1.0 | 105.7 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 108.1 | 2.4% | 8 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 0.5 | 0.5 | 205.6 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 91.2 | 2.0% | 8 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 0.5 | 0.5 | 180.4 | L1/NoC (far above floor) |
| SDPA |  | 4 | 82.5 | 1.8% | 64 | H4 S128 keys128 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 1.6 | 0.6 | 1.6 | 51.0 | compute (far above floor) |
| RotaryEmbedding |  | 4 | 81.7 | 1.8% | 4 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.2 | 0.2 | 403.8 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 79.6 | 1.7% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.5 | 0.5 | 157.5 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 20 | 67.5 | 1.5% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 2.0 | 2.0 | 33.4 | L1/NoC (far above floor) |
| Matmul |  | 4 | 46.3 | 1.0% | 48 | 128x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 4.9 | 13.4 | 13.4 | 3.5 | DRAM (far above floor) |
| Matmul |  | 1 | 44.6 | 1.0% | 48 | 128x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 4.9 | 19.5 | 19.5 | 2.3 | DRAM |
| LayerNorm | RMSNORM | 4 | 37.1 | 0.8% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 0.4 | 0.4 | 89.9 | L1/NoC (far above floor) |
| Slice |  | 24 | 28.4 | 0.6% | 64 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i | 0.0 | 0.6 | 0.6 | 46.7 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 25.9 | 0.6% | 4 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 0.1 | 0.1 | 256.2 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 4 | 20.7 | 0.5% | 24 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.6 | 0.6 | 34.1 | L1/NoC (far above floor) |
| BinaryNg | DIV | 2 | 14.3 | 0.3% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 18.3 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 1 | 9.5 | 0.2% | 4 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 0.3 | 0.3 | 35.1 | DRAM (far above floor) |
| Reduce |  | 1 | 6.7 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.7 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 5.6 | 0.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 112.0 | DRAM (far above floor) |
| Matmul |  | 1 | 5.2 | 0.1% | 48 | 128x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 0.3 | 1.2 | 1.2 | 4.3 | DRAM (far above floor) |
| Embeddings |  | 1 | 4.6 | 0.1% | 4 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.5 | 0.5 | 9.0 | DRAM (far above floor) |
| FillPad |  | 1 | 4.6 | 0.1% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 11.9 | DRAM (far above floor) |
| FillPad |  | 1 | 4.2 | 0.1% | 4 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 65.3 | DRAM (far above floor) |
| Matmul |  | 1 | 3.7 | 0.1% | 24 | 32x128x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 6.2 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 2 | 2.8 | 0.1% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 88.0 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.5 | 0.1% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 78.4 | DRAM (far above floor) |
| Reduce |  | 1 | 2.3 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 58.6 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.2 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 3.8 | DRAM (far above floor) |
| BinaryNg | SUB | 1 | 2.0 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 61.0 | DRAM (far above floor) |
| Unary | SQRT | 1 | 1.8 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 111.5 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 1.7 | 0.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 0.3 | 0.3 | 5.9 | L1/NoC (far above floor) |
| Clone |  | 1 | 1.4 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 3.6 | DRAM (far above floor) |
| Typecast |  | 1 | 1.0 | 0.0% | 4 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 21.0 | DRAM (far above floor) |
| Clone |  | 1 | 0.9 | 0.0% | 4 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 14.7 | DRAM (far above floor) |
