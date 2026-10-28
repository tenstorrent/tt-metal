### b1_s2048: 646 ops, device kernel sum 19.827 ms, eager op-to-op gaps 0.457 ms, roofline floor 2.991 ms (clock 1350 MHz, 130 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 20 | 6085.4 | 30.7% | 130 | H4 S2048 keys1025 D256 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 478.5 | 104.1 | 478.5 | 12.7 | compute (far above floor) |
| SDPA |  | 4 | 1868.4 | 9.4% | 130 | H4 S2048 keys2048 D512 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 382.4 | 22.4 | 382.4 | 4.9 | compute (far above floor) |
| Matmul | GELU_TANH | 24 | 1588.2 | 8.0% | 130 | 2048x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | L1-i/DRAM-i/L1-i | 286.8 | 66.2 | 286.8 | 5.5 | compute (far above floor) |
| LayerNorm | RMSNORM | 144 | 1515.3 | 7.6% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 54.1 | 54.1 | 28.0 | L1/NoC (far above floor) |
| Matmul |  | 44 | 1218.1 | 6.1% | 130 | 2048x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | L1-i/DRAM-i/L1-i | 525.8 | 121.4 | 525.8 | 2.3 | compute |
| MinimalMatmul |  | 28 | 1055.4 | 5.3% | 130 | 2048x2048x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 334.6 | 77.3 | 334.6 | 3.2 | compute (far above floor) |
| RotaryEmbedding |  | 20 | 787.4 | 4.0% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 89.9 | 89.9 | 8.8 | L1/NoC (far above floor) |
| Matmul | GELU_TANH | 24 | 728.9 | 3.7% | 80 | 2048x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 71.7 | 22.0 | 71.7 | 10.2 | compute (far above floor) |
| MinimalMatmul |  | 24 | 574.9 | 2.9% | 130 | 2048x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 71.7 | 22.0 | 71.7 | 8.0 | compute (far above floor) |
| RotaryEmbedding |  | 20 | 553.6 | 2.8% | 128 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 85.9 | 85.9 | 6.4 | L1/NoC (far above floor) |
| MinimalMatmul |  | 20 | 505.2 | 2.5% | 130 | 2048x1024x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 119.5 | 29.5 | 119.5 | 4.2 | compute (far above floor) |
| BinaryNg | ADD | 71 | 433.3 | 2.2% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 39.8 | 39.8 | 10.9 | L1/NoC (far above floor) |
| BinaryNg | MUL | 24 | 349.2 | 1.8% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 28.6 | 28.6 | 12.2 | L1/NoC (far above floor) |
| MinimalMatmul |  | 1 | 276.6 | 1.4% | 130 | 2048x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 71.7 | 115.5 | 115.5 | 2.4 | DRAM |
| Slice |  | 24 | 271.8 | 1.4% | 130 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i | 0.0 | 9.0 | 9.0 | 30.3 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 256.4 | 1.3% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 35.9 | 35.9 | 7.1 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 239.8 | 1.2% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 8.0 | 8.0 | 30.1 | L1/NoC (far above floor) |
| BinaryNg | ADD | 1 | 195.7 | 1.0% | 130 |  | BFLOAT4_B/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 9.5 | 9.5 | 20.7 | DRAM (far above floor) |
| Matmul |  | 4 | 184.9 | 0.9% | 120 | 2048x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | L1-i/DRAM-i/DRAM-i | 71.7 | 66.0 | 71.7 | 2.6 | compute |
| BinaryNg | MUL | 24 | 178.3 | 0.9% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 13.4 | 13.4 | 13.3 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 20 | 156.7 | 0.8% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 15.9 | 15.9 | 9.9 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 147.9 | 0.7% | 128 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 4.0 | 4.0 | 37.1 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 127.4 | 0.6% | 64 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 33.6 | 33.6 | 3.8 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 20 | 111.8 | 0.6% | 128 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | L1-i/L1-i | 0.0 | 4.0 | 4.0 | 28.2 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 74.6 | 0.4% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 3.2 | 3.2 | 23.4 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 4 | 73.3 | 0.4% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | DRAM-i/L1-i/L1-i/L1-i | 0.0 | 54.6 | 54.6 | 1.3 | DRAM |
| LayerNorm | RMSNORM | 4 | 40.4 | 0.2% | 64 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 0.8 | 0.8 | 50.4 | L1/NoC (far above floor) |
| BinaryNg | ADD | 1 | 36.3 | 0.2% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 5.1 | 5.1 | 7.1 | DRAM (far above floor) |
| Matmul |  | 1 | 29.9 | 0.2% | 26 | 32x2048x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.6 | 6.6 | 6.6 | 4.5 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 4 | 29.7 | 0.1% | 64 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | L1-i/L1-i | 0.0 | 0.8 | 0.8 | 37.5 | L1/NoC (far above floor) |
| MinimalMatmul |  | 1 | 27.2 | 0.1% | 130 | 2048x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 4.5 | 7.1 | 7.1 | 3.8 | DRAM (far above floor) |
| Reduce |  | 1 | 15.5 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.5 | 0.5 | 29.8 | DRAM (far above floor) |
| BinaryNg | DIV | 2 | 14.2 | 0.1% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 18.2 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 1 | 13.4 | 0.1% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 4.3 | 4.3 | 3.1 | DRAM (far above floor) |
| Embeddings |  | 1 | 13.4 | 0.1% | 64 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 8.2 | 8.2 | 1.6 | DRAM |
| BinaryNg | ADD | 1 | 11.7 | 0.1% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 4.5 | 4.5 | 2.6 | L1/NoC |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 34.2 | DRAM (far above floor) |
| FillPad |  | 1 | 5.2 | 0.0% | 64 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 1.0 | 1.0 | 5.1 | DRAM (far above floor) |
| FillPad |  | 1 | 4.9 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 12.8 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 2 | 4.5 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 141.8 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.8 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.5 | 0.5 | 5.4 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.7 | 0.0% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 4.7 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.7 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 166.8 | DRAM (far above floor) |
| BinaryNg | SUB | 1 | 2.2 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.5 | 0.5 | 4.3 | DRAM (far above floor) |
| Clone |  | 1 | 1.9 | 0.0% | 64 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 1.0 | 1.0 | 1.9 | DRAM |
| Typecast |  | 1 | 1.9 | 0.0% | 64 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 2.4 | DRAM |
| Clone |  | 1 | 1.6 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.2 | DRAM (far above floor) |
