### b1_s512: 637 ops, device kernel sum 6.944 ms, eager op-to-op gaps 11.093 ms, roofline floor 0.631 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LayerNorm | RMSNORM | 148 | 1419.8 | 20.4% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 15.3 | 15.3 | 93.0 | L1/NoC (far above floor) |
| SDPA |  | 20 | 1073.6 | 15.5% | 64 | H4 S512 keys512 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 64.7 | 11.8 | 64.7 | 16.6 | compute (far above floor) |
| Matmul |  | 68 | 768.9 | 11.1% | 88 | 512x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 220.1 | 165.2 | 220.1 | 3.5 | compute (far above floor) |
| BinaryNg | MUL | 48 | 719.5 | 10.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 29.1 | 29.1 | 24.7 | L1/NoC (far above floor) |
| Matmul |  | 28 | 421.2 | 6.1% | 64 | 512x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 90.6 | 68.0 | 90.6 | 4.6 | compute (far above floor) |
| SDPA |  | 4 | 409.6 | 5.9% | 64 | H4 S512 keys512 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 25.9 | 3.2 | 25.9 | 15.8 | compute (far above floor) |
| Matmul |  | 48 | 281.0 | 4.0% | 64 | 512x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 38.8 | 31.0 | 38.8 | 7.2 | compute (far above floor) |
| RotaryEmbedding |  | 20 | 232.6 | 3.4% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 5.1 | 5.1 | 46.0 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 20 | 229.6 | 3.3% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 3.0 | 3.0 | 75.7 | L1/NoC (far above floor) |
| BinaryNg | ADD | 71 | 187.9 | 2.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 10.8 | 10.8 | 17.4 | L1/NoC (far above floor) |
| Matmul |  | 20 | 176.6 | 2.5% | 64 | 512x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 32.4 | 24.8 | 32.4 | 5.5 | compute (far above floor) |
| LayerNorm | RMSNORM | 20 | 129.3 | 1.9% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 4.1 | 4.1 | 31.8 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 20 | 123.7 | 1.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 8.1 | 8.1 | 15.3 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 117.2 | 1.7% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 2.0 | 2.0 | 57.4 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 99.4 | 1.4% | 32 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 2.0 | 2.0 | 49.1 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 86.5 | 1.2% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 2.0 | 2.0 | 42.7 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 85.7 | 1.2% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.8 | 0.8 | 105.9 | L1/NoC (far above floor) |
| Matmul |  | 1 | 79.7 | 1.1% | 96 | 512x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 19.4 | 38.7 | 38.7 | 2.1 | DRAM |
| Slice |  | 24 | 71.9 | 1.0% | 120 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i | 0.0 | 2.4 | 2.4 | 29.6 | L1/NoC (far above floor) |
| Matmul |  | 4 | 58.0 | 0.8% | 96 | 512x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 19.4 | 14.5 | 19.4 | 3.0 | compute |
| LayerNorm | RMSNORM | 4 | 41.5 | 0.6% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 1.6 | 1.6 | 25.5 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 28.1 | 0.4% | 16 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 0.4 | 0.4 | 69.6 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 4 | 28.0 | 0.4% | 96 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 2.4 | 2.4 | 11.5 | L1/NoC (far above floor) |
| BinaryNg | ADD | 1 | 11.4 | 0.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 27.5 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 1 | 9.8 | 0.1% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 1.1 | 1.1 | 9.1 | DRAM (far above floor) |
| Matmul |  | 1 | 8.9 | 0.1% | 24 | 32x512x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.2 | 1.8 | 1.8 | 4.9 | DRAM (far above floor) |
| Matmul |  | 1 | 8.6 | 0.1% | 96 | 512x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 1.2 | 2.4 | 2.4 | 3.6 | DRAM (far above floor) |
| Embeddings |  | 1 | 7.9 | 0.1% | 16 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2.1 | 2.1 | 3.9 | DRAM (far above floor) |
| BinaryNg | DIV | 1 | 7.0 | 0.1% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 17.9 | DRAM (far above floor) |
| Reduce |  | 1 | 6.7 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.6 | DRAM (far above floor) |
| FillPad |  | 1 | 4.6 | 0.1% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 11.9 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 3.3 | 0.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 1.1 | 1.1 | 2.9 | L1/NoC |
| BinaryNg | MUL | 1 | 2.2 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 3.9 | DRAM (far above floor) |
| Unary | SQRT | 1 | 1.8 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 112.2 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 1 | 1.4 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 88.2 | DRAM (far above floor) |
| Clone |  | 1 | 1.3 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 3.4 | DRAM (far above floor) |
