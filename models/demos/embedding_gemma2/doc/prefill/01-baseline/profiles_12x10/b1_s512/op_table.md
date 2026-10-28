### b1_s512: 645 ops, device kernel sum 6.968 ms, eager op-to-op gaps 10.684 ms, roofline floor 0.632 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LayerNorm | RMSNORM | 148 | 1420.8 | 20.4% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 15.3 | 15.3 | 93.1 | L1/NoC (far above floor) |
| SDPA |  | 20 | 1075.2 | 15.4% | 64 | H4 S512 keys512 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 64.7 | 11.8 | 64.7 | 16.6 | compute (far above floor) |
| Matmul |  | 68 | 771.3 | 11.1% | 88 | 512x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 220.1 | 165.2 | 220.1 | 3.5 | compute (far above floor) |
| BinaryNg | MUL | 48 | 719.8 | 10.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 29.1 | 29.1 | 24.7 | L1/NoC (far above floor) |
| Matmul |  | 28 | 421.2 | 6.0% | 64 | 512x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 90.6 | 68.0 | 90.6 | 4.6 | compute (far above floor) |
| SDPA |  | 4 | 410.4 | 5.9% | 64 | H4 S512 keys512 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 25.9 | 3.2 | 25.9 | 15.9 | compute (far above floor) |
| Matmul |  | 48 | 281.2 | 4.0% | 64 | 512x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 38.8 | 31.0 | 38.8 | 7.2 | compute (far above floor) |
| RotaryEmbedding |  | 20 | 232.2 | 3.3% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 5.1 | 5.1 | 45.9 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 20 | 227.9 | 3.3% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 3.0 | 3.0 | 75.1 | L1/NoC (far above floor) |
| BinaryNg | ADD | 71 | 186.7 | 2.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 10.8 | 10.8 | 17.3 | L1/NoC (far above floor) |
| Matmul |  | 20 | 176.6 | 2.5% | 64 | 512x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 32.4 | 24.8 | 32.4 | 5.5 | compute (far above floor) |
| LayerNorm | RMSNORM | 20 | 129.5 | 1.9% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 4.1 | 4.1 | 31.9 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 20 | 121.3 | 1.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 8.1 | 8.1 | 15.0 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 115.7 | 1.7% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 2.0 | 2.0 | 56.6 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 20 | 99.5 | 1.4% | 32 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 2.0 | 2.0 | 49.2 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 86.3 | 1.2% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 2.0 | 2.0 | 42.7 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 85.4 | 1.2% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i/L1-i | 0.0 | 0.8 | 0.8 | 105.6 | L1/NoC (far above floor) |
| Matmul |  | 1 | 79.4 | 1.1% | 96 | 512x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 19.4 | 38.7 | 38.7 | 2.1 | DRAM |
| Slice |  | 24 | 71.0 | 1.0% | 120 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i | 0.0 | 2.4 | 2.4 | 29.2 | L1/NoC (far above floor) |
| Matmul |  | 4 | 57.8 | 0.8% | 96 | 512x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 19.4 | 14.5 | 19.4 | 3.0 | compute |
| LayerNorm | RMSNORM | 4 | 41.7 | 0.6% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 1.6 | 1.6 | 25.7 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 4 | 28.3 | 0.4% | 96 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 2.4 | 2.4 | 11.7 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 27.7 | 0.4% | 16 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 0.4 | 0.4 | 68.5 | L1/NoC (far above floor) |
| BinaryNg | DIV | 2 | 14.8 | 0.2% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 18.9 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 11.0 | 0.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 26.5 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 1 | 9.9 | 0.1% | 16 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 1.1 | 1.1 | 9.2 | DRAM (far above floor) |
| Matmul |  | 1 | 8.8 | 0.1% | 24 | 32x512x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.2 | 1.8 | 1.8 | 4.9 | DRAM (far above floor) |
| Matmul |  | 1 | 8.6 | 0.1% | 96 | 512x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 1.2 | 2.4 | 2.4 | 3.6 | DRAM (far above floor) |
| Embeddings |  | 1 | 7.9 | 0.1% | 16 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2.1 | 2.1 | 3.9 | DRAM (far above floor) |
| Reduce |  | 1 | 6.8 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 34.0 | DRAM (far above floor) |
| Reduce |  | 1 | 4.9 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 36.4 | DRAM (far above floor) |
| FillPad |  | 1 | 4.6 | 0.1% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 11.9 | DRAM (far above floor) |
| FillPad |  | 1 | 4.4 | 0.1% | 16 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.3 | 0.3 | 17.2 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 3.4 | 0.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 1.1 | 1.1 | 3.0 | L1/NoC (far above floor) |
| Unary | CLAMP_TSS | 2 | 2.8 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 87.7 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.5 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 19.8 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.2 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 3.9 | DRAM (far above floor) |
| BinaryNg | SUB | 1 | 2.0 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.1 | 0.1 | 16.0 | DRAM (far above floor) |
| Unary | SQRT | 1 | 1.8 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 110.6 | DRAM (far above floor) |
| Clone |  | 1 | 1.3 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 3.4 | DRAM (far above floor) |
| Clone |  | 1 | 1.2 | 0.0% | 16 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.3 | 0.3 | 4.9 | DRAM (far above floor) |
| Typecast |  | 1 | 1.2 | 0.0% | 16 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 6.2 | DRAM (far above floor) |
