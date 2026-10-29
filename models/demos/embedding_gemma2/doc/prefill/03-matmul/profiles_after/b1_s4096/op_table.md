### b1_s4096: 638 ops, device kernel sum 41.337 ms, eager op-to-op gaps 0.360 ms, roofline floor 15.425 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 20 | 6438.8 | 15.6% | 64 | H4 S4096 keys1025 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 1036.6 | 1351.7 | 1351.7 | 4.8 | DRAM (far above floor) |
| Matmul |  | 68 | 5453.5 | 13.2% | 110 | 4096x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 1760.6 | 2403.7 | 2403.7 | 2.3 | DRAM |
| SDPA |  | 4 | 5391.5 | 13.0% | 120 | H4 S4096 keys4096 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 1657.0 | 401.4 | 1657.0 | 3.3 | compute (far above floor) |
| BinaryNg | MUL | 24 | 3804.0 | 9.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2359.3 | 2359.3 | 1.6 | DRAM |
| LayerNorm | RMSNORM | 144 | 2666.9 | 6.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 116.8 | 116.8 | 22.8 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 20 | 2572.1 | 6.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 819.2 | 819.2 | 3.1 | DRAM (far above floor) |
| MinimalMatmul |  | 28 | 2093.9 | 5.1% | 120 | 4096x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/L1-i | 724.9 | 989.8 | 989.8 | 2.1 | DRAM |
| RotaryEmbedding |  | 20 | 1594.6 | 3.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 491.5 | 491.5 | 3.2 | DRAM (far above floor) |
| NlpCreateHeads |  | 20 | 1533.1 | 3.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1310.7 | 1310.7 | 1.2 | DRAM |
| Matmul | GELU_TANH | 24 | 1261.7 | 3.1% | 80 | 4096x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 155.3 | 32.5 | 155.3 | 8.1 | compute (far above floor) |
| RotaryEmbedding |  | 4 | 949.8 | 2.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 327.7 | 327.7 | 2.9 | DRAM |
| LayerNorm | RMSNORM | 20 | 934.4 | 2.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 655.4 | 655.4 | 1.4 | DRAM |
| MinimalMatmul |  | 20 | 919.5 | 2.2% | 120 | 4096x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/L1-i | 258.9 | 357.5 | 357.5 | 2.6 | DRAM |
| BinaryNg | ADD | 71 | 760.8 | 1.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 86.2 | 86.2 | 8.8 | L1/NoC (far above floor) |
| Matmul |  | 24 | 582.9 | 1.4% | 80 | 4096x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 155.3 | 32.5 | 155.3 | 3.8 | compute (far above floor) |
| MinimalMatmul |  | 1 | 538.3 | 1.3% | 120 | 4096x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 155.3 | 217.9 | 217.9 | 2.5 | DRAM |
| LayerNorm | RMSNORM | 20 | 532.0 | 1.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 327.7 | 327.7 | 1.6 | DRAM |
| LayerNorm | RMSNORM | 20 | 465.1 | 1.1% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 327.7 | 327.7 | 1.4 | DRAM |
| NlpCreateHeads |  | 4 | 454.3 | 1.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 393.2 | 393.2 | 1.2 | DRAM |
| Matmul |  | 4 | 449.9 | 1.1% | 120 | 4096x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 155.3 | 211.3 | 211.3 | 2.1 | DRAM |
| RotaryEmbedding |  | 4 | 371.1 | 0.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 131.1 | 131.1 | 2.8 | DRAM |
| LayerNorm | RMSNORM | 4 | 362.1 | 0.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 262.2 | 262.2 | 1.4 | DRAM |
| BinaryNg | MUL | 24 | 313.6 | 0.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 29.1 | 29.1 | 10.8 | L1/NoC (far above floor) |
| Slice |  | 24 | 277.4 | 0.7% | 120 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i | 0.0 | 19.4 | 19.4 | 14.3 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 113.9 | 0.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 65.5 | 65.5 | 1.7 | DRAM |
| BinaryNg | ADD | 1 | 108.8 | 0.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 19.5 | 19.5 | 5.6 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 4 | 99.0 | 0.2% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 65.5 | 65.5 | 1.5 | DRAM |
| BinaryNg | ADD | 1 | 88.7 | 0.2% | 120 |  | BFLOAT4_B/BFLOAT4_B/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 55.3 | 55.3 | 1.6 | DRAM |
| Matmul |  | 1 | 58.0 | 0.1% | 24 | 32x4096x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 1.2 | 13.0 | 13.0 | 4.5 | DRAM (far above floor) |
| MinimalMatmul |  | 1 | 52.3 | 0.1% | 120 | 4096x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 9.7 | 13.5 | 13.5 | 3.9 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 1 | 25.7 | 0.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 8.6 | 8.6 | 3.0 | DRAM |
| Embeddings |  | 1 | 25.4 | 0.1% | 120 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 16.4 | 16.4 | 1.5 | DRAM |
| BinaryNg | ADD | 1 | 14.9 | 0.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 9.0 | 9.0 | 1.7 | L1/NoC |
| BinaryNg | DIV | 1 | 7.3 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 18.6 | DRAM (far above floor) |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 34.0 | DRAM (far above floor) |
| FillPad |  | 1 | 5.0 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 13.0 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.9 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 5.1 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.7 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 167.3 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 1 | 2.3 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 142.6 | DRAM (far above floor) |
| Clone |  | 1 | 1.7 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.4 | DRAM (far above floor) |
