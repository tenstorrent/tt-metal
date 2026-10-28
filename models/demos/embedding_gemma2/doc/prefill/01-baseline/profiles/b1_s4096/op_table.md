### b1_s4096: 646 ops, device kernel sum 33.570 ms, eager op-to-op gaps 0.616 ms, roofline floor 8.752 ms (clock 1350 MHz, 130 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 20 | 6029.3 | 18.0% | 130 | H4 S4096 keys1025 D256 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 956.9 | 392.4 | 956.9 | 6.3 | compute (far above floor) |
| SDPA |  | 4 | 5591.2 | 16.7% | 130 | H4 S4096 keys4096 D512 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 1529.5 | 247.8 | 1529.5 | 3.7 | compute (far above floor) |
| Matmul | GELU_TANH | 24 | 2888.2 | 8.6% | 130 | 4096x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | L1-i/DRAM-i/DRAM-i | 573.6 | 479.0 | 573.6 | 5.0 | compute (far above floor) |
| Matmul |  | 44 | 2416.1 | 7.2% | 130 | 4096x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | L1-i/DRAM-i/DRAM-i | 1051.6 | 878.1 | 1051.6 | 2.3 | compute |
| LayerNorm | RMSNORM | 144 | 1841.2 | 5.5% | 128 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 107.8 | 107.8 | 17.1 | L1/NoC (far above floor) |
| MinimalMatmul |  | 28 | 1688.3 | 5.0% | 130 | 4096x2048x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/L1-i | 669.2 | 558.8 | 669.2 | 2.5 | compute |
| BinaryNg | MUL | 24 | 1614.5 | 4.8% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1253.4 | 1253.4 | 1.3 | DRAM |
| RotaryEmbedding |  | 20 | 1296.5 | 3.9% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 179.7 | 179.7 | 7.2 | L1/NoC (far above floor) |
| Matmul | GELU_TANH | 24 | 1289.2 | 3.8% | 80 | 4096x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 143.4 | 31.0 | 143.4 | 9.0 | compute (far above floor) |
| MinimalMatmul |  | 24 | 974.0 | 2.9% | 130 | 4096x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 143.4 | 31.0 | 143.4 | 6.8 | compute (far above floor) |
| MinimalMatmul |  | 20 | 858.0 | 2.6% | 130 | 4096x1024x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 239.0 | 37.2 | 239.0 | 3.6 | compute (far above floor) |
| RotaryEmbedding |  | 20 | 801.0 | 2.4% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/DRAM-i/L1-i | 0.0 | 171.8 | 171.8 | 4.7 | L1/NoC (far above floor) |
| BinaryNg | ADD | 1 | 756.4 | 2.3% | 130 |  | BFLOAT4_B/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 37.4 | 37.4 | 20.2 | DRAM (far above floor) |
| BinaryNg | ADD | 71 | 756.1 | 2.3% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 79.5 | 79.5 | 9.5 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 689.1 | 2.1% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 204.8 | 204.8 | 3.4 | DRAM (far above floor) |
| MinimalMatmul |  | 1 | 546.5 | 1.6% | 130 | 4096x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 143.4 | 217.9 | 217.9 | 2.5 | DRAM |
| NlpCreateHeads |  | 20 | 474.6 | 1.4% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | DRAM-i/L1-i/L1-i/L1-i | 0.0 | 364.0 | 364.0 | 1.3 | DRAM |
| LayerNorm | RMSNORM | 20 | 403.2 | 1.2% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 15.9 | 15.9 | 25.4 | L1/NoC (far above floor) |
| Slice |  | 24 | 377.4 | 1.1% | 130 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1-i | 0.0 | 17.9 | 17.9 | 21.1 | L1/NoC (far above floor) |
| Matmul |  | 4 | 363.3 | 1.1% | 120 | 4096x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | L1-i/DRAM-i/DRAM-i | 143.4 | 119.0 | 143.4 | 2.5 | compute |
| BinaryNg | MUL | 24 | 290.9 | 0.9% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 26.9 | 26.9 | 10.8 | L1/NoC (far above floor) |
| RotaryEmbedding |  | 4 | 261.9 | 0.8% | 128 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 100.4 | 100.4 | 2.6 | DRAM |
| NlpCreateHeads |  | 4 | 260.3 | 0.8% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 208.9 | 208.9 | 1.2 | DRAM |
| LayerNorm | RMSNORM | 20 | 232.6 | 0.7% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 8.0 | 8.0 | 29.2 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 4 | 204.8 | 0.6% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 139.3 | 139.3 | 1.5 | DRAM |
| LayerNorm | RMSNORM | 20 | 179.6 | 0.5% | 130 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | L1-i/L1-i | 0.0 | 7.9 | 7.9 | 22.6 | L1/NoC (far above floor) |
| BinaryNg | ADD | 1 | 105.9 | 0.3% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 19.5 | 19.5 | 5.4 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 4 | 68.8 | 0.2% | 128 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 34.8 | 34.8 | 2.0 | DRAM |
| Matmul |  | 1 | 58.1 | 0.2% | 26 | 32x4096x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 1.1 | 13.0 | 13.0 | 4.5 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 4 | 56.3 | 0.2% | 128 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i | 0.0 | 34.8 | 34.8 | 1.6 | DRAM |
| MinimalMatmul |  | 1 | 51.3 | 0.2% | 130 | 4096x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/DRAM-i | 9.0 | 13.5 | 13.5 | 3.8 | DRAM (far above floor) |
| Reduce |  | 1 | 29.5 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 1.0 | 1.0 | 28.5 | DRAM (far above floor) |
| Embeddings |  | 1 | 23.4 | 0.1% | 128 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 16.4 | 16.4 | 1.4 | DRAM |
| BinaryNg | ADD | 1 | 19.0 | 0.1% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 8.9 | 8.9 | 2.1 | L1/NoC |
| LayerNorm | RMSNORM | 1 | 18.8 | 0.1% | 128 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 8.6 | 8.6 | 2.2 | DRAM |
| BinaryNg | DIV | 2 | 14.0 | 0.0% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 17.8 | DRAM (far above floor) |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 34.1 | DRAM (far above floor) |
| FillPad |  | 1 | 5.8 | 0.0% | 128 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 2.0 | 2.0 | 2.8 | DRAM |
| FillPad |  | 1 | 4.8 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 12.6 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 2 | 4.6 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 142.3 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 3.3 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 1.0 | 1.0 | 3.2 | DRAM (far above floor) |
| Clone |  | 1 | 2.8 | 0.0% | 128 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 2.0 | 2.0 | 1.4 | DRAM |
| BinaryNg | MUL | 1 | 2.8 | 0.0% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 4.8 | DRAM (far above floor) |
| Typecast |  | 1 | 2.7 | 0.0% | 128 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 1.5 | 1.5 | 1.7 | DRAM |
| BinaryNg | SUB | 1 | 2.7 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 1.0 | 1.0 | 2.6 | DRAM |
| Unary | SQRT | 1 | 2.7 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 165.8 | DRAM (far above floor) |
| Clone |  | 1 | 1.6 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.1 | DRAM (far above floor) |
