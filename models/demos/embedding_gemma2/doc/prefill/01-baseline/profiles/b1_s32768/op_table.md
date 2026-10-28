### b1_s32768: 646 ops, device kernel sum 636.357 ms, eager op-to-op gaps 0.571 ms, roofline floor 236.172 ms (clock 1350 MHz, 130 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 4 | 322512.7 | 50.7% | 130 | H4 S32768 keys32768 D512 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 97891.0 | 6111.2 | 97891.0 | 3.3 | compute (far above floor) |
| SDPA |  | 20 | 50251.6 | 7.9% | 130 | H4 S32768 keys1025 D256 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 7655.2 | 27770.9 | 27770.9 | 1.8 | DRAM |
| BinaryNg | ADD | 1 | 47881.7 | 7.5% | 130 |  | BFLOAT4_B/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2363.4 | 2363.4 | 20.3 | DRAM (far above floor) |
| MinimalMatmul | GELU_TANH | 24 | 29274.1 | 4.6% | 130 | 32768x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 4588.6 | 4967.4 | 4967.4 | 5.9 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 145 | 25628.7 | 4.0% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 19005.7 | 19005.7 | 1.3 | DRAM |
| MinimalMatmul | GELU_TANH | 24 | 24310.0 | 3.8% | 130 | 32768x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1147.2 | 3158.8 | 3158.8 | 7.7 | DRAM (far above floor) |
| MinimalMatmul |  | 44 | 20230.4 | 3.2% | 130 | 32768x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 8412.5 | 9106.9 | 9106.9 | 2.2 | DRAM |
| BinaryNg | ADD | 72 | 16321.8 | 2.6% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 14155.8 | 14155.8 | 1.2 | DRAM |
| BinaryNg | MUL | 24 | 13190.2 | 2.1% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 10027.0 | 10027.0 | 1.3 | DRAM |
| RotaryEmbedding |  | 20 | 12121.5 | 1.9% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 4096.0 | 4096.0 | 3.0 | DRAM |
| MinimalMatmul |  | 28 | 10623.4 | 1.7% | 130 | 32768x2048x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 5353.4 | 5795.3 | 5795.3 | 1.8 | DRAM |
| NlpCreateHeads |  | 20 | 6487.2 | 1.0% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 5570.6 | 5570.6 | 1.2 | DRAM |
| BinaryNg | ADD | 1 | 6371.4 | 1.0% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1187.8 | 1187.8 | 5.4 | DRAM (far above floor) |
| RotaryEmbedding |  | 20 | 6263.6 | 1.0% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 2703.4 | 2703.4 | 2.3 | DRAM |
| BinaryNg | MUL | 24 | 5460.2 | 0.9% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 4718.6 | 4718.6 | 1.2 | DRAM |
| MinimalMatmul |  | 24 | 5416.4 | 0.9% | 130 | 32768x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1147.2 | 3158.8 | 3158.8 | 1.7 | DRAM |
| RotaryEmbedding |  | 4 | 4936.4 | 0.8% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1638.4 | 1638.4 | 3.0 | DRAM (far above floor) |
| MinimalMatmul |  | 20 | 4732.6 | 0.7% | 130 | 32768x1024x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1911.9 | 2725.1 | 2725.1 | 1.7 | DRAM |
| Slice |  | 24 | 3959.9 | 0.6% | 130 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i | 0.0 | 3145.7 | 3145.7 | 1.3 | DRAM |
| MinimalMatmul |  | 1 | 3776.6 | 0.6% | 130 | 32768x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1147.2 | 1651.5 | 1651.5 | 2.3 | DRAM |
| LayerNorm | RMSNORM | 20 | 3544.9 | 0.6% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2785.3 | 2785.3 | 1.3 | DRAM |
| MinimalMatmul |  | 4 | 2641.0 | 0.4% | 130 | 32768x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1147.2 | 1110.8 | 1147.2 | 2.3 | compute |
| NlpCreateHeads |  | 4 | 1969.6 | 0.3% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1671.2 | 1671.2 | 1.2 | DRAM |
| LayerNorm | RMSNORM | 20 | 1846.5 | 0.3% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1392.7 | 1392.7 | 1.3 | DRAM |
| LayerNorm | RMSNORM | 20 | 1744.0 | 0.3% | 130 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i | 0.0 | 1392.6 | 1392.6 | 1.3 | DRAM |
| LayerNorm | RMSNORM | 4 | 1515.2 | 0.2% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1114.1 | 1114.1 | 1.4 | DRAM |
| RotaryEmbedding |  | 4 | 1266.1 | 0.2% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 802.8 | 802.8 | 1.6 | DRAM |
| Matmul |  | 1 | 450.1 | 0.1% | 26 | 32x32768x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 9.0 | 102.6 | 102.6 | 4.4 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 4 | 395.6 | 0.1% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 278.5 | 278.5 | 1.4 | DRAM |
| LayerNorm | RMSNORM | 4 | 354.9 | 0.1% | 130 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i | 0.0 | 278.5 | 278.5 | 1.3 | DRAM |
| MinimalMatmul |  | 1 | 297.9 | 0.0% | 130 | 32768x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 71.7 | 164.7 | 164.7 | 1.8 | DRAM |
| Reduce |  | 1 | 225.3 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 8.2 | 8.2 | 27.5 | DRAM (far above floor) |
| Embeddings |  | 1 | 186.0 | 0.0% | 130 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 131.3 | 131.3 | 1.4 | DRAM |
| Clone |  | 1 | 37.8 | 0.0% | 130 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 16.4 | 16.4 | 2.3 | DRAM |
| FillPad |  | 1 | 33.3 | 0.0% | 130 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 16.4 | 16.4 | 2.0 | DRAM |
| Typecast |  | 1 | 24.3 | 0.0% | 130 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 12.3 | 12.3 | 2.0 | DRAM |
| BinaryNg | SUB | 1 | 18.6 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 8.2 | 8.2 | 2.3 | DRAM |
| BinaryNg | MUL | 1 | 17.7 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 8.2 | 8.2 | 2.2 | DRAM |
| BinaryNg | DIV | 2 | 14.2 | 0.0% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 18.1 | DRAM (far above floor) |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 34.0 | DRAM (far above floor) |
| FillPad |  | 1 | 4.9 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 12.8 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 2 | 4.5 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 141.9 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.7 | 0.0% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 4.7 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.6 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 164.6 | DRAM (far above floor) |
| Clone |  | 1 | 1.6 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.1 | DRAM (far above floor) |
