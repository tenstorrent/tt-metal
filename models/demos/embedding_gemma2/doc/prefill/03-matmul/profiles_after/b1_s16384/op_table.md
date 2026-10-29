### b1_s16384: 637 ops, device kernel sum 200.708 ms, eager op-to-op gaps 0.482 ms, roofline floor 117.506 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 4 | 54616.8 | 27.2% | 120 | H4 S16384 keys16384 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 26512.1 | 2490.4 | 26512.1 | 2.1 | compute |
| SDPA |  | 20 | 25268.6 | 12.6% | 120 | H4 S16384 keys1025 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 4146.6 | 9830.4 | 9830.4 | 2.6 | DRAM |
| BinaryNg | MUL | 48 | 18493.2 | 9.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 18874.4 | 18874.4 | 1.0 | DRAM |
| MinimalMatmul |  | 44 | 15547.0 | 7.7% | 120 | 16384x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 4556.8 | 7304.7 | 7304.7 | 2.1 | DRAM |
| LayerNorm | RMSNORM | 149 | 13229.5 | 6.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 9765.2 | 9765.2 | 1.4 | DRAM |
| BinaryNg | ADD | 72 | 8350.9 | 4.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 7077.9 | 7077.9 | 1.2 | DRAM |
| MinimalMatmul |  | 72 | 8225.5 | 4.1% | 120 | 16384x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1864.1 | 4757.8 | 4757.8 | 1.7 | DRAM |
| Matmul |  | 24 | 8205.7 | 4.1% | 110 | 4x4096x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2485.5 | 3984.4 | 3984.4 | 2.1 | DRAM |
| RotaryEmbedding |  | 20 | 7818.3 | 3.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 3276.8 | 3276.8 | 2.4 | DRAM |
| MinimalMatmul |  | 28 | 7173.2 | 3.6% | 120 | 16384x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2899.8 | 4648.4 | 4648.4 | 1.5 | DRAM |
| NlpCreateHeads |  | 20 | 5978.2 | 3.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 5242.9 | 5242.9 | 1.1 | DRAM |
| RotaryEmbedding |  | 20 | 4296.4 | 2.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1966.1 | 1966.1 | 2.2 | DRAM |
| LayerNorm | RMSNORM | 20 | 3549.6 | 1.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2621.5 | 2621.5 | 1.4 | DRAM |
| MinimalMatmul |  | 20 | 3094.3 | 1.5% | 120 | 16384x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1035.6 | 1987.8 | 1987.8 | 1.6 | DRAM |
| RotaryEmbedding |  | 4 | 3079.2 | 1.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1310.7 | 1310.7 | 2.3 | DRAM |
| MinimalMatmul |  | 4 | 2060.1 | 1.0% | 120 | 16384x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 621.4 | 930.6 | 930.6 | 2.2 | DRAM |
| LayerNorm | RMSNORM | 20 | 1862.8 | 0.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1310.7 | 1310.7 | 1.4 | DRAM |
| NlpCreateHeads |  | 4 | 1781.6 | 0.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1572.9 | 1572.9 | 1.1 | DRAM |
| LayerNorm | RMSNORM | 20 | 1743.9 | 0.9% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 1310.7 | 1310.7 | 1.3 | DRAM |
| BinaryNg | ADD | 1 | 1720.2 | 0.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 299.0 | 299.0 | 5.8 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 1465.0 | 0.7% | 120 |  | BFLOAT4_B/BFLOAT4_B/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 884.7 | 884.7 | 1.7 | DRAM |
| LayerNorm | RMSNORM | 4 | 1356.9 | 0.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1048.6 | 1048.6 | 1.3 | DRAM |
| RotaryEmbedding |  | 4 | 953.5 | 0.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 524.3 | 524.3 | 1.8 | DRAM |
| LayerNorm | RMSNORM | 4 | 334.4 | 0.2% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 262.1 | 262.1 | 1.3 | DRAM |
| Matmul |  | 1 | 226.9 | 0.1% | 24 | 32x16384x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 4.9 | 51.4 | 51.4 | 4.4 | DRAM (far above floor) |
| MinimalMatmul |  | 1 | 156.2 | 0.1% | 120 | 16384x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 38.8 | 82.7 | 82.7 | 1.9 | DRAM |
| Embeddings |  | 1 | 91.3 | 0.0% | 120 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 65.7 | 65.7 | 1.4 | DRAM |
| BinaryNg | DIV | 1 | 7.0 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 17.9 | DRAM (far above floor) |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.9 | DRAM (far above floor) |
| FillPad |  | 1 | 5.0 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 13.1 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.9 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 5.1 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.6 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 165.4 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 1 | 2.3 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 143.8 | DRAM (far above floor) |
| Clone |  | 1 | 1.7 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.5 | DRAM (far above floor) |
