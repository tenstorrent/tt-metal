### b1_s16384: 646 ops, device kernel sum 224.524 ms, eager op-to-op gaps 0.545 ms, roofline floor 118.068 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 4 | 54610.8 | 24.3% | 120 | H4 S16384 keys16384 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 26512.1 | 2490.4 | 26512.1 | 2.1 | compute |
| SDPA |  | 20 | 25288.3 | 11.3% | 120 | H4 S16384 keys1025 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 4146.6 | 9830.4 | 9830.4 | 2.6 | DRAM |
| MinimalMatmul | GELU_TANH | 24 | 18672.0 | 8.3% | 120 | 16384x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2485.5 | 3984.4 | 3984.4 | 4.7 | DRAM (far above floor) |
| MinimalMatmul |  | 44 | 15625.8 | 7.0% | 120 | 16384x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 4556.8 | 7304.7 | 7304.7 | 2.1 | DRAM |
| BinaryNg | MUL | 48 | 13520.5 | 6.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 18874.4 | 18874.4 | 0.7 | DRAM |
| LayerNorm | RMSNORM | 149 | 13279.4 | 5.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 9765.2 | 9765.2 | 1.4 | DRAM |
| BinaryNg | ADD | 1 | 12984.7 | 5.8% | 120 |  | BFLOAT4_B/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 591.9 | 591.9 | 21.9 | DRAM (far above floor) |
| BinaryNg | ADD | 72 | 8358.7 | 3.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 7077.9 | 7077.9 | 1.2 | DRAM |
| RotaryEmbedding |  | 20 | 7833.1 | 3.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 3276.8 | 3276.8 | 2.4 | DRAM |
| MinimalMatmul |  | 28 | 7682.9 | 3.4% | 120 | 16384x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2899.8 | 4648.4 | 4648.4 | 1.7 | DRAM |
| Matmul | GELU_TANH | 24 | 7149.1 | 3.2% | 80 | 16384x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 621.4 | 1585.9 | 1585.9 | 4.5 | DRAM (far above floor) |
| NlpCreateHeads |  | 20 | 5986.2 | 2.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 5242.9 | 5242.9 | 1.1 | DRAM |
| RotaryEmbedding |  | 20 | 4310.0 | 1.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1966.1 | 1966.1 | 2.2 | DRAM |
| LayerNorm | RMSNORM | 20 | 3572.9 | 1.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2621.5 | 2621.5 | 1.4 | DRAM |
| MinimalMatmul |  | 20 | 3095.6 | 1.4% | 120 | 16384x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1035.6 | 1987.8 | 1987.8 | 1.6 | DRAM |
| RotaryEmbedding |  | 4 | 3093.7 | 1.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1310.7 | 1310.7 | 2.4 | DRAM |
| MinimalMatmul |  | 24 | 2839.1 | 1.3% | 120 | 16384x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 621.4 | 1585.9 | 1585.9 | 1.8 | DRAM |
| Slice |  | 24 | 2124.6 | 0.9% | 120 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i | 0.0 | 1572.9 | 1572.9 | 1.4 | DRAM |
| MinimalMatmul |  | 4 | 2094.1 | 0.9% | 120 | 16384x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 621.4 | 930.6 | 930.6 | 2.3 | DRAM |
| MinimalMatmul |  | 1 | 1957.6 | 0.9% | 120 | 16384x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 621.4 | 832.3 | 832.3 | 2.4 | DRAM |
| LayerNorm | RMSNORM | 20 | 1869.4 | 0.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1310.7 | 1310.7 | 1.4 | DRAM |
| NlpCreateHeads |  | 4 | 1793.0 | 0.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1572.9 | 1572.9 | 1.1 | DRAM |
| LayerNorm | RMSNORM | 20 | 1738.5 | 0.8% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 1310.7 | 1310.7 | 1.3 | DRAM |
| BinaryNg | ADD | 1 | 1695.7 | 0.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 299.0 | 299.0 | 5.7 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 4 | 1374.0 | 0.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1048.6 | 1048.6 | 1.3 | DRAM |
| RotaryEmbedding |  | 4 | 954.1 | 0.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 524.3 | 524.3 | 1.8 | DRAM |
| LayerNorm | RMSNORM | 4 | 336.4 | 0.1% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 262.1 | 262.1 | 1.3 | DRAM |
| Matmul |  | 1 | 226.1 | 0.1% | 24 | 32x16384x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 4.9 | 51.4 | 51.4 | 4.4 | DRAM (far above floor) |
| MinimalMatmul |  | 1 | 160.7 | 0.1% | 120 | 16384x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 38.8 | 82.7 | 82.7 | 1.9 | DRAM |
| Reduce |  | 1 | 113.5 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 4.1 | 4.1 | 27.6 | DRAM (far above floor) |
| Embeddings |  | 1 | 90.5 | 0.0% | 120 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 65.7 | 65.7 | 1.4 | DRAM |
| Clone |  | 1 | 15.3 | 0.0% | 120 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 8.2 | 8.2 | 1.9 | DRAM |
| FillPad |  | 1 | 14.4 | 0.0% | 120 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 8.2 | 8.2 | 1.8 | DRAM |
| BinaryNg | DIV | 2 | 14.0 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 17.8 | DRAM (far above floor) |
| Typecast |  | 1 | 10.6 | 0.0% | 120 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 6.1 | 6.1 | 1.7 | DRAM |
| BinaryNg | MUL | 1 | 7.9 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 4.1 | 4.1 | 1.9 | DRAM |
| BinaryNg | SUB | 1 | 7.4 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 4.1 | 4.1 | 1.8 | DRAM |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 34.0 | DRAM (far above floor) |
| FillPad |  | 1 | 5.0 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 13.0 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 2 | 4.6 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 143.2 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.7 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 4.8 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.7 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 166.0 | DRAM (far above floor) |
| Clone |  | 1 | 1.7 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.3 | DRAM (far above floor) |
