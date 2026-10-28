### b1_s16384: 646 ops, device kernel sum 222.101 ms, eager op-to-op gaps 0.609 ms, roofline floor 86.967 ms (clock 1350 MHz, 130 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 4 | 80961.6 | 36.5% | 130 | H4 S16384 keys16384 D512 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 24472.7 | 1876.0 | 24472.7 | 3.3 | compute (far above floor) |
| SDPA |  | 20 | 25816.5 | 11.6% | 130 | H4 S16384 keys1025 D256 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 3827.6 | 7987.2 | 7987.2 | 3.2 | DRAM (far above floor) |
| MinimalMatmul | GELU_TANH | 24 | 15200.4 | 6.8% | 130 | 16384x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2294.3 | 2509.8 | 2509.8 | 6.1 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 145 | 12946.6 | 5.8% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 9503.0 | 9503.0 | 1.4 | DRAM |
| BinaryNg | ADD | 1 | 11984.6 | 5.4% | 130 |  | BFLOAT4_B/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 591.9 | 591.9 | 20.2 | DRAM (far above floor) |
| MinimalMatmul |  | 44 | 10368.8 | 4.7% | 130 | 16384x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 4206.3 | 4601.3 | 4601.3 | 2.3 | DRAM |
| BinaryNg | ADD | 72 | 8186.3 | 3.7% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 7077.9 | 7077.9 | 1.2 | DRAM |
| Matmul | GELU_TANH | 24 | 7118.9 | 3.2% | 80 | 16384x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 573.6 | 1585.9 | 1585.9 | 4.5 | DRAM (far above floor) |
| BinaryNg | MUL | 24 | 6565.7 | 3.0% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 5013.5 | 5013.5 | 1.3 | DRAM |
| RotaryEmbedding |  | 20 | 6278.1 | 2.8% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 2048.0 | 2048.0 | 3.1 | DRAM (far above floor) |
| MinimalMatmul |  | 28 | 5544.1 | 2.5% | 130 | 16384x2048x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2676.7 | 2928.1 | 2928.1 | 1.9 | DRAM |
| RotaryEmbedding |  | 20 | 3363.3 | 1.5% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1351.7 | 1351.7 | 2.5 | DRAM |
| NlpCreateHeads |  | 20 | 3280.9 | 1.5% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 2785.3 | 2785.3 | 1.2 | DRAM |
| BinaryNg | MUL | 24 | 2739.4 | 1.2% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2359.3 | 2359.3 | 1.2 | DRAM |
| MinimalMatmul |  | 24 | 2727.0 | 1.2% | 130 | 16384x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 573.6 | 1585.9 | 1585.9 | 1.7 | DRAM |
| RotaryEmbedding |  | 4 | 2470.1 | 1.1% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 819.2 | 819.2 | 3.0 | DRAM (far above floor) |
| MinimalMatmul |  | 20 | 2376.1 | 1.1% | 130 | 16384x1024x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 956.0 | 1373.4 | 1373.4 | 1.7 | DRAM |
| Slice |  | 24 | 2013.1 | 0.9% | 130 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i | 0.0 | 1572.9 | 1572.9 | 1.3 | DRAM |
| MinimalMatmul |  | 1 | 1886.4 | 0.8% | 130 | 16384x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 573.6 | 832.3 | 832.3 | 2.3 | DRAM |
| LayerNorm | RMSNORM | 20 | 1851.0 | 0.8% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1392.7 | 1392.7 | 1.3 | DRAM |
| BinaryNg | ADD | 1 | 1615.9 | 0.7% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 299.0 | 299.0 | 5.4 | DRAM (far above floor) |
| MinimalMatmul |  | 4 | 1374.8 | 0.6% | 130 | 16384x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 573.6 | 561.9 | 573.6 | 2.4 | compute |
| LayerNorm | RMSNORM | 20 | 999.2 | 0.4% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 696.3 | 696.3 | 1.4 | DRAM |
| NlpCreateHeads |  | 4 | 982.1 | 0.4% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 835.6 | 835.6 | 1.2 | DRAM |
| LayerNorm | RMSNORM | 20 | 912.1 | 0.4% | 130 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i | 0.0 | 696.3 | 696.3 | 1.3 | DRAM |
| LayerNorm | RMSNORM | 4 | 764.3 | 0.3% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 557.1 | 557.1 | 1.4 | DRAM |
| RotaryEmbedding |  | 4 | 700.4 | 0.3% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 401.4 | 401.4 | 1.7 | DRAM |
| Matmul |  | 1 | 225.7 | 0.1% | 26 | 32x16384x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 4.5 | 51.4 | 51.4 | 4.4 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 4 | 208.0 | 0.1% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 139.3 | 139.3 | 1.5 | DRAM |
| LayerNorm | RMSNORM | 4 | 183.8 | 0.1% | 130 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i | 0.0 | 139.3 | 139.3 | 1.3 | DRAM |
| MinimalMatmul |  | 1 | 150.3 | 0.1% | 130 | 16384x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 35.8 | 82.7 | 82.7 | 1.8 | DRAM |
| Reduce |  | 1 | 113.4 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 4.1 | 4.1 | 27.6 | DRAM (far above floor) |
| Embeddings |  | 1 | 90.4 | 0.0% | 130 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 65.7 | 65.7 | 1.4 | DRAM |
| Clone |  | 1 | 18.2 | 0.0% | 130 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 8.2 | 8.2 | 2.2 | DRAM |
| FillPad |  | 1 | 16.1 | 0.0% | 130 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 8.2 | 8.2 | 2.0 | DRAM |
| BinaryNg | DIV | 2 | 14.0 | 0.0% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 17.9 | DRAM (far above floor) |
| Typecast |  | 1 | 12.5 | 0.0% | 130 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 6.1 | 6.1 | 2.0 | DRAM |
| BinaryNg | SUB | 1 | 8.9 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 4.1 | 4.1 | 2.2 | DRAM |
| BinaryNg | MUL | 1 | 8.8 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 4.1 | 4.1 | 2.1 | DRAM |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 34.0 | DRAM (far above floor) |
| FillPad |  | 1 | 4.9 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 12.8 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 2 | 4.5 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 140.8 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.8 | 0.0% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 4.8 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.7 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 167.1 | DRAM (far above floor) |
| Clone |  | 1 | 1.6 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.2 | DRAM (far above floor) |
