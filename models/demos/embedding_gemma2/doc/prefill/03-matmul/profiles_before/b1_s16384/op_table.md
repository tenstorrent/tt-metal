### b1_s16384: 637 ops, device kernel sum 211.306 ms, eager op-to-op gaps 0.510 ms, roofline floor 117.506 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 4 | 54632.6 | 25.9% | 120 | H4 S16384 keys16384 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 26512.1 | 2490.4 | 26512.1 | 2.1 | compute |
| SDPA |  | 20 | 25281.1 | 12.0% | 120 | H4 S16384 keys1025 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 4146.6 | 9830.4 | 9830.4 | 2.6 | DRAM |
| MinimalMatmul | GELU_TANH | 24 | 18654.4 | 8.8% | 120 | 16384x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2485.5 | 3984.4 | 3984.4 | 4.7 | DRAM (far above floor) |
| MinimalMatmul |  | 44 | 15652.1 | 7.4% | 120 | 16384x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 4556.8 | 7304.7 | 7304.7 | 2.1 | DRAM |
| BinaryNg | MUL | 48 | 13468.8 | 6.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 18874.4 | 18874.4 | 0.7 | DRAM |
| LayerNorm | RMSNORM | 149 | 13232.2 | 6.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 9765.2 | 9765.2 | 1.4 | DRAM |
| BinaryNg | ADD | 72 | 8328.4 | 3.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 7077.9 | 7077.9 | 1.2 | DRAM |
| RotaryEmbedding |  | 20 | 7810.5 | 3.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 3276.8 | 3276.8 | 2.4 | DRAM |
| MinimalMatmul |  | 28 | 7688.6 | 3.6% | 120 | 16384x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2899.8 | 4648.4 | 4648.4 | 1.7 | DRAM |
| Matmul | GELU_TANH | 24 | 7112.9 | 3.4% | 80 | 16384x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 621.4 | 1585.9 | 1585.9 | 4.5 | DRAM (far above floor) |
| NlpCreateHeads |  | 20 | 5968.2 | 2.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 5242.9 | 5242.9 | 1.1 | DRAM |
| MinimalMatmul |  | 48 | 5653.1 | 2.7% | 120 | 16384x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1242.8 | 3171.8 | 3171.8 | 1.8 | DRAM |
| RotaryEmbedding |  | 20 | 4292.5 | 2.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1966.1 | 1966.1 | 2.2 | DRAM |
| LayerNorm | RMSNORM | 20 | 3559.2 | 1.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2621.5 | 2621.5 | 1.4 | DRAM |
| RotaryEmbedding |  | 4 | 3084.4 | 1.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1310.7 | 1310.7 | 2.4 | DRAM |
| MinimalMatmul |  | 20 | 3079.6 | 1.5% | 120 | 16384x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1035.6 | 1987.8 | 1987.8 | 1.5 | DRAM |
| MinimalMatmul |  | 4 | 2083.2 | 1.0% | 120 | 16384x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 621.4 | 930.6 | 930.6 | 2.2 | DRAM |
| LayerNorm | RMSNORM | 20 | 1872.6 | 0.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1310.7 | 1310.7 | 1.4 | DRAM |
| NlpCreateHeads |  | 4 | 1782.0 | 0.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1572.9 | 1572.9 | 1.1 | DRAM |
| LayerNorm | RMSNORM | 20 | 1735.2 | 0.8% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 1310.7 | 1310.7 | 1.3 | DRAM |
| BinaryNg | ADD | 1 | 1717.2 | 0.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 299.0 | 299.0 | 5.7 | DRAM (far above floor) |
| BinaryNg | ADD | 1 | 1461.6 | 0.7% | 120 |  | BFLOAT4_B/BFLOAT4_B/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 884.7 | 884.7 | 1.7 | DRAM |
| LayerNorm | RMSNORM | 4 | 1378.2 | 0.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1048.6 | 1048.6 | 1.3 | DRAM |
| RotaryEmbedding |  | 4 | 938.8 | 0.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 524.3 | 524.3 | 1.8 | DRAM |
| LayerNorm | RMSNORM | 4 | 333.0 | 0.2% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 262.1 | 262.1 | 1.3 | DRAM |
| Matmul |  | 1 | 225.6 | 0.1% | 24 | 32x16384x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 4.9 | 51.4 | 51.4 | 4.4 | DRAM (far above floor) |
| MinimalMatmul |  | 1 | 159.4 | 0.1% | 120 | 16384x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 38.8 | 82.7 | 82.7 | 1.9 | DRAM |
| Embeddings |  | 1 | 92.3 | 0.0% | 120 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 65.7 | 65.7 | 1.4 | DRAM |
| BinaryNg | DIV | 1 | 7.0 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 17.8 | DRAM (far above floor) |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.8 | DRAM (far above floor) |
| FillPad |  | 1 | 5.0 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 13.0 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.9 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 5.1 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.7 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 166.9 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 1 | 2.3 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 143.4 | DRAM (far above floor) |
| Clone |  | 1 | 1.7 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.5 | DRAM (far above floor) |
