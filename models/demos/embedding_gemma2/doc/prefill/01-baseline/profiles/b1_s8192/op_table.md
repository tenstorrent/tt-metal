### b1_s8192: 646 ops, device kernel sum 92.333 ms, eager op-to-op gaps 0.602 ms, roofline floor 35.807 ms (clock 1350 MHz, 130 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 4 | 20658.2 | 22.4% | 130 | H4 S8192 keys8192 D512 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 6118.2 | 643.1 | 6118.2 | 3.4 | compute (far above floor) |
| SDPA |  | 20 | 13587.4 | 14.7% | 130 | H4 S8192 keys1025 D256 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 1913.8 | 2519.0 | 2519.0 | 5.4 | DRAM (far above floor) |
| Matmul | GELU_TANH | 24 | 7362.5 | 8.0% | 130 | 8192x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1147.2 | 1281.0 | 1281.0 | 5.7 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 145 | 6665.5 | 7.2% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 4751.7 | 4751.7 | 1.4 | DRAM |
| MinimalMatmul |  | 44 | 6211.3 | 6.7% | 130 | 8192x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2103.1 | 2348.5 | 2348.5 | 2.6 | DRAM |
| BinaryNg | ADD | 72 | 4954.1 | 5.4% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 3538.9 | 3538.9 | 1.4 | DRAM |
| RotaryEmbedding |  | 20 | 3377.0 | 3.7% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1024.0 | 1024.0 | 3.3 | DRAM (far above floor) |
| Matmul | GELU_TANH | 24 | 3372.1 | 3.7% | 80 | 8192x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 286.8 | 799.5 | 799.5 | 4.2 | DRAM (far above floor) |
| BinaryNg | MUL | 24 | 3280.3 | 3.6% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2506.8 | 2506.8 | 1.3 | DRAM |
| BinaryNg | ADD | 1 | 3004.7 | 3.3% | 130 |  | BFLOAT4_B/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 148.5 | 148.5 | 20.2 | DRAM (far above floor) |
| MinimalMatmul |  | 28 | 2971.7 | 3.2% | 130 | 8192x2048x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1338.4 | 1494.5 | 1494.5 | 2.0 | DRAM |
| RotaryEmbedding |  | 20 | 1934.7 | 2.1% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 675.8 | 675.8 | 2.9 | DRAM |
| NlpCreateHeads |  | 20 | 1651.0 | 1.8% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1392.6 | 1392.6 | 1.2 | DRAM |
| BinaryNg | MUL | 24 | 1646.1 | 1.8% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 1179.6 | 1179.6 | 1.4 | DRAM |
| MinimalMatmul |  | 24 | 1511.1 | 1.6% | 130 | 8192x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 286.8 | 799.5 | 799.5 | 1.9 | DRAM |
| MinimalMatmul |  | 20 | 1396.0 | 1.5% | 130 | 8192x1024x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 478.0 | 697.6 | 697.6 | 2.0 | DRAM |
| RotaryEmbedding |  | 4 | 1271.5 | 1.4% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 409.6 | 409.6 | 3.1 | DRAM (far above floor) |
| Slice |  | 24 | 1206.8 | 1.3% | 130 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i | 0.0 | 786.4 | 786.4 | 1.5 | DRAM |
| MinimalMatmul |  | 1 | 1073.3 | 1.2% | 130 | 8192x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 286.8 | 422.7 | 422.7 | 2.5 | DRAM |
| LayerNorm | RMSNORM | 20 | 1007.4 | 1.1% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 696.3 | 696.3 | 1.4 | DRAM |
| MinimalMatmul |  | 4 | 796.6 | 0.9% | 130 | 8192x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 286.8 | 287.5 | 287.5 | 2.8 | DRAM |
| LayerNorm | RMSNORM | 20 | 574.3 | 0.6% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 348.2 | 348.2 | 1.6 | DRAM |
| NlpCreateHeads |  | 4 | 508.0 | 0.6% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 417.8 | 417.8 | 1.2 | DRAM |
| LayerNorm | RMSNORM | 20 | 506.3 | 0.5% | 130 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i | 0.0 | 348.2 | 348.2 | 1.5 | DRAM |
| BinaryNg | ADD | 1 | 414.1 | 0.4% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 75.8 | 75.8 | 5.5 | DRAM (far above floor) |
| RotaryEmbedding |  | 4 | 407.5 | 0.4% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 200.7 | 200.7 | 2.0 | DRAM |
| LayerNorm | RMSNORM | 4 | 395.1 | 0.4% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 278.5 | 278.5 | 1.4 | DRAM |
| LayerNorm | RMSNORM | 4 | 119.7 | 0.1% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 69.6 | 69.6 | 1.7 | DRAM |
| Matmul |  | 1 | 113.6 | 0.1% | 26 | 32x8192x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 2.2 | 25.8 | 25.8 | 4.4 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 4 | 101.0 | 0.1% | 130 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i | 0.0 | 69.6 | 69.6 | 1.4 | DRAM |
| MinimalMatmul |  | 1 | 83.6 | 0.1% | 130 | 8192x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 17.9 | 41.8 | 41.8 | 2.0 | DRAM |
| Reduce |  | 1 | 57.5 | 0.1% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 2.1 | 2.1 | 28.0 | DRAM (far above floor) |
| Embeddings |  | 1 | 46.5 | 0.1% | 130 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 32.8 | 32.8 | 1.4 | DRAM |
| BinaryNg | DIV | 2 | 14.3 | 0.0% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 18.2 | DRAM (far above floor) |
| FillPad |  | 1 | 8.1 | 0.0% | 130 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 4.1 | 4.1 | 2.0 | DRAM |
| Clone |  | 1 | 7.0 | 0.0% | 130 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 4.1 | 4.1 | 1.7 | DRAM |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 33.8 | DRAM (far above floor) |
| Typecast |  | 1 | 5.3 | 0.0% | 130 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 3.1 | 3.1 | 1.7 | DRAM |
| FillPad |  | 1 | 5.0 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 12.9 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 4.7 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 2.0 | 2.0 | 2.3 | DRAM |
| Unary | CLAMP_TSS | 2 | 4.6 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 142.9 | DRAM (far above floor) |
| BinaryNg | SUB | 1 | 4.0 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 2.0 | 2.0 | 1.9 | DRAM |
| BinaryNg | MUL | 1 | 2.7 | 0.0% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 4.7 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.7 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 166.7 | DRAM (far above floor) |
| Clone |  | 1 | 1.7 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.3 | DRAM (far above floor) |
