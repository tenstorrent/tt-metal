### b1_s65536: 637 ops, device kernel sum 1418.856 ms, eager op-to-op gaps 0.543 ms, roofline floor 843.936 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 4 | 810746.5 | 57.1% | 120 | H4 S65536 keys65536 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 424194.3 | 24117.2 | 424194.3 | 1.9 | compute |
| SDPA |  | 20 | 90640.1 | 6.4% | 120 | H4 S65536 keys1025 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 16586.3 | 110100.5 | 110100.5 | 0.8 | DRAM |
| MinimalMatmul | GELU_TANH | 24 | 66465.2 | 4.7% | 120 | 65536x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 9942.1 | 15780.9 | 15780.9 | 4.2 | DRAM (far above floor) |
| MinimalMatmul |  | 44 | 61798.3 | 4.4% | 120 | 65536x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 18227.1 | 28931.6 | 28931.6 | 2.1 | DRAM |
| LayerNorm | RMSNORM | 149 | 50243.0 | 3.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 39059.8 | 39059.8 | 1.3 | DRAM |
| BinaryNg | MUL | 24 | 41506.1 | 2.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 37748.7 | 37748.7 | 1.1 | DRAM |
| MinimalMatmul |  | 72 | 31279.1 | 2.2% | 120 | 65536x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 7456.5 | 18913.5 | 18913.5 | 1.7 | DRAM |
| BinaryNg | ADD | 72 | 31108.8 | 2.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 28311.6 | 28311.6 | 1.1 | DRAM |
| RotaryEmbedding |  | 20 | 29657.8 | 2.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 13107.2 | 13107.2 | 2.3 | DRAM |
| MinimalMatmul |  | 28 | 29490.4 | 2.1% | 120 | 65536x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 11599.1 | 18411.0 | 18411.0 | 1.6 | DRAM |
| BinaryNg | ADD | 1 | 24465.4 | 1.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 4735.0 | 4735.0 | 5.2 | DRAM (far above floor) |
| NlpCreateHeads |  | 20 | 23213.7 | 1.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 20971.5 | 20971.5 | 1.1 | DRAM |
| BinaryNg | ADD | 1 | 21262.4 | 1.5% | 120 |  | BFLOAT4_B/BFLOAT4_B/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 14155.8 | 14155.8 | 1.5 | DRAM |
| BinaryNg | MUL | 24 | 15288.5 | 1.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 9437.2 | 9437.2 | 1.6 | DRAM |
| RotaryEmbedding |  | 20 | 15023.5 | 1.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 7864.3 | 7864.3 | 1.9 | DRAM |
| LayerNorm | RMSNORM | 20 | 13702.7 | 1.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 10485.8 | 10485.8 | 1.3 | DRAM |
| RotaryEmbedding |  | 4 | 11948.3 | 0.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 5242.9 | 5242.9 | 2.3 | DRAM |
| MinimalMatmul |  | 20 | 10659.5 | 0.8% | 120 | 65536x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 4142.5 | 7886.1 | 7886.1 | 1.4 | DRAM |
| MinimalMatmul |  | 4 | 7895.4 | 0.6% | 120 | 65536x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2485.5 | 3683.1 | 3683.1 | 2.1 | DRAM |
| NlpCreateHeads |  | 4 | 6975.0 | 0.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 6291.5 | 6291.5 | 1.1 | DRAM |
| LayerNorm | RMSNORM | 20 | 6972.8 | 0.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 5242.9 | 5242.9 | 1.3 | DRAM |
| LayerNorm | RMSNORM | 20 | 6808.0 | 0.5% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 5242.9 | 5242.9 | 1.3 | DRAM |
| LayerNorm | RMSNORM | 4 | 5383.7 | 0.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 4194.3 | 4194.3 | 1.3 | DRAM |
| RotaryEmbedding |  | 4 | 3090.8 | 0.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 2097.2 | 2097.2 | 1.5 | DRAM |
| LayerNorm | RMSNORM | 4 | 1310.6 | 0.1% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 1048.6 | 1048.6 | 1.2 | DRAM |
| Matmul |  | 1 | 929.4 | 0.1% | 24 | 32x65536x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 19.4 | 205.0 | 205.0 | 4.5 | DRAM (far above floor) |
| MinimalMatmul |  | 1 | 616.5 | 0.0% | 120 | 65536x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 155.3 | 328.5 | 328.5 | 1.9 | DRAM |
| Embeddings |  | 1 | 345.8 | 0.0% | 120 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 262.7 | 262.7 | 1.3 | DRAM |
| Reduce |  | 1 | 7.1 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 35.3 | DRAM (far above floor) |
| BinaryNg | DIV | 1 | 7.1 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 18.0 | DRAM (far above floor) |
| FillPad |  | 1 | 5.2 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 13.4 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 3.0 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 5.2 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.8 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 173.6 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 1 | 2.4 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 148.1 | DRAM (far above floor) |
| Clone |  | 1 | 1.7 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.4 | DRAM (far above floor) |
