### b1_s65536: 646 ops, device kernel sum 1661.090 ms, eager op-to-op gaps 0.528 ms, roofline floor 870.954 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 4 | 802399.1 | 48.3% | 120 | H4 S65536 keys65536 D512 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 424194.3 | 24117.2 | 424194.3 | 1.9 | compute |
| BinaryNg | ADD | 1 | 215472.7 | 13.0% | 120 |  | BFLOAT4_B/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 9445.4 | 9445.4 | 22.8 | DRAM (far above floor) |
| SDPA |  | 20 | 91197.6 | 5.5% | 120 | H4 S65536 keys1025 D256 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 16586.3 | 110100.5 | 110100.5 | 0.8 | DRAM |
| MinimalMatmul | GELU_TANH | 24 | 66012.0 | 4.0% | 120 | 65536x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 9942.1 | 15780.9 | 15780.9 | 4.2 | DRAM (far above floor) |
| MinimalMatmul |  | 44 | 61658.8 | 3.7% | 120 | 65536x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 18227.1 | 28931.6 | 28931.6 | 2.1 | DRAM |
| MinimalMatmul | GELU_TANH | 24 | 54157.4 | 3.3% | 120 | 65536x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2485.5 | 6304.5 | 6304.5 | 8.6 | DRAM (far above floor) |
| BinaryNg | MUL | 48 | 53899.9 | 3.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 75497.5 | 75497.5 | 0.7 | DRAM |
| LayerNorm | RMSNORM | 149 | 51754.3 | 3.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 39059.8 | 39059.8 | 1.3 | DRAM |
| BinaryNg | ADD | 72 | 32257.6 | 1.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 28311.6 | 28311.6 | 1.1 | DRAM |
| RotaryEmbedding |  | 20 | 30518.6 | 1.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 13107.2 | 13107.2 | 2.3 | DRAM |
| MinimalMatmul |  | 28 | 29608.3 | 1.8% | 120 | 65536x2048x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 11599.1 | 18411.0 | 18411.0 | 1.6 | DRAM |
| BinaryNg | ADD | 1 | 26542.2 | 1.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 4735.0 | 4735.0 | 5.6 | DRAM (far above floor) |
| NlpCreateHeads |  | 20 | 24121.7 | 1.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 20971.5 | 20971.5 | 1.2 | DRAM |
| RotaryEmbedding |  | 20 | 15510.1 | 0.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 7864.3 | 7864.3 | 2.0 | DRAM |
| LayerNorm | RMSNORM | 20 | 13903.4 | 0.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 10485.8 | 10485.8 | 1.3 | DRAM |
| RotaryEmbedding |  | 4 | 12257.2 | 0.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 5242.9 | 5242.9 | 2.3 | DRAM |
| MinimalMatmul |  | 20 | 10932.0 | 0.7% | 120 | 65536x1024x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 4142.5 | 7886.1 | 7886.1 | 1.4 | DRAM |
| MinimalMatmul |  | 24 | 10745.7 | 0.6% | 120 | 65536x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2485.5 | 6304.5 | 6304.5 | 1.7 | DRAM |
| Slice |  | 24 | 8378.7 | 0.5% | 120 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i | 0.0 | 6291.5 | 6291.5 | 1.3 | DRAM |
| MinimalMatmul |  | 4 | 7920.2 | 0.5% | 120 | 65536x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2485.5 | 3683.1 | 3683.1 | 2.2 | DRAM |
| MinimalMatmul |  | 1 | 7864.8 | 0.5% | 120 | 65536x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2485.5 | 3289.9 | 3289.9 | 2.4 | DRAM |
| NlpCreateHeads |  | 4 | 7232.3 | 0.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 6291.5 | 6291.5 | 1.1 | DRAM |
| LayerNorm | RMSNORM | 20 | 7089.9 | 0.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 5242.9 | 5242.9 | 1.4 | DRAM |
| LayerNorm | RMSNORM | 20 | 7053.3 | 0.4% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 5242.9 | 5242.9 | 1.3 | DRAM |
| LayerNorm | RMSNORM | 4 | 5550.7 | 0.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 4194.3 | 4194.3 | 1.3 | DRAM |
| RotaryEmbedding |  | 4 | 3121.9 | 0.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 2097.2 | 2097.2 | 1.5 | DRAM |
| LayerNorm | RMSNORM | 4 | 1346.5 | 0.1% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 1048.6 | 1048.6 | 1.3 | DRAM |
| Matmul |  | 1 | 925.6 | 0.1% | 24 | 32x65536x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 19.4 | 205.0 | 205.0 | 4.5 | DRAM (far above floor) |
| MinimalMatmul |  | 1 | 609.7 | 0.0% | 120 | 65536x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 155.3 | 328.5 | 328.5 | 1.9 | DRAM |
| Reduce |  | 1 | 468.3 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 16.4 | 16.4 | 28.6 | DRAM (far above floor) |
| Embeddings |  | 1 | 378.2 | 0.0% | 120 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 262.7 | 262.7 | 1.4 | DRAM |
| FillPad |  | 1 | 42.9 | 0.0% | 120 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 32.8 | 32.8 | 1.3 | DRAM |
| Clone |  | 1 | 40.8 | 0.0% | 120 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 32.8 | 32.8 | 1.2 | DRAM |
| Typecast |  | 1 | 33.3 | 0.0% | 120 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 24.6 | 24.6 | 1.4 | DRAM |
| BinaryNg | MUL | 1 | 24.8 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 16.4 | 16.4 | 1.5 | DRAM |
| BinaryNg | SUB | 1 | 21.8 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 16.4 | 16.4 | 1.3 | DRAM |
| BinaryNg | DIV | 2 | 14.2 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 18.1 | DRAM (far above floor) |
| Reduce |  | 1 | 7.1 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 35.4 | DRAM (far above floor) |
| FillPad |  | 1 | 5.1 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 13.3 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 2 | 4.7 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 145.3 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.8 | 0.0% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 4.8 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.7 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 171.2 | DRAM (far above floor) |
| Clone |  | 1 | 1.6 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.2 | DRAM (far above floor) |
