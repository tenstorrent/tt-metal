### b1_s65536: 646 ops, device kernel sum 2013.640 ms, eager op-to-op gaps 0.560 ms, roofline floor 722.115 ms (clock 1350 MHz, 130 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 4 | 1289024.1 | 64.0% | 130 | H4 S65536 keys65536 D512 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 391564.0 | 21659.6 | 391564.0 | 3.3 | compute (far above floor) |
| BinaryNg | ADD | 1 | 191529.9 | 9.5% | 130 |  | BFLOAT4_B/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 9445.4 | 9445.4 | 20.3 | DRAM (far above floor) |
| SDPA |  | 20 | 99082.0 | 4.9% | 130 | H4 S65536 keys1025 D256 | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT4_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i/DRAM-i/DRAM-i | 15310.4 | 102727.7 | 102727.7 | 1.0 | DRAM |
| MinimalMatmul | GELU_TANH | 24 | 57288.0 | 2.8% | 130 | 65536x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 9177.3 | 9882.6 | 9882.6 | 5.8 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 145 | 50760.2 | 2.5% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 38011.2 | 38011.2 | 1.3 | DRAM |
| MinimalMatmul | GELU_TANH | 24 | 47702.3 | 2.4% | 130 | 65536x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2294.3 | 6304.5 | 6304.5 | 7.6 | DRAM (far above floor) |
| MinimalMatmul |  | 44 | 38688.0 | 1.9% | 130 | 65536x512x2048 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 16825.0 | 18118.1 | 18118.1 | 2.1 | DRAM |
| BinaryNg | ADD | 72 | 32025.5 | 1.6% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 28311.6 | 28311.6 | 1.1 | DRAM |
| BinaryNg | MUL | 24 | 25837.4 | 1.3% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 20054.0 | 20054.0 | 1.3 | DRAM |
| BinaryNg | ADD | 1 | 25293.8 | 1.3% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 4735.0 | 4735.0 | 5.3 | DRAM (far above floor) |
| RotaryEmbedding |  | 20 | 23116.9 | 1.1% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 8192.0 | 8192.0 | 2.8 | DRAM |
| MinimalMatmul |  | 28 | 21077.2 | 1.0% | 130 | 65536x2048x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 10706.8 | 11529.7 | 11529.7 | 1.8 | DRAM |
| NlpCreateHeads |  | 20 | 12852.2 | 0.6% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 11141.1 | 11141.1 | 1.2 | DRAM |
| RotaryEmbedding |  | 20 | 11991.0 | 0.6% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 5406.7 | 5406.7 | 2.2 | DRAM |
| BinaryNg | MUL | 24 | 10700.0 | 0.5% | 130 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 9437.2 | 9437.2 | 1.1 | DRAM |
| MinimalMatmul |  | 24 | 10568.4 | 0.5% | 130 | 65536x512x512 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2294.3 | 6304.5 | 6304.5 | 1.7 | DRAM |
| MinimalMatmul |  | 20 | 9547.7 | 0.5% | 130 | 65536x1024x512 | BFLOAT8_B/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 3823.9 | 5428.5 | 5428.5 | 1.8 | DRAM |
| RotaryEmbedding |  | 4 | 9168.1 | 0.5% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 3276.8 | 3276.8 | 2.8 | DRAM |
| Slice |  | 24 | 7858.2 | 0.4% | 130 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i | 0.0 | 6291.5 | 6291.5 | 1.2 | DRAM |
| MinimalMatmul |  | 1 | 7513.8 | 0.4% | 130 | 65536x512x12288 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2294.3 | 3289.9 | 3289.9 | 2.3 | DRAM |
| LayerNorm | RMSNORM | 20 | 6813.9 | 0.3% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 5570.6 | 5570.6 | 1.2 | DRAM |
| MinimalMatmul |  | 4 | 5079.9 | 0.3% | 130 | 65536x512x3072 | BFLOAT16/BFLOAT8_B/BFLOAT8_B | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 2294.3 | 2208.5 | 2294.3 | 2.2 | compute |
| NlpCreateHeads |  | 4 | 3839.0 | 0.2% | 130 |  | BFLOAT8_B/BFLOAT8_B/BFLOAT8_B/BFLOAT8_B |  | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 3342.3 | 3342.3 | 1.1 | DRAM |
| LayerNorm | RMSNORM | 20 | 3503.0 | 0.2% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2785.3 | 2785.3 | 1.3 | DRAM |
| LayerNorm | RMSNORM | 20 | 3377.5 | 0.2% | 130 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i | 0.0 | 2785.3 | 2785.3 | 1.2 | DRAM |
| LayerNorm | RMSNORM | 4 | 2934.2 | 0.1% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2228.2 | 2228.2 | 1.3 | DRAM |
| RotaryEmbedding |  | 4 | 2427.9 | 0.1% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i/DRAM-i | 0.0 | 1605.6 | 1605.6 | 1.5 | DRAM |
| Matmul |  | 1 | 900.7 | 0.0% | 26 | 32x65536x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 17.9 | 205.0 | 205.0 | 4.4 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 4 | 771.9 | 0.0% | 130 |  | BFLOAT8_B/BFLOAT16/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 557.1 | 557.1 | 1.4 | DRAM |
| LayerNorm | RMSNORM | 4 | 697.6 | 0.0% | 130 |  | BFLOAT8_B/BFLOAT8_B | HiFi4 | DRAM-i/DRAM-i | 0.0 | 557.1 | 557.1 | 1.3 | DRAM |
| MinimalMatmul |  | 1 | 596.8 | 0.0% | 130 | 65536x512x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 143.4 | 328.5 | 328.5 | 1.8 | DRAM |
| Reduce |  | 1 | 450.9 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 16.4 | 16.4 | 27.5 | DRAM (far above floor) |
| Embeddings |  | 1 | 380.9 | 0.0% | 130 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 262.7 | 262.7 | 1.5 | DRAM |
| Clone |  | 1 | 53.8 | 0.0% | 130 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 32.8 | 32.8 | 1.6 | DRAM |
| FillPad |  | 1 | 52.5 | 0.0% | 130 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 32.8 | 32.8 | 1.6 | DRAM |
| Typecast |  | 1 | 38.7 | 0.0% | 130 |  | BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 24.6 | 24.6 | 1.6 | DRAM |
| BinaryNg | MUL | 1 | 30.7 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 16.4 | 16.4 | 1.9 | DRAM |
| BinaryNg | SUB | 1 | 27.7 | 0.0% | 130 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 16.4 | 16.4 | 1.7 | DRAM |
| BinaryNg | DIV | 2 | 13.9 | 0.0% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.8 | 0.8 | 17.7 | DRAM (far above floor) |
| Reduce |  | 1 | 6.8 | 0.0% | 1 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.2 | 0.2 | 34.1 | DRAM (far above floor) |
| FillPad |  | 1 | 4.9 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 12.9 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 2 | 4.6 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 142.5 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 2.7 | 0.0% | 130 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 4.6 | DRAM (far above floor) |
| Unary | SQRT | 1 | 2.7 | 0.0% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.0 | 0.0 | 165.7 | DRAM (far above floor) |
| Clone |  | 1 | 1.6 | 0.0% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.4 | 0.4 | 4.3 | DRAM (far above floor) |
