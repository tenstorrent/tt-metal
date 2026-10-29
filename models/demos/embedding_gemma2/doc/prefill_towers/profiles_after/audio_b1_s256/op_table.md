### audio b1_s256: 1079 ops, device kernel sum 6.055 ms, eager op-to-op gaps 33.094 ms, roofline floor 0.836 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Matmul | HARDTANH | 24 | 640.3 | 10.6% | 88 | 256x4096x1024 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 155.3 | 215.0 | 215.0 | 3.0 | DRAM |
| GatherCodegen |  | 12 | 590.1 | 9.7% | 120 |  | BFLOAT16/UINT16/BFLOAT16 |  | L1-i/L1-i/L1-i | 0.0 | 0.8 | 0.8 | 778.0 | L1/NoC (far above floor) |
| Matmul | HARDTANH | 24 | 568.6 | 9.4% | 96 | 256x1024x4096 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 155.3 | 215.0 | 215.0 | 2.6 | DRAM |
| LayerNorm | RMSNORM | 96 | 506.7 | 8.4% | 32 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1_BLOCK-s/DRAM-i/L1_BLOCK-s | 0.0 | 10.1 | 10.1 | 50.2 | L1/NoC (far above floor) |
| BinaryNg | MUL | 60 | 357.3 | 5.9% | 120 |  | BFLOAT16/FLOAT32/FLOAT32 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 24.5 | 24.5 | 14.6 | L1/NoC (far above floor) |
| Unary | SILU | 24 | 252.6 | 4.2% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 9.7 | 9.7 | 26.0 | L1/NoC (far above floor) |
| Matmul |  | 12 | 213.7 | 3.5% | 96 | 256x1024x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 58.3 | 80.8 | 80.8 | 2.6 | DRAM |
| Matmul |  | 48 | 192.3 | 3.2% | 88 | 256x256x1024 | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi2 | DRAM-i/L1-i/L1-i | 19.4 | 17.1 | 19.4 | 9.9 | compute (far above floor) |
| ShardedToInterleaved |  | 96 | 190.5 | 3.1% | 32 |  | BFLOAT16/BFLOAT16 |  | L1_BLOCK-s/L1-i | 0.0 | 9.7 | 9.7 | 19.6 | L1/NoC (far above floor) |
| BinaryNg | ADD | 48 | 188.9 | 3.1% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 14.6 | 14.6 | 13.0 | L1/NoC (far above floor) |
| Matmul | HARDTANH | 12 | 186.1 | 3.1% | 88 | 256x1024x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 38.8 | 54.0 | 54.0 | 3.4 | DRAM (far above floor) |
| InterleavedToSharded |  | 95 | 158.5 | 2.6% | 32 |  | BFLOAT16/BFLOAT16 |  | L1-i/L1_BLOCK-s | 0.0 | 9.6 | 9.6 | 16.5 | L1/NoC (far above floor) |
| BinaryNg | ADD | 47 | 127.4 | 2.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 7.1 | 7.1 | 17.9 | L1/NoC (far above floor) |
| Matmul | HARDTANH | 12 | 125.1 | 2.1% | 88 | 256x1024x1024 | FLOAT32/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 19.4 | 27.9 | 27.9 | 4.5 | L1/NoC (far above floor) |
| Matmul | HARDTANH | 12 | 123.9 | 2.0% | 88 | 256x1024x1024 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 19.4 | 27.3 | 27.3 | 4.5 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 61 | 121.2 | 2.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 6.2 | 6.2 | 19.6 | L1/NoC (far above floor) |
| BinaryNg | MUL | 12 | 104.8 | 1.7% | 120 |  | BFLOAT16/FLOAT32/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 1.6 | 1.6 | 65.6 | L1/NoC (far above floor) |
| ReshapeView |  | 1 | 97.8 | 1.6% | 120 |  | FLOAT32/FLOAT32 |  | L1-i/L1-i | 0.0 | 0.1 | 0.1 | 966.8 | L1/NoC (far above floor) |
| BinaryNg | MUL | 12 | 74.9 | 1.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 1.8 | 1.8 | 41.2 | L1/NoC (far above floor) |
| Unary | SILU | 12 | 68.9 | 1.1% | 120 |  | FLOAT32/FLOAT32 | HiFi4 | L1-i/L1-i | 0.0 | 2.4 | 2.4 | 28.4 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 12 | 64.1 | 1.1% | 32 |  | FLOAT32/BFLOAT16/FLOAT32 | HiFi4 | L1_BLOCK-s/DRAM-i/L1_BLOCK-s | 0.0 | 2.5 | 2.5 | 25.9 | L1/NoC (far above floor) |
| BinaryNg | MUL | 24 | 61.6 | 1.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 0.3 | 0.3 | 203.1 | L1/NoC (far above floor) |
| Concat |  | 24 | 61.2 | 1.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i | 0.0 | 4.9 | 4.9 | 12.6 | L1/NoC (far above floor) |
| BinaryNg | MUL | 12 | 58.6 | 1.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 1.2 | 1.2 | 48.3 | L1/NoC (far above floor) |
| ShardedToInterleaved |  | 12 | 45.1 | 0.7% | 32 |  | FLOAT32/FLOAT32 |  | L1_BLOCK-s/L1-i | 0.0 | 2.4 | 2.4 | 18.6 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 12 | 44.7 | 0.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 3.6 | 3.6 | 12.3 | L1/NoC (far above floor) |
| Matmul |  | 12 | 43.8 | 0.7% | 32 | 8x256x128x32 | BFLOAT16/FLOAT32/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 1.2 | 3.8 | 3.8 | 11.4 | L1/NoC (far above floor) |
| Concat |  | 24 | 43.1 | 0.7% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i | 0.0 | 2.4 | 2.4 | 17.8 | L1/NoC (far above floor) |
| Slice |  | 24 | 41.7 | 0.7% | 120 |  | BFLOAT16/BFLOAT16 |  | L1-i/L1-i | 0.0 | 2.4 | 2.4 | 17.2 | L1/NoC (far above floor) |
| Reduce |  | 24 | 40.4 | 0.7% | 64 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 0.9 | 0.9 | 44.4 | L1/NoC (far above floor) |
| Matmul |  | 12 | 38.4 | 0.6% | 64 | 64x32x64x128 | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 2.4 | 2.1 | 2.4 | 15.8 | compute (far above floor) |
| InterleavedToSharded |  | 1 | 38.2 | 0.6% | 114 |  | FLOAT32/FLOAT32/FLOAT32 |  | L1-i/L1_HEIGHT-s/L1_HEIGHT-s | 0.0 | 0.9 | 0.9 | 44.4 | L1/NoC (far above floor) |
| BinaryNg | MUL | 12 | 36.8 | 0.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 0.8 | 0.8 | 48.5 | L1/NoC (far above floor) |
| Slice |  | 24 | 35.7 | 0.6% | 120 |  | BFLOAT16/BFLOAT16 |  | L1-i/L1-i | 0.0 | 2.1 | 2.1 | 16.8 | L1/NoC (far above floor) |
| BinaryNg | MUL | 1 | 35.4 | 0.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 5.7 | 5.7 | 6.2 | L1/NoC (far above floor) |
| BinaryNg | ADD | 12 | 34.8 | 0.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 0.9 | 0.9 | 38.3 | L1/NoC (far above floor) |
| BinaryNg | ADD | 12 | 34.2 | 0.6% | 120 |  | BFLOAT16/FLOAT32/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 2.1 | 2.1 | 16.0 | L1/NoC (far above floor) |
| InterleavedToSharded |  | 12 | 33.0 | 0.5% | 32 |  | FLOAT32/FLOAT32 |  | L1-i/L1_BLOCK-s | 0.0 | 2.4 | 2.4 | 13.6 | L1/NoC (far above floor) |
| LayerNorm | LAYERNORM | 1 | 32.6 | 0.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 1.6 | 1.6 | 20.2 | L1/NoC (far above floor) |
| Matmul |  | 12 | 31.7 | 0.5% | 64 | 64x32x128x128 | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 4.9 | 2.1 | 4.9 | 6.5 | compute (far above floor) |
| Slice |  | 24 | 31.0 | 0.5% | 32 |  | BFLOAT16/BFLOAT16 |  | L1-i/L1-i | 0.0 | 0.3 | 0.3 | 102.1 | L1/NoC (far above floor) |
| BinaryNg | SUB | 12 | 30.2 | 0.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 0.8 | 0.8 | 39.9 | L1/NoC (far above floor) |
| ReshapeCodegen |  | 1 | 29.4 | 0.5% | 32 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i | 0.0 | 2.0 | 2.0 | 14.4 | DRAM (far above floor) |
| NLPConcatHeads |  | 12 | 28.8 | 0.5% | 64 |  | BFLOAT16/BFLOAT16 |  | L1-i/L1-i | 0.0 | 1.2 | 1.2 | 23.7 | L1/NoC (far above floor) |
| Matmul |  | 1 | 25.2 | 0.4% | 88 | 256x1024x1024 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1.6 | 4.2 | 4.2 | 6.0 | DRAM (far above floor) |
| Unary | CLAMP_TSS | 11 | 22.0 | 0.4% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 1.1 | 1.1 | 19.8 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 1 | 18.3 | 0.3% | 8 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 6.1 | 6.1 | 3.0 | DRAM |
| Matmul |  | 1 | 17.2 | 0.3% | 96 | 256x1024x1536 | BFLOAT16/BFLOAT8_B/FLOAT32/FLOAT32 | HiFi2 | L1-i/DRAM-i/DRAM-i/DRAM-i | 2.4 | 6.8 | 6.8 | 2.5 | DRAM |
| Conv2d |  | 1 | 15.1 | 0.2% | 120 |  | FLOAT32/BFLOAT16/BFLOAT16 | HiFi4 | L1_HEIGHT-s/DRAM-i/L1_HEIGHT-s | 0.0 | 1.3 | 1.3 | 11.3 | L1/NoC (far above floor) |
| Matmul |  | 1 | 12.3 | 0.2% | 64 | 256x1536x512 | FLOAT32/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 1.2 | 5.2 | 5.2 | 2.4 | DRAM |
| ShardedToInterleaved |  | 1 | 12.3 | 0.2% | 114 |  | BFLOAT16/BFLOAT16 |  | L1_HEIGHT-s/L1-i | 0.0 | 1.6 | 1.6 | 7.6 | L1/NoC (far above floor) |
| Conv2d |  | 1 | 12.2 | 0.2% | 96 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1_HEIGHT-s/DRAM-i/L1_HEIGHT-s | 0.0 | 1.2 | 1.2 | 10.4 | L1/NoC (far above floor) |
| InterleavedToSharded |  | 1 | 11.9 | 0.2% | 86 |  | BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1_HEIGHT-s/L1_HEIGHT-s | 0.0 | 2.4 | 2.4 | 4.9 | L1/NoC (far above floor) |
| LayerNorm | LAYERNORM | 1 | 9.2 | 0.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 2.0 | 2.0 | 4.5 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 7.5 | 0.1% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 1.3 | 1.3 | 5.6 | DRAM (far above floor) |
| Halo |  | 1 | 4.6 | 0.1% | 86 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1_HEIGHT-s/L1_HEIGHT-s | 0.0 | 1.8 | 1.8 | 2.5 | L1/NoC |
| Untilize |  | 1 | 4.4 | 0.1% | 32 |  | FLOAT32/FLOAT32 | HiFi4 | L1-i/L1-i | 0.0 | 0.1 | 0.1 | 43.3 | L1/NoC (far above floor) |
| BinaryNg | ADD | 1 | 3.6 | 0.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 1.1 | 1.1 | 3.2 | L1/NoC (far above floor) |
| Unary | RELU | 1 | 3.2 | 0.1% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 2.0 | 2.0 | 1.6 | DRAM |
| RepeatCodegen |  | 1 | 2.5 | 0.0% | 2 |  | UINT16/UINT16 |  | DRAM-i/L1-i | 0.0 | 0.0 | 0.0 | 75.0 | L1/NoC (far above floor) |
| InterleavedToSharded |  | 1 | 2.4 | 0.0% | 32 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1_BLOCK-s | 0.0 | 1.1 | 1.1 | 2.3 | DRAM |
| ShardedToInterleaved |  | 1 | 2.1 | 0.0% | 86 |  | BFLOAT16/BFLOAT16 |  | L1_HEIGHT-s/DRAM-i | 0.0 | 1.1 | 1.1 | 2.0 | DRAM |
| Move |  | 1 | 1.8 | 0.0% | 86 |  | BFLOAT16/BFLOAT16/BFLOAT16 |  | L1_HEIGHT-s/L1_HEIGHT-s/L1_HEIGHT-s | 0.0 | 2.9 | 2.9 | 0.6 | L1/NoC |
| Halo |  | 1 | 1.8 | 0.0% | 114 |  | FLOAT32/FLOAT32 | HiFi4 | L1_HEIGHT-s/L1_HEIGHT-s | 0.0 | 0.9 | 0.9 | 2.0 | L1/NoC |
| Move |  | 1 | 0.9 | 0.0% | 114 |  | FLOAT32/FLOAT32/FLOAT32 |  | L1_HEIGHT-s/L1_HEIGHT-s/L1_HEIGHT-s | 0.0 | 1.4 | 1.4 | 0.6 | L1/NoC |
