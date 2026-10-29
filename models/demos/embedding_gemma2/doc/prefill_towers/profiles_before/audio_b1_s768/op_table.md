### audio b1_s768: 1090 ops, device kernel sum 11.295 ms, eager op-to-op gaps 29.403 ms, roofline floor 2.049 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GatherCodegen |  | 12 | 1193.2 | 10.6% | 120 |  | BFLOAT16/UINT16/BFLOAT16 |  | L1-i/L1-i/L1-i | 0.0 | 2.3 | 2.3 | 524.4 | L1/NoC (far above floor) |
| Matmul | HARDTANH | 24 | 948.1 | 8.4% | 88 | 768x4096x1024 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 466.0 | 227.1 | 466.0 | 2.0 | compute |
| Matmul | HARDTANH | 24 | 947.8 | 8.4% | 96 | 768x1024x4096 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 466.0 | 227.1 | 466.0 | 2.0 | compute |
| LayerNorm | RMSNORM | 96 | 744.7 | 6.6% | 64 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1_BLOCK-s/DRAM-i/L1_BLOCK-s | 0.0 | 29.5 | 29.5 | 25.2 | L1/NoC (far above floor) |
| BinaryNg | MUL | 60 | 705.7 | 6.2% | 120 |  | BFLOAT16/FLOAT32/FLOAT32 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 42.7 | 42.7 | 16.5 | L1/NoC (far above floor) |
| Unary | SILU | 24 | 681.6 | 6.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 29.1 | 29.1 | 23.4 | L1/NoC (far above floor) |
| Matmul |  | 48 | 642.7 | 5.7% | 88 | 768x768x1024 | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi2 | DRAM-i/L1-i/L1-i | 174.8 | 125.2 | 174.8 | 3.7 | compute (far above floor) |
| BinaryNg | ADD | 48 | 403.4 | 3.6% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 43.7 | 43.7 | 9.2 | L1/NoC (far above floor) |
| Matmul |  | 12 | 339.8 | 3.0% | 96 | 768x1024x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 174.8 | 85.6 | 174.8 | 1.9 | compute |
| ShardedToInterleaved |  | 96 | 305.3 | 2.7% | 64 |  | BFLOAT16/BFLOAT16 |  | L1_BLOCK-s/L1-i | 0.0 | 29.1 | 29.1 | 10.5 | L1/NoC (far above floor) |
| Matmul | HARDTANH | 12 | 277.7 | 2.5% | 88 | 768x1024x2048 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 116.5 | 57.7 | 116.5 | 2.4 | compute |
| ReshapeView |  | 1 | 253.2 | 2.2% | 120 |  | FLOAT32/FLOAT32 |  | L1-i/L1-i | 0.0 | 0.3 | 0.3 | 834.5 | L1/NoC (far above floor) |
| BinaryNg | ADD | 47 | 246.7 | 2.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 21.4 | 21.4 | 11.5 | L1/NoC (far above floor) |
| InterleavedToSharded |  | 95 | 230.9 | 2.0% | 64 |  | BFLOAT16/BFLOAT16 |  | L1-i/L1_BLOCK-s | 0.0 | 28.8 | 28.8 | 8.0 | L1/NoC (far above floor) |
| Unary | CLAMP_TSS | 61 | 219.2 | 1.9% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 18.5 | 18.5 | 11.8 | L1/NoC (far above floor) |
| Matmul | HARDTANH | 12 | 216.1 | 1.9% | 88 | 768x1024x1024 | FLOAT32/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 58.3 | 31.6 | 58.3 | 3.7 | compute (far above floor) |
| BinaryNg | MUL | 12 | 213.5 | 1.9% | 120 |  | BFLOAT16/FLOAT32/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 4.0 | 4.0 | 53.0 | L1/NoC (far above floor) |
| Matmul | HARDTANH | 12 | 165.3 | 1.5% | 88 | 768x1024x1024 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 58.3 | 29.8 | 58.3 | 2.8 | compute |
| BinaryNg | MUL | 12 | 159.0 | 1.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 5.5 | 5.5 | 29.1 | L1/NoC (far above floor) |
| Unary | SILU | 12 | 150.8 | 1.3% | 120 |  | FLOAT32/FLOAT32 | HiFi4 | L1-i/L1-i | 0.0 | 7.3 | 7.3 | 20.7 | L1/NoC (far above floor) |
| Concat |  | 24 | 132.2 | 1.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i | 0.0 | 14.6 | 14.6 | 9.1 | L1/NoC (far above floor) |
| InterleavedToSharded |  | 1 | 109.1 | 1.0% | 119 |  | FLOAT32/FLOAT32/FLOAT32 |  | L1-i/L1_HEIGHT-s/L1_HEIGHT-s | 0.0 | 2.6 | 2.6 | 42.0 | L1/NoC (far above floor) |
| BinaryNg | MUL | 12 | 105.7 | 0.9% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 3.6 | 3.6 | 29.0 | L1/NoC (far above floor) |
| BinaryNg | MUL | 1 | 100.5 | 0.9% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 17.1 | 17.1 | 5.9 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 12 | 94.0 | 0.8% | 64 |  | FLOAT32/BFLOAT16/FLOAT32 | HiFi4 | L1_BLOCK-s/DRAM-i/L1_BLOCK-s | 0.0 | 7.3 | 7.3 | 12.8 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 12 | 92.6 | 0.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 10.9 | 10.9 | 8.5 | L1/NoC (far above floor) |
| LayerNorm | LAYERNORM | 1 | 88.6 | 0.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 4.9 | 4.9 | 18.2 | L1/NoC (far above floor) |
| Concat |  | 24 | 85.0 | 0.8% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i | 0.0 | 7.3 | 7.3 | 11.7 | L1/NoC (far above floor) |
| Slice |  | 24 | 84.5 | 0.7% | 120 |  | BFLOAT16/BFLOAT16 |  | L1-i/L1-i | 0.0 | 7.0 | 7.0 | 12.1 | L1/NoC (far above floor) |
| Slice |  | 24 | 82.1 | 0.7% | 120 |  | BFLOAT16/BFLOAT16 |  | L1-i/L1-i | 0.0 | 7.3 | 7.3 | 11.3 | L1/NoC (far above floor) |
| Matmul |  | 12 | 81.4 | 0.7% | 120 | 192x32x64x128 | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 7.3 | 6.4 | 7.3 | 11.2 | compute (far above floor) |
| BinaryNg | ADD | 12 | 77.8 | 0.7% | 120 |  | BFLOAT16/FLOAT32/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 6.4 | 6.4 | 12.1 | L1/NoC (far above floor) |
| ShardedToInterleaved |  | 12 | 75.6 | 0.7% | 64 |  | FLOAT32/FLOAT32 |  | L1_BLOCK-s/L1-i | 0.0 | 7.3 | 7.3 | 10.4 | L1/NoC (far above floor) |
| RepeatCodegen |  | 12 | 73.1 | 0.6% | 2 |  | UINT16/UINT16 |  | DRAM-i/L1-i | 0.0 | 1.0 | 1.0 | 72.7 | L1/NoC (far above floor) |
| Matmul |  | 12 | 69.5 | 0.6% | 120 | 192x32x128x128 | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 14.6 | 6.4 | 14.6 | 4.8 | compute (far above floor) |
| BinaryNg | MUL | 24 | 68.7 | 0.6% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 0.3 | 0.3 | 226.6 | L1/NoC (far above floor) |
| Matmul |  | 12 | 66.9 | 0.6% | 48 | 8x768x128x32 | BFLOAT16/FLOAT32/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 3.6 | 5.3 | 5.3 | 12.5 | L1/NoC (far above floor) |
| Reduce |  | 24 | 59.2 | 0.5% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 2.7 | 2.7 | 21.7 | L1/NoC (far above floor) |
| BinaryNg | MUL | 12 | 57.8 | 0.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 2.3 | 2.3 | 25.4 | L1/NoC (far above floor) |
| ReshapeCodegen |  | 1 | 53.3 | 0.5% | 48 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/DRAM-i | 0.0 | 6.1 | 6.1 | 8.7 | DRAM (far above floor) |
| InterleavedToSharded |  | 12 | 53.2 | 0.5% | 64 |  | FLOAT32/FLOAT32 |  | L1-i/L1_BLOCK-s | 0.0 | 7.3 | 7.3 | 7.3 | L1/NoC (far above floor) |
| BinaryNg | ADD | 12 | 52.0 | 0.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 2.7 | 2.7 | 19.0 | L1/NoC (far above floor) |
| NLPConcatHeads |  | 12 | 46.7 | 0.4% | 120 |  | BFLOAT16/BFLOAT16 |  | L1-i/L1-i | 0.0 | 3.6 | 3.6 | 12.8 | L1/NoC (far above floor) |
| BinaryNg | SUB | 12 | 43.5 | 0.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 2.3 | 2.3 | 19.1 | L1/NoC (far above floor) |
| Unary | CLAMP_TSS | 11 | 39.6 | 0.4% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 3.3 | 3.3 | 11.9 | L1/NoC (far above floor) |
| Matmul |  | 1 | 37.2 | 0.3% | 88 | 768x1024x1024 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 4.9 | 8.3 | 8.3 | 4.5 | DRAM (far above floor) |
| Conv2d |  | 1 | 34.1 | 0.3% | 120 |  | FLOAT32/BFLOAT16/BFLOAT16 | HiFi4 | L1_HEIGHT-s/DRAM-i/L1_HEIGHT-s | 0.0 | 3.8 | 3.8 | 9.1 | L1/NoC (far above floor) |
| Matmul |  | 1 | 34.1 | 0.3% | 96 | 768x1024x1536 | BFLOAT16/BFLOAT8_B/FLOAT32/FLOAT32 | HiFi2 | L1-i/DRAM-i/DRAM-i/DRAM-i | 7.3 | 13.0 | 13.0 | 2.6 | DRAM |
| ShardedToInterleaved |  | 1 | 33.3 | 0.3% | 119 |  | BFLOAT16/BFLOAT16 |  | L1_HEIGHT-s/L1-i | 0.0 | 4.9 | 4.9 | 6.9 | L1/NoC (far above floor) |
| LayerNorm | RMSNORM | 1 | 30.8 | 0.3% | 24 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 18.4 | 18.4 | 1.7 | DRAM |
| Slice |  | 24 | 30.5 | 0.3% | 32 |  | BFLOAT16/BFLOAT16 |  | L1-i/L1-i | 0.0 | 0.3 | 0.3 | 100.6 | L1/NoC (far above floor) |
| Matmul |  | 1 | 30.2 | 0.3% | 64 | 768x1536x512 | FLOAT32/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | 3.6 | 12.4 | 12.4 | 2.4 | DRAM |
| InterleavedToSharded |  | 1 | 26.2 | 0.2% | 110 |  | BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1_HEIGHT-s/L1_HEIGHT-s | 0.0 | 7.3 | 7.3 | 3.6 | L1/NoC (far above floor) |
| Conv2d |  | 1 | 22.9 | 0.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1_HEIGHT-s/DRAM-i/L1_HEIGHT-s | 0.0 | 3.0 | 3.0 | 7.7 | L1/NoC (far above floor) |
| LayerNorm | LAYERNORM | 1 | 17.9 | 0.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 6.1 | 6.1 | 2.9 | DRAM |
| BinaryNg | MUL | 1 | 13.8 | 0.1% | 120 |  | FLOAT32/FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 4.0 | 4.0 | 3.5 | DRAM (far above floor) |
| Unary | RELU | 1 | 8.2 | 0.1% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 6.1 | 6.1 | 1.3 | DRAM |
| BinaryNg | ADD | 1 | 7.1 | 0.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 3.4 | 3.4 | 2.1 | L1/NoC |
| Halo |  | 1 | 6.6 | 0.1% | 110 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1_HEIGHT-s/L1_HEIGHT-s | 0.0 | 5.1 | 5.1 | 1.3 | L1/NoC |
| ShardedToInterleaved |  | 1 | 5.6 | 0.0% | 110 |  | BFLOAT16/BFLOAT16 |  | L1_HEIGHT-s/DRAM-i | 0.0 | 3.2 | 3.2 | 1.7 | DRAM |
| InterleavedToSharded |  | 1 | 5.1 | 0.0% | 64 |  | BFLOAT16/BFLOAT16 |  | DRAM-i/L1_BLOCK-s | 0.0 | 3.2 | 3.2 | 1.6 | DRAM |
| Untilize |  | 1 | 5.1 | 0.0% | 96 |  | FLOAT32/FLOAT32 | HiFi4 | L1-i/L1-i | 0.0 | 0.3 | 0.3 | 16.8 | L1/NoC (far above floor) |
| Halo |  | 1 | 3.7 | 0.0% | 119 |  | FLOAT32/FLOAT32 | HiFi4 | L1_HEIGHT-s/L1_HEIGHT-s | 0.0 | 2.5 | 2.5 | 1.5 | L1/NoC |
| Move |  | 1 | 3.5 | 0.0% | 110 |  | BFLOAT16/BFLOAT16/BFLOAT16 |  | L1_HEIGHT-s/L1_HEIGHT-s/L1_HEIGHT-s | 0.0 | 8.1 | 8.1 | 0.4 | L1/NoC |
| Move |  | 1 | 1.8 | 0.0% | 119 |  | FLOAT32/FLOAT32/FLOAT32 |  | L1_HEIGHT-s/L1_HEIGHT-s/L1_HEIGHT-s | 0.0 | 3.9 | 3.9 | 0.5 | L1/NoC |
