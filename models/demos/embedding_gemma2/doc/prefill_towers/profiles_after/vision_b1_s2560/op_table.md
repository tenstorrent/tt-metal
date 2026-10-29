### vision b1_s2560: 319 ops, device kernel sum 14.999 ms, eager op-to-op gaps 0.563 ms, roofline floor 3.692 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 16 | 5043.5 | 33.6% | 120 | H12 S2560 keys2560 D64 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 970.9 | 139.5 | 970.9 | 5.2 | compute (far above floor) |
| BinaryNg | MUL | 16 | 2309.1 | 15.4% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 72.8 | 72.8 | 31.7 | L1/NoC (far above floor) |
| Matmul |  | 32 | 1742.6 | 11.6% | 120 | 2560x768x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 1165.1 | 217.4 | 1165.1 | 1.5 | compute |
| LayerNorm | RMSNORM | 64 | 1016.8 | 6.8% | 80 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 48.7 | 48.7 | 20.9 | L1/NoC (far above floor) |
| RotaryEmbeddingLlama |  | 32 | 970.2 | 6.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/DRAM-i/DRAM-i/L1-i | 0.0 | 65.4 | 65.4 | 14.8 | L1/NoC (far above floor) |
| Matmul |  | 16 | 940.6 | 6.3% | 120 | 2560x3072x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 582.5 | 108.7 | 582.5 | 1.6 | compute |
| Matmul |  | 16 | 687.5 | 4.6% | 120 | 2560x768x2304 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 436.9 | 83.0 | 436.9 | 1.6 | compute |
| LayerNorm | RMSNORM | 32 | 670.2 | 4.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 24.3 | 24.3 | 27.6 | L1/NoC (far above floor) |
| BinaryNg | ADD | 33 | 351.3 | 2.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 37.5 | 37.5 | 9.4 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 16 | 348.7 | 2.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 36.4 | 36.4 | 9.6 | L1/NoC (far above floor) |
| Matmul |  | 16 | 325.9 | 2.2% | 120 | 2560x768x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 145.6 | 31.7 | 145.6 | 2.2 | compute |
| LayerNorm | RMSNORM | 16 | 299.1 | 2.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 12.1 | 12.1 | 24.6 | L1/NoC (far above floor) |
| Matmul |  | 1 | 63.8 | 0.4% | 120 | 2560x768x768 | FLOAT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 18.2 | 25.3 | 25.3 | 2.5 | DRAM |
| Unary | MUL_UNARY_SFPU | 1 | 56.2 | 0.4% | 120 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 30.7 | 30.7 | 1.8 | DRAM |
| Embeddings |  | 2 | 47.8 | 0.3% | 80 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 1.6 | 1.6 | 30.7 | L1/NoC (far above floor) |
| BinaryNg | ADD | 1 | 45.4 | 0.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 7.8 | 7.8 | 5.8 | DRAM (far above floor) |
| Matmul |  | 1 | 21.8 | 0.1% | 108 | 288x2560x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/L1-i/DRAM-i | 6.8 | 5.0 | 6.8 | 3.2 | compute (far above floor) |
| BinaryNg | ADD | 1 | 17.1 | 0.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 8.4 | 8.4 | 2.0 | L1/NoC |
| Matmul |  | 1 | 14.3 | 0.1% | 72 | 288x768x512 | FLOAT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 1.4 | 3.8 | 3.8 | 3.7 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 1 | 10.7 | 0.1% | 9 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 3.5 | 3.5 | 3.1 | DRAM (far above floor) |
| BinaryNg | MUL | 1 | 6.8 | 0.0% | 120 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 3.5 | 3.5 | 2.0 | DRAM |
| BinaryNg | MUL | 2 | 5.7 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 1.3 | 1.3 | 4.5 | DRAM (far above floor) |
| BinaryNg | SUB | 1 | 2.3 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 3.6 | DRAM (far above floor) |
| Transpose |  | 1 | 1.5 | 0.0% | 80 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 2.4 | DRAM |
