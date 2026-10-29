### vision b1_s2560: 320 ops, device kernel sum 16.114 ms, eager op-to-op gaps 0.410 ms, roofline floor 4.298 ms (clock 1350 MHz, 120 cores)

| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SDPA |  | 16 | 5037.9 | 31.3% | 120 | H12 S2560 keys2560 D64 | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT4_B/BFLOAT16 | HiFi2 | L1-i/L1-i/L1-i/DRAM-i/L1-i | 970.9 | 139.5 | 970.9 | 5.2 | compute (far above floor) |
| BinaryNg | MUL | 16 | 2352.6 | 14.6% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/DRAM-i | 0.0 | 540.1 | 540.1 | 4.4 | L1/NoC (far above floor) |
| Matmul |  | 32 | 1746.9 | 10.8% | 120 | 2560x768x3072 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 1165.1 | 217.4 | 1165.1 | 1.5 | compute |
| MinimalMatmul |  | 16 | 1633.4 | 10.1% | 120 | 2560x3072x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | DRAM-i/DRAM-i/L1-i | 582.5 | 575.9 | 582.5 | 2.8 | compute |
| LayerNorm | RMSNORM | 63 | 999.4 | 6.2% | 80 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 48.0 | 48.0 | 20.8 | L1/NoC (far above floor) |
| RotaryEmbeddingLlama |  | 32 | 968.5 | 6.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/DRAM-i/DRAM-i/L1-i | 0.0 | 65.4 | 65.4 | 14.8 | L1/NoC (far above floor) |
| Matmul |  | 16 | 686.9 | 4.3% | 120 | 2560x768x2304 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 436.9 | 83.0 | 436.9 | 1.6 | compute |
| LayerNorm | RMSNORM | 32 | 670.3 | 4.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/DRAM-i/L1-i | 0.0 | 24.3 | 24.3 | 27.6 | L1/NoC (far above floor) |
| NlpCreateHeads |  | 16 | 350.7 | 2.2% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16/BFLOAT16 |  | L1-i/L1-i/L1-i/L1-i | 0.0 | 36.4 | 36.4 | 9.6 | L1/NoC (far above floor) |
| BinaryNg | ADD | 31 | 328.2 | 2.0% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i/L1-i | 0.0 | 35.3 | 35.3 | 9.3 | L1/NoC (far above floor) |
| Matmul |  | 16 | 325.5 | 2.0% | 120 | 2560x768x768 | BFLOAT16/BFLOAT8_B/BFLOAT16 | HiFi2 | L1-i/DRAM-i/L1-i | 145.6 | 31.7 | 145.6 | 2.2 | compute |
| LayerNorm | RMSNORM | 16 | 299.2 | 1.9% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | L1-i/L1-i | 0.0 | 12.1 | 12.1 | 24.7 | L1/NoC (far above floor) |
| Matmul |  | 1 | 279.3 | 1.7% | 6 | 288x2560x768 | BFLOAT16/BFLOAT16/FLOAT32 | HiFi4 | DRAM-i/L1-i/DRAM-i | 6.8 | 5.0 | 6.8 | 40.9 | compute (far above floor) |
| BinaryNg | ADD | 2 | 77.5 | 0.5% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 46.1 | 46.1 | 1.7 | DRAM |
| Matmul |  | 1 | 64.2 | 0.4% | 120 | 2560x768x768 | FLOAT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 18.2 | 25.3 | 25.3 | 2.5 | DRAM |
| BinaryNg | MUL | 2 | 61.0 | 0.4% | 120 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 61.4 | 61.4 | 1.0 | DRAM |
| Embeddings |  | 2 | 58.7 | 0.4% | 80 |  | UINT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 30.8 | 30.8 | 1.9 | DRAM |
| BinaryNg | SUB | 1 | 54.8 | 0.3% | 120 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 30.7 | 30.7 | 1.8 | DRAM |
| BinaryNg | ADD | 1 | 45.8 | 0.3% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT4_B | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 0.0 | 7.8 | 7.8 | 5.8 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 1 | 21.7 | 0.1% | 80 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/L1-i | 0.0 | 8.1 | 8.1 | 2.7 | DRAM |
| BinaryNg | ADD | 1 | 17.4 | 0.1% | 120 |  | BFLOAT16/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/L1-i/L1-i | 0.0 | 8.4 | 8.4 | 2.1 | L1/NoC |
| Matmul |  | 1 | 14.0 | 0.1% | 72 | 288x768x512 | FLOAT32/BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | 1.4 | 3.8 | 3.8 | 3.6 | DRAM (far above floor) |
| LayerNorm | RMSNORM | 1 | 10.8 | 0.1% | 9 |  | FLOAT32/FLOAT32 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 3.5 | 3.5 | 3.1 | DRAM (far above floor) |
| BinaryNg | MUL | 2 | 5.8 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 1.3 | 1.3 | 4.5 | DRAM (far above floor) |
| BinaryNg | SUB | 1 | 2.2 | 0.0% | 120 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 3.5 | DRAM (far above floor) |
| Transpose |  | 1 | 1.5 | 0.0% | 80 |  | BFLOAT16/BFLOAT16 | HiFi4 | DRAM-i/DRAM-i | 0.0 | 0.6 | 0.6 | 2.4 | DRAM |
