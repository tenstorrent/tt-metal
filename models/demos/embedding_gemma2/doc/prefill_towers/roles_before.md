| shape | role | op | M x K x N | calls | device us | per call | share | cores | fidelity | memory (in/w/out) | bound (model) | device / floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vision_b1_s2560 | up / gate | Matmul | 2560x768x3072 | 32 | 1746.9 | 54.6 | 10.8% | 120 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 1.5 |
| vision_b1_s2560 | down | MinimalMatmul | 2560x3072x768 | 16 | 1633.4 | 102.1 | 10.1% | 120 | HiFi2 | DRAM-i/DRAM-i/L1-i | compute | 2.8 |
| vision_b1_s2560 | qkv | Matmul | 2560x768x2304 | 16 | 686.9 | 42.9 | 4.3% | 120 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 1.6 |
| vision_b1_s2560 | o | Matmul | 2560x768x768 | 16 | 325.5 | 20.3 | 2.0% | 120 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 2.2 |
| vision_b1_s2560 | pool (one-hot) | Matmul | 288x2560x768 | 1 | 279.3 | 279.3 | 1.7% | 6 | HiFi4 | DRAM-i/L1-i/DRAM-i | compute | 40.9 |
| vision_b1_s2560 | patch_embed | Matmul | 2560x768x768 | 1 | 64.2 | 64.2 | 0.4% | 120 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | DRAM | 2.5 |
| vision_b1_s2560 | embed_vision | Matmul | 288x768x512 | 1 | 14.0 | 14.0 | 0.1% | 72 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | DRAM | 3.6 |
| audio_b1_s256 | ffn down | Matmul HARDTANH | 256x4096x1024 | 24 | 729.9 | 30.4 | 11.8% | 88 | HiFi2 | L1-i/DRAM-i/L1-i | DRAM | 3.4 |
| audio_b1_s256 | ffn up | Matmul HARDTANH | 256x1024x4096 | 24 | 568.4 | 23.7 | 9.2% | 96 | HiFi2 | L1-i/DRAM-i/L1-i | DRAM | 2.6 |
| audio_b1_s256 | qkv | Matmul | 256x1024x3072 | 12 | 213.3 | 17.8 | 3.4% | 96 | HiFi2 | L1-i/DRAM-i/L1-i | DRAM | 2.6 |
| audio_b1_s256 | lconv shift (S x S) | Matmul | 256x256x1024 | 48 | 191.9 | 4.0 | 3.1% | 88 | HiFi2 | DRAM-i/L1-i/L1-i | compute | 9.9 |
| audio_b1_s256 | lconv start | Matmul HARDTANH | 256x1024x2048 | 12 | 186.3 | 15.5 | 3.0% | 88 | HiFi2 | L1-i/DRAM-i/L1-i | DRAM | 3.4 |
| audio_b1_s256 | attn post / lconv end | Matmul HARDTANH | 256x1024x1024 | 12 | 125.1 | 10.4 | 2.0% | 88 | HiFi2 | L1-i/DRAM-i/L1-i | L1/NoC | 4.5 |
| audio_b1_s256 | attn post / lconv end | Matmul HARDTANH | 256x1024x1024 | 12 | 123.2 | 10.3 | 2.0% | 88 | HiFi2 | L1-i/DRAM-i/L1-i | DRAM | 4.5 |
| audio_b1_s256 | rel-pos Q E^T (block bmm) | Matmul | 8x256x128x32 | 12 | 60.4 | 5.0 | 1.0% | 16 | HiFi4 | L1-i/DRAM-i/L1-i | L1/NoC | 15.8 |
| audio_b1_s256 | band attention PV | Matmul | 64x32x64x128 | 12 | 38.9 | 3.2 | 0.6% | 64 | HiFi4 | L1-i/L1-i/L1-i | compute | 16.0 |
| audio_b1_s256 | band attention QK^T | Matmul | 64x32x128x128 | 12 | 31.7 | 2.6 | 0.5% | 64 | HiFi4 | L1-i/L1-i/L1-i | compute | 6.5 |
| audio_b1_s256 | input_proj (subsample) | Matmul | 256x1024x1024 | 1 | 25.2 | 25.2 | 0.4% | 88 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | DRAM | 6.0 |
| audio_b1_s256 | output_proj | Matmul | 256x1024x1536 | 1 | 17.1 | 17.1 | 0.3% | 96 | HiFi2 | L1-i/DRAM-i/DRAM-i/DRAM-i | DRAM | 2.5 |
| audio_b1_s256 | embed_audio | Matmul | 256x1536x512 | 1 | 12.3 | 12.3 | 0.2% | 64 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | DRAM | 2.4 |
| audio_b1_s768 | ffn down | Matmul HARDTANH | 768x4096x1024 | 24 | 948.1 | 39.5 | 8.4% | 88 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 2.0 |
| audio_b1_s768 | ffn up | Matmul HARDTANH | 768x1024x4096 | 24 | 947.8 | 39.5 | 8.4% | 96 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 2.0 |
| audio_b1_s768 | lconv shift (S x S) | Matmul | 768x768x1024 | 48 | 642.7 | 13.4 | 5.7% | 88 | HiFi2 | DRAM-i/L1-i/L1-i | compute | 3.7 |
| audio_b1_s768 | qkv | Matmul | 768x1024x3072 | 12 | 339.8 | 28.3 | 3.0% | 96 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 1.9 |
| audio_b1_s768 | lconv start | Matmul HARDTANH | 768x1024x2048 | 12 | 277.7 | 23.1 | 2.5% | 88 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 2.4 |
| audio_b1_s768 | attn post / lconv end | Matmul HARDTANH | 768x1024x1024 | 12 | 216.1 | 18.0 | 1.9% | 88 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 3.7 |
| audio_b1_s768 | attn post / lconv end | Matmul HARDTANH | 768x1024x1024 | 12 | 165.3 | 13.8 | 1.5% | 88 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 2.8 |
| audio_b1_s768 | band attention PV | Matmul | 192x32x64x128 | 12 | 81.4 | 6.8 | 0.7% | 120 | HiFi4 | L1-i/L1-i/L1-i | compute | 11.2 |
| audio_b1_s768 | band attention QK^T | Matmul | 192x32x128x128 | 12 | 69.5 | 5.8 | 0.6% | 120 | HiFi4 | L1-i/L1-i/L1-i | compute | 4.8 |
| audio_b1_s768 | rel-pos Q E^T (block bmm) | Matmul | 8x768x128x32 | 12 | 66.9 | 5.6 | 0.6% | 48 | HiFi4 | L1-i/DRAM-i/L1-i | L1/NoC | 12.5 |
| audio_b1_s768 | input_proj (subsample) | Matmul | 768x1024x1024 | 1 | 37.2 | 37.2 | 0.3% | 88 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | DRAM | 4.5 |
| audio_b1_s768 | output_proj | Matmul | 768x1024x1536 | 1 | 34.1 | 34.1 | 0.3% | 96 | HiFi2 | L1-i/DRAM-i/DRAM-i/DRAM-i | DRAM | 2.6 |
| audio_b1_s768 | embed_audio | Matmul | 768x1536x512 | 1 | 30.2 | 30.2 | 0.3% | 64 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | DRAM | 2.4 |
