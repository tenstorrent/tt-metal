| shape | role | op | M x K x N | calls | device us | per call | share | cores | fidelity | memory (in/w/out) | bound (model) | device / floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vision_b1_s2560 | up / gate | Matmul | 2560x768x3072 | 32 | 1742.6 | 54.5 | 11.6% | 120 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 1.5 |
| vision_b1_s2560 | down | Matmul | 2560x3072x768 | 16 | 940.6 | 58.8 | 6.3% | 120 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 1.6 |
| vision_b1_s2560 | qkv | Matmul | 2560x768x2304 | 16 | 687.5 | 43.0 | 4.6% | 120 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 1.6 |
| vision_b1_s2560 | o | Matmul | 2560x768x768 | 16 | 325.9 | 20.4 | 2.2% | 120 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 2.2 |
| vision_b1_s2560 | patch_embed | Matmul | 2560x768x768 | 1 | 63.8 | 63.8 | 0.4% | 120 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | DRAM | 2.5 |
| vision_b1_s2560 | pool (one-hot) | Matmul | 288x2560x768 | 1 | 21.8 | 21.8 | 0.1% | 108 | HiFi4 | DRAM-i/L1-i/DRAM-i | compute | 3.2 |
| vision_b1_s2560 | embed_vision | Matmul | 288x768x512 | 1 | 14.3 | 14.3 | 0.1% | 72 | HiFi4 | DRAM-i/DRAM-i/DRAM-i | DRAM | 3.7 |
| audio_b1_s256 | ffn down | Matmul HARDTANH | 256x4096x1024 | 24 | 640.3 | 26.7 | 10.6% | 88 | HiFi2 | L1-i/DRAM-i/L1-i | DRAM | 3.0 |
| audio_b1_s256 | ffn up | Matmul HARDTANH | 256x1024x4096 | 24 | 568.6 | 23.7 | 9.4% | 96 | HiFi2 | L1-i/DRAM-i/L1-i | DRAM | 2.6 |
| audio_b1_s256 | qkv | Matmul | 256x1024x3072 | 12 | 213.7 | 17.8 | 3.5% | 96 | HiFi2 | L1-i/DRAM-i/L1-i | DRAM | 2.6 |
| audio_b1_s256 | lconv shift (S x S) | Matmul | 256x256x1024 | 48 | 192.3 | 4.0 | 3.2% | 88 | HiFi2 | DRAM-i/L1-i/L1-i | compute | 9.9 |
| audio_b1_s256 | lconv start | Matmul HARDTANH | 256x1024x2048 | 12 | 186.1 | 15.5 | 3.1% | 88 | HiFi2 | L1-i/DRAM-i/L1-i | DRAM | 3.4 |
| audio_b1_s256 | attn post / lconv end | Matmul HARDTANH | 256x1024x1024 | 12 | 125.1 | 10.4 | 2.1% | 88 | HiFi2 | L1-i/DRAM-i/L1-i | L1/NoC | 4.5 |
| audio_b1_s256 | attn post / lconv end | Matmul HARDTANH | 256x1024x1024 | 12 | 123.9 | 10.3 | 2.0% | 88 | HiFi2 | L1-i/DRAM-i/L1-i | DRAM | 4.5 |
| audio_b1_s256 | rel-pos Q E^T (block bmm) | Matmul | 8x256x128x32 | 12 | 43.8 | 3.6 | 0.7% | 32 | HiFi4 | L1-i/DRAM-i/L1-i | L1/NoC | 11.4 |
| audio_b1_s256 | band attention PV | Matmul | 64x32x64x128 | 12 | 38.4 | 3.2 | 0.6% | 64 | HiFi4 | L1-i/L1-i/L1-i | compute | 15.8 |
| audio_b1_s256 | band attention QK^T | Matmul | 64x32x128x128 | 12 | 31.7 | 2.6 | 0.5% | 64 | HiFi4 | L1-i/L1-i/L1-i | compute | 6.5 |
| audio_b1_s256 | input_proj (subsample) | Matmul | 256x1024x1024 | 1 | 25.2 | 25.2 | 0.4% | 88 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | DRAM | 6.0 |
| audio_b1_s256 | output_proj | Matmul | 256x1024x1536 | 1 | 17.2 | 17.2 | 0.3% | 96 | HiFi2 | L1-i/DRAM-i/DRAM-i/DRAM-i | DRAM | 2.5 |
| audio_b1_s256 | embed_audio | Matmul | 256x1536x512 | 1 | 12.3 | 12.3 | 0.2% | 64 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | DRAM | 2.4 |
| audio_b1_s768 | ffn up | Matmul HARDTANH | 768x1024x4096 | 24 | 947.3 | 39.5 | 8.5% | 96 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 2.0 |
| audio_b1_s768 | ffn down | Matmul HARDTANH | 768x4096x1024 | 24 | 908.8 | 37.9 | 8.1% | 88 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 2.0 |
| audio_b1_s768 | lconv shift (S x S) | Matmul | 768x768x1024 | 48 | 644.4 | 13.4 | 5.8% | 88 | HiFi2 | DRAM-i/L1-i/L1-i | compute | 3.7 |
| audio_b1_s768 | qkv | Matmul | 768x1024x3072 | 12 | 340.1 | 28.3 | 3.0% | 96 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 1.9 |
| audio_b1_s768 | lconv start | Matmul HARDTANH | 768x1024x2048 | 12 | 277.5 | 23.1 | 2.5% | 88 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 2.4 |
| audio_b1_s768 | attn post / lconv end | Matmul HARDTANH | 768x1024x1024 | 12 | 217.7 | 18.1 | 1.9% | 88 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 3.7 |
| audio_b1_s768 | attn post / lconv end | Matmul HARDTANH | 768x1024x1024 | 12 | 165.1 | 13.8 | 1.5% | 88 | HiFi2 | L1-i/DRAM-i/L1-i | compute | 2.8 |
| audio_b1_s768 | band attention PV | Matmul | 192x32x64x128 | 12 | 81.9 | 6.8 | 0.7% | 120 | HiFi4 | L1-i/L1-i/L1-i | compute | 11.3 |
| audio_b1_s768 | band attention QK^T | Matmul | 192x32x128x128 | 12 | 70.4 | 5.9 | 0.6% | 120 | HiFi4 | L1-i/L1-i/L1-i | compute | 4.8 |
| audio_b1_s768 | rel-pos Q E^T (block bmm) | Matmul | 8x768x128x32 | 12 | 66.8 | 5.6 | 0.6% | 48 | HiFi4 | L1-i/DRAM-i/L1-i | L1/NoC | 12.5 |
| audio_b1_s768 | input_proj (subsample) | Matmul | 768x1024x1024 | 1 | 38.0 | 38.0 | 0.3% | 88 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | DRAM | 4.6 |
| audio_b1_s768 | output_proj | Matmul | 768x1024x1536 | 1 | 34.0 | 34.0 | 0.3% | 96 | HiFi2 | L1-i/DRAM-i/DRAM-i/DRAM-i | DRAM | 2.6 |
| audio_b1_s768 | embed_audio | Matmul | 768x1536x512 | 1 | 29.5 | 29.5 | 0.3% | 64 | HiFi2 | DRAM-i/DRAM-i/DRAM-i | DRAM | 2.4 |
