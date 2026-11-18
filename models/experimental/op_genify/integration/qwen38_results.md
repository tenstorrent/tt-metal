| Per | Calls | Stock matmul ms | Generated ms | Generated vs stock | Roofline ms | Generated / roofline |
|---|---:|---:|---:|---|---:|---:|
| prefill M=128 | 256 | 16.20 | 11.97 | 1.35x faster | 8.40 | 1.42x |
| prefill M=512 | 256 | 31.04 | 22.80 | 1.36x faster | 15.01 | 1.52x |
| prefill M=1024 | 256 | 44.16 | 34.80 | 1.27x faster | 29.42 | 1.18x |
| prefill M=2048 | 256 | 80.82 | 61.69 | 1.31x faster | 58.66 | 1.05x |
| prefill M=4096 | 256 | 155.85 | 119.80 | 1.30x faster | 117.32 | 1.02x |
| decode token | 260 | 10.12 | 8.85 | 1.14x faster | 8.17 | 1.08x |

| Per | Stock model device ms | Generated model device ms | Change |
|---|---:|---:|---|
| prefill M=128 | 45.75 | 40.74 | 1.12x faster |
| prefill M=512 | 87.74 | 73.83 | 1.19x faster |
| prefill M=1024 | 131.54 | 110.08 | 1.19x faster |
| prefill M=2048 | 241.48 | 196.44 | 1.23x faster |
| prefill M=4096 | 477.86 | 389.14 | 1.23x faster |
| decode token | 22.38 | 21.04 | 1.06x faster |

| Per | Matmul | Calls | Shape per chip | Stock us | Generated us | Generated vs stock | Roofline us |
|---|---|---:|---|---:|---:|---|---:|
| prefill M=128 | GDN in_proj (qkv/z/b/a) | 48 | `[128 x 5120] . [5120 x 4160]` | 63.7 | 50.4 | 1.26x faster | 28.3 |
| prefill M=128 | GDN out_proj | 48 | `[128 x 1536] . [1536 x 5120]` | 27.3 | 20.0 | 1.36x faster | 12.4 |
| prefill M=128 | MLP gate_up | 48 | `[128 x 5120] . [5120 x 8704]` | 104.3 | 72.2 | 1.44x faster | 59.2 |
| prefill M=128 | MLP down | 48 | `[128 x 4352] . [4352 x 5120]` | 58.5 | 44.5 | 1.32x faster | 30.0 |
| prefill M=128 | Full-attn qkvg (q/gate/k/v) | 16 | `[128 x 5120] . [5120 x 3584]` | 52.9 | 46.0 | 1.15x faster | 27.2 |
| prefill M=128 | Full-attn o_proj | 16 | `[128 x 1536] . [1536 x 5120]` | 26.2 | 19.5 | 1.35x faster | 13.3 |
| prefill M=128 | MLP gate_up | 16 | `[128 x 5120] . [5120 x 8704]` | 113.3 | 74.9 | 1.51x faster | 62.1 |
| prefill M=128 | MLP down | 16 | `[128 x 4352] . [4352 x 5120]` | 58.8 | 46.6 | 1.26x faster | 32.5 |
| prefill M=512 | GDN in_proj (qkv/z/b/a) | 48 | `[512 x 5120] . [5120 x 4160]` | 109.1 | 75.4 | 1.45x faster | 51.2 |
| prefill M=512 | GDN out_proj | 48 | `[512 x 1536] . [1536 x 5120]` | 52.3 | 44.0 | 1.19x faster | 24.4 |
| prefill M=512 | MLP gate_up | 48 | `[512 x 5120] . [5120 x 8704]` | 214.1 | 157.0 | 1.36x faster | 107.2 |
| prefill M=512 | MLP down | 48 | `[512 x 4352] . [4352 x 5120]` | 111.3 | 80.9 | 1.38x faster | 53.6 |
| prefill M=512 | Full-attn qkvg (q/gate/k/v) | 16 | `[512 x 5120] . [5120 x 3584]` | 101.3 | 72.4 | 1.40x faster | 44.1 |
| prefill M=512 | Full-attn o_proj | 16 | `[512 x 1536] . [1536 x 5120]` | 52.4 | 44.2 | 1.19x faster | 24.4 |
| prefill M=512 | MLP gate_up | 16 | `[512 x 5120] . [5120 x 8704]` | 213.8 | 155.4 | 1.38x faster | 107.2 |
| prefill M=512 | MLP down | 16 | `[512 x 4352] . [4352 x 5120]` | 112.0 | 80.6 | 1.39x faster | 53.6 |
| prefill M=1024 | GDN in_proj (qkv/z/b/a) | 48 | `[1024 x 5120] . [5120 x 4160]` | 138.7 | 124.8 | 1.11x faster | 102.4 |
| prefill M=1024 | GDN out_proj | 48 | `[1024 x 1536] . [1536 x 5120]` | 83.6 | 77.9 | 1.07x faster | 39.2 |
| prefill M=1024 | MLP gate_up | 48 | `[1024 x 5120] . [5120 x 8704]` | 312.3 | 220.8 | 1.41x faster | 214.4 |
| prefill M=1024 | MLP down | 48 | `[1024 x 4352] . [4352 x 5120]` | 156.9 | 124.8 | 1.26x faster | 107.2 |
| prefill M=1024 | Full-attn qkvg (q/gate/k/v) | 16 | `[1024 x 5120] . [5120 x 3584]` | 133.4 | 106.8 | 1.25x faster | 88.3 |
| prefill M=1024 | Full-attn o_proj | 16 | `[1024 x 1536] . [1536 x 5120]` | 83.0 | 76.5 | 1.08x faster | 39.2 |
| prefill M=1024 | MLP gate_up | 16 | `[1024 x 5120] . [5120 x 8704]` | 312.9 | 221.5 | 1.41x faster | 214.4 |
| prefill M=1024 | MLP down | 16 | `[1024 x 4352] . [4352 x 5120]` | 156.2 | 125.1 | 1.25x faster | 107.2 |
| prefill M=2048 | GDN in_proj (qkv/z/b/a) | 48 | `[2048 x 5120] . [5120 x 4160]` | 255.2 | 222.1 | 1.15x faster | 204.9 |
| prefill M=2048 | GDN out_proj | 48 | `[2048 x 1536] . [1536 x 5120]` | 156.0 | 130.3 | 1.20x faster | 75.7 |
| prefill M=2048 | MLP gate_up | 48 | `[2048 x 5120] . [5120 x 8704]` | 573.4 | 395.2 | 1.45x faster | 428.7 |
| prefill M=2048 | MLP down | 48 | `[2048 x 4352] . [4352 x 5120]` | 282.8 | 225.7 | 1.25x faster | 214.4 |
| prefill M=2048 | Full-attn qkvg (q/gate/k/v) | 16 | `[2048 x 5120] . [5120 x 3584]` | 241.4 | 187.3 | 1.29x faster | 176.5 |
| prefill M=2048 | Full-attn o_proj | 16 | `[2048 x 1536] . [1536 x 5120]` | 157.0 | 127.4 | 1.23x faster | 75.7 |
| prefill M=2048 | MLP gate_up | 16 | `[2048 x 5120] . [5120 x 8704]` | 564.1 | 394.9 | 1.43x faster | 428.7 |
| prefill M=2048 | MLP down | 16 | `[2048 x 4352] . [4352 x 5120]` | 286.9 | 226.0 | 1.27x faster | 214.4 |
| prefill M=4096 | GDN in_proj (qkv/z/b/a) | 48 | `[4096 x 5120] . [5120 x 4160]` | 514.5 | 402.9 | 1.28x faster | 409.8 |
| prefill M=4096 | GDN out_proj | 48 | `[4096 x 1536] . [1536 x 5120]` | 296.5 | 224.5 | 1.32x faster | 151.3 |
| prefill M=4096 | MLP gate_up | 48 | `[4096 x 5120] . [5120 x 8704]` | 1128.5 | 814.0 | 1.39x faster | 857.4 |
| prefill M=4096 | MLP down | 48 | `[4096 x 4352] . [4352 x 5120]` | 533.0 | 440.1 | 1.21x faster | 428.7 |
| prefill M=4096 | Full-attn qkvg (q/gate/k/v) | 16 | `[4096 x 5120] . [5120 x 3584]` | 366.5 | 365.3 | even | 353.1 |
| prefill M=4096 | Full-attn o_proj | 16 | `[4096 x 1536] . [1536 x 5120]` | 298.3 | 224.6 | 1.33x faster | 151.3 |
| prefill M=4096 | MLP gate_up | 16 | `[4096 x 5120] . [5120 x 8704]` | 1125.9 | 813.6 | 1.38x faster | 857.4 |
| prefill M=4096 | MLP down | 16 | `[4096 x 4352] . [4352 x 5120]` | 532.2 | 439.9 | 1.21x faster | 428.7 |
| decode token | GDN in_proj (qkv/z/b/a) | 48 | `[1 x 5120] . [5120 x 4608]` | 37.5 | 31.3 | 1.20x faster | 28.8 |
| decode token | GDN out_proj | 48 | `[1 x 1536] . [1536 x 5120]` | 15.2 | 11.5 | 1.32x faster | 9.6 |
| decode token | MLP gate_up | 48 | `[1 x 5120] . [5120 x 9216]` | 63.1 | 59.9 | 1.05x faster | 57.6 |
| decode token | MLP down | 48 | `[1 x 4352] . [4352 x 5120]` | 36.6 | 30.8 | 1.19x faster | 27.2 |
| decode token | Full-attn qkvg (q/gate/k/v) | 16 | `[1 x 5120] . [5120 x 3584]` | 29.1 | 24.8 | 1.17x faster | 22.4 |
| decode token | Full-attn o_proj | 16 | `[1 x 1536] . [1536 x 5120]` | 15.2 | 11.6 | 1.32x faster | 9.6 |
| decode token | MLP gate_up | 16 | `[1 x 5120] . [5120 x 9216]` | 63.4 | 60.1 | 1.06x faster | 57.6 |
| decode token | MLP down | 16 | `[1 x 4352] . [4352 x 5120]` | 37.1 | 30.7 | 1.21x faster | 27.2 |
| decode token | LM head (vocab chunk) | 1 | `[1 x 5120] . [5120 x 16384]` | 127.1 | 108.1 | 1.18x faster | 102.4 |
| decode token | LM head (vocab chunk) | 1 | `[1 x 5120] . [5120 x 16384]` | 126.7 | 107.7 | 1.18x faster | 102.4 |
| decode token | LM head (vocab chunk) | 1 | `[1 x 5120] . [5120 x 16384]` | 126.9 | 107.1 | 1.18x faster | 102.4 |
| decode token | LM head (vocab chunk) | 1 | `[1 x 5120] . [5120 x 12928]` | 104.6 | 87.0 | 1.20x faster | 80.8 |
