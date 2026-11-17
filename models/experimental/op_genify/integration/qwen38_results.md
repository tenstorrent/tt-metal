| Per | Calls | Stock matmul ms | Generated ms | Generated vs stock | Roofline ms | Generated / roofline |
|---|---:|---:|---:|---|---:|---:|
| prefill M=128 | 256 | 16.20 | 16.58 | even | 8.40 | 1.97x |
| prefill M=512 | 256 | 31.04 | 29.14 | 1.07x faster | 15.01 | 1.94x |
| prefill M=1024 | 256 | 44.16 | 36.82 | 1.20x faster | 29.42 | 1.25x |
| prefill M=2048 | 256 | 80.82 | 63.29 | 1.28x faster | 58.66 | 1.08x |
| prefill M=4096 | 256 | 155.85 | 121.25 | 1.29x faster | 117.32 | 1.03x |
| decode token | 260 | 10.12 | 9.32 | 1.09x faster | 8.17 | 1.14x |

| Per | Stock model device ms | Generated model device ms | Change |
|---|---:|---:|---|
| prefill M=128 | 45.75 | 44.45 | even |
| prefill M=512 | 87.74 | 80.02 | 1.10x faster |
| prefill M=1024 | 131.54 | 111.04 | 1.18x faster |
| prefill M=2048 | 241.48 | 202.44 | 1.19x faster |
| prefill M=4096 | 477.86 | 386.33 | 1.24x faster |
| decode token | 22.38 | 21.73 | even |

| Per | Matmul | Calls | Shape per chip | Stock us | Generated us | Generated vs stock | Roofline us |
|---|---|---:|---|---:|---:|---|---:|
| prefill M=128 | GDN in_proj (qkv/z/b/a) | 48 | `[128 x 5120] . [5120 x 4160]` | 63.7 | 64.5 | even | 28.3 |
| prefill M=128 | GDN out_proj | 48 | `[128 x 1536] . [1536 x 5120]` | 27.3 | 23.4 | 1.17x faster | 12.4 |
| prefill M=128 | MLP gate_up | 48 | `[128 x 5120] . [5120 x 8704]` | 104.3 | 106.2 | even | 59.2 |
| prefill M=128 | MLP down | 48 | `[128 x 4352] . [4352 x 5120]` | 58.5 | 56.3 | 1.04x faster | 30.0 |
| prefill M=128 | Full-attn qkvg (q/gate/k/v) | 16 | `[128 x 5120] . [5120 x 3584]` | 52.9 | 82.2 | 1.55x slower | 27.2 |
| prefill M=128 | Full-attn o_proj | 16 | `[128 x 1536] . [1536 x 5120]` | 26.2 | 37.6 | 1.43x slower | 13.3 |
| prefill M=128 | MLP gate_up | 16 | `[128 x 5120] . [5120 x 8704]` | 113.3 | 108.6 | 1.04x faster | 62.1 |
| prefill M=128 | MLP down | 16 | `[128 x 4352] . [4352 x 5120]` | 58.8 | 56.4 | 1.04x faster | 32.5 |
| prefill M=512 | GDN in_proj (qkv/z/b/a) | 48 | `[512 x 5120] . [5120 x 4160]` | 109.1 | 102.0 | 1.07x faster | 51.2 |
| prefill M=512 | GDN out_proj | 48 | `[512 x 1536] . [1536 x 5120]` | 52.3 | 53.3 | even | 24.4 |
| prefill M=512 | MLP gate_up | 48 | `[512 x 5120] . [5120 x 8704]` | 214.1 | 198.0 | 1.08x faster | 107.2 |
| prefill M=512 | MLP down | 48 | `[512 x 4352] . [4352 x 5120]` | 111.3 | 103.3 | 1.08x faster | 53.6 |
| prefill M=512 | Full-attn qkvg (q/gate/k/v) | 16 | `[512 x 5120] . [5120 x 3584]` | 101.3 | 97.5 | 1.04x faster | 44.1 |
| prefill M=512 | Full-attn o_proj | 16 | `[512 x 1536] . [1536 x 5120]` | 52.4 | 53.8 | even | 24.4 |
| prefill M=512 | MLP gate_up | 16 | `[512 x 5120] . [5120 x 8704]` | 213.8 | 196.7 | 1.09x faster | 107.2 |
| prefill M=512 | MLP down | 16 | `[512 x 4352] . [4352 x 5120]` | 112.0 | 103.6 | 1.08x faster | 53.6 |
| prefill M=1024 | GDN in_proj (qkv/z/b/a) | 48 | `[1024 x 5120] . [5120 x 4160]` | 138.7 | 138.6 | even | 102.4 |
| prefill M=1024 | GDN out_proj | 48 | `[1024 x 1536] . [1536 x 5120]` | 83.6 | 78.2 | 1.07x faster | 39.2 |
| prefill M=1024 | MLP gate_up | 48 | `[1024 x 5120] . [5120 x 8704]` | 312.3 | 229.0 | 1.36x faster | 214.4 |
| prefill M=1024 | MLP down | 48 | `[1024 x 4352] . [4352 x 5120]` | 156.9 | 134.6 | 1.17x faster | 107.2 |
| prefill M=1024 | Full-attn qkvg (q/gate/k/v) | 16 | `[1024 x 5120] . [5120 x 3584]` | 133.4 | 120.0 | 1.11x faster | 88.3 |
| prefill M=1024 | Full-attn o_proj | 16 | `[1024 x 1536] . [1536 x 5120]` | 83.0 | 76.7 | 1.08x faster | 39.2 |
| prefill M=1024 | MLP gate_up | 16 | `[1024 x 5120] . [5120 x 8704]` | 312.9 | 228.7 | 1.37x faster | 214.4 |
| prefill M=1024 | MLP down | 16 | `[1024 x 4352] . [4352 x 5120]` | 156.2 | 134.7 | 1.16x faster | 107.2 |
| prefill M=2048 | GDN in_proj (qkv/z/b/a) | 48 | `[2048 x 5120] . [5120 x 4160]` | 255.2 | 229.6 | 1.11x faster | 204.9 |
| prefill M=2048 | GDN out_proj | 48 | `[2048 x 1536] . [1536 x 5120]` | 156.0 | 137.1 | 1.14x faster | 75.7 |
| prefill M=2048 | MLP gate_up | 48 | `[2048 x 5120] . [5120 x 8704]` | 573.4 | 395.2 | 1.45x faster | 428.7 |
| prefill M=2048 | MLP down | 48 | `[2048 x 4352] . [4352 x 5120]` | 282.8 | 235.5 | 1.20x faster | 214.4 |
| prefill M=2048 | Full-attn qkvg (q/gate/k/v) | 16 | `[2048 x 5120] . [5120 x 3584]` | 241.4 | 198.9 | 1.21x faster | 176.5 |
| prefill M=2048 | Full-attn o_proj | 16 | `[2048 x 1536] . [1536 x 5120]` | 157.0 | 135.5 | 1.16x faster | 75.7 |
| prefill M=2048 | MLP gate_up | 16 | `[2048 x 5120] . [5120 x 8704]` | 564.1 | 394.7 | 1.43x faster | 428.7 |
| prefill M=2048 | MLP down | 16 | `[2048 x 4352] . [4352 x 5120]` | 286.9 | 234.8 | 1.22x faster | 214.4 |
| prefill M=4096 | GDN in_proj (qkv/z/b/a) | 48 | `[4096 x 5120] . [5120 x 4160]` | 514.5 | 415.5 | 1.24x faster | 409.8 |
| prefill M=4096 | GDN out_proj | 48 | `[4096 x 1536] . [1536 x 5120]` | 296.5 | 226.1 | 1.31x faster | 151.3 |
| prefill M=4096 | MLP gate_up | 48 | `[4096 x 5120] . [5120 x 8704]` | 1128.5 | 814.7 | 1.39x faster | 857.4 |
| prefill M=4096 | MLP down | 48 | `[4096 x 4352] . [4352 x 5120]` | 533.0 | 450.1 | 1.18x faster | 428.7 |
| prefill M=4096 | Full-attn qkvg (q/gate/k/v) | 16 | `[4096 x 5120] . [5120 x 3584]` | 366.5 | 371.0 | even | 353.1 |
| prefill M=4096 | Full-attn o_proj | 16 | `[4096 x 1536] . [1536 x 5120]` | 298.3 | 224.7 | 1.33x faster | 151.3 |
| prefill M=4096 | MLP gate_up | 16 | `[4096 x 5120] . [5120 x 8704]` | 1125.9 | 813.9 | 1.38x faster | 857.4 |
| prefill M=4096 | MLP down | 16 | `[4096 x 4352] . [4352 x 5120]` | 532.2 | 449.5 | 1.18x faster | 428.7 |
| decode token | GDN in_proj (qkv/z/b/a) | 48 | `[1 x 5120] . [5120 x 4608]` | 37.5 | 34.8 | 1.08x faster | 28.8 |
| decode token | GDN out_proj | 48 | `[1 x 1536] . [1536 x 5120]` | 15.2 | 11.5 | 1.32x faster | 9.6 |
| decode token | MLP gate_up | 48 | `[1 x 5120] . [5120 x 9216]` | 63.1 | 62.4 | even | 57.6 |
| decode token | MLP down | 48 | `[1 x 4352] . [4352 x 5120]` | 36.6 | 31.0 | 1.18x faster | 27.2 |
| decode token | Full-attn qkvg (q/gate/k/v) | 16 | `[1 x 5120] . [5120 x 3584]` | 29.1 | 32.4 | 1.11x slower | 22.4 |
| decode token | Full-attn o_proj | 16 | `[1 x 1536] . [1536 x 5120]` | 15.2 | 11.5 | 1.32x faster | 9.6 |
| decode token | MLP gate_up | 16 | `[1 x 5120] . [5120 x 9216]` | 63.4 | 62.2 | even | 57.6 |
| decode token | MLP down | 16 | `[1 x 4352] . [4352 x 5120]` | 37.1 | 31.1 | 1.19x faster | 27.2 |
| decode token | LM head (vocab chunk) | 1 | `[1 x 5120] . [5120 x 16384]` | 127.1 | 108.4 | 1.17x faster | 102.4 |
| decode token | LM head (vocab chunk) | 1 | `[1 x 5120] . [5120 x 16384]` | 126.7 | 107.9 | 1.17x faster | 102.4 |
| decode token | LM head (vocab chunk) | 1 | `[1 x 5120] . [5120 x 16384]` | 126.9 | 107.9 | 1.18x faster | 102.4 |
| decode token | LM head (vocab chunk) | 1 | `[1 x 5120] . [5120 x 12928]` | 104.6 | 102.5 | even | 80.8 |
