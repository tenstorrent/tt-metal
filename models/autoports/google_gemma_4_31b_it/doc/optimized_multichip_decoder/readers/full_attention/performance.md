# Isolated alternating reader trials

Isolated one-op device FW window, maximum across four devices, 32 traced repetitions; per-device weight-only bandwidth, not layer roofline.

| Role | Readers | Device us | Physical GB/s | Peak % | Spread % | PCC |
|---|---:|---:|---:|---:|---:|---:|
| self_attn.qk_proj | 1 | 50.836 | 274.11 | 53.54 | 0.20 | 1.00000000 |
| self_attn.qk_proj | 2 | 42.975 | 324.25 | 63.33 | 0.18 | 1.00000000 |
| self_attn.qk_proj | 3 | 49.936 | 279.05 | 54.50 | 0.33 | 1.00000000 |
| self_attn.o_proj | 1 | 48.377 | 268.23 | 52.39 | 0.04 | 1.00000000 |
| self_attn.o_proj | 2 | 40.919 | 317.12 | 61.94 | 0.29 | 1.00000000 |
| mlp.gate_proj | 1 | 63.186 | 269.54 | 52.64 | 0.29 | 1.00000000 |
| mlp.gate_proj | 2 | 45.247 | 376.40 | 73.52 | 0.37 | 1.00000000 |
| mlp.up_proj | 1 | 63.149 | 269.70 | 52.68 | 0.37 | 1.00000000 |
| mlp.up_proj | 2 | 45.296 | 376.00 | 73.44 | 0.18 | 1.00000000 |
| mlp.down_proj | 1 | 63.490 | 281.02 | 54.89 | 0.10 | 1.00000000 |
| mlp.down_proj | 2 | 52.211 | 341.73 | 66.74 | 0.23 | 1.00000000 |
