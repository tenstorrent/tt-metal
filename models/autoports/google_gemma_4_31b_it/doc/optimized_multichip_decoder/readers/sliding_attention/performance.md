# Isolated alternating reader trials

Isolated one-op device FW window, maximum across four devices, 32 traced repetitions; per-device weight-only bandwidth, not layer roofline.

| Role | Readers | Device us | Physical GB/s | Peak % | Spread % | PCC |
|---|---:|---:|---:|---:|---:|---:|
| self_attn.qkv_proj | 1 | 48.764 | 254.01 | 49.61 | 0.29 | 1.00000000 |
| self_attn.qkv_proj | 2 | 36.095 | 343.16 | 67.02 | 0.73 | 1.00000000 |
| self_attn.o_proj | 1 | 28.694 | 226.11 | 44.16 | 0.20 | 1.00000000 |
| self_attn.o_proj | 2 | 23.512 | 275.95 | 53.90 | 0.56 | 1.00000000 |
| mlp.gate_proj | 1 | 62.968 | 270.47 | 52.83 | 0.07 | 1.00000000 |
| mlp.gate_proj | 2 | 45.378 | 375.32 | 73.30 | 1.17 | 1.00000000 |
| mlp.up_proj | 1 | 63.081 | 269.99 | 52.73 | 0.06 | 1.00000000 |
| mlp.up_proj | 2 | 45.300 | 375.96 | 73.43 | 0.32 | 1.00000000 |
| mlp.down_proj | 1 | 63.515 | 280.91 | 54.87 | 0.11 | 1.00000000 |
| mlp.down_proj | 2 | 52.214 | 341.71 | 66.74 | 0.16 | 1.00000000 |
