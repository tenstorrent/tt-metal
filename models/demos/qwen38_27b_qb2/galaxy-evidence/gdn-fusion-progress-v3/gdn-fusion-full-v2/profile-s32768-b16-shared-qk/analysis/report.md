# Qwen long-context layer profile

Warm eager two-layer diagnostic with real weights and synthetic populated caches; not natural prompt activations, model accuracy, traced TPOT or a complete P0 pass.

Durations remain separate per device. Inclusive stages overlap; exclusive stages partition each device total. Firmware sums may include overlapping execution. Per-RISC durations include waits and alone cannot establish a bandwidth or compute bottleneck. Absent compute timings are marked not applicable only when source/hash lists are empty and all compute binary sizes are zero; they are not zero-time measurements. Device-op rows are not program counts.

All device timings present: True. Full P0 gate: still pending.

| Input tokens | Batch | Device | Device-op rows | Sum firmware (ms) | Sum kernel (ms) |
|---:|---:|---:|---:|---:|---:|
| 32768 | 16 | 0 | 130 | 4.478 | 2.998 |
| 32768 | 16 | 4 | 130 | 4.486 | 3.003 |
| 32768 | 16 | 8 | 130 | 4.542 | 3.052 |
| 32768 | 16 | 12 | 130 | 4.431 | 2.980 |

## Most expensive device ops per geometry

### Input 32768, batch 16, device 8

Device with the largest firmware-duration sum; other ranks remain in the CSV/JSON.

| Exclusive stage | Op | Calls | Sum firmware (ms) | Sum kernel (ms) |
|---|---|---:|---:|---:|
| P0_S32768_B16_MODEL | MatmulDeviceOperation | 4 | 0.915 | 0.902 |
| P0_S32768_B16_L3_paged_decode | SdpaDecodeDeviceOperation | 1 | 0.791 | 0.790 |
| P0_S32768_B16_MODEL | ShardedToInterleavedDeviceOperation | 4 | 0.699 | 0.016 |
| P0_S32768_B16_L3_paged_decode | SliceDeviceOperation | 1 | 0.693 | 0.002 |
| P0_S32768_B16_L3__linear_mlp_gate_up | MatmulDeviceOperation | 1 | 0.118 | 0.117 |
| P0_S32768_B16_L0__linear_mlp_gate_up | MatmulDeviceOperation | 1 | 0.117 | 0.117 |
| P0_S32768_B16_L0__delta_recurrence | GenericOpDeviceOperation | 2 | 0.114 | 0.112 |
| P0_S32768_B16_L0_packed_decode_conv | TilizeWithValPaddingDeviceOperation | 3 | 0.076 | 0.073 |
| P0_S32768_B16_L0__linear_linear_attn_packed | MatmulDeviceOperation | 1 | 0.070 | 0.069 |
| P0_S32768_B16_L0__linear_mlp_down_proj | MatmulDeviceOperation | 1 | 0.067 | 0.067 |
| P0_S32768_B16_L3__linear_mlp_down_proj | MatmulDeviceOperation | 1 | 0.067 | 0.066 |
| P0_S32768_B16_L0__delta | GenericOpDeviceOperation | 1 | 0.059 | 0.058 |
| P0_S32768_B16_L3__linear_self_attn_qkvg | MatmulDeviceOperation | 1 | 0.054 | 0.053 |
| P0_S32768_B16_L3__rope_decode | ReshapeViewDeviceOperation | 4 | 0.041 | 0.039 |
| P0_S32768_B16_L0__linear_mlp_down_proj | AllReduceAsyncDeviceOperation | 1 | 0.032 | 0.031 |
| P0_S32768_B16_L0__delta | SliceDeviceOperation | 5 | 0.030 | 0.026 |
| P0_S32768_B16_L0__linear_linear_attn_packed | ReshapeViewDeviceOperation | 1 | 0.030 | 0.029 |
| P0_S32768_B16_L0__linear_linear_attn_out_proj | AllReduceAsyncDeviceOperation | 1 | 0.030 | 0.029 |
| P0_S32768_B16_L0__linear_linear_attn_out_proj | MatmulDeviceOperation | 1 | 0.029 | 0.029 |
| P0_S32768_B16_L3__linear_self_attn_o_proj | MatmulDeviceOperation | 1 | 0.029 | 0.029 |
