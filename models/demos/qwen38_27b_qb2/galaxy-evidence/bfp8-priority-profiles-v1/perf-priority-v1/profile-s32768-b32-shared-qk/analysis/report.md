# Qwen long-context layer profile

Warm eager two-layer diagnostic with real weights and synthetic populated caches; not natural prompt activations, model accuracy, traced TPOT or a complete P0 pass.

Durations remain separate per device. Inclusive stages overlap; exclusive stages partition each device total. Firmware sums may include overlapping execution. Per-RISC durations include waits and alone cannot establish a bandwidth or compute bottleneck. Absent compute timings are marked not applicable only when source/hash lists are empty and all compute binary sizes are zero; they are not zero-time measurements. Device-op rows are not program counts.

All device timings present: True. Full P0 gate: still pending.

| Input tokens | Batch | Device | Device-op rows | Sum firmware (ms) | Sum kernel (ms) |
|---:|---:|---:|---:|---:|---:|
| 32768 | 32 | 0 | 156 | 6.512 | 4.226 |
| 32768 | 32 | 4 | 156 | 6.505 | 4.205 |
| 32768 | 32 | 8 | 156 | 6.527 | 4.236 |
| 32768 | 32 | 12 | 156 | 6.455 | 4.225 |

## Most expensive device ops per geometry

### Input 32768, batch 32, device 8

Device with the largest firmware-duration sum; other ranks remain in the CSV/JSON.

| Exclusive stage | Op | Calls | Sum firmware (ms) | Sum kernel (ms) |
|---|---|---:|---:|---:|
| P0_S32768_B32_L3_paged_decode | SdpaDecodeDeviceOperation | 1 | 1.497 | 1.496 |
| P0_S32768_B32_L3_paged_decode | SliceDeviceOperation | 1 | 1.432 | 0.004 |
| P0_S32768_B32_MODEL | MatmulDeviceOperation | 4 | 0.912 | 0.900 |
| P0_S32768_B32_MODEL | ShardedToInterleavedDeviceOperation | 4 | 0.699 | 0.016 |
| P0_S32768_B32_L0__delta_recurrence | GenericOpDeviceOperation | 2 | 0.188 | 0.187 |
| P0_S32768_B32_L0__linear_mlp_gate_up | MatmulDeviceOperation | 1 | 0.118 | 0.117 |
| P0_S32768_B32_L3__linear_mlp_gate_up | MatmulDeviceOperation | 1 | 0.118 | 0.117 |
| P0_S32768_B32_L0_packed_decode_conv | TilizeWithValPaddingDeviceOperation | 3 | 0.104 | 0.101 |
| P0_S32768_B32_L3__rope_decode | ReshapeViewDeviceOperation | 4 | 0.077 | 0.075 |
| P0_S32768_B32_L0__delta_recurrence | TilizeWithValPaddingDeviceOperation | 5 | 0.072 | 0.067 |
| P0_S32768_B32_L0__delta_recurrence | UntilizeWithUnpaddingDeviceOperation | 9 | 0.070 | 0.062 |
| P0_S32768_B32_L0__linear_linear_attn_packed | MatmulDeviceOperation | 1 | 0.070 | 0.069 |
| P0_S32768_B32_L0__linear_mlp_down_proj | MatmulDeviceOperation | 1 | 0.068 | 0.067 |
| P0_S32768_B32_L3__linear_mlp_down_proj | MatmulDeviceOperation | 1 | 0.067 | 0.066 |
| P0_S32768_B32_L0__linear_linear_attn_packed | ReshapeViewDeviceOperation | 1 | 0.057 | 0.057 |
| P0_S32768_B32_L3__linear_self_attn_qkvg | MatmulDeviceOperation | 1 | 0.054 | 0.054 |
| P0_S32768_B32_L0__delta | SliceDeviceOperation | 5 | 0.053 | 0.049 |
| P0_S32768_B32_L0__delta | SigmoidGatedRmsNormOperation | 1 | 0.052 | 0.051 |
| P0_S32768_B32_L0__linear_linear_attn_packed | SliceDeviceOperation | 1 | 0.045 | 0.045 |
| P0_S32768_B32_L0__delta | FillPadDeviceOperation | 3 | 0.042 | 0.040 |
