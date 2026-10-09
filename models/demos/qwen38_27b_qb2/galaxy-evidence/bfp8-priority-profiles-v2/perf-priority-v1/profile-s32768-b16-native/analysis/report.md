# Qwen long-context layer profile

Warm eager two-layer diagnostic with real weights and synthetic populated caches; not natural prompt activations, model accuracy, traced TPOT or a complete P0 pass.

Durations remain separate per device. Inclusive stages overlap; exclusive stages partition each device total. Firmware sums may include overlapping execution. Per-RISC durations include waits and alone cannot establish a bandwidth or compute bottleneck. Absent compute timings are marked not applicable only when source/hash lists are empty and all compute binary sizes are zero; they are not zero-time measurements. Device-op rows are not program counts.

All device timings present: True. Full P0 gate: still pending.

| Input tokens | Batch | Device | Device-op rows | Sum firmware (ms) | Sum kernel (ms) |
|---:|---:|---:|---:|---:|---:|
| 32768 | 16 | 0 | 161 | 4.971 | 3.475 |
| 32768 | 16 | 4 | 161 | 4.971 | 3.473 |
| 32768 | 16 | 8 | 161 | 5.015 | 3.518 |
| 32768 | 16 | 12 | 161 | 4.907 | 3.444 |

## Most expensive device ops per geometry

### Input 32768, batch 16, device 8

Device with the largest firmware-duration sum; other ranks remain in the CSV/JSON.

| Exclusive stage | Op | Calls | Sum firmware (ms) | Sum kernel (ms) |
|---|---|---:|---:|---:|
| P0_S32768_B16_MODEL | MatmulDeviceOperation | 4 | 0.909 | 0.896 |
| P0_S32768_B16_L3_paged_decode | SdpaDecodeDeviceOperation | 1 | 0.789 | 0.788 |
| P0_S32768_B16_L3_paged_decode | SliceDeviceOperation | 1 | 0.699 | 0.002 |
| P0_S32768_B16_MODEL | ShardedToInterleavedDeviceOperation | 4 | 0.682 | 0.016 |
| P0_S32768_B16_L0__delta_recurrence | ChunkGdnPrepOperation | 2 | 0.158 | 0.156 |
| P0_S32768_B16_L0__delta_recurrence | ChunkGdnScanOperation | 2 | 0.121 | 0.120 |
| P0_S32768_B16_L0__linear_mlp_gate_up | MatmulDeviceOperation | 1 | 0.118 | 0.118 |
| P0_S32768_B16_L3__linear_mlp_gate_up | MatmulDeviceOperation | 1 | 0.118 | 0.117 |
| P0_S32768_B16_L0__delta_recurrence | SliceDeviceOperation | 12 | 0.115 | 0.107 |
| P0_S32768_B16_L0__delta_recurrence | ConcatDeviceOperation | 2 | 0.083 | 0.082 |
| P0_S32768_B16_L0_packed_decode_conv | TilizeWithValPaddingDeviceOperation | 3 | 0.076 | 0.073 |
| P0_S32768_B16_L0__linear_linear_attn_packed | MatmulDeviceOperation | 1 | 0.070 | 0.069 |
| P0_S32768_B16_L3__linear_mlp_down_proj | MatmulDeviceOperation | 1 | 0.067 | 0.067 |
| P0_S32768_B16_L0__linear_mlp_down_proj | MatmulDeviceOperation | 1 | 0.067 | 0.067 |
| P0_S32768_B16_L0__delta_recurrence | CopyDeviceOperation | 1 | 0.064 | 0.064 |
| P0_S32768_B16_L0__delta_recurrence | ReshapeViewDeviceOperation | 8 | 0.055 | 0.049 |
| P0_S32768_B16_L3__linear_self_attn_qkvg | MatmulDeviceOperation | 1 | 0.055 | 0.054 |
| P0_S32768_B16_L3__rope_decode | ReshapeViewDeviceOperation | 4 | 0.041 | 0.038 |
| P0_S32768_B16_L0__delta | FillPadDeviceOperation | 3 | 0.034 | 0.031 |
| P0_S32768_B16_L0__delta | SigmoidGatedRmsNormOperation | 1 | 0.034 | 0.033 |
