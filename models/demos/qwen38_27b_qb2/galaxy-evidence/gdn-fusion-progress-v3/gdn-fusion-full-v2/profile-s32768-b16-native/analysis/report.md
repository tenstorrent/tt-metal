# Qwen long-context layer profile

Warm eager two-layer diagnostic with real weights and synthetic populated caches; not natural prompt activations, model accuracy, traced TPOT or a complete P0 pass.

Durations remain separate per device. Inclusive stages overlap; exclusive stages partition each device total. Firmware sums may include overlapping execution. Per-RISC durations include waits and alone cannot establish a bandwidth or compute bottleneck. Absent compute timings are marked not applicable only when source/hash lists are empty and all compute binary sizes are zero; they are not zero-time measurements. Device-op rows are not program counts.

All device timings present: True. Full P0 gate: still pending.

| Input tokens | Batch | Device | Device-op rows | Sum firmware (ms) | Sum kernel (ms) |
|---:|---:|---:|---:|---:|---:|
| 32768 | 16 | 0 | 160 | 4.639 | 3.124 |
| 32768 | 16 | 4 | 160 | 4.649 | 3.128 |
| 32768 | 16 | 8 | 160 | 4.679 | 3.163 |
| 32768 | 16 | 12 | 160 | 4.601 | 3.113 |

## Most expensive device ops per geometry

### Input 32768, batch 16, device 8

Device with the largest firmware-duration sum; other ranks remain in the CSV/JSON.

| Exclusive stage | Op | Calls | Sum firmware (ms) | Sum kernel (ms) |
|---|---|---:|---:|---:|
| P0_S32768_B16_MODEL | MatmulDeviceOperation | 4 | 0.903 | 0.891 |
| P0_S32768_B16_L3_paged_decode | SdpaDecodeDeviceOperation | 1 | 0.787 | 0.786 |
| P0_S32768_B16_L3_paged_decode | SliceDeviceOperation | 1 | 0.700 | 0.002 |
| P0_S32768_B16_MODEL | ShardedToInterleavedDeviceOperation | 4 | 0.693 | 0.016 |
| P0_S32768_B16_L0__linear_mlp_gate_up | MatmulDeviceOperation | 1 | 0.118 | 0.118 |
| P0_S32768_B16_L3__linear_mlp_gate_up | MatmulDeviceOperation | 1 | 0.118 | 0.118 |
| P0_S32768_B16_L0__delta_recurrence | GenericOpDeviceOperation | 2 | 0.110 | 0.109 |
| P0_S32768_B16_L0_packed_decode_conv | TilizeWithValPaddingDeviceOperation | 3 | 0.076 | 0.073 |
| P0_S32768_B16_L0__linear_linear_attn_packed | MatmulDeviceOperation | 1 | 0.070 | 0.069 |
| P0_S32768_B16_L0__linear_mlp_down_proj | MatmulDeviceOperation | 1 | 0.067 | 0.066 |
| P0_S32768_B16_L3__linear_mlp_down_proj | MatmulDeviceOperation | 1 | 0.067 | 0.066 |
| P0_S32768_B16_L0__delta_recurrence | UntilizeWithUnpaddingDeviceOperation | 9 | 0.057 | 0.048 |
| P0_S32768_B16_L3__linear_self_attn_qkvg | MatmulDeviceOperation | 1 | 0.054 | 0.054 |
| P0_S32768_B16_L0__delta_recurrence | TilizeWithValPaddingDeviceOperation | 5 | 0.052 | 0.047 |
| P0_S32768_B16_L3__rope_decode | ReshapeViewDeviceOperation | 4 | 0.041 | 0.039 |
| P0_S32768_B16_L0__delta | FillPadDeviceOperation | 3 | 0.034 | 0.031 |
| P0_S32768_B16_L0__delta | SigmoidGatedRmsNormOperation | 1 | 0.033 | 0.033 |
| P0_S32768_B16_L0__linear_linear_attn_out_proj | AllReduceAsyncDeviceOperation | 1 | 0.031 | 0.030 |
| P0_S32768_B16_L0__delta | SliceDeviceOperation | 5 | 0.030 | 0.026 |
| P0_S32768_B16_L0__linear_linear_attn_packed | ReshapeViewDeviceOperation | 1 | 0.030 | 0.029 |
