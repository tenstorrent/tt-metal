# Candidate host timings

Ranking only: fixed decode position4096, 128 trace replays per trial; not device-time telemetry.

| Matrix/candidate | Prefill host ms | Traced host us (median) |
|---|---:|---:|
| precision/0_attn4_timing | 54.579 | 2206.670 |
| precision/0_attn_lofi_timing | 54.836 | 2214.660 |
| precision/0_bfp8_timing | 55.610 | 2217.436 |
| precision/0_gate4_timing | 52.555 | 2019.071 |
| precision/0_mlp4_timing | 51.703 | 2017.779 |
| precision/0_mlp_lofi_timing | 51.819 | 2216.435 |
| precision/5_attn4_timing | 81.478 | 2246.115 |
| precision/5_attn_lofi_timing | 81.497 | 2327.644 |
| precision/5_bfp8_timing | 83.262 | 2327.096 |
| precision/5_gate4_timing | 80.120 | 2129.321 |
| precision/5_mlp4_timing | 79.173 | 2127.992 |
| precision/5_mlp_lofi_timing | 79.461 | 2327.186 |
| geometry_0_bfp4/control_timing | 55.300 | 2340.299 |
| geometry_0_bfp4/down_c28_b12_timing | 55.336 | 2068.858 |
| geometry_0_bfp4/down_c56_b12_timing | 55.338 | 2070.364 |
| geometry_0_bfp4/down_c8_b14_timing | 55.326 | 2054.426 |
| geometry_0_bfp4/down_c8_b7_timing | 55.313 | 2067.982 |
| geometry_0_bfp4/down_r3_timing | 55.352 | 2061.766 |
| geometry_0_bfp4/gate_c14_b6_timing | 55.305 | 1916.291 |
| geometry_0_bfp4/gate_c28_b6_timing | 55.348 | 1864.520 |
| geometry_0_bfp4/gate_c56_b3_timing | 55.317 | 1858.176 |
| geometry_0_bfp4/gate_c8_b3_timing | 55.294 | 2001.725 |
| geometry_0_bfp4/gate_r2_timing | 55.274 | 1926.639 |
| geometry_0_bfp4/gate_r3_timing | 55.340 | 2002.165 |
| geometry_0_bfp4/o_b16_timing | 55.478 | 2231.917 |
| geometry_0_bfp4/o_b4_timing | 55.347 | 2245.944 |
| geometry_0_bfp4/o_b8_timing | 55.515 | 2235.715 |
| geometry_0_bfp4/o_r3_timing | 55.456 | 2234.177 |
| geometry_0_bfp4/qkv_b3_timing | 55.388 | 2142.064 |
| geometry_0_bfp4/qkv_b7_timing | 55.380 | 2111.668 |
| geometry_0_bfp4/qkv_r2_timing | 55.444 | 2073.432 |
| combined_0/combined_timing | 55.447 | 1190.920 |
| combined_0/down_b14_r3_timing | 55.363 | 1195.913 |
| combined_0/down_b28_r3_timing | 55.300 | 1195.218 |
| combined_0/gate56_b12_r2_timing | 55.314 | 1045.243 |
| combined_0/gate56_b12_r3_timing | 55.346 | 1066.805 |
| combined_0/gate56_b6_r1_timing | 55.479 | 1170.331 |
| combined_0/gate56_b6_r2_timing | 55.313 | 1039.547 |
| combined_0/gate56_b6_r3_timing | 55.276 | 1061.052 |
| combined_0/o_b16_r3_timing | 55.366 | 1192.541 |
| combined_0/o_b32_r3_timing | 55.399 | 1193.642 |
| combined_0/q_b12_timing | 55.352 | 1236.204 |
| combined_0/q_b21_timing | 55.370 | 1198.023 |
| attention_packing_0/control_timing | 55.373 | 1039.346 |
| attention_packing_0/head_l1_timing | 55.337 | 1030.823 |
| attention_packing_0/kv8_timing | 46.395 | 1041.125 |
| attention_packing_0/packed_qkv_timing | 56.083 | 1045.261 |
| attention_packing_0/prefill128_timing | 38.761 | 1039.708 |
| attention_packing_0/prefill64_timing | 44.098 | 1039.662 |
| attention_packing_0/sdpa_hifi2_timing | 38.610 | 1024.698 |
| attention_packing_0/sdpa_k128_timing | 55.392 | 963.943 |
| attention_packing_0/sdpa_k256_timing | 55.275 | 974.919 |
| attention_packing_0/sdpa_k64_timing | 55.334 | 960.074 |
