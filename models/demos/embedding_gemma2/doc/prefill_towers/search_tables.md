### v1: vision 1 x 2560: pooler matmul config, MLP down config / family, h placement

Source `search/vision_v1.jsonl`; grid 12 x 10. Burst = median of 5 windows after 2 s idle; delta vs the mean of the first and last control of the session.

| candidate | 1x2560 burst ms | 1x2560 delta | 1x2560 soft-token PCC / emb cos vs HF fp32 | 1x2560 max abs vs control | result |
|---|---|---|---|---|---|
| control | 16.269 | -0.01 % | 0.99730 / 0.99973 | 0 | control (drift 0.004 ms) |
| pool_2d_12x9 | 16.016 | -1.57 % | 0.99730 / 0.99973 | 0 | kept (in v4 combination) (drift 0.004 ms) |
| pool_2d_12x9_bw16 | 16.012 | -1.59 % | 0.99730 / 0.99973 | 0 | rejected: a better candidate for the same role kept (drift 0.004 ms) |
| pool_2d_12x3 | 16.026 | -1.51 % | 0.99730 / 0.99973 | 0 | rejected: a better candidate for the same role kept (drift 0.004 ms) |
| down_min_4_8_4 | 16.199 | -0.45 % | 0.99730 / 0.99973 | 0 | rejected: a better candidate for the same role kept (drift 0.004 ms) |
| down_min_4_8_2 | 16.600 | +2.02 % | 0.99786 / 0.99977 | 0.781 | rejected: slower (drift 0.004 ms) |
| down_min_8_8_4 | 16.252 | -0.12 % | 0.99730 / 0.99973 | 0 | rejected: a better candidate for the same role kept (drift 0.004 ms) |
| down_min_8_4_8 | - | - | - | - | error: `RuntimeError: TT_FATAL @ /local/ttuser/gtobar/tt-metal/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.cpp:288: cfg.` |
| h_l1 | 16.240 | -0.20 % | 0.99730 / 0.99973 | 0 | rejected: a better candidate for the same role kept (drift 0.004 ms) |
| h_l1_down_2d | 15.551 | -4.43 % | 0.99762 / 0.99976 | 0.77 | rejected: a better candidate for the same role kept (drift 0.004 ms) |
| h_l1_down_2d_bw4 | 15.740 | -3.26 % | 0.99756 / 0.99974 | 1.25 | rejected: a better candidate for the same role kept (drift 0.004 ms) |
| h_l1_down_2d_bw12 | 15.514 | -4.65 % | 0.99778 / 0.99975 | 0.652 | kept (in v4 combination) (drift 0.004 ms) |
| control_end | 16.273 | +0.01 % | 0.99730 / 0.99973 | 0 | control (drift 0.004 ms) |

### v2: vision 1 x 2560: qkv / up / gate / o subblocks and in0_block_w

Source `search/vision_v2.jsonl`; grid 12 x 10. Burst = median of 5 windows after 2 s idle; delta vs the mean of the first and last control of the session.

| candidate | 1x2560 burst ms | 1x2560 delta | 1x2560 soft-token PCC / emb cos vs HF fp32 | 1x2560 max abs vs control | result |
|---|---|---|---|---|---|
| control | 16.270 | -0.02 % | 0.99730 / 0.99973 | 0 | control (drift 0.007 ms) |
| qkv_sb22 | 16.282 | +0.05 % | 0.99730 / 0.99973 | 0 | rejected: slower (drift 0.007 ms) |
| qkv_bw12 | 16.298 | +0.15 % | 0.99786 / 0.99978 | 0.688 | rejected: slower (drift 0.007 ms) |
| qkv_bw4 | 16.313 | +0.24 % | 0.99761 / 0.99974 | 0.785 | rejected: slower (drift 0.007 ms) |
| up_gate_bw12 | 16.312 | +0.24 % | 0.99776 / 0.99977 | 0.609 | rejected: slower (drift 0.007 ms) |
| up_gate_bw4 | 16.415 | +0.88 % | 0.99782 / 0.99977 | 0.953 | rejected: slower (drift 0.007 ms) |
| up_gate_sb22 | 16.380 | +0.66 % | 0.99730 / 0.99973 | 0 | rejected: slower (drift 0.007 ms) |
| o_bw12 | 16.283 | +0.06 % | 0.99760 / 0.99975 | 0.496 | rejected: slower (drift 0.007 ms) |
| o_bw24 | 16.316 | +0.26 % | 0.99760 / 0.99975 | 0.496 | rejected: slower (drift 0.007 ms) |
| o_min_4_8_2 | 16.625 | +2.16 % | 0.99794 / 0.99977 | 0.891 | rejected: slower (drift 0.007 ms) |
| control_end | 16.276 | +0.02 % | 0.99730 / 0.99973 | 0 | control (drift 0.007 ms) |

### v3: vision 1 x 2560: patch-embedder topology and placement

Source `search/vision_v3.jsonl`; grid 12 x 10. Burst = median of 5 windows after 2 s idle; delta vs the mean of the first and last control of the session.

| candidate | 1x2560 burst ms | 1x2560 delta | 1x2560 soft-token PCC / emb cos vs HF fp32 | 1x2560 max abs vs control | result |
|---|---|---|---|---|---|
| control | 16.278 | +0.01 % | 0.99730 / 0.99973 | 0 | control (drift 0.003 ms) |
| affine_one_op | 16.219 | -0.35 % | 0.99730 / 0.99973 | 0 | kept (in the pair) (drift 0.003 ms) |
| patch_l1 | 16.236 | -0.24 % | 0.99730 / 0.99973 | 0 | kept (in the pair) (drift 0.003 ms) |
| affine_one_op+patch_l1 | 16.175 | -0.62 % | 0.99730 / 0.99973 | 0 | kept (topology, every grid) (drift 0.003 ms) |
| control_end | 16.274 | -0.01 % | 0.99730 / 0.99973 | 0 | control (drift 0.003 ms) |

### v4: vision 1 x 2560: v1 winners combined, down in0_block_w 8 ... 24

Source `search/vision_v4.jsonl`; grid 12 x 10. Burst = median of 5 windows after 2 s idle; delta vs the mean of the first and last control of the session.

| candidate | 1x2560 burst ms | 1x2560 delta | 1x2560 soft-token PCC / emb cos vs HF fp32 | 1x2560 max abs vs control | result |
|---|---|---|---|---|---|
| control | 16.270 | -0.02 % | 0.99730 / 0.99973 | 0 | control (drift 0.007 ms) |
| pool+h_l1_down_bw12 | 15.259 | -6.23 % | 0.99778 / 0.99975 | 0.652 | kept (12x10 table) (drift 0.007 ms) |
| pool+h_l1_down_bw8 | 15.287 | -6.06 % | 0.99762 / 0.99976 | 0.77 | rejected: a better candidate for the same role kept (drift 0.007 ms) |
| pool+h_l1_down_bw16 | 15.257 | -6.24 % | 0.99760 / 0.99975 | 0.578 | rejected: a better candidate for the same role kept (drift 0.007 ms) |
| pool+h_l1_down_bw24 | - | - | - | - | error: `RuntimeError: TT_THROW @ /local/ttuser/gtobar/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3072: tt::exception
info:
Statically allocated dataflow` |
| pool+h_l1_down_bw12_sb41 | 15.258 | -6.24 % | 0.99778 / 0.99975 | 0.652 | rejected: a better candidate for the same role kept (drift 0.007 ms) |
| control_end | 16.277 | +0.02 % | 0.99730 / 0.99973 | 0 | control (drift 0.007 ms) |

### a0: audio 1 x 256 / 1 x 768: rel-shift index once per forward (control = default) vs per layer

Source `search/audio_a0.jsonl`; grid 12 x 10. Burst = median of 5 windows after 2 s idle; delta vs the mean of the first and last control of the session.

| candidate | 1x256 burst ms | 1x256 delta | 1x256 soft-token PCC / emb cos vs HF fp32 | 1x256 max abs vs control | 1x768 burst ms | 1x768 delta | 1x768 soft-token PCC / emb cos vs HF fp32 | 1x768 max abs vs control | result |
|---|---|---|---|---|---|---|---|---|---|
| control | 7.566 | +0.01 % | 0.99849 / 0.99992 | 0 | 12.139 | -0.00 % | 0.99767 / 0.99992 | 0 | kept: once per forward is the default (control) (drift 0.002 ms) |
| shift_idx_per_layer | 7.585 | +0.26 % | 0.99849 / 0.99992 | 0 | 12.190 | +0.42 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.002 ms) |
| control_end | 7.564 | -0.01 % | 0.99849 / 0.99992 | 0 | 12.139 | +0.00 % | 0.99767 / 0.99992 | 0 | control (drift 0.002 ms) |

### a1: audio 1 x 768: transposed 2D multicast per role

Source `search/audio_a1.jsonl`; grid 12 x 10. Burst = median of 5 windows after 2 s idle; delta vs the mean of the first and last control of the session.

| candidate | 1x768 burst ms | 1x768 delta | 1x768 soft-token PCC / emb cos vs HF fp32 | 1x768 max abs vs control | result |
|---|---|---|---|---|---|
| control | 12.141 | +0.02 % | 0.99767 / 0.99992 | 0 | control (drift 0.004 ms) |
| qkv_T | 12.318 | +1.47 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.004 ms) |
| up_T | 12.493 | +2.92 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.004 ms) |
| down_T | 12.879 | +6.10 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.004 ms) |
| lstart_T | 12.240 | +0.83 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.004 ms) |
| sq_T | 12.461 | +2.65 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.004 ms) |
| shift_T | 12.570 | +3.55 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.004 ms) |
| up_T_bw4 | 12.496 | +2.94 % | 0.99770 / 0.99992 | 0.148 | rejected: slower (drift 0.004 ms) |
| down_T_bw16 | 13.040 | +7.42 % | 0.99767 / 0.99992 | 0.781 | rejected: slower (drift 0.004 ms) |
| up_T_pc2x14 | 12.569 | +3.55 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.004 ms) |
| up_T_pc2x16 | 12.630 | +4.05 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.004 ms) |
| lstart_T_pc2x8 | 12.273 | +1.11 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.004 ms) |
| all_T | 14.255 | +17.43 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.004 ms) |
| control_end | 12.137 | -0.02 % | 0.99767 / 0.99992 | 0 | control (drift 0.004 ms) |

### a2: audio 1 x 256: 1D in0-multicast and transposed 2D per role

Source `search/audio_a2.jsonl`; grid 12 x 10. Burst = median of 5 windows after 2 s idle; delta vs the mean of the first and last control of the session.

| candidate | 1x256 burst ms | 1x256 delta | 1x256 soft-token PCC / emb cos vs HF fp32 | 1x256 max abs vs control | result |
|---|---|---|---|---|---|
| control | 7.564 | -0.01 % | 0.99849 / 0.99992 | 0 | control (drift 0.002 ms) |
| up_1d | 7.650 | +1.13 % | 0.99849 / 0.99992 | 0 | rejected: slower (drift 0.002 ms) |
| down_1d | 8.677 | +14.69 % | 0.99849 / 0.99992 | 0 | rejected: slower (drift 0.002 ms) |
| qkv_1d | 7.644 | +1.04 % | 0.99849 / 0.99992 | 0 | rejected: slower (drift 0.002 ms) |
| lstart_1d | 7.576 | +0.14 % | 0.99849 / 0.99992 | 0 | rejected: slower (drift 0.002 ms) |
| sq_1d | 7.764 | +2.63 % | 0.99849 / 0.99992 | 0 | rejected: slower (drift 0.002 ms) |
| up_1d_8x8 | 7.605 | +0.53 % | 0.99849 / 0.99992 | 0 | rejected: slower (drift 0.002 ms) |
| down_1d_bw16 | 8.565 | +13.22 % | 0.99902 / 0.99991 | 2.04 | rejected: slower (drift 0.002 ms) |
| up_T | 8.160 | +7.86 % | 0.99849 / 0.99992 | 0 | rejected: slower (drift 0.002 ms) |
| down_T | 8.153 | +7.76 % | 0.99849 / 0.99992 | 0 | rejected: slower (drift 0.002 ms) |
| shift_T | 7.630 | +0.86 % | 0.99849 / 0.99992 | 0 | rejected: slower (drift 0.002 ms) |
| control_end | 7.566 | +0.01 % | 0.99849 / 0.99992 | 0 | control (drift 0.002 ms) |

### a3: audio both buckets: non-transposed 2D in0_block_w 4 / 16 per role

Source `search/audio_a3.jsonl`; grid 12 x 10. Burst = median of 5 windows after 2 s idle; delta vs the mean of the first and last control of the session.

| candidate | 1x256 burst ms | 1x256 delta | 1x256 soft-token PCC / emb cos vs HF fp32 | 1x256 max abs vs control | 1x768 burst ms | 1x768 delta | 1x768 soft-token PCC / emb cos vs HF fp32 | 1x768 max abs vs control | result |
|---|---|---|---|---|---|---|---|---|---|
| control | 7.566 | +0.00 % | 0.99849 / 0.99992 | 0 | 12.140 | +0.00 % | 0.99767 / 0.99992 | 0 | control (drift 0.001 ms) |
| 256_up_bw16 | 7.602 | +0.49 % | 0.99849 / 0.99992 | 0 | 12.139 | +0.00 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.001 ms) |
| 256_up_bw4 | 7.602 | +0.48 % | 0.99849 / 0.99992 | 0.0938 | 12.139 | -0.00 % | 0.99767 / 0.99992 | 0 | rejected: inside drift / noise (drift 0.001 ms) |
| 256_down_bw16 | 7.493 | -0.96 % | 0.99902 / 0.99991 | 2.04 | 12.142 | +0.02 % | 0.99767 / 0.99992 | 0 | rejected: a better candidate for the same role kept (drift 0.001 ms) |
| 256_down_bw4 | 7.777 | +2.80 % | 0.99830 / 0.99992 | 0.707 | 12.140 | +0.01 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.001 ms) |
| 256_qkv_bw16 | 7.583 | +0.23 % | 0.99827 / 0.99992 | 0.535 | 12.136 | -0.02 % | 0.99767 / 0.99992 | 0 | rejected: inside drift / noise (drift 0.001 ms) |
| 256_sq_bw16 | 7.566 | +0.01 % | 0.99849 / 0.99992 | 0.121 | 12.139 | +0.00 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.001 ms) |
| 256_shift_bw4 | 7.564 | -0.02 % | 0.99849 / 0.99992 | 0 | 12.137 | -0.02 % | 0.99767 / 0.99992 | 0 | rejected: inside drift / noise (drift 0.001 ms) |
| 768_up_bw16 | 7.568 | +0.03 % | 0.99849 / 0.99992 | 0 | 12.207 | +0.56 % | 0.99767 / 0.99992 | 0.266 | rejected: slower (drift 0.001 ms) |
| 768_up_bw4 | 7.567 | +0.02 % | 0.99849 / 0.99992 | 0 | 12.128 | -0.09 % | 0.99770 / 0.99992 | 0.148 | rejected: a better candidate for the same role kept (drift 0.001 ms) |
| 768_down_bw16 | 7.566 | +0.01 % | 0.99849 / 0.99992 | 0 | 12.098 | -0.34 % | 0.99767 / 0.99992 | 0.781 | kept (12x10 table) (drift 0.001 ms) |
| 768_down_bw4 | 7.563 | -0.03 % | 0.99849 / 0.99992 | 0 | 12.352 | +1.75 % | 0.99770 / 0.99991 | 0.582 | rejected: inside drift / noise (drift 0.001 ms) |
| 768_qkv_bw16 | 7.567 | +0.02 % | 0.99849 / 0.99992 | 0 | 12.172 | +0.27 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.001 ms) |
| 768_sq_bw16 | 7.563 | -0.03 % | 0.99849 / 0.99992 | 0 | 12.140 | +0.01 % | 0.99766 / 0.99992 | 0.234 | rejected: inside drift / noise (drift 0.001 ms) |
| 768_shift_bw4 | 7.564 | -0.03 % | 0.99849 / 0.99992 | 0 | 12.303 | +1.35 % | 0.99767 / 0.99992 | 0 | rejected: inside drift / noise (drift 0.001 ms) |
| control_end | 7.566 | -0.00 % | 0.99849 / 0.99992 | 0 | 12.139 | -0.00 % | 0.99767 / 0.99992 | 0 | control (drift 0.001 ms) |

### a4: audio both buckets: FFN down in0_block_w 32, combinations

Source `search/audio_a4.jsonl`; grid 12 x 10. Burst = median of 5 windows after 2 s idle; delta vs the mean of the first and last control of the session.

| candidate | 1x256 burst ms | 1x256 delta | 1x256 soft-token PCC / emb cos vs HF fp32 | 1x256 max abs vs control | 1x768 burst ms | 1x768 delta | 1x768 soft-token PCC / emb cos vs HF fp32 | 1x768 max abs vs control | result |
|---|---|---|---|---|---|---|---|---|---|
| control | 7.567 | -0.01 % | 0.99849 / 0.99992 | 0 | 12.140 | +0.00 % | 0.99767 / 0.99992 | 0 | control (drift 0.001 ms) |
| 256_down_bw32 | 7.470 | -1.29 % | 0.99907 / 0.99991 | 2.01 | 12.140 | +0.00 % | 0.99767 / 0.99992 | 0 | kept (12x10 table) (drift 0.001 ms) |
| 768_down_bw32 | 7.566 | -0.01 % | 0.99849 / 0.99992 | 0 | 12.139 | -0.00 % | 0.99833 / 0.99992 | 2.8 | rejected: inside drift / noise (drift 0.001 ms) |
| down_bw16_both | 7.493 | -0.98 % | 0.99902 / 0.99991 | 2.04 | 12.098 | -0.34 % | 0.99767 / 0.99992 | 0.781 | 768 entry kept (12x10 table); 256 superseded by bw32 (drift 0.001 ms) |
| down_bw16_both+768_up_bw4 | 7.493 | -0.98 % | 0.99902 / 0.99991 | 2.04 | 12.096 | -0.36 % | 0.99770 / 0.99992 | 0.781 | rejected: a better candidate for the same role kept (drift 0.001 ms) |
| control_end | 7.568 | +0.01 % | 0.99849 / 0.99992 | 0 | 12.139 | -0.00 % | 0.99767 / 0.99992 | 0 | control (drift 0.001 ms) |

### a5: audio both buckets: rel-pos Q E^T block-matmul per_core_M

Source `search/audio_a5.jsonl`; grid 12 x 10. Burst = median of 5 windows after 2 s idle; delta vs the mean of the first and last control of the session.

| candidate | 1x256 burst ms | 1x256 delta | 1x256 soft-token PCC / emb cos vs HF fp32 | 1x256 max abs vs control | 1x768 burst ms | 1x768 delta | 1x768 soft-token PCC / emb cos vs HF fp32 | 1x768 max abs vs control | result |
|---|---|---|---|---|---|---|---|---|---|
| control | 7.566 | -0.01 % | 0.99849 / 0.99992 | 0 | 12.139 | -0.00 % | 0.99767 / 0.99992 | 0 | control (drift 0.001 ms) |
| rel_pm1_256 | 7.585 | +0.25 % | 0.99849 / 0.99992 | 0 | 12.139 | +0.00 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.001 ms) |
| rel_pm2_256 | 7.552 | -0.19 % | 0.99849 / 0.99992 | 0 | 12.139 | +0.00 % | 0.99767 / 0.99992 | 0 | kept (12x10 table) (drift 0.001 ms) |
| rel_pm1_768 | 7.568 | +0.02 % | 0.99849 / 0.99992 | 0 | 12.232 | +0.77 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.001 ms) |
| rel_pm2_768 | 7.566 | -0.01 % | 0.99849 / 0.99992 | 0 | 12.156 | +0.14 % | 0.99767 / 0.99992 | 0 | rejected: inside drift / noise (drift 0.001 ms) |
| rel_pm1_both | 7.584 | +0.24 % | 0.99849 / 0.99992 | 0 | 12.231 | +0.76 % | 0.99767 / 0.99992 | 0 | rejected: slower (drift 0.001 ms) |
| control_end | 7.567 | +0.01 % | 0.99849 / 0.99992 | 0 | 12.139 | +0.00 % | 0.99767 / 0.99992 | 0 | control (drift 0.001 ms) |
