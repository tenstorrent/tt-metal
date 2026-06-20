# writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp — v7

- Group: G3 conv (BS sender)
- Commit: 65d233ef34803ce8b257cf91cc0703bb6872b4f2
- Status: migrated v7 — PASS
- Validation: test_conv_features BLOCK_SHARDED input_channels=128 output_channels=128 input_height=32 input_width=32 (BFLOAT8_B/BFLOAT8_B HiFi4 fp32_accum=True) — PASS (co-compiled with the BS receiver)

## Delta
SenderPipe reordered to v7 + helper McastRect templated:
  SenderPipe<weights_mcast_num_dests_ct, weights_mcast_receiver_sem_id, weights_mcast_sender_sem_id>(noc, McastRect{...})
  -> SenderPipe<noc_index, weights_mcast_receiver_sem_id, weights_mcast_num_dests_ct, /*PRE_HANDSHAKE=*/true, weights_mcast_sender_sem_id>(noc, McastRect<>{...})
Note: the line-95 `McastRect` is the conv-local conv_reader_common.hpp struct (fields .noc_x_start...), NOT the helper type — left as-is; its corners feed the helper McastRect<>. .send(s,d,sz) already v7. diff_lines_removed: 0.
