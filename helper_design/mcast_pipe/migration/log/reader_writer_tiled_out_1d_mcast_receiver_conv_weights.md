# reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp — v7

- Group: G3 conv (HS receiver)
- Commit: d93798c8e0793bd2c104815e7a9d5b45c8f0e04f
- Status: migrated v7 — PASS
- Validation: test_conv_features HEIGHT_SHARDED input_channels=16 output_channels=16 input_height=256 input_width=256 config={'act_block_h':32} (resolved via collect-only) — PASS

## Delta
Pre-v7 2-arg ReceiverPipe -> insert PRE_HANDSHAKE:
  ReceiverPipe<weights_mcast_receiver_sem_id, weights_mcast_sender_sem_id>
  -> ReceiverPipe<weights_mcast_receiver_sem_id, /*PRE_HANDSHAKE=*/true, weights_mcast_sender_sem_id>
.receive(x,y) already v7. diff_lines_removed: 0.
