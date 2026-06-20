# writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp — v7

- Group: G3 conv (BS receiver, co-compiled with BS sender)
- Commit: b37b1853d20cc977fdb532288faac51338c76d30
- Status: migrated v7 — PASS
- Validation: same BLOCK_SHARDED nodeid as the BS sender — PASS

## Delta
Pre-v7 2-arg ReceiverPipe -> insert PRE_HANDSHAKE (same fix as the HS receiver). .receive(x,y) already v7. diff_lines_removed: 0.
