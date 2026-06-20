# reader_bmm_tile_layout_in0_receiver.cpp — v7
- Group G4 matmul, receiver. Commit 85cc48b1373d2f6d0faf48ad3665de9a62ebed5d. migrated v7 — PASS.
- Validation: 1D interleaved: test_matmul_1d_multiple_output_blocks_per_core[uneven_width=0-mcast_in0=True-...-grid_size=(8, 2)-...n=2048-k=1024-m=256] PASS (exercises in0 receiver).
## Delta
Dead Pipe<> + McastRect::single_core -> ReceiverPipe v7. data_ready=receiver_sem(CTA5), consumed=sender_sem(CTA4).
  Pipe<>(noc, McastRect::single_core(sx,sy), 1, receiver_sem, sender_sem) + .receive()
  -> ReceiverPipe<get_compile_time_arg_val(5), /*PRE_HANDSHAKE=*/true, get_compile_time_arg_val(4)>(noc) + .receive(in0_mcast_sender_noc_x, in0_mcast_sender_noc_y)
Kept raw sems (batch-valid path lines 64-68). diff_lines_removed: ~4 (ctor rect/count args).
