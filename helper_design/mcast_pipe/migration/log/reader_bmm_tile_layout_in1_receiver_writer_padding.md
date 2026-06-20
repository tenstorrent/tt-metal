# reader_bmm_tile_layout_in1_receiver_writer_padding.cpp — v7
- Group G4 matmul, receiver. Commit 592e92169277b7ae09d83348b3ebb481a4493f20. migrated v7 — PASS.
- Validation: 2D in0-sharded: test_matmul_2d_multiple_output_blocks_per_core[...transpose_mcast=True...in0_sharded=True-grid_size=(8, 4)...n=1024-k=512-m=512-b=1] PASS (exercises in1 receiver; verified PASS even with raw role-flip during bisect).
## Delta
Dead Pipe<> + McastRect::single_core -> ReceiverPipe v7. data_ready=receiver_sem(CTA5), consumed=sender_sem(CTA4).
  -> ReceiverPipe<get_compile_time_arg_val(5), /*PRE_HANDSHAKE=*/true, get_compile_time_arg_val(4)>(noc) + .receive(in1_mcast_sender_noc_x, in1_mcast_sender_noc_y) at both call sites (in1 + in3/bias). diff_lines_removed: ~4.
