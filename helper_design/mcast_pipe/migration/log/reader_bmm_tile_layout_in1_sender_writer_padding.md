# reader_bmm_tile_layout_in1_sender_writer_padding.cpp — v7
- Group G4 matmul, hybrid (sender). Commit 3a90bb8928626886801d3db7373b54772e2cb8c7. migrated v7 — PASS.
- Validation: 1D interleaved: test_matmul_1d_multiple_output_blocks_per_core[uneven_width=0-mcast_in0=True-...-grid_size=(8, 2)-...n=2048-k=1024-m=256] PASS (exercises in1 sender).
## Delta
Dead Pipe<> -> SenderPipe v7. data_ready=receiver_sem(CTA11), consumed=sender_sem(CTA10), count=in1_mcast_num_dests.
  -> SenderPipe<noc_index, get_compile_time_arg_val(11), in1_mcast_num_dests, /*PRE_HANDSHAKE=*/true, get_compile_time_arg_val(10)>(noc, McastRect<>{...})
Dropped pre-loop receiver_sem.set(VALID). Same pipe serves in1 + in3/bias send()s. diff_lines_removed: 1.
