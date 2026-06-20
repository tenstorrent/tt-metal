# reader_bmm_tile_layout_in0_sender_padding.cpp — v7
- Group G4 matmul, sender. Commit 3d31e412f9d4a5c9a05f4ca625dd72d8d9b27e85. migrated v7 — PASS.
- Validation: 1D interleaved: test_matmul_1d_multiple_output_blocks_per_core[uneven_width=0-mcast_in0=True-...-grid_size=(8, 2)-...n=2048-k=1024-m=256] PASS (exercises in0 sender).
## Delta
Dead Pipe<> -> SenderPipe v7. data_ready=receiver_sem(CTA16), consumed=sender_sem(CTA15), count=in0_mcast_num_dests.
  Pipe<>(noc, McastRect{...}, in0_mcast_num_dests, receiver_sem, sender_sem)
  -> SenderPipe<noc_index, get_compile_time_arg_val(16), in0_mcast_num_dests, /*PRE_HANDSHAKE=*/true, get_compile_time_arg_val(15)>(noc, McastRect<>{...})
Dropped pre-loop receiver_sem.set(VALID) (ctor owns it). Kept raw receiver_sem/sender_sem objects (used by the separate batch-valid set_multicast path lines 210-221). .send() already 3-arg. diff_lines_removed: 1 (set(VALID)).
