# reader_bmm_tile_layout_in0_receiver.cpp — v7
- Group G4 matmul, receiver. Commit 85cc48b1373d2f6d0faf48ad3665de9a62ebed5d. migrated v7 — PASS.
- Validation: 1D interleaved: test_matmul_1d_multiple_output_blocks_per_core[uneven_width=0-mcast_in0=True-...-grid_size=(8, 2)-...n=2048-k=1024-m=256] PASS (exercises in0 receiver).
## Delta
Dead Pipe<> + McastRect::single_core -> ReceiverPipe v7. data_ready=receiver_sem(CTA5), consumed=sender_sem(CTA4).
  Pipe<>(noc, McastRect::single_core(sx,sy), 1, receiver_sem, sender_sem) + .receive()
  -> ReceiverPipe<get_compile_time_arg_val(5), /*PRE_HANDSHAKE=*/true, get_compile_time_arg_val(4)>(noc) + .receive(in0_mcast_sender_noc_x, in0_mcast_sender_noc_y)
Kept raw sems (batch-valid path lines 64-68). diff_lines_removed: ~4 (ctor rect/count args).

## v8 re-verify (2026-06-20, Tier 0a)
ReceiverPipe is UNCHANGED in v8 (v7→v8 only touched SenderPipe). **NO code edit.** Re-verified alongside
the v8 in0 sender in the 1D suite: `test_matmul_1d_multiple_output_blocks_per_core` --run-all =
48 passed, 16 skipped, 0 failed. JIT-built confirmed (`generated/inspector/kernels.yaml`).
Ledger: migrated_api_version bumped 7 -> 8 (commit kept = 85cc48b...; code identical).
