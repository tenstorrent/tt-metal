# reader_bmm_tile_layout_in1_receiver_writer_padding.cpp — v7
- Group G4 matmul, receiver. Commit 592e92169277b7ae09d83348b3ebb481a4493f20. migrated v7 — PASS.
- Validation: 2D in0-sharded: test_matmul_2d_multiple_output_blocks_per_core[...transpose_mcast=True...in0_sharded=True-grid_size=(8, 4)...n=1024-k=512-m=512-b=1] PASS (exercises in1 receiver; verified PASS even with raw role-flip during bisect).
## Delta
Dead Pipe<> + McastRect::single_core -> ReceiverPipe v7. data_ready=receiver_sem(CTA5), consumed=sender_sem(CTA4).
  -> ReceiverPipe<get_compile_time_arg_val(5), /*PRE_HANDSHAKE=*/true, get_compile_time_arg_val(4)>(noc) + .receive(in1_mcast_sender_noc_x, in1_mcast_sender_noc_y) at both call sites (in1 + in3/bias). diff_lines_removed: ~4.

## v8 re-verify (2026-06-20, Tier 0a)
ReceiverPipe is UNCHANGED in v8. **NO code edit.** Re-verified alongside the v8 block_sharded sender +
in1 sender in the 2D suite: `test_matmul_2d_multiple_output_blocks_per_core` --run-all =
56 passed, 72 skipped, 0 failed. JIT-built confirmed (`generated/watcher/kernel_names.txt`).
Ledger: migrated_api_version bumped 7 -> 8 (commit kept = 592e921...; code identical).
