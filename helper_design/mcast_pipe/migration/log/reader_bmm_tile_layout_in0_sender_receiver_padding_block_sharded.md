# reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp — QUARANTINED
- Group G4 matmul, role-flip (refactor). Commit 49c5bd8f96f2a8716b936a324c233cc5ee27b18e (revert to raw).
- Status: quarantined (migrated_api_version=null).
- Validation: 2D in0-sharded: test_matmul_2d_multiple_output_blocks_per_core[...transpose_mcast=True...in0_sharded=True-grid_size=(8, 4)...n=1024-k=512-m=512-b=1] PASS
- Result: v7 helper migration = HANG (physical cores 18-26/19-26 never finish; TT_THROW system_memory_manager:757). Raw pre-helper = PASS.

## Why quarantined
R6 rotating-role kernel: every grid core runs BOTH the SenderPipe face (top of loop) and a per-round
ReceiverPipe face (rebuilt inside the loop, ack target rotates with block_id). The send() loopback is
inferred per call (extract src==dst -> EXCLUDE n-1; non-extract cb_in2->cb_in0 -> INCLUDE n; out-of-grid
-> EXCLUDE n; in-grid single-core active==0 -> local-copy degenerate).

The v7 translation was API-correct (SenderPipe<noc_index, CTA10, in0_pipe_active_cores, true, CTA9>,
ReceiverPipe<CTA10, true, CTA9>, coords -> receive()), but HANGS on the 2D in0-sharded matmul test.

A/B diagnostic (decisive): swapping in the raw pre-helper role-flip kernel (parent of c9f5f852bdf) while
KEEPING the migrated in1-receiver made the SAME 2D test PASS. So the role-flip kernel is the sole offender;
the migrated in0/in1 senders+receivers are good (1D + 2D both green without it).

Likely cause: the helper send()'s loopback-mode inference and/or the active==0 local-copy degenerate path
do not reproduce this rotating self-mcast's exact data+flag ordering across rounds (same failure class as
conv-WS activation_reader). Needs tune-dm-helper-level work on the rotating-role loopback contract.
Reverted to raw to keep the tree green. diff_lines_removed: full helper sender+receiver blocks.
