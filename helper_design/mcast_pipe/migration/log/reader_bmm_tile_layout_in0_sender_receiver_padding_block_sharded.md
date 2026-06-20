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

## ROOT CAUSE — CONFIRMED ON DEVICE (2026-06-20)
Decisive A/B on the v7-migrated kernel (commit fa561f3b584), same nodeid, WH:
- CONTROL: v7 unmodified -> HANG (watcher fired, triage written).
- TREATMENT: v7 + ONE line `receiver_sem.set(VALID);` immediately before `in0_send_pipe.send(...)` -> PASS.
The only delta is re-establishing the data-ready flag VALID per send round.

Mechanism: the v7 `SenderPipe` ctor sets the local data-ready cell (receiver_sem, CTA10) VALID **once**
and `send()` never re-sets it (Round-6 "flag-set lifecycle" change — the per-send local set was dropped
as redundant for a STAR sender). But this is a ROTATING-ROLE core: the SAME receiver_sem cell is also
its receiver cell, clobbered to INVALID every round by (a) the per-round `ReceiverPipe` ctor (inits its
own data_ready = INVALID), (b) `receive()`'s post-wait clear, and (c) the top-of-loop raw
`receiver_sem.set(INVALID)`. So by the time this core sends again, the ctor-once VALID is stale -> the
flag mcast broadcasts INVALID -> receivers wait VALID forever -> hang.

The raw (working) kernel does exactly the dropped set: `receiver_sem.set(VALID)` immediately before each
`set_multicast` (HEAD lines ~286/314), with a comment warning that overwriting it INVALID too early makes
"receivers see INVALID and hang." So Round 6's "VALID once in ctor" is correct for a STAR sender but WRONG
for a sender whose data-ready cell doubles as a rotating receiver cell.

tune-dm-helper fix options: (1) a rotating/relay-role face that re-sets VALID per send, or (2) make the
Flag-signal `send()` re-assert the local VALID each call (reverts the Round-6 optimization for this path).
NOT an apply-dm-helper call-site bug.

## LIFTED FROM QUARANTINE (2026-06-20, re-entry @ v7)
- Helper fix 20cf0df46ee (per-send flag-VALID re-assert inside SenderPipe.send for Flag-signal path)
  removes the Round-6 ctor-once-VALID staleness for rotating-role cores.
- Restored the clean v7 call site from fa561f3b584 (no manual workaround line needed — the helper now
  re-asserts VALID per send internally).
- Validation: test_matmul_2d_multiple_output_blocks_per_core --run-all = 56 passed, 72 skipped, 0 failed,
  NO HANG (smoke + full function). transpose_mcast=True + in0_sharded=True params hit this kernel.
- Ledger: quarantined -> migrated@v7.

## REMIGRATED TO v8 (2026-06-20, Tier 0a)
Commit `5a9d07277dfb07178185f131ad5290b0d0cbe7c0`. migrated v8 — PASS.

Transform: v8 SenderPipe dropped the 3rd template param. **PURE DELETION** of the 3rd template arg
(`in0_pipe_active_cores,`) AND the now-dead `constexpr uint32_t in0_pipe_active_cores = ...` declaration
(its only use was that template arg). The per-round ReceiverPipe arm (lines ~256-260) is UNCHANGED — v8
ReceiverPipe is identical to v7.

Dense justification: the factory always sets `in0_mcast_num_dests == in0_mcast_num_cores == rect area`, so
the EXCLUDE fan-out the rect derives — `area() - (in_rect ? 1 : 0)` — reproduces the old
`in0_pipe_active_cores` (`core_in_in0_receiver_mcast_grid ? num_dests-1 : num_dests`) exactly: in-grid cores
get `area-1`, out-of-grid cores get `area`. The rect's `in_rect_` containment IS `core_in_in0_receiver_mcast_grid`.
So the default `consumer_ack_count = ACK_EQUALS_FANOUT` is correct and the explicit count is gone. The
per-send flag-VALID re-assert (helper fix 20cf0df46ee that lifted this from quarantine) is preserved in v8.

Validation: `test_matmul_2d_multiple_output_blocks_per_core` --run-all = **56 passed, 72 skipped, 0 failed,
NO HANG**; 2D smoke (transpose_mcast=True + in0_sharded=True, grid (8,4)) PASS. JIT-built confirmed
(`generated/watcher/watcher.log`). diff_lines_removed: 3 (template arg + 2-line constexpr decl).
