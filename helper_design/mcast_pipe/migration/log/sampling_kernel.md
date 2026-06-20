# sampling_kernel.cpp (deepseek_v3_b1 micro_ops/sampling) — DEFERRED (v7)

- Group: ccl / deepseek / examples (SENDER, flag-only loop barrier), tag clean
- Status: DEFERRED — coverage-gap (single-chip cannot run the validating test: needs 101 worker cores)
- Validation attempted: test_sampling_topk_single_device[test_1] (num_internal_iterations=100, k=32 ->
  the k>1 path that exercises this kernel's single-device loop-barrier mcast).
  Result: SKIPPED — "Test requires at least 101 worker cores but got 64 (8x8)". The only tests that hit
  the migrated path are all `@pytest.mark.requires_grid_size(101)`; this WH n150 has 64 worker cores.

## What the migration would be (prepared, then reverted)
The loop-barrier block (NCRISC, num_internal_iterations>1, !mesh_mode, final core) was carrying a STALE
pre-v7 `Pipe<>` spelling (single class, 5-field `McastRect{x,y,x,y,num_dests}`, two Semaphore ctor args,
`send_signal(1)`). v7 form (mirrors reader_final_topk):
  SenderPipe<noc_index, loop_ready_sem_id, num_dests, /*PRE_HANDSHAKE=*/false>
      loop_pipe(pipe_noc, McastRect<>{mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y});
  loop_pipe.send_signal();   // was send_signal(1)
EXCLUDE_SRC (final core not in the non-final receiver rect) -> NUM_ACTIVE_RECEIVER_CORES == num_dests.
The trailing local-cell reset to 0 kept (now redundant — send_signal re-asserts VALID). Compiles cleanly
(JIT cache 16/16 hits on the smoke attempt). NOT device-verifiable here.

## Disposition
- Kernel reverted to its stale pre-v7 `Pipe<>` spelling (tree unchanged at this path) — NOT a regression
  this round, since that spelling predates this tier.
- Ledger: status=deferred, flags add `coverage-gap`. Reason: single-chip (64 cores) cannot run the
  101-core test that drives the loop-barrier mcast path. Hand to a >=101-core machine or a mesh harness.
