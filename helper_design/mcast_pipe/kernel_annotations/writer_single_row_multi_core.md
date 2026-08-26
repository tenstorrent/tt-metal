DERIVED FROM: the 2026-08-03 sort coordinator/reader/writer snapshot, `sort_program_factory.cpp`, `mcast_pipe.hpp`/`.inl` API v9, the three prior migration logs, and `archive/reconciliation/reconcile_2026-08-03.md`.

# writer_single_row_multi_core.cpp (sort) — API-v9 re-audit

Path: `ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/writer_single_row_multi_core.cpp`

Role: **worker completion emitter**. It is the return leg of the sort protocol, not a multicast sender or control-signal receiver.

Current API era: object `Noc` / `Semaphore<>` primitives.

## Required behavior (independent of the current spelling)

For every pair assigned to this worker in a sub-stage:

1. Complete all value and index writes for the pair.
2. Only then increment the coordinator's done counter once.
3. Across all workers, produce exactly `Wt/2` done increments for that sub-stage.

The completion channel must remain distinct from the reader's once-per-row readiness channel. Otherwise an early next-row readiness increment can overshoot the coordinator's exact `Wt/2` done wait.

## Current protocol annotation

### Setup — lines 54–55

- L55 constructs `cores_to_coordinator_done_sem` from the done-semaphore runtime arg.
- L17 reads `coordinator_to_cores_semaphore_arg`, but the writer does not use that value; it is stale ABI baggage from the cross-kernel bundle.

### Per-pair completion — lines 83–155

- L87–149 issue each output write and use `noc.async_write_barrier()` before popping/reusing the corresponding source buffer.
- L152–154 increment the coordinator's **done** counter only after all writes for the processed pair have completed.
- L155 drains the non-posted atomic increment.
- L165 performs a final atomic drain before exit.

## Mapping to `mcast_pipe` API v9

There is no `SenderPipe` or `ReceiverPipe` block to substitute here:

- This kernel does not multicast a signal.
- It does not wait for or clear the coordinator doorbell.
- Its `done_sem.up(...)` is an operation-specific completion event whose target count is `Wt/2`, not a Pipe data-ready or uniform consumer-ready handshake.

API v9 therefore needs no new method for this file. Keep the `Semaphore::up` and atomic ordering explicit. When the coordinator/reader pair is migrated as one protocol unit, the host/kernel ABI can also stop passing the unused doorbell ID to this writer and make the done ID compile-time: the factory already declares it as core-uniform `constexpr` ID `2` at L1189.

## Hazards and invariants

- **Write-before-done:** the done increment must remain after the output write barriers; otherwise the coordinator can release the next sub-stage while data is still in flight.
- **Atomic completion:** the `up()` is a non-posted atomic and retains its atomic drain discipline.
- **Channel separation:** use only the done semaphore here; never fold this increment back into the reader-ready semaphore.
- **Remote-writer lifecycle:** the coordinator's done counter is host-initialized to `0`; it must not be initialized in a coordinator Pipe constructor because writer increments can race such an init.

## Verdict

**defer/raw as a standalone Pipe candidate; helper-neutral companion cleanup completed in
`7337302b564`.**

This file is not evidence of a helper design gap. It remains on the object semaphore API; only the
coupled semaphore-ABI cleanup was applied when coordinator and reader moved to API v9.
