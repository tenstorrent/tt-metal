# reader_mcast_receiver_unary_sharded_ln.cpp — mcast_pipe migration (Tier 2a)

**Status:** migrated @ v7 | **Validation:** PASS

## What changed
Phase-1 of the C3 two-phase sharded-LN reduce receiver migrated to `ReceiverPipe`. Mirrors the
already-migrated sender twin `reader_mcast_sender_unary_sharded_ln.cpp` (which migrated phase-1 via
`send_signal()` and DEFERRED phase-2 raw).

- Added `#include ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp`.
- Constructed
  `ReceiverPipe<reduce_sender_sem_id, /*PRE_HANDSHAKE=*/true, reduce_receiver_sem_id> reduce_pipe(noc)`.
  - `DATA_READY_SEM_ID = reduce_sender_sem` (CTA 1, S->R level FLAG this core waits on).
  - `CONSUMED_SEM_ID  = reduce_receiver_sem` (CTA 0, R->S consumed counter, up'd on the sender).
- Phase-1 handshake `reduce_receiver_sem.up(sender)` + `reduce_sender_sem.wait(VALID)` collapsed to
  `reduce_pipe.receive(in0_remote_noc_x[0], in0_remote_noc_y[0])`.
- RETAINED the leading `reduce_sender_sem.set(INVALID)`: the lambda runs up to twice (mean pass +
  var pass), and phase-2 (below) reuses this SAME sem cell as a monotone counter, so the cell must be
  reset to INVALID BEFORE phase-1's wait. `receive()`'s clear-AFTER does not cover that cross-phase
  reset.

## What stayed raw (deferred, matches the sender twin)
Phase-2 — the final multicast receive loop `reduce_sender_sem.wait_min(block + 2)` — is left RAW.
It reuses the phase-1 flag cell as a MONOTONE counter (streaming), which the Flag/Counter Pipe verbs
cannot express per-side without desyncing the receiver's counter base. The raw `Semaphore<>
reduce_sender_sem` object is retained for that wait_min (it aliases the same L1 cell the pipe drives).
The two-stage `reduce_receiver_sem`/`reduce_second_stage_sem` acks (`up`) and the gather reads (HOLE)
stay raw — protocol gates / NoC reads the Pipe does not own.

## Validation
`tests/ttnn/unit_tests/operations/fused/test_layer_norm_sharded.py::test_layer_norm_sharded_single_stage`
(note: actual path is under `fused/`, ledger validation_set said `normalization/`).
- Smoke (`--dev`, one bf16 use_welford=True nodeid): PASS.
- Full family (`--run-all`): **64 passed, 0 failed**, no hang.
- JIT build confirmed: `reader_mcast_receiver_unary_sharded_ln.cpp` present in
  `generated/inspector/kernels.yaml`.

## Lines
~3 raw handshake lines (set/up/wait) → 1 `receive()` call (+ retained set(INVALID)); net ~2 removed.
