# reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp — mcast_pipe migration (Tier 2a)

**Status:** migrated @ v7 | **Validation:** PASS

## What changed
C2 flag receiver (atomic-barrier flush, FLAG + CTR-up, PRE_HANDSHAKE=true). Migrated to
`ReceiverPipe::receive()`.

- Added `#include ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp`.
- Dropped the raw `Semaphore<> reduce_receiver_sem` + `reduce_sender_sem` (the pipe owns both).
- Constructed
  `ReceiverPipe<reduce_sender_sem_id, /*PRE_HANDSHAKE=*/true, reduce_receiver_sem_id> reduce_pipe(noc)`.
  - DATA_READY_SEM_ID = reduce_sender_sem (S->R flag this core waits on).
  - CONSUMED_SEM_ID  = reduce_receiver_sem (R->S consumed counter, up'd on the sender).
- Replaced `set(INVALID)` + `up(sender)` + `wait(VALID)` with
  `reduce_pipe.receive(in0_remote_noc_x[0], in0_remote_noc_y[0])`.
  - The old leading `set(INVALID)` is DROPPED — the ReceiverPipe ctor inits the flag cell INVALID, and
    this lambda runs once (single flag round, no cross-phase counter reuse), so clear-after is safe.
- `receive()` is the correct verb (not `receive_signal()`): no data lands here, but the receiver must
  ACK the sender, and only `receive()` does the consumer up() ack.
- KEPT raw: the trailing `noc.async_atomic_barrier()` (drains the up() atomic); the gather reads (HOLE).

## Validation
`test_pre_allgather_layernorm` — co-compiled with the sender #4. **32 passed, 0 failed**, no hang.
JIT build confirmed in `generated/inspector/kernels.yaml`.

## Lines
~3 raw handshake lines (set/up/wait) → 1 `receive()`; net ~2 removed.
