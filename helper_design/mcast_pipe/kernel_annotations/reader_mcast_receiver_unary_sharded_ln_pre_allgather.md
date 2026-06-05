# reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp (RECEIVE side)

Path: ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp

API spelling: experimental OO wrapper, raw `RemoteCoord`.
Role: pre-allgather receiver counterpart. Block in `global_reduce_receiver` lambda L121-204; called once L205.

## Block map

### Phase-1 handshake (level flag + counter-up), EXCLUDE_SRC
- L133 `cb_partial_obj.wait_front(...)` — local partial ready.
- L135 `reduce_sender_sem.set(INVALID)` — clear flag.
- L136 `reduce_receiver_sem.up(noc, in0_remote_noc_x[0], in0_remote_noc_y[0], 1)` — COUNTER inc to sender.
- L137 `reduce_sender_sem.wait(VALID)` — LEVEL FLAG wait.
- L139-199 gather reads (HOLE), second-stage two-stage handling.
- L201-202 `cb_reduce_first_stage_obj.wait_front(...)` then `reduce_second_stage_sem.up(...)` — second-stage combine-done counter-up.
- L206 `noc.async_atomic_barrier()` at top level — **NEW SPELLING: atomic barrier** (drains the sem `up` atomic increments, not a write barrier).

## Variant signature
- **F1 = atomic barrier** (L206, `async_atomic_barrier`) — distinct from `async_write_barrier`. Receiver-side flush of outstanding semaphore atomics.
- **F2 = phase-1 LEVEL FLAG + counter-up**. No phase-2 (no data receive; data mcast happens in post_allgather).
- **F3 = EXCLUDE_SRC**.
- **pre_handshake**: full handshake present (set INVALID → up → wait VALID). No data multicast received here.

## Hazards / invariants
- INV: set(INVALID) L135 → up L136 → wait(VALID) L137 ordering (same race window as sharded_ln receiver).
- NEW SPELLING: `noc.async_atomic_barrier()` (L206) — Pipe must expose an atomic-barrier flush distinct from write-barrier (F1 has THREE values now: write-barrier, writes-flushed, atomic-barrier).
- HOLE: gather reads dominate; only L133-137 + L201-202 + L206 are Pipe-relevant.
