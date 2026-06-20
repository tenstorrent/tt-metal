# writer_local_topk.cpp — mcast_pipe migration (tier 2b)

## Summary
SENDER companion of the migrated `reader_final_topk.cpp`. This local-topk core RECEIVES the readiness
invite flag broadcast by the final aggregator core's `SenderPipe::send_signal()`, then unicast-scatters
its TopK values/indices to the final core and acks via a fan-in counter.

## Change
- Replaced the per-iteration `receiver_sem.wait(VALID)` (top of loop) + `receiver_sem.set(INVALID)`
  (bottom of loop) with a single `ReceiverPipe<receiver_sem_id, PRE_HANDSHAKE=false>` and
  `ready_pipe.receive_signal()` (waits VALID + clears INVALID in one call).
- Dropped the local `Semaphore<> receiver_sem` object (now owned by the Pipe).
- KEPT RAW (Pipe does not own these): the unicast data scatter (values + indices `noc.async_write`),
  the fan-in ack `sender_sem.up(noc, noc_final_x, noc_final_y, Kt)` (multi-producer counter, INV9),
  and the trailing atomic barrier.

## Validation
- `tests/ttnn/unit_tests/operations/reduce/test_topk.py::test_topk` (path is `reduce/`, not `reduction/`)
- Smoke (--dev): W=8192 k=50 BFLOAT16_B → PASS; kernel JIT-built.
- Full (--run-all): 80 passed, 80 xfailed, 0 failed.

## Lines removed: ~3 (1 sem obj decl, the explicit wait/set folded into receive_signal)
