# reader_mcast_sender_unary_sharded_ln_pre_allgather — v7→v8 remigration

**Tier:** 0b (layernorm sharded), v7→v8 REMIGRATION run.
**Status:** migrated, migrated_api_version=8, commit=4c7f3b919f2.

## Transform: PURE DELETION
Dropped the 3rd `SenderPipe` template arg `NUM_ACTIVE_RECEIVER_CORES` (= `num_blocks - 1`).

### Why dense (pure deletion)
- FLAG-ONLY sender: broadcasts a doorbell via `send_signal()` (no data payload).
- Sender sits OUTSIDE the receiver rect, so rect `area()` == `num_blocks - 1` == the EXCLUDE
  fan-out the helper derives. Old explicit count == derived fan-out -> pure deletion.
- `PRE_HANDSHAKE=false` -> no consumer-ack wait; `consumer_ack_count` never consulted; nothing
  added to the ctor.
- The consumer-drain gate (`reduce_receiver_sem.wait`) + gather HOLE stay raw (unchanged).

## Diff
- Template arg list: 4 -> 3 (removed `num_blocks - 1`). Comment rewritten. clang-format reflowed
  the call site (whitespace only).
- diff_lines_removed: 1 template arg + comment churn.

## Validation
`tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py::test_pre_allgather_layernorm`
- Smoke (--dev): bf8b, 8x4 grid, num_devices=4, rmsnorm=True -> PASSED.
- Full (--run-all): 32 passed, 0 failed, no hang.
- JIT build confirmed.
