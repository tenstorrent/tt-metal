# reader_mcast_sender_unary_sharded_ln_post_allgather — v7→v8 remigration

> Historical v8 log. Current v9 status (2026-07-30): **blocked and restored to
> baseline**, together with its receiver. The existing 136-case inventory
> exercises only `mcast_1d`. On the accepted non-1D host path, the baseline
> `MCAST_INCL_SRC` operation loops back to a sender outside the receiver
> rectangle, while `SenderPipe` derives loopback only from rectangle
> membership. The raw host fan-out may also differ from the helper's
> rectangle-derived area on ragged grids. The baseline-restored pair passed all 136 mapped cases on
> 2026-07-30; that validates the rollback only because those cases remain
> `mcast_1d`.

**Tier:** 0b (layernorm sharded), v7→v8 REMIGRATION run.
**Status:** migrated, migrated_api_version=8, commit=65d3debd959.

## Transform: PURE DELETION
Dropped the 3rd `SenderPipe` template arg `NUM_ACTIVE_RECEIVER_CORES` (= `num_blocks - 1`).

### Why dense (pure deletion), the loopback case
- One-shot loopback (INCLUDE_SRC) `send()`: the sender reads its own reduced stats and writes
  them into cb_ex_global on ALL num_blocks cores INCLUDING itself (src != dst).
- The rect covers all num_blocks cores including the sender, so rect `area()` == num_blocks.
  The helper derives the EXCLUDE fan-out as `area - (in_rect?1:0)` == num_blocks - 1, which
  equals the old explicit count. `send()` detects the in-box src != dst and adds +1 for the
  self-copy (INCLUDE_SRC) automatically -> the loopback +1 the old code wanted is preserved.
- `PRE_HANDSHAKE=false` (each receiver mcasts into a fresh reserve_back slot) -> no consumer-ack
  count to override; nothing added to the ctor.

## Diff
- Template arg list: 4 -> 3 (removed `num_blocks - 1`). Comment rewritten. clang-format reflowed
  the call site (whitespace only).
- diff_lines_removed: 1 template arg + comment churn.

## Validation
`tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py::test_post_allgather_layernorm`
- Smoke (--dev): bf8b in/out/weights, 8x2 grid, num_devices=4, rmsnorm=True -> PASSED.
- Full (--run-all): 64 passed, 0 failed, no hang.
- JIT build confirmed.
