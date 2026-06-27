# reader_mcast_sender_unary_sharded_ln — v7→v8 remigration

**Tier:** 0b (layernorm sharded), v7→v8 REMIGRATION run.
**Status:** migrated, migrated_api_version=8, commit=2b662e1ba1f.

## Transform: PURE DELETION
Dropped the 3rd `SenderPipe` template arg `NUM_ACTIVE_RECEIVER_CORES` (= `num_blocks - 1`).

### Why dense (pure deletion), not divergent
- Phase-1 sender broadcasts a FLAG-ONLY doorbell via `send_signal()`.
- Sender sits ABOVE the receiver rect (rect starts one row below) -> sender NOT in box.
- Therefore rect `area()` == `num_blocks - 1` == the EXCLUDE fan-out the helper derives.
  The old explicit count equalled the derived fan-out -> pure deletion.
- `PRE_HANDSHAKE=false` -> there is no consumer-ack wait, so `consumer_ack_count` is never
  consulted; nothing added to the ctor.
- Phase-2 (monotone `set(block+2)` streaming) remains DEFERRED RAW (unchanged) — reuses the
  same sem cell as a counter; Flag/Counter Pipe verbs cannot express per-side without
  desyncing the receiver base.

## Diff
- Template arg list: 4 -> 3 (removed `num_blocks - 1`). Comment block rewritten.
- diff_lines_removed: 1 template arg + comment churn.

## Validation
`tests/ttnn/unit_tests/operations/fused/test_layer_norm_sharded.py::test_layer_norm_sharded_single_stage`
- Smoke (--dev): bf16, welford=True, h=256 w=512, 4x4 cores -> PASSED.
- Full (--run-all): 64 passed, 0 failed, no hang.
- JIT build confirmed (kernel present in generated/).
