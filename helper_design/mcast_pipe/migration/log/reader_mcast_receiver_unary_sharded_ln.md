# reader_mcast_receiver_unary_sharded_ln — v7→v8 re-verify (NO code edit)

**Tier:** 0b (layernorm sharded), v7→v8 REMIGRATION run.
**Status:** migrated, migrated_api_version=8 (RE-VERIFIED, code unchanged).

## Transform: NONE
`ReceiverPipe` is UNCHANGED across v7->v8 (only `SenderPipe` dropped a template arg).
Receiver-only kernel: no edit. Bumped migrated_api_version 7 -> 8 after device re-verify.

Phase-2 monotone `wait_min(block+2)` stays raw (shared-cell counter reuse, matches sender twin) —
unchanged, as before.

## Validation
`tests/ttnn/unit_tests/operations/fused/test_layer_norm_sharded.py::test_layer_norm_sharded_single_stage`
- Full (--run-all): 64 passed, 0 failed, no hang. (Same run that validated the sender twin.)
- JIT build confirmed (kernel present in generated/).
