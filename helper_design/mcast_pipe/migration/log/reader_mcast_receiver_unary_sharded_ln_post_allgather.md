# reader_mcast_receiver_unary_sharded_ln_post_allgather — v7→v8 re-verify (NO code edit)

**Tier:** 0b (layernorm sharded), v7→v8 REMIGRATION run.
**Status:** migrated, migrated_api_version=8 (RE-VERIFIED, code unchanged).

## Transform: NONE
`ReceiverPipe` is UNCHANGED across v7->v8. Receiver-only kernel: no edit. Bumped
migrated_api_version 7 -> 8 after device re-verify.

## Validation
`tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py::test_post_allgather_layernorm`
- Full (--run-all): 64 passed, 0 failed, no hang. (Same run that validated the sender twin.)
- JIT build confirmed.
