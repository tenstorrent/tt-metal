# reader_mcast_receiver_unary_sharded_gn_v2.cpp — remigrate to mcast_pipe v7

- Group: G1 groupnorm
- Role: receiver
- Commit: 720f0ede3ac93b542a5bb32796480775341af4df
- Status: migrated, migrated_api_version=7
- Validation nodeid: `tests/ttnn/unit_tests/operations/fused/test_group_norm.py::test_group_norm_with_block_sharded_v2_8x4_grid[specify_grid=True-legacy-N=1-C=1280-H=16-W=16-num_groups=32-device_params={'l1_small_size': 0}]`
- Result: PASS (JIT 0/29 hits = fresh compile against v7)

## Delta
Kernel was frozen at the intermediate (pre-v7) `ReceiverPipe<DATA_READY, CONSUMED>` 2-arg form.
v7 inserted `PRE_HANDSHAKE` as the 2nd template param, so the consumed-sem id had drifted into the
PRE_HANDSHAKE slot (would mis-bind + trip the static_assert). One-line fix:

```
- ReceiverPipe<reduce_sender_semaphore_id, reduce_receiver_semaphore_id> reduce_pipe(noc);
+ ReceiverPipe<reduce_sender_semaphore_id, /*PRE_HANDSHAKE=*/true, reduce_receiver_semaphore_id> reduce_pipe(noc);
```

No call-site lines removed (`.receive(x,y)` already v7-correct). GN sender sibling is `pending` (raw
primitives) so it co-compiles.

- diff_lines_removed: 0 (signature-only edit)

## Remigration v7->v8 (2026-06-20)
NO CODE CHANGE. At v8 only `SenderPipe` changed (dropped 3rd template param
`NUM_ACTIVE_RECEIVER_CORES`; fan-out derived from `McastRect::area()`; divergent ack -> runtime ctor
arg). `ReceiverPipe` is UNCHANGED. This kernel uses ONLY `ReceiverPipe<reduce_sender_semaphore_id,
/*PRE_HANDSHAKE=*/true, reduce_receiver_semaphore_id>` — confirmed no `SenderPipe<` usage. The gn_v2
SENDER twin is a deferred design-gap (multi-rect runtime num_dests) and was NOT touched.

Re-verified under MCAST_PIPE_API_VERSION 8:
- Smoke (--dev) `[...-legacy-...C=1280-H=16-W=16-num_groups=32...]` -> 1 passed.
- Family (--run-all) `test_group_norm_with_block_sharded_v2_8x4_grid` -> 6 passed (3 legacy + 3 welford), no hang.
- JIT-build confirmed: `grep -rl reader_mcast_receiver_unary_sharded_gn_v2 generated/` (legacy variant hits this kernel).

Ledger: migrated_api_version 7->8, last_verified=2026-06-20, commit kept
(720f0ede3ac93b542a5bb32796480775341af4df — code unchanged). diff_lines_removed: 0.
