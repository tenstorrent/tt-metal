# welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp — remigrate to mcast_pipe v7

- Group: G1 groupnorm
- Role: receiver
- Commit: 7e8e0a90bcb036f5a43c02c9431470e3d5e42087
- Status: migrated, migrated_api_version=7
- Validation nodeid: `tests/ttnn/unit_tests/operations/fused/test_group_norm.py::test_group_norm_with_block_sharded_v2_8x4_grid[specify_grid=True-legacy-N=1-C=1280-H=16-W=16-num_groups=32-device_params={'l1_small_size': 0}]`
- Result: PASS (validated together with the non-welford G1 receiver; JIT fresh compile)

## Delta
Same pre-v7 `ReceiverPipe<DATA_READY, CONSUMED>` 2-arg form. v7 fix — insert PRE_HANDSHAKE:

```
- ReceiverPipe<reduce_sender_semaphore_id, reduce_receiver_semaphore_id> reduce_pipe(noc);
+ ReceiverPipe<reduce_sender_semaphore_id, /*PRE_HANDSHAKE=*/true, reduce_receiver_semaphore_id> reduce_pipe(noc);
```

`.receive(x,y)` already v7-correct. diff_lines_removed: 0.

## Remigration v7->v8 (2026-06-20)
NO CODE CHANGE. At v8 only `SenderPipe` changed; `ReceiverPipe` is UNCHANGED. This kernel uses ONLY
`ReceiverPipe<reduce_sender_semaphore_id, /*PRE_HANDSHAKE=*/true, reduce_receiver_semaphore_id>` —
confirmed no `SenderPipe<` usage. The gn_v2 welford SENDER twin
(`welford_reader_mcast_sender_unary_sharded_gn_v2.cpp`) is a deferred design-gap (multi-rect runtime
num_dests) and was NOT touched.

Re-verified under MCAST_PIPE_API_VERSION 8:
- This kernel is exercised by the `welford` variant of the 8x4 grid test.
- Smoke (--dev) `[...-welford-...C=1280-H=16-W=16-num_groups=32...]` -> 1 passed.
- Family (--run-all) -> 6 passed (3 legacy + 3 welford), no hang.
- JIT-build confirmed: `grep -rl welford_reader_mcast_receiver_unary_sharded_gn_v2 generated/`
  -> kernels.yaml, programs_log.yaml, kernel_elf_paths.txt (after the --dev welford smoke).

Ledger: migrated_api_version 7->8, last_verified=2026-06-20, commit kept
(7e8e0a90bcb036f5a43c02c9431470e3d5e42087 — code unchanged). diff_lines_removed: 0.
