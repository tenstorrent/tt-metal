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
