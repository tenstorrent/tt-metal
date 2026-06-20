# reader_final_topk.cpp — remigrate to mcast_pipe v7

- Group: G2 topk
- Role: sender (control-only)
- Commit: 15c9fa959849196d0f1ae4f501370444d7cd33d4
- Status: migrated, migrated_api_version=7
- Validation nodeid: `tests/ttnn/unit_tests/operations/reduce/test_topk.py::test_topk[sub_core_grids=None-largest=True-sorted=True-N=1-C=1-H=32-W=8192-dim=3-k=50-BFLOAT16_B]`
- Result: PASS (JIT 16/30 hits — the topk reader recompiled fresh)

## Delta
Frozen at the intermediate `SenderPipe<NUM, DATA_READY, CONSUMED>` order with un-templated `McastRect{}`
and a `send_signal(VALID)` taking an arg. This is a pure control-only doorbell (the fan-in counter
`sender_sem` is a separate multi-producer channel kept raw, not owned by the pipe), so PRE_HANDSHAKE=false
and the consumer sem is omitted.

```
- SenderPipe<num_dests, receiver_sem_id, sender_sem_id> ready_pipe(
-     noc, McastRect{noc_start_x, noc_start_y, noc_end_x, noc_end_y});
+ SenderPipe<noc_index, receiver_sem_id, num_dests, /*PRE_HANDSHAKE=*/false> ready_pipe(
+     noc, McastRect<>{noc_start_x, noc_start_y, noc_end_x, noc_end_y});
...
- ready_pipe.send_signal(VALID);
+ ready_pipe.send_signal();
```

The raw `Semaphore<> receiver_sem(receiver_sem_id)` at line 27 is a pre-existing dead local (unused
before and after); left untouched to keep the diff to Pipe constructs only.

- diff_lines_removed: 0 net (2 ctor lines + 1 call rewritten in place)
