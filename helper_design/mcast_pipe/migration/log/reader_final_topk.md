# reader_final_topk.cpp — mcast_pipe v7→v8 remigration

- Group: G2 topk / Tier 0d
- Role: sender (control-only, flag broadcast via `send_signal()`)
- Commit: f8879d8c5ce4d5bda50a3257b9be7e32c684b0f9
- Status: migrated, migrated_api_version=8
- Validation nodeid: `tests/ttnn/unit_tests/operations/reduce/test_topk.py::test_topk[sub_core_grids=None-largest=True-sorted=True-N=1-C=1-H=32-W=8192-dim=3-k=50-BFLOAT16_B]`
- Result: smoke `--dev` PASS (1.31s); `--run-all` 80 passed / 80 xfailed / 0 failed

## v7→v8 transform (Round 10 — D2 count split)
**Dense case → PURE DELETION** of the 3rd template arg `NUM_ACTIVE_RECEIVER_CORES`.

```
- SenderPipe<noc_index, receiver_sem_id, num_dests, /*PRE_HANDSHAKE=*/false> ready_pipe(
-     noc, McastRect<>{noc_start_x, noc_start_y, noc_end_x, noc_end_y});
+ SenderPipe<noc_index, receiver_sem_id, /*PRE_HANDSHAKE=*/false> ready_pipe(
+     noc, McastRect<>{noc_start_x, noc_start_y, noc_end_x, noc_end_y});
```

### Why dense / pure deletion (decision evidence)
The deleted 3rd-arg expression was `num_dests`. The kernel comment documents that
the aggregator (this sender) sits OUTSIDE the mcast rect, so the EXCLUDE-source
fan-out the rect derives (`McastRect::area()`) equals `num_dests` exactly →
ack == fan-out (dense). Nothing added to the ctor.

Independently, `PRE_HANDSHAKE=false` (fire-and-forget flag doorbell via
`send_signal()`) makes `consumer_ack_count` inert — `ack_count_` is only consulted
under PRE_HANDSHAKE. So even if it were divergent, no ctor arg would be required on
this control path. Trailing `CONSUMER_READY_SEM_ID` correctly omitted (defaults to
`UNUSED_SEM_ID`; the `!PRE_HANDSHAKE || ...` static_assert is satisfied).

## Edit detail
- Deleted `num_dests` (3rd template arg).
- CT arg `num_dests = get_compile_time_arg_val(8)` LEFT IN PLACE — the host program
  factory still populates the same CT-arg vector; removing it would shift the CT-arg
  layout. Now an unused `constexpr` (no warning).
- Refreshed the comment block: rect-derived fan-out replaces the removed
  `NUM_ACTIVE_RECEIVER_CORES` description.
- diff_lines_removed: 0 net code (template arg deletion + comment rewrite in place).

## Validation
- W=8192 multicore case exercises BOTH reader_final_topk + writer_local_topk.
- All 80 xfails are pre-existing `BFLOAT8_B not supported by pad operation in topk`
  — unrelated to mcast.
- JIT-built confirmed: `grep -rl reader_final_topk generated/` → kernel_names.txt,
  kernel_elf_paths.txt, inspector logs.
