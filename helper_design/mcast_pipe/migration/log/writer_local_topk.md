# writer_local_topk.cpp — mcast_pipe v7→v8 re-verify (receiver, NO code edit)

- Group: G2 topk / Tier 0d
- Role: receiver (ReceiverPipe, control-only via `receive_signal()`)
- Commit: a362b90343a (UNCHANGED — kept from v7 migration; no code edit this round)
- Status: migrated, migrated_api_version=8 (re-verified)

## v7→v8: no change
`ReceiverPipe` is UNCHANGED across the v7→v8 (Round 10 D2 count split) move — the
fan-out/count split only touched `SenderPipe`'s template signature. This receiver-only
kernel needs NO code edit; it is re-verified against the v8 helper and its ledger
api_version bumped to 8 while keeping its existing commit hash.

The kernel still uses `ReceiverPipe<receiver_sem_id, /*PRE_HANDSHAKE=*/false>` +
`ready_pipe.receive_signal()` (waits the invite flag VALID, clears INVALID in one
call). The unicast data scatter (values + indices `noc.async_write`), the fan-in ack
`sender_sem.up(...)` (multi-producer counter, INV9), and the trailing atomic barrier
remain raw (Pipe does not own them).

## Validation (shared device-verify with reader_final_topk)
- `tests/ttnn/unit_tests/operations/reduce/test_topk.py::test_topk`
- Smoke (`--dev`): W=8192 k=50 BFLOAT16_B multicore → PASS (exercises this receiver).
- Full (`--run-all`): 80 passed, 80 xfailed, 0 failed.
- JIT-built confirmed: `grep -rl writer_local_topk generated/` → kernel_names.txt,
  kernel_elf_paths.txt, watcher.log, inspector logs.
