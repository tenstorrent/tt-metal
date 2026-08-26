# writer_local_topk.cpp — MIGRATED API v10 (verified 2026-08-06)

Atomic unit: `topk-multicore-final-readiness`; helper-facing role: readiness
receiver. The factory prepends the complete no-handshake Counter `Mcast2D`
helper CT/RT block and explicitly initializes adopted readiness descriptor 1 to
`INVALID` (`0`).

`McastArgs<0, 0>` reports both opaque boundaries. The kernel chains operation CT
fields from `next_compile_time_args_offset()`, reads `start_wt` from
`next_runtime_args_offset()`, and replaces readiness wait/reset with
`receive_signal()`. Value/index unicast, its data write barrier, the
operation-owned arrival-counter increment, atomic barriers, and CB ownership
are unchanged.

Validation:

- `./build_metal.sh`: passed.
- Exact W=8192, k=50, BFLOAT16_B node under `--dev` from a fresh isolated
  cache: passed; `reader_final_topk` and `writer_local_topk` artifacts confirmed.
- `TOPK-MULTICORE`: 14 passed, 12 expected BFLOAT8_B pad xfails, 26 selected.
- `McastHostFixture.*`: 25/25; `test_mcast_pipe.py`: 77/77.

Current exact-node profiling reports a 238,281 ns TopK device-kernel duration.
There is no operation-matched pre-migration TopK bakeoff, and the 238,280 ns
writer/BRISC envelope also contains `writer_final_topk`; therefore the
per-kernel delta is explicitly N/A.

Production diff: kernel **+18 / -19**; atomic unit **+77 / -75**.
`ledger.json` records production migration commit `b5c99d43fd5`.

## Historical v8 record

> Historical v8/v9 record. At the v9 checkpoint: **blocked and at baseline** because the
> paired no-handshake receiver initialization is not race-free.

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
