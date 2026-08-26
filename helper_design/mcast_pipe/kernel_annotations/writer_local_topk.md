# writer_local_topk.cpp (TopK) — migrated API v10

Path: `ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_local_topk.cpp`

Role: local worker, and the **receiver** face of the readiness-only multicast.
The helper-facing role is intentionally different from the operation's local
writer name.

## Helper formulation

- `McastArgs<0, 0>` decodes the opaque helper CT/RT prefix. Operation CT fields
  chain from `next_compile_time_args_offset()` and `start_wt` is read from
  `next_runtime_args_offset()`.
- Per row, `readiness_pipe.receive_signal()` replaces the raw readiness
  `wait(VALID)` and local `set(INVALID)` reset.
- The adopted readiness semaphore is host-initialized to `INVALID` (`0`), so
  no receiver-kernel startup reset can erase the first no-handshake signal.

Only the readiness channel moved. Values and indices still use operation-owned
unicast writes into their final-core offsets; the write barrier, arrival
counter increment by `Kt`, atomic barrier, and CB ownership are unchanged.

## Validation

- Host build passed.
- Exact W=8192, k=50, BFLOAT16_B node passed under `--dev` from a fresh
  isolated cache; the `writer_local_topk` JIT artifact was confirmed.
- `TOPK-MULTICORE`: 14 passed and 12 expected BFLOAT8_B pad xfails.
- `McastHostFixture.*`: 25/25; `test_mcast_pipe.py`: 77/77.

Current exact-node profile: TopK device-kernel duration 238,281 ns. The writer
shares its 238,280 ns BRISC profiler processor envelope with
`writer_final_topk`, so it cannot
be isolated as a per-kernel duration, and no operation-matched pre-migration
TopK bakeoff exists. Therefore the per-kernel delta is explicitly N/A.
