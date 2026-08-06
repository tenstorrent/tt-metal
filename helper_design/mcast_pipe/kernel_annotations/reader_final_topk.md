# reader_final_topk.cpp (TopK) — migrated API v10

Path: `ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_final_topk.cpp`

Role: final/coordinator core, and the **sender** face of the readiness-only
multicast. The helper-facing role is intentionally different from the
operation's final-reader name.

## Helper formulation

- One sender-separate `Mcast2D` describes the local-worker rectangle; the final
  core is outside that rectangle and the sender uses `EXCLUDE_SRC` semantics.
- The adopted readiness semaphore is descriptor 1 and is explicitly
  host-initialized to `INVALID` (`0`). The channel is no-handshake Counter mode,
  so a sender cannot race a receiver-side kernel reset.
- `McastArgs<0, 0>` owns the opaque helper CT/RT prefix. Every operation CT
  field chains from `next_compile_time_args_offset()`.
- Per row, `readiness_pipe.send_signal()` replaces raw `set(VALID)`, semaphore
  multicast, and the readiness write barrier.

The arrival semaphore (descriptor 0) remains operation-owned. Its per-row
`set(INVALID)` and `wait(Wt_final)` pair is not part of the helper channel.
Circular-buffer reserve/push and the final write barrier are also preserved.

## Validation

- Host build passed.
- Exact W=8192, k=50, BFLOAT16_B node passed under `--dev` from a fresh
  isolated cache; the `reader_final_topk` JIT artifact was confirmed.
- `TOPK-MULTICORE`: 14 passed and 12 expected BFLOAT8_B pad xfails.
- `McastHostFixture.*`: 25/25; `test_mcast_pipe.py`: 77/77.

Current exact-node profile: TopK device-kernel duration 238,281 ns. The rollout
contains no operation-matched pre-migration TopK bakeoff, so a per-kernel delta
is not comparable; the generic F2 microkernel bakeoff is different work and
geometry and is not used as a baseline.
