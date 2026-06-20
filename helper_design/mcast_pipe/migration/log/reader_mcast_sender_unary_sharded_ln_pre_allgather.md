# reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp — mcast_pipe migration (Tier 2a)

**Status:** migrated @ v7 | **Validation:** PASS

## What changed
C2 flag-only sender (no data payload). Migrated to `SenderPipe::send_signal()`.

- Added `#include ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp`.
- Dropped the raw `Semaphore<> reduce_sender_sem` (the pipe owns the flag cell + its VALID init).
- Constructed
  `SenderPipe<noc_index, reduce_sender_sem_id, num_blocks - 1, /*PRE_HANDSHAKE=*/false> reduce_pipe`
  with `McastRect<>{mcast_dest_noc_start/end}`.
  - EXCLUDE_SRC (sender outside the receiver rect), num_dests = `num_blocks - 1` (constexpr).
  - PRE_HANDSHAKE=false → CONSUMER_READY_SEM_ID omitted.
- Replaced `set(VALID)` + `set_multicast(EXCLUDE_SRC, num_blocks-1)` with `reduce_pipe.send_signal()`.
- KEPT raw: the consumer-drain gate `reduce_receiver_sem.wait(num_blocks-1)` / `set(0)` (protocol gate
  that must precede the flag broadcast); the interleaved gather reads after the signal (the HOLE).

## Validation
`tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py::test_pre_allgather_layernorm`
(actual path `fused/`; num_devices is simulated on one chip, so single-chip coverage is real).
- Smoke (`--dev`, bf16 num_devices=4 is_rmsnorm=False): PASS.
- Full family (`--run-all`, co-compiled with the receiver #5): **32 passed, 0 failed**, no hang.
- JIT build confirmed in `generated/inspector/kernels.yaml`.

## Lines
~5 raw flag lines (set + 6-arg set_multicast) → 1 `send_signal()`; net ~4 removed.
