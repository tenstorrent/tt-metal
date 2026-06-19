# persistent_h2d_receiver.cpp — annotation (added by reconcile 2026-06-19)

File: `models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/persistent_h2d_receiver.cpp`
Family: ccl / deepseek (host_io micro-op). Role: **sender** (the name says "receiver" — it receives
host→device data, then **multicasts it to worker cores**). Tag: **refactor**.

A persistent host-to-device receiver loop: pulls a block from the host staging region, then multicasts
it to the worker rectangle and bumps a counter the workers wait on. The counter (`inc_multicast`)
rather than a set-flag is the F2-counter spelling — same as `reader_dispatch`.

## BLOCK (sender leg)
| step | lines | call |
|------|-------|------|
| build worker / metadata mcast addrs | 68, 79 | `get_noc_multicast_addr(...)` ×2 |
| **mcast DATA to workers** | 128 | `noc_async_write_multicast(...)` |
| **mcast SEM (counter inc)** | 144 | `noc_semaphore_inc_multicast(worker_mcast_addr, /*incr=*/1, /*num_dests=*/num_workers)` |

## Forks
- **F2 = COUNTER** (`noc_semaphore_inc_multicast`, monotone; workers `wait_min`).
- **F3 = EXCLUDE_SRC** (`num_workers` recipient count; the h2d core is not a worker).
- **persistent loop** — the block repeats per host push (a streaming barrier flavour).

## Migration-blocker audit → REFACTOR
- **Counter (inc) mcast** — maps to `SenderPipe::send_signal` on the counter staging path (the
  Staging::Counter arm), not the set-flag arm. Confirm the helper's counter path is exercised (the
  reader_dispatch counter exemplar is the reference).
- **`models/demos/...` location** — outside the ttnn op tree; verify it has a runnable test / harness
  before it can be device-verified by apply-dm-helper (coverage may be a gap).
- Persistent loop reuse of the source staging region → flush-before-reuse hazard; confirm the helper's
  flush fence covers it.
