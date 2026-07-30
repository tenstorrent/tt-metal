# persistent_h2d_writer.cpp — current-tree annotation

File: `models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/persistent_h2d_writer.cpp`
Family: ccl / deepseek (host_io micro-op). Role: **sender**. Tag: **refactor**.

> Current-tree replacement for the historical `persistent_h2d_receiver.cpp`
> entry added by the 2026-06-19 reconcile. Line references below describe that
> historical implementation; the current writer retains the same multicast
> roles but now splits socket reading onto `persistent_h2d_reader.cpp`.

A persistent host-to-device writer loop consumes staged socket pages, writes
them to the backing tensor, optionally multicasts metadata to the worker
rectangle, and bumps the worker counter.

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

## Current migration blockers
- **GlobalSemaphore targeting** — the metadata/data-ready addresses are
  GlobalSemaphore L1 addresses; `SenderPipe` binds program semaphore ids and
  cannot target them.
- **Counter (inc) mcast** — maps to the counter signal path only after the
  address-model gap is resolved.
- **`models/demos/...` location** — outside the ttnn op tree; verify it has a runnable test / harness
  before it can be device-verified by apply-dm-helper (coverage may be a gap).
- Persistent loop reuse of source storage requires the helper's source-lifetime
  fence; sender-consumed loopback would additionally require ACKed completion.
