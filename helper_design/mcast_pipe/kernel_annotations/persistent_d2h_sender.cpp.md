# persistent_d2h_sender.cpp — annotation (added by reconcile 2026-06-27)

File: `models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/persistent_d2h_sender.cpp`
Family: **ccl / deepseek** (host_io micro-op). Role: **sender** (the name says "sender"; it pulls a tensor
from device DRAM, drains it to the host over PCIe, and multicasts a worker-sync counter). Tag: **refactor**.
Status: **PENDING**.

The device→host twin of the already-censused `persistent_h2d_receiver.cpp` (host→device). Same micro-op
family, mirror direction. A persistent loop: wait for `num_workers` write-acks, then for each socket page
read the tensor slice from DRAM into the scratch CB and wide-write it into the host-pinned socket FIFO over
PCIe, with an optional metadata page. The worker-sync block multicasts a `transfer_done` counter inc to the
worker rectangle.

## CARVE-OUT — what is in scope vs out of scope
- **OUT OF SCOPE (PCIe / socket bulk leg).** The bulk data drain is a PCIe wide-write-with-state into the
  host socket FIFO: `noc_write_init_state` (L59) + `noc_async_wide_write_any_len_with_state` (L117-123,
  L140-146) + `noc.async_writes_flushed()` + `socket_push_pages` / `socket_notify_receiver`. The kernel
  comment (L56-58) notes this path has NO Device-2.0 equivalent (`Noc::inline_dw_write` is single-DW), so
  it stays raw. Not a NoC rectangle mcast.
- **IN SCOPE (intra-chip worker-grid mcast).** The worker-sync block:

### THE BLOCK (worker-sync inc) — lines 76-82, `if constexpr (worker_sync_enabled)`
| step | lines | call |
|------|-------|------|
| build worker mcast addr | 62-69 | `get_noc_multicast_addr(... transfer_done_sem_addr)` |
| **mcast SEM (counter inc)** | 80 | `noc_semaphore_inc_multicast(worker_mcast_addr, /*incr=*/1, /*num_dests=*/num_workers)` |
| fence | 81 | `noc.async_atomic_barrier()` |
| wait num_workers write-acks | 84-95 | spin on `write_ack_ptr` (`cur - last_write_ack == num_workers`), termination-aware |

## Forks
- **F2 = COUNTER** (`noc_semaphore_inc_multicast`, monotone; workers `wait_min`). Same spelling as the
  h2d twin / `reader_dispatch`. R→S half is a plain spin on the local `write_ack` counter (workers `up`
  it remotely), not a mcast.
- **F1 = ATOMIC-BARRIER** (`noc.async_atomic_barrier()`, L81) — the correct fence after an atomic
  `inc_multicast` (counter path), distinct from a data write-barrier.
- **F3 = EXCLUDE_SRC** — `num_workers` recipient count; the d2h service core is not a worker.
- **persistent loop** — the block repeats per host push (a streaming barrier flavour); termination-aware.

## Migration-blocker audit → REFACTOR (apply-dm-helper will likely DEFER on coverage)
- **Twin of `persistent_h2d_receiver`** — that censused twin ended up **deferred + coverage-gap**
  (`models/demos/...` location, no clear runnable device test/harness for apply-dm-helper to verify).
  This d2h sender inherits the SAME coverage risk, so apply-dm-helper will most likely DEFER it for the
  same reason even though the block itself is migratable.
- **Counter (inc) mcast** — maps to the SenderPipe counter arm (Staging::Counter), not the set-flag arm;
  the `reader_dispatch` counter exemplar is the reference.
- **GlobalSemaphore targeting** — `transfer_done_sem_addr` is a GlobalSemaphore address; `Semaphore<>`
  binds per-program ids via `get_semaphore<>(id)` with no GlobalSemaphore wrapper, so
  `Semaphore::inc_multicast` cannot target it (kernel comment L77-79). The Pipe must accept a raw
  L1/GlobalSemaphore address or this block stays raw.
- **`models/demos/...` location** — outside the ttnn op tree; verify a runnable test/harness before
  device-verification. Coverage gap is the dominant blocker.
