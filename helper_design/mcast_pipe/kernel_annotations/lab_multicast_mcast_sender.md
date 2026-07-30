# lab_multicast/mcast_sender.cpp — annotation (added by reconcile 2026-06-27)

File: `ttnn/examples/lab_multicast/kernels/dataflow/mcast_sender.cpp`
Family: **ccl / deepseek / examples**. Role: **SENDER** (didactic). Tag: **ref** (prior-art / didactic
example). Status: **deferred**.

A teaching example: reads tiles from DRAM and multicasts them to receiver cores with double-buffering.
Legacy raw free-function API (`noc_semaphore_*`, `get_noc_multicast_addr`, `noc_async_write_multicast`).
The same textbook single-sender shape as the censused `ref` example
`programming_examples/contributed/multicast/.../coordinator_kernel.cpp` — heavily commented for instruction,
not production. Like all didactic examples → **ref**, not a migration target.

## THE BLOCK (sender half) — main loop, lines 43-90
| step | lines | call |
|------|-------|------|
| reserve CB slot (double-buffer) | 45-46 | `cb_reserve_back` + `get_write_ptr` |
| read tile DRAM→L1 | 49-50 | `noc_async_read_page` + `noc_async_read_barrier` |
| push to CB | 53 | `cb_push_back` |
| pre-handshake: wait all receivers ready | 56 | `noc_semaphore_wait(receivers_ready_sem_ptr, num_receivers)` |
| reset ready counter | 57 | `noc_semaphore_set(receivers_ready_sem_ptr, 0)` |
| build mcast addr (data) | 71-72 | `get_noc_multicast_addr(rx0,ry0,rx1,ry1, cb_read_addr)` |
| **mcast DATA tile** | 73 | `noc_async_write_multicast(cb_read_addr, tile_mcast_addr, tile_size_bytes, num_receivers)` |
| flush (data SENT before flag) | 79 | `noc_async_writes_flushed()` |
| set local flag VALID | 82 | `*tile_sent_sem_ptr = VALID` |
| **mcast SEM flag** | 83 | `noc_semaphore_set_multicast(tile_sent_semaphore_addr, tile_sent_mcast_addr, num_receivers)` |
| barrier before freeing CB | 86 | `noc_async_write_barrier()` |
| free CB slot | 89 | `cb_pop_front` |

## Forks
- **F1 = FLUSH then BARRIER.** A `noc_async_writes_flushed()` (L79) orders the data mcast SENT before the
  flag mcast (the kernel comment L75-78 explains the separate-command-buffer ordering need); a trailing
  `noc_async_write_barrier()` (L86) gates the CB slot free. Demonstrates both fences explicitly — a
  teaching point.
- **F2 = FLAG** (level VALID: local `*ptr = VALID` L82, mcast set L83; receiver waits `==VALID` then
  resets). The pre-handshake side (L56-57) is a **counter** (`wait(num_receivers)` + reset to 0) — so this
  kernel mixes F2: counter for the ready-collect, flag for the data-ready signal.
- **F3 = EXCLUDE_SRC** — the sender core is the data source, not in the receiver rectangle. `num_receivers`
  is the full receiver count. No loopback.
- **KNOB pre_handshake = present** — receivers signal readiness into `receivers_ready_semaphore` before
  the sender multicasts; the CB-pointer-sync trick (comment L64-69) means the sender reuses its own CB
  read pointer as the dest address (receivers' CBs advance in lockstep).

## VERDICT
**REF / DEFER** — didactic example, double-buffered single-sender star, raw API. Closest sibling to the
`ref` `coordinator_kernel.cpp`; a clean reference of the flag-style sender (flush-then-barrier, CB-pointer
sync) but not a fleet migration target. Useful as a Pipe-shape reference, not migrated.
