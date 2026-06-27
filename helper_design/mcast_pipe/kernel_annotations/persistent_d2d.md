# persistent D2D kernels — annotation (added by reconcile 2026-06-27)

Covers BOTH halves of the D2DStreamService persistent transfer pair:
- `ttnn/core/tensor/kernels/persistent_d2d_sender.cpp`
- `ttnn/core/tensor/kernels/persistent_d2d_receiver.cpp`

Family: **ccl / deepseek / examples**. Role: **sender** (both halves multicast to a worker grid).
Tag: **refactor**. Status: **PENDING**.

A persistent device-to-device stream service. Step-1 direct-DRAM: the sender writes the full tensor
STRAIGHT into the receiver mesh's DRAM backing tensor over tt-fabric — there is no receiver-side L1 FIFO
copy of the bulk data. Each side runs an intra-chip worker handshake (multicast counter) around the
fabric transfer.

## CARVE-OUT — what is in scope vs out of scope
This is the load-bearing distinction for the migration:

- **OUT OF SCOPE (fabric cross-chip leg).** The *bulk* tensor data, and the per-transfer "data-landed"
  signal (`bytes_sent`) and "overwrite-OK" signal (`bytes_acked`), all travel over `tt_fabric`
  (`WorkerToFabricEdmSender`, `to_noc_unicast_write` / `to_noc_unicast_atomic_inc`,
  `send_payload_flush_blocking_from_address`). Sender: `stream_half` (sender L177-206) +
  `flush_inc_data_landed` (L212-220). Receiver: the upstream `bytes_acked` inc (receiver L242-246). These
  belong to the `tt_fabric` / ring family — NOT a local NoC rectangle mcast. DEFER-RAW, same family as
  `all_reduce_create_qkv_heads/worker_writer` fabric leg.
- **Sender lane↔lane sync is NOT mcast either.** On Blackhole the sender forks into master (lane 0) + sub
  (lane 1) on one service core; they sync only through plain shared-L1 monotonic counters `go_count` /
  `done_count` (sender L344-347, L377-382, L422-425, L454) — volatile + `invalidate_l1_cache`, no NoC
  mcast. Out of scope.

- **IN SCOPE (intra-chip worker-grid mcast).** Each half has a genuine `mcast_pipe`-shaped block that
  multicasts to its local worker rectangle:

### Sender — master-only, consumed_sem mcast (sender L385-387)
| step | lines | call |
|------|-------|------|
| build worker-grid mcast addr | 283-288 | `get_noc_multicast_addr(... consumed_sem_addr)` (if `worker_sync_enabled`) |
| **mcast SEM (counter inc)** | 386 | `noc_semaphore_inc_multicast(consumed_mcast_addr, /*incr=*/1, /*num_dests=*/num_workers)` |
Releases the sender worker grid to overwrite the SENDER backing after the receiver acked. Flag-only
counter mcast; no data mcast.

### Receiver — metadata data-mcast + data_ready counter-mcast (receiver L197-236)
| step | lines | call |
|------|-------|------|
| build worker-grid mcast addr | 128-133 | `get_noc_multicast_addr(... data_ready_sem_addr)` |
| **mcast DATA (opt metadata blob)** | 198-208 | `noc.async_write_multicast(CoreLocalMem(read_ptr), MulticastEndpoint{}, metadata_size_bytes, num_workers, {}, {.noc_x_start=..., .addr=metadata_l1_addr})` |
| flush/barrier (data) | 209 | `noc.async_write_barrier()` |
| **mcast SEM (data_ready counter inc)** | 217 | `noc_semaphore_inc_multicast(worker_mcast_addr, /*incr=*/1, /*num_dests=*/num_workers)` |
| wait num_workers acks | 221-232 | spin on `consumed_ptr` (`cur - last_consumed == num_workers`) |

## Forks (in-scope blocks)
- **F2 = COUNTER** (`noc_semaphore_inc_multicast`, monotone; workers `wait_min`). Same counter spelling as
  `persistent_h2d_receiver` / `reader_dispatch`. R→S half is a plain spin on a local consumed-counter
  (workers `up` it), not a mcast.
- **F1 = atomic-barrier on the counter path** — `noc.async_atomic_barrier()` follows the inc on the
  twin h2d kernel; here the receiver relies on the wait loop. The receiver's metadata DATA mcast uses
  **F1 = write-barrier** (L209), and ordering REQUIRES the metadata barrier complete BEFORE the
  data_ready inc (L196 comment) so workers see consistent DRAM + metadata-L1 state.
- **F3 = EXCLUDE_SRC** — `num_workers` recipient count; the service core is not a worker.
- **persistent loop** — the block repeats per transfer (a streaming barrier flavour); both halves are
  termination-aware (`termination_semaphore`).
- **R1 multi-payload, single rect** — the receiver issues a data mcast THEN a counter mcast to the SAME
  worker bbox; one rectangle, two payloads (data + flag).

## Migration-blocker audit → REFACTOR (pending)
- **Fabric/intra-chip split** — only the worker-grid mcast block is migratable; the fabric data + ack
  legs must stay raw. A Pipe migration touches a small fragment of each kernel, not the data path.
- **Counter (inc) mcast** — maps to the SenderPipe counter arm (the Staging::Counter path), not the
  set-flag arm; same reference as `reader_dispatch` / `persistent_h2d_receiver`.
- **`ttnn/core/tensor/` location + persistent service** — verify a runnable device test/harness exists
  for D2DStreamService before apply-dm-helper can device-verify; coverage may be a gap (same risk as the
  h2d twin).
- **GlobalSemaphore targeting** — note the twin h2d kernel keeps `noc_semaphore_inc_multicast` raw
  because the target is a GlobalSemaphore address and `Semaphore<>` binds per-program ids via
  `get_semaphore<>(id)`; the same constraint likely applies here. The Pipe must be able to target a raw
  L1/GlobalSemaphore address, not only a program-id sem, or this block stays raw.
