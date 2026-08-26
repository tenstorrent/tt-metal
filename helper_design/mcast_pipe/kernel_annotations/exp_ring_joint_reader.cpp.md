# exp_ring_joint_reader.cpp — annotation (transformer/sdpa)

Path: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_ring_joint_reader.cpp`
Role: reader (NCRISC). **Open-coded duplicate** of the chain_link.hpp Pipe pattern,
embedded inside a ring/all-gather outer loop. Two rectangle-mcast block instances (K, V).

IMPORTANT — two distinct sync mechanisms coexist in this file:
- **Rectangle-mcast chain forward** (IN SCOPE) — the Pipe block, K @ 299-320, V @ 364-400.
- **Ring/all-gather per-link sync** (OUT OF SCOPE) — `noc_semaphore_wait_min(per_link_sem_ptrs[lnk], ...)`
  at L271 and L430-431 (and the commented L430). This is the AllGather fused-op signaler
  (monotone counter, `wait_min`), a SEPARATE cross-chip/ring protocol, not a rectangle mcast.
  Do not fold it into Pipe.

## Setup (lines 101-122)
- L106 sender_sem noc addr → prev_physical (injector).
- L108-119 mcast branch: L111-116 `get_noc_multicast_addr(prev, next, 0)` rectangle base (injector only),
  L117 `mcast_sem_noc_addr = base | receiver_sem_addr`, L118 `sender_wait_count = mcast_sender_wait`.
- L120-121 unicast branch (OUT OF SCOPE).
- `should_forward`/`should_receive` computed L218-221.

## BLOCK 1 — K chunk

### Receiver half (lines 281-284)
- L282 `noc_semaphore_set(receiver_semaphore_addr_ptr, INVALID)`.
- L283 `noc_semaphore_inc(sender_semaphore_noc_addr, 1)`.
- L284 `noc_semaphore_wait(receiver_semaphore_addr_ptr, VALID)`.

### Sender half (lines 299-320)
- L300 `noc_semaphore_wait(sender_semaphore_addr_ptr, sender_wait_count)` — handshake (pre_handshake YES).
- L301 `noc_semaphore_set(sender_semaphore_addr_ptr, 0)` — F2 flag reset.
- L304-309 `noc_async_write_multicast(..., mcast_num_dests, true /*linked*/)`.
- L310 `noc_semaphore_set_multicast(valid_semaphore_addr, mcast_sem_noc_addr, mcast_num_dests)` — companion.
- L316 `noc_async_writes_flushed()` — **F1 = FLUSH**.
- L317-318 unicast remote sem set (OUT OF SCOPE).

## BLOCK 2 — V chunk

### Receiver half (lines 364-366) — identical 3-op shape.
### Sender half (lines 382-400)
- L382-383 handshake wait + reset.
- L386-392 mcast data (linked=true) + companion sem mcast L392.
- L398 `noc_async_writes_flushed()` — F1 FLUSH.
- L400 unicast remote sem set (OUT OF SCOPE).

## End-of-kernel flush (lines 434-435)
- L434 `noc_async_writes_flushed()` + L435 `noc_async_write_barrier()` — drain before exit. Generic
  teardown, not part of the per-iter block.

## VARIANT TAGS (both blocks)
- **F1 = FLUSH**.
- **F2 = FLAG** (VALID/INVALID + reset-0; wait_count = mcast_sender_wait).
- **F3 = EXCLUDE_SRC** (injector self-fills via fetch_block; `should_receive` false for injector).
- **KNOB pre_handshake = YES**.

## HAZARD / INVARIANT mapping
- **Linked companion ordering**: same as reader_interleaved — comment L297-298 ("before push_back").
  Data write (linked=true) + companion sem mcast back-to-back. INVARIANT shared with all chain kernels.
- **Ring sync interleave**: the `wait_min` ring sync (L271) gates the injector before it READS the
  chunk it will forward. Pipe::send must NOT subsume this — it is an upstream data-availability gate,
  orthogonal to the mcast handshake. Note as a co-located but separate protocol.
- mux-writer path (`is_mux_writer`, cb_k_writer_in) is fabric-MUX forwarding, OUT OF SCOPE.

## HOLES
- HOLE: F2 reset (L301) and ring wait_min reset (L431, `noc_semaphore_set(per_link_sem_ptrs[lnk],0)`)
  operate on different semaphores but share the "level flag" mental model — Pipe should be explicit
  that it owns ONLY the chain valid/sender sems, not the ring per-link sems.
