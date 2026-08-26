# reader_interleaved.cpp — annotation (transformer/sdpa)

Path: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp`
Role: reader (NCRISC). **Open-coded duplicate** of the `chain_link.hpp` Pipe pattern,
inlined. Two block instances: K-chunk and V-chunk forward/receive.

This is a PRIME migration target — it is `ring_joint_reader.cpp` BEFORE the ChainLink
extraction. Same protocol, hand-rolled.

## Setup (lines 160-186)
- L166 `*valid_semaphore_addr_ptr = VALID` — valid-flag init.
- L168-181 mcast branch: L170 sender_sem noc addr (to prev_physical = injector),
  L173-178 `get_noc_multicast_addr(prev, next, 0)` rectangle base (injector only),
  L179 `mcast_sem_noc_addr = base | receiver_sem_l1_addr`, L180 `sender_wait_count = mcast_sender_wait`.
- L182-185 unicast branch (OUT OF SCOPE).

## BLOCK 1 — K chunk

### Receiver half (lines 393-400)
- L395-396 cb_reserve_back / get_write_ptr.
- L397 `noc_semaphore_set(receiver_semaphore_addr_ptr, INVALID)`.
- L398 `noc_semaphore_inc(sender_semaphore_noc_addr, 1)`.
- L399 `noc_semaphore_wait(receiver_semaphore_addr_ptr, VALID)`.
- L400 cb_push_back.

### Sender half (lines 447-468 + completion 508-518)
- L448 `noc_semaphore_wait(sender_semaphore_addr_ptr, sender_wait_count)` — handshake (pre_handshake YES).
- L449 `noc_semaphore_set(sender_semaphore_addr_ptr, 0)` — F2 flag reset.
- L452-457 `noc_async_write_multicast(..., mcast_num_dests, true /*linked*/)`.
- L458 `noc_semaphore_set_multicast(valid_semaphore_addr, mcast_sem_noc_addr, mcast_num_dests)` — companion.
- L459 `noc_async_writes_flushed()` — **F1 = FLUSH**.
- L460-462 cb_push_back guarded by `!should_receive` (loopback EXCLUDE — injector pushes its own copy).
- L508-518 unicast-only completion (`if constexpr (!mcast_enabled)`): flush + remote sem set. OUT OF SCOPE.

## BLOCK 2 — V chunk

### Receiver half (lines 547-554) — identical shape to K (L551-553 the 3 sem ops).
### Sender half (lines 606-630)
- L607-608 handshake wait + reset.
- L609-617 mcast: data write (linked=true L616) + companion sem mcast L617.
- L623 `noc_async_writes_flushed()` — F1 FLUSH.
- L624-626 unicast remote sem set (OUT OF SCOPE).
- L627-629 cb_push_back guarded `!should_receive`.

## VARIANT TAGS (both blocks)
- **F1 = FLUSH**.
- **F2 = FLAG** (VALID/INVALID + reset-0; wait_count = mcast_sender_wait = num receivers).
- **F3 = EXCLUDE_SRC** (injector self-fills via DRAM read path; `should_receive` false for injector;
  cb_push_back gated on `!should_receive` so injector pushes its own buffer).
- **KNOB pre_handshake = YES** (sender waits before write; source CB slot reused).

## HAZARD / INVARIANT mapping
- **Linked companion ordering** (L443-446 comment, authoritative): data mcast (linked=true) and
  companion sem mcast MUST be back-to-back with NO read barrier between — a `noc_async_read_barrier()`
  in the gap DEADLOCKS (read barrier blocks while linked write awaits companion). This is why the
  mask read (L470-506) is placed AFTER the linked pair completes, and Q subblock push (L527-542)
  after K forward. INVARIANT the Pipe MUST encode: send() emits data+sem as an uninterruptible pair.
- **BH read-barrier-vs-inflight-write** (L524-526 comment): `noc_async_read_barrier` inside a read
  deadlocks on BH when NOC writes are in-flight → forward must fully flush before subsequent reads.
- **cb_push ownership** (L460/513/627): only the originator (`!should_receive`) pushes; a pure
  receiver's push happens in its own receive half. Pipe must keep CB push OUT of send/receive or
  parameterize it.

## HOLES
- HOLE: mcast and unicast share the same `should_forward` scaffold but split cb_push timing
  (mcast pushes inside forward block L461; unicast inside completion block L514). Pipe unifying
  these must pick one push site.
- HOLE: no barrier after flush; relies on downstream compute ordering (same as chain_link.hpp).
