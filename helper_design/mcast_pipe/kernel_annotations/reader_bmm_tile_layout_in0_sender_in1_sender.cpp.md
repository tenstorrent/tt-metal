# (programming_examples) reader_bmm_tile_layout_in0_sender_in1_sender.cpp — annotation

Role: **SENDER half ×2** (in0 and in1), pedagogical. **RAW C API** (not the object wrapper). Pairs with the receiver programming examples.

## Recall note
This is the canonical RAW spelling: `noc_async_write_multicast`, `noc_semaphore_set_multicast`, `noc_semaphore_wait/set`, `get_noc_multicast_addr`. The original grep family DID list these — but note the production matmul kernels do NOT use them (they use the object API), so a grep for only-raw OR only-object would miss half the corpus. See recall-misses in the return.

## Fork signature (both in0 and in1 blocks identical)
- **F1**: FLUSH. BH-only `noc_async_writes_flushed()` between data and flag (L149, L211). No final barrier in this simple example (relies on push_back + host teardown). In-order-NoC comment L143-145.
- **F2**: LEVEL FLAG. `*receiver_sem_ptr = VALID` direct write pre-loop (L79, L84). `noc_semaphore_wait(sender_ptr, num_dests)` + `noc_semaphore_set(sender_ptr, 0)` reset (L129-130, L191-192).
- **F3**: EXCLUDE_SRC. `noc_async_write_multicast(..., num_dests)` "must not include source" (L139, L201). Self-fill via `noc_async_read_tile` loop.
- **KNOB pre_handshake**: YES — `noc_semaphore_wait` precedes data mcast (L129 before L140).

## Protocol steps (in0 block; in1 is the mirror at L166-225)
- L77-79: **invariant** set local `receiver_sem` = VALID.
- L87-93: alias sender_sem ptrs (R→S readiness counters).
- L104-105: `cb_reserve_back` + `get_write_ptr` → `in0_start_address` (source). [sender-fill bookkeeping]
- L110-125: **sender-fill** `noc_async_read_tile` loop + `noc_async_read_barrier()`.
- L129-130: **R→S wait+reset** `noc_semaphore_wait(...,num_dests)` / `noc_semaphore_set(...,0)`.
- L133-141: build mcast addr `get_noc_multicast_addr(...)` (open-coded tell) + **data-mcast** `noc_async_write_multicast(...)`.
- L146-150: **flush** (BH).
- L153-161: build flag mcast addr + **flag-mcast** `noc_semaphore_set_multicast(...)`.
- L163: `cb_push_back`.
- in1 block (L166-225): identical structure.

## HOLEs
- None. Two clean back-to-back SENDER blocks. This is the cleanest reference of the canonical pattern.
