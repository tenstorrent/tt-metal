# sdpa_decode/dataflow_common.hpp — annotation (transformer/sdpa_decode)

Path: `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp`
Role: **shared header**. The Pipe block lives inside `read_k<...>(...)` (template), lines 573-661.
Consumer: `reader_decode_all.cpp` (only caller of read_k). Migration unit = this header function,
NOT the .cpp.

This is a DISTINCT dialect from the sdpa chain pattern: **F1 = BARRIER**, vertical-column
rectangle (same x, y0..y1), and the sender/receiver split is one function with a `do_mcast` flag.

## KMcastParams struct (lines 562-571)
do_mcast (true=sender, false=receiver), mcast_x, mcast_y0, mcast_y1, num_dests, mcast_sem_addr,
mcast_sem_ptr. Vertical multicast: same x, y-range.

## THE BLOCK — sender half (`do_mcast == true`, lines 600-654)
- L601-627 read K^T chunk from DRAM into the source CB (with interior read barriers L621-624 + final L627).
- L629-634 `get_noc_multicast_addr(mcast_x, mcast_y0, mcast_x, mcast_y1, k_write_ptr)` — vertical rectangle.
- L635-640 `noc_async_write_multicast(k_write_ptr, dst_mcast_addr, k_chunk_tiles*k_tile_bytes, num_dests, /*linked=*/false)` — **linked=FALSE** (no companion-linked pairing; data and sem are separate barrier-fenced steps).
- L642 `noc_async_write_barrier()` — **F1 = BARRIER** (ensures data lands before signalling).
- L644-645 `noc_semaphore_set(mcast_sem_ptr, VALID)` — set own flag (self/loopback include — see F3).
- L646-652 `get_noc_multicast_addr(...mcast_sem_addr)` + `noc_semaphore_set_multicast(mcast_sem_addr, sem_mcast_addr, num_dests, false)` — sem-flag mcast, linked=false.
- L653 `cb_push_back(cb_k_in, k_chunk_tiles)` — sender pushes its own copy (loopback INCLUDE behavior).
- L654 `noc_async_write_barrier()` — second barrier, drain the sem mcast.

## THE BLOCK — receiver half (`do_mcast == false`, lines 655-661)
- L657 `noc_semaphore_wait(mcast_sem_ptr, 1)` — wait for VALID flag.
- L658 `noc_semaphore_set(mcast_sem_ptr, 0)` — F2 flag reset.
- L659 `noc_async_atomic_barrier()` — barrier the inbound mcast write before consuming.
- L660 `cb_push_back(cb_k_in, k_chunk_tiles)`.

## VARIANT TAGS
- **F1 = BARRIER** (`noc_async_write_barrier`, L642+L654; receiver `noc_async_atomic_barrier` L659).
  This is the OPPOSITE fork from the sdpa chain kernels (which use FLUSH).
- **F2 = FLAG** (VALID(1)/0 reset). Sender SETs its own ptr to VALID (L645) then mcasts; receiver
  waits-on-1 then resets-0. Level flag, single-shot per chunk.
- **F3 = INCLUDE_SRC (loopback)**: sender SETs its own sem VALID (L645) and pushes its own CB (L653),
  i.e. it counts itself as a destination of the logical broadcast. NOTE: the *data* mcast (L635) does
  NOT use loopback_src (linked=false, and the rectangle y0..y1 — whether it includes sender's own y is
  set host-side); the sender's own copy is the already-in-L1 DRAM-read data, so it does not need the
  data echoed. Effectively INCLUDE on the sem/CB-push side, self-fill on the data side.
- **KNOB pre_handshake = NO (fresh slot)**: sender does NOT wait for receivers before writing — it
  reads fresh DRAM into the CB and pushes once. There is no sender_sem ready-handshake here. Receivers
  do not signal readiness upstream. This is the FRESH-SLOT variant (vs chain_link's dest-reused).

## HAZARD / INVARIANT mapping
- **No-companion / barrier discipline**: because linked=false, data correctness depends on the
  L642 barrier completing BEFORE the sem mcast (L645-652). INVARIANT: barrier MUST separate data
  mcast from sem mcast. Contrast with chain_link where linked=true makes them an atomic pair with
  NO barrier allowed between. These two forks are mutually exclusive and a Pipe must pick one or
  dispatch on it.
- **One-way handshake**: receivers never signal readiness, so the sender can clobber the source CB
  on the next call only if cb_reserve_back (L594) blocks until consumers pop. Ordering relies on the
  CB back-pressure, NOT on a sender_sem. Pipe pre_handshake=NO mode must rely on CB reserve for safety.
- **Self-set before mcast** (L645 vs L652): sender sets local VALID first, then mcasts the same value
  to the rectangle. If the local set were ordered after a partial mcast it would be fine (idempotent),
  but Pipe should keep local-set-then-remote-mcast order.

## HOLES
- HOLE: two barriers (L642, L654) bracket the sem mcast — the L654 one is arguably redundant with the
  next iteration's reserve/back-pressure. Bake-off should measure whether F1=BARRIER here costs vs
  the FLUSH dialect.
- HOLE: linked=false + barrier vs linked=true + flush is the single biggest cross-family divergence.
  This is the F1 fork the tune-helper bake-off must resolve empirically (BH #19201-class hangs are
  cited elsewhere in this op as barrier-workarounds → barrier may be load-bearing on BH).
