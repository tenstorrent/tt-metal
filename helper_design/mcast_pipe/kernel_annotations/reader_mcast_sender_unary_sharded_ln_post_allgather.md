# reader_mcast_sender_unary_sharded_ln_post_allgather.cpp (SEND side) — CLEANEST EXEMPLAR

Path: ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln_post_allgather.cpp

API spelling: experimental OO wrapper.
Role: post-allgather sender. Mcasts the final [E[x],E[x^2]] to all cores then sets a flag. This is the **canonical minimal send() block** — no gather, no loops, INCLUDE_SRC.

## Block map (top-level L70-75, two lambdas L38-68)

### THE BLOCK — set up src → data mcast → flag mcast → barrier, INCLUDE_SRC
- L70 `cb_stats_reduced_obj.wait_front(stats_tiles*block_h)` — wait local stats ready (producer = compute). **set up source L1.**
- L71 `cb_ex_global_obj.reserve_back(stats_tiles*block_h)` — **fresh dest slot**.
- L72 `global_reduce_sender(...)`:
  - L54 `cb_ex_global_obj_inner.get_read_ptr()` — src addr.
  - L55-66 `noc.async_write_multicast<Noc::McastMode::INCLUDE_SRC>(cb_ex_obj, mcast_ep, stats_tiles*num_tiles_per_worker_bytes, num_blocks, {}, {...}, false)` — **DATA mcast, INCLUDE_SRC (F3 loopback), linked=false, num_dests = num_blocks (includes self)**.
  - L67 `noc.async_write_barrier()` — **F1 = barrier** inside data-mcast lambda.
- L73 `cb_ex_global_obj.push_back(...)` — hand the (self-filled, via INCLUDE_SRC) result to local compute.
- L74 `cb_stats_reduced_obj.pop_front(...)`.
- L75 `global_semaphore_set()`:
  - L39 `reduce_sender_sem.set(VALID)` — arm flag locally (also sets own slot since INCLUDE_SRC mcast covers self).
  - L40-47 `reduce_sender_sem.set_multicast<Noc::McastMode::INCLUDE_SRC>(... num_blocks, false)` — **FLAG mcast, INCLUDE_SRC, num_dests = num_blocks**.
  - L48 `noc.async_write_barrier()` — **F1 = barrier** after flag mcast.

## Variant signature
- **F1 = barrier** (L67 after data, L48 after flag).
- **F2 = LEVEL FLAG only** (`set(VALID)` + flag mcast; receiver does `set INVALID`/`wait VALID`). **NO counter, NO wait_min.** This is the pure flag staging style.
- **F3 = INCLUDE_SRC** (loopback — sender is also a destination; data + flag both mcast with INCLUDE_SRC so sender self-fills via the mcast itself, not a separate self-read).
- **pre_handshake = NONE.** **This is the no-pre-handshake / no-R→S-wait case**: the sender mcasts into a **fresh `reserve_back` CB slot** (L71) WITHOUT any prior receiver→sender handshake. Receivers were pre-positioned (their `reserve_back` + `wait(VALID)` blocks until this flag). Sender never waits on receivers before mcasting. **THE canonical "fresh slot, no pre-handshake" exemplar.**

## Hazards / invariants
- INV: data mcast (L55) MUST be barriered (L67) before the flag mcast (L40) — receiver keys off the flag to read the data. Ordering: data → data-barrier → flag → flag-barrier.
- INV: INCLUDE_SRC means `num_dests = num_blocks` (NOT num_blocks-1). Off-by-one vs EXCLUDE_SRC kernels is a real hazard the helper must encode per F3.
- CLEANLINESS: zero gather, zero internal loops, single pass → directly migratable to `Pipe::send(data, flag, INCLUDE_SRC, barrier)`.
