# reader_mcast_sender_unary_gn.cpp (SEND side) — interleaved (non-sharded) gn, 3-pass

Path: ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_sender_unary_gn.cpp

API spelling: experimental OO wrapper, `CoreLocalMem`, `TensorAccessor` (interleaved DRAM reads), multi-rectangle mcast.
Role: interleaved gn coordinator. 3 passes (mean/var/final) over batch×group, with out_blocks chunking. The mcast+handshake block runs only for cur_read_iteration 0 (mean) and 1 (var).

## Block map (nested loops: batch L302 → group L306 → cur_read_iteration L312)

### Pre: arm flag, optional zero-fill
- L196 `reduce_sender_sem.set(VALID)` — flag armed once.
- L297-299 `if needs_cb_ex_external_zero_fill: zero_whole_cb(...)` — startup zero-fill (HOLE, not Pipe).

### Phase-1 gather + handshake (counter) — inside iter 0/1, EXCLUDE_SRC
- L350/L352 `cb_ex_partial.wait_front(1)` / `cb_ex2_partial.wait_front(1)`.
- L362-367 self scalar read (HOLE).
- L369-381 `if num_mcast_cores>1`: L371 `reduce_receiver_sem.wait(num_mcast_cores-1)` COUNTER wait; L372 `reduce_receiver_sem.set(0)` reset; L375-380 remote scalar reads (HOLE).
- L382-386 pop partial.
- L388-417 `if num_mcast_cores>1`: **EARLY FLAG mcast** — `reduce_sender_sem.set_multicast(... mid/first/last rects)` (L389, L398, L408). NOTE: this flag mcast happens here, BEFORE the data mcast below, signalling gather-complete to receivers. **This is a flag-only multicast at this point (no data yet).**

### Phase-2 data mcast (multi-rect) + flag mcast again — L447-524, EXCLUDE_SRC, BARRIER
- L451-458 `cb_ex.wait_front(1)` / `cb_ex2.wait_front(1)` — combined result ready.
- L461-468 data mcast MID rect (`async_write_multicast`, linked=true).
- L469-476 flag mcast MID rect.
- L478-496 first-group data+flag mcast.
- L498-516 last-group data+flag mcast.
- L517 `noc.async_write_barrier()` — **F1 = barrier**.
- L518-522 pop cb_ex/cb_ex2.

### iter 2 (final pass): L418-444 — re-read output for residual add (HOLE, no mcast).

## Variant signature
- **F1 = barrier** (L517).
- **F2 = COUNTER (phase-1 gather `wait`/`set(0)`) + LEVEL FLAG (set once + repeated set_multicast)**. Flag is mcast TWICE per iter: once after gather (L388-417, flag-only) and once after data (L469-516, with data). The first is a "go" signal; the second releases the data. **TWO flag-mcasts per logical exchange** — finer-grained than the post_allgather single-flag pattern.
- **F3 = EXCLUDE_SRC** (self scalar via explicit self-read).
- **pre_handshake = YES** (counter gather sync precedes data mcast). Data mcast from producer-owned cb_ex slot.
- **MULTI-RECTANGLE** (mid/first/last), same as sharded_gn_v2.

## Hazards / invariants
- INV: the early flag mcast (L388-417) and the late data+flag mcast (L447-524) are separated by the out_block loop closing — the receiver must distinguish "gather done" from "data ready". Here both use the SAME `reduce_sender_sem`; the receiver (reader_mcast_receiver_unary_gn) does TWO `wait(VALID)`/`set(INVALID)` pairs per iter (one per flag mcast). **Pipe must support a two-step flag exchange (go-flag then data-flag) over a single semaphore.**
- HAZARD: zero-fill (L294-299) is a correctness prereq for the gather reduce; orthogonal to Pipe.
- NEW FORK: double-flag-per-exchange (go + release). Multi-rectangle. Data source = producer CB (not fresh).
