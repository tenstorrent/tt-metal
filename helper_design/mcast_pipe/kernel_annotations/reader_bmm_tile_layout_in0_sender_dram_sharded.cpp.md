# reader_bmm_tile_layout_in0_sender_dram_sharded.cpp — annotation

Role: **HYBRID** — three runtime-selected core types in one kernel: (1) mcast-sender-no-compute, (2) mcast-sender+compute (sender ∈ receiver rect), (3) mcast-receiver+compute. The block appears in all three forms. Object API.

## Fork signature (varies BY core type — this is the stress)
- **F1**: FLUSH (type-2 only, conditional). Type-1 has NO flush between data and flag mcast (L115-135) — relies purely on same-noc/vc in-order. Type-2 adds `async_writes_flushed()` after the flag mcast (L203) and only when `in0_mcast_num_cores > 1`. Final `async_write_barrier()` + `async_atomic_barrier()` at L236-237.
- **F2**: LEVEL FLAG with per-type INVALID reset. `receiver_sem.set(VALID)` pre-loop (L67). Type-2/3 set INVALID each iter (L147, L227) then `wait(VALID)` (L213, L231). `sender_sem.wait(...)` + `set(0)` reset (L87-88, L154-155). Counter-ish on the receiver→sender side: receivers `sem.up(+1)` (L210, L229), sender waits for the count.
- **F3**: BOTH variants present in one file:
  - Type-1 (sender NOT in rect): EXCLUDE_SRC, `num_cores - 1`, default `async_write_multicast` (L115-119).
  - Type-2 (sender IN rect): **INCLUDE_SRC loopback**, `async_write_multicast<Noc::McastMode::INCLUDE_SRC>` (L178), `num_cores`, and INCLUDE_SRC flag mcast (L194).
- **KNOB pre_handshake**: YES. `sender_sem.wait` precedes data mcast (L87 / L154).

## Protocol steps
- L58-62: ctors `Noc`, `cb_in0`, `cb_in2` (sharded src), `sender_sem` (cta4), `receiver_sem` (cta5).
- L67: **invariant** `receiver_sem.set(VALID)`.
- L71: `local_read_addr = cb_in2.get_read_ptr()` — source is the sharded CB, no per-iter fill read (data already resident).
- **Type 1 (L73-138)**: per block: capture write ptr (double-buffer offset L80-82) → **R→S wait+reset** (L87-88) → pad last ktile (L92-110) → **data-mcast EXCLUDE_SRC** (L115-126) → **flag-mcast** (L129-135). No flush, no receiver-wait (this core never consumes).
- **Type 2 (L140-215)**: per block: `reserve_back` → `receiver_sem.set(INVALID)` (L147) → if sender: R→S wait `wait(num_dests-1)`+reset (L154-155), pad, **data-mcast INCLUDE_SRC** (L178), `receiver_sem.set(VALID)` (L192) + **flag-mcast INCLUDE_SRC** (L194) + **flush** (L203); else: `sender_sem.up(...)` to remote sender (L210). Then **receiver-wait** `receiver_sem.wait(VALID)` (L213), `push_back`.
- **Type 3 (L216-234)**: pure receiver: `reserve_back` → `set(INVALID)` (L227) → **receiver-signal-back** `sender_sem.up(noc, sender_x,sender_y,1)` (L229) → **receiver-wait** `receiver_sem.wait(VALID)` (L231) → `push_back`.
- L236-237: final `async_write_barrier()` + `async_atomic_barrier()` (atomic barrier needed for the `sem.up` increments).

## SKIP_MCAST
Only the `async_write_multicast` calls are guarded (L112/L176); the flag mcast and handshake remain — so SKIP_MCAST here means "skip data move, keep sync." Asymmetric vs the padding kernels.

## HOLEs
- L80-82 double-buffer pointer offset (`block_id % 2`) — manual double-buffering of the mcast source addr, computed outside CB API. Classify as **sender-fill bookkeeping** but flag as a generality wrinkle: the Pipe `send()` source addr is not always `cb.get_write_ptr()`; here it's a hand-rolled ping-pong over `cb_in2`.
