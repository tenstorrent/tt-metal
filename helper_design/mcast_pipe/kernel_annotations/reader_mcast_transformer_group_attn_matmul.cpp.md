# reader_mcast_transformer_group_attn_matmul.cpp — annotation

Role: **HYBRID sender OR receiver per `tile_row_id`** (the per-32-row sender rotates). Object API. group_attn_matmul op.

## Fork signature
- **F1**: **BARRIER, not flush** (the divergent one). After the flag mcast the sender does `noc.async_write_barrier()` (L305) with comment "Write barrier needed to make sure we finish sending mcast flag before we modify locally." BH still does an extra `async_writes_flushed()` between data and flag (L291). So this kernel uses *flush-between + barrier-after*, unlike the matmul senders' *flush-between + barrier-only-at-end*.
- **F2**: LEVEL FLAG. `receiver_sem.set(VALID)` pre-loop (L88); receiver `set(INVALID)` (L312) + `wait(VALID)` (L320); sender `sender_sem.wait(num_dests)` + `set(0)` (L234-235); receiver `sem.up(+1)` (L315).
- **F3**: BOTH, runtime-selected:
  - sharded + no-blocking + sender∈rect → **INCLUDE_SRC** loopback from the sharded CB directly (L243, `num_cores+1`).
  - sharded + no-blocking + sender∉rect → **EXCLUDE_SRC** from sharded CB (L257, `num_cores`).
  - otherwise (local-copy path) → **EXCLUDE_SRC** from `l1_write_addr_in1` (L272).
- **KNOB pre_handshake**: YES — `sender_sem.wait` precedes the data mcast (L234 before L238+).

## Protocol steps
- L73-75: ctors `Noc`, `cb_in1_obj` (mcast dest), `cb_in2_obj` (sharded src). Semaphores from RUNTIME args (L54-55), not compile-time — note for API (sem id source differs).
- L88: **invariant** `receiver_sem.set(VALID)`.
- L91-115: precompute `sender_sem_noc_x/y_vec[32]` — per-tile_row sender coords (rotating sender, like the block-sharded kernel).
- L121-132: `in1_sender_in_receiver_grid`, `mcast_in1_to_local_cb` predicates → choose loopback/self-mcast.
- Per tile_row_id (L165): `if (tile_row_id == in1_mcast_sender_id)` → **SENDER arm**:
  - L168: `cb_in1_obj.reserve_back`.
  - L176-228: **sender-fill** (sharded local copy via `noc.async_read` from own CB, or interleaved `noc.async_read` from DRAM) + `async_read_barrier`.
  - L234-235: **R→S wait+reset** `sender_sem.wait(num_dests)` / `set(0)`.
  - L238-283: **data-mcast** (3 F3 branches).
  - L288-292: **flush** (BH).
  - L295-301: **flag-mcast** `receiver_sem.set_multicast(...)`.
  - L305: **barrier** `noc.async_write_barrier()`.
- L306 `else if (in1_sender_in_receiver_grid)` → **RECEIVER arm**: `set(INVALID)` (L312) → **signal-back** `sender_sem.up(sender_sem_noc_x/y_vec[tile_row_id],1)` (L315) → **receiver-wait** `wait(VALID)` (L320).
- L322-329: `push_back` (and immediate `pop_front` for send-only cores — same lockstep invariant as block-sharded).

## SKIP_MCAST
No SKIP_MCAST guard here — mcast is unconditional.

## HOLEs
- Semaphores constructed from runtime args (L54-55) rather than compile-time — minor: Pipe ctor must accept a sem id from either source.
- Rotating per-tile_row sender (L91-115, L174) — same generality breaker as block-sharded: one binary, sender identity varies per iteration.
- No final barrier at kernel end beyond the per-iter L305 barrier — the per-iteration barrier subsumes it. Note: this is the *only* group member that barriers per-iteration; performance-relevant for the F1 bake-off.
