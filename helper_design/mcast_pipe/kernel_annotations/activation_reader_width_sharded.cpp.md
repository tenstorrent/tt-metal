# Annotation — `activation_reader_width_sharded.cpp`

Path: `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/`
Role: **reader, HYBRID (sender + receiver + loopback)** — width-sharded activation mcast, round-robin.
Substrate: new object API.

---

## Variant signature

| Fork | Value | Lines |
|---|---|---|
| **F1** flush vs barrier | **barrier** (sender, after flag mcast). `async_write_barrier` `L253`. No flush anywhere. | 253 |
| **F2** flag vs counter | **MIXED.** S→R = **flag** (`set(VALID)`+`set_multicast`+receiver `wait(VALID)`+receiver `set(INVALID)` reset). R→S = **counter** (`up` + sender **`wait_min`** `L213`, then `set(0)`). | 213,214,216,245,257,271 |
| **F3** loopback | **INCLUDE_SRC loopback** (`L227`, data) and `L246` (flag). Sender ∈ rect, fills self. No EXCLUDE_SRC / degenerate path in this file. | 227,246 |
| **KNOB** pre_handshake | **YES (dest reused).** Sender `wait_min`s receivers' "ready" count before mcasting (`L213`); `act_cb` dest reused across the `num_input_cores` round-robin. | 213 |

> **F2-MIXED is the salient finding here**: this kernel uses a **monotone-counter `wait_min`** on the
> R→S ack (`L213`, `num_mcast_cores-1`) but a **level flag `wait(VALID)`** on the S→R data-ready
> (`L271`). The `set(0)` at `L214` resets the counter each round, so it is *really* a counter used as
> a per-round flag (wait_min then immediately reset) — a hybrid that the `Pipe` must decide to
> canonicalize. Contrast the weights kernels which use exact `wait(N)` (true flag) for R→S.

---

## Block, block-by-block

### Setup
- `L114-121` `MulticastEndpoint mcast_ep` + `McastDst mcast_dst` prebuilt from `mcast_rect`; `.addr`
  patched per call (`L226`). The "ANY rect + ANY dst addr" carrier.
- `L124` `act_mcast_receiver_sem.set(VALID)` — stages the level value for later `set_multicast` (A5
  dual-role). Done once before the loop.

### Round-robin self-mcast loop (`L205-276`)
- `L206` `act_cb.reserve_back(act_block_num_tiles)` — dest CB slot (consumer/producer pairing).

#### SENDER branch (`act_w_outer_i == this_core_id`, `L207-253`)
- `L213` `act_mcast_sender_sem.wait_min(num_mcast_cores - 1)` — **R→S pre-handshake (H2 / KNOB):**
  block until every *other* core has incremented its readiness ack. `wait_min` (A7) ⇒ counter style.
- `L214` `act_mcast_sender_sem.set(0)` — reset counter for next round (so the next `wait_min`
  threshold is met only by fresh acks). Mitigates **H3 on the ack counter**.
- `L216` `act_mcast_receiver_sem.set(INVALID)` — pre-arm S→R level flag low (H3 on data-ready flag).
- `L219` `tilized_in0_cb.wait_front(act_block_num_tiles)` — wait for compute to tilize the block
  (CB-producer sync, NOT mcast). Establishes invariant: only mcast a fully-tilized source.
- `L222-223` `tilized_src = use<READ_PTR>(tilized_in0_cb)` — source L1 (the tilized block).
- `L226` `mcast_dst.addr = act_cb.get_write_ptr()` — dest = receiver CB write ptr (ANY dst addr).
- `L227-234` `async_write_multicast<INCLUDE_SRC>(...)` `num_dests = num_reader_cores` (counts self,
  **H8 INCLUDE_SRC rule**), `linked=true`. **DATA mcast — enqueued (A1).** F3 = M7b loopback.
- `L245` `act_mcast_receiver_sem.set(VALID)` — re-stage the level value (the cell IS the mcast src, A5).
- `L246-252` `set_multicast<INCLUDE_SRC>(...)` — **FLAG mcast** (broadcasts the VALID cell). INV4:
  issued AFTER data mcast on same `Noc`/VC ⇒ data-before-flag at every receiver.
- `L253` `async_write_barrier()` — **F1 = barrier.** Comment `L242-244` explains the *reason*: because
  the new `Semaphore::set_multicast` API uses the sem cell itself as src (can't clear-and-wait-loopback
  like old code), it barriers instead. Mitigates **H1 (sem-cell clobber) → ACKed.**

#### RECEIVER branch (`L254-272`)
- `L257` `set(INVALID)` — reset own data-ready flag low before acking (H3).
- `L260-265` compute sender physical coords from iteration index + lookup tables (geometry).
- `L268` `act_mcast_sender_sem.up(noc, sender_x, sender_y, 1)` — **R→S ack** (A8 remote atomic inc).
- `L271` `act_mcast_receiver_sem.wait(VALID)` — **S→R wait** (A6 exact). Data landed by INV4.
- `L274` `act_cb.push_back` — release dest slot.

### Teardown
- `L277` `tilized_in0_cb.pop_front` — drain source CB.
- `L281-282` final `async_read_barrier` + `async_write_barrier` — flush at exit.

## Unclassifiable / HOLE
- None. Every primitive call maps to a protocol step. The F2-MIXED counter/flag split is a deliberate
  design choice, not a hole — flagged above for the `Pipe` canonicalization decision.
