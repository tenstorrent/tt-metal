# Annotation — `reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp`

Path: `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/`
Role: **RECEIVER (pure).** Pairs with `_sender_` file. Receives weights + bias mcast.
Substrate: new object API.

> **RECALL NOTE:** the bare-recognition grep (`noc_async_write_multicast` etc.) **MISSES this file** —
> it has no mcast/flush/barrier primitives, only the receiver half spelled as `Semaphore` *methods*
> (`.set`, `.up`, `.wait`). The block grep must include the method spellings or it under-counts every
> pure-receiver kernel in conv. See return / recall-miss section.

---

## Variant signature (receiver half)

| Fork | Value | Lines |
|---|---|---|
| **F2** flag vs counter | **flag.** S→R: `wait(VALID)` exact `L185/202`, with own `set(INVALID)` reset `L178/196`. R→S: `up(...,1)` ack `L181/199`. | 178,181,185,196,199,202 |
| **F3** loopback | N/A (receiver). Sem args: `weights_mcast_sender_noc_x/y` (`L68-69`) — unicast back to the one sender. | 68,69 |
| **KNOB** pre_handshake | **YES** — receiver acks readiness (`up` `L181`) BEFORE waiting for data (`wait` `L185`), the R→S half of the sender's pre-handshake. | 181,185 |

## Setup
- `L68-71` `weights_mcast_sender_noc_x/y` (the single sender to ack), `Semaphore<>
  weights_mcast_sender_sem` (the ack target on the sender), `weights_mcast_receiver_sem` (own
  data-ready flag, written by the sender's `set_multicast`).

## Block — WEIGHTS receiver (`L174-188`), inside `if (bh == 0)`
- `L175` `cb_weight_obj.reserve_back` — reserve dest CB slot (the mcast will land here).
- `L178` `weights_mcast_receiver_sem.set(INVALID)` — **reset own data-ready flag low** before
  acking, so the upcoming `wait(VALID)` can't return on a stale VALID from last round (H3 / A6).
- `L181` `weights_mcast_sender_sem.up(noc, sender_noc_x, sender_noc_y, 1)` — **R→S ack** (A8 remote
  atomic inc): tell the sender "my dest slot is reserved, you may mcast". KNOB pre_handshake half.
- `L185` `weights_mcast_receiver_sem.wait(VALID)` — **S→R wait** (A6 exact). Data already in
  `cb_weight` by INV4 (data-before-flag) once VALID is observed.
- `L188` `cb_weight_obj.push_back` — release the now-filled slot to compute.

## Block — BIAS receiver (`L191-205`), inside `if (load_bias)` — identical protocol
- `L193` reserve; `L196` `set(INVALID)`; `L199` `up(...,1)` ack; `L202` `wait(VALID)`; `L204`
  push_back. Same H3 reset + KNOB-ack + INV4-implicit-ordering shape.

## Holes
- None. Clean canonical **`Pipe::receive`** shape: reset-flag → ack → wait → consume.
