# Annotation — `writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp`

Path: `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/`
Role: **RECEIVER (pure)**, 2D weights. Pairs with `writer_..._2d_mcast_sender_...`.
Substrate: new object API.

> **RECALL NOTE:** like the 1D receiver, the bare grep MISSES this file (only `Semaphore` methods).

---

## Variant signature (receiver half)

| Fork | Value | Lines |
|---|---|---|
| **F2** flag vs counter | **flag.** S→R: `wait(VALID)` `L184/208` + own `set(INVALID)` reset `L178/202`. R→S: `up(...,1)` `L181/205`. | 178,181,184,202,205,208 |
| **KNOB** pre_handshake | **YES** — `up` ack `L181/205` before `wait` `L184/208`. | 181,184 |

## Setup
- `L54-55` `reserve_done_sem` / `write_done_sem` (compile-time, only when `split_reader_cb_shared`).
- `L69-75` `weights_mcast_sender_noc_x/y`, `weights_mcast_sender_sem` (ack target), `weights_mcast_receiver_sem` (own flag).

## Block — WEIGHTS receiver (≈`L174-188`)
- `cb_weight_obj.reserve_back` — reserve dest slot.
- `L178` `weights_mcast_receiver_sem.set(INVALID)` — reset own flag (H3).
- `L181` `weights_mcast_sender_sem.up(noc, sender_x, sender_y, 1)` — **R→S ack** (A8).
- `L184` `weights_mcast_receiver_sem.wait(VALID)` — **S→R wait** (A6). Data landed by INV4.
- push_back.

## Block — BIAS receiver (≈`L196-208`) — identical
- `L202` `set(INVALID)`; `L205` `up(...,1)`; `L208` `wait(VALID)`.

## Unclassifiable / out-of-block (NOT mcast)
- `L134-163` **intra-core split-reader shared-CB handshake** (the receiver-side counterpart):
  - `L134` `reserve_done_sem.wait(VALID)`, `L135` `reserve_done_sem.set(INVALID)`.
  - `L163` `write_done_sem.set(VALID)`.
  - Same two-reader CB coordination as the 2D sender writer / 2D activation reader. **HOLE w.r.t.
    mcast** — not part of `Pipe`.

## Holes
- Mcast half: none. Canonical `Pipe::receive` (reset → ack → wait → consume), twice.
