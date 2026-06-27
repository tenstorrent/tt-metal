# Annotation — `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp`

Path: `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/`
Role: **reader, HYBRID (sender + receiver + loopback in one file)** — the conv generality stress test (mirror of the matmul `in0_sender_receiver` hybrid). Activation mcast.
Substrate: new object API (`Noc`, `Semaphore<>`, `MulticastEndpoint`, `McastDst`).

> NOTE — this kernel hoists the block into **two file-local helper functions**: `multicast_data`
> (`L22-49`, the F3 dispatcher) and `mcast_block_chunked` (`L61-132`, the burst-split data mcast).
> They are NOT in the shared `conv_reader_common.hpp` — they are private to this kernel, so the
> migration unit is still this file. But they are a *near-`Pipe`-send* already (see below).

---

## Variant signature

| Fork | Value | Lines |
|---|---|---|
| **F1** flush vs barrier | **MIXED / dual-path.** Data+flag mcast path: **neither** (relies on INV4 VC-4 FIFO, comment `L288-290`). Degenerate-rect local-write path: **barrier** (`L127-131`, `async_write_barrier`). Loopback flag path adds an explicit `async_write_barrier` `L303`. | 129, 303 |
| **F2** flag vs counter | **flag (level VALID/INVALID + reset).** S→R: `set(VALID)`+`set_multicast` (`L295-296`/`L306-307`); receiver `wait(VALID)` `L328`. R→S: `up`+ sender `wait(N)` exact `L276` + `set(0)` reset `L277`. | 276,277,279,295,306,318,328 |
| **F3** loopback | **all THREE sub-cases, runtime-dispatched** in `multicast_data` `L32-48`: (a) `is_receiver_core && num_cores>0` → **INCLUDE_SRC loopback** `L34`; (b) `is_receiver_core && num_cores==0` → **local `async_write` unicast-to-self** `L38` (loopback-with-1-dest hang dodge = INV5); (c) `!is_receiver_core` → **EXCLUDE_SRC** `L46`. | 34,38,46 |
| **KNOB** pre_handshake | **YES (dest reused).** Sender `wait`s receivers' ack before mcasting (`L276`) — round-robin self-mcast reuses `cb_act` dest across `act_w_num_outer` iters. | 276 |

---

## Helper `multicast_data` (`L22-49`) — the F3 dispatcher (= INV5 + H7 + H8 in one place)
- `L31` `src = use<READ_PTR>(src_cb)` — source L1 = tilized block read ptr.
- `L32-35` is_receiver & num_cores>0 ⇒ **INCLUDE_SRC** mcast, `num_dests = act_mcast_num_cores+1`
  (counts self — **H8 num_dests rule for INCLUDE_SRC**). Protocol step: SENT-enqueue (H1 deferred to caller).
- `L36-44` is_receiver & num_cores==0 ⇒ **INV5 degenerate guard**: loopback-1-dest may hang, so
  fall back to a plain `async_write` to **own** `{my_x,my_y,dst.addr}`. Hazard mitigated: **H5**.
- `L45-48` !is_receiver ⇒ **EXCLUDE_SRC**, `num_dests = act_mcast_num_cores+1` (sender not a dest but
  still must reach the output-only core — see header comment `L20-21`). **H7 M7a path.**
- HOLE? No. This is exactly the `Pipe::send` F3 dual-path, hand-rolled. **Direct migration target.**

## Helper `mcast_block_chunked` (`L61-132`) — burst-split streaming send
- `L75-90` compile-time burst arithmetic (full bursts, leftover, tiles-to-wait). Bookkeeping.
- `L93-98` builds `McastDst` once, `.addr` advanced per burst (`L107`) — the "ANY dst addr" carrier.
- `L102-119` per-burst loop: `src_cb.wait_front(wait_tile_curr)` (`L103`, CB-producer sync, NOT the
  mcast handshake) → `multicast_data(...)` (`L104`) → advance `src_offset`/`dst.addr`. Establishes
  **invariant: only mcast tiles already produced by compute** (streams a block bigger than one NoC
  burst). This is a `Pipe`-feature the current sketch does NOT have: **chunked send below
  `NOC_MAX_BURST_SIZE`**. → catalog amendment candidate (see return).
- `L120-124` leftover partial burst.
- `L126-131` **F1 barrier** only on the degenerate local-write path (`num_dest_cores==0`): there is
  no flag mcast to fence it, so an explicit `async_write_barrier` is the only ARRIVED proof. Mitigates
  **H1 + the missing-INV4** (no flag ⇒ must barrier).

## Main block (`L266-334`, inside the `act_w_num_outer` round-robin)
- `L205` `act_mcast_receiver_sem.set(VALID)` (once, pre-loop) — stages the level value that
  `set_multicast` will broadcast. **A4/A5 dual-role: local cell = mcast source.**
- `L270` `cb_act.reserve_back` — dest CB slot (consumer side).
- **SENDER branch `L271-314`:**
  - `L276` `act_mcast_sender_sem.wait(act_mcast_num_dests + (is_receiver_core?0:1))` — **R→S
    pre-handshake (KNOB pre_handshake / H2):** wait until every receiver acked "ready". The
    `+1` when `!is_receiver_core` accounts for the output-only core that also acks (H8 accounting).
  - `L277` `set(0)` — **flag reset (M3a)** so next round's `wait` doesn't re-trigger on stale count (H3).
  - `L279` `act_mcast_receiver_sem.set(INVALID)` — pre-arm the level flag low before the data mcast,
    so receivers' `wait(VALID)` can't see a stale VALID (H3 on the S→R flag).
  - `L281-286` `mcast_block_chunked(...)` — the data mcast (enqueued, chunked).
  - `L292-304` is_receiver loopback: `set(VALID)` (`L295`) then `set_multicast<INCLUDE_SRC>` (`L296`),
    then **`async_write_barrier` `L303`** — because the flag's own source is the sem cell (A5
    gotcha) and this is the loopback path, it barriers to be safe (F1=barrier here).
  - `L305-313` non-receiver: `set(VALID)` + `set_multicast` (EXCLUDE_SRC default), **no barrier**
    (INV4 VC-4 FIFO carries data-then-flag). F1=neither.
  - **INV4 preserved:** data mcast (`L286`) issued before flag mcast (`L296/307`) on same `Noc`/VC.
    Comment `L288-290` states the same-NoC-same-VC reasoning explicitly.
- **RECEIVER branch `L315-329`:**
  - `L318` `set(INVALID)` — reset own flag low before signalling readiness (H3).
  - `L321-325` `act_mcast_sender_sem.up(noc, sender_x, sender_y, 1)` — **R→S ack** (A8 remote inc).
    `transpose_mcast` swaps x/y (geometry, not protocol).
  - `L328` `act_mcast_receiver_sem.wait(VALID)` — **S→R wait** (A6 exact). Protocol: data already
    landed (INV4) by the time VALID is seen.
- `L330` `cb_act.push_back` — release dest slot to compute.
- `L333` `cb_tilized_in0.pop_front` — drain the source CB.
- `L346` final `async_write_barrier` — drain any in-flight writes at kernel exit.

## Unclassifiable / out-of-block (NOT mcast handshake)
- `L237-262` `reserve_done_sem` / `write_done_sem` (`set(VALID)` `L238`, `wait/set(INVALID)`
  `L260-261`): **intra-core split-reader shared-CB handshake** between the two readers, local NoC
  unicast-free sem ping-pong. NOT part of the `Pipe` block. Flag-style, but a different channel.
  → **HOLE w.r.t. mcast** (correctly excluded; note it so the migration doesn't swallow it).
