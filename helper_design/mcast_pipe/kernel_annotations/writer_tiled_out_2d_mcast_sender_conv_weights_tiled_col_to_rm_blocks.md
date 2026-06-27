# Annotation — `writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp`

Path: `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/`
Role: **SENDER (pure)**, 2D weights mcast. Pairs with `writer_..._2d_mcast_receiver_...`.
Substrate: new object API. Block appears **twice** — weights (`L250-281`) and bias (`L316-347`).

---

## Variant signature

| Fork | Value | Lines |
|---|---|---|
| **F1** flush vs barrier | **NEITHER** (INV4 VC-4 FIFO; comment `L269-271`). | (none) |
| **F2** flag vs counter | **flag.** R→S: sender `wait(weights_mcast_num_dests)` exact `L254/320` + `set(0)` reset `L255/321`. S→R: `set_multicast(VALID)` `L274/340`. | 254,255,274,320,321,340 |
| **F3** loopback | **EXCLUDE_SRC** (default mcast, comment `L257-258`/`L323-324` "num_dests must not include source"). | 260,274,326,340 |
| **KNOB** pre_handshake | **YES (dest reused)** — sender pre-acks-waits `L254/320`. | 254,320 |

This is the **2D twin of the 1D sender** — same mcast block, identical fork signature
(EXCLUDE_SRC / flag / no-barrier / pre_handshake). Mergeable into the same `Pipe::send`.

---

## Setup
- `L145` `weights_mcast_receiver_sem.set(VALID)` (once) — stage level value (A5). Same continuous-cell
  assumption as the 1D sender (see that file's HOLE note — applies here too).
- `mcast_rect` / `mcast_ep` / `mcast_dst` / sems built from rt args (header region).

## Block — WEIGHTS (`L228-283`)
- `L228` `cb_weight_obj.reserve_back` — dest CB slot.
- `L233-247` DRAM read weight block → source L1; `L248` `async_read_barrier` (set up source).
- `L254` `weights_mcast_sender_sem.wait(weights_mcast_num_dests)` — **R→S pre-handshake** (H2/KNOB,
  exact `wait` ⇒ F2 flag).
- `L255` `set(0)` — reset ack counter (H3 on ack cell).
- `L259` `mcast_dst.addr = cb_weight_obj.get_write_ptr()` — ANY dst addr.
- `L260-267` `async_write_multicast(...)` EXCLUDE_SRC, `num_dests = weights_mcast_num_cores`,
  `linked=true`. **DATA mcast — enqueued.** F3 = M7a.
- `L274-280` `set_multicast(VALID)` — **FLAG mcast.** INV4: after data, same Noc/VC, **no barrier**
  (comment `L269-271`). F1 = neither.
- `L282` `cb_weight_obj.push_back`.

## Block — BIAS (`L296-351`) — identical protocol
- `L302-313` DRAM read bias → source; `L313` read barrier.
- `L320` `wait(N)` + `L321` `set(0)`; `L325` dst addr; `L326-333` EXCLUDE_SRC data mcast;
  `L340-346` flag `set_multicast`. Same INV4 / F1=neither.

## Unclassifiable / out-of-block (NOT mcast handshake) — IMPORTANT
- `L183-215` **intra-core split-reader shared-CB handshake** (`split_reader_cb_shared` path):
  - `L185` `reserve_done_sem.wait(VALID)`, `L186` `reserve_done_sem.set(INVALID)` — this (sender)
    core waits for the *other* reader to signal "CB slot reserved".
  - `L214` `write_done_sem.set(VALID)` — signal back "I finished writing my half of the shared CB".
  - This is a **two-core producer/producer CB-coordination handshake**, local sem flag ping-pong,
    NOT a multicast and NOT part of the `Pipe` block. **HOLE w.r.t. mcast** — flag so the migration
    leaves it untouched. (Same channel as in the 2D activation reader's `reserve_done`/`write_done`.)
- `L238-245` DRAM `async_read` of weights — source fill, not part of the mcast primitive set.
