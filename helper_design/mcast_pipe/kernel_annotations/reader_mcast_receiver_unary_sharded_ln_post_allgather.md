# reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp (RECEIVE side) — CLEANEST EXEMPLAR

Path: ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp

API spelling: experimental OO wrapper. 27 lines — the minimal receive() block.
Role: post-allgather receiver. Pre-positions a CB slot, waits for the sender's flag, accepts the multicasted data. NO gather, NO loop.

## Block map (top-level L23-26)

### THE BLOCK — arm flag → reserve dest slot → wait flag → push
- L23 `reduce_sender_sem.set(INVALID)` — clear the level flag so `wait(VALID)` blocks.
- L24 `cb_ex_global_obj.reserve_back(stats_tiles*block_h)` — **fresh dest slot pre-positioned BEFORE any handshake** (sender will mcast into this address; receiver only guarantees space + clears flag).
- L25 `reduce_sender_sem.wait(VALID)` — **LEVEL FLAG wait**: block until sender's INCLUDE_SRC flag mcast lands. When it returns, the data mcast (which the sender barriered before the flag) is guaranteed present.
- L26 `cb_ex_global_obj.push_back(stats_tiles*block_h)` — hand received data to compute.

## Variant signature
- **F1 = NONE** (no barrier/flush on receive — purely flag-gated).
- **F2 = LEVEL FLAG** (`set(INVALID)` / `wait(VALID)`). No counter, no wait_min. Pure flag style — the partner of the post_allgather sender.
- **F3 = INCLUDE_SRC** on the wire (the sender's INCLUDE_SRC mcast also writes the sender's own slot; from the receiver's POV it is a plain destination).
- **pre_handshake = NONE.** The receiver does NOT send any R→S signal before the data arrives. It just `set(INVALID)` + `reserve_back` + `wait(VALID)`. **This is the canonical no-pre-handshake receive() into a fresh slot.**

## Hazards / invariants
- INV: `set(INVALID)` L23 MUST precede `reserve_back` is irrelevant to ordering but MUST precede `wait(VALID)` L25 (else a stale VALID from a prior iter is consumed). reserve_back L24 between them is fine.
- INV: the data correctness depends entirely on the sender's ordering (data mcast → barrier → flag mcast). The receiver has no way to detect partial data; the flag is the sole gate.
- CLEANLINESS: 4-line block, single pass, no loop, no gather → the **reference receive() shape** for the Pipe helper. Pair directly with post_allgather sender.
