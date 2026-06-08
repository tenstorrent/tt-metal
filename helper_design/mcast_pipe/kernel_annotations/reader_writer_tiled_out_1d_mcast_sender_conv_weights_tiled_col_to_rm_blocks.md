# Annotation — `reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp`

Path: `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/`
Role: **SENDER (pure).** 1D weights mcast (col→RM blocks). Pairs with the `_receiver_` file.
Substrate: new object API. Block appears **twice** — weights (`L245-277`) and bias (`L305-337`),
identical protocol.

---

## Variant signature

| Fork | Value | Lines |
|---|---|---|
| **F1** flush vs barrier | **NEITHER** — relies on INV4 VC-4 FIFO (comment `L264-266`). No flush, no barrier on the mcast path. | (none) |
| **F2** flag vs counter | **flag (level + exact wait + reset).** R→S: sender `wait(weights_mcast_num_dests)` exact `L249/309` + `set(0)` reset `L250/310`. S→R: `set_multicast(VALID)` `L269/329` (receiver waits — in the paired file). | 249,250,269,309,310,329 |
| **F3** loopback | **EXCLUDE_SRC** (default `async_write_multicast`, no `<INCLUDE_SRC>`). Sender keeps its DRAM-read copy; `num_dests = weights_mcast_num_cores` excludes self. Comment `L253-254`/`L313-314`: "num_dests must not include source, since we are NOT really doing a local copy". | 255,269,315,329 |
| **KNOB** pre_handshake | **YES (dest reused).** Sender `wait`s the receiver-ack count BEFORE mcasting (`L249/309`) — receiver `cb_weight` dest reused across blocks. | 249,309 |

---

## Setup
- `L77-95` `McastRect` from rt args; `weights_mcast_num_dests`, `weights_mcast_num_cores`;
  `Semaphore<> weights_mcast_sender_sem`, `weights_mcast_receiver_sem`; `MulticastEndpoint mcast_ep`;
  `McastDst mcast_dst` prebuilt, `.addr` patched per call. ANY-rect/ANY-addr carrier.
- `L123` `weights_mcast_receiver_sem.set(VALID)` (once, guarded by `#ifndef SKIP_MCAST`) — stages the
  level value the later `set_multicast` broadcasts (A5 dual-role).

## Block — WEIGHTS (`L218-282`), inside `if (bh == 0)`
- `L218` `cb_weight_obj.reserve_back` — dest CB slot.
- `L226-242` DRAM read of the weight block into local L1 (TensorAccessor `async_read`), `L243`
  `async_read_barrier` — fills the **source** L1 before mcast (the "set up source" protocol step).
- `L249` `weights_mcast_sender_sem.wait(weights_mcast_num_dests)` — **R→S pre-handshake (H2/KNOB):**
  exact-wait until all receivers acked ready. **F2 flag (exact `wait`).**
- `L250` `set(0)` — reset ack counter (M3a-style reset, H3 on the ack cell).
- `L254` `mcast_dst.addr = cb_weight_obj.get_write_ptr()` — dest addr (ANY).
- `L255-262` `async_write_multicast(...)` EXCLUDE_SRC, `num_dests = weights_mcast_num_cores`,
  `linked=true`. **DATA mcast — enqueued (A1).** F3 = M7a (no loopback; sender keeps DRAM copy).
- `L269-276` `weights_mcast_receiver_sem.set_multicast(...)` — **FLAG mcast** (broadcasts VALID cell),
  `linked=false`. **INV4: issued AFTER data on same Noc/VC** ⇒ no barrier needed (comment `L264-266`).
- **F1 = neither.** H1 on the sem cell is tolerated because: the cell is re-`set(VALID)` only once
  (`L123`) and never overwritten between rounds here, and the data buffer (`cb_weight`) is fenced by
  the next round's CB `reserve_back` + the receiver-ack `wait` (`L249`). *Subtle — see HOLE below.*
- `L282` `cb_weight_obj.push_back` — release dest slot.

## Block — BIAS (`L286-339`), inside `if (load_bias)` — identical protocol
- `L288-302` DRAM read bias → source L1; `L302` read barrier.
- `L309` `wait(weights_mcast_num_dests)` + `L310` `set(0)` — R→S pre-handshake + reset.
- `L314` dst addr; `L315-322` EXCLUDE_SRC data mcast; `L329-336` flag `set_multicast`. Same INV4,
  same F1=neither.

## Holes / notes
- **HOLE (benign):** the receiver sem `set(VALID)` happens only once at `L123`, NOT re-staged before
  each `set_multicast` (unlike the hybrid readers which re-`set(VALID)` every round). This is correct
  **only because** the sender never sets the cell to any other value — the cell stays VALID for the
  kernel lifetime. If the `Pipe` re-uses the same sem cell for a counter or toggles it, this
  assumption breaks. Flag for the `Pipe` contract: *sender's flag-source cell must hold the broadcast
  value continuously across all rounds, OR be re-staged per round.*
- `L249`+`L269` (and `L309`+`L329`) are the canonical **`Pipe::send`** shape: pre-ack-wait → data
  mcast → flag mcast, no barrier. The cleanest sender in the conv group.
