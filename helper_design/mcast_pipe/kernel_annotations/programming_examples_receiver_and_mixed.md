# (programming_examples) in0_receiver_in1_receiver / in0_receiver_in1_sender / in0_sender_in1_receiver — annotation

Three pedagogical readers, **RAW C API**, that are the mirror halves of in0_sender_in1_sender. Annotated together (structurally identical halves; line refs from grep).

## reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp — RECEIVER ×2
- in0 block: `noc_semaphore_set(in0_receiver_ptr, INVALID)` (L79) → **signal-back** `noc_semaphore_inc(in0_sender_noc_addr, 1)` (L84) → **receiver-wait** `noc_semaphore_wait(in0_receiver_ptr, VALID)` (L87).
- in1 block: `set(INVALID)` (L95) → `noc_semaphore_inc(...,1)` (L99) → `wait(VALID)` (L102).
- Fork: F1 N/A; F2 LEVEL FLAG exact-wait; F3/pre_handshake = receiver back-half (inc before wait).
- `noc_semaphore_inc` is the RAW spelling of `Semaphore::up`.

## reader_bmm_tile_layout_in0_receiver_in1_sender.cpp — RECEIVER (in0) + SENDER (in1)
- in0 = receiver block (set INVALID L95 / inc L100 / wait VALID L103).
- in1 = sender block: pre-set local VALID (L78); R→S `noc_semaphore_wait(sender_ptr, num_dests)` + `set(0)` (L133-134); data-mcast `noc_async_write_multicast` via `get_noc_multicast_addr` (L137,L144); flush BH (L153); flag-mcast `noc_semaphore_set_multicast` (L157,L164).
- This single file is BOTH a receiver (in0) and a sender (in1) — but for DIFFERENT operands (clean separation, unlike block-sharded where the SAME operand flips role per block). Still informs the API: one kernel may instantiate a `receive()`-Pipe and a `send()`-Pipe simultaneously.

## reader_bmm_tile_layout_in0_sender_in1_receiver.cpp — SENDER (in0) + RECEIVER (in1)
- Mirror of the above: in0 = sender block (wait L115 / set0 L116 / mcast L118,L125 / flush L134 / flag-mcast L138,L145); in1 = receiver block (set INVALID L154 / inc L158 / wait VALID L161).

## Fork signatures (all)
F1 FLUSH (BH-only between data/flag) · F2 LEVEL FLAG (exact wait, reset on R→S side) · F3 EXCLUDE_SRC · pre_handshake YES (sender) / inc-before-wait (receiver).

## HOLEs
- None. These are textbook clean halves. Key generality note: in0/in1 mixed files prove the API needs to support **two independent Pipe instances per kernel** (distinct semaphore pairs, distinct CBs), which is straightforward — distinct from the block-sharded same-operand role-flip which is NOT.
