# reader_mcast_receiver_unary_sharded_ln.cpp (RECEIVE side)

Path: ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln.cpp

API spelling: experimental OO wrapper (`Noc`, `Semaphore<>`, `UnicastEndpoint`, `CircularBuffer`).
Role: non-coordinator receiver counterpart to reader_mcast_sender_unary_sharded_ln.cpp. Block in `global_reduce_receiver` lambda L122-242; runs up to twice (mean L246, var L250).

## Block map

### Phase-1 handshake (S↔R), level flag + counter-up, EXCLUDE_SRC
- L138 `cb_partial_obj.wait_front(...)` — local partial ready.
- L140 `reduce_sender_sem.set(INVALID)` — clear the level flag locally (so the upcoming `wait(VALID)` blocks until sender mcasts VALID).
- L141 `reduce_receiver_sem.up(noc, in0_remote_noc_x[0], in0_remote_noc_y[0], 1)` — **COUNTER inc** to the sender (remote unicast inc). This is the receiver→sender signal that the sender's `wait(num_blocks-1)` counts.
- L142 `reduce_sender_sem.wait(VALID)` — **LEVEL FLAG wait**: block until sender's mcast set VALID arrives.

### Gather (NoC reads) L154-213 — HOLE, unicast reads of remote partials, not Pipe block.
- L218-230: after gather, `cb_*.wait_front` then `reduce_receiver_sem.up(...)` (L221, L229) or `reduce_second_stage_sem.up(...)` (L224) — second counter-up signalling combine-done to coordinator.

### Phase-2 receive multicast (streaming), MONOTONE COUNTER `wait_min`
- L236-241 loop over `num_all_to_all_workers`:
  - L238 `cb_ex_global_obj.reserve_back(...)` — fresh dest slot per streamed block.
  - **L239 `reduce_sender_sem.wait_min(block+2)`** — **MONOTONE COUNTER wait_min**: pairs with sender L257 `set(block+2)`. Each streamed block released as the mcast counter crosses its threshold.
  - L240 `cb_ex_global_obj.push_back(...)` — hand the received block to compute.

## Variant signature
- **F1 = n/a on receive** (no flush/barrier; receiver only waits + push). The data lands via sender's mcast; receiver does NOT barrier.
- **F2 = phase-1 LEVEL FLAG (`set INVALID`/`wait VALID`) + counter-up; phase-2 MONOTONE COUNTER (`wait_min`)**. Confirms the two-phase flag-then-counter split.
- **F3 = EXCLUDE_SRC** (receiver is a pure destination).
- **pre_handshake**: phase-1 IS a full handshake (set INVALID, up, wait VALID). Phase-2 receive has NO per-block pre-handshake — it relies purely on the monotone counter crossing. The dest slots are **fresh `reserve_back`** each block (L238). **This is exactly the "fresh CB slot, no R→S wait before each streamed block" case** flagged in the brief, on the receive side.

## Hazards / invariants
- INV: `set(INVALID)` (L140) MUST precede `up()` (L141) MUST precede `wait(VALID)` (L142). Reordering races: if VALID arrives before set(INVALID), the wait could consume a stale VALID.
- INV: `wait_min(block+2)` consumes a *level*, not a token — receiver must not reset the flag during phase-2 (it doesn't). The flag is monotone within the op-iter.
- HAZARD: phase-2 `reserve_back` before `wait_min` — the slot is reserved before the data is known-arrived; correctness depends on sender's per-block `async_write_barrier` completing the data write before bumping the counter the receiver waits on. Ordering lives on the **sender** side (data mcast L263 then counter mcast L275 then barrier L284). Receiver trusts that ordering.
- HOLE: gather reads L154-213 interleave between phase-1 and phase-2; cannot be inside a single Pipe `receive()`.
