# reader_mcast_receiver_unary_sharded_gn_v2.cpp (RECEIVE side)

Path: ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_gn_v2.cpp

API spelling: experimental OO wrapper.
Role: sharded gn v2 receiver. Tiny block (L73-83) in the (batch_group, j∈{0,1}) loop.

## Block map

### Per iteration L73-82 — full handshake, LEVEL FLAG + counter-up, EXCLUDE_SRC
- L75 `cb_ex_partial.wait_front(1)` — local partial ready.
- L76 `reduce_sender_sem.set(INVALID)` — clear flag.
- L77 `cb_ex_global.reserve_back(1)` — **fresh dest slot**.
- L78 `reduce_receiver_sem.up(noc, mcast_sender_noc_x, mcast_sender_noc_y, 1)` — **COUNTER inc to sender** (R→S gather signal).
- L79 `reduce_sender_sem.wait(VALID)` — **LEVEL FLAG wait** (data arrives via sender's mcast into the reserved slot).
- L80 `cb_ex_global.push_back(1)` — hand received global stat to compute.
- L81 `cb_ex_partial.pop_front(1)`.

## Variant signature
- **F1 = NONE** (receiver does no barrier/flush).
- **F2 = LEVEL FLAG (`set INVALID`/`wait VALID`) + counter-up (`up`)**. Single-phase here (the gather signal and the data-release flag are in one tight block — no separate streaming phase like ln-sharded).
- **F3 = EXCLUDE_SRC** (pure destination).
- **pre_handshake = YES** (the `up` at L78 is the pre-mcast R→S handshake; sender's `wait(num-1)` counts it before mcasting). Data lands in **fresh reserve_back slot** (L77), but a handshake DOES gate. This is the "pre_handshake = true, fresh slot" variant.

## Hazards / invariants
- INV ordering: `wait_front` L75 → `set(INVALID)` L76 → `reserve_back` L77 → `up` L78 → `wait(VALID)` L79 → `push_back` L80. The `set(INVALID)` before `up` is critical: it ensures the flag is cleared before the receiver announces readiness, so the sender's subsequent flag mcast is what releases (no stale VALID).
- INV: reserve_back L77 BEFORE up L78 — guarantees the dest L1 region is owned before the sender (which may already be mcasting after counting the up) writes into it.
- CLEAN: 8-line block, single pass within the inner loop → directly migratable to `Pipe::receive(flag, up_target, fresh_slot=true)`.
