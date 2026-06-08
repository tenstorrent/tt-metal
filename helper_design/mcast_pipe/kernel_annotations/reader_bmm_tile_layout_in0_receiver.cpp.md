# reader_bmm_tile_layout_in0_receiver.cpp — annotation

Role: **RECEIVER half** (pure). Object API. Pairs with in0_sender_padding.

## Fork signature (receiver view)
- **F1**: N/A (receiver issues no mcast/flush; no barrier of its own).
- **F2**: LEVEL FLAG, with one COUNTER spelling. Steady-state blocks use `receiver_sem.wait(VALID)` (L84). The sparsity batch path uses `receiver_sem.wait_min(VALID)` (L56) — `wait_min` is the monotone/counter spelling (F2 evidence that wait_min appears even in a level-flag kernel, to tolerate IGNORE_BATCH=0x2 ≥ VALID=0x1).
- **F3**: N/A from receiver side.
- **KNOB pre_handshake**: this receiver's `sem.up` (signal-ready) BEFORE `wait(VALID)` IS the back-half of the sender's pre_handshake — receiver advertises readiness, then blocks for data.

## Protocol steps
- L37-40: ctors `Noc`, `cb_in0`, `sender_sem` (cta4), `receiver_sem` (cta5).
- L42-43: alias `receiver_sem` L1 ptr (to read the actual flag value for batch-valid decode).
- Sparsity batch path (L46-68): `set(INVALID)` (L52) → **signal-back** `sender_sem.up(sender_x,sender_y,1)` (L54) → **receiver-wait** `wait_min(VALID)` (L56) → read flag value, decode batch-valid, mailbox to compute (L58-63).
- Steady-state (L73-87): per block: `cb_in0.reserve_back` → `receiver_sem.set(INVALID)` (L78) → **signal-back** `sender_sem.up(...,1)` (L81) → **receiver-wait** `receiver_sem.wait(VALID)` (L84) → `cb_in0.push_back` (L86).
- No final barrier (receiver does no NoC writes except the atomic inc; no `async_atomic_barrier` here — relies on op teardown).

## HOLEs
- L56 `wait_min` vs L84 `wait` divergence inside one kernel — the receiver-side Pipe `receive()` must support both an exact-match and a min-threshold flag predicate. Flag.
- L58 reading the raw flag value as data (VALID vs IGNORE_BATCH) — the flag is overloaded as a 3-state payload, not a pure boolean. Pipe `receive()` should expose the flag value, not just block-until-valid.
