# reader_mcast_receiver_unary_gn.cpp (RECEIVE side) — interleaved gn, 3-pass

Path: ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_receiver_unary_gn.cpp

API spelling: experimental OO wrapper, TensorAccessor.
Role: interleaved gn receiver. TWO flag exchanges per (iter 0/1): a "go" handshake inside the out_block loop, then a "data-ready" wait after the loop.

## Block map (batch L177 → group L182 → cur_read_iteration L188)

### Exchange A — go handshake (per out_block iteration), LEVEL FLAG + counter-up
- L223 `reduce_sender_sem.set(INVALID)` — clear flag.
- L226/L229 `cb_ex_partial.wait_front(1)` / `cb_ex2_partial.wait_front(1)` — local partial ready.
- L231 `reduce_receiver_sem.up(noc, mcast_sender_noc_x, mcast_sender_noc_y, 1)` — COUNTER inc to sender (gather signal).
- L233 `reduce_sender_sem.wait(VALID)` — LEVEL FLAG wait (sender's L388-417 early flag mcast).
- L234-238 pop partial.

### Exchange B — data-ready wait (after out_block loop), LEVEL FLAG, fresh slot
- L269 `reduce_sender_sem.set(INVALID)` — clear flag again.
- L271/L275 `cb_ex_global.reserve_back(1)` / `cb_ex2_global.reserve_back(1)` — **fresh dest slot**.
- L272/L276 `reduce_sender_sem.wait(VALID)` — LEVEL FLAG wait (sender's L469-516 data+flag mcast). Data has already landed in the reserved slot.
- L273/L277 `cb_ex_global.push_back(1)` / `cb_ex2_global.push_back(1)`.

### iter 2: re-read output (L239-264, HOLE).

## Variant signature
- **F1 = NONE** on receive.
- **F2 = LEVEL FLAG (both exchanges) + counter-up (exchange A only)**. **Double-flag-per-logical-exchange** confirmed: exchange A = go, exchange B = data-release, both over the single `reduce_sender_sem`.
- **F3 = EXCLUDE_SRC**.
- **pre_handshake**: exchange A has the counter-up; exchange B is a **bare `wait(VALID)` into a fresh slot with NO R→S handshake** (no `up` before B). **Exchange B is the no-pre-handshake fresh-slot receive.**

## Hazards / invariants
- INV: two `set(INVALID)` (L223, L269) — the flag must be re-cleared before exchange B, otherwise exchange A's VALID leaks into B's wait and B returns immediately before data lands. **This double-clear is the crux hazard of the double-flag pattern.**
- INV: reserve_back (L271/L275) before wait (L272/L276) — own the slot before sender writes it.
- NEW: a single Pipe `receive()` here would have to be called TWICE per iter with different semantics (with-up vs without-up), OR Pipe must model a two-step receive (go-ack then data-wait). Flagged as REFACTOR cost.
