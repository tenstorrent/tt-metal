# reader_mcast_receiver_unary_sharded_ln.cpp — TIER 3.1 (refactor-high)

**Status: FAILED (left untouched, no commit)**
**Validation: n/a** (no edit made; tree stays green on the baseline)

## Op / dispatch
ttnn.layer_norm sharded (same nodeid as the sender). Object API. `reduce_sender_sem`,
`reduce_receiver_sem`, `reduce_second_stage_sem`.

## Handshake blocks (C3 two-phase)
1. **Phase-1 (L140-142)** — `reduce_sender_sem.set(INVALID); reduce_receiver_sem.up(sender,1); reduce_sender_sem.wait(VALID)`.
2. **Phase-2 (L236-241) monotone-counter streaming** — per block: `cb_ex_global.reserve_back(...); reduce_sender_sem.wait_min(block+2); push_back(...)`.

## Assessment — NO FIT
- **Phase-1 vs `receive()`.** `Pipe::receive()` (PRE_HANDSHAKE) does `consumed_.up(sender,1)` then
  `data_ready_.wait(VALID); data_ready_.set(INVALID)` — i.e. it clears the flag AFTER the wait
  (H11 clear-after-wait, the Pipe's pinned ordering). The raw clears `reduce_sender_sem` to INVALID
  BEFORE the up/wait. That leading clear is **load-bearing**: `global_reduce_receiver` is called
  TWICE in the non-welford dispatch (L246 mean + L250 var) on the SAME `reduce_sender_sem` cell,
  and after the first call's phase-2 the cell holds a high counter value (e.g. 4). The second call's
  phase-1 must reset that counter back to flag-0 BEFORE waiting for VALID=1. A `receive()` (clear
  after wait) would `wait(VALID=1)` while the cell is still ≥4 and return immediately (stale) —
  breaking the non-welford path. The validation nodeid uses welford (single call) so the test would
  not catch it, but a migration must not break the non-welford dispatch. NOT migratable.
- **Phase-1 vs `receive_signal()`.** `receive_signal()` Flag mode is `wait(VALID); set(INVALID)` with
  NO ack — but phase-1 requires the `reduce_receiver_sem.up(sender,1)` ack BETWEEN the clear and the
  wait. Wrapping only the trailing `wait` is a 1-line non-verb; not a clean migration.
- **Phase-2 vs `receive_signal()` Counter.** `Staging::Counter receive_signal()` does
  `wait_min(++round_)` starting at 1 (round_ is private, no constructor seed). The raw base is
  `block+2` (2,3,4,...) because phase-1 left the cell at VALID=1 and the sender's phase-2 sets 2,3,4.
  `wait_min(1)` would return immediately against the pre-set 1, skipping the wait for block-0 data.
  Cannot align the base per-side without changing the (raw, un-migrated) sender's counter base.

No clean verb fits either phase. Left RAW.
