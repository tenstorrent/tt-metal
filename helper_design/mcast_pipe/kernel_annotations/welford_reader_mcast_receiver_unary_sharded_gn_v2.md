# welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp (RECEIVE side)

Path: ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp

API spelling: experimental OO wrapper, `combine_welford_stats`.
Role: welford sharded gn v2 receiver. Per (batch, group): local welford combine, signal sender, wait for global result mcast. Block in batch L81 → group L92 loop.

## Block map

### Per group L92-115:
- L84 `cb_ex_partial.wait_front(2)` — local mean+var ready (per batch, outside group loop).
- L93-103 local welford combine + write to global slot (HOLE, CPU).
- L106 `reduce_receiver_sem.up(noc, mcast_sender_noc_x, mcast_sender_noc_y, 1)` — **COUNTER inc to sender** (gather signal).
- L109 `reduce_sender_sem.wait(VALID)` — **LEVEL FLAG wait** (data mcast lands in cb_ex_global region).
- L110 `reduce_sender_sem.set(INVALID)` — clear flag AFTER wait (note: order is wait-then-clear here, opposite of the gn_v2 non-welford receiver which clears before up).
- L112-115 advance pointers.
- L118-119 (per batch) pop cb_ex_partial, push cb_ex_global.

## Variant signature
- **F1 = NONE** on receive.
- **F2 = LEVEL FLAG + counter-up**. Single exchange per group.
- **F3 = EXCLUDE_SRC**.
- **pre_handshake = YES** (the `up` L106 precedes the `wait` L109).
- **CB ownership**: cb_ex_global.reserve_back(2*num_groups) is done ONCE per batch (L149-ish in sender; here the receiver reserves once per batch and the per-group writes land into the pre-reserved region). So NOT a fresh-slot-per-group; the slot is reserved per batch, written per group by the sender's mcast. Different ownership model from the tile-CB receivers.

## Hazards / invariants
- INV ORDER ANOMALY: here `wait(VALID)` (L109) precedes `set(INVALID)` (L110) — wait-then-clear. This works because each group's flag VALID is consumed exactly once and the sender re-mcasts VALID for the next group AFTER this receiver cleared it. **But** there is NO `set(INVALID)` BEFORE the first `up`/`wait`: the very first group relies on the flag being INVALID at kernel entry (default). **HAZARD**: if a prior op left the flag VALID, the first group's wait returns early. Contrast non-welford gn_v2 receiver which clears before up. The Pipe helper should normalize this ordering (clear-before-signal) to avoid the cross-op stale-VALID hazard.
- INV: counter-up per group; sender resets counter per group.
- NEW: clear-after-wait ordering (vs clear-before-signal elsewhere) — an inconsistency the helper must standardize.
