# welford_reader_mcast_receiver_unary_gn.cpp (RECEIVE side) — interleaved welford gn

Path: ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_receiver_unary_gn.cpp

API spelling: experimental OO wrapper, `combine_welford_stats`, `TensorAccessor`.
Role: interleaved welford gn receiver. Per batch: input reads, local welford combine, per-group signal+wait. Block in batch L114 → group L153 loop. Mcast/handshake identical to welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp; differs only in DRAM input reads (L116-143, L186-212, HOLE).

## Block map

### Per group L153-180:
- L145 `cb_ex_partial.wait_front(2)` (per batch).
- L155-167 local welford combine (HOLE).
- L170 `reduce_receiver_sem.up(noc, mcast_sender_noc_x, mcast_sender_noc_y, 1)` — COUNTER inc to sender.
- L173 `reduce_sender_sem.wait(VALID)` — LEVEL FLAG wait.
- L174 `reduce_sender_sem.set(INVALID)` — clear AFTER wait.
- L176-179 advance pointers.
- L182-183 (per batch) pop cb_ex_partial, push cb_ex_global.

## Variant signature
- **F1 = NONE**.
- **F2 = LEVEL FLAG + counter-up**, single exchange per group.
- **F3 = EXCLUDE_SRC**.
- **pre_handshake = YES** (up L170 before wait L173).
- CB ownership: cb_ex_global reserved once per batch; per-group sender mcasts write into the pre-reserved region.

## Hazards / invariants
- SAME clear-after-wait ordering anomaly as welford sharded_gn_v2 receiver (L173 wait then L174 clear; no clear before first up — relies on default-INVALID at entry). Cross-op stale-VALID hazard. Helper should standardize clear-before-signal.
- HOLE: DRAM input reads bracket the block (pre L116-143, post L186-212).
- DEDUP NOTE: receive block ≈ welford sharded_gn_v2 receive block.
