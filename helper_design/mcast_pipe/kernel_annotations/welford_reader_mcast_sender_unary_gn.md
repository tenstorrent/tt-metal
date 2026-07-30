# welford_reader_mcast_sender_unary_gn.cpp (SEND side) — interleaved welford gn

Path: ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_sender_unary_gn.cpp

API spelling: experimental OO wrapper, `CoreLocalMem`, `combine_welford_stats`, `TensorAccessor`, multi-rectangle mcast.
Role: interleaved welford gn coordinator. Per batch: read input (pre+post around the combine), local+global welford combine, mcast result. Block in batch L185 → group L224 loop. Structurally identical mcast/handshake to welford_reader_mcast_sender_unary_sharded_gn_v2.cpp; differs only in the DRAM input reads (L197-214, L355-382, HOLE).

## Block map

### Pre: arm flag
- L123 `reduce_sender_sem.set(VALID)`.

### Per batch L185:
- L197-214 input tile reads (TensorAccessor, HOLE).
- L216 `cb_ex_partial.wait_front(2)`.

### Per group L224-350:
- L226-238 local welford combine (HOLE, CPU).
- Phase-1 gather + handshake, EXCLUDE_SRC:
  - L242 `reduce_receiver_sem.wait(num_mcast_cores-1)` — COUNTER wait.
  - L243 `reduce_receiver_sem.set(0)` — reset.
  - L245-259 remote (mean,var) reads (HOLE).
  - L260 `async_read_barrier`.
- L264-266 global welford combine (HOLE).
- Phase-2 data mcast (multi-rect) + flag mcast, BARRIER:
  - L275-286 data mcast MID; L287-294 flag mcast MID.
  - L296-318 first-group data+flag.
  - L320-342 last-group data+flag.
  - L343 `noc.async_write_barrier()` — **F1 = barrier**.
  - L346-349 advance pointers.
- L352-353 (per batch) pop cb_ex_partial, push cb_ex_global.
- L355-382 post-combine input reads (HOLE).

## Variant signature
- **F1 = barrier** (L343).
- **F2 = COUNTER (gather) + LEVEL FLAG (single mcast per group)**.
- **F3 = EXCLUDE_SRC**.
- **pre_handshake = YES**. Data source = raw L1 (`CoreLocalMem(global_means_ptr)`). MULTI-RECTANGLE. Payload = 2 tiles.

## Hazards / invariants
- Identical mcast+handshake hazards to welford_reader_mcast_sender_unary_sharded_gn_v2.cpp (same combine→mcast pointer RAW; same per-group flag/counter reuse).
- HOLE: DRAM input reads bracket the combine block (pre L197-214, post L355-382). The Pipe block is purely the gather+combine+mcast in the middle; input I/O stays at call site.
- DEDUP NOTE: this kernel's send block ≈ welford sharded_gn_v2 send block. Strong candidate for the same Pipe.send() call.
