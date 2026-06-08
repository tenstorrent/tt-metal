# welford_reader_mcast_sender_unary_sharded_gn_v2.cpp (SEND side)

Path: ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_sender_unary_sharded_gn_v2.cpp

API spelling: experimental OO wrapper, `CoreLocalMem`, `combine_welford_stats` (host-of-kernel CPU combine on uint16 ptrs), multi-rectangle mcast.
Role: welford sharded gn v2 coordinator. Per (batch, group): local welford combine → gather remote (mean,var) pairs → global welford combine → mcast result. Block in batch L152 → group L164 loop.

## Block map

### Pre: arm flag
- L107 `reduce_sender_sem.set(VALID)`.

### Per group L164-285:
#### Local combine (CPU on L1) L165-175 — HOLE (not Pipe; pointer arithmetic + welford).
#### Phase-1 gather + handshake (counter), EXCLUDE_SRC
- L156 `cb_ex_partial.wait_front(2)` — local mean+var tiles ready (outside group loop, per batch).
- L177-198 `if num_mcast_cores>1`:
  - L179 `reduce_receiver_sem.wait(num_mcast_cores-1)` — COUNTER wait.
  - L180 `reduce_receiver_sem.set(0)` — reset.
  - L182-196 remote (mean,var) reads via `async_read<..., NOC_L1_READ_ALIGNMENT_BYTES>` into global_means/vars buffers (HOLE, gather; alignment-strided).
  - L197 `async_read_barrier`.
#### Global combine L201-207 — HOLE (CPU welford).
#### Phase-2 data mcast (multi-rect) + flag mcast, EXCLUDE_SRC, BARRIER
- L210-279 `if num_mcast_cores>1`:
  - L212-223 data mcast MID rect: `noc.async_write_multicast(CoreLocalMem(global_means_ptr), ..., 2*single_tile_size_bytes, num_mcast_cores_mid_group, ..., true)` — mean+var together, linked=true.
  - L224-231 flag mcast MID rect.
  - L233-254 first-group data+flag.
  - L256-277 last-group data+flag.
  - L278 `noc.async_write_barrier()` — **F1 = barrier**.
- L281-284 advance per-group pointers.
- L286-287 (per batch) pop cb_ex_partial, push cb_ex_global.

## Variant signature
- **F1 = barrier** (L278).
- **F2 = COUNTER (phase-1 gather) + LEVEL FLAG (phase-2 set once + set_multicast)**. Single flag mcast per group (no double-flag — unlike interleaved gn).
- **F3 = EXCLUDE_SRC** (own stat computed locally, written into global buffer slot 0; remote slots gathered).
- **pre_handshake = YES** (counter gather sync). Data source = raw L1 addr (`CoreLocalMem(global_means_ptr)`), NOT a CB object — mcasts from a hand-managed buffer region.
- **MULTI-RECTANGLE**. Data payload = 2 tiles (mean+var contiguous).

## Hazards / invariants
- INV: global combine (L201) writes back into global_means_ptr[0]/global_vars_ptr[0] BEFORE the data mcast (L212) reads them. Same-pointer RAW handled by being on the same RISC.
- HAZARD: gather reads use `NOC_L1_READ_ALIGNMENT_BYTES` stride to keep each remote pair in its own aligned slot; the mcast then sends `2*single_tile_size_bytes` from global_means_ptr. The src layout (mean then +single_tile var) must match dest expectation. Orthogonal to Pipe but a data-layout invariant.
- NEW: data source is a raw L1 region (not CB). Multi-rectangle. Per-group loop reuses the same flag/counter sems across groups (counter reset per group L180; flag re-mcast per group).
