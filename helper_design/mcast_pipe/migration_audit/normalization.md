# Migration Audit — normalization group (layernorm + groupnorm dataflow)

Helper under design: `Pipe` (two-sided: sender `send()`, receiver `receive()`) wrapping
**set up src L1 → mcast a block (data and/or 4-byte sem flag) to a rectangle → handshake (set/wait/inc) → flush/barrier**.

## API-spelling note (RECALL: NEW SPELLING)
None of these kernels use the bare `noc_async_write_multicast` / `noc_semaphore_*` recognition spellings.
They ALL use the **experimental OO wrapper API**:
- `Noc noc;` → `noc.async_write_multicast<McastMode>(...)`, `noc.async_read(...)`, `noc.async_write_barrier()`, `noc.async_atomic_barrier()`.
- `Semaphore<> s(id);` → `s.set(VALID/INVALID/val)`, `s.wait(v)`, `s.wait_min(v)`, `s.up(noc,x,y,v)`, `s.set_multicast<McastMode>(...)`, `s.inc_multicast(...)`.
- `MulticastEndpoint` / `UnicastEndpoint` / `CircularBuffer` / `CoreLocalMem<uint32_t>`.
The recognition grep MUST be extended with these or the entire normalization group is missed.
(Header: build_Release/.../api/dataflow/noc_semaphore.h confirms `set_multicast` wraps
`noc_semaphore_set_multicast` / `_loopback_src`; `McastMode::INCLUDE_SRC` = loopback = F3.)

## Per-kernel verdict (14 kernels: 7 send, 7 receive)

| Kernel | Side | Verdict | (F1,F2,F3,pre_hs) | Note |
|---|---|---|---|---|
| reader_mcast_receiver_unary_sharded_ln_post_allgather | recv | **CLEAN** | (none, FLAG, INCLUDE_SRC, NO) | 4-line reference receive() |
| reader_mcast_sender_unary_sharded_ln_post_allgather | send | **CLEAN** | (barrier, FLAG, INCLUDE_SRC, NO) | reference send(): data+flag, no gather |
| reader_mcast_receiver_unary_sharded_gn_v2 | recv | **CLEAN** | (none, FLAG, EXCLUDE_SRC, YES) | 8-line receive(), fresh slot + up |
| reader_mcast_sender_unary_sharded_ln_pre_allgather | send | **REFACTOR (low)** | (barrier, FLAG+CTR, EXCLUDE_SRC, n/a) | flag-only mcast (no payload) + interleaved gather HOLE |
| reader_mcast_receiver_unary_sharded_ln_pre_allgather | recv | **REFACTOR (low)** | (atomic-barrier, FLAG+CTRup, EXCLUDE_SRC, YES) | atomic-barrier flush; gather HOLE |
| welford_reader_mcast_receiver_unary_sharded_gn_v2 | recv | **REFACTOR (low)** | (none, FLAG+CTRup, EXCLUDE_SRC, YES) | clear-after-wait anomaly; raw-L1 dest |
| welford_reader_mcast_receiver_unary_gn | recv | **REFACTOR (low)** | (none, FLAG+CTRup, EXCLUDE_SRC, YES) | dup of above + DRAM read HOLE |
| reader_mcast_sender_unary_sharded_gn_v2 | send | **REFACTOR (med)** | (barrier, FLAG+CTR, EXCLUDE_SRC, YES) | multi-rectangle (×3); raw-L1 or CB src |
| welford_reader_mcast_sender_unary_sharded_gn_v2 | send | **REFACTOR (med)** | (barrier, FLAG+CTR, EXCLUDE_SRC, YES) | multi-rect; raw-L1 src; per-group loop |
| welford_reader_mcast_sender_unary_gn | send | **REFACTOR (med)** | (barrier, FLAG+CTR, EXCLUDE_SRC, YES) | dup of above + DRAM read HOLE |
| reader_mcast_sender_unary_sharded_ln | send | **REFACTOR (high)** | (barrier, FLAG+CTR ↦ MONOTONE-CTR, EXCLUDE_SRC, YES) | two-stage; phase-2 monotone `set(block+2)` streaming; interleaved gather |
| reader_mcast_receiver_unary_sharded_ln | recv | **REFACTOR (high)** | (none, FLAG+CTRup ↦ wait_min, EXCLUDE_SRC, mixed) | phase-2 `wait_min(block+2)` streaming, no per-block pre-hs |
| reader_mcast_sender_unary_gn | send | **REFACTOR (high)** | (barrier, CTR + DOUBLE-FLAG, EXCLUDE_SRC, YES) | 3-pass; go-flag + data-flag per iter; multi-rect; out_block chunking |
| reader_mcast_receiver_unary_gn | recv | **REFACTOR (high)** | (none, DOUBLE-FLAG + CTRup, EXCLUDE_SRC, mixed) | two `wait(VALID)` per iter (go + data); double clear |

No **defer-raw** verdicts: every kernel's mcast+handshake is expressible by the Pipe model, given the forks below are supported. The cost differences are about how many forks/HOLE-interleavings the call site must thread.

## Counts
- Total block kernels: **14** (7 send / 7 receive).
- CLEAN: **3** (post_allgather send+recv, sharded_gn_v2 recv).
- REFACTOR-low: **4** (pre_allgather send+recv, 2 welford recv).
- REFACTOR-med: **3** (sharded_gn_v2 send, 2 welford send).
- REFACTOR-high: **4** (sharded_ln send+recv, interleaved gn send+recv).
- DEFER-RAW: **0**.

## Headline blockers (drive the helper API / bake-off)
1. **NEW SPELLING (recall)** — OO wrapper API (`Noc`/`Semaphore<>`/`MulticastEndpoint`). The Pipe helper should likely be built ON TOP of this API (it is the de-facto normalization dataflow API), not the bare `noc_*` calls.
2. **Two-phase flag/counter split (the brief's headline)** — confirmed in layernorm sharded: **phase-1 = level-flag + bounded counter (`wait(N-1)`/`set(0)`)**, **phase-2 = MONOTONE counter (`set(block+2)` / `wait_min(block+2)`) streaming**. Pipe.send()/receive() must support BOTH a level-flag mode AND a monotone-counter (wait_min) streaming mode (F2 fork is not binary — it's flag | bounded-counter | monotone-counter).
3. **No-pre-handshake fresh-slot mcast** — post_allgather send/recv: sender mcasts (INCLUDE_SRC) into a fresh `reserve_back` slot with NO R→S handshake; receiver just `set(INVALID)`+`reserve_back`+`wait(VALID)`. Pipe must allow `pre_handshake=false`. Also appears as the *data-flag* half (exchange B) of interleaved gn.
4. **Double-flag per exchange** — interleaved gn (reader_mcast_*_unary_gn): a "go" flag mcast after gather, then a "data" flag mcast after combine, both on ONE semaphore; receiver does two `set(INVALID)`/`wait(VALID)`. Needs either two Pipe calls or a two-step send/receive primitive.
5. **Multi-rectangle fan-out** — all groupnorm senders mcast to up to 3 NoC rectangles (mid/first/last) per logical exchange. Pipe.send() must accept a *list* of rectangles (1–3), each with its own num_dests, with a SINGLE trailing barrier.
6. **F3 = INCLUDE_SRC vs EXCLUDE_SRC + the num_dests off-by-one** — INCLUDE_SRC (post_allgather) uses `num_dests = num_blocks`; EXCLUDE_SRC uses `num_blocks-1` and a separate self-read/self-fill. Pipe must encode this per F3 to avoid off-by-one.
7. **F1 has THREE values, not two** — `async_write_barrier` (most), `async_writes_flushed` (none seen here but in family), and **`async_atomic_barrier`** (pre_allgather receiver L206, for draining semaphore `up` atomics). Pipe flush mode must include atomic-barrier.
8. **Data source polymorphism** — src is sometimes a `CircularBuffer` object (post_allgather), sometimes a raw L1 addr via `CoreLocalMem<uint32_t>` (gn_v2, welford). Pipe.send() must accept either.
9. **Ordering inconsistency to standardize** — receivers split into clear-before-signal (gn_v2, sharded_ln) vs clear-after-wait (welford gn/gn_v2). The latter has a cross-op stale-VALID hazard on the first iteration. Pipe should standardize on clear-before-signal.
10. **HOLE interleaving** — in all REFACTOR kernels the mcast+handshake is interleaved with NoC gather reads and/or DRAM input reads that CANNOT be subsumed by Pipe. The call site must keep those between Pipe phases → Pipe phases must be separately callable (e.g. `send_signal()` / `gather...` / `send_data()`), not a single monolithic `send()`.
