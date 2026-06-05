# reader_mcast_sender_unary_sharded_gn_v2.cpp (SEND side)

Path: ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_gn_v2.cpp

API spelling: experimental OO wrapper. Adds `CoreLocalMem<uint32_t>` for self/remote scalar reads, and **multi-rectangle mcast** (mid/first/last groups).
Role: sharded gn v2 coordinator. Gathers per-core scalars into cb_ex_external, then mcasts the global stat to up to 3 NoC rectangles. Main block in the `m`/`n` loop L157-287.

## Block map

### Pre-loop: arm flag eagerly
- L113 `reduce_sender_sem.set(VALID)` — flag armed ONCE before the loop (not per-iter). NOTE: differs from ln sender which arms inside phase-1.

### Per (batch_group, n∈{0,1}) iteration L158-285:
#### Gather + phase-1 handshake (counter), EXCLUDE_SRC
- L160 `cb_ex_partial.wait_front(1)` — local partial ready.
- L163-195 self-read of own scalar via `CoreLocalMem` (doubles as zero-init of tile, see L166-186 comment). HOLE (gather).
- L197 `reduce_receiver_sem.wait(num_mcast_cores-1)` — COUNTER wait.
- L198 `reduce_receiver_sem.set(0)` — counter reset (reused slot).
- L199-209 remote scalar reads (HOLE, gather).
- L210 `cb_ex_external.push_back(1)`.

#### Data mcast + flag mcast (multi-rectangle), EXCLUDE_SRC, BARRIER
- L212 `cb_ex.wait_front(1)` — combined result ready (producer = compute).
- L213 `cb_ex_partial.pop_front(1)`.
- L217-228 `noc.async_write_multicast(cb_ex (CoreLocalMem), mcast_dst, num_bytes_read, num_mcast_cores_mid_group, {}, {mid rect...}, true)` — **DATA mcast to MID rectangle**, linked=true, EXCLUDE_SRC.
- L229-236 `reduce_sender_sem.set_multicast(... mid rect, num_mcast_cores_mid_group, false)` — **FLAG mcast to MID rect**, linked=false.
- L238-259 `if has_mcast_first_group`: data mcast + flag mcast to FIRST rect.
- L261-282 `if has_mcast_last_group`: data mcast + flag mcast to LAST rect.
- L283 `noc.async_write_barrier()` — **F1 = barrier** (single barrier draining all 3 rect mcasts).
- L284 `cb_ex.pop_front(1)`.

## Variant signature
- **F1 = barrier** (L283).
- **F2 = COUNTER (phase-1 gather signal `wait`+`set(0)`) + LEVEL FLAG (phase-2 `set(VALID)` once + `set_multicast`)**. The flag is NOT monotone here (set once pre-loop, re-mcast each iter); receivers do `set INVALID`/`wait VALID` each iter. Different from ln-sharded's monotone phase-2 counter.
- **F3 = EXCLUDE_SRC** (self scalar gathered via explicit self-read L188-193; not loopback).
- **pre_handshake**: phase-2 data mcast IS preceded by the phase-1 counter handshake (gather sync). Data goes into the **producer-owned cb_ex slot** (read via get_read_ptr, popped after) — sender mcasts its own combined tile, not a fresh reserve. Receiver side is fresh slot.
- **MULTI-RECTANGLE fan-out**: one logical mcast = up to 3 `async_write_multicast` + 3 `set_multicast` calls (mid/first/last). **NEW FORK**: Pipe.send() must accept a *list of rectangles* (1-3), not a single rectangle.

## Hazards / invariants
- INV: data mcast (each rect) before flag mcast (each rect); single barrier at end (L283) is sufficient because all writes are on the same NoC and the flag carries the release.
- HAZARD: `static_assert datum_size_bytes <= cb_ex_external_slot_pitch_bytes` (L33-36) — per-slot scalar packing into cb_ex_external; gap bytes zero-init relies on the full-tile self-read trick (L166-186). Not a Pipe concern but a gather invariant.
- HAZARD: flag armed once at L113 but counter `reduce_receiver_sem` reset every iter (L198). The level flag `reduce_sender_sem` is re-mcast each iter via set_multicast; never re-set to VALID locally inside the loop → relies on remote receivers resetting it to INVALID then sender's set_multicast pushing VALID. Subtle: sender's local copy stays VALID, only remote copies toggle.
- NEW FORK: multi-rectangle. NEW: data via `CoreLocalMem` raw-addr mcast (not CircularBuffer-object mcast) — Pipe.send() must accept either a CB handle or a raw L1 addr as the source.
