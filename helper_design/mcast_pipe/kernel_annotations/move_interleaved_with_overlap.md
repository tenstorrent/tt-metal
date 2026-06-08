# move_interleaved_with_overlap.cpp

Path: `ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_interleaved_with_overlap.cpp`
Role: combined reader+writer for in-place `move` with overlapping src/dst. One pass, no outer loop.
API era: legacy free-function (`noc_semaphore_*`, `get_noc_multicast_addr`).

## Block instances

### Block 1 — barrier handshake, SENDER half (controller) — lines 64-78
- `noc_semaphore_wait(semaphore_addr_ptr, control_value)` (L65): controller waits for all N non-controller cores to report in (counter reached `control_value`).
- mcast cluster (sem-flag mcast, NO data mcast):
  - `get_noc_multicast_addr(... range0 ...)` (L68-69) + `noc_semaphore_set_multicast(semaphore_addr, addr0, range_0_size)` (L70)
  - same for range1 (L71-73) and optional range2 (L74-78, guarded by `do_third_multicast`).
- Purpose: once every source core has finished reading its tiles into L1, controller broadcasts a "go" flag to up to 3 rectangles so all cores may now write to the (overlapping) destination. Guarantees no writer clobbers a tile another core has not yet read.

### Block 1 — RECEIVER half (non-controller) — lines 79-85
- `noc_semaphore_inc(controller_noc_address, 1)` (L82): report "my read is done" to controller's counter.
- `noc_semaphore_wait(semaphore_addr_ptr, control_value)` (L84): wait for controller's mcast go-flag.

Setup before the block: read all `num_tiles` into CB (L55-62). After the block: write all tiles to dst (L87-94).

## Mapping to Pipe
- This is a **single mcast of a 4-byte flag** wrapping a global read→write reorder barrier. Data is NOT multicast; data movement is plain interleaved read/write.
- SENDER = `Pipe::send()` of a bare flag to a multi-rectangle destination set, predicated on a counter wait.
- RECEIVER = `Pipe::receive()` of the flag, after incrementing the sender's counter.
- The 3 explicit `get_noc_multicast_addr`+`set_multicast` calls collapse to one fan-out send over a list of rectangles.

## Forks
- F1: **barrier** — `noc_async_write_barrier()` (L91), but that barrier guards the *data* writes, not the mcast. The sem mcast itself has no explicit flush before the consumer reads it (relies on `set_multicast` posted-write ordering + the wait on the receiver side).
- F2: **counter** — controller waits `== control_value` (monotone count of arrivals); single-shot, no reset (kernel runs once).
- F3: cannot tell INCLUDE/EXCLUDE_SRC from the legacy `set_multicast` API — the rectangle ranges are host-computed; controller is presumably excluded from its own ranges (it does not wait on its own flag). Treat as **EXCLUDE_SRC + self is the sender**. HOLE: loopback semantics opaque at kernel level.
- KNOB pre_handshake: dest slot (`semaphore_addr`) is reused for both the inbound counter (controller side) and the outbound flag (receiver side) — same L1 word, **dest reused**, no fresh slot. Single-shot so no reset needed.

## HOLEs
- Multi-rectangle fan-out (2-3 rectangles per send) — Pipe must support a destination *set*, not a single rectangle, or the call site keeps an explicit loop.
- Same L1 word serves as both arrival-counter (on controller) and go-flag (on workers); a typed Pipe would need to model this dual use or allocate two slots.
