# Intent: `Pipe` — NoC multicast + semaphore-handshake dataflow helper (tune-helper run)

> **Skill:** `tune-helper` (API-first, style choices decided **empirically on device**, not by argument).
> **Track:** dataflow (correctness = data landed bit-exact + no hang; bake-off under `--dev`).

---

## Primitives to find (drives Step A — the authoritative input)

Two roles, as the prompt specifies.

### Implementation substrate (what the helper is BUILT FROM — object-style API, hard constraint)
The helper and **every Step E bake-off kernel** are written against the new object API,
NOT the legacy free functions:
- `Noc` — `tt_metal/hw/inc/api/dataflow/noc.h`
  - `async_write_multicast<McastMode, ...>(src, dst_mcast_noc_addr, size, num_dests, linked)`
  - enums: `McastMode {INCLUDE_SRC, EXCLUDE_SRC}`, `VcSelection {DEFAULT, CUSTOM}`,
    `BarrierMode {TXN_ID, FULL}`, `ResponseMode {NON_POSTED, POSTED}`
  - `async_writes_flushed()`, `async_write_barrier<BarrierMode>()`, `get_noc_id()`
- `Semaphore<>` — `tt_metal/hw/inc/api/dataflow/noc_semaphore.h`
  - `set(v)`, `wait(v)`, `wait_min(v)`, `up(...)`, `inc_multicast(...)`,
    `set_multicast<McastMode>(noc, x0,y0,x1,y1, num_dests)`

### Recognition family (what the Step D census must SPOT in existing kernels, however spelled)
The census recognizes the block by ANY of these spellings (legacy free functions included —
they are the recognition net, never the implementation):
- `noc_async_write_multicast` (+ `_loopback_src`, `_one_packet`)
- `noc_semaphore_set_multicast` (+ `_loopback_src`)
- `noc_semaphore_set`, `noc_semaphore_wait`, `noc_semaphore_wait_min`, `noc_semaphore_inc`
- `noc_async_writes_flushed`, `noc_async_write_barrier`
- `get_noc_multicast_addr` (open-coded multicast-address tell)
- plus the wrapper-layer spellings (`Noc::async_write_multicast`, `Semaphore<>::set_multicast`, …)

> The net stays open: Step A may add any further **mcast-related** primitive it finds
> (e.g. `Semaphore<>::inc_multicast`, `noc_async_write_multicast_one_packet`, the
> `*_multicast_loopback_src` family) as long as it belongs to the multicast/handshake block.

## Vague sketch of the block to wrap (NOT a signature — firms up at Step ★)

The recurring producer/consumer dataflow sequence:

> **set up a source L1 region → multicast a block to a receiver rectangle →
> handshake with the receivers (sem set / wait / inc) → flush/barrier before the next iteration.**

Shape it as a **`Pipe` OBJECT that exists on BOTH sides** of the channel:
- the **sender** side exposes `send()`,
- the **receiver** side exposes `receive()`.

Fully general: multicast **ANY rectangle** of cores, to **ANY destination L1 address**, from
**ANY source L1 address** — `send()`/`receive()` take the addresses + size; the rectangle (and
the handshake semaphores) are configured on the `Pipe` at construction. Designed fresh.

## Style choices to resolve EMPIRICALLY (the reason this is tune-helper, not design-helper)
These are decided by on-device coverage→perf→L1 bake-off in Step E, not by paper argument.
The catalog (Step B) is the authority on which become true forks; this is the expected backlog:
- **flush vs barrier** before next iteration (SENT vs ACKed; `async_writes_flushed` vs `async_write_barrier<FULL>`)
- **flag sem (VALID/INVALID, exact-match `wait`) vs counter sem (monotone, `wait_min`)** for staging
- **loopback handling** when the sender core is inside the receiver rectangle
  (`McastMode::EXCLUDE_SRC` skip-self vs `INCLUDE_SRC` fill-own-dst)
- (any others the hazards catalog surfaces with ≥2 viable mitigations)

## Bookkeeping (not design decisions)
- **Track:** dataflow. Correctness = bit-exact landed data + no hang. Perf = NoC ns (tracy /
  `DeviceZoneScoped`). Footprint = L1 bytes (sem slots, CB depth).
- **Standalone run.** This run does not seed, censuses, or argue from any prior helper-design
  work. Contracts (Step A), hazards (Step B), and the census (Step D) are built fresh from
  source.
- **Known instances** (Step D census recall-check — the census MUST contain at least these):
  - `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp` — role: reader (sender)
  - `…/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` — role: reader (**hybrid: sender+receiver+loopback in one file — the generality stress test**)
  - `…/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` — role: hybrid
  - Then sweep every op-family: conv, layernorm, sdpa, group-attn, groupnorm, data-movement, deepseek-moe.

## Explicit exclusions
- Compute-kernel patterns (dataflow-only).
- Ethernet / cross-chip mcast (CCL) — out of scope.
