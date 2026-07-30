# quasar move kernels — annotation (added by reconcile 2026-06-27)

Covers BOTH Metal-2.0 quasar port twins of the production `move` overlap kernels:
- `ttnn/cpp/ttnn/operations/experimental/quasar/move/device/kernels/dataflow/move_interleaved_with_overlap.cpp` (tiled)
- `ttnn/cpp/ttnn/operations/experimental/quasar/move/device/kernels/dataflow/move_stick_layout_interleaved_with_overlap.cpp` (RM / stick)

Family: **quasar (experimental metal 2.0 port)**. Role: **hybrid** (controller = sender, non-controller = receiver).
Tag: **refactor**. Status: **deferred**. Flag: **`quasar-metal2-port`**.

These are the object-API (`Noc`, `Semaphore`, `DataflowBuffer`/`CircularBuffer`) re-spellings of the
production `data_movement/move/.../move_*_with_overlap.cpp` pair (both censused `refactor`). Same in-place
overlapping-`move` algorithm: every core reads its tile/page slice into the scratch CB, a global
read→write reorder barrier mcasts a "go" flag, then every core writes to the (overlapping) destination so
no writer clobbers a tile another core has not yet read.

## THE BLOCK (flag-only "go" barrier) — both files, lines ~63-80
Identical handshake in both; the only difference is the data path (CB `async_read`/`async_write` vs
`CoreLocalMem` page reads at `aligned_page_size` stride). **Data is NOT multicast** — the mcast carries
only the 4-byte go flag.

| step | tiled L | stick L | call |
|------|---------|---------|------|
| read all tiles/pages into scratch CB | 53-61 | 53-61 | `noc.async_read` + `async_read_barrier` (per ublock) |
| **controller** waits all workers reported | 64 | 64 | `sem.wait(control_value)` (counter == arrival count) |
| **controller** mcasts go-flag to rect 0 | 67-68 | 67-68 | `sem.set_multicast<NocOptions::DEFAULT>(noc, r0_xs,r0_ys,r0_xe,r0_ye, r0_size)` |
| **controller** mcasts go-flag to rect 1 | 69-70 | 69-70 | `sem.set_multicast<DEFAULT>(... range1 ...)` |
| **controller** mcasts go-flag to rect 2 (opt) | 71-74 | 71-74 | guarded by `do_third_multicast` |
| **worker** reports "my read done" | 77 | 77 | `sem.up(noc, controller_noc_x, controller_noc_y, 1)` (R→S counter inc) |
| **worker** waits the go-flag | 79 | 79 | `sem.wait(control_value)` |
| everyone writes to dst | 82-90 | 82-90 | `noc.async_write` + `async_write_barrier` (per ublock) |

## Forks
- **F1 = barrier** — `noc.async_write_barrier()` (L87) guards the *data* writes, not the mcast. The
  sem mcast itself has no explicit flush before consumers read it; relies on `set_multicast` ordering +
  the receiver wait. (Same as the production twin.)
- **F2 = counter / single-shot** — controller waits `== control_value` (monotone count of worker
  arrivals); kernel runs once, no reset. The go-flag side is a level value (`set_multicast` of the sem
  cell), the workers `wait(control_value)` on it; single-shot so no stale-retrigger (H3) risk.
- **F3 = opaque (EXCLUDE_SRC + self is sender).** The 2-3 rectangles are host-computed; controller is
  presumably excluded from its own ranges (it does not wait on its own flag). HOLE: loopback semantics
  not visible at kernel level.
- **KNOB pre_handshake = present (dual-use L1 word).** The single `sem::sem` slot serves BOTH as the
  controller's inbound arrival-counter AND as the workers' outbound go-flag — same L1 word, dest reused.
  Single-shot so no reset needed.

## Migration-blocker audit → REFACTOR (deferred)
- **Multi-rectangle fan-out (R1)** — 2-3 host-computed rectangles per send; Pipe must support a
  destination *set*, not a single rectangle, or the call site keeps the explicit loop.
- **Dual-use sem word** — one L1 word is both arrival-counter (controller) and go-flag (workers); a
  typed Pipe would need two slots or to model the dual use. Same HOLE as the production twin.
- **`quasar-metal2-port`** — this is an actively-churning experimental Metal-2.0 API port. Even though it
  already uses the object API, the port is unstable; DEFER until the production twin migrates and the
  quasar surface settles, then mirror that migration here. Object-API spelling does NOT promote it to
  `clean` — the multi-rect flag-only barrier + dual-use word are the same `refactor` shape as production.
