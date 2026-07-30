# quasar/conv2d mcast call sites — annotation (added by reconcile 2026-06-27)

**Ledger state:** all 8 kernels below are recorded `status=deferred` with flag `quasar-metal2-port`. The three
`_metal2` forks that carry active debug instrumentation —
`reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2_metal2.cpp`,
`reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks_metal2.cpp`,
`writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks_metal2.cpp` — additionally carry
`hang:#47797`. Per-file `tag` below is the **intrinsic eventual migration target** (clean weights senders /
refactor activation hybrids / defer for the deadlocking v2_metal2), *not* a green-light — the whole group is
gated behind "quasar metal-2.0 port lands + #47797 act-mcast/weights-mcast hang closed".

All 8 are Metal-2.0-port copies of EXISTING census conv twins (same basenames under
`ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/`). The mcast+handshake block — the part `mcast_pipe`
cares about — is **byte-for-byte identical** between each non-metal2 / `_metal2` pair; the `_metal2` file is a
**host-binding-only fork that REPLACES** the legacy kernel for the quasar factory, never a coexisting runtime
variant. All 8 use the OBJECT API (`Noc::`, `Semaphore<>::`, `MulticastEndpoint`/`McastDst`) — no raw free
functions. `op_family: conv` for every entry.

The `_metal2` delta in every pair (does NOT touch the handshake protocol):
- CB-index CTAs → `dfb::` tokens; sem-id CTAs → `sem::`; remaining positional CTAs → `get_arg(args::name)`
- per-core NoC-coord lookup tables → positional runtime varargs via `get_vararg(i)`
- `experimental::CB` → `DataflowBuffer`; `get_tile_size(cb)` → `cb.get_entry_size()`
- mcast source spelling `use<…AddrSelector::READ_PTR/WRITE_PTR>(cb)` → `CoreLocalMem<uint32_t>(cb.get_*_ptr())`
- feature `#ifdef` gating (FUSE_BIAS / SPLIT_READER); DRAM config-tensor read via `tensor::reader_indices`

---

## 1. `activation_reader_width_sharded.cpp` (non-metal2 original)
Path: `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/activation_reader_width_sharded.cpp`
Role: **reader, HYBRID (sender + receiver + loopback)** — width-sharded activation round-robin self-mcast.
Tag (eventual): **refactor**. op_family: conv. Substrate: object API.

| Fork | Value | Lines |
|---|---|---|
| **F1** flush/barrier | **barrier** (sender, after flag mcast). `noc.async_write_barrier()` `L252`. No flush. | 252 |
| **F2** flag/counter | **MIXED.** R→S = **counter** (`wait_min(num_mcast_cores-1)` `L213` + `set(0)` `L214`; receiver `up` `L267`). S→R = **level flag** (`set(VALID)` `L244` + `set_multicast<INCL_SRC>` `L245-251`; receiver `set(INVALID)` `L256`+`wait(VALID)` `L270`). | 213,214,244,256,267,270 |
| **F3** loopback | **INCLUDE_SRC loopback** — data `L226-233` (`MCAST_INCL_SRC`, `num_dests=num_reader_cores` counts self), flag `L245`. No EXCLUDE/degenerate path in this file. | 226,245 |
| **KNOB** pre_handshake | **YES (dest reused)** — sender `wait_min`s receiver acks before mcasting (`L213`); `act_cb` dest reused across the `num_input_cores` round-robin. | 213 |

**Migration blocker — F2-MIXED.** A monotone-counter `wait_min` on the R→S ack but a level flag `wait(VALID)`
on the S→R data-ready, with a `set(0)` reset (`L214`) that turns the counter into a per-round flag (wait_min
then reset). The `Pipe` must decide to canonicalize this hybrid (contrast the weights senders, which use exact
`wait(N)` for R→S). F1=barrier is forced by the new `set_multicast` API: the sem cell is its own mcast source
(A5 gotcha), so the old clear-and-wait-loopback fence is gone and it barriers instead (H1, comment `L240-243`).

## 2. `activation_reader_width_sharded_metal2.cpp`
Path: `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/activation_reader_width_sharded_metal2.cpp`
Role: hybrid. Tag (eventual): **refactor**. op_family: conv. **`_metal2` fork — REPLACES #1, host-binding only.**

Handshake body identical to #1: same INCLUDE_SRC loopback (`L208-215` data / `L218` flag), same F2-MIXED
(`wait_min` `L198`+`set(0)` `L199` R→S counter; `set(VALID)` `L217`+`set_multicast` S→R flag; receiver
`up` `L236` / `wait(VALID)` `L238`), same F1=barrier (`L225`), same pre_handshake (`L198`). Only deltas: the
X/Y lookup tables are positional varargs (`get_vararg(sender_logical_x)` / `get_vararg(num_cores_x+…)`
`L233-234`) and the mcast source is `CoreLocalMem<uint32_t>(tilized_in0_cb.get_read_ptr())` `L205`. **No active
debug scaffolding** in this file. Same F2-MIXED canonicalization blocker as #1.

## 3. `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp` (non-metal2 original)
Path: `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp`
Role: **reader, HYBRID** — block-sharded 2D-mcast activation reader; the conv generality stress test.
Tag (eventual): **refactor**. op_family: conv. Substrate: object API. Block hoisted into two file-local
helpers: `multicast_data` (`L22-49`, F3 dispatcher) and `mcast_block_chunked` (`L61-132`, burst-split send).

| Fork | Value | Lines |
|---|---|---|
| **F1** flush/barrier | **MIXED / dual-path.** Data+flag mcast path: **neither** (INV4 VC-4 FIFO, comment `L288-290`). Degenerate local-write path: **barrier** `L129`. Loopback flag path adds explicit `async_write_barrier` `L303`. | 129,303 |
| **F2** flag/counter | **flag (level + exact wait + reset).** R→S: sender `wait(act_mcast_num_dests + (is_receiver?0:1))` `L276` + `set(0)` `L277`; receiver `up` `L322/324`. S→R: `set(VALID)`+`set_multicast` `L295-296`/`L306-307`; receiver `set(INVALID)` `L318`+`wait(VALID)` `L328`. | 276,277,295,306,318,322,328 |
| **F3** loopback | **all THREE sub-cases, runtime-dispatched** in `multicast_data` `L32-48`: (a) is_receiver & num_cores>0 → INCLUDE_SRC `L34`; (b) is_receiver & num_cores==0 → local `async_write` to self `L38` (INV5 degenerate guard, loopback-1-dest hang dodge); (c) !is_receiver → EXCLUDE_SRC `L46`. | 34,38,46 |
| **KNOB** pre_handshake | **YES (dest reused)** — sender pre-acks-waits `L276`; `cb_act` dest reused across `act_w_num_outer`. | 276 |

**Migration blockers.** (1) The F3 tri-path (`multicast_data`) is exactly the `Pipe::send` F3 dual/tri-path
hand-rolled, including INV5 degenerate-unicast — a direct migration target once the `Pipe` grows the tri-path.
(2) **R4 chunked-streaming-send GAP (the hard one):** `mcast_block_chunked` `L61-132` interleaves
`src_cb.wait_front(wait_tile_curr)` with per-burst mcasts to start sending before the whole block is tilized
(splits into ≤`NOC_MAX_BURST_SIZE` chunks, `.addr` advanced per burst `L107`). The current `Pipe` sketch
auto-chunks a *fully-ready* block but does NOT stream a not-yet-complete one — **R4 streaming is DEFERRED this
round** (catalog R4), so this stays on raw API until the `Pipe` grows phase-granular streaming. HOLE: the
`reserve_done_sem`/`write_done_sem` ping-pong (`L238`,`L260-261`) is an intra-core split-reader shared-CB
handshake, NOT mcast — leave untouched.

## 4. `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2_metal2.cpp`  ⚠ UNRESOLVED DEADLOCK
Path: `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2_metal2.cpp`
Role: hybrid. Tag (eventual): **defer**. op_family: conv. Ledger: `status=deferred`, flags
`quasar-metal2-port` + **`hang:#47797`**. **`_metal2` fork — REPLACES #3, host-binding only + ACTIVE DEBUG.**

Handshake body identical to #3: same F3 tri-path in `multicast_data` (`L48-75`) + `mcast_block_chunked`
(`L87-158`) streaming send; same F2 flag (`wait` `L298`+`set(0)` `L299`; `set_multicast` `L296-326-343`;
receiver `up` `L353/355`, `wait(VALID)` `L362`); same F1 dual-path (neither on mcast / barrier `L155` & `L333`
on degenerate+loopback). `split_reader_cb_shared` is host-DROPPED (factory TT_FATAL-rejects it; the
`reserve_done`/`write_done` semaphores are gone). Sender NoC coords come from `get_vararg(act_w_outer_i)`
(`L353/355`).

**LIVE DEBUG SCAFFOLDING (do NOT migrate over it):** this file is laced with issue **#47797** act-mcast
handshake instrumentation — `#include "api/debug/dprint.h"` `L32`; a watcher ring-buffer marker scheme
(`RB_ITER` macro `L40`, `WATCHER_RING_BUFFER_PUSH` at `L212-214`, `L296`, `L303-304`, `L315`, `L358-359`,
`L363`); and per-phase `DPRINT(...)` traces (`L297`,`L307`,`L316`,`L360`,`L364`). The markers localize an
act-mcast **deadlock that is NOT yet resolved** (sender stalls between "got all bumps" `0xA2` and "mcast data
done" `0xA3`, i.e. the compute never produced this `nbh`'s tilized act for the sender's last reader line —
see comment `L305-306`). **defer** is therefore stronger than #3's refactor: same R4 chunked-streaming-send
blocker as #3, PLUS an open hang on the exact handshake we'd migrate. Revisit only after #47797 closes.

## 5. `reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` (non-metal2 original)
Path: `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp`
Role: **SENDER (pure)** — 1D weights mcast (col→RM blocks); pairs with the `_receiver_` file. Block appears
**twice** (weights `L245-277`, bias `L305-337`), identical protocol. Tag (eventual): **clean**. op_family: conv.

| Fork | Value | Lines |
|---|---|---|
| **F1** flush/barrier | **NEITHER** — INV4 VC-4 FIFO (comment `L264-266`). No flush, no barrier on the mcast path. | (none) |
| **F2** flag/counter | **flag (level + exact wait + reset).** R→S: sender `wait(weights_mcast_num_dests)` `L249/309` + `set(0)` `L250/310`. S→R: `set_multicast(VALID)` `L269/329`. | 249,250,269,309,310,329 |
| **F3** loopback | **EXCLUDE_SRC** (default `async_write_multicast`, `num_dests=weights_mcast_num_cores` excludes self; comment `L253/313`). | 255,269,315,329 |
| **KNOB** pre_handshake | **YES (dest reused)** — pre-ack `wait` `L249/309`. | 249,309 |

**Migration: clean.** The canonical `Pipe::send` shape — pre-ack-wait → EXCLUDE_SRC data mcast → flag
`set_multicast`, no barrier. The cleanest sender in the conv group. Benign note (same as prod twin): the
receiver sem `set(VALID)` happens once at `L123` and is never re-staged per round — correct only because the
sender never writes the cell to any other value; flag for the `Pipe` contract.

## 6. `reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks_metal2.cpp`  (active debug)
Path: `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks_metal2.cpp`
Role: SENDER (pure). Tag (eventual): **clean**. op_family: conv. Ledger: `status=deferred`, flags
`quasar-metal2-port` + **`hang:#47797`**. **`_metal2` fork — REPLACES #5, host-binding only + ACTIVE DEBUG.**

Handshake body identical to #5: weights block `wait(weights_mcast_num_dests)` `L277` + `set(0)` `L278` →
EXCLUDE_SRC data mcast `L283-290` → flag `set_multicast` `L297-304`, F1=neither (INV4); bias block identical
(`L338-365`). Deltas vs #5: source `use<WRITE_PTR>(cb)` → `CoreLocalMem<uint32_t>(cb_weight_obj.get_write_ptr())`
(`L284`,`L345`); FUSE_BIAS / SPLIT_READER `#ifdef`-gated; final fence `async_write_barrier` → **`noc.async_full_barrier()`**
`L381` (Metal-2.0 FW epilogue does not drain atomics — comment `L380`).

**LIVE DEBUG SCAFFOLDING:** `#include "api/debug/dprint.h"` `L28` with `DPRINT("WS start\n")` `L31` /
`DPRINT("WS end\n")` `L382`, tagged "DEBUG: conv2d layer3 hang". Pattern itself is clean, but the file is
mid-bringup for the #47797 hang — defer-while-debugging.

## 7. `writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` (non-metal2 original)
Path: `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp`
Role: **SENDER (pure)** — 2D weights mcast; pairs with `writer_..._2d_mcast_receiver_...`. Block appears twice
(weights `L250-282`, bias `L296-351`). Tag (eventual): **clean**. op_family: conv. The 2D twin of #5 — same
fork signature, mergeable into the same `Pipe::send`.

| Fork | Value | Lines |
|---|---|---|
| **F1** flush/barrier | **NEITHER** (INV4 VC-4 FIFO, comment `L269-271`). | (none) |
| **F2** flag/counter | **flag.** R→S: `wait(weights_mcast_num_dests)` `L254/320` + `set(0)` `L255/321`. S→R: `set_multicast(VALID)` `L274/340`. | 254,255,274,320,321,340 |
| **F3** loopback | **EXCLUDE_SRC** (comment `L258/313` "num_dests must not include source"). | 260,274,326,340 |
| **KNOB** pre_handshake | **YES (dest reused)** — pre-ack `L254/320`. | 254,320 |

**Migration: clean.** HOLE (NOT mcast): `reserve_done_sem`/`write_done_sem` (`L185-186` wait/set, `L214`
set(VALID)) is a two-core split-reader producer/producer shared-CB handshake — local sem flag ping-pong on the
`split_reader_cb_shared` path, leave untouched.

## 8. `writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks_metal2.cpp`  (active debug)
Path: `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks_metal2.cpp`
Role: SENDER (pure). Tag (eventual): **clean**. op_family: conv. Ledger: `status=deferred`, flags
`quasar-metal2-port` + **`hang:#47797`**. **`_metal2` fork — REPLACES #7, host-binding only + ACTIVE DEBUG.**

Handshake body identical to #7: weights `wait(weights_mcast_num_dests)` `L244` + `set(0)` `L245` → EXCLUDE_SRC
data mcast `L250-257` → flag `set_multicast` `L264-270`, F1=neither; bias identical (`L315-341`). Deltas vs #7:
source → `CoreLocalMem<uint32_t>(cb.get_write_ptr())` (`L251`,`L322`); **`split_reader_cb_shared` path DROPPED
on host** (factory TT_FATAL-rejects it), so the `reserve_done`/`write_done` HOLE from #7 is **absent** here —
ACT is single-producer; FUSE_BIAS / SPLIT_READER `#ifdef`-gated.

**LIVE DEBUG SCAFFOLDING:** `#include "api/debug/ring_buffer.h"` `L29` with a weights-mcast-sender deadlock
localization scheme — `rb_wcnt` load counter `L33`, `WATCHER_RING_BUFFER_PUSH` markers `0xE1` pre-reserve `L216`,
`0xE2` pre-wait-bumps `L243`, `0xE3` mcast-done `L272` (marker layout comment `L30`). Pattern is clean, but the
file is mid-bringup for the #47797 weights-mcast hang — defer-while-debugging.
