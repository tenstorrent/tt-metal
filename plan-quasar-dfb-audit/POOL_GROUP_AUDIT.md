# CB→DFB Kernel Audit: `pool` Group (consolidated)

**Date:** 2026-07-15
**Group:** pool ops from the do-list — **3 ops / 8 factory-variant rows / 17 in-scope kernel `.cpp`+`.hpp` files** (grid_sample, rotate, upsample)
**Audit spec:** `plan-quasar-dfb-audit/cb_dfb_kernel_audit_helper.md`

## Group verdict: YELLOW

**Bottom line:** The pool group is dominated by canonical Class-1 FIFO CBs and portable pointer-only (sharded / borrowed) buffers, but it is held at **YELLOW** by two things, both mechanical, neither a design decision:

1. **Shared-helper `get_local_cb_interface(...)` field reads** in `pool_kernels_common.hpp` (`zero_out_tiles` → `.fifo_num_pages`, `zero_out_page` → `.fifo_page_size`) — pulled in by **every grid_sample and rotate factory**. These are the exact hits the spec pre-classified as **mechanical NEEDS-FIX** with **merged/existing DFB getters** (`get_total_num_entries()` [PR #49197], `get_entry_size()`). They must be swapped before the kernel port merges (strict hard-GATE policy), but no getter is missing and no redesign is pending.
2. **The upsample-bilinear Class-4 flagship** — `bilinear.cpp` `llk_push_pages_bilinear` (`fifo_wr_ptr +=` / manual ring-wrap via `fifo_limit`/`fifo_size`, **no `push_back` credits**). This is the spec's named Class-4 flagship and classifies **WEIRD-OK → Portable (workaround) → GREEN** (LLK tile-stream coupling forces DFB ptr/credit surgery). It also carries one mechanical field read (`.fifo_page_size` → `get_entry_size()`).

No silent-wrong (`get_cb_tiles_*_ptr`), no Quasar-runtime block (`read_tile_value`/`get_tile_address`), no LTA prerequisite (`get_pointer_to_cb_data`) anywhere in the group. **RED is not warranted** — there are no unresolved design decisions or missing runtime APIs; the only blockers are mechanical getter swaps (getters exist) plus the documented Class-4 ptr-surgery workaround.

> **Rollup policy (Reading B — adopted):** a `get_local_cb_interface(cb).fifo_page_size` / `.fifo_num_pages` **read** where the DFB getter already exists (`get_entry_size()` / `get_total_num_entries()`, both merged) is a **mechanical NEEDS-FIX → YELLOW (Portable, GATE-clear before merge)**, not RED. RED is reserved for no-getter fields, structural blockers, and unresolved design decisions. Every `get_local_cb_interface(...)` hit in this group resolves to a merged getter (mechanical) or the sanctioned Class-4 DFB ptr-surgery workaround → **YELLOW**. Applied consistently across all audited groups.

## Group-wide classification scan (SCAN_FILES only)

Run over the 21-file `SCAN_FILES` closure (17 kernels + 4 shared/donor headers):

| Pattern (classification) | Hits |
|--------------------------|------|
| `get_local_cb_interface(...).<field>` (GATE) | **9 lines / 2 files** — `pool_kernels_common.hpp` (×4: L46,47,76,129), `bilinear.cpp` (×5: L17,19,20,22,23) |
| `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr` (silent-wrong) | **none** |
| `read_tile_value` / `get_tile_address` (quasar-blocked) | **none** |
| `get_pointer_to_cb_data` (LTA prereq) | **none** |
| `llk_push_pages` / `push_back_hold` (Class-4 credit decoupling) | **1** — `bilinear.cpp` `llk_push_pages_bilinear` (L15,58) |
| `fifo_wr_ptr` / `fifo_rd_ptr` (ptr surgery) | **3** — `bilinear.cpp` L19,22,23 (all via `get_local_cb_interface`) |
| `fifo_page_size` / `fifo_num_pages` (field reads) | **5** — `pool_kernels_common.hpp` L46,47,76,129 + `bilinear.cpp` L17 |
| `get_read_ptr` / `get_write_ptr` (portable pointer / WEIRD-OK) | 14 lines across 8 kernels — L1 addresses + NOC offset arithmetic |

All pool kernels use the `experimental::CB` portable wrapper (from `experimental_device_api.hpp`); the GATE hits are places where code reaches *around* the wrapper to the private `LocalCBInterface` struct.

## Per-op / per-factory rollup

| Op | Factory | Kernel state | CB classes | Verdict |
|----|---------|--------------|------------|---------|
| `grid_sample` | **GridSampleBilinearProgramFactory** | Class-1 FIFO + shared-helper field reads | 1, 6 | **YELLOW** — NEEDS-FIX getter swap (`zero_out_tiles`/`zero_out_page`) |
| `grid_sample` | **GridSampleNearestProgramFactory** | pointer-only writer-does-all + `zero_out_page` field read | 6 | **YELLOW** — NEEDS-FIX getter swap; CBs portable pointer |
| `rotate` | **BilinearProgramFactory** | Class-1 FIFO + `zero_out_page` field read | 1, 6 | **YELLOW** — NEEDS-FIX getter swap |
| `rotate` | **NearestProgramFactory** | Class-1 FIFO + `zero_out_page` field read | 1, 6 | **YELLOW** — NEEDS-FIX getter swap |
| `upsample` | **UpsampleBilinearProgramFactory** | **Class-4 flagship** `llk_push_pages_bilinear` + field read | 1, 4, 6 | **YELLOW** — Class-4 WEIRD-OK (GREEN) + NEEDS-FIX field read (L17) |
| `upsample` | **UpsampleMultiCoreInterleavedProgramFactory** | canonical FIFO (reader/writer/untilize donor) | 1 | **GREEN** |
| `upsample` | **UpsampleMultiCoreShardedProgramFactory** | pointer-only sharded copy (get_read_ptr) | 6 | **GREEN** (workaround-flagged) |
| `upsample` | **UpsampleNearestFloatProgramFactory** | canonical FIFO producer/consumer | 1 | **GREEN** |

## CB portability — `upsample` / UpsampleBilinearProgramFactory (Class-4 flagship)

**Scope:** `upsample_bilinear_program_factory_multicore.cpp` → kernels: `reader_bilinear_multi_core_sharded.cpp` (bound as both reader and writer; `is_reader` CT-arg branch), `bilinear.cpp` (compute), `bilinear_weights_lut.hpp`.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `halo_cb` (input) | 6 | `reader_bilinear_multi_core_sharded.cpp` | Portable (workaround) | **undesirable but OK hack:** sync-free borrowed halo-padded input shard read via `halo_cb.get_read_ptr()` + byte offset as NOC source (L304); no FIFO. Bare L1 address → do-not-churn. Uplift optional: LTA. | Portable (workaround) | same |
| `tilize_reduce_cb0/1` | 1 | `reader_bilinear_multi_core_sharded.cpp`, `bilinear.cpp` | Portable | linear FIFO — reader `reserve_back(4)`/`push_back(4)` (L360/407), compute `wait_front(4)`/`pop_front(4)` (L33/51). Double-buffered even/odd. | Portable | — |
| `scalar_cb` (in_scalar 1/2) | 1 | `reader_bilinear_multi_core_sharded.cpp`, `bilinear.cpp` | Portable | producer fills weights at `scalar_cb.get_write_ptr()` (L400) then `push_back(1)` (L405); compute `pop_front(1)`. `get_write_ptr` fill is canonical producer body. | Portable | — |
| `out_cb` | **4** | `bilinear.cpp` (produce), `reader_bilinear_multi_core_sharded.cpp` (writer-role drain) | **Portable (workaround)** | **undesirable but OK hack:** Class-4 credit/address decoupling — `pack_untilize_dest` writes tiles to L1, then `llk_push_pages_bilinear` (L15-25) does `get_local_cb_interface(out).fifo_wr_ptr += num_words` + manual ring-wrap on `fifo_limit`/`fifo_size` (L19-23) with **no `push_back` / no `tiles_received` posting** — nothing waits on credits, the writer-role drains L1 directly. Port target: replace `get_local_cb_interface` field writes with **DFB `get_write_ptr()` + manual advance** (ptr/credit surgery); ring-span via `get_entry_size()`/`get_total_num_entries()`. LLK `pack_untilize_dest(out_cb_id)` still targets the CB/DFB id → LTA/scratchpad do not apply. | **Portable (workaround)** | **undesirable but OK hack:** same Class-4 ptr surgery. **NOT a compute self-loop candidate** (credits decoupled from address — spec Class-4 self-loop table = No). Retain DFB ptr/credit surgery and **`Gen2Config::disable_implicit_sync_for` on this DFB**. Alternative if ptr surgery is later rejected: scratchpad + semaphores (would require a design decision — not chosen now). |

**Also in this factory (mechanical):** `bilinear.cpp` L17 reads `get_local_cb_interface(operand).fifo_page_size` inside `llk_push_pages_bilinear` → **NEEDS-FIX** `get_entry_size()`.

## CB portability — `grid_sample` / GridSampleBilinearProgramFactory

**Scope:** `grid_sample_bilinear_program_factory.cpp` → kernels: `reader_grid_sample_sharded.cpp` **or** `reader_grid_sample_interleaved_start_id.cpp` (+ `grid_sample_reader_common.hpp`), `compute_pool_2d.cpp` (donor: pool/generic), `writer_grid_sample_interleaved.cpp`, `pool_kernels_common.hpp` (helper).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `input_cb` (c_0) | 1 | reader (+common), `compute_pool_2d.cpp` | Portable | linear FIFO — `input_cb.reserve_back`/`push_back` in `grid_sample_reader_common.hpp` (L389/407), compute consumes. **NEEDS-FIX:** cleared via `zero_out_tiles<input_cb_index>` → `get_local_cb_interface(cb).fifo_num_pages` (`pool_kernels_common.hpp` L76) → `get_total_num_entries()`. | Portable | same NEEDS-FIX |
| `grid_cb` | 6 / 1 | reader (+common) | Portable (workaround) | **undesirable but OK hack:** sharded → `grid_cb.get_read_ptr()` (borrowed grid shard, L104); interleaved → `grid_cb.get_write_ptr()` NOC dest for grid sticks (L78). Pointer-only. | Portable (workaround) | same; uplift optional: LTA (borrowed) / linear DFB (interleaved) |
| `scalar_cb` | 1 | reader (+common), `compute_pool_2d.cpp` | Portable | producer `reserve_back`/fill@`get_write_ptr`/`push_back` (common L369-372); compute consumes. **NEEDS-FIX (sharded reader):** `zero_out_page(scalar_cb)` → `get_local_cb_interface(cb).fifo_page_size` (`pool_kernels_common.hpp` L129) → `get_entry_size()`. | Portable | same NEEDS-FIX |
| `out_cb` (pool out) | 1 | `compute_pool_2d.cpp`, `writer_grid_sample_interleaved.cpp` | Portable | compute `reserve_back`/`push_back` (L145/226/257), writer `wait_front`/`pop_front` (L32/40). Canonical cross-kernel FIFO. | Portable | — |
| `pre_tilize_cb`, `fast_tilize_cb` | 1 | `compute_pool_2d.cpp` | Portable | intra-compute tilize staging — canonical `reserve_back`/`push_back`/`wait_front`/`pop_front` (L186-230). No field access. | Portable | on Quasar, PACK→UNPACK staging → **SELF-LOOP-CANDIDATE** (informational) |

## CB portability — `rotate` / BilinearProgramFactory

**Scope:** `rotate_bilinear_program_factory.cpp` → kernels: `reader_rotate_bilinear_interleaved.cpp` (+ `grid_sample_reader_common.hpp`, `fixed_point_arithmetic.hpp`, `pool_kernels_common.hpp`), `compute_pool_2d.cpp` (donor: pool/generic), `writer_grid_sample_interleaved.cpp` (donor: grid_sample).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `input_cb` (c_0) | 1 | reader, `compute_pool_2d.cpp` | Portable | linear FIFO — `input_cb.reserve_back`/`push_back` (L121/143); compute consumes. | Portable | — |
| `scalar_cb` | 1 | reader, `compute_pool_2d.cpp` | Portable | producer `reserve_back`/fill/`push_back` (L137/140); compute consumes. | Portable | — |
| `fill_cb` | 6 | reader | Portable (workaround) | **undesirable but OK hack:** pinned-L1 fill scratch — `fill_cb.get_write_ptr()` (L54) + `zero_out_page(fill_cb)` (L56); no FIFO. **NEEDS-FIX:** `zero_out_page` → `get_local_cb_interface(cb).fifo_page_size` (`pool_kernels_common.hpp` L129) → `get_entry_size()`. Uplift: ScratchpadSpec. | Portable (workaround) | same; autoportable end-state: **ScratchpadSpec** (private L1, no FIFO) |
| `out_cb` (pool out) | 1 | `compute_pool_2d.cpp`, `writer_grid_sample_interleaved.cpp` | Portable | compute produces, donor writer consumes (`wait_front`/`pop_front`). | Portable | — |
| `pre_tilize_cb`, `fast_tilize_cb` | 1 | `compute_pool_2d.cpp` | Portable | intra-compute tilize staging — canonical. | Portable | Quasar PACK→UNPACK → SELF-LOOP-CANDIDATE (informational) |

## CB portability — `grid_sample` / GridSampleNearestProgramFactory

**Scope:** `grid_sample_nearest_program_factory.cpp` → kernels: `writer_grid_sample_nearest_sharded.cpp` (writer0 + writer1 split-reader; also used for interleaved) (+ `grid_sample_reader_common.hpp`, `pool_kernels_common.hpp`). No separate reader/compute — the "writer" does NOC input reads + output writes.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `grid_cb` | 6 | `writer_grid_sample_nearest_sharded.cpp` | Portable (workaround) | **undesirable but OK hack:** `grid_cb.get_write_ptr()` base addr (L192); interleaved NOC-reads grid sticks into it, sharded already resident. Pointer-only, no FIFO. | Portable (workaround) | same; uplift optional: LTA (borrowed grid) |
| `output_cb` | 6 | `writer_grid_sample_nearest_sharded.cpp` | Portable (workaround) | **undesirable but OK hack:** resident output shard (factory attaches `output_tensor.buffer()`); `process_grid_point_nearest` NOC-writes directly to it via offset arithmetic; no FIFO. | Portable (workaround) | same; synchronized pack-into-resident-shard end-state: `borrowed_from` DFB |
| `fill_cb` | 6 | `writer_grid_sample_nearest_sharded.cpp` | Portable (workaround) | **undesirable but OK hack:** pinned-L1 fill scratch — `fill_cb.get_write_ptr()` (L198) + `zero_out_page(fill_cb)` (L199). **NEEDS-FIX:** `zero_out_page` → `get_local_cb_interface(cb).fifo_page_size` (`pool_kernels_common.hpp` L129) → `get_entry_size()`. | Portable (workaround) | autoportable end-state: **ScratchpadSpec** |

## CB portability — `rotate` / NearestProgramFactory

**Scope:** `rotate_nearest_program_factory.cpp` → kernels: `reader_rotate_nearest_interleaved.cpp`, `writer_rotate_nearest_interleaved.cpp` (+ `pool_kernels_common.hpp`).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `output_cb` | 1 | `reader_rotate_nearest_interleaved.cpp`, `writer_rotate_nearest_interleaved.cpp` | Portable | linear FIFO — reader `reserve_back`/`push_back` (L60/108), writer `wait_front`/`pop_front` (L29/43). | Portable | — |
| `fill_cb` | 6 | `reader_rotate_nearest_interleaved.cpp` | Portable (workaround) | **undesirable but OK hack:** pinned-L1 fill scratch — `fill_cb.get_write_ptr()` (L41) + `zero_out_page(fill_cb)` (L43). **NEEDS-FIX:** `zero_out_page` → `get_local_cb_interface(cb).fifo_page_size` (`pool_kernels_common.hpp` L129) → `get_entry_size()`. Uplift: ScratchpadSpec. | Portable (workaround) | autoportable end-state: **ScratchpadSpec** |

## CB portability — `upsample` / UpsampleMultiCoreShardedProgramFactory

**Scope:** `upsample_program_factory_multicore_sharded.cpp` → kernels: `writer_upsample_multi_core_sharded.cpp` (bound as both reader and writer; `is_reader` CT-arg branch).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `in_cb` | 6 | `writer_upsample_multi_core_sharded.cpp` | Portable (workaround) | **undesirable but OK hack:** sync-free resident input shard — `in_cb.get_read_ptr()` + `offset*stick_nbytes` as NOC source (L33,60); no FIFO. Bare L1 address → do-not-churn. | Portable (workaround) | same; uplift optional: LTA (borrowed) |
| `out_cb` | 6 | `writer_upsample_multi_core_sharded.cpp` | Portable (workaround) | **undesirable but OK hack:** resident output shard — NOC `async_read` gathers neighbor sticks into it at `write_offset` (L63-69); no FIFO. | Portable (workaround) | same; synchronized-drain end-state: `borrowed_from` DFB |
| `config_cb` | 6 | `writer_upsample_multi_core_sharded.cpp` | Portable (workaround) | **undesirable but OK hack:** resident config tensor read via `config_cb.get_read_ptr()` as a `uint32_t*` view (L39-41); pointer-only. | Portable (workaround) | same; uplift optional: LTA (borrowed config) |

## Trivially-GREEN factories (summarized, no non-canonical CBs)

- **`upsample` / UpsampleMultiCoreInterleavedProgramFactory** — `reader_upsample_unary_stick_layout_interleaved_start_id.cpp` (`in_cb`: `reserve_back`/`push_back`, Class 1), `writer_upsample_interleaved.cpp` (`out_cb`: `wait_front`/`pop_front`, Class 1), optional compute donor `data_movement/untilize/.../untilize.cpp` (`src_cb`/`out_cb`, Class 1 — uses DFB getter `get_dfb_num_pages` in ASSERTs, no field access). **GREEN, Portable on both arches.**
- **`upsample` / UpsampleNearestFloatProgramFactory** — `reader_upsample_nearest_float.cpp` (`out_cb`: `reserve_back`/`push_back`, Class 1 producer), `writer_upsample_nearest_float.cpp` (`out_cb`: `wait_front`/`pop_front`, Class 1 consumer). **GREEN, Portable on both arches.**

## GATE hits (must be empty to merge)

All 9 are `get_local_cb_interface(...).<field>` accesses. Every one resolves mechanically (merged/existing getter) or via the sanctioned Class-4 DFB ptr-surgery workaround — none needs a new getter or a design decision.

- `ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp:76` — `get_local_cb_interface(cb_id).fifo_num_pages` (`zero_out_tiles`) → **NEEDS-FIX** `get_total_num_entries()` [PR #49197 merged]. Used by grid_sample bilinear readers.
- `ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp:129` — `get_local_cb_interface(cb.get_cb_id()).fifo_page_size` (`zero_out_page`) → **NEEDS-FIX** `get_entry_size()`. Used by grid_sample nearest writer, grid_sample sharded reader, rotate bilinear reader, rotate nearest reader.
- `ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp:46` — `.fifo_num_pages` (`clear_out_tiles`) → **NEEDS-FIX** `get_total_num_entries()`. *Definition present but no in-scope caller in the audited factories* (dead for this group; lives in shared header).
- `ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp:47` — `.fifo_page_size` (`clear_out_tiles`) → **NEEDS-FIX** `get_entry_size()`. Same (no in-scope caller).
- `ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/compute/bilinear.cpp:17` — `get_local_cb_interface(operand).fifo_page_size` (`llk_push_pages_bilinear`) → **NEEDS-FIX** `get_entry_size()`.
- `ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/compute/bilinear.cpp:19` — `get_local_cb_interface(output).fifo_wr_ptr += num_words` — **Class-4 WEIRD-OK** → port to DFB `get_write_ptr()` + manual advance (documented ptr/credit surgery; disable implicit sync on Quasar).
- `ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/compute/bilinear.cpp:20` — `get_local_cb_interface(output).fifo_wr_tile_ptr = 0` — Class-4 WEIRD-OK (same).
- `ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/compute/bilinear.cpp:22` — `... .fifo_wr_ptr >= ... .fifo_limit` (ring-wrap test) — Class-4 WEIRD-OK; ring span via `get_total_num_entries()` × `get_entry_size()`.
- `ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/compute/bilinear.cpp:23` — `... .fifo_wr_ptr -= ... .fifo_size` (ring-wrap) — Class-4 WEIRD-OK (same).

## Blocked on runtime (2xx rollup)

- (none) — no `read_tile_value` / `get_tile_address` / typed-read dependency in any pool kernel. Quasar treatment of the one Class-4 buffer (`bilinear.cpp` `out_cb`) is a documented ptr/credit workaround with `disable_implicit_sync`, not a runtime-API block.

## Notes & follow-ups

- **Class-4 flagship (`upsample` bilinear `out_cb`) is WEIRD-OK, not a blocker.** `llk_push_pages_bilinear` advances the write pointer without posting `tiles_received`/`push_back` credits precisely because the writer-role drain (same `reader_bilinear_multi_core_sharded.cpp` kernel, `is_reader=0`) reads the packed L1 directly. Per the spec Class-4 self-loop table this is **credit-decoupled → NOT a compute self-loop candidate**; the v1 port keeps DFB ptr/credit surgery (`get_write_ptr()` + manual advance, ring span from DFB getters) and opts the DFB out of Quasar implicit sync. Scratchpad/LTA are not applicable because `pack_untilize_dest(out_cb_id)` still targets the CB/DFB id.
- **The YELLOW is entirely the shared `pool_kernels_common.hpp` getter swap.** Four of the five non-trivial factories (both grid_sample, both rotate) pull `zero_out_page`/`zero_out_tiles`, whose only issue is `get_local_cb_interface(...).fifo_page_size`/`.fifo_num_pages` field reads → mechanical swap to `get_entry_size()`/`get_total_num_entries()` (getters landed). Fixing `pool_kernels_common.hpp` once clears the GATE for all four factories at once. `pool_kernels_common` is explicitly named in the spec's "Merged → unblocks" and NEEDS-FIX tables.
- **Pointer-only sharded/nearest CBs are portable, do-not-churn.** grid_sample-nearest (`grid_cb`/`output_cb`/`fill_cb`), rotate `fill_cb`, upsample-sharded (`in_cb`/`out_cb`/`config_cb`), and upsample-bilinear `halo_cb` never form a FIFO; they use `get_read_ptr()`/`get_write_ptr()` as L1 addresses with NOC offset arithmetic (Step-5 WEIRD-OK / already-portable). Optional uplift to `ScratchpadSpec` (private fill scratch) or `LocalTensorAccessor` (borrowed tensor shards) is available but **not required** for the port; no `get_pointer_to_cb_data` anywhere means no LTA is a hard prerequisite.
- **Cross-op donor kernels (in scope, follow-the-reference):**
  - `ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp` — donor (**pool/generic**); compute kernel for **grid_sample bilinear** and **rotate bilinear**. Clean (Class 1, no field access).
  - `ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp` — donor (**grid_sample**); writer for **rotate bilinear**. Clean (Class 1).
  - `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp` (+ `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` / `.inl`) — donor (**data_movement/untilize**); optional compute for **upsample interleaved**. Clean (uses DFB getters).
  - `ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/grid_sample_reader_common.hpp` — shared reader logic for **both grid_sample readers** and (cross-op) the **rotate bilinear reader**. Clean (`get_write_ptr` fill only).
  - `ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp` — shared pool helper (the GATE-bearing header); `ttnn/cpp/ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp` — rotate bilinear math helper (clean).
- **Host-side factories are NOT audited here.** This is a kernel-only CB→DFB readiness audit; host `ProgramSpec`/`DataflowBufferSpec` legality and binding multiplicity are out of scope (tracked by the host audit).
- **Unreferenced kernels under the pool tree** (present but not assigned to any audited factory's `kernel_source`): none material — every `*/kernels/*.cpp` under the three op trees is referenced by a factory variant on the do-list.
