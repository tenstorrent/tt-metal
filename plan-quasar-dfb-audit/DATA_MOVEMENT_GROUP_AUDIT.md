# CB→DFB Kernel Audit: `data_movement` Group (consolidated)

**Date:** 2026-07-15
**Group:** Data-movement ops from the do-list — **17 distinct ops / 35 factory-variant rows / 76 own kernel `.cpp`+`.hpp` files + 12 cross-op donor kernels**
**Audit spec:** `plan-quasar-dfb-audit/cb_dfb_kernel_audit_helper.md`
**Repo state:** #49392 ("Migrate Data Movement Kernels to DataflowBuffer") has **not** landed. Data-movement kernels are already on the **Device 2.0 object API** (`CircularBuffer` / `experimental::CB` / `Noc`) with canonical FIFO ops (`.reserve_back/.push_back/.wait_front/.pop_front`), but not yet renamed to `DataflowBuffer`. The remaining port is the mechanical Class-1 `CircularBuffer`→`DataflowBuffer` rename.

## Group verdict: RED

**Bottom line:** All **17 data-movement ops' own kernels are 100% clean** — canonical Class-1 linear FIFOs plus a small set of pointer-only / private-L1 scratch buffers (Class 6), every one of which is **Portable** or **Portable (workaround)** → GREEN. The group is dragged to **RED by exactly two shared cross-op donor kernels** owned by **`eltwise/unary`** that still read a legacy `LocalCBInterface` field while already declaring a `DataflowBuffer` — a hard **GATE** that is **silent-wrong on Quasar** (get_local_cb_interface returns `LocalCBInterface`; the DFB uses `LocalDFBInterface`). These donors are pulled in by **3 factory variants** (bcast `MultiCoreHW`, copy `DefaultTilized`, pad `PadTileCore`). The fix is a **one-line getter swap** (`get_local_cb_interface(cb).fifo_page_size` → `dfb.get_entry_size()`) owned by the eltwise group, and it clears all three ops (and many other ops repo-wide) at once. No silent-wrong `get_cb_tiles_*`, no `read_tile_value`/`get_tile_address` Quasar-runtime dependency, and no LTA prerequisite (`get_pointer_to_cb_data`) anywhere in the group.

## Group-wide classification scan

Run over all 76 own kernel files + 12 referenced cross-op donor kernels + 6 shared kernel headers (blocker patterns from spec Step 4):

| Pattern (classification) | Hits |
|--------------------------|------|
| `get_local_cb_interface(...).<field>` (**GATE**) | **2 — cross-op `eltwise/unary` donors only** (see GATE hits) |
| `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr` (silent-wrong) | none |
| `read_tile_value` / `get_tile_address` (Quasar-blocked) | none |
| `get_pointer_to_cb_data` (LTA prereq) | none |
| `fifo_wr_ptr` / `fifo_rd_ptr` / `push_back_hold` / `llk_push_pages` (ptr surgery) | none |
| bare `fifo_page_size` / `fifo_num_pages` (field read outside get_local_cb_interface) | none |
| `use<...AddrSelector::WRITE_PTR>` (write-ptr-as-NOC-source; sanctioned DFB API) | 6 (fold ×2, non_zero_indices ×2, sort coordinator ×2) |
| `reserve_back` without `push_back` / private-L1 staging (Class 6 fake-CB) | pad ×1, non_zero_indices ×2, moe_routing_remap ×1 |
| pointer-only `get_read_ptr`/`get_write_ptr` in-place / borrowed shard (Class 6) | move ×1, fold ×1, pad-sharded ×2 |

## Per-op rollup (17 ops)

"Factories" = factory-variant rows on the do-list. All ops' own kernels are on the Device 2.0 object API with canonical CBs (Class 1) unless a CB-table row says otherwise.

| Op | Factories | Kernel CB summary | Verdict |
|----|-----------|-------------------|---------|
| `bcast` | 5 | Class 1 own kernels (readers/compute/writer). **`MultiCoreHW` pulls GATE donor `writer_unary_interleaved_start_id.cpp`** | **RED** (donor GATE; other 4 variants GREEN) |
| `clone` | 1 | Class 1 (read/compute/write, RM+tiled+sharded) | GREEN |
| `copy` | 3 | Class 1 own kernels. **`DefaultTilized` pulls GATE donors `reader_`+`writer_unary_interleaved_start_id.cpp`** | **RED** (donor GATE; RowMajor + SameMemoryConfig GREEN) |
| `fill_pad` | 2 | Class 1 (reader/compute/writer, interleaved+L1-sharded) | GREEN |
| `fill_rm` | 1 | Class 1 producer (reserve/push) | GREEN |
| `fold` | 2 | Class 1 readers + Class 6 pointer-only cb2s writer; writers use sanctioned `WRITE_PTR` NOC source | GREEN (workaround) |
| `gather` | 2 | Class 1 (reader/writer canonical FIFO) | GREEN |
| `indexed_fill` | 1 | data CB globally aliased to output (borrowed); reader fills, writer drains — Class 1-ish WEIRD-OK | GREEN (workaround) |
| `moe_expert_token_remap` | 1 | own writer Class 1; reader = clean CCL donor `reader_all_to_all_combine.cpp` | GREEN |
| `moe_routing_remap` | 1 | Class 1 consumer CBs + one private-L1 scratch CB (`local_weights_cb`) | GREEN (workaround) |
| `move` | 2 | `Move` Class 1; `MoveOverlap` adds Class 6 pointer-only in-place L1 copy | GREEN (workaround) |
| `non_zero_indices` | 1 | Class 1 `input_cb` + 2 Class 6 private-L1 staging CBs (`WRITE_PTR`) | GREEN (workaround) |
| `pad` | 6 | RM/sharded/tile factories Class 1 + one fake-CB scratch (`cb_out1`). **`PadTileCore` pulls GATE donor `reader_unary_interleaved_start_id.cpp`** | **RED** (donor GATE; other 5 variants GREEN) |
| `repeat` | 2 | Class 1 (reserve/push, RM+tiled, interleaved+sharded) | GREEN |
| `sharded_to_interleaved_partial` | 1 | **no own kernels** — all clean Class 1 donors (see notes) | GREEN |
| `sort` | 3 | Class 1 reader/compute/writer + coordinator `WRITE_PTR` NOC source + one credit-only `push_back` on unused CB | GREEN (workaround) |
| `split` | 1 | Class 1 (reader reserve/push, writer wait/pop) | GREEN |

## CB portability (ops that are not trivially Class-1-GREEN)

Only representative / non-canonical CBs are listed per op; every other CB in these ops is canonical Class 1 → Portable both arches.

### `bcast` — factory `BcastMultiCoreHW`
Op root: `ttnn/cpp/ttnn/operations/data_movement/bcast/`
Scope: `bcast_multi_core_hw_program_factory.cpp` → own `reader_bcast_hw_interleaved*.cpp`, `bcast_hw.cpp` (compute) + **donor** `eltwise/unary/.../writer_unary_interleaved_start_id.cpp`.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0`, `cb_in1` | 1 | `reader_bcast_hw_interleaved.cpp`, `bcast_hw.cpp` | Portable | linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_out0` (donor DFB) | 1 | **donor** `writer_unary_interleaved_start_id.cpp` | **Blocked** | **GATE:** `get_local_cb_interface(cb_id_out).fifo_page_size` (line 19) — kernel already uses `DataflowBuffer dfb` but reads legacy field; fix → `dfb.get_entry_size()` | **Blocked** | same field read is **silent-wrong on Quasar** (wrong interface struct) |

*(bcast's other 4 variants — MultiCoreH, MultiCoreW, ShardedHOptimised, ShardedH — use only own Class-1 kernels → GREEN.)*

### `copy` — factory `DefaultTilized`
Op root: `ttnn/cpp/ttnn/operations/data_movement/copy/`
Scope: `copy_default_tilized_program_factory.cpp` → **donors** `eltwise/unary/.../reader_unary_interleaved_start_id.cpp`, `.../writer_unary_interleaved_start_id.cpp`, `ttnn/kernel/compute/eltwise_copy.cpp`.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_in0` (donor DFB) | 1 | **donor** `reader_unary_interleaved_start_id.cpp` | **Blocked** | **GATE:** `get_local_cb_interface(cb_id_in0).fifo_page_size` (line 20) → `dfb.get_entry_size()` | **Blocked** | silent-wrong on Quasar |
| `cb_id_out` (donor DFB) | 1 | **donor** `writer_unary_interleaved_start_id.cpp` | **Blocked** | **GATE:** `get_local_cb_interface(cb_id_out).fifo_page_size` (line 19) → `dfb.get_entry_size()` | **Blocked** | silent-wrong on Quasar |
| compute CBs | 1 | `eltwise_copy.cpp` | Portable | canonical copy pipeline | Portable | — |

*(copy's DefaultRowMajor + SameMemoryConfig variants use own Class-1 kernels — `reader/writer_unary_start_id.cpp`, `reader/writer_unary_stick_start_id.cpp`, `redistribute_pages_row_major_*` — all clean → GREEN.)*

### `pad` — factory `PadTileCore`
Op root: `ttnn/cpp/ttnn/operations/data_movement/pad/`
Scope: `pad_tile_program_factory.cpp` → **donor** `eltwise/unary/.../reader_unary_interleaved_start_id.cpp` + own `writer_pad_tiled.cpp`.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_in0` (donor DFB) | 1 | **donor** `reader_unary_interleaved_start_id.cpp` | **Blocked** | **GATE:** `get_local_cb_interface(cb_id_in0).fifo_page_size` (line 20) → `dfb.get_entry_size()` | **Blocked** | silent-wrong on Quasar |
| `cb_out0` | 1 | `writer_pad_tiled.cpp` | Portable | canonical wait/pop | Portable | — |
| `cb_out1` | 6 | `writer_unary_pad_dims_interleaved.cpp` (variant `PadRmReaderWriter*`) | Portable (workaround) | **undesirable but OK hack:** fake CB — `reserve_back(1)` with no `push_back` (comment: "not pushing anything into CBs, just using the space"); `get_write_ptr()` filled with pad value, DMA'd via `CoreLocalMem`. Uplift: **ScratchpadSpec**. | Portable | autoportable: private-L1 → **ScratchpadSpec** |
| `cb_out0` (sharded) | 6 | `writer_pad_dims_rm_sharded.cpp` | Portable | pointer-only `get_write_ptr()` into output shard as NOC dst (bare L1 cursor — already-portable class); `cb_pad` scratch → **ScratchpadSpec** | Portable | same |

*(pad's other 5 variants — PadRmReaderWriterMultiCoreDefault, PadRmReaderWriterMultiCore, PadRmReaderWriter, PadRmShardedWidthOnly, PadTileMulticore — do not pull the GATE donor; their own kernels are Class 1 / Class 6 pointer-only → GREEN.)*

### `fold` — factories `MultiCore`, `MultiCoreDRAMFold`
| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `src_cb` / `dst_cb` | 6 | `writer_cb2s_row_major.cpp` | Portable | pointer-only `get_read_ptr()`/`get_write_ptr()` as L1 cursors for in-L1 fold copy (bare-pointer already-portable class); no FIFO | Portable | borrowed shards → LTA optional, but bare-ptr copy is portable as-is |
| `input_cb` / `cb_in1` | 6 | `writer_cb2dram_for_tiled_input.cpp`, `writer_cb2dram_for_rm_input.cpp` | Portable (workaround) | **undesirable but OK hack:** `use<experimental::CB::AddrSelector::WRITE_PTR>` as NOC source (sanctioned DFB write-ptr API) | Portable | maps to `dfb.get_write_ptr()` NOC source |
| reader CBs | 1 | `reader_dram2cb_*.cpp` | Portable | canonical reserve/push | Portable | — |

### `non_zero_indices` — factory `NonZeroIndicesProgramFactory`
Spec note said "index_fill WEIRD-OK Class 1-ish" — verified: `non_zero_indices` is single-core and uses staging CBs, `indexed_fill` (separate op below) is the borrowed-to-output one.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `input_cb` | 1 | `non_zero_indices_sc_reader.cpp` | Portable | canonical reserve→read→push→pop | Portable | — |
| `output_cb_0` (count), `output_cb_1` (indices) | 6 | `non_zero_indices_sc_reader.cpp` | Portable (workaround) | **undesirable but OK hack:** private-L1 staging — `reserve_back(1)` up front, fill via `get_write_ptr()`, DMA out via `use<CircularBuffer::AddrSelector::WRITE_PTR>` (lines 249/255), `push_back(1)` at end; no consumer. Uplift: **ScratchpadSpec**. | Portable | autoportable: private-L1 → **ScratchpadSpec** |

### `move` — factory `MoveOverlapProgramFactory`
| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `src_cb` / `dst_cb` | 6 | `reader_unary_local_l1_copy_backwards.cpp` | Portable | pointer-only `get_read_ptr()`/`get_write_ptr()` as L1 base for in-place overlapping L1 copy; no FIFO (single-kernel) | Portable | same (bare-ptr already-portable class) |
| overlap CBs | 1 | `move_interleaved_with_overlap.cpp`, `move_stick_layout_interleaved_with_overlap.cpp` | Portable | canonical reserve/push/wait/pop | Portable | — |

*(`MoveProgramFactory` = plain copy, Class 1 → GREEN.)*

### `moe_routing_remap` — factory `SingleCore`
| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `local_weights_cb` | 6 | `writer_moe_routing_remap.cpp` | Portable (workaround) | **undesirable but OK hack:** `reserve_back(1)`→`get_read_ptr()`→`push_back(1)` used as private scratch (fill + memmove + NOC write), self-consumed within kernel. Uplift: **ScratchpadSpec**. | Portable | autoportable: **ScratchpadSpec** |
| `routing_weights_cb`, `local_weights_idxs_cb` | 1 | `writer_moe_routing_remap.cpp`, `reader_moe_routing_remap.cpp` | Portable | cross-kernel producer→consumer FIFO (`wait_front`; single-shot, no `pop_front` needed) | Portable | — |

### `sort` — factories `CrossCoreDataExchange`, `SingleRowMultiCore`, `SingleRowSingleCore`
| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| value/index/data CBs | 1 | `sort_*.cpp` (compute), `reader_*`, `writer_*` | Portable | canonical FIFO sort pipeline (`generate_index_tile`, wait/pop) | Portable | — |
| `rm_coord_value_row`, `rm_coord_index_row` | 6 | `coordinator_single_row_multi_core.cpp` | Portable (workaround) | **undesirable but OK hack:** `use<CircularBuffer::AddrSelector::WRITE_PTR>` NOC source (lines 87/113; sanctioned write-ptr DFB API) | Portable | maps to `dfb.get_write_ptr()` NOC source |
| `physical_core_lookup_table_cb` | 4 | `writer_cross_core_data_exchange.cpp` | Portable (workaround) | **undesirable but OK hack:** credit-only `push_back(1)` with no `reserve_back`/data (line 102) on an "unused — future improvements" CB; credit decoupled from data. Uplift: **SemaphoreSpec** or delete. | Portable (workaround) | same; on Quasar prefer explicit semaphore over implicit-sync DFB |

### `indexed_fill` — factory `IndexedFillProgramFactory`
| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` (data) | 1 | `indexed_fill_reader.cpp`, `indexed_fill_writer.cpp` | Portable (workaround) | **undesirable but OK hack:** data CB globally allocated to the **output** buffer (borrowed); reader writes into it (== writing output), writer just `wait_front`/`pop_front` to drain credits — Class 1-ish WEIRD-OK. Optional hardening: `borrowed_from` DFB. | Portable | Class 1 FIFO over borrowed output shard |

### `sharded_to_interleaved_partial` — factory `ShardedToInterleavedPartialProgramFactory`
Op root: `ttnn/cpp/ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/`
**No own kernels** — factory (`sharded_to_interleaved_partial_program_factory.cpp`) references only cross-op donors; all clean Class 1.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| input shard CB | 1 | **donor** `eltwise/unary/.../reader_unary_sharded.cpp` | Portable | sharded producer `push_back` (pre-resident shard) | Portable | — |
| output CB | 1 | **donors** `data_movement/sharded/.../writer_unary_sharded_blocks_interleaved_start_id.cpp`, `.../writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | Portable | canonical `wait_front`/`pop_front` → interleaved NOC write | Portable | — |
| compute CBs | 1 | **donor** `ttnn/kernel/compute/eltwise_copy.cpp` | Portable | canonical copy | Portable | — |

*(This op is an interleaved↔sharded transfer, but the kernels use plain canonical FIFO CBs — no Class 6 scratch CB was found in the referenced donors, unlike the generic `interleaved_to_sharded` flagship. The spec's Class-6 scratch warning did not materialize for this specific factory.)*

## GATE hits (must be empty to merge)

Both are **cross-op donor kernels owned by `eltwise/unary`**, already migrated to `DataflowBuffer` but still reading a `LocalCBInterface` field — **silent-wrong on Quasar** (`get_local_cb_interface` returns `LocalCBInterface`; the live buffer is a DFB using `LocalDFBInterface`). Fix is a one-line getter swap; not owned by data_movement.

- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:19` — `get_local_cb_interface(cb_id_out).fifo_page_size` → `dfb.get_entry_size()`. **Referenced by:** bcast `bcast_multi_core_hw_program_factory.cpp:150`; copy `copy_default_tilized_program_factory.cpp:28`.
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp:20` — `get_local_cb_interface(cb_id_in0).fifo_page_size` → `dfb.get_entry_size()`. **Referenced by:** copy `copy_default_tilized_program_factory.cpp:26`; pad `pad_tile_program_factory.cpp:105`.

## Blocked on runtime (2xx rollup)

- (none) — no `read_tile_value` / `get_tile_address` / `get_pointer_to_cb_data` anywhere in the group. No Quasar runtime-API dependency.

## Notes & follow-ups

- **Group RED is entirely donor-driven.** 14 of 17 ops are GREEN. The other 3 (bcast, copy, pad) are GREEN in their own code and go RED only via specific factory variants that borrow the two `eltwise/unary` `*_unary_interleaved_start_id.cpp` donors. Fixing those two donor lines (getter swap) clears bcast `MultiCoreHW`, copy `DefaultTilized`, and pad `PadTileCore` simultaneously — and those donors are shared by a large number of ops repo-wide, so the fix is high-leverage and belongs to the eltwise/unary owner, not data_movement.
- **`use<...AddrSelector::WRITE_PTR>` is not a GATE.** It is the sanctioned Device 2.0 API to use a CB/DFB's write pointer as a NOC source address (maps to `dfb.get_write_ptr()`). It appears in fold, non_zero_indices, and sort; all classified Portable / Portable (workaround). Distinct from the moreh_arange fake-CB pattern only in that most here also pair `push_back`.
- **No fake-CB pattern like moreh_arange in the strict sense except pad `cb_out1`.** pad's `writer_unary_pad_dims_interleaved.cpp` `cb_out1` is the one true `reserve_back`-without-`push_back` private-L1 staging buffer → ScratchpadSpec (autoportable).
- **Cross-op donor kernels referenced by data_movement factories** (all scanned; only the two eltwise readers/writers above carry a GATE):
  - `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` **(GATE)** — bcast, copy
  - `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` **(GATE)** — copy, pad
  - `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` (clean) — sharded_to_interleaved_partial
  - `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp`, `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` (clean) — sharded_to_interleaved_partial
  - `data_movement/sharded/device/kernels/compute/eltwise_copy.cpp`, `ttnn/kernel/compute/eltwise_copy.cpp` (clean) — copy
  - `ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp`, `writer_unary_stick_layout_interleaved_start_id.cpp` (clean) — copy
  - `data_movement/untilize/device/kernels/compute/untilize.cpp` (clean) — fold
  - `ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp` (clean) — moe_expert_token_remap (this donor **is** moe_expert_token_remap's reader; the op has only 1 own kernel — the writer)
- **Verification caveat:** this audit reads factory `.cpp`/`.hpp` only to extract kernel paths; it does not audit host `ProgramSpec` / binding legality. Kernel-side GREEN (for the 14 clean ops) does not certify end-to-end Metal 2.0 host-side portability.
- **Shared kernel headers scanned clean:** `data_movement/common/kernels/common.hpp`, `ccl/common/kernels/moe_utils.hpp`, `ccl/kernel_common/sharding_addrgen.hpp`, `ccl/shared_with_host/sharded_tensor_addr_gen.hpp`, `ttnn/kernel/dataflow/generate_bcast_scalar.hpp`, `pool/device/kernels/experimental_device_api.hpp`.
