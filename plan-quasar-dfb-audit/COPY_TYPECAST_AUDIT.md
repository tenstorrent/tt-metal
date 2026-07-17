# CB→DFB Kernel Audit: `copy/typecast` (consolidated — all 4 factories)

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/copy/typecast/`
**Audit spec:** `plan-quasar-dfb-audit/cb_dfb_kernel_audit_helper.md`

**Scope (4 factories on the do-list, from `select_program_factory` in `device/typecast_device_op.cpp`):**

- `TypecastProgramFactory` (`device/typecast_program_factory.cpp:17`) → kernels:
  `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` *(donor)*,
  `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` *(donor)*,
  `copy/typecast/device/kernels/compute/eltwise_typecast.cpp` *(own)*
- `TypecastSubgridProgramFactory` (**same file**, `device/typecast_program_factory.cpp:193`) → **identical kernel set** to `TypecastProgramFactory` (interleaved reader/writer donors + own compute)
- `TypecastShardedProgramFactory` (`device/typecast_sharded_program_factory.cpp:17`) → kernels:
  `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` *(donor)*,
  `copy/typecast/device/kernels/compute/eltwise_typecast.cpp` *(own)* — **no writer kernel** (output shard resident)
- `TypecastRowMajorChunkedProgramFactory` (`device/typecast_rm_chunked_program_factory.cpp:65`) → kernels:
  `copy/typecast/device/kernels/dataflow/reader_typecast_rm_chunked.cpp` *(own)*,
  `copy/typecast/device/kernels/dataflow/writer_typecast_rm_chunked.cpp` *(own)*,
  `copy/typecast/device/kernels/compute/eltwise_typecast.cpp` *(own)*

**SCAN_FILES = 6 kernels** (3 own + 3 cross-op donors). No in-tree ttnn headers pulled in — every `#include` in these kernels resolves under `api/` (HW-infra: `dataflow_api.h`, `noc.h`, `circular_buffer.h`, `dataflow_buffer.h`, `core_local_mem.h`, `compute/*`), which the spec excludes from the closure.

> **Note — out of scope:** `experimental/quasar/typecast` and `experimental/copy/typecast` are **different ops**, not audited here.

## Overall verdict: YELLOW

> **Rollup policy (Reading B):** a `get_local_cb_interface(cb).fifo_page_size` / `.fifo_num_pages` **read** where the DFB getter already exists (`get_entry_size()` / `get_total_num_entries()`, both merged) is a **mechanical NEEDS-FIX → YELLOW (Portable, "GATE-clear required before merge")**, not RED. It is still silent-wrong on Quasar until swapped. RED is reserved for no-getter fields (`fifo_size`/`fifo_limit`), field *writes* / ptr-surgery, structural blockers, and unresolved design decisions — **none of which occur here.**

**Bottom line:** `copy/typecast` is **YELLOW**, driven entirely by **2 inherited donor GATEs** — both `get_local_cb_interface(...).fifo_page_size` reads in the shared eltwise/unary interleaved reader/writer (`reader_unary_interleaved_start_id.cpp:20`, `writer_unary_interleaved_start_id.cpp:19`). Both donors are **already `DataflowBuffer`**, so each is a one-line mechanical swap to `dfb.get_entry_size()`. These are the **exact same two donor GATEs** already flagged in the reduction and data_movement group audits — the typecast op authors **zero GATE lines of its own**. Clearing the two donor lines flips the two interleaved factories to GREEN.

Everything else is clean: **no** silent-wrong (`get_cb_tiles_*_ptr`), **no** `read_tile_value`/`get_tile_address` (typecast is **not** Quasar-blocked), **no** `get_pointer_to_cb_data` (no LTA prerequisite), **no** `fifo_wr_ptr`/`fifo_rd_ptr`/`push_back_hold`/`llk_push_pages` ptr surgery, **no** `fifo_size`/`fifo_limit` no-getter fields. The RM-chunked and sharded factories are fully GREEN; their `get_read_ptr()`/`get_write_ptr()` uses are bare L1 cursors for NOC (portable, already fine today). All CBs are canonical Class-1 linear FIFOs (interleaved/RM-chunked) or borrowed resident shards (sharded).

## Group-wide classification scan (all 6 in-scope kernels)

| Pattern (classification) | Hits |
|--------------------------|------|
| `get_local_cb_interface(...).field` (GATE) | **2** — both `.fifo_page_size` (mechanical getter swap), both in eltwise/unary **donors** |
| `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr` (silent-wrong) | **none** |
| `read_tile_value` / `get_tile_address` (Quasar-blocked) | **none** |
| `get_pointer_to_cb_data` (LTA prereq) | **none** |
| `fifo_wr_ptr` / `fifo_rd_ptr` / `push_back_hold` / `llk_push_pages` / `fifo_size` / `fifo_limit` (ptr surgery / no-getter fields) | **none** |
| `fifo_page_size` / `fifo_num_pages` (mechanical field reads) | **2** (== the 2 GATE hits above) |
| `get_read_ptr` / `get_write_ptr` (portable L1 cursor) | **4** — all in own RM-chunked reader/writer (bare NOC addresses) |

## Per-factory rollup

| Factory | Kernels (own / donor) | CB classes | GATE | Verdict |
|---------|----------------------|------------|------|---------|
| `TypecastProgramFactory` | own compute + 2 interleaved donors | 1 (both CBs) | 2 donor `.fifo_page_size` (mechanical) | **YELLOW** (donor GATE-clear before merge) |
| `TypecastSubgridProgramFactory` | **same kernel set** as above | 1 (both CBs) | same 2 donor GATEs | **YELLOW** (donor GATE-clear before merge) |
| `TypecastShardedProgramFactory` | own compute + sharded donor reader | 1/6 borrowed (both CBs) | none | **GREEN** |
| `TypecastRowMajorChunkedProgramFactory` | all own (reader/writer/compute) | 1 (both CBs) | none | **GREEN** |

---

## CB portability — `TypecastProgramFactory` (interleaved, TILE / non-sharded fallback)

**Kernels:** donor reader `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`, donor writer `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`, own compute `copy/typecast/device/kernels/compute/eltwise_typecast.cpp`. Both donors are already `DataflowBuffer`; compute is on `CircularBuffer`.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `src0_cb` (`c_0`, input) | 1 | donor reader, compute (`cb_in`) | Portable | **NEEDS-FIX (mechanical, donor):** `reader_unary_interleaved_start_id.cpp:20` `get_local_cb_interface(cb_id_in0).fifo_page_size` — CB is already a `DataflowBuffer`; swap to `dfb.get_entry_size()`. Buffer itself is canonical Class-1 linear FIFO (`reserve_back`→NOC read→`push_back` / `wait_front`→`copy_tile`→`pop_front`). **GATE-clear before merge.** | Portable | same swap; silent-wrong on Quasar until fixed |
| `output_cb` (`c_2`, output) | 1 | compute (`cb_out`), donor writer | Portable | **NEEDS-FIX (mechanical, donor):** `writer_unary_interleaved_start_id.cpp:19` `get_local_cb_interface(cb_id_out).fifo_page_size` → `dfb.get_entry_size()`. Buffer canonical Class-1 (`reserve_back`/`push_back` on compute; `wait_front`/NOC write/`pop_front` on writer). **GATE-clear before merge.** | Portable | same swap; silent-wrong on Quasar until fixed |

## CB portability — `TypecastSubgridProgramFactory` (interleaved, sub_core_grids)

**Kernels:** **identical set** to `TypecastProgramFactory` (donor `reader_unary_interleaved_start_id.cpp`, donor `writer_unary_interleaved_start_id.cpp`, own `eltwise_typecast.cpp`). Only the host core-distribution differs — kernel CB usage and GATE hits are the same.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `src0_cb` (`c_0`, input) | 1 | donor reader, compute | Portable | same as `TypecastProgramFactory` — `reader_unary_interleaved_start_id.cpp:20` `.fifo_page_size` → `dfb.get_entry_size()` (mechanical, donor). Canonical FIFO. **GATE-clear before merge.** | Portable | same swap; silent-wrong on Quasar until fixed |
| `output_cb` (`c_2`, output) | 1 | compute, donor writer | Portable | same as `TypecastProgramFactory` — `writer_unary_interleaved_start_id.cpp:19` `.fifo_page_size` → `dfb.get_entry_size()` (mechanical, donor). Canonical FIFO. **GATE-clear before merge.** | Portable | same swap; silent-wrong on Quasar until fixed |

## CB portability — `TypecastShardedProgramFactory` (2D sharded, L1-resident)

**Kernels:** donor reader `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` (`DataflowBuffer`; only `dfb.push_back(num_tiles_per_core)` — data pre-resident, no NOC move), own compute `eltwise_typecast.cpp`. **No writer kernel** — output shard stays resident in L1. Both CBs overlay the tensor buffers (`.buffer = input.buffer()` / `output.buffer()`).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `in_cb` (`c_0`, borrowed input shard) | 1/6 | donor reader, compute (`cb_in`) | Portable | resident **borrowed input shard**: reader is single-ended synchronized producer — posts `dfb.push_back(num_tiles_per_core)` credits for pre-loaded L1 data (no `get_write_ptr`/NOC read); compute consumes canonically (`wait_front`/`copy_tile`/`pop_front`). No ptr surgery, no field reads. → `borrowed_from` DFB | Portable | `borrowed_from` DFB (or scratchpad); no GATE |
| `out_cb` (`c_2`, borrowed output shard) | 1/6 | compute (`cb_out`) | Portable | single-ended synchronized pack into **resident output shard**: compute `reserve_back`/`pack_tile`/`push_back`; no drain kernel (data resident). → `borrowed_from` DFB | Portable | `borrowed_from` DFB; on Quasar single-ended packer into resident shard (borrowed) — no ptr hack needed |

## CB portability — `TypecastRowMajorChunkedProgramFactory` (ROW_MAJOR, DRAM chunked)

**Kernels:** all **own** — reader `reader_typecast_rm_chunked.cpp`, writer `writer_typecast_rm_chunked.cpp`, compute `eltwise_typecast.cpp` (all on `CircularBuffer`). Trivially Class-1 GREEN.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `input_cb` (`c_0`) | 1 | reader, compute | Portable | canonical linear FIFO: reader `reserve_back`→`get_write_ptr()` (bare L1 addr for NOC `async_read` via `CoreLocalMem`)→`push_back`; compute `wait_front`/`pop_front`. `get_write_ptr()` is a portable L1 cursor (reader_typecast_rm_chunked.cpp:38,55) | Portable | — |
| `output_cb` (`c_2`) | 1 | compute, writer | Portable | canonical linear FIFO: compute `reserve_back`/`push_back`; writer `wait_front`→`get_read_ptr()` (bare L1 addr for NOC `async_write`)→`pop_front`. `get_read_ptr()` portable L1 cursor (writer_typecast_rm_chunked.cpp:38,55) | Portable | — |

---

## GATE hits (must be empty to merge)

Both are `get_local_cb_interface(...).fifo_page_size` reads on CBs that are **already `DataflowBuffer`** → **mechanical getter swap** to `DataflowBuffer::get_entry_size()` (spec P4 / "Runtime fixes: `get_entry_size()` merged"). No design work. **Both are inherited from shared eltwise/unary donors — fix belongs in the donor tree, not `copy/typecast`.**

1. **(donor)** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp:20` — `get_local_cb_interface(cb_id_in0).fifo_page_size` → `dfb.get_entry_size()`. Affects `TypecastProgramFactory` + `TypecastSubgridProgramFactory` (input CB `c_0`). Same donor line flagged in the reduction (`prod`) and data_movement audits.
2. **(donor)** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:19` — `get_local_cb_interface(cb_id_out).fifo_page_size` → `dfb.get_entry_size()`. Affects `TypecastProgramFactory` + `TypecastSubgridProgramFactory` (output CB `c_2`). Repo-wide shared donor — same line flagged across reduction (`generic` Reduce, Welford, `prod`) and data_movement.

`TypecastShardedProgramFactory` and `TypecastRowMajorChunkedProgramFactory` have **no GATE hits**.

## Blocked on runtime (2xx rollup)

- **(none)** — no `read_tile_value` / `get_tile_address` / `get_pointer_to_cb_data` in any in-scope typecast kernel. Typecast is **not** Quasar-blocked and has **no LTA prerequisite**.

## Notes & follow-ups

- **Why YELLOW not RED (Reading B):** the only findings are 2 `.fifo_page_size` reads where the DFB getter (`get_entry_size()`) already exists — the lowest-severity kind of GATE. Per the rollup policy these are mechanical **NEEDS-FIX → YELLOW (Portable, GATE-clear before merge)**, not RED. After the 2 donor line swaps, all four factories are **GREEN**.
- **Both GATEs are inherited, not authored.** `copy/typecast` writes zero `get_local_cb_interface` lines itself. Both hits live in `reader_/writer_unary_interleaved_start_id.cpp` (eltwise/unary donors used repo-wide). Fixing those two donor lines clears the GATE for the two interleaved typecast factories **and** the reduction/data_movement ops that share them — a single shared fix.
- **`TypecastSubgridProgramFactory` shares the interleaved kernel set** — it is a second `create_descriptor` in `typecast_program_factory.cpp` (lines 193–334), selected only when `sub_core_grids` is set (TILE input). Same donors, same 2 GATEs, same CB layout (`c_0`/`c_2`) as `TypecastProgramFactory`; only host core distribution differs.
- **Sharded factory is genuinely GREEN (no LTA prereq).** Unlike the reduction reduce-H sharded input (which read a borrowed shard sync-free via `get_write_ptr()` + offset → LTA), the typecast sharded reader (`reader_unary_sharded.cpp`) does a **synchronized single-ended credit post** (`dfb.push_back`) with a real `push_back`→`wait_front` handoff to compute. That is a `borrowed_from` DFB (Portable), not a sync-free pointer read — so **no LocalTensorAccessor migration is required**. Both sharded CBs overlay tensor buffers (borrowed backing) but move data through canonical credits.
- **RM-chunked factory is all own kernels, fully canonical Class-1.** `get_write_ptr()`/`get_read_ptr()` are used only as bare L1 byte addresses for `noc.async_read`/`async_write` via `CoreLocalMem` — the ~600-file "already portable, do not churn" idiom. No workaround/hack rows anywhere in this op.
- **DFB migration state:** the own compute (`eltwise_typecast.cpp`) and own RM-chunked reader/writer are still on `CircularBuffer`; the eltwise/unary donors are already `DataflowBuffer`. The `CircularBuffer`→`DataflowBuffer` rename on the own kernels is the standard mechanical Class-1 port and changes no verdict here.
- **Host-side factories not covered.** Kernel-only audit; SPSC/endpoint legality and `DataflowBufferSpec` fit are tracked by the host audit.
