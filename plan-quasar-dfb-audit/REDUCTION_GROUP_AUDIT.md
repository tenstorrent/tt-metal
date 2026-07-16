# CB→DFB Kernel Audit: `reduction` Group (consolidated)

**Date:** 2026-07-15
**Group:** Reduction ops from the do-list — **7 op families / 12 factory-variant rows / 34 in-scope kernel files** (incl. 3 cross-op donor kernels + 7 in-tree `kernel_lib`/donor headers)
**Audit spec:** `plan-quasar-dfb-audit/cb_dfb_kernel_audit_helper.md`

## Group verdict: YELLOW

> **Rollup policy (Reading B):** a `get_local_cb_interface(cb).fifo_page_size` / `.fifo_num_pages` **read** where the DFB getter already exists (`get_entry_size()` / `get_total_num_entries()`, both merged) is a **mechanical NEEDS-FIX → YELLOW (Portable, note fix)**, not RED. It is still a **GATE-clear required before merge** (silent-wrong on Quasar until swapped). RED is reserved for no-getter fields, structural blockers, and unresolved design decisions.

**Bottom line:** The reduction group is **YELLOW** — driven by **3 `get_local_cb_interface(...).fifo_page_size` reads** (the "mechanical getter swap" the spec flags at **P4** for `reduction/generic`, eltwise readers, and `*_interleaved_start_id*` kernels) plus **one LTA prerequisite**. Each field read is a one-line swap to `DataflowBuffer::get_entry_size()`. **Two of the three hits live in shared eltwise/unary donor kernels** (`writer_unary_interleaved_start_id.cpp`, `reader_unary_interleaved_start_id.cpp`) that most reduction ops pull in — so the GATE-clear is *inherited*, not authored in the reduction tree. The third is `reader_unary_reduce_rm.cpp` (generic RM path). Clearing these three lines flips those ops to **GREEN**; the **reduce-H sharded reader** stays **YELLOW** via its **LTA** prerequisite (borrowed input shard).

Everything else is clean: **no** silent-wrong (`get_cb_tiles_*_ptr`), **no** `read_tile_value`/`get_tile_address` (Welford is **NOT** Quasar-blocked here — the W-combine runs as CPU float math in `writer_welford_hw.cpp`, not LLK tile reads), **no** `get_pointer_to_cb_data`, **no** `fifo_wr_ptr`/`fifo_rd_ptr`/`push_back_hold` ptr surgery anywhere. `accumulation`, `ema`, `argmax` (both factories), and `moe` are fully GREEN.

## Group-wide classification scan (all in-scope reduction kernels + donors)

| Pattern (classification) | Hits |
|--------------------------|------|
| `get_local_cb_interface(...).field` (GATE) | **3** — all `.fifo_page_size` (mechanical getter swap) |
| `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr` (silent-wrong) | **none** |
| `read_tile_value` / `get_tile_address` (Quasar-blocked) | **none** |
| `get_pointer_to_cb_data` (LTA prereq) | **none** |
| `fifo_wr_ptr` / `fifo_rd_ptr` / `push_back_hold` / `llk_push_pages` (ptr surgery) | **none** |
| `fifo_page_size` / `fifo_num_pages` (mechanical field reads) | **3** (== the 3 GATE hits above) |
| `use<CircularBuffer::AddrSelector::WRITE_PTR>` / raw `get_write_ptr` scratch (Class 6) | argmax readers, moe/prod index-fill, rm clear-template |

## Per-op rollup (7 op families / 12 factory variants)

| Op / DeviceOperation | Factory | Kernel state | CB classes | Verdict |
|----|-----------|--------------|------------|---------|
| `accumulation` (cumsum/cumprod) | AccumulationProgramFactory | CircularBuffer | 1 + 5 (self-loop acc) | **GREEN** |
| `accumulation/ema` | EmaProgramFactory | CircularBuffer | 1 + 5 (self-loop scratch) | **GREEN** |
| `argmax` | ArgMaxMultiCoreProgramFactory | CircularBuffer | 6 (scratch + semaphores) | **GREEN** |
| `argmax` | ArgMaxSingleCoreProgramFactory | CircularBuffer | 6 (scratch) | **GREEN** |
| `generic` ReduceDeviceOp | ReduceMultiCoreH | CB (+DFB donor) | 1, 6, **GATE**; sharded→6 LTA | **YELLOW** (mech. GATE-clear); sharded variant YELLOW |
| `generic` ReduceDeviceOp | ReduceMultiCoreW | CB (+DFB donor) | 1, 6, **GATE** | **YELLOW** (mech. GATE-clear) |
| `generic` ReduceDeviceOp | ReduceSingleCoreHw | CB (+DFB donor) | 1, **GATE** | **YELLOW** (mech. GATE-clear) |
| `generic` WelfordReduceDeviceOp | WelfordReduceProgramFactory | CircularBuffer (+DFB donor) | 1, **GATE** (W/H variant) | **YELLOW** (mech. GATE-clear); HW variant clean |
| `moe` | MoeProgramFactory | CircularBuffer | 1 + 6 (index fill) | **GREEN** |
| `prod` ProdAllDeviceOp | ProdAllProgramFactory | DFB donors | 1, **GATE** (both donors) | **YELLOW** (mech. GATE-clear) |
| `prod` ProdNcDeviceOp | ProdNcProgramFactory | CB (+DFB donor) | 1, 6 (fill), **GATE** | **YELLOW** (mech. GATE-clear) |

> `argmax_nc` (ArgMaxNcDeviceOperation: `argmax_nc_program_factory.cpp`, kernels `reader_argmax_nc.cpp`/`writer_argmax_nc.cpp`/`argmax_nc_compute.cpp`) exists in the tree but is **NOT on the do-list** — not audited. A quick pre-scan showed no GATE/blocker in it.

---

## CB portability — `accumulation` (AccumulationProgramFactory)

**Kernels:** `device/kernels/dataflow/accumulation_reader.cpp`, `device/kernels/compute/accumulation_compute.cpp`, `device/kernels/dataflow/accumulation_writer.cpp` (shared `device/kernels/accumulation_common.hpp`)

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `CB_IN` (`c_0`) | 1 | reader, compute | Portable | linear FIFO → `DataflowBuffer` | Portable | — |
| `CB_OUT` (`c_1`) | 1 | compute, writer | Portable | linear FIFO → `DataflowBuffer` | Portable | — |
| `CB_ACC` (`c_2`) | 5 | compute only | Portable | canonical PACK→UNPACK accumulator, **no ptr surgery** (uses `reserve_back`/`push_back`/`wait_front`/`pop_front` for the recirculation) | Portable | **SELF-LOOP-CANDIDATE**: compute self-loop DFB (PRODUCER+CONSUMER on one compute kernel) |

## CB portability — `accumulation/ema` (EmaProgramFactory)

**Kernels:** `ema/kernels/dataflow/ema_reader.cpp`, `ema/kernels/compute/ema_compute.cpp`, `ema/kernels/dataflow/ema_writer.cpp`

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `src_cb` (`c_0`) | 1 | reader, compute | Portable | linear FIFO | Portable | — |
| `dst_cb` (`c_1`) | 1 | compute, writer | Portable | linear FIFO | Portable | — |
| `trp_cb` (`c_2`) | 5 | compute only | Portable | transpose scratch: `pack_tile`→`push_back` then `wait_front`→`pop_front` on same kernel, canonical | Portable | **SELF-LOOP-CANDIDATE**: compute self-loop DFB |

## CB portability — `argmax` (ArgMaxMultiCoreProgramFactory)

**Kernel:** `device/kernels/reader_argmax_interleaved_multicore.cpp` (headers `argmax_common.hpp`, `argmax_tile_layout.hpp`, `argmax_tile_h_col.hpp`). No compute/writer kernel — argmax runs on the reader RISC.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `src_cb` | 6 | reader | Portable (workaround) | **undesirable but OK hack:** fake CB — NOC-read landing region read directly via `get_write_ptr()`, no `reserve_back`/`push_back`. Uplift: **ScratchpadSpec** | Portable | autoportable: private-L1 → **ScratchpadSpec** |
| `dst_cb` | 6 | reader | Portable (workaround) | **undesirable but OK hack:** `get_write_ptr()` + `use<CircularBuffer::AddrSelector::WRITE_PTR>` staging for NOC out. Uplift: **ScratchpadSpec** | Portable | autoportable: **ScratchpadSpec** |
| `red_idx_cb`, `red_val_cb` | 6 | reader | Portable (workaround) | **undesirable but OK hack:** cross-core partial buffers written at `get_write_ptr() + core_id*offset` via remote unicast, coordinated by `start_sem`/`done_sem`. Uplift: **ScratchpadSpec + SemaphoreSpec** | Portable | autoportable: **ScratchpadSpec + SemaphoreSpec** (semaphores already present) |

## CB portability — `argmax` (ArgMaxSingleCoreProgramFactory)

**Kernels:** `device/kernels/reader_argmax_interleaved.cpp`, `reader_argmax_tile_layout.cpp`, `reader_argmax_tile_layout_h.cpp` (headers as above). No compute/writer.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `src_cb` | 6 | all 3 readers | Portable (workaround) | **undesirable but OK hack:** NOC-read landing region, read via `get_write_ptr()`, no FIFO ops. Uplift: **ScratchpadSpec** | Portable | autoportable: **ScratchpadSpec** |
| `dst_cb` | 6 | all 3 readers | Portable (workaround) | **undesirable but OK hack:** `get_write_ptr()` + `use<...WRITE_PTR>` staging for NOC out. Uplift: **ScratchpadSpec** | Portable | autoportable: **ScratchpadSpec** |

## CB portability — `generic` ReduceDeviceOperation (ReduceMultiCoreH / ReduceMultiCoreW / ReduceSingleCoreHw)

**Kernels:** compute `reduce.cpp` / `reduce_h_neg.cpp` / `reduce_w_neg.cpp` / `reduce_hw_neg.cpp` / `reduce_rm.cpp`; readers `reader_unary_reduce_rm.cpp`, `reader_unary_reduce_universal_start_id.cpp`, `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp`, `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp`; writers `writer_reduce_rm_scalar.cpp`, **donor** `eltwise/unary/.../writer_unary_interleaved_start_id.cpp`, **donor** `data_movement/sharded/.../writer_unary_sharded.cpp`. Headers: `reduce_rm_dataflow_common.hpp`, `kernel_lib/reduce_helpers_{common,compute,dataflow}.hpp`, `dest_helpers.hpp`, `pool/.../experimental_device_api.hpp`.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` (`c_0`) | 1 | universal/transpose readers, compute | Portable | linear FIFO | Portable | — |
| `cb_scaler` (`c_2`) | 1 | readers (`prepare_reduce_scaler`), compute | Portable | scaler tile pushed once, canonical | Portable | — |
| `cb_out` (`output_cb_index`) | 1 | compute, writers | Portable | linear FIFO → writer | Portable | — |
| `cb_id_rm` (`c_24`) | 1 | `reader_unary_reduce_rm.cpp`, compute (`reduce_rm.cpp`) | Portable | **NEEDS-FIX (mechanical):** `reader_unary_reduce_rm.cpp:82` `get_local_cb_interface(cb_id_rm).fifo_page_size` — swap to `get_entry_size()`; buffer itself is canonical Class 1. **GATE-clear before merge.** | Portable | same swap; silent-wrong on Quasar until fixed |
| `cb_clear_value` (`c_4`) | 6 | `reader_unary_reduce_rm.cpp` | Portable | RM identity-pad template: filled once via `get_write_ptr()`, `push_back(1)` w/ no consumer, reused as persistent NOC-source via `get_read_ptr()`. Private-L1 → **ScratchpadSpec** | Portable | autoportable: **ScratchpadSpec** |
| `cb_out` (donor writer, `c_16`/`output_cb_index`) | 1 | `writer_unary_interleaved_start_id.cpp` (interleaved H/W/Hw paths) | Portable | **NEEDS-FIX (mechanical, donor):** `writer_unary_interleaved_start_id.cpp:19` `get_local_cb_interface(cb_id_out).fifo_page_size` — CB is already a `DataflowBuffer`; swap to `dfb.get_entry_size()`. Fix belongs to eltwise/unary donor | Portable | same swap; silent-wrong on Quasar until fixed |
| `cb_out` (sharded writer) | 1/6 | `writer_unary_sharded.cpp` (reduce-H sharded output) | Portable | resident output shard; `wait_front`/`pop_front` readiness handshake only, no data move | Portable | `borrowed_from` DFB or scratchpad |
| `cb_in1` (sharded input) | 6 | `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` | Portable (prereq: LTA) | sync-free **borrowed input shard**: `reserve_back(num_tiles)` once, then read via `get_write_ptr()` + offset as self-unicast NOC source, no per-tile FIFO handoff → **LocalTensorAccessor** | Portable (prereq: LTA) | same |

## CB portability — `generic` WelfordReduceDeviceOperation (WelfordReduceProgramFactory)

**Kernels:** compute `welford_reduce_h.cpp` / `welford_reduce_w.cpp` / `welford_reduce_hw.cpp`; readers `reader_unary_reduce_universal_start_id.cpp`, `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp`; writers `writer_welford_hw.cpp` (HW variant), **donor** `writer_unary_interleaved_start_id.cpp` (W/H variants); donor header `normalization/groupnorm/.../welford_combine.h`.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` (`c_0`) | 1 | readers, compute | Portable | linear FIFO | Portable | — |
| `cb_scaler` (`c_2`) | 1 | reader, compute | Portable | scaler tile, canonical | Portable | — |
| `cb_partial` (`c_21`) | 1 | compute → `writer_welford_hw.cpp` | Portable | compute packs mean/var partials; writer consumes via `wait_front`/`get_read_ptr()` (bare L1 read, CPU float combine) → `pop_front`. **No `read_tile_value`** | Portable | — |
| `cb_combined` (`c_22`) | 1 | `writer_welford_hw.cpp` → compute | Portable | writer produces combined scalar via `get_write_ptr()` write + `push_back`; compute reads back. Canonical FIFO | Portable | — |
| `cb_out` (`c_16`) | 1 | compute → writer | Portable | HW: `writer_welford_hw.cpp` NOC-writes; canonical | Portable | — |
| `cb_out` (donor writer) | 1 | `writer_unary_interleaved_start_id.cpp` (W/H variants) | Portable | **NEEDS-FIX (mechanical, donor):** `writer_unary_interleaved_start_id.cpp:19` `.fifo_page_size` → `dfb.get_entry_size()` (mechanical) | Portable | same swap; silent-wrong on Quasar until fixed |

## CB portability — `moe` (MoeProgramFactory)

**Kernels:** `device/kernels/dataflow/reader_create_index_tensor.cpp`, `device/kernels/compute/moe.cpp`, `device/kernels/dataflow/writer_unary_interleaved.cpp` (moe-local, **not** the eltwise donor).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` / `input_cb` (`c_0`) | 1 | reader, compute | Portable | linear FIFO (double-buffered input stream) | Portable | — |
| `topk_mask_cb`, `expert_mask_cb` | 1 | reader, compute | Portable | resident mask rows, canonical `reserve_back`/`push_back` | Portable | — |
| `scale_cb` (`c_3`) | 1 | writer (`calculate_and_prepare_reduce_scaler`), compute | Portable | scaler tile | Portable | — |
| `index_cb` (intermed) | 6→1 | reader (`generate_index_tile`), compute | Portable | index tile built via `CoreLocalMem(get_write_ptr())` then `push_back(1)` — canonical producer (has consumer) | Portable | — |
| `input_transposed_cb`, `index_transposed_cb`, `values_cb`, `output_ind_cb`, `masked_input_cb`, `cb_cur_max`, `cb_cur_sum`, `out_cb` | 1 | compute, writer | Portable | topk/reduce pipeline CBs — all canonical FIFO (`reserve_back`/`push_back`/`wait_front`/`pop_front`), no ptr surgery, no field reads | Portable | — |

## CB portability — `prod` ProdAllDeviceOperation (ProdAllProgramFactory)

**Kernels:** compute `device/kernels/compute/prod_all.cpp`; **donor** reader `eltwise/unary/.../reader_unary_interleaved_start_id.cpp`; **donor** writer `eltwise/unary/.../writer_unary_interleaved_start_id.cpp`.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `input_cb` (`c_0`) | 1 | donor reader, compute | Portable | **NEEDS-FIX (mechanical, donor):** `reader_unary_interleaved_start_id.cpp:20` `get_local_cb_interface(cb_id_in0).fifo_page_size` → `dfb.get_entry_size()` (mechanical). Buffer itself canonical Class 1 | Portable | same swap; silent-wrong on Quasar until fixed |
| `final_output_cb` (`c_3`) | 1 | compute, donor writer | Portable | **NEEDS-FIX (mechanical, donor):** `writer_unary_interleaved_start_id.cpp:19` `.fifo_page_size` → `dfb.get_entry_size()` (mechanical) | Portable | same swap; silent-wrong on Quasar until fixed |

## CB portability — `prod` ProdNcDeviceOperation (ProdNcProgramFactory)

**Kernels:** reader `device/kernels/dataflow/reader_prod_nc.cpp` (+ `device/kernels/dataflow/utils.hpp`); compute `device/kernels/compute/prod_nc.cpp`; **donor** writer `eltwise/unary/.../writer_unary_interleaved_start_id.cpp`.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` (`c_0`) | 1 | reader, compute | Portable | linear FIFO | Portable | — |
| scaler/fill CB | 6→1 | reader (`utils.hpp` `fill_cb_with_value`) | Portable | value tile built via `CoreLocalMem(get_write_ptr())`, `reserve_back`/`push_back(1)` — canonical producer | Portable | — |
| `cb_out0` (`c_3`) | 1 | compute, donor writer | Portable | **NEEDS-FIX (mechanical, donor):** `writer_unary_interleaved_start_id.cpp:19` `.fifo_page_size` → `dfb.get_entry_size()` (mechanical) | Portable | same swap; silent-wrong on Quasar until fixed |

---

## GATE hits (must be empty to merge)

All three are `get_local_cb_interface(...).fifo_page_size` → **mechanical getter swap** to `DataflowBuffer::get_entry_size()`. No design work; spec **P4**.

1. `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_rm.cpp:82` — `get_local_cb_interface(cb_id_rm).fifo_page_size` — affects `generic` ReduceMultiCoreH (RM path) + ReduceMultiCoreW (RM path).
2. **(donor)** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:19` — `get_local_cb_interface(cb_id_out).fifo_page_size` (CB is already a `DataflowBuffer`) — affects `generic` Reduce H/W/Hw (interleaved), `generic` Welford (W/H variants), `prod` ProdAll, `prod` ProdNc. **Shared repo-wide donor — fix in eltwise/unary, not the reduction tree.**
3. **(donor)** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp:20` — `get_local_cb_interface(cb_id_in0).fifo_page_size` — affects `prod` ProdAll. **Shared repo-wide donor.**

## Blocked on runtime (2xx rollup)

- (none) — **no** `read_tile_value` / `get_tile_address` in any in-scope reduction kernel. In particular, **Welford is not Quasar-blocked**: the parallel W-combine is done as scalar CPU float math in `writer_welford_hw.cpp` (via `welford_combine.h`) reading `cb_partial` through a bare `get_read_ptr()` L1 pointer, not an LLK tile read.

## Notes & follow-ups

- **Why YELLOW not RED (Reading B):** every GATE here is the lowest-severity kind — a mechanical `fifo_page_size` → `get_entry_size()` swap where the getter already exists — so per the rollup policy it is **NEEDS-FIX → YELLOW (Portable, GATE-clear before merge)**, not RED. After the 3 line-level swaps, those ops are **GREEN**; the one **YELLOW** (LTA) below remains until the LocalTensorAccessor migration.
- **2 of 3 GATEs are inherited from shared donors.** `writer_unary_interleaved_start_id.cpp` (and its sibling reader) are used across the whole repo; the reduction ops themselves author only 1 GATE line (`reader_unary_reduce_rm.cpp`). Fixing the donors clears the GATE for `prod`, most of `generic` Reduce, and the Welford W/H writer path simultaneously. This matches the spec NEEDS-FIX table entry for `*_interleaved_start_id*.cpp`.
- **One LTA prerequisite (YELLOW):** the reduce-H **sharded** input CB `cb_in1` in `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` is a sync-free borrowed input shard → must migrate to **LocalTensorAccessor** in the port (host `TensorBinding` + kernel ctor). This variant has **no GATE** — it is YELLOW, not RED, on its own.
- **Self-loop candidates (2xx, informational):** `CB_ACC` (accumulation) and `trp_cb` (ema) are canonical PACK→UNPACK accumulators on a single compute kernel — flag as **SELF-LOOP-CANDIDATE** for Quasar compute self-loop binding. Neither uses ptr surgery, so both are cleanly Portable on 1xx too.
- **Class-6 scratch is autoportable:** argmax's `src`/`dst`/`red_idx`/`red_val` CBs and the RM `cb_clear_value` template are private-L1 fake-CBs (`reserve_back` w/o consumer, `use<...WRITE_PTR>`, or NOC-landing regions) → **ScratchpadSpec** (+ existing semaphores for argmax multicore cross-core reduction). GREEN.
- **DFB migration state:** unlike the moreh group (mechanically ported by #49430), reduction kernels are mostly **still on `CircularBuffer`** — only the eltwise/unary donors and `kernel_lib` reduce helpers use `DataflowBuffer`. The `CircularBuffer`→`DataflowBuffer` rename is the standard mechanical Class-1 port and does not change any verdict here.
- **Host-side factories not covered.** Kernel-only audit; SPSC/endpoint legality and `DataflowBufferSpec` fit are tracked by the host audit.
- **`argmax_nc` not on do-list** — present in the tree but excluded from this audit (pre-scan showed no GATE/blocker).
