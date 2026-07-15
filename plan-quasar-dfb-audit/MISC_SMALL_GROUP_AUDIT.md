# CB→DFB Kernel Audit: Misc Small Group (consolidated)

**Date:** 2026-07-15
**Group:** `full`, `examples/example`, `sliding_window/halo`, `prefetcher`, `point_to_point`, `debug`
**Audit spec:** `plan-quasar-dfb-audit/cb_dfb_kernel_audit_helper.md`
**Arch columns:** **1xx** = WH/BH · **2xx** = Quasar

## Group verdict: RED

**Bottom line:** Two ops are clean (**full**, **point_to_point** → GREEN) and one has no CB at all (**debug** → N/A). The group is dragged **RED** by three ops:

- **example** — its two data kernels are shared **eltwise/unary donor** kernels that read `get_local_cb_interface(cb).fifo_page_size` → **GATE** (mechanical getter swap, but a hard stop until cleared).
- **halo** — Class 3 scatter into the output shard (`out_cb`), sync-free borrowed config-CB reads (LTA prereq), and a delicately-balanced split-reader shared input shard. 1xx is largely WEIRD-OK; **2xx needs a scatter design decision**.
- **prefetcher** — a **remote / global CB** (`c_31`) driven by `experimental::remote_cb_*` APIs (a non-DFB primitive whose backing header does `fifo_wr_ptr`/`fifo_rd_ptr` surgery on both remote and local CB interfaces), plus a hand-rolled ring buffer with wrap-around pointer + trid credit decoupling. **Structural — no DFB equivalent today.**

## Per-op rollup

| Op | Factory/-ies audited | CBs | Worst class | 1xx | 2xx | Verdict |
|----|----------------------|-----|-------------|-----|-----|---------|
| `full` | Interleaved / Sharded / NDSharded | 1 (`cb_value`) | 6 (DM self-loop) | Portable (workaround) | Portable (ScratchpadSpec) | **GREEN** |
| `examples/example` | MultiCore, SingleCore | 2 | 1 + **GATE** | Blocked (GATE) | Blocked (GATE) | **RED** |
| `sliding_window/halo` | UntilizeWithHalo | 6 | 3 (scatter) | Portable (workaround) | Blocked (needs-design) | **RED** |
| `prefetcher` | DramPrefetcher | 4 | 6 (remote CB) | Portable (workaround) | Blocked (structural) | **RED** |
| `point_to_point` | Send, Receive, LocalCopy | 5 | 6 (fabric scratch) | Portable / workaround | Portable / workaround | **GREEN** |
| `debug` | ApplyDeviceDelay | 0 | — | N/A (no CB) | N/A (no CB) | **GREEN (N/A)** |

---

## `full`

**Op root:** `ttnn/cpp/ttnn/operations/full/`
**Scope:** `FullInterleavedProgramFactory`, `FullShardedProgramFactory`, `FullNDShardedProgramFactory` → kernels: `device/kernels/writer_full.cpp`, `writer_full_sharded.cpp`, `writer_full_nd_sharded.cpp`, `full_kernel_common.hpp`

### Overall verdict: GREEN

All three writer kernels are structurally identical: a **single DM kernel** does `reserve_back(1)` → fill one page at `get_write_ptr()` → `push_back(1)` → `wait_front(1)` → NOC-broadcast that page to many output pages → `pop_front(1)`. Producer and consumer are the **same kernel** on one CB → **Class 6 (DM self-loop / private-L1 staging)**, not a cross-kernel FIFO. Backing is private L1 (the fill value is a scalar, not a tensor). Kernels use the object API `CircularBuffer` from `api/dataflow/circular_buffer.h`; only clean `get_write_ptr()` L1-address use — no field surgery, no `get_local_cb_interface`. The interleaved factory binds the same kernel to two CBs (`c_0` writer / `c_1` reader) as two independent self-loops; this is a host binding-multiplicity concern (OUT-OF-SCOPE for this kernel audit).

### CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_value` (`c_0`/`c_1`) | 6 | `writer_full.cpp`, `writer_full_sharded.cpp`, `writer_full_nd_sharded.cpp` | Portable (workaround) | **undesirable but OK hack:** Class 6 DM self-loop — `reserve_back`/`push_back`/`wait_front`/`pop_front` all in one writer kernel used as private-L1 staging; `get_write_ptr()` as L1 fill address; broadcast via `Noc::async_write(cb, …)`. Uplift: **ScratchpadSpec**. | Portable | autoportable: private-L1 staging → **ScratchpadSpec** (Quasar rejects DM self-loop DFB; no FIFO credits / no sem needed — single kernel) |

### GATE hits: (none)
### Blocked on runtime: (none)

---

## `examples/example`

**Op root:** `ttnn/cpp/ttnn/operations/examples/example/`
**Scope:** `ExampleDeviceOperation::MultiCore` + `SingleCore` → **cross-op donor kernels** (eltwise/unary): `reader_unary_interleaved_start_id.cpp`, `writer_unary_interleaved_start_id.cpp`, `compute/eltwise_sfpu.cpp`. The op's own `device/kernels/**` (`blank.cpp`, `reader_unary.cpp`, `reader_binary_diff_lengths.cpp`, `writer_unary.cpp`, `compute/eltwise_sfpu.cpp`) are **unreferenced** (informational — not scanned/gated).

### Overall verdict: RED

The CBs are canonical Class 1 linear FIFOs (compute already uses `DataflowBuffer` cleanly), **but** both donor data kernels read the page size via `get_local_cb_interface(cb).fifo_page_size` — a **GATE** hit. Fix is mechanical (`get_entry_size()` exists today), but per spec a non-empty unresolved GATE list rolls up **RED** and must clear before the port merges. This is exactly the P4 "eltwise … 1 + GATE (field read) — mechanical getter swap" case.

### CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_in0` (`c_0`) | 1 | `reader_unary_interleaved_start_id.cpp`, `eltwise_sfpu.cpp` | Blocked | **GATE:** `get_local_cb_interface(cb_id_in0).fifo_page_size` (reader:20) — swap to `get_entry_size()` before port merges | Blocked | same |
| `cb_output` (`c_2`) | 1 | `eltwise_sfpu.cpp`, `writer_unary_interleaved_start_id.cpp` | Blocked | **GATE:** `get_local_cb_interface(cb_id_out).fifo_page_size` (writer:19) — swap to `get_entry_size()` | Blocked | same |

### GATE hits (must be empty to merge)
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp:20` — `get_local_cb_interface(cb_id_in0).fifo_page_size` → `get_entry_size()`
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:19` — `get_local_cb_interface(cb_id_out).fifo_page_size` → `get_entry_size()`

### Blocked on runtime: (none — GATE is a mechanical getter swap, available today)

---

## `sliding_window/halo`

**Op root:** `ttnn/cpp/ttnn/operations/sliding_window/halo/`
**Scope:** `UntilizeWithHaloProgramFactory` → kernels: `device/kernels/dataflow/halo_gather.cpp`, `device/kernels/compute/pack_untilize.cpp`; closure donors: `pool/device/kernels/experimental_device_api.hpp` (`experimental::CB = CircularBuffer`), `kernel_lib/untilize_helpers.hpp` (DataflowBuffer-based, canonical).

### Overall verdict: RED

The gather kernel uses `experimental::CB` (a `CircularBuffer` alias) with **pointer-only** access (`get_read_ptr()`/`get_write_ptr()` + computed offsets, `use<AddrSelector::READ_PTR>`) — **no field surgery, no `get_local_cb_interface`**. But it mixes several non-canonical patterns: a **Class 3 scatter** into the output shard (`out_cb`), **sync-free borrowed** config reads (LTA prereqs), a **split-reader shared input shard** (`src_cb`) with hand-balanced reserve/push/wait/pop confined to one reader, and private padding scratch. 1xx keeps the Gen1 ptr scatter (WEIRD-OK); **2xx must choose a scatter strategy** (strided multi-producer DFB vs multi-DFB combine vs disable split reader) → design decision → RED for 2xx.

### CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `untilize_out_cb0/1` (compute→gather `in_cb`) | 1 | `pack_untilize.cpp`, `halo_gather.cpp` | Portable | double-buffered untilize output; canonical FIFO via `untilize_helpers` (DFB); gather consumes with `wait_front`/`pop_front` + `AddrSelector::READ_PTR` NOC source | Portable | — |
| `src_cb` (`c_0`, input shard) | 6 | `halo_gather.cpp` | Portable (workaround) | **undesirable but OK hack:** shared resident input shard read in-place by both split readers; reserve/push/wait/pop confined to `block_start_offset==0` reader to keep received/acked balanced (halo_gather:316-372). Uplift: scratchpad + per-reader sems, or LTA (borrowed shard) | Blocked | needs-design: shared-shard split-reader handshake → **scratchpad + semaphores** (or LTA borrowed view); degenerate FIFO does not map to a Quasar DFB |
| `out_cb` (output shard) | 3 | `halo_gather.cpp` | Portable (workaround) | **undesirable but OK hack:** Class 3 scatter — `get_write_ptr()` base + computed `dst_offset` stick writes (halo_gather:108,205,246). Keep Gen1 ptr scatter (no strided DFB on 1xx) | Blocked | needs-design: **strided multi-producer DFB** (preferred) vs multi-DFB combine vs disable split reader — arch fork, factory must choose |
| `padding_config0/1` | 6 | `halo_gather.cpp` | Portable (prereq: LTA) | sync-free borrowed config read via `get_read_ptr()` (halo_gather:105); DRAM mode stages via `async_read` then reads → **LocalTensorAccessor** (borrowed) or ScratchpadSpec (DRAM-staged) | Portable (prereq: LTA) | same |
| `gather_config0/1` | 6 | `halo_gather.cpp` | Portable (prereq: LTA) | sync-free borrowed config read via `get_read_ptr()` (halo_gather:349) → **LocalTensorAccessor** | Portable (prereq: LTA) | same |
| `pad_cb0/1` | 6 | `halo_gather.cpp` | Portable | private-L1 padding scratch: `get_write_ptr()` fill + `get_read_ptr()` source (halo_gather:330,335) → **ScratchpadSpec** | Portable | same |

### GATE hits: (none)
### Blocked on runtime: (none — 2xx blocks are design decisions, not runtime-API waits)

---

## `prefetcher`

**Op root:** `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/`
**Scope:** `DramPrefetcherOperation` (single-descriptor) → kernels: `device/kernels/reader_dram.cpp`, `writer_l1.cpp`; closure donors: `ccl/kernel_common/worker_sync_utils.hpp` (clean), **`tt_metal/hw/inc/api/remote_circular_buffer.h`** (remote-CB primitive).

### Overall verdict: RED

The writer pushes to a **remote / global circular buffer** (`c_31`, from `operation_attributes.global_cb`) via `experimental::remote_cb_reserve_back` / `remote_cb_push_back_and_write_pages` / `remote_cb_sender_barrier` / `update_remote_cb_config_in_l1`. This is a **non-DFB primitive**: `remote_circular_buffer.h` reads and **writes** `fifo_wr_ptr`, `fifo_rd_ptr`, `fifo_size`, `fifo_page_size` on both `RemoteSender/ReceiverCBInterface` and (in `update_remote_cb_config_in_l1`) on the **`LocalCBInterface`** itself (`remote_circular_buffer.h:434-438`). The op kernel calls sanctioned `experimental::` helpers (not direct field surgery), so this is not a per-kernel GATE, but the primitive has **no DFB equivalent** — port is **structural / blocked** on a Quasar remote-/global-CB story. Separately, the reader hand-rolls a ring buffer with wrap-around `l1_write_addr` and trid-based multi-block credit decoupling (`cb_reserve_back(… *2)`) → Class 2/4.

### CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id` / `local_cb_id` (read ring) | 2/4 | `reader_dram.cpp`, `writer_l1.cpp` | Portable (workaround) | **undesirable but OK hack:** manual ring — `get_write_ptr(cb_id)` base + wrap at `l1_buffer_end_addr`, trid barriers, `cb_reserve_back(…*2)` reserving two blocks ahead so credits are decoupled from write addresses (reader_dram:57-109). Writer side is a clean Class-1 consumer. Uplift: scratchpad + semaphores | Blocked | needs-design: ring + trid pipeline → **scratchpad + semaphores** (or disable double-block prefetch); ptr/credit decoupling does not map to a canonical DFB |
| `addrs_cb_id` | 6 | `reader_dram.cpp` | Portable (prereq: LTA) | sync-free borrowed read of the tensor-address table via `get_read_ptr(addrs_cb_id)` (reader_dram:37-38) → **LocalTensorAccessor** | Portable (prereq: LTA) | same |
| `sync_cb_id` | 6 | `reader_dram.cpp`, `writer_l1.cpp` | Portable | used purely as a cross-kernel exit signal (`cb_push_back`/`cb_wait_front`, 1 credit) → **SemaphoreSpec** | Portable | same |
| `remote_cb_id` (`c_31`, global CB) | 6 | `writer_l1.cpp` | Portable (workaround) | **undesirable but OK hack:** remote/global CB via `experimental::remote_cb_*` (remote_circular_buffer.h); works on WH/BH today. Not a DFB — a distinct cross-core primitive | Blocked | **STRUCTURAL:** no DFB equivalent; backing primitive writes `fifo_wr_ptr`/`fifo_rd_ptr` on remote + local CB interfaces (remote_circular_buffer.h:167,225,434-438). Needs a Quasar global/remote-CB design |

### GATE hits: (none in op kernels — field surgery is confined to the framework `remote_circular_buffer.h` primitive, invoked via sanctioned `experimental::` helpers)
### Blocked on runtime (2xx rollup)
- Remote/global CB (`remote_cb_id`, `c_31`) — needs a Quasar remote-/global-CB primitive (no DFB mapping). Structural, not a listed in-flight runtime fix.

---

## `point_to_point`

**Op root:** `ttnn/cpp/ttnn/operations/point_to_point/`
**Scope:** `PointToPointOp` SendReceive (+ LocalCopy) → kernels: `device/kernels/dataflow/writer_send.cpp`, `reader_receive.cpp`, `reader_unary_interleaved_start_id_gen.cpp`, `writer_unary_interleaved_start_id_gen.cpp`, `common.hpp`; closure donors: `data_movement/common/kernels/common.hpp`, `ccl/common/kernels/moe_utils.hpp`, fabric headers (all clean).

### Overall verdict: GREEN

Cross-chip transfer is done over **fabric** (`FabricConnectionManager`, packet headers, `noc_semaphore_wait`), with local CBs for data FIFO and packet staging. All kernels use the **legacy free-function CB API** (`cb_reserve_back(id)`, `get_write_ptr(id)`, …) — a mechanical rename to `DataflowBuffer`, not a blocker. **No `get_local_cb_interface`, no field surgery, no scatter.** The data CBs are canonical Class 1; the fabric packet-header / packet buffers are single-kernel Class 6 scratch → ScratchpadSpec.

### CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_in0` (send reader → sender) | 1 | `reader_unary_interleaved_start_id_gen.cpp`, `writer_send.cpp` | Portable | Class 1 producer/consumer FIFO; free-function API → `DataflowBuffer` | Portable | — |
| `cb_id_out` (receiver → receive writer) | 1 | `reader_receive.cpp`, `writer_unary_interleaved_start_id_gen.cpp` | Portable | Class 1 producer (`receiver_cb_id`) → consumer FIFO | Portable | — |
| `packet_header_cb_id` | 6 | `writer_send.cpp`, `reader_receive.cpp` | Portable | single-kernel fabric packet-header staging: `reserve_back(1)`/`get_read_ptr`|`get_write_ptr`/`push_back(1)`, no cross-kernel consumer → **ScratchpadSpec** | Portable | same |
| `packet_cb_id` | 6 | `writer_send.cpp`, `reader_receive.cpp` | Portable | single-kernel coalesced-packet working buffer → **ScratchpadSpec** | Portable | same |
| `receiver_cb_id` (receive) | 1 | `reader_receive.cpp` | Portable | Class 1 producer into the receive writer FIFO | Portable | — |

### GATE hits: (none)
### Blocked on runtime: (none)

---

## `debug`

**Op root:** `ttnn/cpp/ttnn/operations/debug/`
**Scope:** `ApplyDeviceDelayDeviceOperation` (single-descriptor) → kernel: `device/kernels/dataflow/device_delay_spin.cpp`

### Overall verdict: GREEN (N/A — no CB)

The kernel only reads the wall-clock registers and busy-spins for `delay_cycles` (`tt::data_movement::common::spin`). **No CB / DFB is declared or used** in the kernel, and the factory allocates none. Nothing to port. **N/A — no CB.**

### CB portability: N/A — no CB
### GATE hits: (none)
### Blocked on runtime: (none)

---

## GATE hits (group)

| File:line | Field | Op | Fix |
|-----------|-------|----|-----|
| `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp:20` | `get_local_cb_interface(cb_id_in0).fifo_page_size` | example (donor) | `get_entry_size()` |
| `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:19` | `get_local_cb_interface(cb_id_out).fifo_page_size` | example (donor) | `get_entry_size()` |

## Blocked on runtime (2xx rollup)

- **prefetcher** `remote_cb_id` (`c_31`, global CB) — no DFB equivalent; needs a Quasar remote/global-CB primitive. Structural (not a listed in-flight runtime fix).
- **halo** `out_cb` (Class 3 scatter) and `src_cb` (split-reader shared shard) — 2xx design decisions (strided DFB / multi-DFB / scratchpad+sems), not runtime-API waits.
- LTA prereqs (YELLOW, both arches): halo `padding_config*`/`gather_config*`; prefetcher `addrs_cb_id`.

## Notes & follow-ups

- **Cross-op donor kernels (in scope, note donor origin):**
  - `example` → `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`, `.../writer_unary_interleaved_start_id.cpp`, `.../compute/eltwise_sfpu.cpp` (eltwise/unary donors — carry the GATE).
  - `halo` → `ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp` (`experimental::CB`), `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`.
  - `prefetcher` → `tt_metal/hw/inc/api/remote_circular_buffer.h` (remote-CB primitive), `ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp`.
  - `point_to_point` → `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp`, `ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp`, fabric headers.
- **Migration state varies:** `full`/halo use the object `CircularBuffer` API; `example` compute + eltwise donors use `DataflowBuffer`; `point_to_point`/`prefetcher` still use the **legacy free-function `cb_*(id)` API** (mechanical rename, not a blocker).
- **example GATE is trivial:** `get_entry_size()` exists today; once the two donor kernels swap `fifo_page_size` → `get_entry_size()`, example is Class 1 → GREEN. Because these donors are shared repo-wide, the fix benefits many ops.
- **full — host binding note (OUT-OF-SCOPE):** the interleaved factory binds the same `writer_full.cpp` source to two CBs (`c_0`/`c_1`) as two independent DM self-loops; binding multiplicity / endpoint legality is a host-audit concern, not gated here.
- **Kernel-only audit:** host `ProgramSpec` / `DataflowBufferSpec` feasibility, SPSC/endpoint legality, and remote/global-CB host modeling are tracked by the host audit, not here.
