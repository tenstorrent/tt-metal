# CB→DFB Kernel Audit: `moreh` Group (consolidated)

**Date:** 2026-07-15
**Group:** Moreh ops from `gchoudhary_likely_to_do_list.csv` — **28 distinct ops / 49 factory-variant rows / 138 kernel `.cpp` files**
**Audit spec:** `plan-quasar-dfb-audit/cb_dfb_kernel_audit_helper.md`

## Group verdict: GREEN

**Bottom line:** The moreh group is effectively **already ported**. PR **#49430** ("[Cleanup] Migrate Moreh Kernels from CircularBuffer to DataflowBuffer", merged — commit `af9c372a48c`, ancestor of current HEAD) performed the mechanical Class-1 `CircularBuffer`→`DataflowBuffer` conversion across the moreh tree: **136 / 138** kernel files now declare `DataflowBuffer`. The audit is therefore verification-by-inheritance plus classification of the one deferred exception. Every group-wide blocker scan returns **zero hits**, and the sole non-canonical pattern is `moreh_arange`'s two writers (deferred by #49430, tracked in issue **#49368**), which classify as Class 6 → `ScratchpadSpec` (autoportable, GREEN). No GATE, no silent-wrong, no Quasar runtime dependency, no LTA prerequisite anywhere in the group.

## Group-wide classification scan (all moreh kernels)

Run over `ttnn/cpp/ttnn/operations/moreh/**/kernels/**`:

| Pattern (classification) | Hits |
|--------------------------|------|
| `get_local_cb_interface` (GATE) | **none** |
| `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr` (silent-wrong) | **none** |
| `read_tile_value` / `get_tile_address` (quasar-blocked) | **none** |
| `get_pointer_to_cb_data` (LTA prereq) | **none** |
| `fifo_wr_ptr` / `fifo_rd_ptr` / `fifo_page_size` / `fifo_num_pages` / `push_back_hold` / `llk_push_pages` (ptr surgery / field reads) | **none** |
| `packer_l1_acc` / `matmul_partials` / `*_partials_cb` (Class 5 in-place accumulator) | **none** |
| `AddrSelector::WRITE_PTR` / residual `CircularBuffer` (fake-CB scratchpad) | **2 files — `moreh_arange` writers only** |

## Per-op rollup (28 ops)

All ops are fully `DataflowBuffer`-migrated (via #49430) with canonical FIFO CBs → Class 1 → **Portable / GREEN**, except `moreh_arange` (see deep-dive). "Factories" = factory-variant rows on the do-list.

| Op | Factories | Kernel state | CB classes | Verdict |
|----|-----------|--------------|------------|---------|
| `moreh_abs_pow` | 1 | DFB | 1 | GREEN |
| `moreh_arange` | 1 | **2 files residual CircularBuffer** | **6** (fake-CB scratch) | GREEN (workaround — see deep-dive) |
| `moreh_clip_grad_norm/step1` | 1 | DFB | 1 | GREEN |
| `moreh_clip_grad_norm/step2` | 1 | DFB | 1 | GREEN |
| `moreh_clip_grad_norm/step3` | 1 | DFB | 1 | GREEN |
| `moreh_dot` | 1 | DFB | 1 | GREEN |
| `moreh_dot_backward` | 1 | DFB | 1 | GREEN |
| `moreh_fold` | 1 | DFB | 1 | GREEN |
| `moreh_getitem` | 2 | DFB | 1 | GREEN |
| `moreh_group_norm` | 1 | DFB | 1 | GREEN |
| `moreh_group_norm_backward` | 2 | DFB | 1 | GREEN |
| `moreh_layer_norm` | 1 | DFB | 1 | GREEN |
| `moreh_layer_norm_backward` | 2 | DFB | 1 | GREEN |
| `moreh_linear_backward` | 2 | DFB | 1 | GREEN |
| `moreh_matmul` | 1 | DFB | 1 | GREEN |
| `moreh_mean` | 3 | DFB | 1 | GREEN |
| `moreh_mean_backward` | 1 | DFB | 1 | GREEN |
| `moreh_nll_loss/step1` | 1 | DFB | 1 | GREEN |
| `moreh_nll_loss/step2` | 1 | DFB | 1 | GREEN |
| `moreh_nll_loss_backward` | 1 | DFB | 1 | GREEN |
| `moreh_nll_loss_unreduced_backward` | 1 | DFB | 1 | GREEN |
| `moreh_norm` | 3 | DFB | 1 | GREEN |
| `moreh_norm_backward` | 1 | DFB | 1 | GREEN |
| `moreh_sgd` | 1 | DFB | 1 | GREEN |
| `moreh_softmax` | 5 | DFB | 1 | GREEN |
| `moreh_softmax_backward` | 5 | DFB | 1 | GREEN |
| `moreh_sum` | 6 | DFB | 1 | GREEN |
| `moreh_sum_backward` | 1 | DFB | 1 | GREEN |

## Deep-dive: `moreh_arange` (the one exception)

**Op root:** `ttnn/cpp/ttnn/operations/moreh/moreh_arange/`
**Kernels (writers only; no reader, no compute):** `device/kernels/writer_moreh_arange.cpp`, `device/kernels/writer_moreh_arange_rm.cpp`

Both writers keep the legacy `CircularBuffer` and use it as a **fake CB / private-L1 staging buffer**, not a FIFO:

```cpp
constexpr uint32_t cb_out = tt::CBIndex::c_16;
CircularBuffer cb_out_obj(cb_out);
...
cb_out_obj.reserve_back(1);                 // reserve, but NO push_back / no consumer
uint32_t w_addr = cb_out_obj.get_write_ptr();
... compute arange values into w_addr ...
noc.async_write(use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_out_obj), s0, num_bytes_per_tile);
```

`reserve_back` with no matching `push_back`, no consuming kernel, and NOC-write straight to the output tensor → **Class 6 (structural non-FIFO)** per the spec. This is exactly the "fake-CB WRITE_PTR scratchpad" pattern #49430 intentionally **deferred** (tracked in issue **#49368**), which is why these two files were left on `CircularBuffer` while the other 136 moved to `DataflowBuffer`.

### CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_out` (`c_16`) | 6 | `writer_moreh_arange.cpp`, `writer_moreh_arange_rm.cpp` | Portable (workaround) | **undesirable but OK hack:** fake CB — `reserve_back` without `push_back`, no consumer; `get_write_ptr()` + `use<CircularBuffer::AddrSelector::WRITE_PTR>` as private-L1 NOC staging. Uplift: **ScratchpadSpec**. Deferred by #49430 (issue #49368). | Portable | autoportable: private-L1 → **ScratchpadSpec** (no FIFO credits, no semaphore needed — single-kernel staging) |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Notes & follow-ups

- **Prior art / this audit's relationship to it:** the moreh mechanical port is done by **#49430** (moreh) — the sibling PR **#49392** does the same for data-movement kernels (~176 files). Both defer raw write-ptr scratchpads and remote CBs to **issue #49368**. This audit confirms nothing beyond that residual needs design work in the moreh group.
- **Only remaining moreh port task:** convert `moreh_arange`'s two writers from the fake-CB `WRITE_PTR` scratchpad to `ScratchpadSpec` (per #49368). Everything else is complete and clean.
- **Verification caveat:** this audit trusts #49430's mechanical `CircularBuffer`→`DataflowBuffer` rename as semantics-preserving (DFB wraps `LocalCBInterface` on WH/BH, `LocalDFBInterface` on Quasar). The blocker scans confirm the rename introduced no `get_local_cb_interface` field access, ptr surgery, or Quasar-blocked calls.
- **Host-side factories are NOT migrated.** #49430 converted kernels only. **26 moreh host program-factory `.cpp` files still construct `CircularBuffer`** (e.g. `moreh_matmul`, `moreh_norm`, `moreh_sum`, `moreh_adam(w)`, `moreh_group_norm(_backward)`, `moreh_layer_norm(_backward)`, `moreh_sgd`, `moreh_linear_backward`, plus `moreh_helper_functions.{hpp,cpp}`). Kernel-side "GREEN" does **not** mean the ops are end-to-end Metal 2.0 ported — the host CB→DFB spec migration is separate work.
- **Host-side feasibility not covered here.** Kernel-only audit; SPSC/endpoint legality and `DataflowBufferSpec` fit are tracked by the host audit (`port_op_to_metal2_audit.md`).
