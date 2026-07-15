# CB→DFB Kernel Audit: `normalization` Group (consolidated)

**Date:** 2026-07-15
**Group:** normalization ops from the do-list — **3 ops / 11 factory-variant rows / 51 kernel files** (incl. 5 moreh_softmax cross-op donors + 1 shared Welford `memory.h` donor)
**Audit spec:** `plan-quasar-dfb-audit/cb_dfb_kernel_audit_helper.md`

## Group verdict: RED

**Bottom line:** The group is dragged to **RED by a single groupnorm buffer**: `cb_ex_external` is zero-filled via `groupnorm_zero_fill.hpp::zero_whole_cb`, which reads `get_local_cb_interface(cb_id).fifo_size` — this is **both** a hard **GATE** (`get_local_cb_interface(...).<field>`) **and** the no-getter `fifo_size` case the spec flags for **file-to-Almeet** (needs `get_total_buffer_size_bytes()` / ring-span getter). Blocked on **both** arches until the getter lands or the zero-fill is redesigned. Separately, groupnorm's Welford reciprocal LUT (`cb_reciprocals`) is **Portable (prereq: LTA)** (YELLOW) — sync-free borrowed read via `get_pointer_to_cb_data` (implemented over `get_tile_address` in the shared `memory.h`). **batch_norm** and **softmax** are clean Class-1 FIFO throughout → **GREEN** (softmax general kernels already on `DataflowBuffer` via the moreh port; batch_norm + softmax-attention + groupnorm still on legacy `CircularBuffer`, a mechanical rename that is not itself a blocker).

## Group-wide classification scan (all 62 SCAN_FILES: 51 kernels + 11 shared headers)

| Pattern (classification) | Hits |
|--------------------------|------|
| `get_local_cb_interface(...).<field>` (GATE) | **1** — `groupnorm_zero_fill.hpp:38,40` (`.fifo_size`), live-called by `reader_mcast_sender_unary_gn.cpp:298` |
| `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr` (silent-wrong) | **none** |
| `read_tile_value` / `get_tile_address` (QUASAR-BLOCKED) | **1 def** — `memory.h:31` `get_tile_address` (reached only via `get_pointer_to_cb_data`; LTA is the sanctioned end-state, see below) |
| `get_pointer_to_cb_data` (LTA prereq) | **1 call** — `welford_groupnorm.cpp:247` (`cb_reciprocals`, via `memory.h:30`) |
| `fifo_size` / `fifo_limit` (no getter → file-to-Almeet) | **1** — `groupnorm_zero_fill.hpp:40` `iface.fifo_size` (same hit as GATE) |
| `fifo_page_size` / `fifo_num_pages` (mechanical getter swap) | **none** |
| `fifo_wr_ptr` / `fifo_rd_ptr` / `push_back_hold` / `llk_push_pages` (ptr surgery) | **none** |
| `packer_l1_acc` / `matmul_partials` / `*_partials_cb` (Class 5 accumulator) | **none** |

Unreferenced kernel `.cpp` under the three op trees: **none** (every kernel file is referenced by a factory).

## Per-op / per-factory rollup

| Op | Factory (do-list) | Kernels | Kernel state | Notable CB classes | Verdict |
|----|-------------------|---------|--------------|--------------------|---------|
| `batch_norm` | BatchNormFactory | `reader_batch_norm`, `writer_batch_norm`, `batch_norm_{kernel,sfpu_kernel}` | CircularBuffer | all 1 | GREEN |
| `batch_norm` | (RunningStatistics — adjacent, same tree) | `reader/writer_running_statistics`, `running_statistics_{kernel,sfpu_kernel}` | CircularBuffer | all 1 | GREEN |
| `softmax` | AttentionOptimized | `softmax.cpp`, `softmax_large_tensor.cpp`, `reader_unary_interleaved_sm(_large_tensor)`, `writer_unary_interleaved_start_id_blocked_sm` | CircularBuffer | all 1 | GREEN |
| `softmax` | ShardedAttentionOptimized | `softmax_sharded.cpp`, `reader_unary_sharded_sm{,_causal_mask_hw_dims,_rm_mask}` | CircularBuffer | all 1 | GREEN |
| `softmax` | GeneralCLarge / HLarge / HSmall / WLarge / WSmall | moreh_softmax donors: `moreh_softmax_{c_large,h_large,h,w_large,w}` + `reader_`/`writer_` | **DataflowBuffer** (already ported) | all 1 | GREEN |
| `groupnorm` | GroupNormMcastProgramFactory | `groupnorm.cpp` / `welford_groupnorm.cpp`; `reader_mcast_{sender,receiver}_unary_gn` (+welford); `writer_unary_gn_rm_gb` (+welford) | CircularBuffer | 1, **6 (LUT/LTA)**, **GATE** | **RED** |
| `groupnorm` | GroupNormNoMcastProgramFactory | `groupnorm.cpp` / `welford_groupnorm.cpp`; `reader_mcast_sender_unary_gn` (+welford); `writer_unary_gn_rm_gb` (+welford) | 1, **6 (LUT/LTA)**, **GATE** | **RED** |
| `groupnorm` | GroupNormShardedProgramFactory | `groupnorm_sharded_v2.cpp` / `welford_groupnorm_sharded_v2.cpp`; `reader_mcast_{sender,receiver}_unary_sharded_gn_v2` (+welford); `writer_unary_sharded_gn_rm_gb_v2` (+welford) | 1 (no LUT, no live zero-fill) | GREEN* |

*Sharded factory is clean on its own kernels (sharded Welford uses `empty_reciprocal_lut`, no `get_pointer_to_cb_data`; the `zero_whole_cb` reference in `groupnorm_sharded_v2.cpp:276` is a **comment**, not a call). But `groupnorm` rolls up **RED** as an op because the mcast/no-mcast factories share the GATE'd zero-fill and the LTA-prereq LUT.

## CB portability — `batch_norm` (BatchNormFactory + running_statistics)

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_input`, `cb_batch_mean`, `cb_batch_var`, `cb_eps`, `cb_weight`, `cb_bias` (inputs) | 1 | `reader_batch_norm.cpp`, `batch_norm_{kernel,sfpu_kernel}.cpp` | Portable | linear FIFO; `get_write_ptr()`/`get_tile_size()` as L1 cursor; mechanical `CircularBuffer`→`DataflowBuffer` | Portable | — |
| `cb_den`, `cb_tmp_1` (intermediates) | 1 | `batch_norm_{kernel,sfpu_kernel}.cpp` | Portable | scratch FIFO in normalize pipeline | Portable | — |
| `cb_output_0` (output) | 1 | `writer_batch_norm.cpp`, `batch_norm_{kernel,sfpu_kernel}.cpp` | Portable | pack → output FIFO | Portable | — |
| running-stats CBs (`cb_*` in `running_statistics_{kernel,sfpu_kernel}`, `reader/writer_running_statistics`) | 1 | running_statistics kernels | Portable | linear FIFO throughout; identical pattern | Portable | — |

## CB portability — `softmax` (all 7 factories)

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` (c_0) | 1 | all compute + readers | Portable | input FIFO; mechanical rename (attention) / already DFB (general) | Portable | — |
| `cb_max_scaler`, `cb_sum_scaler`, `cb_fused_scale`, `cb_fused_attn`, `cb_scale_mask`, `cb_mask(_padded)` | 1 | `softmax.cpp`, `softmax_sharded.cpp`, moreh donors, readers | Portable | scalar/mask input FIFOs | Portable | — |
| `cb_exps` (c_6/c_24) | 1 | compute kernels | Portable | exp intermediate FIFO | Portable | — |
| `cb_max`, `cb_x` (c_8/c_9/c_10) | 1 | compute kernels | Portable | running-max / shifted-input FIFO | Portable | — |
| `cb_recipsumexps` (c_7/c_25) | 1 | compute kernels | Portable | **in-kernel** reduce + `recip_tile` (not a borrowed LUT); canonical FIFO | Portable | — |
| `cb_x_m_max`, `cb_tmp` (moreh donors) | 1 | `moreh_softmax_*` | Portable | intermediates; already DataflowBuffer | Portable | — |
| `cb_out0` (c_11/c_16) | 1 | writers + compute | Portable | pack → output FIFO | Portable | — |

## CB portability — `groupnorm` (3 factories, welford + non-welford)

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` (c_0), `cb_in` (c_29) | 1 | `groupnorm.cpp`, `groupnorm_sharded_v2.cpp`, welford variants | Portable | input FIFO; mechanical `CircularBuffer`→`DataflowBuffer` | Portable | — |
| `cb_scaler`, `cb_scaler_global`, `cb_eps`, `cb_input_mask` | 1 | compute + readers | Portable | scalar/mask input FIFOs | Portable | — |
| `cb_x` (c_24), `cb_xmm` (c_25), `cb_ex2pe` (c_27) | 1 | compute kernels | Portable | normalize-pipeline intermediates | Portable | — |
| `cb_ex_partial` (c_8), `cb_ex2_partial` (c_21), `cb_ex` (c_9), `cb_ex2` (c_13) | 1 | compute + mcast readers | Portable | partial mean/var reduce FIFOs | Portable | — |
| `cb_ex_global` (c_15), `cb_ex2_global` (c_14) | 1 | compute kernels | Portable | reduced global mean/var FIFOs | Portable | — |
| welford-stats CBs feeding `combine_welford_stats` (`welford_combine.h`) | 1 | `welford_reader_mcast_{sender,receiver}_unary_gn(_sharded_v2)` | Portable | `get_read_ptr()` as L1 source for stats combine; sync-free but pointer-only on FIFO'd partials | Portable | — |
| `cb_gamma` (c_5), `cb_beta` (c_6) | 1 | `writer_unary_gn_rm_gb.cpp`, `welford_writer_unary_gn_rm_gb.cpp`, `writer_unary_sharded_gn_rm_gb_v2.cpp` (+welford) | Portable (workaround) | **undesirable but OK hack:** `get_write_ptr() + gamma/beta_tile_bytes` offset + `+512`/`+32` for masked NOC weight writes (Class 1-ish); uplift: keep byte arithmetic on DFB `get_write_ptr()` | Portable (workaround) | same |
| `cb_repack`, `cb_repack_out`, `cb_reread_out`, `cb_reread_write_out`, `cb_out0`/`cb_out` | 1 | compute + writers | Portable | repack/output FIFOs | Portable | — |
| **`cb_reciprocals`** (c_18) | 6 | `welford_groupnorm.cpp:247` (mcast + no-mcast Welford compute) | **Portable (prereq: LTA)** | sync-free borrowed reciprocal LUT via `get_pointer_to_cb_data<recip_lut_t>` → **LocalTensorAccessor** (replaces `memory.h` ptr read) | **Portable (prereq: LTA)** | same; LTA removes the `get_tile_address` dependency entirely (LLK does not need a DFB id for this pointer-only read) |
| **`cb_ex_external`** (c_10) | 6 | `reader_mcast_sender_unary_gn.cpp:298` → `groupnorm_zero_fill.hpp::zero_whole_cb` (mcast + no-mcast **non-welford**) | **Blocked** | **GATE:** `get_local_cb_interface(cb).fifo_size` in `groupnorm_zero_fill.hpp:38,40`. Also no-getter `fifo_size` → **file-to-Almeet** (`get_total_buffer_size_bytes()`/ring-span getter). Interim 1xx: buffer size is knowable at build → pass it as CTA instead of reading the interface | **Blocked** | same — hard stop until getter lands or zero-fill is redesigned (e.g. compile-time size arg, or `ScratchpadSpec`-backed staging) |

## GATE hits (must be empty to merge)

- `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/groupnorm_zero_fill.hpp:38` — `auto& iface = get_local_cb_interface(cb_id);`
- `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/groupnorm_zero_fill.hpp:40` — `noc.async_write_zeros(cb, iface.fifo_size);` (`.fifo_size` field read)
  - **Live caller:** `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_sender_unary_gn.cpp:298` — `zero_whole_cb(cb_ex_external_id, noc);` (GroupNormMcast + GroupNormNoMcast, non-welford path)
  - Header is also `#include`d by `compute/groupnorm_sharded_v2.cpp`, but the `zero_whole_cb` reference there (`:276`) is a **comment only** — no live call in the sharded path.
  - **Resolution:** dual-flagged — GATE **and** no-getter `fifo_size` → **file issue to Almeet** for `get_total_buffer_size_bytes()` / ring-span getter; port stays RED until getter lands or the whole-CB zero-fill is redesigned (compile-time size CTA or `ScratchpadSpec`).

## Blocked on runtime (2xx rollup)

- **`cb_reciprocals`** (groupnorm Welford, `welford_groupnorm.cpp`): the current `memory.h::get_pointer_to_cb_data` is implemented over `get_tile_address` (QUASAR-BLOCKED). The **sanctioned end-state is LocalTensorAccessor** (sync-free borrowed pointer-only LUT read), which removes the `get_tile_address` dependency — so this is tracked as **Portable (prereq: LTA)** (YELLOW), not a 2xx runtime block. If LTA is *not* taken and the code is ported as-is, 2xx would be **Blocked (runtime)** until the DFB `get_tile_address`/`read_tile_value` API lands.
- No other `read_tile_value` / `get_tile_address` compute paths in the group (softmax's `1/sum` uses in-kernel `recip_tile`, not tile-address reads).

## Notes & follow-ups

- **Two distinct groupnorm port tasks (the only blockers in the whole group):**
  1. **`cb_ex_external` zero-fill (RED / GATE):** `groupnorm_zero_fill.hpp` reads `get_local_cb_interface(cb).fifo_size`. File-to-Almeet for a byte-size getter, or redesign the whole-CB zero-fill (the fill span is build-time knowable → pass as a compile-time arg; the buffer is genuinely private-L1 pre-first-`push_back`, so a `ScratchpadSpec` also fits). Affects GroupNormMcast + GroupNormNoMcast (non-welford).
  2. **`cb_reciprocals` LUT (YELLOW / LTA):** migrate `get_pointer_to_cb_data` → `LocalTensorAccessor` (host `TensorBinding` + kernel ctor). Same fix as the layernorm sharded-Welford example; shared donor header `memory.h` (`ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/memory.h`) is used only by `welford_groupnorm.cpp` in this group. Affects GroupNormMcast + GroupNormNoMcast (welford).
- **Legacy `CircularBuffer` vs `DataflowBuffer`:** softmax **general** kernels (moreh_softmax donors) are already on `DataflowBuffer`; batch_norm, softmax **attention/sharded**, and groupnorm kernels still declare `CircularBuffer`. That mechanical Class-1 rename is required to port but is **not** an audit blocker (no `get_local_cb_interface` field access was introduced anywhere except the zero-fill GATE above).
- **Cross-op donor kernels (in scope, flagged):**
  - softmax General C/H/W (large/small) factories build kernel paths against `SOFTMAX_KERNEL_PATH_GENERAL = "ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels"` — **15 moreh_softmax donor kernels** (`moreh_softmax_{c_large,h_large,h,w_large,w}.cpp` + matching `reader_`/`writer_`). All Class-1, already `DataflowBuffer` (consistent with the moreh group audit, GREEN).
  - Welford reciprocal LUT donor header: `ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/memory.h` (`get_pointer_to_cb_data` → `get_tile_address`). Same header feeds the layernorm / layernorm_distributed Welford kernels (out of this group's scope but the same LTA fix applies).
  - groupnorm-local shared helper `welford_combine.h` (`combine_welford_stats`) — Class-1 pointer math over FIFO'd partial stats; clean.
- **Scope handling:** BatchNormFactory is the named do-list factory; `running_statistics` is a separate `DeviceOperation` in the same tree and is included here only as an adjacent verification (equally clean Class-1). No OUT-OF-SCOPE (moe-gate / DeepSeek) paths touched.
- **Host-side factories NOT audited:** kernel-only audit. Host `ProgramSpec` / `DataflowBufferSpec` legality and the CB→DFB host-spec migration are separate work.
