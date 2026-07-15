# CB→DFB Blocked / RED Summary (cross-family rollup)

**Date:** 2026-07-15
**Source:** the family `CB_DFB_KERNEL_AUDIT.md` files under `ttnn/cpp/ttnn/operations/**`, now including a **full per-op sweep of `experimental/`** (124 op docs + master rollup at `experimental/CB_DFB_KERNEL_AUDIT.md`: 4 RED / 19 YELLOW / 96 GREEN / 5 OUT-OF-SCOPE).
**Purpose:** consolidate every **RED** op, its root cause, and the **shared patterns** that repeat across families so they can be fixed once rather than op-by-op.
**Recent landings:** `eltwise` kernels are already CB→DFB-migrated ([PR #49410](https://github.com/tenstorrent/tt-metal/pull/49410), 50 kernel files) — the audit finds it **YELLOW** (4 mechanical `fifo_page_size` swaps left, no new RED).
**Experimental sweep (2026-07-15):** all 124 `experimental/` sub-ops now have their own `CB_DFB_KERNEL_AUDIT.md`. It added **2 new RED ops** (`experimental/quasar/matmul`, `experimental/ccl/llama_all_gather_matmul_async`) — both the **same gathered-matmul kernel** already tracked under Patterns A/B — and **no new problem class**. Rollup: 4 RED / 19 YELLOW / 96 GREEN / 5 OUT-OF-SCOPE.

---

## 1. Rollup of every audit

| Audit (family) | Verdict | RED driver(s) |
|----------------|---------|---------------|
| `data_movement` | 🟡 YELLOW | none — 1 mechanical `fifo_page_size` swap |
| `eltwise` | 🟡 YELLOW | none — kernels already migrated to DFB ([PR #49410](https://github.com/tenstorrent/tt-metal/pull/49410)); 4 mechanical `fifo_page_size` swaps remain |
| `reduction` | 🟡 YELLOW | none — 1 mechanical swap + `manual_seed` 2xx `read_tile_value` |
| `matmul` | 🔴 **RED** | ring-all-gather **gathered** variant — `fifo_rd_ptr` writes + no-getter `fifo_size`/`fifo_limit` reads |
| `normalization` | 🔴 **RED** | `groupnorm_zero_fill.hpp` — no-getter `fifo_size` read |
| `transformer` | 🔴 **RED** | `sdpa` family — `compute_streaming.hpp` `fifo_wr_ptr` write + no-getter ring-span reads |
| `experimental/quasar/conv2d` | 🔴 **RED** | 42 `fifo_wr_ptr`/`fifo_rd_ptr` **writes** (Class 2+3+4+5) |
| `experimental/quasar/matmul` | 🔴 **RED** | gathered variant `cb_in1` ring window — `fifo_rd_ptr` writes + no-getter `fifo_size`/`fifo_limit` reads |
| `experimental/ccl/ring_attention_all_gather_async` | 🔴 **RED** | reader manual ring wrap — no-getter `fifo_limit`/`fifo_size` reads |
| `experimental/ccl/llama_all_gather_matmul_async` | 🔴 **RED** | reuses gathered-matmul ring window (`fifo_rd_ptr` writes + no-getter reads) |
| `experimental/` (full per-op rollup, 124 ops) | 🔴 **RED** | 4 RED / 19 YELLOW / 96 GREEN / 5 OUT-OF-SCOPE — aggregates the four experimental REDs above |

**Seven distinct RED ops** (the experimental rollup double-counts four):
1. `matmul` (ring-all-gather gathered variant only)
2. `normalization/groupnorm`
3. `transformer/sdpa` (+ all sparse/joint/ring variants)
4. `experimental/quasar/conv2d`
5. `experimental/quasar/matmul` (gathered variant)
6. `experimental/ccl/ring_attention_all_gather_async`
7. `experimental/ccl/llama_all_gather_matmul_async` (gathered variant)

---

## 2. The RED patterns (root causes)

Every RED reduces to one or both of two hard blockers on `get_local_cb_interface(...)`. A third pattern (`read_tile_value`) only blocks 2xx and is never the sole RED driver.

### Pattern A — `fifo_wr_ptr` / `fifo_rd_ptr` **WRITES** (pointer surgery)
Kernel hand-manipulates the CB's internal FIFO pointer to build a manual ring/window/held-base. This is **credit/address decoupling** (Class 2/3/4/5) and requires a **redesign**, not a getter swap. It is the deepest blocker.

| Op | Location(s) | Class |
|----|-------------|-------|
| `quasar/conv2d` | `conv_reader_common.hpp:91,108`; `reader_conv_activations_*:89,116`; `reader_writer_tiled_out_1d_*` (metal2 `:180,137`, legacy `:155,113`); `conv_bmm_tilize.cpp:71,77,111,163,369,519-632` (+ `_metal2:104,110,140,184,413,558-669`) — **42 total** | 2+3+4+5 |
| `matmul` (gathered) | `bmm_large_block_zm_fused_bias_activation_gathered.cpp:51,88,120,125,130,133` — `fifo_rd_ptr` ring wrap | 2+4 |
| `experimental/quasar/matmul` (gathered) | same `bmm_*_gathered.cpp` `cb_in1` ring window (referenced by `matmul_multicore_reuse_mcast_1d_program_factory.cpp`) | 2+4 |
| `experimental/ccl/llama_all_gather_matmul_async` | reuses the gathered-matmul ring window (`fifo_rd_ptr` writes + no-getter `fifo_size`/`fifo_limit` reads) | 2+4 |
| `transformer/sdpa` | `compute_streaming.hpp:85,88` — `cb_push_back_hold_wr_ptr` rewinds `fifo_wr_ptr` | 4 |
| `ccl/ring_attention` | `ring_attention_all_gather_reader.cpp:70-71` — hand-wraps `l1_write_addr` against snapshotted limit/size (address-side variant of the same idea) | 4 |

### Pattern B — `fifo_size` / `fifo_limit` **READS** with **NO existing getter**
Kernel reads the ring span to compute wrap math. There is **no DFB getter today** (`get_total_buffer_size_bytes()` / ring-span getters). **File one issue to Almeet** — it unblocks all four ops below at once.

| Op | Location(s) |
|----|-------------|
| `ccl/ring_attention` | `ring_attention_all_gather_reader.cpp:148,149` |
| `ccl/llama_all_gather_matmul_async` | gathered-matmul `cb_in1` ring span (same reads as matmul gathered) |
| `transformer/sdpa` | `compute_streaming.hpp:86` |
| `matmul` (gathered) | `bmm_large_block_zm_fused_bias_activation_gathered.cpp:56-57,65,121-122,127` |
| `experimental/quasar/matmul` (gathered) | same `bmm_*_gathered.cpp` `fifo_size`/`fifo_limit` reads |
| `normalization/groupnorm` | `groupnorm_zero_fill.hpp:38,40` — `fifo_size` read (**not** ptr surgery; a plain no-getter read) |

### Pattern C — `read_tile_value` / `get_tile_address` on Quasar DFB (2xx only)
Sanctioned DFB read API that is **in progress** (Runtime team). **1xx is clear**; only **2xx** is blocked. Never the sole RED driver, but rides along on several RED/YELLOW ops.

- `transformer/sdpa` ctrl CBs (`cb_ctrl`, `cb_chunk_start_idx`), `sdpa_decode` `cb_cur_pos`
- `experimental/ccl/moe_compute`, `ccl/moe_gpt`; `deepseek_prefill/unified_routed_expert_ffn` (2xx path)
- `reduction/manual_seed` (YELLOW)
- Welford `cb_reciprocals` LUT via `get_pointer_to_cb_data`/`get_tile_address` in shared `kernel_util/compute/memory.h` — layernorm, layernorm_distributed, groupnorm (+ `transformer/dit_layernorm_pre_all_gather`). On 1xx these port via **LocalTensorAccessor (LTA)**.

---

## 3. Duplicate / similar cases across families

These are the "fix once, unblock many" clusters. The most important finding is that **six of the seven RED ops are the same bug** (the seventh, `groupnorm`, is the read half of it).

### Cluster 1 — Manual ring/window buffer (Patterns A + B together) 🔴
`conv2d`, `matmul` (gathered), `sdpa`, `ring_attention`, plus the experimental gathered-matmul twins `experimental/quasar/matmul` and `experimental/ccl/llama_all_gather_matmul_async` are **the same anti-pattern**: a CB used as a hand-indexed ring/window via `fifo_wr_ptr`/`fifo_rd_ptr` writes, with wrap math driven by no-getter `fifo_size`/`fifo_limit` reads. The two experimental twins are **literally the same kernel** as the non-experimental `matmul` gathered variant (`bmm_large_block_zm_fused_bias_activation_gathered.cpp`), so one redesign covers all three matmul-gathered instances. They share **one recommended fix**:
> **scratchpad + semaphores** (`ScratchpadSpec` + `SemaphoreSpec`) for the window/ring, replacing manual pointer/credit surgery. On Quasar, alternative is **strided / multi-producer DFB** (`stride_in_entries`) or disabling the split reader. If retained as a v1 ptr fallback: **WEIRD-OK**, disable implicit sync on Quasar (`Gen2Config::disable_implicit_sync_for`).

### Cluster 2 — No-getter `fifo_size`/`fifo_limit` read 🔴
Same missing runtime API blocks all six ring/window ops (`ring_attention`, `sdpa`, `matmul`-gathered, `experimental/quasar/matmul`, `experimental/ccl/llama_all_gather_matmul_async`) **plus** `groupnorm`. **One Almeet issue** (`get_total_buffer_size_bytes()` / ring-span getters) is a prerequisite for every one of them. `groupnorm` is the cheapest: its read can also be removed entirely by passing the backing byte size as a compile-time/runtime arg, or migrating `cb_ex_external` to `ScratchpadSpec`.

### Cluster 3 — Welford `cb_reciprocals` LUT (Pattern C, LTA) 🟡/2xx
`layernorm`, `layernorm_distributed`, `groupnorm`, and `transformer/dit_layernorm_pre_all_gather` all read the reciprocal LUT through the **single shared** `kernel_util/compute/memory.h`. **One `memory.h` fix** ports all of them: LTA on 1xx, and Quasar `get_tile_address` (in progress) on 2xx.

### Cluster 4 — `read_tile_value` mailbox/ctrl scalar (Pattern C) 🟡/2xx
Repeats across `sdpa`/`sdpa_decode` ctrl CBs, `ccl/moe_compute`, `ccl/moe_gpt`, `deepseek_prefill/unified_routed_expert_ffn`, and `reduction/manual_seed`. All wait on the **same** Quasar DFB `read_tile_value` API; all are 1xx-clear. The experimental full sweep confirmed the three experimental members (`moe_compute`, `moe_gpt`, `unified_routed_expert_ffn`) and found **no new** `read_tile_value`/`get_tile_address` sites outside the OUT-OF-SCOPE routing ops.

### Cluster 5 — Mechanical getter swaps (NOT red, noted for contrast) 🟢
`fifo_page_size → get_entry_size()` and `fifo_num_pages → get_total_num_entries()` (PR #49197) are the dominant "YELLOW" across the tree — trivial one-line swaps that clear their GATE immediately. They are **not** RED and should not be conflated with Patterns A/B. They recur across:
- **Core families:** `data_movement`, `eltwise`, `reduction/generic`, `matmul` (standard).
- **Experimental (16 of the 19 experimental YELLOWs):** 8 `quasar/*` ops (`interleaved_to_sharded`, `pool_generic`, `reduction/generic`, `tilize`, `tilize_with_val_padding`, `typecast`, `untilize`, `untilize_with_unpadding`), `deepseek_prefill/update_padded_kv_cache`, `padded_slice`, and `matmul/attn_matmul` + `unary_backward/gelu_backward` (these two share **one** donor writer `writer_unary_interleaved_start_id.cpp:19` — fixing that line clears both), plus the 3 ccl composites (`all_gather_matmul_async`, `llama_reduce_scatter_matmul`, `matmul_reduce_scatter_async`) that inherit the swaps from the standard `matmul` mcast donors.

**`eltwise` is already CB→DFB-migrated** ([PR #49410](https://github.com/tenstorrent/tt-metal/pull/49410), 50 kernel files); its four remaining `fifo_page_size` reads (`unary/` reader/writer kernels) each already hold a `DataflowBuffer` handle for the swap.

### Cluster 6 — OUT-OF-SCOPE (experimental) ⬜
The experimental sweep confirmed **5 OUT-OF-SCOPE ops** that must not gate other ports (firmware-style CB reconfig / DeepSeek-prefill routing): `deepseek/moe/deepseek_moe_gate`, `deepseek/moe/generalized_moe_gate` (both `reconfig_cbs_for_mask` + `get_cb_tiles_*_ptr` writes — SILENT-WRONG), and `deepseek_prefill/{combine,dispatch,post_combine_reduce}` (`read_tile_value` routing). Track separately with a firmware-style reinit story on Quasar.

---

## 4. What actually needs to happen (deduped action list)

| # | Action | Unblocks | Type |
|---|--------|----------|------|
| 1 | **File Almeet issue:** ring-span getters (`get_total_buffer_size_bytes()` / `fifo_size`/`fifo_limit`) | ring_attention, sdpa, matmul-gathered, **experimental/quasar/matmul, experimental/ccl/llama_all_gather_matmul_async**, groupnorm (Pattern B / Cluster 2) | Runtime API — needed |
| 2 | **Redesign manual ring/window → scratchpad + semaphores** (or strided/multi-DFB on Quasar) | conv2d, matmul-gathered (+ **experimental/quasar/matmul & experimental/ccl/llama_all_gather_matmul_async**, same kernel), sdpa, ring_attention (Pattern A / Cluster 1) | Kernel redesign |
| 3 | **One `memory.h` fix** (LTA on 1xx; Quasar `get_tile_address` on 2xx) | layernorm, layernorm_distributed, groupnorm, dit_layernorm (Cluster 3) | Runtime API in progress + LTA |
| 4 | **Land Quasar `read_tile_value` on DFB** (Runtime, in progress) | sdpa ctrl, sdpa_decode, moe_compute, moe_gpt, manual_seed, unified_routed_expert_ffn (Cluster 4) | Runtime API in progress (2xx only) |
| 5 | Apply mechanical getter swaps | data_movement, eltwise, reduction/generic, matmul standard, **16 experimental YELLOWs** (8 quasar/* + attn_matmul/gelu_backward shared donor + padded_slice + update_padded_kv_cache + 3 ccl composites) (Cluster 5) | Trivial swap |
| 6 | **Track separately (do not gate):** firmware-style reinit for moe-gate + DeepSeek-prefill routing | deepseek_moe_gate, generalized_moe_gate, deepseek_prefill/{combine,dispatch,post_combine_reduce} (Cluster 6) | Out-of-scope |

**Bottom line:** the seven RED ops collapse to **two real problems** — (A) manual ring/window pointer surgery and (B) the missing ring-span getter — and they overlap heavily. The experimental sweep did **not** add a new problem class: its two REDs (`experimental/quasar/matmul`, `experimental/ccl/llama_all_gather_matmul_async`) are the **same gathered-matmul kernel** already covered by (A)+(B), and its 19 YELLOWs are all either mechanical getter swaps (Cluster 5), the shared `memory.h`/LTA fix (Cluster 3), or the `read_tile_value` cluster (Cluster 4). Fixing the Almeet getter (action 1) plus a single reusable scratchpad+semaphore ring pattern (action 2) clears the structural RED across `conv2d`, all three matmul-gathered instances, `sdpa`, and `ring_attention`, leaving `groupnorm` cleared by action 1 alone (or by removing its field read).
