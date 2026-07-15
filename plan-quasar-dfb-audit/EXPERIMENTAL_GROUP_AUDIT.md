# CB→DFB Kernel Audit: `experimental` Group (consolidated)

**Date:** 2026-07-15
**Group:** Experimental ops from the do-list — **18 audited ops + `all_gather_async` (already audited, skipped) + 4 `deepseek_prefill` sub-ops (OUT-OF-SCOPE) / ~60 kernel `.cpp`/`.hpp` files**
**Audit spec:** `plan-quasar-dfb-audit/cb_dfb_kernel_audit_helper.md`

## Group verdict: YELLOW

**Bottom line:** The experimental group is **kernel-clean for the CB→DFB port** with a single YELLOW dependency. Every hard-blocker scan (GATE `get_local_cb_interface`, silent-wrong `get_cb_tiles_*_ptr`, ptr-surgery `fifo_wr_ptr`/`fifo_rd_ptr`/`push_back_hold`/`llk_push_pages`, field-reads `fifo_page_size`/`fifo_num_pages`, moe-gate `reconfig_cbs_for_mask`) returns **zero hits** across all in-scope kernels **and their transitive include closure** (Welford `memory.h`/`combine_welford.h`, `reduce_helpers_compute.hpp`, CCL `minimal_ccl_common.hpp`, integral_image `common.hpp`). The kernels are **not yet DFB-migrated** — they use the object-style `CircularBuffer` API (or the legacy free-function `cb_*` API), both of which map 1:1 to `DataflowBuffer`.

The YELLOW is driven by **two ops with a single LTA-prerequisite CB each**:
1. `dit_layernorm_pre_all_gather` — Welford compute reads a sync-free reciprocal LUT via `get_pointer_to_cb_data` (norm `memory.h`), the canonical **LocalTensorAccessor** prerequisite (exactly the spec's layernorm-Welford example).
2. `group_attn_matmul` — `cb_in2` (c_2) is a sync-free borrowed sharded-in1 tensor view: `get_read_ptr()` used **only** as a hardware-mcast source with **zero** reserve/push/wait/pop — the "pinned CB as L1 pointer" idiom LTA replaces.

Both are **Portable (prereq: LTA)**, **not** quasar-blocked. **No kernel in the group uses `read_tile_value`/`get_tile_address` directly on a DFB**, so there is **no QUASAR-BLOCKED runtime dependency** anywhere. The remaining ops are GREEN: Class 1 canonical FIFOs, plus WEIRD-OK `get_read_ptr()/get_write_ptr() + offset` NOC source/dest tile-fills, plus Class 3 tile-generation scatter (deepseek_grouped_gate writer) and Class 6 private-L1 scratchpads (CCL packet-header CBs, conv3d/MSDA reader arenas, fused-reducer staging), all of which are autoportable or documented ptr workarounds.

## Group-wide classification scan (all in-scope experimental kernels + include closure)

| Pattern (classification) | Hits |
|--------------------------|------|
| `get_local_cb_interface` / `cb_interface.` (GATE) | **none** |
| `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr` (silent-wrong) | **none** |
| `read_tile_value` / `get_tile_address` (quasar-blocked) — **direct kernel calls** | **none** (only *inside* norm `memory.h`, reached via `get_pointer_to_cb_data`) |
| `get_pointer_to_cb_data` (LTA prereq) | **1** — `dit_layernorm_pre_all_gather` welford compute (`cb_reciprocals`) |
| sync-free borrowed `get_read_ptr()` view, no FIFO ops (LTA prereq) | **1** — `group_attn_matmul` `cb_in2` (mcast source only) |
| `fifo_wr_ptr` / `fifo_rd_ptr` / `push_back_hold` / `llk_push_pages` (ptr surgery) | **none** |
| `fifo_page_size` / `fifo_num_pages` (field reads) | **none** |
| `reconfig_cbs_for_mask` / moe-gate `LocalCBInterface` rewrite | **none** |

## Per-op rollup

Arch columns: **1xx** = WH/BH, **2xx** = Quasar. "Factory" = do-list factory. Trivially-GREEN Class-1 ops are rolled up here; the interesting ops have full CB tables below.

| Op | Factory | Kernel API | CB classes | Verdict |
|----|---------|-----------|------------|---------|
| `conv3d` | Conv3dProgramFactory | object CB | 1 + 6 (reader arenas) + WEIRD-OK | **GREEN (workaround)** |
| `matmul/attn_matmul` | AttnMatmulProgramFactory | object CB | 1 + WEIRD-OK ptr | **GREEN (workaround)** |
| `matmul/group_attn_matmul` | GroupAttnMatmulProgramFactory | object CB | 1 + 6 (LTA view) + WEIRD-OK | **YELLOW (LTA prereq)** |
| `multi_scale_deformable_attn` | MSDAOperation | object CB | 1 + 6 (reader/writer arenas) | **GREEN** |
| `reduction/deepseek_grouped_gate` | ProgramFactory | object CB | 1 + 3 (writer tile scatter) | **GREEN (workaround)** |
| `reduction/deepseek_moe_fast_reduce_nc` | DeepseekMoEFastReduceNCProgramFactory | object CB | 1 | **GREEN** |
| `reduction/deepseek_moe_fast_reduce_nc_fused` | DeepseekMoEFastReduceNCFusedMeshWorkloadFactory | object CB | 1 + 6 (reader staging) | **GREEN** |
| `reduction/fast_reduce_nc` | FastReduceNCProgramFactory | object CB | 1 | **GREEN** |
| `reduction/integral_image` | IntImgDeviceOperation | object CB + RAII guard | 1 | **GREEN** |
| `ssm/hc_sum_reduce` | HCSumReduceProgramFactory | object CB | 1 + WEIRD-OK ptr | **GREEN (workaround)** |
| `ssm/prefix_scan` | PrefixScanProgramFactory | object CB | 1 + WEIRD-OK pad | **GREEN (workaround)** |
| `ssm/repeat_and_interleave_eltwise_mul` | RepeatAndInterleaveEltwiseMulProgramFactory | object CB | 1 + WEIRD-OK ptr | **GREEN (workaround)** |
| `transformer/all_reduce_create_qkv_heads` | AllReduceCreateQkvHeadsMeshWorkloadFactory | legacy free-fn | 1 + 6 (pkt hdr) + WEIRD-OK | **GREEN (workaround)** |
| `transformer/dit_layernorm_post_all_gather` | PostAllGatherWelfordProgramFactory | mixed CB | 1 | **GREEN** |
| `transformer/dit_layernorm_pre_all_gather` | PreAllGatherWelfordProgramFactory | mixed CB | 1 + 6 (LTA LUT) | **YELLOW (LTA prereq)** |
| `transformer/fused_distributed_rmsnorm` | FusedRMSNormPost/PreAllGather | object CB | 1 | **GREEN** |
| `transformer/rotary_embedding_hf` | RotaryEmbeddingHfMultiCore(Sharded) | object CB | 1 | **GREEN** |
| `unary_backward/gelu_backward` | GeluBackwardProgramFactory | object CB | 1 | **GREEN** |
| **`ccl/all_gather_async`** | AllGatherViaBroadcastFactory | — | 1 | **GREEN — already audited (SKIPPED)**, see `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/CB_DFB_KERNEL_AUDIT.md` |
| **`deepseek_prefill/combine`** | — | — | — | **OUT-OF-SCOPE** (spec path table) |
| **`deepseek_prefill/dispatch`** | — | — | — | **OUT-OF-SCOPE** (spec path table) |
| **`deepseek_prefill/offset_cumsum`** | — | — | — | **OUT-OF-SCOPE** (spec path table) |
| **`deepseek_prefill/post_combine_reduce`** | — | — | — | **OUT-OF-SCOPE** (spec path table) |

---

## CB portability

Full tables for every non-trivially-GREEN op. Trivially-GREEN Class-1 ops (`fast_reduce_nc`, `deepseek_moe_fast_reduce_nc`, `integral_image`, `dit_layernorm_post_all_gather`, `fused_distributed_rmsnorm`) are summarized in the rollup; representative rows below.

### `reduction/fast_reduce_nc` — GREEN

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` (c_0) | 1 | `reader_reduce_nc.cpp`, `reduce_nc.cpp` | Portable | producer `reserve_back(input_granularity)` → seq NOC fill → `push_back`; consumer `wait_front`/`add_tiles`/`pop_front` | Portable | — |
| `cb_in1` (c_1) | 1 | `reader_reduce_nc.cpp`, `reduce_nc.cpp` | Portable | single zero-scaler tile, single-push/single-pop broadcast FIFO | Portable | — |
| `cb_out0` (c_16) | 1 | `reduce_nc.cpp`, `writer_reduce_nc.cpp` | Portable | producer pack/`push_back(1)`; consumer NOC-write/`pop_front` | Portable | — |

### `reduction/deepseek_moe_fast_reduce_nc` — GREEN

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_compute_input_0` | 1 | `_reader.cpp`, `_reduce.cpp` | Portable | reserve/seq-fill/push → wait/add/pop, canonical multi-tile FIFO | Portable | — |
| `cb_compute_input_1` | 1 | `_reader.cpp`, `_reduce.cpp` | Portable | single zero-scaler broadcast tile | Portable | — |
| `cb_compute_output` | 1 | `_reduce.cpp`, `_writer.cpp` | Portable | producer pack/push; consumer NOC-write/pop | Portable | — |

*No moe-gate pattern — confirmed clean reduction op (no `reconfig_cbs_for_mask` / `LocalCBInterface` / remote_cb).*

### `reduction/deepseek_moe_fast_reduce_nc_fused` — GREEN

Reader + compute live here; the output CB's consumer is the **reused sibling** `deepseek_moe_fast_reduce_nc_writer.cpp` (cross-op donor via program factory).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in_act` | 1 | `_fused_reader.cpp`, `_fused_compute.cpp` | Portable | reserve/seq-fill/push → wait/`mul_tiles_bcast_cols`/pop | Portable | — |
| `cb_scores` | 1 | `_fused_reader.cpp`, `_fused_compute.cpp` | Portable | full-block reserve/fill/push (not scatter), indexed consume/pop | Portable | — |
| `cb_scores_rm` | 6 | `_fused_reader.cpp` | Portable | autoportable: **ScratchpadSpec** — reserve/fill/push but no consumer; self-consumed via raw L1 ptr (not borrowed tensor → not LTA) | Portable | same |
| `cb_expert_indices` | 6 | `_fused_reader.cpp` | Portable | autoportable: **ScratchpadSpec** — self-consumed staging | Portable | same |
| `cb_expert_mapping` | 6 | `_fused_reader.cpp` | Portable | autoportable: **ScratchpadSpec** — self-consumed staging | Portable | same |
| `cb_out` | 1 | `_fused_compute.cpp`, `deepseek_moe_fast_reduce_nc_writer.cpp` (donor) | Portable | producer pack/push; consumer (donor writer) NOC-write/pop | Portable | — |

*No moe-gate pattern — confirmed clean.*

### `reduction/deepseek_grouped_gate` — GREEN (workaround)

Reader + compute are fully canonical Class 1. The **writer** synthesizes routing tiles by filling reserved regions at computed face/tile offsets via `get_write_ptr() + offset` (NOC-read dests and direct L1 writes) with **linear reserve/push credits** — Class 3 tile-generation scatter, WEIRD-OK on 1xx, strided-DFB uplift on 2xx. **No moe-gate `reconfig_cbs_for_mask` / `LocalCBInterface` rewrite** — this is genuine data scatter, not firmware CB reconfig.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in_scores`, `cb_in_bias` | 1 | `reader_deepseek_grouped_gate.cpp`, `compute` | Portable | reserve → NOC read into `get_write_ptr()` → push; canonical | Portable | — |
| `cb_sigmoid_scores`, `cb_biased_scores` | 1 | `compute` | Portable | compute-internal reserve/push/wait/pop | Portable | — |
| `cb_sorted_group_scores`, `cb_sorted_expert_indices_temp`, `cb_group_summed_scores`, `cb_sorted_group_order`, `cb_reduce_intermediate`, `cb_reciprocal_sums`, `cb_normalized_scores`, `cb_final_indices_transposed`, `cb_out_weights`, `cb_out_indices` | 1 | `compute`, `writer` | Portable | canonical reserve/push/wait/pop in compute; `out_*` consumed by writer NOC-write/pop | Portable | — |
| `cb_expert_index_template`, `cb_group_index_template` | 3 | `writer_deepseek_grouped_gate.cpp` | Portable (workaround) | **undesirable but OK hack:** Class 3 tile-fill — `reserve_back(1)` then structured writes at `get_write_ptr()+face_offset` + NOC face-copies, then `push_back(1)`; uplift: strided DFB | Portable (workaround) | **undesirable but OK hack:** ptr tile-fill scatter; uplift: strided multi-producer DFB |
| `cb_top_experts_per_group` | 3 | `writer_deepseek_grouped_gate.cpp` | Portable (workaround) | **undesirable but OK hack:** `reserve_back(N)` then NOC reads into `get_write_ptr() + i*tile + width*faceline` across the reserved N-tile region, single `push_back(N)`; uplift: strided DFB | Portable (workaround) | **undesirable but OK hack:** ptr scatter into reserved region; uplift: strided DFB |
| `cb_winning_group_scores`, `cb_winning_group_indices` | 3 | `writer_deepseek_grouped_gate.cpp` | Portable (workaround) | **undesirable but OK hack:** `reserve_back(topk)` then scatter NOC reads into `dest_base + k*tile + faceline` offsets, `push_back(topk)`; uplift: strided DFB | Portable (workaround) | **undesirable but OK hack:** ptr scatter; uplift: strided DFB |
| `cb_gathered_sigmoid` | 3 | `writer_deepseek_grouped_gate.cpp` | Portable (workaround) | **undesirable but OK hack:** `reserve_back(N)` then gather-writes `gathered_ptr[computed_offset]` within reserved region, `push_back(N)`; uplift: strided DFB / scratchpad | Portable (workaround) | same |
| `cb_epsilon_scalar`, `cb_route_scale_scalar`, `cb_reduce_ones_scalar` | 1 | `writer` | Portable | scalar-fill tiles: reserve/write-at-base/push | Portable | — |

### `reduction/integral_image` — GREEN

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `input_cb` | 1 | `intimg_reader.cpp`, `intimg_compute.cpp` | Portable | reader `load_from_dram` (WriteCBGuard reserve/push); compute wait/add/pop | Portable | — |
| `start_cb` | 1 | `intimg_reader.cpp`, `intimg_compute.cpp` | Portable | initial-zero tile, ReadCBGuard wait/pop | Portable | — |
| `acc_cb` | 1 | `intimg_compute.cpp` | Portable | compute-internal running accumulator, clean reserve/push/wait/pop each iter (self-loop-candidate on 2xx, informational) | Portable | — |
| `cumsum_stage_0/1/2_cb` | 1 | `intimg_compute.cpp` | Portable | compute-internal, RAII Read/Write guards → balanced FIFO | Portable | — |
| `axis_2_buffer_cb` | 1 | `intimg_compute.cpp` | Portable | "save last tile for propagation": reserve/push then wait/pop; canonical credits | Portable | — |
| `axis_3_buffer_cb` | 1 | `intimg_reader.cpp`/`intimg_writer.cpp`, `intimg_compute.cpp` | Portable | upper-block re-read, ReadCBGuard consume | Portable | — |
| `output_cb` | 1 | `intimg_compute.cpp`, `intimg_writer.cpp` | Portable | producer WriteCBGuard; consumer `write_to_dram` NOC-write/pop | Portable | — |

*`common.hpp` `ReadCBGuard`/`WriteCBGuard` are RAII wrappers around `wait_front`/`pop_front` and `reserve_back`/`push_back` — guarantee balanced canonical FIFO ops.*

### `ssm/hc_sum_reduce` — GREEN (workaround)

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` (c_0) | 1 | `reader_ssm_1d_sum_reduce.cpp`, `ssm_1d_sum_reduce.cpp` | Portable | reserve/NOC-read/push → transpose wait/pop | Portable | — |
| `cb_scalar` (c_2) | 1 | `reader...`, `ssm_1d_sum_reduce.cpp` | Portable | reduce-scaler tile, persistent broadcast | Portable | — |
| `intermed_cb_id0` | 1 | `ssm_1d_sum_reduce.cpp` | Portable | compute-internal transpose→reduce FIFO | Portable | — |
| `intermed_cb_id1` | 1 | `ssm_1d_sum_reduce.cpp`, `writer_ssm_1d_sum_reduce.cpp` | Portable (workaround) | **undesirable but OK hack:** consumer writer reads `get_read_ptr() + FACE_SIZE_BYTES` as NOC source; uplift: none needed (Class-1-ish) | Portable (workaround) | same |
| `intermed_cb_id2` | 1 | `writer...` (producer), `ssm_1d_sum_reduce.cpp` (consumer) | Portable (workaround) | **undesirable but OK hack:** writer assembles tile via `get_write_ptr()+face` NOC-read dests, then push; uplift: none needed | Portable (workaround) | same |
| `output_cb_id` | 1 | `ssm_1d_sum_reduce.cpp`, `writer...` | Portable | producer pack/push; consumer NOC-write (offset 0)/pop | Portable | — |

### `ssm/repeat_and_interleave_eltwise_mul` — GREEN (workaround)

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0`, `cb_in1` | 1 | `reader_ssm_eltwise_mul.cpp`, `ssm_eltwise_mul.cpp` | Portable | reserve/NOC-read/push → transpose or mul wait/pop | Portable | — |
| `cb_in0_transposed`, `cb_out_transposed` | 1 | `ssm_eltwise_mul.cpp` | Portable | compute-internal transpose FIFO | Portable | — |
| `cb_out` | 1 | `ssm_eltwise_mul.cpp`, `writer_ssm_eltwise_mul.cpp` | Portable | producer pack/push; consumer NOC-write/pop | Portable | — |
| `cb_in1_transposed` | 1 | `ssm_eltwise_mul.cpp` (producer), `reader_ssm_eltwise_mul.cpp` (consumer) | Portable (workaround) | **undesirable but OK hack:** reader reads `get_read_ptr() + one_face_bytes` (advancing read ptr) as NOC source; uplift: none needed | Portable (workaround) | same |
| `cb_in1_bcast_row` | 1 | `reader_ssm_eltwise_mul.cpp` (producer), `ssm_eltwise_mul.cpp` (consumer) | Portable (workaround) | **undesirable but OK hack:** reader writes tile via `get_write_ptr()+one_face_bytes` NOC-read dests; uplift: none needed | Portable (workaround) | same |

### `ssm/prefix_scan` — GREEN (workaround)

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in` (shared a/bx staging) | 1 | `reader_ssm_prefix_scan.cpp`, `ssm_prefix_scan.cpp` | Portable (workaround) | **undesirable but OK hack:** on short row tails, tail-zero-fill via `CoreLocalMem(get_write_ptr() + bytes_to_copy)` before `push_back`; otherwise canonical reserve/NOC-read/push | Portable (workaround) | same |
| `cb_h_in` | 1 | `reader...`, `ssm_prefix_scan.cpp` | Portable | reserve/NOC-read/push → copy/pop (non-shard-backed to keep low L1 addr) | Portable | — |
| `cb_a_tilize_in`, `cb_bx_tilize_in`, `cb_h_prev`, `cb_ah`, `cb_h`, `cb_tilize_out`, `cb_h_acc` | 1 | `ssm_prefix_scan.cpp` | Portable | compute-internal recurrence/tilize FIFOs, canonical reserve/push/wait/pop (PACK→UNPACK self-loop-candidates on 2xx, informational) | Portable | — |
| `cb_out` | 1 | `ssm_prefix_scan.cpp`, `writer_ssm_prefix_scan.cpp` | Portable | producer pack/push; consumer NOC-write/pop | Portable | — |

### `transformer/all_reduce_create_qkv_heads` — GREEN (workaround)

Fused all-reduce + create-qkv-heads CCL op (fabric + semaphores, MeshWorkload). CBs are Class 1 / Class 6 scratchpad / WEIRD-OK; sync via semaphores + fabric is the intended Metal 2.0 pattern (not a CB concern).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb0_id` / `cb_in0` | 1 | `worker_reader.cpp` (producer), `worker_writer.cpp` (consumer), `reduction.cpp` (consumer) | Portable | reader reserve/NOC-read/push; writer wait/`get_read_ptr` fabric-write/pop; compute wait/add-reduce | Portable | — |
| `cb_id` (reduction input signal) | 1 | `reduction_receiver.cpp` | Portable | receiver `cb_push_back` after semaphore wait → signals compute; canonical push credits | Portable | — |
| `cb_out0` / `cb_id_reduction_out` | 1 | `reduction.cpp` (producer), `reduction_receiver.cpp` (consumer) | Portable (workaround) | **undesirable but OK hack:** receiver reads `get_read_ptr(cb_id_reduction_out) + in_tile_offset_by_batch + i*TILE_ELEMENTS*2` as NOC source for the qkv-head split-write; uplift: LTA (borrowed resident read) | Portable (workaround) | same |
| `cb_batch_offset_id` | 1 | `reduction_receiver.cpp` | Portable | index tile: reserve/NOC-read/push then wait; canonical | Portable | — |
| `reserved_packet_header_cb_id` | 6 | `worker_writer.cpp` | Portable | autoportable: **ScratchpadSpec** — 3× reserve/push allocating packet-header slots, no FIFO consumer (private L1 for fabric pkt headers); standard CCL pattern | Portable | same |

### `transformer/dit_layernorm_post_all_gather` — GREEN

Welford post-all-gather combine (`combine_welford.h` donor — **zero blocker hits**; no `read_tile_value`/`get_pointer_to_cb_data`). All CBs canonical Class 1.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_inp`, `cb_stats`, `cb_stats_reduced`, `cb_intermediate`, `cb_recip_sqrt_var`, `cb_gamma`, `cb_beta`, `cb_eps`, `norm_target_cb`, `gamma_out_cb`, `cb_out` | 1 | `layernorm_post_allgather_welford.cpp`, `reader_layernorm_postallgather_dit.cpp`, `writer_layernorm_postallgather_dit.cpp` | Portable | canonical FIFO throughout; reader/compute/writer producer-consumer; combine via `combine_welford.h` (no CB field access) | Portable | — |

### `transformer/dit_layernorm_pre_all_gather` — YELLOW (LTA prereq)

Welford pre-all-gather. `cb_reciprocals` is a sync-free borrowed reciprocal LUT read via `get_pointer_to_cb_data<recip_lut_t>` (norm `memory.h`, line 40). Per spec this is the **LocalTensorAccessor** prerequisite (same as the reference layernorm-Welford example) — **not** quasar-blocked (the kernel does not call `read_tile_value`/`get_tile_address` directly; the LUT is a borrowed L1 view).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_inp` (c_0) | 1 | `layernorm_pre_allgather_welford.cpp`, `reader_layernorm_preallgather_dit.cpp` | Portable | consumer `cb_wait_front`/`transpose_tile`/`cb_pop_front`; producer = reader | Portable | — |
| `cb_x2` (c_1) | 1 | `layernorm_pre_allgather_welford.cpp` | Portable | compute-internal reserve/pack/push then wait/transpose/pop (PACK→UNPACK self-loop-candidate on 2xx) | Portable | — |
| `cb_reciprocals` (c_2) | 6 | `layernorm_pre_allgather_welford.cpp` | Portable (prereq: LTA) | sync-free borrowed reciprocal LUT via `get_pointer_to_cb_data` (norm `memory.h`) → **LocalTensorAccessor** (host `TensorBinding` + kernel ctor) | Portable (prereq: LTA) | same — LTA (not quasar-blocked; no direct DFB `read_tile_value`) |
| `cb_out` (c_14) | 1 | `layernorm_pre_allgather_welford.cpp`, `writer_layernorm_preallgather_dit.cpp` | Portable | producer reserve/pack/push; consumer writer NOC-write/pop | Portable | — |

### `transformer/fused_distributed_rmsnorm` — GREEN

RMSNorm pre/post all-gather (`FusedRMSNormPreAllGather` + `FusedRMSNormPostAllGather`). Compute uses `reduce_helpers_compute.hpp` (**not** the Welford `memory.h`/`welford.h` path) — **zero blocker hits**; no `read_tile_value`/`get_pointer_to_cb_data`. The `writer_unary_interleaved_start_id_blocked.cpp` donor writer has **no `fifo_page_size`** field read. All CBs canonical Class 1.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_input`, `cb_stats`, `cb_intermediate`, `cb_reduce_result`, `cb_reduce_scalar`, `cb_epsilon`, `cb_weight`, `cb_mul_rms_result`, `cb_mul_weight_result`, `cb_output` | 1 | `rmsnorm_pre_allgather.cpp`, `rmsnorm_post_allgather.cpp`, `rms_*_reader.cpp`, `rms_*_writer.cpp`, `writer_unary_interleaved_start_id_blocked.cpp` | Portable | canonical reserve/push/wait/pop; reduce helpers use no CB field access | Portable | — |
| `cb_rope_cos`, `cb_rope_sin`, `cb_transformation_mat`, `cb_rotated_input` | 1 | `rms_post_allgather_reader.cpp`, `rmsnorm_post_allgather.cpp` | Portable | RoPE fusion inputs, canonical FIFO | Portable | — |

---

### `conv3d` — GREEN (workaround)

Kernels: `compute.cpp`, `reader_vol2col.cpp`, `writer.cpp` (headers `conv3d_gather_tuning.hpp`, `conv3d_weight_share.hpp` are tuning constants / role enums — no CBs).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_vol2col` (CTA0) | 1 | `reader_vol2col.cpp`, `compute.cpp` | Portable | reader `ChunkWriter` reserve/NOC-write/push → compute `tilize` wait/pop | Portable | — |
| `cb_vol2col_tiled` (CTA1) | 1 | `compute.cpp` | Portable | tilize produce → matmul wait/pop; internal FIFO | Portable | — |
| `cb_weight_tiled` (CTA2) | 1 | `writer.cpp`, `compute.cpp` | Portable | writer reserve/read/push; compute wait/pop; mcast uses `get_write_ptr()` base (no +offset) | Portable | — |
| `cb_bias_tiled` (CTA3) | 1 | `writer.cpp`, `compute.cpp` | Portable | reducer reserve/read/push → compute wait/pop | Portable | — |
| `cb_matmul_interm_tiled` (CTA4) | 1 | `compute.cpp`, `writer.cpp` | Portable (workaround) | **undesirable but OK hack:** reducer in writer reads remote worker's `get_read_ptr()` as NOC source (no offset); uplift: LTA (borrowed remote read) | Portable (workaround) | same |
| `cb_matmul_result_rm` / `cb_out` (CTA5) | 1 | `compute.cpp`, `writer.cpp` | Portable | untilize produce; writer `noc.async_write(cb_out, …, {.offset_bytes=…})` via CB object (canonical) | Portable | — |
| `cb_reduction_tiled` (CTA6) | 1 | `compute.cpp`, `writer.cpp` | Portable | genuine cross-core partials FIFO: worker reserve/push, reducer reserve/NOC-read/push, consume wait/pop | Portable | — |
| `cb_worker_ack_back` / `cb_ack` (CTA7) | 1 | `compute.cpp`, `writer.cpp` | Portable | small ack FIFO reserve/push ↔ wait/pop | Portable | — |
| `cb_input_shard` / `shard_cb` (CTA31) | 6 | `reader_vol2col.cpp` | Portable | autoportable: **ScratchpadSpec** — `reserve_back(shard_total+coalesced)` once, never pushed; fixed L1 gather arena via `get_write_ptr()` | Portable | same |
| `cb_dram_read_scratch` (CTA39) | 6 | `reader_vol2col.cpp` | Portable | autoportable: **ScratchpadSpec** — single reused staging page, never pushed | Portable | same |

### `matmul/attn_matmul` — GREEN (workaround)

Kernels: `compute/transformer_attn_matmul.cpp`, `dataflow/reader_transformer_attn_matmul.cpp` (output writer is outside this dir).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` (c_0) | 1 | reader, compute | Portable | reserve(Kt)/read/push → wait/pop | Portable | — |
| `cb_in1` (c_1) | 1 | reader, compute | Portable | reserve/read/push → wait/pop | Portable | — |
| `cb_intermed0` (c_2) | 1 | compute | Portable | compute-internal reserve/pack/push → untilize consume | Portable | — |
| `cb_intermed1` (c_3) | 1 | compute, reader | Portable (workaround) | **undesirable but OK hack:** reader reads `get_read_ptr() + row_bytes` as NOC transpose source; uplift: none needed (Class-1-ish) | Portable (workaround) | same |
| `cb_intermed2` (c_4) | 1 | reader, compute | Portable (workaround) | **undesirable but OK hack:** reader fills tile via `get_write_ptr()+offset` local NOC dest; uplift: none needed | Portable (workaround) | same |
| `out_cb_id` (c_5) | 1 | compute (+ external writer) | Portable | tilize produce; canonical output FIFO | Portable | — |

### `matmul/group_attn_matmul` — YELLOW (LTA prereq)

Kernels: `compute/transformer_group_attn_matmul.cpp`, `dataflow/reader_mcast_transformer_group_attn_matmul.cpp`, `dataflow/writer_transformer_group_attn_matmul.cpp`.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` (c_0) | 1 | writer, compute | Portable | reserve(in0_block_w)/read/push → wait/pop (IN0_SHARDED skips read, keeps credits) | Portable | — |
| `cb_in1` (c_1) | 1 | reader (mcast), compute | Portable | reserve → hardware mcast into CB → push; `get_write_ptr()` as mcast base (no +offset); compute wait/pop | Portable | — |
| `cb_in2` (c_2) | 6 | `reader_mcast_transformer_group_attn_matmul.cpp` | Portable (prereq: LTA) | sync-free borrowed sharded-in1 view: `get_read_ptr()` used **only** as mcast source (`CoreLocalMem mcast_src`), **zero reserve/push/wait/pop** → **LocalTensorAccessor** | Portable (prereq: LTA) | same (not quasar-blocked — no DFB `read_tile_value`) |
| `cb_intermed0` (c_3) | 1 | compute, writer | Portable (workaround) | **undesirable but OK hack:** writer reads `get_read_ptr() + row_offset_bytes` as NOC transpose source; uplift: none needed | Portable (workaround) | same |
| `cb_intermed1` (c_4) | 1 | writer, compute | Portable (workaround) | **undesirable but OK hack:** writer fills via `get_write_ptr()+offset` local NOC dest; uplift: none needed | Portable (workaround) | same |
| `out_cb_id` / `cb_id_out` (c_5) | 1 | compute, writer | Portable | reserve/tilize/push; writer `noc.async_write(cb_out, …, {.offset_bytes=…})` via CB object; OUT_SHARDED = wait-only resident | Portable | — |

### `multi_scale_deformable_attn` — GREEN

Kernels: `compute/msda_compute.cpp`, `dataflow/reader_msda.cpp`, `dataflow/writer_msda.cpp` (`msda_tile_layout.hpp` = offset helpers, no CBs).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `value_scratch_cb` (reader CTA0) | 6 | `reader_msda.cpp` | Portable | autoportable: **ScratchpadSpec** — `reserve_back` once, never pushed; fixed staging arena | Portable | same |
| `grid_cb` (reader CTA1) | 6 | `reader_msda.cpp` | Portable | autoportable: **ScratchpadSpec** — arena, never pushed | Portable | same |
| `attn_cb` (reader CTA2) | 6 | `reader_msda.cpp` | Portable | autoportable: **ScratchpadSpec** — arena, never pushed | Portable | same |
| `input_tile_cb` (reader CTA3 / compute c0) | 1 | reader, compute | Portable | reader reserve → scatter value sticks into faces via `get_write_ptr()`+face-offset → push; compute wait/pop (producer fills reserved region then pushes = canonical) | Portable | — |
| `scalar_tile_cb` (reader CTA4 / compute c1) | 1 | reader, compute | Portable | reserve → col-0 lane fill → push; compute wait/pop | Portable | — |
| `output_cb` (compute c2 / writer CTA0) | 1 | compute, writer | Portable | compute `reserve_back(1)` → `reduction_size` `pack_tile<true>` with `pack_reconfig_l1_acc` accumulating into the single reserved slot → `push_back(1)` (L1-acc **within** one produce step, not a re-read FIFO → linear Class 1, **not** Class 5); writer wait/`get_read_ptr()`+face-offset local reads/NOC-write/pop | Portable | — |
| `output_scratch_cb` (writer CTA1) | 6 | `writer_msda.cpp` | Portable | autoportable: **ScratchpadSpec** — staging stick, never pushed | Portable | same |

### `transformer/rotary_embedding_hf` — GREEN

All 9 kernels (3 compute variants + 6 dataflow) use only canonical reserve/push/wait/pop — uniformly Class 1, no scratch/offset/accumulate hacks. CB indices vary per variant; roles are uniform.

| CB (role) | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `in_cb` / `cb_input` | 1 | readers, compute | Portable | interleaved reader reserve/fill/push → compute wait/pop; sharded compute claims+consumes resident shard via reserve(Wt)/push(Wt)/wait(Wt)/pop(Wt) = linear FIFO | Portable | — |
| `cos_cb`, `sin_cb`, `cos_interm`, `sin_interm` | 1 | readers, compute | Portable | readers produce or compute self-produces resident shard; interm CBs compute-internal reserve/push→wait/pop | Portable | — |
| `scalar_cb` | 1 | readers, compute | Portable | reserve/fill/push → wait (popped at end) | Portable | — |
| `rotated_in_cb`, `rotated_in_interm_cb` | 1 | reader_interleaved, compute | Portable | reader produces `cb_rotated_input`; compute produces/consumes interm | Portable | — |
| `trans_mat_cb` | 1 | single-tile reader/compute | Portable | reserve/fill/push → wait | Portable | — |
| `out_cb` / `cb_output` | 1 | compute, writer | Portable | compute reserve/pack/push; interleaved writer wait/`get_read_ptr()` whole-tile NOC source (no offset)/pop; sharded = wait-only resident | Portable | — |

### `unary_backward/gelu_backward` — GREEN

Kernels: `compute/eltwise_bw_gelu_poly.cpp`, `compute/eltwise_bw_gelu_approx_tanh.cpp` (identical CB structure; input readers / output writer are outside this dir). All multi-step math is in DST registers, not CBs.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_grad_out` (c_0) | 1 | compute (consumer) | Portable | `wait_front(1)`/`pop_front(1)` | Portable | — |
| `cb_input` (c_1) | 1 | compute (consumer) | Portable | `wait_front(1)`/`pop_front(1)` | Portable | — |
| `cb_grad_in` (c_2) | 1 | compute (producer) | Portable | `reserve_back(1)`/`pack_tile`/`push_back(1)` | Portable | — |

---

## GATE hits (must be empty to merge)

- **(none)** — zero `get_local_cb_interface(...).<field>` reads/writes across all in-scope kernels and their include closure.

## Blocked on runtime (2xx rollup)

- **(none)** — no kernel calls `read_tile_value` / `get_tile_address` directly on a DFB. The Welford `dit_layernorm_pre_all_gather` LUT uses `get_pointer_to_cb_data` → **LTA prereq (YELLOW)**, not a runtime-API block. There is **no QUASAR-BLOCKED op** in this group.

## LTA prerequisites (YELLOW)

- `dit_layernorm_pre_all_gather` — `cb_reciprocals` (c_2) via `get_pointer_to_cb_data` at `layernorm_pre_allgather_welford.cpp:40` (helper `norm::kernel_util::compute::memory::get_pointer_to_cb_data`, `normalization/kernel_util/compute/memory.h:30`). Migrate to `LocalTensorAccessor`.
- `group_attn_matmul` — `cb_in2` (c_2) at `reader_mcast_transformer_group_attn_matmul.cpp:123` (`in1_sharded_cb_addr = cb_in2_obj.get_read_ptr()`, used as `CoreLocalMem mcast_src` at line 239; no reserve/push/wait/pop on the CB). Sync-free borrowed sharded-in1 tensor view → migrate to `LocalTensorAccessor`.

## Cross-op donor kernels

- `reduction/deepseek_moe_fast_reduce_nc/device/kernels/deepseek_moe_fast_reduce_nc_writer.cpp` — reused as the output writer for `deepseek_moe_fast_reduce_nc_fused` (no writer of its own).
- `normalization/kernel_util/compute/memory.h` — `get_pointer_to_cb_data` (→ `get_tile_address`) donor for `dit_layernorm_pre_all_gather` welford. **LTA prereq source.**
- `normalization/kernel_util/compute/combine_welford.h` — Welford-combine donor for `dit_layernorm_post_all_gather` (clean, no CB field access).
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` / `reduce_helpers_dataflow.hpp` — reduce-scaler helpers for `fused_distributed_rmsnorm`, `deepseek_grouped_gate` (clean).
- `ccl/common/kernels/minimal_ccl_common.hpp` + `ccl/shared_with_host/*` — fabric/CCL donors for `all_reduce_create_qkv_heads` (clean).

## Notes & follow-ups

- **Kernels are not yet DFB-migrated.** All ops use object-style `CircularBuffer` (or legacy free-fn `cb_*`), which maps 1:1 to `DataflowBuffer`. Kernel-side "GREEN/YELLOW" here does **not** imply end-to-end Metal 2.0 port — host `ProgramSpec`/`DataflowBufferSpec` migration and SPSC/endpoint legality are tracked by the host audit, not this kernel audit.
- **`deepseek_grouped_gate` is NOT moe-gate.** Despite the name, it contains no `reconfig_cbs_for_mask` / `LocalCBInterface` rewrite / `get_cb_tiles_*_ptr`. Its writer does genuine Class 3 data scatter (tile generation), which is WEIRD-OK on 1xx and strided-DFB-uplift on 2xx — GREEN (workaround), not SILENT-WRONG.
- **The three `deepseek_*` reduction ops** (`deepseek_grouped_gate`, `deepseek_moe_fast_reduce_nc`, `deepseek_moe_fast_reduce_nc_fused`) were held IN scope (not in the spec OUT-OF-SCOPE table) and are all clean of moe-gate patterns.
- **CCL ops** (`all_reduce_create_qkv_heads`) rely on semaphores + fabric for cross-kernel sync — this is the intended Metal 2.0 mechanism and does not affect CB classification. `all_gather_async` is already audited (GREEN) and was not re-audited.
- **Welford runtime API:** the `read_tile_value`/`get_tile_address`-on-DFB in-flight runtime fix is **not required** by this group — no kernel uses those APIs directly. Only the `get_pointer_to_cb_data` LUT (LTA path) touches `get_tile_address` transitively, and its port strategy is LocalTensorAccessor, independent of the runtime fix.
