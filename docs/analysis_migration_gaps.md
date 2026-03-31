# Migration Gap Analysis — Instance 4, Phase 1

## Methodology

Read the current `matmul_block` helper (`matmul_block_helpers.hpp/inl`), the prior llk3 attempts (`matmul_tile_helpers.hpp`, `matmul_block_fused_bias_helpers.hpp/inl`), and every "not migratable" kernel in full. For each kernel, identified the standard matmul_block pattern lines vs. custom lines, the exact features the current helper lacks, feature interactions, and composability potential.

---

## Per-Kernel Analysis

### 1. bmm_large_block_zm_fused_bias_activation.cpp

- **File**: `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`
- **Total LOC**: 500
- **Standard matmul_block LOC**: ~60 (core matmul loop lines 258-300 + sub-block iteration structure)
- **Custom LOC**: ~440 (all `#ifdef` paths, bias phase, transpose, untilize, DRAM sharded, etc.)
- **Missing features**:
  1. **PACKER_L1_ACC support in pack loop**: With L1_ACC, partial results accumulate in L1 via packer instead of software spill/reload through intermediate CB. The current helper always does software spill/reload (`copy_tile_to_dst` from interm_cb). With L1_ACC the helper needs: `llk_pack_reconfig_l1_acc(0)` on block 0, `llk_pack_reconfig_l1_acc(1)` on block 1+, and different reload/pop semantics depending on whether FUSE_BIAS is enabled.
  2. **FP32_DEST_ACC_EN**: Requires extra `pack_reconfig_data_format(mm_out_cb_id)` before packing the final block. Currently the helper only does `pack_reconfig_data_format(out_cb)` once at init.
  3. **FUSE_BIAS with row broadcast**: A separate post-matmul phase (lines 404-463): data format reconfig from matmul to bias-add, `add_bcast_rows_init_short`, per-subblock `add_tiles_bcast_rows`, optional SFPU, then pack to untilize_mode_out_cb. The helper has no post-matmul bias phase.
  4. **SFPU activation with bias interaction**: Without bias, SFPU runs on DST tiles after matmul on the last K-block (line 307-309). With bias, SFPU runs after bias addition (lines 442-444). The current PostComputeFn fires before pack on the last K-block, which is wrong for the FUSE_BIAS path.
  5. **PACK_RELU toggle**: `llk_pack_relu_config(ReluType::NO_RELU)` at loop start, `ZERO_RELU` on last output block. Toggled differently depending on FUSE_BIAS. Not exposed by the helper.
  6. **in0_transpose_tile**: Pre-matmul WH transpose of A tiles via `transpose_tile_block` into a separate CB, with packer L1_ACC disabled during transpose and format reconfig. The helper has a `transpose` template param but it controls B transpose (`in1_transpose_tile`), not A transpose.
  7. **SKIP_COMPUTE**: Skip the matmul loop entirely (bias-only path). The helper always runs the matmul.
  8. **untilize_out**: Post-matmul (or post-bias) pack untilize via `reblock_and_untilize` or `pack_untilize_dest`. The helper packs tiles normally.
  9. **MATMUL_DRAM_SHARDED**: Early return for non-worker cores. The helper has no core classification.
  10. **Outer H/W dimension loops**: The production kernel has `num_blocks_h_dim` x `num_blocks_w_dim` outer loops. The helper only has a batch loop.
  11. **Half-sync DST management**: The production kernel uses `tile_regs_acquire/commit/wait/release` (half-sync mode). The helper uses `acquire_dst/release_dst` (full-sync wrappers). Half-sync allows MATH to start the next sub-block while PACK finishes the previous one.
  12. **get_batch_from_reader**: Dynamic batch validity via mailbox from BRISC. Minor; could be handled outside the helper.
- **Feature interactions**:
  - PACKER_L1_ACC + FUSE_BIAS: Never reload; instead pop interm after each non-last block. Changes the enable_reload logic entirely.
  - PACKER_L1_ACC + in0_transpose_tile: L1_ACC must be disabled during the transpose phase (`llk_pack_reconfig_l1_acc(0)` before transpose), re-enabled after.
  - FP32_DEST_ACC_EN + PACKER_L1_ACC: Both require `pack_reconfig_data_format` before packing the final block.
  - FUSE_BIAS + SFPU: Activation fires after bias, not after matmul.
  - FUSE_BIAS + PACK_RELU: RELU enabled only for the final bias-add output, not the matmul output.
  - FUSE_BIAS + untilize_out: Output destination changes (`mm_partials_cb_id` vs `out_cb_id`).
- **Composability assessment**: The production kernel has deeply interleaved `#ifdef` paths that can't be captured by a single monolithic helper (this is exactly what the failed `matmul_block_fused_bias` attempt proved). However, decomposing into: (a) a K-blocking matmul core with L1_ACC support, (b) a separate bias-add phase, (c) separate PACK_RELU/untilize phases — with the caller composing them — could work. The key insight is that L1_ACC fundamentally changes the spill/reload strategy, so it's not just an additive feature.

---

### 2. bmm_large_block_zm_fused_bias_activation_gathered.cpp

- **File**: `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`
- **Total LOC**: 464
- **Standard matmul_block LOC**: ~50 (core matmul loop lines 341-363)
- **Custom LOC**: ~414 (gathered infrastructure, CB pointer management, ring distribution, sync)
- **Missing features** (beyond production kernel #1):
  1. **Per-batch output/partials CB arrays**: `mm_out_cb_ids[b]` and `mm_partials_cb_ids[b]` indexed by batch. The helper uses a single `out_cb` and `interm_cb` template param.
  2. **Dynamic K dim per block**: `unpadded_in0_block_w` varies per ring iteration based on runtime `unpadded_in0_shard_widths_in_tiles[curr_ring_idx]`. The helper's K dim (`block_w`) is fixed for all blocks.
  3. **Alternating in0 CB source**: `input0_cb_id = block == 0 ? in0_cb_id : in2_cb_id`. The helper uses a single `in0_cb` for all blocks.
  4. **ENABLE_GLOBAL_CB path**: Direct `LocalCBInterface` manipulation for ring-based CB read pointer arithmetic (`calculate_next_block_index_and_update_rd_ptr`, `update_rd_ptr_to_ring_index`, `update_local_cb_rd_ptr`).
  5. **Cross-core sync primitives**: `sync_cb`/`sync_cb2` wait/push for coordinating ring data movement.
  6. **CORE_TYPE classification**: Idle/worker/hop core early-return.
  7. **No FUSE_BIAS in gathered variant**: The gathered kernel doesn't have the same bias fusion (at least not in the current code). But it does have SFPU_OP_INIT_ACTIVATION, PACK_RELU, PACKER_L1_ACC, FP32_DEST_ACC_EN, SKIP_COMPUTE, IN1_TRANSPOSE_TILE.
  8. **Inline `pack_untilize_dest` for untilize_out**: Different from production kernel's `reblock_and_untilize`.
- **Feature interactions**: Same as production kernel for shared features. Additionally, dynamic K dim per block interacts with the matmul_block call (the inner loop uses `unpadded_in0_block_w` not `in0_block_w`).
- **Composability assessment**: The gathered kernel is fundamentally different from the standard matmul because of the ring distribution. The matmul core is the same sub-blocked pattern, but the surrounding infrastructure (CB pointer management, ring sync, per-batch CBs, dynamic K dim) makes it impractical for a helper to absorb. Best approach: make the matmul core a composable building block that the gathered kernel calls within its ring loop.

---

### 3. conv_bmm_tilize.cpp

- **File**: `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize.cpp`
- **Total LOC**: 640
- **Standard matmul_block LOC**: ~30 (inner matmul_block loop lines 424-441)
- **Custom LOC**: ~610 (fused tilize input, activation reuse, CB pointer management, PACKER_L1_ACC, FUSE_BIAS, untilize output, skip_compute, split reader)
- **Missing features**:
  1. **Fused tilize input**: Pre-matmul tilize phase using `compute_kernel_lib::tilize` helper with multiple modes (split reader, activation reuse, pretilize CB). The matmul helper doesn't handle input tilization.
  2. **PACKER_L1_ACC with complex toggling**: `pack_reconfig_l1_acc(0)` for first block, `pack_reconfig_l1_acc(1)` for middle blocks, `pack_reconfig_l1_acc(fuse_bias ? 1 : 0)` for last block. Different from production kernel's L1_ACC pattern.
  3. **FUSE_BIAS with row broadcast**: Same pattern as production kernel (lines 543-595).
  4. **SFPU activation**: Applied after bias (with FUSE_BIAS) or after matmul (without), same as production.
  5. **PACK_RELU**: Same toggle pattern.
  6. **untilize_out**: Both `pack_untilize_dest` (packer_untilize=true) and `compute_kernel_lib::untilize` helper (packer_untilize=false) paths.
  7. **CB pointer reset between blocks**: Direct `fifo_rd_ptr`/`fifo_wr_ptr` manipulation when L1_ACC or spill is active. Lines 480-483, 492-493, 503-506.
  8. **check_skip_compute**: Runtime skip for non-output cores in block-sharded conv2d.
  9. **Activation reuse with CB pointer offsets**: Complex window-based CB pointer update for spatial reuse across convolution windows.
  10. **40+ compile-time args**: The kernel is heavily parameterized at compile time.
  11. **mm_block_init_short_with_both_dt**: Uses a different init variant than the helper's `mm_block_init`.
- **Feature interactions**: PACKER_L1_ACC + FUSE_BIAS + spill: CB pointer resets differ depending on all three. Tilize + matmul transitions require data format reconfig between phases.
- **Composability assessment**: Could be expressed as: tilize helper (already exists) → matmul helper → bias helper → untilize helper (already exists). The matmul core is standard sub-blocked pattern. The complexity is in the surrounding orchestration (L1_ACC toggling, CB pointer management, mode switches). A composable matmul helper that handles L1_ACC and can be called between tilize and untilize phases would cover this kernel.

---

### 4. compute_streaming.hpp (SDPA)

- **File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp`
- **Total LOC**: 1528
- **Standard matmul_block LOC**: 0 (uses custom `blocked_matmul_and_pack` function, not the standard pattern)
- **Custom LOC**: ~1528 (entire file is SDPA-specific pipeline)
- **Missing features**:
  1. **Out-of-order packing**: `pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset + c)` — tiles packed at absolute positions, not sequentially.
  2. **Architecture-specific matmul**: `matmul_block_no_mop` on Blackhole, standard `matmul_block` on Wormhole.
  3. **L1_ACC for row-sum accumulation**: `llk_pack_reconfig_l1_acc` used for accumulating row sums across columns, not for matmul spill/reload.
  4. **CB write pointer manipulation**: `cb_push_back_hold_wr_ptr` — push tiles for UNPACK visibility but rewind wr_ptr for subsequent `pack_tile<true>`.
  5. **HW semaphore pipelining**: `t6_semaphore_post<p_stall::NONE>(semaphore::FPU_SFPU)` for triggering reduce from matmul.
  6. **Blocked pack mode on Blackhole**: `llk_pack_mop_config` for block-level packing.
  7. **Complex multi-stage pipeline**: matmul → max-reduction → sub+exp → softmax → SALAD corrections → output normalization. Matmul is one stage of ~10.
  8. **Ring accumulator state**: Persistent ping-pong buffers across ring iterations.
  9. **exp_packthread_tile**: Exp operation running on PACK thread.
  10. **Architecture-specific sub_bcast_cols_custom on Blackhole**.
- **Feature interactions**: The matmul is deeply coupled with the SDPA pipeline — the output packing positions are determined by the SDPA row/column structure, and the matmul triggers subsequent reduce operations via hardware semaphores.
- **Composability assessment**: **NOT composable with matmul helpers.** The matmul is a small inner operation within a complex SDPA pipeline. The `blocked_matmul_and_pack` function is already a self-contained building block (~55 lines) that packs at absolute offsets. Any helper would need to be a very thin wrapper around the raw `matmul_block` LLK call — essentially what `blocked_matmul_and_pack` already is. Attempting to use the matmul_block helper here would add complexity, not reduce it.

---

### 5. minimal_matmul/compute.cpp

- **File**: `ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp`
- **Total LOC**: 421
- **Standard matmul_block LOC**: ~55 (the `matmul_blocks` function lines 226-281 follows a sub-blocked pattern)
- **Custom LOC**: ~366 (copy_block, add_bias_block, add_bias_and_addcmul_block, ternary ops, L1_ACC, dynamic subblocks)
- **Missing features**:
  1. **Out-of-order packing**: `matmul_blocks` uses `pack_tile<true>(write_dst_index, out_cb, out_tile_id)` with computed absolute tile IDs.
  2. **PACKER_L1_ACC with different pattern**: `llk_pack_reconfig_l1_acc(1)` after block 0, `llk_pack_reconfig_l1_acc(0)` at end. No spill/reload; accumulation is fully in L1.
  3. **FUSE_TERNARY (addcmul)**: Post-matmul ternary operation: `output = matmul_result * ternary_b * scalar + ternary_a`. A multi-stage fusion not expressible via PostComputeFn.
  4. **FUSE_BIAS**: Same row-broadcast bias addition as production kernel, but also combined with FUSE_TERNARY in `add_bias_and_addcmul_block`.
  5. **Dynamic subblock sizes**: `current_M_block_tiles` and `current_subblock_h` adjusted at runtime based on tile range boundaries (`M_end_tile`, `N_end_tile`).
  6. **LLK bug workaround**: `unary_bcast`/`mul_binary_tile` path for FP32 vs `mul_tiles_bcast` for BF16, selected via `TERNARY_B_IS_FLOAT32` define.
  7. **mm_init instead of mm_block_init**: Uses `mm_init` (the tile-level init) not `mm_block_init`.
  8. **In-place CB write**: `pack_tile(DST_ID, intermediate_cb)` writes back to the same CB that was read, with explicit `cb_pop_front` + `cb_reserve_back` + `cb_push_back` for packer/unpacker sync.
- **Feature interactions**: FUSE_BIAS + FUSE_TERNARY: bias is applied first, then ternary multiply+add. L1_ACC applies only during the K-blocking matmul phase, not the post-matmul phases.
- **Composability assessment**: The matmul core (`matmul_blocks`) could use a helper. The post-matmul pipeline (bias, ternary) is too specialized for the helper to absorb. A composable approach where the matmul helper handles K-blocking + L1_ACC and the caller handles post-processing would work.

---

### 6. transformer_group_attn_matmul.cpp

- **File**: `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/compute/transformer_group_attn_matmul.cpp`
- **Total LOC**: 180
- **Standard matmul_block LOC**: 0 (uses `matmul_tiles` API, not `matmul_block`)
- **Custom LOC**: ~180 (tile-by-tile matmul, pack_untilize_dest, retilize, group attention head skipping)
- **Missing features**:
  1. **matmul_tiles API support**: Uses `matmul_tiles` (tile-by-tile LLK) not `matmul_block`. The helper only wraps `matmul_block`.
  2. **pack_untilize_dest**: Intermediate output packed as untilized tiles.
  3. **Re-tilize pipeline**: After matmul+untilize, the result is re-tilized via `tilize_block` for the output CB.
  4. **Group attention head skipping**: `cb_pop_front(num_kv_heads_skip)` before matmul, `cb_pop_front(num_kv_heads_remaining)` after.
  5. **experimental::CircularBuffer API**: Object-oriented CB interface instead of raw function calls.
  6. **Per-tile matmul with explicit indexing**: Tile-by-tile matmul with manually computed indices (not sub-blocked).
  7. **Commented-out matmul_block code**: There's commented-out code suggesting matmul_block was tried but disabled due to "didt" issues.
- **Feature interactions**: None significant — features are orthogonal.
- **Composability assessment**: This kernel uses `matmul_tiles`, which was already removed as a helper per PR feedback (only one call site: bmm.cpp). The matmul_block helper doesn't apply here. This kernel is better left as inline code per the PR feedback principle: "don't create a helper for a pattern with only one call site."

---

### 7. deepseek/mla/matmul_wo/compute.cpp

- **File**: `ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/compute.cpp`
- **Total LOC**: 105
- **Standard matmul_block LOC**: ~15 (matmul_block call lines 78-89)
- **Custom LOC**: ~90 (cross-block accumulation, custom tile offset, pipeline drain)
- **Missing features**:
  1. **Cross-block DST accumulation without spill/reload**: `tile_regs_acquire()` is called once before the entire multi-block loop (line 74), and `tile_regs_commit()` after all blocks. The helper does acquire/release per sub-block with spill/reload between K-blocks.
  2. **Custom per-core tile index offset**: `in0_index_base` computed from `K_TILES_PER_CORE_A[dram_bank_id]` — different cores read different K slices.
  3. **Pipeline drain**: Dummy final `cb_wait_front` + `cb_pop_front` at the end.
  4. **No sub-block iteration**: Single rt_dim=1, ct_dim=7 — no in0/in1 sub-block loops.
- **Feature interactions**: None — the features are independent.
- **Composability assessment**: This kernel's matmul pattern is fundamentally different from the helper's sub-blocked pattern. It accumulates across many blocks in DST without spilling. The helper's value (managing spill/reload, sub-block iteration) doesn't apply here. Better left as inline code — it's only 105 lines and the matmul is straightforward.

---

### 8. deepseek/moe/moe_gate_mm/compute.cpp

- **File**: `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm/device/kernels/compute.cpp`
- **Total LOC**: 377
- **Standard matmul_block LOC**: ~30 (two matmul patterns: send_core ct_dim=2, worker ct_dim=1)
- **Custom LOC**: ~347 (multi-core dispatch, sigmoid, copy_dest_values, custom bias, topk, top8_merge, out-of-order pack)
- **Missing features**:
  1. **Cross-block DST accumulation**: Same as matmul_wo — acquire once, accumulate across blocks.
  2. **Multi-core role dispatch**: `is_send_core` vs worker vs collector (COLLECTOR_CORE_ID) — entirely different compute paths per role.
  3. **binary_dest_reuse_tiles**: Add partial results from other cores into DST tiles in-place.
  4. **Custom SFPU operations**: `sigmoid_tile`, `sum_top2_tile`, `top4_tile`, `top8_tile`, `top8_merge` — none of which are matmul-related.
  5. **copy_dest_values**: Duplicate a DST tile for later use.
  6. **Custom bias via copy_tile + add_bias**: Not the standard `add_tiles_bcast_rows` pattern.
  7. **Out-of-order packing**: `pack_tile<true>(1, cb_s2c_out, 0)`.
  8. **transpose_wh_tile**: Tile transpose as part of the compute pipeline.
- **Feature interactions**: The matmul is a small initial phase; the rest is MoE-specific scoring/routing logic.
- **Composability assessment**: **NOT practical for matmul helper.** The matmul is ~30 lines in a 377-line kernel. The value-add of a helper is minimal. The kernel's complexity is in the post-matmul MoE routing logic, not the matmul itself.

---

### 9. ccl/moe_compute/device/kernels/compute.cpp

- **File**: `ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/compute.cpp`
- **Total LOC**: 343
- **Standard matmul_block LOC**: ~30 (two matmul stages: W0/W1 and W2)
- **Custom LOC**: ~313 (PACK-thread SFPU, ring sync, double buffer, pack_untilize, expert loop, metadata decode)
- **Missing features**:
  1. **PACK-thread SFPU operations**: `llk_math_eltwise_unary_sfpu_silu` and `llk_math_eltwise_binary_sfpu_binop` run on the PACK thread (via `PACK((...))` wrappers). This is a fundamentally different execution model from the helper's MATH-thread-only approach.
  2. **Two sequential matmul stages**: First matmul (in @ W0/W1), then SILU+eltwise_multiply, then second matmul (in2 @ W2). Different weight CBs, different dimensions.
  3. **Cross-block DST accumulation**: Same pattern as matmul_wo/moe_gate.
  4. **Double-buffered input**: `use_second_half_buffer` toggle changes `in0_index` base between halves.
  5. **Ring all-to-all synchronization**: `noc_semaphore_wait_min` for inter-core coordination.
  6. **pack_untilize_dest**: `pack_untilize_dest<4, 20>` for output.
  7. **Dynamic expert iteration**: `NUM_CHUNKS_PER_EXPERT[expert_id]` computed from runtime metadata.
  8. **PACK-thread TT_SETC16 register control**: Direct hardware register manipulation for DEST_TARGET switching.
- **Feature interactions**: The two matmul stages share DST state indirectly (through CBs), and the SILU+mul between them runs on the PACK thread while MATH may have already started other work.
- **Composability assessment**: **NOT practical for matmul helper.** The matmul is interleaved with PACK-thread SFPU operations and ring synchronization. The helper's single-function abstraction can't express the two-stage matmul-with-activation-in-between pattern, especially with PACK-thread execution.

---

### 10. topk_router_gpt/compute.cpp

- **File**: `ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/compute.cpp`
- **Total LOC**: 329
- **Standard matmul_block LOC**: ~15 (matmul_block call lines 103-113)
- **Custom LOC**: ~314 (sender/worker/collector roles, binary_dest_reuse, bias, topk HW instructions, softmax pipeline)
- **Missing features**:
  1. **Cross-block DST accumulation**: Accumulates across blocks without spill/reload.
  2. **Sender/worker/collector dispatch**: Three distinct compute paths depending on role.
  3. **binary_dest_reuse_tiles**: Add 2 sender partial tiles into matmul result in-place.
  4. **Bias via binary_dest_reuse**: Bias added as another `binary_dest_reuse_tiles<ELWADD>`.
  5. **TopK hardware instructions**: `topk_tile_init`, `topk_local_sort`, `topk_merge`, `topk_rebuild` — entirely unrelated to matmul.
  6. **Softmax pipeline**: max reduction → sub+exp → sum reduction → recip → mul_bcast. ~100 lines of non-matmul compute.
  7. **transpose_wh_tile** for topk input preparation.
  8. **Multiple data format reconfigurations** between pipeline stages.
- **Feature interactions**: None significant between matmul and post-matmul features.
- **Composability assessment**: **NOT practical for matmul helper.** The matmul is ~15 lines in a 329-line kernel. The complexity is topk + softmax, not matmul. Same pattern as moe_gate_mm — matmul is a small initial phase.

---

### 11. ccl/llama_all_gather_matmul_async/.../bmm_large_block_zm_fused_bias_activation_gathered.cpp

- **File**: `ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`
- **Total LOC**: 417
- **Standard matmul_block LOC**: ~50 (same core matmul loop as gathered variant #2)
- **Custom LOC**: ~367 (same gathered infrastructure + fabric integration)
- **Missing features**: Nearly identical to kernel #2 (gathered variant). Differences:
  1. Uses positional `get_compile_time_arg_val(N)` instead of `get_named_compile_time_arg_val("name")`.
  2. Uses `fill_array_with_next_n_args` instead of `fill_named_cb_array`.
  3. Different ring index direction: `(ring_idx - block + ring_size) % ring_size` (counterclockwise).
  4. Includes `edm_fabric/compile_time_arg_tmp.hpp` for fabric integration.
  5. Uses `input0_cb_id = in0_cb_id` for all blocks (no alternating CB source).
- **Feature interactions**: Same as kernel #2.
- **Composability assessment**: Same as kernel #2. A copy of the gathered variant adapted for the all-gather-matmul pipeline.

---

## Summary Table: Missing Features Ranked by Migration Coverage

| Missing Feature | Kernels It Would Unlock (fully or partially) | Estimated Complexity |
|----------------|----------------------------------------------|---------------------|
| **PACKER_L1_ACC support** (L1 accumulation in pack loop) | #1 (production), #2 (gathered), #3 (conv), #5 (minimal_matmul), #11 (llama_gathered) | **High** — fundamentally changes spill/reload strategy; 3 different L1_ACC patterns across kernels |
| **FUSE_BIAS with row broadcast** (post-matmul bias phase) | #1 (production), #3 (conv), #5 (minimal_matmul) | **Medium** — separate phase after matmul; proven pattern from llk3 attempt (but that attempt caused device hangs) |
| **Half-sync DST management** (tile_regs_acquire/commit/wait/release) | #1, #2, #3, #4, #5, #7, #8, #9, #10, #11 — all production and experimental kernels | **Low** — change helper to use tile_regs_* instead of acquire_dst/release_dst |
| **PACK_RELU toggle** (pack-level relu config) | #1 (production), #2 (gathered), #3 (conv), #11 (llama_gathered) | **Low** — `llk_pack_relu_config` calls at loop boundaries |
| **FP32_DEST_ACC_EN** (extra pack reconfig) | #1 (production), #2 (gathered), #3 (conv), #11 (llama_gathered) | **Low** — extra `pack_reconfig_data_format` before pack on last block |
| **untilize_out** (pack untilize output) | #1 (production), #2 (gathered), #3 (conv) | **Medium** — `reblock_and_untilize` or `pack_untilize_dest` post-pack phase |
| **SFPU activation with bias interaction** (apply after bias, not after matmul) | #1 (production), #3 (conv), #5 (minimal_matmul) | **Low** — adjust when PostComputeFn fires based on whether bias is fused |
| **Out-of-order packing** (pack_tile\<true\>) | #4 (SDPA), #5 (minimal_matmul), #8 (moe_gate), #9 (moe_compute) | **Medium** — requires different pack loop with absolute tile indexing |
| **Cross-block DST accumulation** (no spill/reload) | #7 (matmul_wo), #8 (moe_gate), #9 (moe_compute), #10 (topk_router) | **Medium** — fundamentally different from the helper's per-block acquire/release pattern |
| **Outer H/W dimension loops** | #1 (production) | **Low** — add outer loop params |
| **in0_transpose_tile** (pre-matmul A transpose) | #1 (production) | **Medium** — transpose phase with CB switching and L1_ACC reconfiguration |
| **SKIP_COMPUTE** (bypass matmul) | #1 (production) | **Low** — conditional skip around matmul loop |
| **Per-batch output CB arrays** | #2 (gathered), #11 (llama_gathered) | **Medium** — runtime CB selection instead of compile-time template param |
| **Dynamic K dim per block** | #2 (gathered), #11 (llama_gathered) | **Low** — runtime inner loop bound instead of fixed block_w |
| **Fused tilize input** | #3 (conv) | **Low for helper** — already handled by separate tilize helper; the matmul helper just needs to work after tilize |
| **matmul_tiles API support** | #6 (group_attn) | **N/A** — already removed per PR feedback; only 1 call site |
| **binary_dest_reuse_tiles** | #8 (moe_gate), #9 (moe_compute), #10 (topk_router) | **N/A** — not a matmul feature; these are eltwise additions of partial results |
| **PACK-thread SFPU** | #9 (moe_compute) | **N/A** — fundamentally different execution model; not a matmul helper concern |
| **Custom CB pointer management** | #2, #3, #11 | **N/A** — infrastructure concern, not a matmul helper feature |

---

## Key Observations

### 1. The production kernel (#1) is the highest-value target

Migrating `bmm_large_block_zm_fused_bias_activation.cpp` would validate the helper design against all major feature combinations. The features needed are: PACKER_L1_ACC, FP32_DEST_ACC_EN, FUSE_BIAS, SFPU activation, PACK_RELU, in0_transpose_tile, SKIP_COMPUTE, untilize_out, outer H/W loops, and half-sync DST. These features also cover conv_bmm_tilize (#3), which needs a subset.

### 2. The gathered variants (#2, #11) need infrastructure, not helper features

The gathered kernels share the same matmul core as #1 but add ring-distribution infrastructure (per-batch CBs, dynamic K dim, CB pointer management, sync primitives). These are caller-side concerns that shouldn't be absorbed into the helper. A composable helper that exposes the matmul core would let the gathered kernel call it within its ring loop.

### 3. DeepSeek/MoE/TopK kernels (#7-#10) don't benefit from the matmul helper

These kernels use matmul as a small initial phase (~15-30 lines) in much larger pipelines (350+ lines). The matmul pattern is also different: cross-block DST accumulation without spill/reload, single rt_dim=1 sub-block, no sub-block iteration. Attempting to force-fit them into the helper would add complexity. They're better left as inline code.

### 4. SDPA (#4) is fundamentally incompatible

The SDPA kernel has its own `blocked_matmul_and_pack` function with out-of-order packing, architecture-specific matmul calls, and hardware semaphore pipelining. It's a self-contained building block already. Wrapping it in the matmul helper would add overhead without benefit.

### 5. group_attn_matmul (#6) uses matmul_tiles, not matmul_block

This kernel uses the tile-by-tile `matmul_tiles` API, which was already removed as a helper per PR feedback. It has only one call site. Leave as inline.

### 6. The llk3 fused_bias attempt failed for the right reasons

The `matmul_block_fused_bias` helper on llk3 duplicated the entire matmul loop from `matmul_block_helpers.inl` and added a bias phase. This monolithic approach:
- Used param structs (removed per PR feedback)
- Used InitUninitMode/ReconfigureRegisterDatatypeMode enums (removed per PR feedback)
- Did not support PACKER_L1_ACC (which changes the spill/reload fundamentally)
- Used full-sync DST (`tile_regs_acquire/commit` + `tile_regs_wait/release` in pairs, but within a monolithic function)
- Caused device hangs — likely because the `cb_pop_front(interm_cb)` before `cb_reserve_back(out_cb)` ordering is critical when interm and out share L1, and the monolithic approach didn't get this right for all code paths.

### 7. Three tiers of migration feasibility

**Tier 1 — High value, feasible with improved helpers:**
- #1 (production kernel) — PRIMARY TARGET
- #3 (conv_bmm_tilize) — shares most features with #1

**Tier 2 — Feasible but requires caller-side infrastructure:**
- #2 (gathered) — needs composable helper within ring loop
- #11 (llama_gathered) — same as #2
- #5 (minimal_matmul) — needs composable helper + post-matmul ternary

**Tier 3 — Not practical for helper migration:**
- #4 (SDPA) — matmul is one stage in a complex pipeline
- #6 (group_attn) — uses matmul_tiles, only 1 call site
- #7 (matmul_wo) — different accumulation pattern, only 105 lines
- #8 (moe_gate) — matmul is ~30 lines in 377-line kernel
- #9 (moe_compute) — PACK-thread SFPU, two matmul stages
- #10 (topk_router) — matmul is ~15 lines in 329-line kernel
