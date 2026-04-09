// Migration Examples: Mode 2 (Semi-Automatic) -- Fused BMM kernels
// Call sites: B1, B2, B3, B16
//
// These call sites use the semi-automatic pattern because they interleave
// custom fusion operations (bias addition, SFPU activation, untilize) between
// the matmul accumulation and the final pack. MatmulOp provides begin_subblock(),
// accumulate(), end_to_output(), end_to_partials(), and reload_partials() to
// handle DST ownership and spill/reload, while the caller inserts fusion code.
//
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/matmul_op.h"
#include "api/compute/bcast.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"
#include "experimental/circular_buffer.h"

// ============================================================================
// B1: bmm_large_block_zm_fused_bias_activation.cpp
// Source: ttnn/.../matmul/.../bmm_large_block_zm_fused_bias_activation.cpp
//
// This is the most complex production matmul kernel. It includes:
//   - Block-mode matmul with subblocking (in0_num_subblocks * in1_num_subblocks)
//   - K-dimension accumulation with spill/reload (num_blocks_inner_dim > 1)
//   - Optional in0 transpose (transpose_tile_block before matmul)
//   - Fused bias addition via add_tiles_bcast_rows (separate phase after matmul)
//   - Fused SFPU activation (applied in DST before pack)
//   - Optional untilize output path
//   - PACKER_L1_ACC support (orthogonal to MatmulOp)
//   - Sparse batch gating (get_batch_from_reader)
//
// The migration replaces:
//   1. mm_block_init(...) -> mm.init()
//   2. The reload_from_cb_to_dst() helper -> mm.reload_partials()
//   3. tile_regs_acquire() -> mm.begin_subblock()
//   4. The inner_dim matmul_block loop -> mm.accumulate()
//   5. The pack-to-partials path -> mm.end_to_partials()
//   6. The pack-to-output path (without fusion) -> mm.end_to_output()
//
// What is NOT replaced:
//   - FUSE_BIAS phase (separate loop after all K blocks complete)
//   - SFPU activation (applied between accumulate and pack)
//   - PACKER_L1_ACC configuration
//   - in0 transpose preprocessing
//   - Untilize output path
//   - pack_tile_block calls (custom pack pattern)
// ============================================================================
namespace b1_fused_bias_activation {

void kernel_main() {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(6);
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(7);
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(8);
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(9);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(11);
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t batch = get_compile_time_arg_val(13);
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(14);
    constexpr bool untilize_out = get_compile_time_arg_val(15);

    constexpr uint32_t in0_cb_id = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in1_cb_id = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t out_cb_id = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t mm_partials_cb_id = get_named_compile_time_arg_val("cb_intermed0");
    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? mm_partials_cb_id : out_cb_id;

#ifdef FUSE_BIAS
    constexpr uint32_t bias_cb_id = get_named_compile_time_arg_val("cb_bias");
    constexpr uint32_t mm_out_cb_id = mm_partials_cb_id;
#else
    constexpr uint32_t mm_out_cb_id = untilize_mode_out_cb_id;
#endif

#ifdef IN1_TRANSPOSE_TILE
    constexpr bool in1_transpose_tile = true;
#else
    constexpr bool in1_transpose_tile = false;
#endif

    constexpr bool spill = num_blocks_inner_dim > 1;

    experimental::CircularBuffer in0_cb(in0_cb_id);
    experimental::CircularBuffer in1_cb(in1_cb_id);
    experimental::CircularBuffer out_cb(out_cb_id);
    experimental::CircularBuffer mm_partials_cb(mm_partials_cb_id);
    experimental::CircularBuffer mm_out_cb(mm_out_cb_id);

    // --- NEW: MatmulOp configuration ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = in0_cb_id,
        .in1_cb_id = in1_cb_id,
        .out_cb_id = mm_out_cb_id,
        .ct_dim = out_subblock_w,
        .rt_dim = out_subblock_h,
        .kt_dim = in0_block_w,
        .transpose = in1_transpose_tile,
        .partials_cb_id = spill ? mm_partials_cb_id : 0u,
    };
    ckernel::BlockMatmulOp mm(cfg);
    mm.init();  // replaces mm_block_init(...)
    // --- END NEW ---

    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                bool enable_reload = false;
                uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

                // UNCHANGED: PACK_RELU, pack_reconfig_data_format setup
#ifdef PACK_RELU
                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((llk_pack_relu_config(ReluType::NO_RELU)));
                }
#endif
                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((pack_reconfig_data_format(mm_partials_cb_id)));
                }

                for (uint32_t block = 0; block < num_blocks_inner_dim; block++) {
                    bool last_out = block == (num_blocks_inner_dim - 1);

                    // UNCHANGED: in0 transpose preprocessing, PACK_RELU config
#if not defined FUSE_BIAS and defined PACK_RELU
                    if (last_out) {
                        PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
                    }
#endif

                    // UNCHANGED: CB waits for the full block
                    in0_cb.wait_front(in0_block_num_tiles);
                    in1_cb.wait_front(in1_block_num_tiles);

                    int in0_index_subblock_offset = 0;
                    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                        int in1_index_subblock_offset = 0;
                        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                            // --- NEW: begin_subblock replaces tile_regs_acquire ---
                            mm.begin_subblock();

                            // --- NEW: reload_partials replaces reload_from_cb_to_dst ---
                            if (enable_reload) {
                                mm.reload_partials(out_subblock_num_tiles);
                            }

#ifndef SKIP_COMPUTE
                            // --- NEW: accumulate replaces the inner_dim matmul_block loop ---
                            mm.accumulate(
                                in0_index_subblock_offset,
                                in1_index_subblock_offset,
                                /*dst_index_start=*/0,
                                in0_block_w,   // inner_dim
                                in1_block_w);  // in1_stride
                                               // --- END NEW ---
#endif

                            if (last_out) {
                                // UNCHANGED: optional SFPU activation in DST
#if not defined FUSE_BIAS and defined SFPU_OP_INIT_ACTIVATION
                                for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                    SFPU_OP_FUNC_ACTIVATION
                                }
#endif
                                // Pack to output -- uses custom pack path (pack_tile_block),
                                // so we do NOT use mm.end_to_output(). Instead, manual DST
                                // commit/wait/pack/release.
                                tile_regs_commit();
                                mm_out_cb.reserve_back(out_subblock_num_tiles);
                                tile_regs_wait();

                                // UNCHANGED: FP32_DEST_ACC_EN / PACKER_L1_ACC reconfig
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                                PACK((pack_reconfig_data_format(mm_out_cb_id)));
#endif
#ifdef PACKER_L1_ACC
                                PACK((llk_pack_reconfig_l1_acc(0)));
#endif
                                uint32_t start_dst_index = 0;
                                pack_tile_block(start_dst_index, mm_out_cb_id, out_subblock_num_tiles);
                                tile_regs_release();
                                mm_out_cb.push_back(out_subblock_num_tiles);
                            } else {
                                // Spill to partials -- uses custom pack path (pack_tile_block).
                                tile_regs_commit();
                                if (block == 0) {
                                    out_cb.reserve_back(out_num_tiles_to_wait);
                                    out_num_tiles_to_wait += out_subblock_num_tiles;
                                }
                                mm_partials_cb.reserve_back(out_subblock_num_tiles);
                                tile_regs_wait();

#ifdef PACKER_L1_ACC
                                // UNCHANGED: L1_ACC configuration per block
                                if (block == 0) {
                                    PACK((llk_pack_reconfig_l1_acc(0)));
                                } else if (block == 1) {
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                }
#endif
                                uint32_t start_dst_index = 0;
                                pack_tile_block(start_dst_index, mm_partials_cb_id, out_subblock_num_tiles);
                                tile_regs_release();
                                mm_partials_cb.push_back(out_subblock_num_tiles);
                            }

                            in1_index_subblock_offset += out_subblock_w;
                        }
                        in0_index_subblock_offset += in0_subblock_num_tiles;
                    }

                    // UNCHANGED: PACKER_L1_ACC enable_reload logic
                    if constexpr (spill) {
                        enable_reload = true;
                    }

                    in0_cb.pop_front(in0_block_num_tiles);
                    in1_cb.pop_front(in1_block_num_tiles);
                }

                // UNCHANGED: FUSE_BIAS phase (separate loop after all K blocks)
#ifdef FUSE_BIAS
                // ... bias addition via add_tiles_bcast_rows ...
                // ... SFPU activation after bias ...
                // ... pack to untilize_mode_out_cb ...
#endif

                // UNCHANGED: untilize output path
                if constexpr (untilize_out) {
                    // ... reblock_and_untilize ...
                }

                // UNCHANGED: Reinit matmul for next output block
                if constexpr (batch > 1 || num_blocks_w_dim > 1 || num_blocks_h_dim > 1) {
                    // --- NEW: init_short replaces mm_block_init_short ---
                    mm.init_short();
                    // --- END NEW ---
                }
            }
        }
    }
}

}  // namespace b1_fused_bias_activation

// ============================================================================
// B2: bmm_large_block_zm_fused_bias_activation_gathered.cpp
// Source: ttnn/.../matmul/.../bmm_large_block_zm_fused_bias_activation_gathered.cpp
//
// This kernel is structurally identical to B1 in its matmul core logic.
// The only difference is how the input CBs are managed (gathered input).
// The matmul portion migrates identically to B1.
//
// MIGRATION: Same as B1 above. The gathered CB management is UNCHANGED.
// ============================================================================
namespace b2_gathered {
// Migration is identical to B1. The additional gathered CB logic (reading from
// multiple input buffers, managing gathered semaphores) is orthogonal to MatmulOp.
// See B1 above for the complete matmul migration pattern.
}  // namespace b2_gathered

// ============================================================================
// B3: conv_bmm_tilize.cpp
// Source: ttnn/.../conv/conv2d/.../conv_bmm_tilize.cpp
//
// Same inner matmul loop as B1, but with:
//   - Tilize preprocessing (tilize_in before matmul blocks)
//   - SFPU activation applied per-subblock on last inner dim block
//   - PACKER_L1_ACC with different accumulation gating
//
// The matmul_block loop and reload pattern are identical to B1.
// ============================================================================
namespace b3_conv_bmm_tilize {

void kernel_main_snippet() {
    // Compile-time args (same structure as B1)
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(1);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(2);
    constexpr uint32_t out_subblock_num_tiles = out_subblock_w * out_subblock_h;
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(3);

    constexpr uint32_t mm_in0_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t matmul_partials_cb = tt::CBIndex::c_24;

    // --- NEW: MatmulOp configuration ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = mm_in0_cb_id,
        .in1_cb_id = in1_cb_id,
        .out_cb_id = matmul_partials_cb,  // B3 packs to partials CB always
        .ct_dim = out_subblock_w,
        .rt_dim = out_subblock_h,
        .kt_dim = in0_block_w,
        .transpose = false,  // conv2d does not transpose in1
        .partials_cb_id = matmul_partials_cb,
    };
    ckernel::BlockMatmulOp mm(cfg);
    mm.init();
    // --- END NEW ---

    // Inside the subblock loop (same pattern as B1):
    // The key difference from B1 is that B3 always does:
    //   tile_regs_acquire() (or reload + acquire)
    //   matmul_block loop
    //   optional SFPU on last_inner_dim_block
    //   tile_regs_commit() + pack to curr_matmul_out_cb

    // Example of the inner subblock (one iteration of in0_sub x in1_sub):
    uint32_t in0_index_subblock_offset = 0;
    uint32_t in1_index_subblock_offset = 0;
    bool enable_reload = false;

    if (enable_reload) {
        // --- NEW: reload_partials replaces manual copy_tile_to_dst_init_short_with_dt
        //          + copy_block_matmul_partials + mm_block_init_short_with_dt ---
        mm.begin_subblock();
        mm.reload_partials(out_subblock_num_tiles);
    } else {
        mm.begin_subblock();
    }

    // --- NEW: accumulate replaces inner_dim matmul_block loop ---
    mm.accumulate(
        in0_index_subblock_offset,
        in1_index_subblock_offset,
        /*dst_index_start=*/0,
        in0_block_w,
        in1_block_w);
    // --- END NEW ---

    // UNCHANGED: SFPU activation and custom pack logic follow
}

}  // namespace b3_conv_bmm_tilize

// ============================================================================
// B16: llama_all_gather_matmul -- gathered variant
// Source: ttnn/.../ccl/llama_all_gather_matmul_async/.../bmm_large_block_zm_fused_bias_activation_gathered.cpp
//
// This is the same kernel as B2, just in a different directory for the
// llama_all_gather pipeline. The matmul migration is identical to B1/B2.
// ============================================================================
namespace b16_llama_all_gather {
// Migration is identical to B1/B2. See above.
}  // namespace b16_llama_all_gather
