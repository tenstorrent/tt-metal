// Migration Examples: Mode 1 (Low-Level Block) -- MOE operations
// Call sites: B8, B11, B12, B13, B14
//
// MOE (Mixture of Experts) kernels use matmul_block with various ct_dim values
// (1, 2, 4, 7) and complex control flow including ring accumulation, bias via
// ones-tile matmul, SwiGLU activation, and untilize packing. All require
// Mode 1 due to the custom control flow surrounding each matmul_block call.
//
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/matmul_op.h"
#include "experimental/circular_buffer.h"

// ============================================================================
// B8: topk_router_gpt -- 1x1x1 tile-by-tile block matmul
// Source: ttnn/.../experimental/topk_router_gpt/.../compute.cpp
//
// ORIGINAL CODE:
//   mm_block_init(cb_input, cb_weight, cb_local_out, 0, 1, 1, 1);
//   tile_regs_acquire();
//   while (tiles_done < num_k_tiles) {
//       cb_wait_front(cb_input, block); cb_wait_front(cb_weight, block);
//       for (k < block) {
//           matmul_block(cb_input, cb_weight, k, k, 0, false, 1, 1, 1);
//       }
//       cb_pop_front(cb_input, block); cb_pop_front(cb_weight, block);
//   }
//   // Send or compute path: pack result
//
// Uses ct_dim=1, rt_dim=1, kt_dim=1. Two code paths: send core (ct=2) and
// compute core (ct=1). The send core uses pack_tile<true> (out-of-order).
// ============================================================================
namespace b8_topk_router {

void kernel_main_snippet() {
    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_weight = tt::CBIndex::c_1;
    constexpr uint32_t cb_local_out = tt::CBIndex::c_2;

    uint32_t num_k_tiles = 100;  // from runtime args
    constexpr uint32_t BLOCK_SIZE = 16;
    bool is_sender = false;  // from runtime args

    // --- NEW: Separate MatmulOp for send vs compute path ---
    // Send core uses ct_dim=2, compute core uses ct_dim=1
    if (is_sender) {
        ckernel::MatmulOpConfig cfg{
            .in0_cb_id = cb_input,
            .in1_cb_id = cb_weight,
            .out_cb_id = cb_local_out,
            .ct_dim = 2,
            .rt_dim = 1,
            .kt_dim = 1,
        };
        ckernel::BlockMatmulOp mm(cfg);
        mm.init();

        tile_regs_acquire();

        uint32_t tile_index = 2 * 76;  // offset for send core
        uint32_t tiles_done = 0;
        while (tiles_done < num_k_tiles) {
            uint32_t block = num_k_tiles - tiles_done;
            if (block > BLOCK_SIZE) {
                block = BLOCK_SIZE;
            }

            cb_wait_front(cb_weight, block);

            for (uint32_t k = 0; k < block; k += 2) {
                // --- NEW: mm.matmul replaces matmul_block ---
                mm.matmul(tile_index++, k, 0);
                // --- END NEW ---
            }
            cb_pop_front(cb_weight, block);
            tiles_done += block;
        }

        tile_regs_commit();
        cb_reserve_back(cb_local_out, 1);
        tile_regs_wait();
        // Out-of-order pack: send each subblock tile separately
        pack_tile<true>(1, cb_local_out, 0);
        cb_push_back(cb_local_out, 1);
        cb_reserve_back(cb_local_out, 1);
        pack_tile<true>(0, cb_local_out, 0);
        cb_push_back(cb_local_out, 1);
        tile_regs_release();
    } else {
        // Compute core: ct_dim=1
        ckernel::MatmulOpConfig cfg{
            .in0_cb_id = cb_input,
            .in1_cb_id = cb_weight,
            .out_cb_id = cb_local_out,
            .ct_dim = 1,
            .rt_dim = 1,
            .kt_dim = 1,
        };
        ckernel::BlockMatmulOp mm(cfg);
        mm.init();

        tile_regs_acquire();

        uint32_t tile_index = 0;
        uint32_t tiles_done = 0;
        while (tiles_done < num_k_tiles) {
            uint32_t block = num_k_tiles - tiles_done;
            if (block > BLOCK_SIZE) {
                block = BLOCK_SIZE;
            }

            cb_wait_front(cb_input, block);
            cb_wait_front(cb_weight, block);

            for (uint32_t k = 0; k < block; k++) {
                // --- NEW: mm.matmul replaces matmul_block ---
                mm.matmul(tile_index++, k, 0);
                // --- END NEW ---
            }

            cb_pop_front(cb_input, block);
            cb_pop_front(cb_weight, block);
            tiles_done += block;
        }

        // UNCHANGED: binary_dest_reuse_tiles for partial addition, then pack
        tile_regs_commit();
        // ... add partials, pack ...
        tile_regs_release();
    }
}

}  // namespace b8_topk_router

// ============================================================================
// B11: MOE gate matmul -- ct_dim=2 (send core) or ct_dim=1 (compute core)
// Source: ttnn/.../experimental/deepseek/moe/moe_gate_mm/.../compute.cpp
//
// ORIGINAL CODE (send core path, lines 96-156):
//   mm_block_init(cb_s2c_in, cb_r2c_w, cb_s2c_out, false, 2, 1, 1);
//   tile_regs_acquire();
//   for (block_id) {
//       cb_wait_front(cb_r2c_w, w_tiles_per_block);
//       for (tile_id = 0; tile_id < w_tiles_per_block; tile_id += 2) {
//           matmul_block(cb_s2c_in, cb_r2c_w, tile_index++, tile_id, 0, false, 2, 1, 1);
//       }
//       cb_pop_front(cb_r2c_w, w_tiles_per_block);
//   }
//   tile_regs_commit(); tile_regs_wait();
//   pack_tile<true>(1, cb_s2c_out, 0);
//   pack_tile<true>(0, cb_s2c_out, 0);
//   tile_regs_release();
//
// ORIGINAL CODE (compute core path, lines 160-210):
//   mm_block_init(cb_s2c_in, cb_r2c_w, cb_s2c_out, false, 1, 1, 1);
//   // same loop but ct_dim=1, sequential pack
//
// 4 matmul_block call sites total (2 per path, main blocks + last block).
// ============================================================================
namespace b11_moe_gate_mm {

void kernel_main_snippet() {
    constexpr uint32_t cb_s2c_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_r2c_w = tt::CBIndex::c_1;
    constexpr uint32_t cb_s2c_out = tt::CBIndex::c_2;
    constexpr uint32_t cb_c2w_rdy = tt::CBIndex::c_3;

    uint32_t w_num_blocks = 10;           // from compile time args
    uint32_t w_tiles_per_block = 8;       // from compile time args
    uint32_t w_tiles_per_block_last = 4;  // from compile time args
    bool is_send_core = true;             // from runtime args

    if (is_send_core) {
        // --- NEW: MatmulOp with ct_dim=2 ---
        ckernel::MatmulOpConfig cfg{
            .in0_cb_id = cb_s2c_in,
            .in1_cb_id = cb_r2c_w,
            .out_cb_id = cb_s2c_out,
            .ct_dim = 2,
            .rt_dim = 1,
            .kt_dim = 1,
        };
        ckernel::BlockMatmulOp mm(cfg);
        mm.init();
        // --- END NEW ---

        tile_regs_acquire();

        uint32_t tile_index = 2 * 76;  // send core offset
        for (uint32_t block_id = 0; block_id < w_num_blocks; ++block_id) {
            cb_wait_front(cb_r2c_w, w_tiles_per_block);
            for (uint32_t tile_id = 0; tile_id < w_tiles_per_block; tile_id += 2) {
                // --- NEW: mm.matmul replaces matmul_block ---
                mm.matmul(tile_index++, tile_id, 0);
                // --- END NEW ---
            }
            cb_pop_front(cb_r2c_w, w_tiles_per_block);
        }

        // Last block
        cb_wait_front(cb_r2c_w, w_tiles_per_block);
        for (uint32_t tile_id = 0; tile_id < w_tiles_per_block_last; tile_id += 2) {
            mm.matmul(tile_index++, tile_id, 0);
        }
        cb_pop_front(cb_r2c_w, w_tiles_per_block);

        tile_regs_commit();
        tile_regs_wait();

        // UNCHANGED: Out-of-order pack for send core
        cb_reserve_back(cb_c2w_rdy, 1);
        pack_tile<true>(1, cb_s2c_out, 0);
        cb_push_back(cb_c2w_rdy, 1);

        cb_reserve_back(cb_c2w_rdy, 1);
        pack_tile<true>(0, cb_s2c_out, 0);
        cb_push_back(cb_c2w_rdy, 1);

        tile_regs_release();
    }
    // Compute core path: identical but ct_dim=1
}

}  // namespace b11_moe_gate_mm

// ============================================================================
// B12: MLA matmul_wo -- ct_dim=7, K tiles streamed in blocks
// Source: ttnn/.../experimental/deepseek/mla/matmul_wo/.../compute.cpp
//
// ORIGINAL CODE:
//   mm_block_init(cb_s2c_in, cb_r2c_w, cb_c2w_out, false, 7, 1, 1);
//   for (iter_id) {
//       tile_regs_acquire();
//       for (block_id) {
//           cb_wait_front(cb_r2c_w, w_tiles_per_block);
//           for (k = 0; k < w_tiles_per_block; k += 7) {
//               matmul_block(cb_s2c_in, cb_r2c_w, in0_index++, k, 0, false, 7, 1, 1);
//           }
//           cb_pop_front(cb_r2c_w, w_tiles_per_block);
//       }
//       tile_regs_commit(); tile_regs_wait();
//       pack_tile_block(0, cb_c2w_out, num_n_tiles_per_iter);
//       tile_regs_release();
//   }
//
// ct_dim=7 means each matmul_block call processes 7 output tiles in the
// column dimension. The in1 tiles are arranged in groups of 7.
// ============================================================================
namespace b12_mla_matmul_wo {

void kernel_main() {
    constexpr uint32_t cb_s2c_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_r2c_w = tt::CBIndex::c_1;
    constexpr uint32_t cb_c2w_out = tt::CBIndex::c_2;

    uint32_t num_iters = 4;             // from compile time args
    uint32_t num_blocks_per_iter = 2;   // from compile time args
    uint32_t w_tiles_per_block = 14;    // from compile time args
    uint32_t num_n_tiles_per_iter = 7;  // from compile time args
    uint32_t dram_bank_id = 0;          // from runtime args
    uint32_t in0_index_base = 0;        // computed from dram_bank_id

    // --- NEW: MatmulOp with ct_dim=7 ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = cb_s2c_in,
        .in1_cb_id = cb_r2c_w,
        .out_cb_id = cb_c2w_out,
        .ct_dim = 7,
        .rt_dim = 1,
        .kt_dim = 1,
    };
    ckernel::BlockMatmulOp mm(cfg);
    mm.init();  // replaces mm_block_init(...)
    // --- END NEW ---

    for (uint32_t iter_id = 0; iter_id < num_iters; ++iter_id) {
        uint32_t in0_index = in0_index_base;

        tile_regs_acquire();
        for (uint32_t block_id = 0; block_id < num_blocks_per_iter; ++block_id) {
            cb_wait_front(cb_r2c_w, w_tiles_per_block);

            for (uint32_t k = 0; k < w_tiles_per_block; k += 7) {
                // --- NEW: mm.matmul replaces matmul_block ---
                mm.matmul(in0_index++, k, 0);
                // --- END NEW ---
            }
            cb_pop_front(cb_r2c_w, w_tiles_per_block);
        }

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_c2w_out, num_n_tiles_per_iter);
        pack_tile_block(0, cb_c2w_out, num_n_tiles_per_iter);
        cb_push_back(cb_c2w_out, num_n_tiles_per_iter);
        tile_regs_release();
    }
}

}  // namespace b12_mla_matmul_wo

// ============================================================================
// B13: CCL MOE compute -- ct_dim=4, W0/W1 + W2 matmul with SwiGLU
// Source: ttnn/.../experimental/ccl/moe_compute/.../compute.cpp
//
// ORIGINAL CODE (W0/W1 phase, lines 188-236):
//   mm_block_init(cb_s2c_in, cb_r2c_w0_w1, cb_s2c_in2, false, 4, 1, 1);
//   for (tile_id += 2) {
//       tile_regs_acquire();
//       for (block_id) {
//           cb_wait_front(cb_r2c_w0_w1, w0_w1_tiles_per_block);
//           for (k = 0; k < tiles_per_block; k += 4) {
//               matmul_block(cb_s2c_in, cb_r2c_w0_w1, in0_index++, k, 0, false, 4, 1, 1);
//           }
//           cb_pop_front(cb_r2c_w0_w1, w0_w1_tiles_per_block);
//       }
//       tile_regs_commit(); tile_regs_wait();
//       // SiLU activation + eltwise multiply (SwiGLU)
//       pack_tile<true>(...);
//       tile_regs_release();
//   }
//
// ORIGINAL CODE (W2 phase, lines 254-284):
//   Same pattern but reading from cb_s2c_in2 and cb_r2c_w2, with inter-step
//   synchronization (dm1_tiles_remaining tracking).
//
// 2 matmul_block call sites (W0/W1 and W2).
// ============================================================================
namespace b13_ccl_moe_compute {

void kernel_main_snippet() {
    constexpr uint32_t cb_s2c_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_r2c_w0_w1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_s2c_in2 = tt::CBIndex::c_2;
    constexpr uint32_t cb_r2c_w2 = tt::CBIndex::c_3;

    uint32_t w0_w1_tiles_per_block = 8;
    uint32_t w0_w1_blocks_per_two_elt_tile = 4;
    uint32_t tiles_per_step = 8;
    bool use_second_half_buffer = false;
    uint32_t num_w0_w1_tiles_h = 16;

    // --- NEW: MatmulOp for W0/W1 phase (ct_dim=4) ---
    ckernel::MatmulOpConfig w0_w1_cfg{
        .in0_cb_id = cb_s2c_in,
        .in1_cb_id = cb_r2c_w0_w1,
        .out_cb_id = cb_s2c_in2,
        .ct_dim = 4,
        .rt_dim = 1,
        .kt_dim = 1,
    };
    ckernel::BlockMatmulOp mm_w0_w1(w0_w1_cfg);
    mm_w0_w1.init();
    // --- END NEW ---

    // W0/W1 matmul phase
    for (uint32_t tile_id = 0; tile_id < tiles_per_step; tile_id += 2) {
        uint32_t in0_index = use_second_half_buffer ? num_w0_w1_tiles_h : 0;

        tile_regs_acquire();
        for (uint32_t block_id = 0; block_id < w0_w1_blocks_per_two_elt_tile; ++block_id) {
            cb_wait_front(cb_r2c_w0_w1, w0_w1_tiles_per_block);

            for (uint32_t k = 0; k < w0_w1_tiles_per_block; k += 4) {
                // --- NEW: mm_w0_w1.matmul replaces matmul_block ---
                mm_w0_w1.matmul(in0_index++, k, 0);
                // --- END NEW ---
            }
            cb_pop_front(cb_r2c_w0_w1, w0_w1_tiles_per_block);
        }

        tile_regs_commit();

        // UNCHANGED: SwiGLU activation (SiLU + eltwise mul) on packer thread
        // PACK(TTI_SEMWAIT(...)); PACK(TT_SETC16(...));
        // PACK((llk_math_eltwise_unary_sfpu_silu<true, false>(0)));
        // PACK((llk_math_eltwise_unary_sfpu_silu<true, false>(2)));
        // PACK((llk_math_eltwise_binary_sfpu_binop<true, MUL>(0, 1, 0)));
        // PACK((llk_math_eltwise_binary_sfpu_binop<true, MUL>(2, 3, 2)));
        // PACK(TTI_STALLWAIT(...));
        // pack_tile<true>(0, cb_s2c_in2, tile_id);
        // pack_tile<true>(2, cb_s2c_in2, tile_id + 1);
        tile_regs_release();
    }

    // --- NEW: Separate MatmulOp for W2 phase (same ct_dim=4 but different CBs) ---
    ckernel::MatmulOpConfig w2_cfg{
        .in0_cb_id = cb_s2c_in2,
        .in1_cb_id = cb_r2c_w2,
        .out_cb_id = cb_s2c_in2,  // reuses same CB
        .ct_dim = 4,
        .rt_dim = 1,
        .kt_dim = 1,
    };
    ckernel::BlockMatmulOp mm_w2(w2_cfg);
    // NOTE: No separate init needed -- the MATH engine config is the same
    // (ct_dim=4, rt_dim=1, kt_dim=1). init_short() after data format reconfig.
    // --- END NEW ---

    // W2 matmul phase (with inter-step synchronization)
    // ... same pattern, using mm_w2.matmul() instead of matmul_block ...
}

}  // namespace b13_ccl_moe_compute

// ============================================================================
// B14: CCL MOE GPT -- ct_dim=4, 8 matmul_block calls, bias via ones tile
// Source: ttnn/.../experimental/ccl/moe_gpt/.../compute.cpp (lines 190-480)
//
// This is the most complex Mode 1 block-mode kernel. It has:
//   - W0/W1 matmul with bias via ones-tile matmul (matmul(ones, bias_row))
//   - SwiGLU activation (sfpu_swiglu on packer thread)
//   - W2 matmul with bias via same ones-tile pattern
//   - Untilize output via pack_untilize_dest
//   - 6-buffer cycling for inter-step synchronization
//   - Both fused and non-fused code paths (#ifdef FUSE_COMPUTE)
//
// 8 matmul_block call sites across the two phases and two code paths.
//
// The ones-tile bias pattern: instead of a separate add_bcast_rows pass,
// bias is applied by doing matmul(ones_tile, bias_row) which accumulates
// the bias into the DST that already has the matmul result. This works
// because ct_dim=4 and the bias row has 4 tiles matching the output width.
//
// MIGRATED: Each matmul_block call is replaced by mm.matmul(). The bias
// via ones-tile is also a matmul_block call with a different in0 CB (ones),
// so it uses the same mm.matmul() (with the ones CB as in0_tile_index source).
//
// NOTE: The bias-via-ones pattern means we call matmul_block with a DIFFERENT
// in0 CB than what the MatmulOp was configured with. Since Mode 1 matmul()
// passes cfg_.in0_cb_id, we need a separate MatmulOp for the bias calls.
// ============================================================================
namespace b14_ccl_moe_gpt {

void kernel_main_snippet() {
    constexpr uint32_t cb_s2c_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_r2c_w0_w1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_s2c_in2 = tt::CBIndex::c_2;
    constexpr uint32_t cb_c2c_ones_tile = tt::CBIndex::c_4;
    constexpr uint32_t cb_r2c_w2 = tt::CBIndex::c_5;
    constexpr uint32_t cb_c2s_out = tt::CBIndex::c_6;

    uint32_t w0_w1_tiles_per_block = 8;
    uint32_t num_w0_w1_tiles_h = 16;
    uint32_t tiles_per_step = 8;
    uint32_t w0_w1_blocks_per_two_elt_tile = 4;

    // --- NEW: Two MatmulOp instances for data matmul and bias matmul ---

    // Data matmul: input @ weight
    ckernel::MatmulOpConfig data_cfg{
        .in0_cb_id = cb_s2c_in,
        .in1_cb_id = cb_r2c_w0_w1,
        .out_cb_id = cb_s2c_in2,
        .ct_dim = 4,
        .rt_dim = 1,
        .kt_dim = 1,
    };
    ckernel::BlockMatmulOp mm_data(data_cfg);
    mm_data.init();

    // Bias matmul: ones_tile @ bias_row (same ct/rt/kt, different in0 CB)
    ckernel::MatmulOpConfig bias_cfg{
        .in0_cb_id = cb_c2c_ones_tile,
        .in1_cb_id = cb_r2c_w0_w1,
        .out_cb_id = cb_s2c_in2,
        .ct_dim = 4,
        .rt_dim = 1,
        .kt_dim = 1,
    };
    ckernel::BlockMatmulOp mm_bias(bias_cfg);
    // NOTE: No separate init needed for bias -- shares the same MATH config.
    // --- END NEW ---

    // W0/W1 phase (per expert, per chunk)
    for (uint32_t tile_id = 0; tile_id < tiles_per_step; tile_id += 2) {
        uint32_t in0_index = 0;

        tile_regs_acquire();
        uint32_t k_tracker = 0;
        for (uint32_t block_id = 0; block_id < w0_w1_blocks_per_two_elt_tile; ++block_id) {
            cb_wait_front(cb_r2c_w0_w1, w0_w1_tiles_per_block);
            uint32_t last_k_index = 0;
            for (uint32_t k = 0; k < w0_w1_tiles_per_block; k += 4) {
                if (k_tracker == num_w0_w1_tiles_h) {
                    last_k_index = k;
                    break;
                }
                // --- NEW: mm_data.matmul replaces matmul_block ---
                mm_data.matmul(in0_index++, k, 0);
                // --- END NEW ---
                k_tracker++;
            }
            if (k_tracker == num_w0_w1_tiles_h) {
                // Bias addition: matmul(ones_tile, bias_row)
                // --- NEW: mm_bias.matmul replaces matmul_block with cb_c2c_ones_tile ---
                mm_bias.matmul(0, last_k_index, 0);
                // --- END NEW ---
            }
            cb_pop_front(cb_r2c_w0_w1, w0_w1_tiles_per_block);
        }

        tile_regs_commit();

        // UNCHANGED: SwiGLU activation on packer thread
        // PACK(TTI_SEMWAIT(...));
        // PACK(TT_SETC16(...));
        // PACK((llk_math_eltwise_binary_sfpu_swiglu<true, false>(0, 1, 0)));
        // PACK((llk_math_eltwise_binary_sfpu_swiglu<true, false>(2, 3, 2)));
        // PACK(TTI_STALLWAIT(...));

        pack_tile<true>(0, cb_s2c_in2, tile_id);
        pack_tile<true>(2, cb_s2c_in2, tile_id + 1);
        tile_regs_release();
    }

    // W2 phase: Same pattern with cb_r2c_w2 and different in0/out CBs
    // Uses separate MatmulOp instances for W2 data and W2 bias:
    ckernel::MatmulOpConfig w2_data_cfg{
        .in0_cb_id = cb_s2c_in2,
        .in1_cb_id = cb_r2c_w2,
        .out_cb_id = cb_c2s_out,
        .ct_dim = 4,
        .rt_dim = 1,
        .kt_dim = 1,
    };
    ckernel::BlockMatmulOp mm_w2_data(w2_data_cfg);

    ckernel::MatmulOpConfig w2_bias_cfg{
        .in0_cb_id = cb_c2c_ones_tile,
        .in1_cb_id = cb_r2c_w2,
        .out_cb_id = cb_c2s_out,
        .ct_dim = 4,
        .rt_dim = 1,
        .kt_dim = 1,
    };
    ckernel::BlockMatmulOp mm_w2_bias(w2_bias_cfg);

    // W2 loop: same structure as W0/W1 but with untilize at end
    // for (iter) {
    //     tile_regs_acquire();
    //     for (block_id) {
    //         for (k += 4) { mm_w2_data.matmul(in2_index++, k, 0); }
    //         if (bias) { mm_w2_bias.matmul(0, last_k_index, 0); }
    //     }
    //     tile_regs_commit(); tile_regs_wait();
    //     pack_untilize_dest<4, 8>(cb_c2s_out, 1, iter);
    //     tile_regs_release();
    // }
}

}  // namespace b14_ccl_moe_gpt
