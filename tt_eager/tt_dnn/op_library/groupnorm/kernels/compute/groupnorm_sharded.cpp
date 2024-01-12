// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/matmul.h"

// #include "debug/dprint.h"


inline void tilize_in(
    uint32_t in_cb_id,
    uint32_t out_cb_id,
    uint32_t block_h,
    uint32_t block_w
) {
    tilize_init_short(in_cb_id, block_w);
    for (uint32_t h = 0; h < block_h; ++h) {
        cb_reserve_back(out_cb_id, block_w);
        tilize_block(in_cb_id, block_w, out_cb_id);
        cb_push_back(out_cb_id, block_w);
        cb_pop_front(in_cb_id, block_w);
    }
    tilize_uninit();
}

inline void untilize_out(
    uint32_t in_cb_id,
    uint32_t out_cb_id,
    uint32_t block_h,
    uint32_t block_w
) {
    untilize_init_short(in_cb_id);
    for (uint32_t h = 0; h < block_h; ++h) {
        cb_wait_front(in_cb_id, block_w);
        cb_reserve_back(out_cb_id, block_w);
        untilize_block(in_cb_id, block_w, out_cb_id);
        cb_pop_front(in_cb_id, block_w);
        cb_push_back(out_cb_id, block_w);
    }
    untilize_uninit(in_cb_id);
}

// SPLIT REDUCE across Cores
namespace NAMESPACE {
void MAIN {

    constexpr uint32_t is_mcast_sender                = get_compile_time_arg_val(0);
    constexpr uint32_t do_gamma                       = get_compile_time_arg_val(1);
    constexpr uint32_t do_beta                        = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores_per_mcast_group      = get_compile_time_arg_val(3);

    constexpr uint32_t batch                          = get_compile_time_arg_val(4);
    constexpr uint32_t group                          = get_compile_time_arg_val(5);

    constexpr uint32_t num_batch_group                = get_compile_time_arg_val(6);

    volatile uint32_t block_h                        = get_compile_time_arg_val(7);
    constexpr uint32_t block_w                        = get_compile_time_arg_val(8);
    constexpr uint32_t block_hw                       = get_compile_time_arg_val(9);

    constexpr uint32_t subblock_w                     = get_compile_time_arg_val(10);
    constexpr uint32_t num_subblocks_w                = get_compile_time_arg_val(11);

    constexpr uint32_t tilize_in0                      = get_compile_time_arg_val(12);

    constexpr uint32_t per_core_M                       = get_compile_time_arg_val(13);
    constexpr uint32_t per_core_N                       = get_compile_time_arg_val(14);
    constexpr uint32_t per_core_MN                       = get_compile_time_arg_val(15);
    constexpr uint32_t block_bhw                       = get_compile_time_arg_val(16);


    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    constexpr uint32_t cb_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_scaler = tt::CB::c_in2;
    constexpr uint32_t cb_eps = tt::CB::c_in3;
    constexpr uint32_t cb_scaler_global = tt::CB::c_in4;
    constexpr uint32_t cb_gamma = tt::CB::c_in5;
    constexpr uint32_t cb_beta = tt::CB::c_in6;
    constexpr uint32_t cb_x = tt::CB::c_intermed0; // x minus mean
    constexpr uint32_t cb_xmm = tt::CB::c_intermed1; // x minus mean
    constexpr uint32_t cb_ex_partial = tt::CB::dataflow0; // E[x] partial reduce
    constexpr uint32_t cb_ex = tt::CB::dataflow1; // E[x] global reduce
    constexpr uint32_t cb_ex_external = tt::CB::dataflow2;
    constexpr uint32_t cb_xmm2 = cb_x; // xmm^2
    constexpr uint32_t cb_ex2pe = tt::CB::c_intermed3; // E[(x-E[x])^2]+eps
    constexpr uint32_t cb_fusion = cb_xmm; // stream gamma/beta
    constexpr uint32_t cb_out = tt::CB::c_out0;
    constexpr uint32_t cb_ex_global = num_cores_per_mcast_group == 1 ? cb_ex_partial : tt::CB::dataflow7;

    uint32_t index_subblock_w_offset = 0;
    uint32_t index_h_offset = 0;
    uint32_t index_b_offset = 0;
    uint32_t index_g_offset = 0;
    uint32_t index_b_offset_ex = 0;

    constexpr int cb_in = tilize_in0 ? cb_x : cb_in0;
    constexpr int cb_im = (do_gamma | do_beta) ? cb_x : cb_out;
    constexpr int cb_outgamma = do_beta ? cb_fusion : cb_out;

    binary_op_init_common(cb_in0, cb_in0, cb_xmm);

    if constexpr (tilize_in0) {
        tilize_in(cb_in0, cb_in, per_core_M, per_core_N);
        cb_wait_front(cb_in, per_core_MN);
    }

    // Partial-E[x] for each core
    unpack_reconfig_data_format(cb_in, cb_scaler);
    reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
    cb_reserve_back(cb_ex_partial, num_batch_group);
    cb_wait_front(cb_scaler, 1);
    index_b_offset = 0;
    for (uint32_t b = 0; b < batch; ++b) {
        index_g_offset = 0;
        for (uint32_t g = 0; g < group; ++g) {
            uint32_t index_bg_offset = index_b_offset + index_g_offset;
            index_h_offset = 0;
            tile_regs_acquire();
            for (uint32_t h = 0; h < block_h; ++h) {
                for (uint32_t w = 0; w < block_w; ++w) {
                    uint32_t index = index_bg_offset + index_h_offset + w;
                    reduce_tile(REDUCE_OP, REDUCE_DIM, cb_in, cb_scaler, index, scaler0, dst0);
                }
                index_h_offset += per_core_N;
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex_partial);
            tile_regs_release();
            index_g_offset += block_w;
        }
        index_b_offset += block_bhw;
    }
    cb_push_back(cb_ex_partial, num_batch_group);
    reduce_revert_delta();
    unpack_reconfig_data_format(cb_xmm, cb_xmm);

    if constexpr(is_mcast_sender and num_cores_per_mcast_group > 1) {
        index_b_offset = 0;
        reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
        cb_reserve_back(cb_ex_global, num_batch_group);
        cb_reserve_back(cb_ex, num_batch_group);
        for (uint32_t bg = 0; bg < num_batch_group; ++bg) {
            tile_regs_acquire();
            cb_wait_front(cb_scaler_global, 1);
            for (uint32_t w = 0; w < num_cores_per_mcast_group; w++) {
                cb_wait_front(cb_ex_external, 1);
                reduce_tile(REDUCE_OP, REDUCE_DIM, cb_ex_external, cb_scaler_global, 0, scaler0, dst0);
                cb_pop_front(cb_ex_external, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex_global);
            tile_regs_release();
        }
        reduce_revert_delta();
        cb_push_back(cb_ex_global, num_batch_group);
        cb_push_back(cb_ex, num_batch_group);
    }

    // x - E[x]
    sub_tiles_bcast_scalar_init_short();
    cb_reserve_back(cb_xmm, per_core_MN);
    cb_wait_front(cb_ex_global, num_batch_group);
    unpack_reconfig_data_format(cb_in, cb_ex_global);
    index_b_offset = 0;
    index_b_offset_ex = 0;
    for (uint32_t b = 0; b < batch; ++b) {
        index_h_offset = 0;
        for (uint32_t i = 0; i < block_h; i++) {
            index_g_offset = 0;
            for (uint32_t g = 0; g < group; ++g) {
                uint32_t index_bhg_offset = index_b_offset + index_h_offset + index_g_offset;
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; j++) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t index = index_bhg_offset + index_subblock_w_offset + w;
                        uint32_t index_ex = index_b_offset_ex + g;
                        sub_tiles_bcast_scalar(cb_in, cb_ex_global, index, index_ex, w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; i++) {
                        pack_tile(i, cb_xmm);
                    }
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                index_g_offset += block_w;
            }
            index_h_offset += per_core_N;
        }
        index_b_offset_ex += group;
        index_b_offset += block_bhw;
    }
    cb_pop_front(cb_in, per_core_MN);
    cb_pop_front(cb_ex_global, num_batch_group);
    cb_push_back(cb_xmm, per_core_MN);
    cb_wait_front(cb_xmm, per_core_MN);
    unpack_reconfig_data_format(cb_xmm2, cb_xmm2);

    // (x - E[x])^2
    mul_tiles_init();
    cb_reserve_back(cb_xmm2, per_core_MN);
    index_b_offset = 0;
    for (uint32_t b = 0; b < batch; ++b) {
        index_h_offset = 0;
        for (uint32_t i = 0; i < block_h; i++) {
            index_g_offset = 0;
            for (uint32_t g = 0; g < group; ++g) {
                uint32_t index_bhg_offset = index_b_offset + index_h_offset + index_g_offset;
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; j++) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t index = index_bhg_offset + index_subblock_w_offset + w;
                        mul_tiles(cb_xmm, cb_xmm, index, index, w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; i++) {
                        pack_tile(i, cb_xmm2);
                    }
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                index_g_offset += block_w;
            }
            index_h_offset += per_core_N;
        }
        index_b_offset += block_bhw;
    }
    cb_push_back(cb_xmm2, per_core_MN);

    // Partial-Var(x)
    index_b_offset = 0;
    reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
    cb_reserve_back(cb_ex_partial, num_batch_group);
    cb_wait_front(cb_xmm2, per_core_MN);
    cb_wait_front(cb_scaler, 1);
    for (uint32_t b = 0; b < batch; ++b) {
        index_g_offset = 0;
        for (uint32_t g = 0; g < group; ++g) {
            uint32_t index_bg_offset = index_b_offset + index_g_offset;
            index_h_offset = 0;
            tile_regs_acquire();
            for (uint32_t h = 0; h < block_h; ++h) {
                for (uint32_t w = 0; w < block_w; ++w) {
                    uint32_t index = index_bg_offset + index_h_offset + w;
                    reduce_tile(REDUCE_OP, REDUCE_DIM, cb_xmm2, cb_scaler, index, scaler0, dst0);
                }
                index_h_offset += per_core_N;
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex_partial);
            tile_regs_release();
            index_g_offset += block_w;
        }
        index_b_offset += block_bhw;
    }
    cb_push_back(cb_ex_partial, num_batch_group);
    cb_pop_front(cb_xmm2, per_core_MN);
    reduce_revert_delta();

    // global reduce
    if constexpr(is_mcast_sender and num_cores_per_mcast_group > 1) {
        index_b_offset = 0;
        reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
        cb_reserve_back(cb_ex_global, num_batch_group);
        cb_reserve_back(cb_ex, num_batch_group);
        for (uint32_t bg = 0; bg < num_batch_group; ++bg) {
            tile_regs_acquire();
            cb_wait_front(cb_scaler_global, 1);
            for (uint32_t w = 0; w < num_cores_per_mcast_group; w++) {
                cb_wait_front(cb_ex_external, 1);
                reduce_tile(REDUCE_OP, REDUCE_DIM, cb_ex_external, cb_scaler_global, 0, scaler0, dst0);
                cb_pop_front(cb_ex_external, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex_global);
            tile_regs_release();
        }
        reduce_revert_delta();
        cb_push_back(cb_ex_global, num_batch_group);
        cb_push_back(cb_ex, num_batch_group);
    }

    cb_wait_front(cb_ex_global, num_batch_group);
    cb_reserve_back(cb_ex2pe, num_batch_group);
    for (uint32_t bg = 0; bg < num_batch_group; ++bg) {
        // (Var + eps)
        tile_regs_acquire();
        add_tiles_init();
        add_tiles(cb_ex_global, cb_eps, bg, 0, dst0);
        tile_regs_wait();
        // sqrt(Var + eps)
        sqrt_tile_init();
        sqrt_tile(dst0);
        tile_regs_wait();
        // 1/[sqrt(Var + eps)]
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex2pe);
        tile_regs_release();
    }
    cb_push_back(cb_ex2pe, num_batch_group);
    cb_pop_front(cb_ex_global, num_batch_group);

    // (x - Ex) * 1/[sqrt(Var + eps)]
    if constexpr(do_gamma == 0 && do_beta == 0) {
        pack_reconfig_data_format(cb_out);
    }
    mul_tiles_bcast_scalar_init_short();
    cb_reserve_back(cb_im, per_core_MN);
    cb_wait_front(cb_ex2pe, num_batch_group);
    index_b_offset = 0;
    index_b_offset_ex = 0;
    for (uint32_t b = 0; b < batch; ++b) {
        index_h_offset = 0;
        for (uint32_t i = 0; i < block_h; i++) {
            index_g_offset = 0;
            for (uint32_t g = 0; g < group; ++g) {
                uint32_t index_bhg_offset = index_b_offset + index_h_offset + index_g_offset;
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; j++) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t index = index_bhg_offset + index_subblock_w_offset + w;
                        uint32_t index_ex2pe = index_b_offset_ex + g;
                        mul_tiles_bcast_scalar(cb_xmm, cb_ex2pe, index, index_ex2pe, w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; i++) {
                        pack_tile(i, cb_im);
                    }
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                index_g_offset += block_w;
            }
            index_h_offset += per_core_N;
        }
        index_b_offset_ex += group;
        index_b_offset += block_bhw;
    }
    cb_push_back(cb_im, per_core_MN);
    cb_pop_front(cb_ex2pe, num_batch_group);
    cb_pop_front(cb_xmm, per_core_MN);
    cb_wait_front(cb_im, per_core_MN);

    if constexpr(do_gamma) {
        unpack_reconfig_data_format(cb_im, cb_gamma);
        if constexpr(do_beta == 0) {
            pack_reconfig_data_format(cb_out);
        }
        mul_bcast_rows_init_short();
        cb_reserve_back(cb_outgamma, per_core_MN);
        cb_wait_front(cb_gamma, per_core_N);
        index_b_offset = 0;
        index_b_offset_ex = 0;
        for (uint32_t b = 0; b < batch; ++b) {
            index_h_offset = 0;
            for (uint32_t i = 0; i < block_h; i++) {
                index_g_offset = 0;
                for (uint32_t g = 0; g < group; ++g) {
                    uint32_t index_bhg_offset = index_b_offset + index_h_offset + index_g_offset;
                    index_subblock_w_offset = 0;
                    for (uint32_t j = 0; j < num_subblocks_w; j++) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; w++) {
                            uint32_t index = index_bhg_offset + index_subblock_w_offset + w;
                            uint32_t index_gm = index_subblock_w_offset + w + index_g_offset;
                            mul_tiles_bcast_rows(cb_im, cb_gamma, index, index_gm, w);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; i++) {
                            pack_tile(i, cb_outgamma);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
                    index_g_offset += block_w;
                }
                index_h_offset += per_core_N;
            }
            index_b_offset_ex += group;
            index_b_offset += block_bhw;
        }
        cb_push_back(cb_outgamma, per_core_MN);
        cb_pop_front(cb_im, per_core_MN);
        cb_wait_front(cb_outgamma, per_core_MN);
    }

    if constexpr(do_beta) {
        unpack_reconfig_data_format(cb_fusion, cb_beta);
        pack_reconfig_data_format(cb_out);
        add_bcast_rows_init_short();
        cb_reserve_back(cb_out, per_core_MN);
        cb_wait_front(cb_beta, per_core_N);
        index_b_offset = 0;
        index_b_offset_ex = 0;
        for (uint32_t b = 0; b < batch; ++b) {
            index_h_offset = 0;
            for (uint32_t i = 0; i < block_h; i++) {
                index_g_offset = 0;
                for (uint32_t g = 0; g < group; ++g) {
                    uint32_t index_bhg_offset = index_b_offset + index_h_offset + index_g_offset;
                    index_subblock_w_offset = 0;
                    for (uint32_t j = 0; j < num_subblocks_w; j++) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; w++) {
                            uint32_t index = index_bhg_offset + index_subblock_w_offset + w;
                            uint32_t index_gm = index_subblock_w_offset + w + index_g_offset;
                            add_tiles_bcast_rows(cb_fusion, cb_beta, index, index_gm, w);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; i++) {
                            pack_tile(i, cb_out);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
                    index_g_offset += block_w;
                }
                index_h_offset += per_core_N;
            }
            index_b_offset_ex += group;
            index_b_offset += block_bhw;
        }
        cb_push_back(cb_out, per_core_MN);
        cb_pop_front(cb_im, per_core_MN);
        cb_wait_front(cb_out, per_core_MN);
    }

}
}
