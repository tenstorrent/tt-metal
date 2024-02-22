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
// #include "debug/status.h"



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
    tilize_uninit(in_cb_id);
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

    constexpr uint32_t is_num_channel_div_by_tile   = get_compile_time_arg_val(17);
    constexpr uint32_t is_num_rows_per_batch_div_by_tile   = get_compile_time_arg_val(18);


    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    // constexpr uint32_t cb_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_in = tt::CB::c_in7;
    constexpr uint32_t cb_im_out = tt::CB::c_intermed2;
    constexpr uint32_t cb_zero_mask = tt::CB::c_intermed4;
    constexpr uint32_t cb_zero_mask_full_tile = tt::CB::c_intermed6;
    constexpr uint32_t cb_scaler = tt::CB::c_in2;
    constexpr uint32_t cb_eps = tt::CB::c_in3;
    constexpr uint32_t cb_scaler_global = tt::CB::c_in4;
    constexpr uint32_t cb_gamma = tt::CB::c_in5;
    constexpr uint32_t cb_beta = tt::CB::c_in6;
    constexpr uint32_t cb_x = tt::CB::c_intermed0; // x minus mean
    constexpr uint32_t cb_xmm = tt::CB::c_intermed1; // x minus mean
    constexpr uint32_t cb_xmm_temp = tt::CB::c_intermed5; // x minus mean
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

    constexpr int cb_im = cb_x;
    constexpr int cb_outgamma = cb_fusion;
    constexpr int cb_inbeta = not do_gamma ? cb_x : cb_fusion;
    constexpr int cb_outbeta = not do_gamma ? cb_fusion : cb_x;
    constexpr int cb_untilize_in = (do_gamma and not do_beta) ? cb_outgamma : do_beta ? cb_outbeta : cb_im;

    binary_op_init_common(cb_in, cb_in, cb_xmm);

    // UNPACK(( DPRINT << "is_num_rows_per_batch_div_by_tile " << is_num_rows_per_batch_div_by_tile << ENDL() ));
    // UNPACK(( DPRINT << "is_num_channel_div_by_tile " << is_num_channel_div_by_tile << ENDL() ));
    // UNPACK(( DPRINT << "batch " << batch << ENDL() ));
    // UNPACK(( DPRINT << "group " << group << ENDL() ));
    // UNPACK(( DPRINT << "block_h " << block_h << ENDL() ));
    // UNPACK(( DPRINT << "block_w " << block_w << ENDL() ));
    // UNPACK(( DPRINT << "block_hw " << block_hw << ENDL() ));
    // UNPACK(( DPRINT << "num_subblocks_w " << num_subblocks_w << ENDL() ));
    // UNPACK(( DPRINT << "subblock_w " << subblock_w << ENDL() ));


    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t gamma_beta_group_offset = 0;
        for (uint32_t g = 0; g < group; ++g) {

            // tilize input
            for (uint32_t h = 0; h < block_h; ++h) {
                for (uint32_t w = 0; w < block_w; ++w) {
                    tilize_init_short(cb_in, 1);
                    cb_wait_front(cb_in, 1);
                    cb_reserve_back(cb_x, 1);
                    tilize_block(cb_in, 1, cb_x);
                    cb_push_back(cb_x, 1);
                    cb_pop_front(cb_in, 1);
                    tilize_uninit(cb_in);
                }
            }

            // Partial-E[x] for each core
            reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
            cb_reserve_back(cb_ex_partial, 1);
            tile_regs_acquire();
            cb_wait_front(cb_scaler, 1);
            cb_wait_front(cb_x, block_hw);
            index_h_offset = 0;
            for (uint32_t h = 0; h < block_h; ++h) {
                for (uint32_t w = 0; w < block_w; ++w) {
                    uint32_t index = index_h_offset + w;
                    reduce_tile(REDUCE_OP, REDUCE_DIM, cb_x, cb_scaler, index, scaler0, dst0);
                }
                index_h_offset += block_w;
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex_partial);
            tile_regs_release();
            cb_push_back(cb_ex_partial, 1);
            reduce_revert_delta();
            cb_wait_front(cb_ex_partial, 1);


            if constexpr(is_mcast_sender and num_cores_per_mcast_group > 1) {
                index_b_offset = 0;
                reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
                cb_reserve_back(cb_ex_global, 1);
                cb_reserve_back(cb_ex, 1);
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
                reduce_revert_delta();
                cb_push_back(cb_ex_global, 1);
                cb_push_back(cb_ex, 1);
            }

            // x - E[x]
            sub_tiles_bcast_scalar_init_short();
            cb_reserve_back(cb_xmm_temp, block_hw);
            cb_wait_front(cb_ex_global, 1);
            unpack_reconfig_data_format(cb_x, cb_ex_global);
            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; j++) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t index = w + index_subblock_w_offset;
                        sub_tiles_bcast_scalar(cb_x, cb_ex_global, index, 0, w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; i++) {
                        pack_tile(i, cb_xmm_temp);
                    }
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                cb_pop_front(cb_x, block_w);
            }
            cb_pop_front(cb_ex_global, 1);
            cb_push_back(cb_xmm_temp, block_hw);
            cb_wait_front(cb_xmm_temp, block_hw);


            // zero out the garbage values
            cb_reserve_back(cb_xmm, block_hw);
            index_h_offset = 0;
            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t w = 0; w < block_w; w++) {
                    if (w == block_w - 1 and not is_num_channel_div_by_tile and is_num_rows_per_batch_div_by_tile) {
                        cb_wait_front(cb_zero_mask, 1);
                        mul_bcast_rows_init_short();
                        tile_regs_acquire();
                        uint32_t index = w + index_h_offset;
                        mul_tiles_bcast_rows(cb_xmm_temp, cb_zero_mask, index, 0, 0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile(0, cb_xmm);
                        tile_regs_release();
                    } else if (w == block_w - 1 and not is_num_channel_div_by_tile and not is_num_rows_per_batch_div_by_tile) {
                        cb_wait_front(cb_zero_mask, 1);
                        mul_tiles_init();
                        tile_regs_acquire();
                        uint32_t index = w + index_h_offset;
                        mul_tiles(cb_xmm_temp, cb_zero_mask, index, 0, 0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile(0, cb_xmm);
                        tile_regs_release();
                    } else if (not is_num_rows_per_batch_div_by_tile) {
                        cb_wait_front(cb_zero_mask_full_tile, 1);
                        mul_tiles_init();
                        tile_regs_acquire();
                        uint32_t index = w + index_h_offset;
                        mul_tiles(cb_xmm_temp, cb_zero_mask_full_tile, index, 0, 0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile(0, cb_xmm);
                        tile_regs_release();
                    } else {
                        copy_tile_init();
                        tile_regs_acquire();
                        uint32_t index = w + index_h_offset;
                        copy_tile(cb_xmm_temp, index, 0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile(0, cb_xmm);
                        tile_regs_release();
                    }
                }
                index_h_offset += block_w;
            }
            cb_push_back(cb_xmm, block_hw);
            cb_wait_front(cb_xmm, block_hw);
            cb_pop_front(cb_xmm_temp, block_hw);

            // (x - E[x])^2
            mul_tiles_init();
            cb_reserve_back(cb_xmm2, block_hw);
            index_h_offset = 0;
            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; j++) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t index = w + index_subblock_w_offset + index_h_offset;
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
                index_h_offset += block_w;
            }
            cb_push_back(cb_xmm2, block_hw);
            cb_wait_front(cb_xmm2, block_hw);

            // Partial-Var(x)
            index_b_offset = 0;
            reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
            cb_reserve_back(cb_ex_partial, 1);
            cb_wait_front(cb_xmm2, block_hw);
            cb_wait_front(cb_scaler, 1);
            tile_regs_acquire();
            index_h_offset = 0;
            for (uint32_t h = 0; h < block_h; ++h) {
                for (uint32_t w = 0; w < block_w; ++w) {
                    uint32_t index = index_h_offset + w;
                    reduce_tile(REDUCE_OP, REDUCE_DIM, cb_xmm2, cb_scaler, index, scaler0, dst0);
                }
                index_h_offset += block_w;
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex_partial);
            tile_regs_release();
            cb_push_back(cb_ex_partial, 1);
            cb_pop_front(cb_xmm2, block_hw);
            reduce_revert_delta();

            // global reduce
            if constexpr(is_mcast_sender and num_cores_per_mcast_group > 1) {
                reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
                cb_reserve_back(cb_ex_global, 1);
                cb_reserve_back(cb_ex, 1);
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
                reduce_revert_delta();
                cb_push_back(cb_ex_global, 1);
                cb_push_back(cb_ex, 1);
            }


            cb_wait_front(cb_ex_global, 1);
            cb_reserve_back(cb_ex2pe, 1);
            // (Var + eps)
            tile_regs_acquire();
            add_tiles_init();
            add_tiles(cb_ex_global, cb_eps, 0, 0, dst0);
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
            cb_push_back(cb_ex2pe, 1);
            cb_pop_front(cb_ex_global, 1);

            // (x - Ex) * 1/[sqrt(Var + eps)]
            mul_tiles_bcast_scalar_init_short();
            cb_reserve_back(cb_im, block_hw);
            cb_wait_front(cb_ex2pe, 1);
            index_h_offset = 0;
            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; j++) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t index = w + index_subblock_w_offset + index_h_offset;
                        mul_tiles_bcast_scalar(cb_xmm, cb_ex2pe, index, 0, w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; i++) {
                        pack_tile(i, cb_im);
                    }
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                index_h_offset += block_w;
            }
            cb_push_back(cb_im, block_hw);
            cb_pop_front(cb_ex2pe, 1);
            cb_pop_front(cb_xmm, block_hw);
            cb_wait_front(cb_im, block_hw);

            if constexpr(do_gamma) {
                mul_bcast_rows_init_short();
                cb_wait_front(cb_gamma, block_w + gamma_beta_group_offset);
                index_h_offset = 0;
                cb_reserve_back(cb_outgamma, block_hw);
                for (uint32_t i = 0; i < block_h; i++) {
                    index_subblock_w_offset = 0;
                    for (uint32_t j = 0; j < num_subblocks_w; j++) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; w++) {
                            uint32_t index = w + index_subblock_w_offset;
                            mul_tiles_bcast_rows(cb_im, cb_gamma, index + index_h_offset, index + gamma_beta_group_offset, w);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; i++) {
                            pack_tile(i, cb_outgamma);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
                    index_h_offset += block_w;
                }
                cb_push_back(cb_outgamma, block_hw);
                cb_pop_front(cb_im, block_hw);
                cb_wait_front(cb_outgamma, block_hw);
            }

            if constexpr(do_beta) {
                add_bcast_rows_init_short();
                cb_wait_front(cb_beta, block_w + gamma_beta_group_offset);
                index_h_offset = 0;
                cb_reserve_back(cb_outbeta, block_hw);
                for (uint32_t i = 0; i < block_h; i++) {
                    index_subblock_w_offset = 0;
                    for (uint32_t j = 0; j < num_subblocks_w; j++) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; w++) {
                            uint32_t index = w + index_subblock_w_offset;
                            add_tiles_bcast_rows(cb_inbeta, cb_beta, index + index_h_offset, index + gamma_beta_group_offset, w);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; i++) {
                            pack_tile(i, cb_outbeta);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
                    index_h_offset += block_w;
                }
                cb_push_back(cb_outbeta, block_hw);
                cb_pop_front(cb_inbeta, block_hw);
                cb_wait_front(cb_outbeta, block_hw);
            }

            for (uint32_t h = 0; h < block_h; ++h) {
                for (uint32_t w = 0; w < block_w; ++w) {
                    untilize_init_short(cb_untilize_in);
                    cb_wait_front(cb_untilize_in, 1);
                    cb_reserve_back(cb_im_out, 1);
                    untilize_block(cb_untilize_in, 1, cb_im_out);
                    cb_pop_front(cb_untilize_in, 1);
                    cb_push_back(cb_im_out, 1);
                    untilize_uninit(cb_untilize_in);
                    cb_wait_front(cb_im_out, 1);
                }
            }
            gamma_beta_group_offset += block_w;
        }
    }
}
}
