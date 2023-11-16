// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include "compute_kernel_api/tile_move_copy.h"

// SPLIT REDUCE across Cores
namespace NAMESPACE {
void MAIN {

    constexpr uint32_t is_top_row                     = get_compile_time_arg_val(0);
    constexpr uint32_t do_gamma                       = get_compile_time_arg_val(1);
    constexpr uint32_t do_beta                        = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks                     = get_compile_time_arg_val(3);
    constexpr uint32_t block_h                        = get_compile_time_arg_val(4);
    constexpr uint32_t block_w                        = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_w                     = get_compile_time_arg_val(6);
    constexpr uint32_t num_subblocks_w                = get_compile_time_arg_val(7);
    const bool is_allgather_worker                    = get_compile_time_arg_val(8) == 1;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0, tt::CB::c_intermed0);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    constexpr uint32_t cb_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_in1 = tt::CB::c_in1;
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
    constexpr uint32_t cb_ex_partial2 = tt::CB::dataflow3; // E[(x-E[x])^2] partial reduce
    constexpr uint32_t cb_ex2 = tt::CB::dataflow4; // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_ex_external2 = tt::CB::dataflow5;
    constexpr uint32_t cb_ex_global = tt::CB::dataflow7; // E[x] global reduce
    constexpr uint32_t cb_xmm2 = tt::CB::c_intermed2; // xmm^2
    constexpr uint32_t cb_ex2pe = tt::CB::c_intermed3; // E[(x-E[x])^2]+eps
    constexpr uint32_t cb_fusion = tt::CB::c_intermed4; // stream gamma/beta
    constexpr uint32_t cb_out = tt::CB::c_out0;

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

    #ifdef FUSE_PRE_ADD
    constexpr int cb_in = cb_x;
    #else
    constexpr int cb_in = cb_in0;
    #endif
    constexpr int cb_im = (do_gamma | do_beta) ? cb_fusion : cb_out;
    constexpr int cb_outgamma = do_beta ? cb_fusion : cb_out;

    for (uint32_t i = 0; i < block_h; i++) {
        // pre-add x + y
        #ifdef FUSE_PRE_ADD
        unpack_reconfig_data_format(tt::CB::c_in0, tt::CB::c_in0);
        pack_reconfig_data_format(tt::CB::c_intermed0);
        add_tiles_init();
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                add_tiles(cb_in0, cb_in1, index, index, w);
            }
            tile_regs_commit();
            cb_reserve_back(cb_in, subblock_w);
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_in);
            }
            tile_regs_release();
            cb_push_back(cb_in, subblock_w);
            index_subblock_w_offset += subblock_w;
        }
        cb_wait_front(cb_in, block_w+index_h_offset);
        unpack_reconfig_data_format(tt::CB::c_intermed0, tt::CB::c_intermed0);
        #endif

        // E[x],
        reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
        cb_wait_front(cb_scaler, 1);
        cb_reserve_back(cb_ex_partial, 1);
        tile_regs_acquire();
        for (uint32_t w = 0; w < block_w; w++) {
            reduce_tile(REDUCE_OP, REDUCE_DIM, cb_in, cb_scaler, w+index_h_offset, scaler0, dst0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex_partial);
        tile_regs_release();
        reduce_revert_delta();
        cb_push_back(cb_ex_partial, 1);
        index_h_offset += block_w;
    }

    // global reduce, cb_ex <-- cb_ex_external, cb_ex_partial
    if constexpr(is_allgather_worker) {
        constexpr int num_allgather_workers = 6;
        constexpr int num_tiles_per_allgather_worker = block_h / num_allgather_workers;
        reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
        cb_reserve_back(cb_ex, num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            cb_wait_front(cb_ex_external, num_blocks);
            cb_wait_front(cb_scaler_global, 1);
            tile_regs_acquire();
            for (uint32_t w = 0; w < num_blocks; w++) {
                reduce_tile(REDUCE_OP, REDUCE_DIM, cb_ex_external, cb_scaler_global, w, scaler0, dst0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex);
            tile_regs_release();
            cb_pop_front(cb_ex_external, num_blocks);
        }
        reduce_revert_delta();
        cb_push_back(cb_ex, num_tiles_per_allgather_worker);
        cb_wait_front(cb_ex, num_tiles_per_allgather_worker);
    }
    cb_wait_front(cb_ex_global, block_h);

    index_h_offset = 0;
    for (uint32_t i = 0; i < block_h; i++) {
        // x - E[x]
        sub_bcast_cols_init_short();
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                sub_tiles_bcast_cols(cb_in, cb_ex_global, index, 0, w);
            }
            tile_regs_commit();
            cb_reserve_back(cb_xmm, subblock_w);
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_xmm);
            }
            tile_regs_release();
            cb_push_back(cb_xmm, subblock_w);
            index_subblock_w_offset += subblock_w;
        }
        cb_pop_front(cb_ex_global, 1);
        cb_pop_front(cb_in, block_w);
        cb_wait_front(cb_xmm, block_w+index_h_offset);

        // (x - E[x])^2, cb_mm2 <-- cb_xmm
        mul_tiles_init();
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles(cb_xmm, cb_xmm, index, index, w);
            }
            tile_regs_commit();
            cb_reserve_back(cb_xmm2, subblock_w);
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_xmm2);
            }
            tile_regs_release();
            cb_push_back(cb_xmm2, subblock_w);
            index_subblock_w_offset += subblock_w;
        }

        // Var(x)
        cb_reserve_back(cb_ex_partial2, 1);
        reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
        tile_regs_acquire();
        cb_wait_front(cb_xmm2, block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            reduce_tile(REDUCE_OP, REDUCE_DIM, cb_xmm2, cb_scaler, w, scaler0, dst0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex_partial2);
        tile_regs_release();
        reduce_revert_delta();
        cb_push_back(cb_ex_partial2, 1);
        cb_pop_front(cb_xmm2, block_w);
        cb_wait_front(cb_ex_partial2, 1);
        index_h_offset += block_w;
    }

    // global reduce, cb_ex <-- cb_ex_external, cb_ex_partial
    if constexpr(is_allgather_worker) {
        constexpr int num_allgather_workers = 6;
        constexpr int num_tiles_per_allgather_worker = block_h / num_allgather_workers;
        reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
        cb_reserve_back(cb_ex2, num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            cb_wait_front(cb_ex_external2, num_blocks);
            cb_wait_front(cb_scaler_global, 1);

            tile_regs_acquire();
            for (uint32_t w = 0; w < num_blocks; w++) {
                reduce_tile(REDUCE_OP, REDUCE_DIM, cb_ex_external2, cb_scaler_global, w, scaler0, dst0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex2);
            tile_regs_release();
            cb_pop_front(cb_ex_external2, num_blocks);
        }
        reduce_revert_delta();
        cb_push_back(cb_ex2, num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            // 1/[sqrt(Var + eps)],
            cb_wait_front(cb_ex2, 1);
            cb_reserve_back(cb_ex2pe, 1);
            tile_regs_acquire();
            add_tiles_init();
            add_tiles(cb_ex2, cb_eps, i, 0, dst0);
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
            cb_push_back(cb_ex2pe, 1);
            tile_regs_release();
            cb_wait_front(cb_ex2pe, 1);
        }

    }
    cb_wait_front(cb_ex_global, block_h);

    index_h_offset = 0;
    for (uint32_t i = 0; i < block_h; i++) {
        // (x - Ex) * 1/[sqrt(Var + eps)]
        if constexpr(do_gamma == 0 && do_beta == 0) {
            pack_reconfig_data_format(cb_out);
        } else {
            pack_reconfig_data_format(tt::CB::c_intermed0);
        }
        cb_wait_front(cb_xmm, block_w);
        mul_bcast_cols_init_short();
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                mul_tiles_bcast_cols(cb_xmm, cb_ex_global, index, 0, w);
            }
            tile_regs_commit();
            cb_reserve_back(cb_im, subblock_w);
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_im);
            }
            tile_regs_release();
            cb_push_back(cb_im, subblock_w);
            index_subblock_w_offset += subblock_w;
        }
        cb_pop_front(cb_ex_global, 1);
        cb_pop_front(cb_xmm, block_w);
        cb_wait_front(cb_im, block_w);

        if constexpr(do_gamma) {
            if constexpr(do_beta == 0) {
                pack_reconfig_data_format(cb_out);
            }
            cb_wait_front(cb_im, block_w);
            cb_wait_front(cb_gamma, block_w);
            mul_bcast_rows_init_short();
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    index = w + index_subblock_w_offset;
                    mul_tiles_bcast_rows(cb_im, cb_gamma, index, index, w);
                }
                tile_regs_commit();
                cb_reserve_back(cb_outgamma, subblock_w);
                tile_regs_wait();
                for (uint32_t i = 0; i < subblock_w; i++) {
                    pack_tile(i, cb_outgamma);
                }
                tile_regs_release();
                cb_push_back(cb_outgamma, subblock_w);
                index_subblock_w_offset += subblock_w;
            }
            cb_pop_front(cb_im, block_w);
        }

        if constexpr(do_beta) {
            pack_reconfig_data_format(cb_out);
            cb_wait_front(cb_beta, block_w);
            cb_wait_front(cb_fusion, block_w);
            add_bcast_rows_init_short();
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    index = w + index_subblock_w_offset;
                    add_tiles_bcast_rows(cb_fusion, cb_beta, index, index, w);
                }
                tile_regs_commit();
                cb_reserve_back(cb_out, subblock_w);
                tile_regs_wait();
                for (uint32_t i = 0; i < subblock_w; i++) {
                    pack_tile(i, cb_out);
                }
                tile_regs_release();
                cb_push_back(cb_out, subblock_w);
                index_subblock_w_offset += subblock_w;
            }
            cb_pop_front(cb_fusion, block_w);
            cb_wait_front(cb_out, block_w+index_h_offset);
        }
        index_h_offset += block_w;
    }

}

}
