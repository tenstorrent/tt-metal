// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/debug/assert.h"
#include "api/debug/dprint.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/binary_max_min.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/softplus.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/reduce_custom.h"

using std::uint32_t;

template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t head_dim_t>
void sdpa_inner_loop(
    const uint32_t cb_q_in,
    const uint32_t cb_kt_in,
    const uint32_t cb_qkt_out) {
    const uint32_t in0_block_w = 4;
    const uint32_t subblock_h = 2;
    const uint32_t subblock_w = 4;
    const uint32_t q_num_subblocks = 4;
    const uint32_t kt_num_subblocks = 4;

    const uint32_t q_subblock_num_tiles = subblock_h * in0_block_w;
    uint32_t q_wait_tiles = q_subblock_num_tiles;
    const uint32_t output_num_tiles_per_row = Sq_chunk_t * subblock_h;

    uint32_t q_index_offset = 0;
    uint32_t kt_index_offset = 0;

    //
    // Fill the pipeline
    //
    pack_reconfig_data_format(cb_qkt_out);
    reconfig_data_format(cb_q_in, cb_kt_in);
    cb_reserve_back(cb_qkt_out, Sq_chunk_t * Sk_chunk_t);
    mm_block_init_short(
        cb_q_in, cb_kt_in, true /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);
    cb_wait_front(cb_q_in, q_wait_tiles);
    cb_wait_front(cb_kt_in, head_dim_t * Sk_chunk_t);

    for (uint32_t kt_subblock = 0; kt_subblock < kt_num_subblocks; kt_subblock++) {
        // Compute a single subblock
        tile_regs_acquire();
        uint32_t dst_index = 0;
        uint32_t q_index = q_index_offset;
        uint32_t kt_index = kt_index_offset;
        for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
            matmul_block(
                cb_q_in,
                cb_kt_in,
                q_index,
                kt_index,
                dst_index,
                true /*transpose*/,
                subblock_w,
                subblock_h,
                in0_block_w);
            q_index++;
            kt_index += Sq_chunk_t;
        }
        tile_regs_commit();

        // Pack the subblock
        tile_regs_wait();
        uint32_t dst_idx = 0;
        uint32_t out_col_offset = kt_subblock * subblock_w;
        for (uint32_t r = 0; r < subblock_h; r++) {
            uint32_t out_row_offset = r * Sq_chunk_t;
            for (uint32_t c = 0; c < subblock_w; c++) {
                pack_tile<true>(dst_idx, cb_qkt_out, out_row_offset + out_col_offset + c);
                dst_idx++;
            }
        }
        tile_regs_release();
        kt_index_offset += subblock_w;
    }
    // Max reduce here
    //...
    // end max reduce
    q_index_offset += subblock_h * in0_block_w;
    q_wait_tiles += q_subblock_num_tiles;

    //
    // Pipeline steady state
    //
    for (uint32_t q_subblock = 1; q_subblock < q_num_subblocks; q_subblock++) {
        MATH(DPRINT << "Steady state, row " << q_subblock << ENDL());
        // Sub max (prev row) here
        //...
        // end sub max
        // Exp here
        //...
        // end exp
        MATH(DPRINT << "Waiting for Q " << q_wait_tiles << " tiles." << ENDL());
        cb_wait_front(cb_q_in, q_wait_tiles);
        kt_index_offset = 0;
        for (uint32_t kt_subblock = 0; kt_subblock < kt_num_subblocks; ++kt_subblock) {
            // MATH(DPRINT << "Computing matmul for Kt subblock " << kt_subblock << ENDL());
            //  Compute a single subblock
            tile_regs_acquire();
            uint32_t dst_index = 0;
            uint32_t q_index = q_index_offset;
            uint32_t kt_index = kt_index_offset;
            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                matmul_block(
                    cb_q_in,
                    cb_kt_in,
                    q_index,
                    kt_index,
                    dst_index,
                    true /*transpose*/,
                    subblock_w,
                    subblock_h,
                    in0_block_w);
                q_index++;
                kt_index += Sq_chunk_t;
            }
            tile_regs_commit();

            // Pack the subblock
            tile_regs_wait();
            // PACK(DPRINT << "Packing subblock " << kt_subblock << ENDL());
            uint32_t dst_idx = 0;
            uint32_t out_col_offset = kt_subblock * subblock_w;
            for (uint32_t r = 0; r < subblock_h; r++) {
                uint32_t out_row_offset = (r + q_subblock * subblock_h) * Sk_chunk_t;
                for (uint32_t c = 0; c < subblock_w; c++) {
                    pack_tile<true>(dst_idx, cb_qkt_out, out_row_offset + out_col_offset + c);
                    dst_idx++;
                }
            }
            tile_regs_release();
            kt_index_offset += subblock_w;
        }
        // Max reduce here
        //...
        // end max reduce
        q_index_offset += subblock_h * in0_block_w;
        q_wait_tiles += q_subblock_num_tiles;
    }
    cb_pop_front(cb_q_in, head_dim_t * Sq_chunk_t);
    cb_pop_front(cb_kt_in, head_dim_t * Sk_chunk_t);
    cb_push_back(cb_qkt_out, Sq_chunk_t * Sk_chunk_t);

    // dummy: use QKT somehow
    cb_wait_front(cb_qkt_out, Sq_chunk_t * Sk_chunk_t);
    cb_pop_front(cb_qkt_out, Sq_chunk_t * Sk_chunk_t);
}

void kernel_main() {
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(0);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t head_dim_t = get_compile_time_arg_val(2);
    constexpr uint32_t num_iter = get_compile_time_arg_val(3);

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_kt_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_qkt_out = tt::CBIndex::c_2;

    mm_init(cb_q_in, cb_kt_in, cb_qkt_out);

    for (uint32_t iter = 0; iter < num_iter; iter++) {
        MATH(DPRINT << "Iteration " << iter << ENDL());

        sdpa_inner_loop<Sq_chunk_t, Sk_chunk_t, head_dim_t>(cb_q_in, cb_kt_in, cb_qkt_out);
    }
}
