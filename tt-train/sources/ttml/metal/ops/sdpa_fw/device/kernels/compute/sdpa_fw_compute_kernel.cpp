// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compute_kernel_api/cb_api.h>
#include <compute_kernel_api/pack.h>
#include <compute_kernel_api/reconfig_data_format.h>
#include <compute_kernel_api/reg_api.h>
#include <hostdevcommon/kernel_structs.h>
#include <tensix.h>

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/transpose_wh.h"

namespace NAMESPACE {

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);  // rows to process in this kernel
constexpr uint32_t block_size = get_compile_time_arg_val(1);         // size of block
constexpr uint32_t Wt = get_compile_time_arg_val(2);
constexpr uint32_t Ht = get_compile_time_arg_val(3);

constexpr uint32_t cb_query = tt::CBIndex::c_0;
constexpr uint32_t cb_key = tt::CBIndex::c_1;
constexpr uint32_t cb_value = tt::CBIndex::c_2;
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_3;
constexpr uint32_t cb_scaler = tt::CBIndex::c_4;
constexpr uint32_t cb_reduction_scaler = tt::CBIndex::c_5;
constexpr uint32_t cb_transpose_key = tt::CBIndex::c_6;  // isn't used right now, for debugging only
constexpr uint32_t cb_temp_accum = tt::CBIndex::c_7;     // used for accumulating results
constexpr uint32_t cb_output = tt::CBIndex::c_8;

const uint32_t onetile = 1U;

const uint32_t q_chunk_size = Wt;
const uint32_t k_chunk_size = Wt;
const uint32_t v_chunk_size = Wt;
const uint32_t kv_chunks_size = Wt;
const uint32_t kv_chunks_number = Ht;

void MAIN {
    init_sfpu(cb_query, cb_output);
    binary_op_init_common(cb_query, cb_key, cb_value);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        cb_wait_front(cb_query, q_chunk_size);
        cb_reserve_back(cb_output, q_chunk_size);

        const uint32_t matmul_accum_reg = 0;
        for (uint32_t h = 0; h < kv_chunks_number; ++h) {  // read all
            cb_wait_front(cb_key, kv_chunks_size);
            cb_wait_front(cb_value, kv_chunks_size);

            mm_init(cb_query, cb_key, cb_temp_accum, /* transpose */ 1);
            // TODO[check]: check whether I can use mm_init_short here instead of full init
            // mm_init_short(cb_query, cb_key, /* transpose */ 1);
            tile_regs_acquire();
            for (uint32_t tile_idx = 0; tile_idx < q_chunk_size; tile_idx++) {
                matmul_tiles(
                    cb_query,
                    cb_key,
                    /* tile_idx */ tile_idx,
                    /* tile_idx */ tile_idx,
                    /* dst_reg_idx*/ matmul_accum_reg,
                    /* transpose */ 1);  // accumulate in dest_reg 0
            }
            tile_regs_commit();

            tile_regs_wait();
            cb_reserve_back(cb_temp_accum, onetile);
            pack_reconfig_data_format(cb_temp_accum);
            pack_tile(matmul_accum_reg, cb_temp_accum);
            tile_regs_release();
            cb_push_back(cb_temp_accum, onetile);

            // pop key data to make space for next key chunk
            cb_pop_front(cb_key, kv_chunks_size);

            // wait for Q_chunk by K_chunk result to be ready
            cb_wait_front(cb_temp_accum, onetile);
            mm_init(cb_temp_accum, cb_value, cb_output, /* transpose */ 0);
            // TODO[check]: check whether I can use mm_init_short here instead of full init
            // mm_init_short(cb_temp_accum, cb_value, /* transpose */ 0);
            for (uint32_t tile_idx = 0; tile_idx < kv_chunks_size; tile_idx += block_size) {
                tile_regs_acquire();
                for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                    matmul_tiles(
                        cb_temp_accum,
                        cb_value,
                        /* tile_idx */ 0,
                        /* tile_idx */ tile_idx + block_idx,
                        block_idx,
                        0);
                }
                tile_regs_commit();

                tile_regs_wait();
                if (h > 0) {
                    // if in the same q_chunk(row) continue accumulating
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }

                // TODO[improve]: change block_idx/size naming to dst_reg_idx/count to be more clear
                for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                    pack_tile<true>(
                        /* dst_reg_idx */ block_idx, /* cb_idx */ cb_output, /* tile_idx */ tile_idx + block_idx);
                }
                tile_regs_release();
                PACK((llk_pack_reconfig_l1_acc(0)));
            }

            // pop data from circular buffers
            cb_pop_front(cb_temp_accum, onetile);
            cb_pop_front(cb_value, kv_chunks_size);

            // mm_init_short(cb_query, cb_key, 1);
            // for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
            //     tile_regs_acquire();
            //     for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            //         matmul_tiles(
            //             cb_query,
            //             cb_key,
            //             /* tile_idx */ tile_idx + block_idx,
            //             /* tile_idx */ tile_idx + block_idx,
            //             block_idx,
            //             1);
            //     }
            //     tile_regs_commit();

            //     tile_regs_wait();
            //     if (tile_idx > 0) {
            //         // if in the same row continue accumulating
            //         PACK((llk_pack_reconfig_l1_acc(1)));
            //     }

            //     for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            //         pack_tile<true>(/* dst_reg_idx */ block_idx, /* cb_idx */ cb_output, /* tile_idx */ h);
            //         if (tile_idx == 0 && block_idx == 0) {
            //             // If this was the first tile of a row, start accumulating
            //             PACK((llk_pack_reconfig_l1_acc(1)));
            //         }
            //     }
            //     tile_regs_release();
            // }
            // PACK((llk_pack_reconfig_l1_acc(0)));

            // tile_regs_acquire();
            // mm_init_short(cb_query, cb_key, 1);
            // for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx++) {
            //     matmul_tiles(
            //         cb_query,
            //         cb_key,
            //         /* tile_idx */ tile_idx,
            //         /* tile_idx */ tile_idx,
            //         0,
            //         1);
            // }
            // tile_regs_commit();

            // tile_regs_wait();
            // pack_reconfig_data_format(cb_output);
            // pack_tile(0, cb_output);
            // tile_regs_release();

            // cb_pop_front(cb_key, Wt);
        }
        cb_push_back(cb_output, q_chunk_size);
        cb_pop_front(cb_query, q_chunk_size);
    }
}

}  // namespace NAMESPACE
