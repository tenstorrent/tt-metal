// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compute_kernel_api/cb_api.h>
#include <compute_kernel_api/pack.h>
#include <compute_kernel_api/reconfig_data_format.h>
#include <compute_kernel_api/reg_api.h>
#include <debug/dprint.h>
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
#include "metal/ops/common/dataflow_utils.hpp"

namespace NAMESPACE {

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);  // rows to process in this kernel
constexpr uint32_t block_size = get_compile_time_arg_val(1);         // size of block
constexpr uint32_t Wt = get_compile_time_arg_val(2);

constexpr uint32_t cb_query = tt::CBIndex::c_0;
constexpr uint32_t cb_key = tt::CBIndex::c_1;
constexpr uint32_t cb_value = tt::CBIndex::c_2;
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_3;
constexpr uint32_t cb_scaler = tt::CBIndex::c_4;
constexpr uint32_t cb_reduction_scaler = tt::CBIndex::c_5;
constexpr uint32_t cb_transpose_key = tt::CBIndex::c_6;  // used for transposing key tiles
constexpr uint32_t cb_temp_accum = tt::CBIndex::c_7;     // used for accumulating results
constexpr uint32_t cb_output = tt::CBIndex::c_8;

const uint32_t onetile = 1U;

void MAIN() {
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // run throught all tiles in query row
        for (uint32_t col = 0; col < Wt; col += block_size) {
            cb_wait_front(cb_query, block_size);

            // process  block_size tiles from query row
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                const uint32_t accum_register = 0;
                const uint32_t tile_register = 1U;
                cb_reserve_back(cb_output, block_size);

                // run through all tiles in key row and matmul with query row
                for (uint32_t key_col = 0; key_col < Wt;) {
                    cb_wait_front(cb_key, block_size);
                    tile_regs_acquire();
                    for (uint32_t block_idx_key = 0; block_idx_key < block_size; ++block_idx_key) {
                        transpose_wh_tile(cb_key, block_idx_key, block_idx_key);
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    cb_reserve_back(cb_transpose_key, block_size);
                    pack_reconfig_data_format(cb_transpose_key);
                    for (uint32_t block_idx_key = 0; block_idx_key < block_size; ++block_idx_key) {
                        pack_tile(block_idx_key, cb_transpose_key);
                    }
                    tile_regs_release();
                    cb_push_back(cb_transpose_key, block_size);

                    cb_wait_front(cb_transpose_key, block_size);
                    tile_regs_acquire();

                    // TODO[check]: maybe I can call this func once before the loop and then call mm_init_short()
                    mm_init(cb_query, cb_transpose_key, cb_output, 0);
                    for (uint32_t block_idx_key = 0; block_idx_key < block_size; ++block_idx_key, ++key_col) {
                        const uint32_t working_register = (key_col == 0) ? accum_register : tile_register;
                        matmul_tiles(
                            cb_query,
                            cb_transpose_key,
                            /* tile_idx */ 0,
                            /* tile_idx */ block_idx_key,
                            working_register,
                            0);

                        if (j > 0) {
                            add_binary_tile_init();
                            add_binary_tile(accum_register, working_register);
                        }
                    }

                    if (key_col >= block_size) {  // we’re processing the second (and subsequent) blocks of the key row
                        copy_tile_init(cb_temp_accum);
                        copy_tile(cb_temp_accum, /* tile_idx */ 0, /* register_idx */ tile_register);

                        add_binary_tile_init();
                        add_binary_tile(accum_register, tile_register);
                        pop_front(cb_temp_accum, onetile);
                    }
                    tile_regs_commit();

                    cb_reserve_back(cb_temp_accum, onetile);
                    tile_regs_wait();
                    pack_reconfig_data_format(cb_temp_accum);
                    pack_tile(accum_register, cb_temp_accum);
                    tile_regs_release();
                    cb_push_back(cb_temp_accum, onetile);
                }  // end of processing key row

                tile_regs_acquire();
                copy_tile_init(cb_temp_accum);
                copy_tile(cb_temp_accum, /* tile_idx */ 0, /* register_idx */ accum_register);
                tile_regs_commit();

                tile_regs_wait();
                pack_reconfig_data_format(cb_output);
                pack_tile(accum_register, cb_output);
                tile_regs_release();
            }  // end of processing query row block_size tiles
            cb_push_back(cb_output, block_size);
            cb_pop_front(cb_query, block_size);
        }  // end of processing query row
    }
}

}  // namespace NAMESPACE
