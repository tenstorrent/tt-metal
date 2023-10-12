// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    #ifdef SFPU_OP_INIT_PRE_IN0_0
        constexpr auto cb_inp0 = tt::CB::c_intermed0;
    #else
        constexpr auto cb_inp0 = tt::CB::c_in0;
    #endif

    #ifdef SFPU_OP_INIT_PRE_IN1_0
        constexpr auto cb_inp1 = tt::CB::c_intermed1;
    #else
        constexpr auto cb_inp1 = tt::CB::c_in1;
    #endif

    binary_op_init_common(cb_inp0, cb_inp1);

    #if not PRE_SCALE
    binary_op_specific_init<false>(ELTWISE_OP_CODE);
    #endif

    for(uint32_t block = 0; block < per_core_block_cnt; ++block) {

        cb_reserve_back(tt::CB::c_out0, per_core_block_size);

        #ifdef SFPU_OP_INIT_PRE_IN0_0
        cb_wait_front(tt::CB::c_in0, per_core_block_size);
        cb_reserve_back(cb_inp0, per_core_block_size);
        copy_tile_init(); // need to copy from CB to DST to be able to run sfpu math

        tile_regs_acquire();
        SFPU_OP_INIT_PRE_IN0_0
        for(uint32_t i = 0; i < per_core_block_size; ++i)
        {
            copy_tile(tt::CB::c_in0, i, i); // copy from c_in[0] to DST[0]
            SFPU_OP_FUNC_PRE_IN0_0
        }
        tile_regs_commit();

        tile_regs_wait();
        for(uint32_t i = 0; i < per_core_block_size; ++i)
        {
            pack_tile(i, cb_inp0); // DST[0]->cb
        }
        tile_regs_release();

        cb_pop_front(tt::CB::c_in0, per_core_block_size);
        cb_push_back(cb_inp0, per_core_block_size);
        #endif

        #ifdef SFPU_OP_INIT_PRE_IN1_0
        cb_wait_front(tt::CB::c_in1, per_core_block_size);
        cb_reserve_back(cb_inp1, per_core_block_size);
        copy_tile_init(); // need to copy from CB to DST to be able to run sfpu math

        tile_regs_acquire();
        SFPU_OP_INIT_PRE_IN1_0
        for(uint32_t i = 0; i < per_core_block_size; ++i)
        {
            copy_tile(tt::CB::c_in1, i, i); // copy from c_in[0] to DST[0]
            SFPU_OP_FUNC_PRE_IN1_0
        }
        tile_regs_commit();

        tile_regs_wait();
        for(uint32_t i = 0; i < per_core_block_size; ++i)
        {
            pack_tile(i, cb_inp1); // DST[0]->cb
        }
        tile_regs_release();

        cb_pop_front(tt::CB::c_in1, per_core_block_size);
        cb_push_back(cb_inp1, per_core_block_size);
        #endif

        cb_wait_front(cb_inp0, per_core_block_size);
        cb_wait_front(cb_inp1, per_core_block_size);

        #if PRE_SCALE
        binary_op_specific_init<true>(ELTWISE_OP_CODE);
        #endif

        tile_regs_acquire();
        for(uint32_t i = 0; i < per_core_block_size; ++i)
        {
            ELTWISE_OP(cb_inp0, cb_inp1, i, i, i);

            #ifdef SFPU_OP_INIT_0
            SFPU_OP_INIT_0
            SFPU_OP_FUNC_0
            #endif

            #ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
            #endif
        }
        tile_regs_commit();

        tile_regs_wait();
        for(uint32_t i = 0; i < per_core_block_size; ++i)
        {
            pack_tile(i, tt::CB::c_out0);
        }
        tile_regs_release();

        cb_pop_front(cb_inp0, per_core_block_size);
        cb_pop_front(cb_inp1, per_core_block_size);
        cb_push_back(tt::CB::c_out0, per_core_block_size);
    }

}
}
