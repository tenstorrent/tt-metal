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

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    constexpr auto cb_out0 =  tt::CB::c_out0;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);

    for(uint32_t block = 0; block < per_core_block_cnt; ++block) {

        copy_tile_to_dst_init_short(); // need to copy from CB to DST to be able to run sfpu math
        cb_wait_front(cb_in0, per_core_block_size);
        cb_wait_front(cb_in1, per_core_block_size);
        cb_reserve_back(cb_out0, per_core_block_size);

        acquire_dst(tt::DstMode::Half);
        for(uint32_t i = 0; i < per_core_block_size; ++i)
        {
            copy_tile(cb_in0, i, i); // copy from c_in[0] to DST[0]
        }

        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in1);

        for(uint32_t i = 0; i < per_core_block_size; ++i)
        {
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in1, i, i);
        }

        for(uint32_t i = 0; i < per_core_block_size; ++i)
        {
            pack_tile(i, cb_out0);
        }
        release_dst(tt::DstMode::Half);

        cb_pop_front(cb_in0, per_core_block_size);
        cb_pop_front(cb_in1, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }
}
}
