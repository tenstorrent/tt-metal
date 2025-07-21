// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/eltwise_binary.h"

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/tile_move_copy.h"
namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);
    uint32_t acc_to_dst = get_arg_val<uint32_t>(2);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_inp0 = cb_in0;
    constexpr auto cb_inp1 = cb_in1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    constexpr auto cb_in2 = tt::CBIndex::c_2;

    binary_op_init_common(cb_inp0, cb_inp1, cb_out0);

#if not defined ELTWISE_DEST_REUSE_TYPE
#ifdef FULL_INIT
    binary_tiles_init<true, ELTWISE_OP_TYPE>(cb_in0, cb_in1);
#else
    binary_tiles_init<false, ELTWISE_OP_TYPE>(cb_in0, cb_in1);
#endif
#endif

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        cb_wait_front(cb_inp0, per_core_block_size);
        cb_wait_front(cb_inp1, per_core_block_size);
        cb_reserve_back(cb_out0, per_core_block_size);

        tile_regs_acquire();

#if defined(DST_ACCUM_MODE) || defined(ELTWISE_DEST_REUSE_TYPE)
        cb_wait_front(cb_in2, per_core_block_size);
        copy_tile_to_dst_init_short(cb_in2);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_in2, i, i);  // copy from c_in[0] to DST[0]
        }
        cb_pop_front(cb_in2, per_core_block_size);
#endif

#ifdef DST_ACCUM_MODE
// The following define is needed if mul_tiles/_init is used
#ifdef MUL_TILES_WITH_DST_ACCUM
        ELTWISE_OP_INIT(cb_inp0, cb_inp1);
#else
        ELTWISE_OP_INIT(cb_inp0, cb_inp1, true);
#endif
#endif

#ifdef ELTWISE_DEST_REUSE_TYPE
        binary_dest_reuse_tiles_init<ELTWISE_OP_TYPE, ELTWISE_DEST_REUSE_TYPE>(cb_inp0);
#endif

        for (uint32_t i = 0; i < per_core_block_size; ++i) {
#ifdef ELTWISE_DEST_REUSE_TYPE
            binary_dest_reuse_tiles<ELTWISE_OP_TYPE, ELTWISE_DEST_REUSE_TYPE>(cb_inp0, i, i);
#else
            ELTWISE_OP(cb_inp0, cb_inp1, i, i, i);
#endif

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_out0);
        }
        tile_regs_release();

        cb_pop_front(cb_inp0, per_core_block_size);
        cb_pop_front(cb_inp1, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }
}
}  // namespace NAMESPACE
