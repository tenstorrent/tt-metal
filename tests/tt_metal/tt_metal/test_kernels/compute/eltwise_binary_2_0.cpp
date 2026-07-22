// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_binary.h"

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    uint32_t per_core_block_size = get_arg(args::per_core_block_size);
    uint32_t acc_to_dst = get_arg(args::acc_to_dst);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_in2(dfb::in2);
    DataflowBuffer dfb_out(dfb::out);
    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);
#if not defined ELTWISE_DEST_REUSE_TYPE
#ifdef FULL_INIT
    binary_tiles_init<true, ELTWISE_OP_TYPE>(dfb::in0, dfb::in1);
#else
    binary_tiles_init<false, ELTWISE_OP_TYPE>(dfb::in0, dfb::in1);
#endif
#endif

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        dfb_in0.wait_front(per_core_block_size);
        dfb_in1.wait_front(per_core_block_size);
        dfb_out.reserve_back(per_core_block_size);
        tile_regs_acquire();

#if defined(DST_ACCUM_MODE) || defined(ACC_TO_DEST) || defined(ELTWISE_DEST_REUSE_TYPE)
        dfb_in2.wait_front(per_core_block_size);
        copy_tile_to_dst_init_short(dfb::in2);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(dfb::in2, i, i);  // copy from c_in[0] to DST[0]
        }
        dfb_in2.pop_front(per_core_block_size);
#endif

#if defined(DST_ACCUM_MODE) || defined(ACC_TO_DEST)
// The following define is needed for WH/BH if mul_tiles/_init is used
#if defined(MUL_TILES_WITH_DST_ACCUM)
        ELTWISE_OP_INIT(dfb::in0, dfb::in1);
#else
        ELTWISE_OP_INIT(dfb::in0, dfb::in1, true);
#endif
#endif

#ifdef ELTWISE_DEST_REUSE_TYPE
        // Dest-reuse init is folded into the per-op inits via the binary_reuse_dest template param;
        // dispatch on the compile-time op type since there is no generic binary_init.
        if constexpr (ELTWISE_OP_TYPE == EltwiseBinaryType::ELWADD) {
            add_init<ELTWISE_DEST_REUSE_TYPE>(dfb::in0, dfb::in0);
        } else if constexpr (ELTWISE_OP_TYPE == EltwiseBinaryType::ELWSUB) {
            sub_init<ELTWISE_DEST_REUSE_TYPE>(dfb::in0, dfb::in0);
        } else {
            mul_init<ELTWISE_DEST_REUSE_TYPE>(dfb::in0, dfb::in0);
        }
#endif

        for (uint32_t i = 0; i < per_core_block_size; ++i) {
#ifdef ELTWISE_DEST_REUSE_TYPE
            binary_dest_reuse_tiles<ELTWISE_OP_TYPE, ELTWISE_DEST_REUSE_TYPE>(dfb::in0, i, i);
#else
            ELTWISE_OP(dfb::in0, dfb::in1, i, i, i);
#endif

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, dfb::out);
        }
        tile_regs_release();

        dfb_in0.pop_front(per_core_block_size);
        dfb_in1.pop_front(per_core_block_size);
        dfb_out.push_back(per_core_block_size);
    }
}
