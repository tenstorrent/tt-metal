// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_binary.h"

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"

#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif

void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);
    uint32_t acc_to_dst = get_arg_val<uint32_t>(2);


    #ifdef ARCH_QUASAR
        constexpr uint32_t dfb_in0_id = get_compile_time_arg_val(0);
        constexpr uint32_t dfb_in1_id = get_compile_time_arg_val(1);
        constexpr uint32_t dfb_in2_id = get_compile_time_arg_val(2);
        constexpr uint32_t dfb_out_id = get_compile_time_arg_val(3);
        experimental::DataflowBuffer dfb_in0(dfb_in0_id);
        experimental::DataflowBuffer dfb_in1(dfb_in1_id);
        experimental::DataflowBuffer dfb_in2(dfb_in2_id);
        experimental::DataflowBuffer dfb_out(dfb_out_id);
        binary_op_init_common(dfb_in0.get_id(), dfb_in1.get_id(), dfb_out.get_id());
        #if not defined ELTWISE_DEST_REUSE_TYPE
            #ifdef FULL_INIT
                binary_tiles_init<true, ELTWISE_OP_TYPE>(dfb_in0.get_id(), dfb_in1.get_id());
            #else
                binary_tiles_init<false, ELTWISE_OP_TYPE>(dfb_in0.get_id(), dfb_in1.get_id());
            #endif
        #endif
    #else
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
    #endif

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        #ifdef ARCH_QUASAR
            dfb_in0.wait_front(per_core_block_size);
            dfb_in1.wait_front(per_core_block_size);
            dfb_out.reserve_back(per_core_block_size);
        #else
            cb_wait_front(cb_inp0, per_core_block_size);
            cb_wait_front(cb_inp1, per_core_block_size);
            cb_reserve_back(cb_out0, per_core_block_size);
        #endif
        tile_regs_acquire();

#if defined(DST_ACCUM_MODE) || defined(ELTWISE_DEST_REUSE_TYPE)
    #ifdef ARCH_QUASAR
            dfb_in2.wait_front(per_core_block_size);
            copy_tile_to_dst_init_short(dfb_in2.get_id());
            for (uint32_t i = 0; i < per_core_block_size; ++i) {
                copy_tile(dfb_in2.get_id(), i, i);  // copy from c_in[0] to DST[0]
            }
            dfb_in2.pop_front(per_core_block_size);
    #else
            cb_wait_front(cb_in2, per_core_block_size);
            copy_tile_to_dst_init_short(cb_in2);
            for (uint32_t i = 0; i < per_core_block_size; ++i) {
                copy_tile(cb_in2, i, i);  // copy from c_in[0] to DST[0]
            }
            cb_pop_front(cb_in2, per_core_block_size);
    #endif
#endif

#ifdef DST_ACCUM_MODE
// The following define is needed if mul_tiles/_init is used
#ifdef MUL_TILES_WITH_DST_ACCUM
    #ifdef ARCH_QUASAR
        ELTWISE_OP_INIT(dfb_in0.get_id(), dfb_in1.get_id());
    #else
        ELTWISE_OP_INIT(cb_inp0, cb_inp1);
    #endif
#else
    #ifdef ARCH_QUASAR
        ELTWISE_OP_INIT(dfb_in0.get_id(), dfb_in1.get_id(), true);
    #else
        ELTWISE_OP_INIT(cb_inp0, cb_inp1, true);
    #endif
#endif
#endif

#ifdef ELTWISE_DEST_REUSE_TYPE
    #ifdef ARCH_QUASAR
        binary_dest_reuse_tiles_init<ELTWISE_OP_TYPE, ELTWISE_DEST_REUSE_TYPE>(dfb_in0.get_id());
    #else
        binary_dest_reuse_tiles_init<ELTWISE_OP_TYPE, ELTWISE_DEST_REUSE_TYPE>(cb_inp0);
    #endif
#endif

        for (uint32_t i = 0; i < per_core_block_size; ++i) {
#ifdef ELTWISE_DEST_REUSE_TYPE
    #ifdef ARCH_QUASAR
            binary_dest_reuse_tiles<ELTWISE_OP_TYPE, ELTWISE_DEST_REUSE_TYPE>(dfb_in0.get_id(), i, i);
    #else
            binary_dest_reuse_tiles<ELTWISE_OP_TYPE, ELTWISE_DEST_REUSE_TYPE>(cb_inp0, i, i);
    #endif
#else
    #ifdef ARCH_QUASAR
            ELTWISE_OP(dfb_in0.get_id(), dfb_in1.get_id(), i, i, i);
    #else
            ELTWISE_OP(cb_inp0, cb_inp1, i, i, i);
    #endif
#endif

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
        #ifdef ARCH_QUASAR
            pack_tile(i, dfb_out.get_id());
        #else
            pack_tile(i, cb_out0);
        #endif
        }
        tile_regs_release();

        #ifdef ARCH_QUASAR
            dfb_in0.pop_front(per_core_block_size);
            dfb_in1.pop_front(per_core_block_size);
            dfb_out.push_back(per_core_block_size);
        #else
            cb_pop_front(cb_inp0, per_core_block_size);
            cb_pop_front(cb_inp1, per_core_block_size);
            cb_push_back(cb_out0, per_core_block_size);
        #endif
    }
}
