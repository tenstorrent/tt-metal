// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#if defined(ARCH_BLACKHOLE) && defined(UCK_CHLKC_UNPACK) && defined(ELTWISE_DEST_REUSE_TYPE)
constexpr std::uint32_t stall_clear_control_mask = 1U << 5;
static_assert(
    (llk_unpack_a_detail::dest_reuse_dummy_unpack<ELTWISE_DEST_REUSE_TYPE>() & stall_clear_control_mask) != 0);
#endif

#ifdef ELTWISE_BROADCAST_TYPE
template <
    EltwiseBinaryType eltwise_binary_type,
    EltwiseBinaryReuseDestType binary_reuse_dest,
    BroadcastType broadcast_type>
ALWI void binary_dest_reuse_broadcast_tiles_init(uint32_t icb0) {
    state_configure(icb0, __builtin_LINE());
#ifndef ARCH_QUASAR
    UNPACK(constexpr bool acc_to_dest = true);
#else
    UNPACK(constexpr bool acc_to_dest = false);
#endif
    UNPACK((llk_unpack_A_init<broadcast_type, acc_to_dest, binary_reuse_dest>(false, false, icb0)));
    MATH((llk_math_eltwise_binary_init<eltwise_binary_type, broadcast_type, MATH_FIDELITY, binary_reuse_dest>(
        icb0, icb0, false /* acc_to_dest */)));
}

template <
    EltwiseBinaryType eltwise_binary_type,
    EltwiseBinaryReuseDestType binary_reuse_dest,
    BroadcastType broadcast_type>
ALWI void binary_dest_reuse_broadcast_tiles(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
#ifndef ARCH_QUASAR
    UNPACK(constexpr bool acc_to_dest = true);
#else
    UNPACK(constexpr bool acc_to_dest = false);
#endif
    UNPACK((llk_unpack_A<broadcast_type, acc_to_dest, binary_reuse_dest>(in_cb_id, in_tile_index)));
    MATH(
        (llk_math_eltwise_binary<eltwise_binary_type, broadcast_type, DST_ACCUM_MODE, MATH_FIDELITY, binary_reuse_dest>(
            in_cb_id, in_cb_id, dst_tile_index, true /* clear_fp32_dst_acc */)));
}
#endif

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
    // full_init=true: compute_kernel_hw_startup does not run llk_unpack_AB_init, so a math-only
    // (binary_tiles_init<false>) init is no longer sufficient on its own; always do the full init.
    binary_tiles_init<true, ELTWISE_OP_TYPE>(dfb::in0, dfb::in1);
#endif

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif

#ifdef PRECEDE_DEST_REUSE_WITH_COL_BROADCAST
    dfb_in1.wait_front(1);
    dfb_in2.wait_front(1);
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        dfb_in0.wait_front(per_core_block_size);
#ifndef PRECEDE_DEST_REUSE_WITH_COL_BROADCAST
        dfb_in1.wait_front(per_core_block_size);
#endif
        dfb_out.reserve_back(per_core_block_size);
        tile_regs_acquire();

#if (defined(DST_ACCUM_MODE) || defined(ACC_TO_DEST) || defined(ELTWISE_DEST_REUSE_TYPE)) && \
    !defined(PRECEDE_DEST_REUSE_WITH_COL_BROADCAST)
        dfb_in2.wait_front(per_core_block_size);
        copy_init(dfb::in2);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(dfb::in2, i, i);  // copy from c_in[0] to DST[0]
        }
        dfb_in2.pop_front(per_core_block_size);
#endif

#ifdef PRECEDE_DEST_REUSE_WITH_COL_BROADCAST
        reconfig_data_format(dfb::in0, dfb::in1);
        sub_bcast_cols_init(dfb::in0, dfb::in1);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            sub_tiles_bcast_cols(dfb::in0, dfb::in1, i, 0, i);
        }
        dfb_in0.pop_front(per_core_block_size);
        reconfig_data_format_srca(dfb::in0, dfb::in2);
        if constexpr (ELTWISE_OP_TYPE == EltwiseBinaryType::ELWADD) {
            add_reuse_dest_init<ELTWISE_DEST_REUSE_TYPE>(dfb::in2);
        } else if constexpr (ELTWISE_OP_TYPE == EltwiseBinaryType::ELWSUB) {
            sub_reuse_dest_init<ELTWISE_DEST_REUSE_TYPE>(dfb::in2);
        } else {
            mul_reuse_dest_init<ELTWISE_DEST_REUSE_TYPE>(dfb::in2);
        }
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            if constexpr (ELTWISE_OP_TYPE == EltwiseBinaryType::ELWADD) {
                add_reuse_dest_tiles<ELTWISE_DEST_REUSE_TYPE>(dfb::in2, 0, i);
            } else if constexpr (ELTWISE_OP_TYPE == EltwiseBinaryType::ELWSUB) {
                sub_reuse_dest_tiles<ELTWISE_DEST_REUSE_TYPE>(dfb::in2, 0, i);
            } else {
                mul_reuse_dest_tiles<ELTWISE_DEST_REUSE_TYPE>(dfb::in2, 0, i);
            }
        }
#else
#if defined(DST_ACCUM_MODE) || defined(ACC_TO_DEST)
// The following define is needed for WH/BH if mul_tiles/_init is used
#if defined(MUL_TILES_WITH_DST_ACCUM)
        ELTWISE_OP_INIT(dfb::in0, dfb::in1);
#else
        ELTWISE_OP_INIT(dfb::in0, dfb::in1, true);
#endif
#endif

#ifdef ELTWISE_DEST_REUSE_TYPE
#ifdef ELTWISE_BROADCAST_TYPE
        binary_dest_reuse_broadcast_tiles_init<ELTWISE_OP_TYPE, ELTWISE_DEST_REUSE_TYPE, ELTWISE_BROADCAST_TYPE>(
            dfb::in0);
#else
        // Dest-reuse init is the per-op {add,sub,mul}_reuse_dest_init<reuse_dest>; dispatch on the
        // compile-time op type since there is no generic binary_init.
        if constexpr (ELTWISE_OP_TYPE == EltwiseBinaryType::ELWADD) {
            add_reuse_dest_init<ELTWISE_DEST_REUSE_TYPE>(dfb::in0);
        } else if constexpr (ELTWISE_OP_TYPE == EltwiseBinaryType::ELWSUB) {
            sub_reuse_dest_init<ELTWISE_DEST_REUSE_TYPE>(dfb::in0);
        } else {
            mul_reuse_dest_init<ELTWISE_DEST_REUSE_TYPE>(dfb::in0);
        }
#endif
#endif

        for (uint32_t i = 0; i < per_core_block_size; ++i) {
#ifdef ELTWISE_DEST_REUSE_TYPE
#ifdef ELTWISE_BROADCAST_TYPE
            binary_dest_reuse_broadcast_tiles<ELTWISE_OP_TYPE, ELTWISE_DEST_REUSE_TYPE, ELTWISE_BROADCAST_TYPE>(
                dfb::in0, i, i);
#else
            // Dispatch on the compile-time op type; the dest-reuse execute is per-op.
            if constexpr (ELTWISE_OP_TYPE == EltwiseBinaryType::ELWADD) {
                add_reuse_dest_tiles<ELTWISE_DEST_REUSE_TYPE>(dfb::in0, i, i);
            } else if constexpr (ELTWISE_OP_TYPE == EltwiseBinaryType::ELWSUB) {
                sub_reuse_dest_tiles<ELTWISE_DEST_REUSE_TYPE>(dfb::in0, i, i);
            } else {
                mul_reuse_dest_tiles<ELTWISE_DEST_REUSE_TYPE>(dfb::in0, i, i);
            }
#endif
#else
            ELTWISE_OP(dfb::in0, dfb::in1, i, i, i);
#endif

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
        }
#endif
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, dfb::out);
        }
        tile_regs_release();

#ifndef PRECEDE_DEST_REUSE_WITH_COL_BROADCAST
        dfb_in0.pop_front(per_core_block_size);
        dfb_in1.pop_front(per_core_block_size);
#endif
        dfb_out.push_back(per_core_block_size);
    }

#ifdef PRECEDE_DEST_REUSE_WITH_COL_BROADCAST
    dfb_in1.pop_front(1);
    dfb_in2.pop_front(1);
#endif
}
