// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute-only microbenchmark: does pack-thread SFPU work overlap math-thread matmul?
//
// The math thread runs the matmul's last-K-block subblock loop (1 x 4 output tiles, kt = k_block, FP32 DEST,
// half-sync), as bmm_large_block_zm_fused_bias_activation does. Per subblock the pack thread applies the
// FP32-accurate sigmoid to the first sfpu_tiles tiles of the committed DEST half (as apply_activation_from_pack does),
// then packs the four tiles. No reader or writer: the operands are whatever the circular buffers hold, which does not
// change the timing. k_block 0 times the pack thread alone.
//
// sfpu_mode: 1 = sigmoid_tile_pack (its SFPU start waits for an idle math unit),
//            2 = the same SFPU code without that start stall.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    constexpr uint32_t k_block = get_compile_time_arg_val(0);
    constexpr uint32_t num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t sfpu_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t sfpu_mode = get_compile_time_arg_val(3);
    constexpr uint32_t in0_cb = 0, in1_cb = 1, out_cb = 16;
    constexpr uint32_t sub_w = 4;

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, out_cb);
    matmul_block_init(in0_cb, in1_cb, 0, sub_w, 1, k_block > 0 ? k_block : 1);
    if constexpr (sfpu_tiles > 0) {
        sigmoid_tile_init_pack<false>();
    }

    for (uint32_t s = 0; s < num_subblocks; s++) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < k_block; k++) {
            matmul_block(in0_cb, in1_cb, k, k * sub_w, 0, 0, sub_w, 1, k_block);
        }
        tile_regs_commit();

        if constexpr (sfpu_tiles > 0) {
            PACK(TTI_SEMWAIT(
                p_stall::STALL_TDMA | p_stall::STALL_CFG,
                semaphore::t6_sem(semaphore::MATH_PACK),
                p_stall::STALL_ON_ZERO));
            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
            for (uint32_t i = 0; i < sfpu_tiles; i++) {
                if constexpr (sfpu_mode == 1) {
                    sigmoid_tile_pack<VectorMode::RC, false>(i);
                } else {
                    PACK((math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(i)));
                    PACK((_llk_math_eltwise_sfpu_apply_vector_mode_(
                        ckernel::sfpu::calculate_sigmoid<false, DST_ACCUM_MODE, 8>, VectorMode::RC)));
                    PACK((math::clear_dst_reg_addr()));
                }
            }
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        } else {
            tile_regs_wait();
        }
        for (uint32_t i = 0; i < sub_w; i++) {
            pack_tile<true>(i, out_cb, i);
        }
        tile_regs_release();
    }
}
