// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Step-0 microbench of the routed-expert program: the cost of one BF4 x bf16 LoFi matmul tile of moe_compute's
// compute kernel (same config: LoFi, fp32 dest off, math approx, the same hw startup and reconfigs).  No data
// movement: the CBs hold whatever the L1 holds, `reps` repeats of the unit named by the define BENCH_UNIT:
//   0 = a gate/up pair unit (K = 80, ct 4, 8 blocks of 10 K rows) with the SiLU * up pack, packed to 2 in2 tiles
//   1 = the same matmul only (no SFPU, one pack of 4 tiles)
//   2 = a W2 unit (K = 20, ct 4)
//   3 = a gate/up half unit (K = 80 stored as 40 rows, ct 2) with the SiLU * up pack
// Runtime arg 0 = reps.  Time(reps) - Time(0) over reps x tiles = the per-tile cost.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

#include "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/moe_activation.h"

void kernel_main() {
    const uint32_t reps = get_arg_val<uint32_t>(0);
    constexpr auto cb_in = tt::CBIndex::c_0;   // x tiles, Float16_b
    constexpr auto cb_out = tt::CBIndex::c_1;  // untilized rows, Float16_b
    constexpr auto cb_w = tt::CBIndex::c_3;    // weights, Bfp4_b
    constexpr auto cb_in2 = tt::CBIndex::c_6;  // gate/up tiles, Float16_b
    constexpr auto activation = ttnn::experimental::prim::detail::MoEActivationFunction::SILU;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in, cb_w, cb_in2);
    pack_reconfig_data_format(cb_in2);
    reconfig_data_format_srcb(cb_in);
    reconfig_data_format_srca(cb_w);
    MATH((ckernel::zeroacc()));
    ::moe_activation::pack_init_activation<activation>();

    for (uint32_t rep = 0; rep < reps; ++rep) {
#if BENCH_UNIT == 0 || BENCH_UNIT == 1
        matmul_block_init(cb_in, cb_w, false, 4, 1, 1);
        tile_regs_acquire();
        for (uint32_t block = 0; block < 8; ++block) {
            for (uint32_t k = 0; k < 40; k += 4) {
                matmul_block(cb_in, cb_w, block * 10 + k / 4, k, 0, false, 4, 1, 1);
            }
        }
        tile_regs_commit();
        PACK(TTI_SEMWAIT(
            p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
#if BENCH_UNIT == 0
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
        ::moe_activation::PackActivation<activation, 2>::compute();
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        pack_tile</*out_of_order_output=*/true>(0, cb_in2, 0);
        pack_tile</*out_of_order_output=*/true>(2, cb_in2, 1);
#else
        pack_tile</*out_of_order_output=*/true>(0, cb_in2, 0);
        pack_tile</*out_of_order_output=*/true>(1, cb_in2, 1);
        pack_tile</*out_of_order_output=*/true>(2, cb_in2, 2);
        pack_tile</*out_of_order_output=*/true>(3, cb_in2, 3);
#endif
        tile_regs_release();
#elif BENCH_UNIT == 2
        matmul_block_init(cb_in2, cb_w, false, 4, 1, 1);
        tile_regs_acquire();
        for (uint32_t block = 0; block < 2; ++block) {
            for (uint32_t k = 0; k < 40; k += 4) {
                matmul_block(cb_in2, cb_w, block * 10 + k / 4, k, 0, false, 4, 1, 1);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_untilize_dest_init<4, 4>(cb_out);
        pack_untilize_dest<4, 4>(cb_out, 1, 0);
        pack_untilize_uninit(cb_out);
        tile_regs_release();
        pack_reconfig_data_format(cb_in2);
#elif BENCH_UNIT == 3
        matmul_block_init(cb_in, cb_w, false, 2, 1, 1);
        tile_regs_acquire();
        for (uint32_t block = 0; block < 4; ++block) {
            for (uint32_t k = 0; k < 40; k += 2) {
                matmul_block(cb_in, cb_w, block * 20 + k / 2, k, 0, false, 2, 1, 1);
            }
        }
        tile_regs_commit();
        PACK(TTI_SEMWAIT(
            p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
        ::moe_activation::PackActivation<activation, 1>::compute();
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        pack_tile</*out_of_order_output=*/true>(0, cb_in2, 0);
        tile_regs_release();
#endif
    }
}
