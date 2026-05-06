// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for mul-relu-overlap: y = relu(a * b), bfloat16, FPU + SFPU
// both on MATH (TRISC1). One tile per cycle to match the binary_ng-style
// dataflow kernel pacing.
//
// Runtime args:
//   [0] num_tiles - tiles to process for this core

#include <cstdint>

#include "api/compute/eltwise_binary.h"

#ifdef TRISC_PACK
#include "api/compute/eltwise_unary/relu.h"
#endif  // TRISC_PACK

ALWI void relu_packthread_tile_init() { PACK(SFPU_UNARY_KERNEL_INIT(relu_min, APPROX)); }

ALWI void relu_packthread_tile(uint32_t idst) {
    PACK(SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_min_, RC, APPROX, idst, 0));
}

#define OVERLAP_MATH_PACK 1

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto c_a = tt::CBIndex::c_0;
    constexpr auto c_b = tt::CBIndex::c_1;
    constexpr auto c_out = tt::CBIndex::c_2;

    binary_op_init_common(c_a, c_b, c_out);
    mul_tiles_init(c_a, c_b);

    MATH((ckernel::zeroacc()));

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(c_a, 1);
        cb_wait_front(c_b, 1);
        cb_reserve_back(c_out, 1);

        tile_regs_acquire();
        mul_tiles(c_a, c_b, 0, 0, 0);  // FPU: A * B -> DST[0]

#ifdef OVERLAP_MATH_PACK
        tile_regs_commit();

        // tile_regs_wait();
        PACK(TTI_SEMWAIT(
                 p_stall::STALL_TDMA | p_stall::STALL_CFG,
                 semaphore::t6_sem(semaphore::MATH_PACK),
                 p_stall::STALL_ON_ZERO););
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()););

        // SFPU: relu(DST[0]) in place
        PACK(relu_packthread_tile_init(); relu_packthread_tile(0);)

        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);)
#else
        relu_tile_init();
        relu_tile(0);
        tile_regs_commit();

        tile_regs_wait();
#endif
        pack_tile(0, c_out);
        tile_regs_release();

        cb_push_back(c_out, 1);
        cb_pop_front(c_a, 1);
        cb_pop_front(c_b, 1);
    }
}
