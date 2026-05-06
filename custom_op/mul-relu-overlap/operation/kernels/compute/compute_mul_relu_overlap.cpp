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
#include "api/compute/eltwise_unary/relu.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto c_a = tt::CBIndex::c_0;
    constexpr auto c_b = tt::CBIndex::c_1;
    constexpr auto c_out = tt::CBIndex::c_2;

    binary_op_init_common(c_a, c_b, c_out);
    mul_tiles_init(c_a, c_b);
    relu_tile_init();

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(c_a, 1);
        cb_wait_front(c_b, 1);
        cb_reserve_back(c_out, 1);

        tile_regs_acquire();
        mul_tiles(c_a, c_b, 0, 0, 0);  // FPU: A * B -> DST[0]
        relu_tile(0);                  // SFPU: relu(DST[0]) in place
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, c_out);
        tile_regs_release();

        cb_push_back(c_out, 1);
        cb_pop_front(c_a, 1);
        cb_pop_front(c_b, 1);
    }
}
