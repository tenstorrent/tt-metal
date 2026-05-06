// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for mul-relu-add-overlap: y = relu(a * b) + c, bfloat16.
// Naive sequential baseline — FPU mul and SFPU relu and FPU add all run on
// MATH (TRISC1), in order, one tile per cycle. A future iteration will move
// relu to PACK (TRISC2) to overlap with the surrounding FPU ops.
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
    constexpr auto c_c = tt::CBIndex::c_2;
    constexpr auto c_out = tt::CBIndex::c_3;

    binary_op_init_common(c_a, c_b, c_out);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(c_a, 1);
        cb_wait_front(c_b, 1);
        cb_wait_front(c_c, 1);
        cb_reserve_back(c_out, 1);

        tile_regs_acquire();

        // FPU: A * B -> DST[0]
        mul_tiles_init(c_a, c_b);
        mul_tiles(c_a, c_b, 0, 0, 0);

        // SFPU: relu(DST[0]) in place
        relu_tile_init();
        relu_tile(0);

        // FPU: DST[0] + C -> DST[0] via DST->SrcA reuse
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_c);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_c, 0, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, c_out);
        tile_regs_release();

        cb_push_back(c_out, 1);
        cb_pop_front(c_a, 1);
        cb_pop_front(c_b, 1);
        cb_pop_front(c_c, 1);
    }
}
