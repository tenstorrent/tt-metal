// SPDX-License-Identifier: Apache-2.0
//
// Exercise 04 — compute kernel, reference solution.
//
// Identical to lesson 03. The compute kernel sees only circular buffers, so
// distributing work across cores does not change it at all.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);

    constexpr uint32_t dst = 0;

    binary_op_init_common(cb_a, cb_b, cb_out);
    add_tiles_init(cb_a, cb_b);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);

        tile_regs_acquire();
        // The FPU reads both operands from L1 directly — no staging in DST.
        add_tiles(cb_a, cb_b, 0, 0, dst);
        tile_regs_commit();

        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(dst, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }
}
