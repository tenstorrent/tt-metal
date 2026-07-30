// SPDX-License-Identifier: Apache-2.0
//
// Exercise 05 — compute kernel, reference solution.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t block = get_compile_time_arg_val(3);

    binary_op_init_common(cb_a, cb_b, cb_out);
    add_tiles_init(cb_a, cb_b);

    for (uint32_t i = 0; i < n_tiles; i += block) {
        cb_wait_front(cb_a, block);
        cb_wait_front(cb_b, block);
        cb_reserve_back(cb_out, block);

        // One handshake for the whole block instead of one per tile.
        tile_regs_acquire();
        for (uint32_t t = 0; t < block; t++) {
            // t indexes the CB's visible window (block tiles wide) and the DST
            // slot. DST holds 8 tiles in half-sync mode, so block <= 8.
            add_tiles(cb_a, cb_b, t, t, t);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t t = 0; t < block; t++) {
            pack_tile(t, cb_out);
        }
        tile_regs_release();

        cb_push_back(cb_out, block);
        cb_pop_front(cb_a, block);
        cb_pop_front(cb_b, block);
    }
}
