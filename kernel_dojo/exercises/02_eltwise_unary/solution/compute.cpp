// SPDX-License-Identifier: Apache-2.0
//
// Exercise 02 — compute kernel, reference solution.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);

    constexpr uint32_t dst = 0;

    // One-time hardware configuration: unpacker/packer data formats and DST
    // sync mode. Does MMIO writes, so it must come before any other compute
    // call and must not be repeated mid-kernel.
    init_sfpu(cb_in, cb_out);
    // Program the SFPU for exponential. Needed once per kind of SFPU op.
    exp_tile_init();

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_in, 1);

        tile_regs_acquire();
        // The SFPU cannot read a CB directly, so stage the tile in DST first.
        copy_tile(cb_in, 0, dst);
        exp_tile(dst);
        tile_regs_commit();

        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(dst, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
    }
}
