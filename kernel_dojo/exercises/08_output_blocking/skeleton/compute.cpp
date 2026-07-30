// SPDX-License-Identifier: Apache-2.0
//
// Exercise 08 — 2-D blocked matmul compute kernel.
//
// For each B column, produce Mb output tiles at once using Mb separate DST
// accumulators. Every B tile read now feeds Mb accumulations instead of one.
//
// DST holds 8 tiles in half-sync mode, so Mb <= 8.

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    const uint32_t Kt = get_arg_val<uint32_t>(0);
    const uint32_t Nt = get_arg_val<uint32_t>(1);
    const uint32_t n_blocks = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t Mb = get_compile_time_arg_val(3);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_out);
    matmul_init(cb_a, cb_b);

    for (uint32_t blk = 0; blk < n_blocks; blk++) {
        // TODO: wait for the whole Mb * Kt sub-block of A. Don't pop it until
        //       the block is finished.

        for (uint32_t nt = 0; nt < Nt; nt++) {
            // TODO: wait for Kt tiles of B, reserve Mb output pages.

            // TODO: acquire DST once, then for m in [0, Mb) and kt in [0, Kt):
            //         matmul_tiles(cb_a, cb_b, <A window index>, kt, m)
            //       A's tile (m, kt) is at window index m * Kt + kt.
            //       The same B tile is used for every m — that reuse is the
            //       entire point of this exercise.

            // TODO: commit, wait, pack all Mb DST slots, release.

            // TODO: push Mb output pages, pop Kt tiles of B.
        }

        // TODO: pop A's Mb * Kt tiles.
    }
}
