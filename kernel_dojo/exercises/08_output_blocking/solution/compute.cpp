// SPDX-License-Identifier: Apache-2.0
//
// Exercise 08 — 2-D blocked matmul compute kernel, reference solution.
//
// One B column, Mb output tiles. Each B tile that crosses the NoC now feeds Mb
// accumulations instead of one, which is what finally makes the FPU the
// bottleneck rather than DRAM.

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
        // Mb x Kt tiles of A, resident for the whole block.
        cb_wait_front(cb_a, Mb * Kt);

        for (uint32_t nt = 0; nt < Nt; nt++) {
            cb_wait_front(cb_b, Kt);
            cb_reserve_back(cb_out, Mb);

            // Mb independent accumulators, one DST slot each. DST holds 8
            // tiles in half-sync mode, so Mb <= 8.
            tile_regs_acquire();
            for (uint32_t m = 0; m < Mb; m++) {
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    // A tile (m, kt) is at window index m * Kt + kt; the B
                    // tile is shared across all m — that reuse is the point.
                    matmul_tiles(cb_a, cb_b, m * Kt + kt, kt, m);
                }
            }
            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t m = 0; m < Mb; m++) {
                pack_tile(m, cb_out);
            }
            tile_regs_release();

            cb_push_back(cb_out, Mb);
            cb_pop_front(cb_b, Kt);
        }

        cb_pop_front(cb_a, Mb * Kt);
    }
}
