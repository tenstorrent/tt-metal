// SPDX-License-Identifier: Apache-2.0
//
// Exercise 07 — matmul compute kernel with A-row reuse, reference solution.

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    const uint32_t Kt = get_arg_val<uint32_t>(0);
    const uint32_t Nt = get_arg_val<uint32_t>(1);
    const uint32_t n_rows = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);

    constexpr uint32_t dst = 0;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_out);
    matmul_init(cb_a, cb_b);

    for (uint32_t m = 0; m < n_rows; m++) {
        // A's row becomes visible and stays visible: no pop inside the nt loop.
        cb_wait_front(cb_a, Kt);

        for (uint32_t nt = 0; nt < Nt; nt++) {
            cb_wait_front(cb_b, Kt);

            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                // Kt tiles are visible in both windows, so the CB-relative
                // index is kt. Using 0 here would multiply the same pair Kt
                // times.
                matmul_tiles(cb_a, cb_b, kt, kt, dst);
            }
            tile_regs_commit();

            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_tile(dst, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);

            cb_pop_front(cb_b, Kt);
        }

        // Only now is A's row finished with.
        cb_pop_front(cb_a, Kt);
    }
}
