// SPDX-License-Identifier: Apache-2.0
//
// Exercise 07 — matmul compute kernel with A-row reuse.
//
// A's row stays resident in cb_a for all Nt output tiles of that row. Pop cb_b
// after each output tile; pop cb_a only once the row is finished.

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
        // TODO: wait for the whole row of A — Kt tiles — but do NOT pop it yet.

        for (uint32_t nt = 0; nt < Nt; nt++) {
            // TODO: wait for Kt tiles of B.

            // TODO: acquire DST, accumulate Kt tile-matmuls into slot `dst`.
            //       With Kt tiles visible in each window, the CB-relative
            //       index is kt for both operands.

            // TODO: commit, reserve an output page, wait, pack, release, push.

            // TODO: pop Kt tiles of B — this column is done.
        }

        // TODO: now the row is finished, pop A's Kt tiles.
    }
}
