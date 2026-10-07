// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// gated_rmsnorm forward: per (tile-row, group) work item
//   inv = rsqrt(mean_c(x^2) + eps)
//   out = x * gamma * inv * silu(gate)

#include "gated_rmsnorm_compute_common.hpp"

void kernel_main() {
    compute_kernel_hw_startup(cb::x, cb::gamma_b, cb::out);

    cb_wait_front(cb::gamma_b, Gt);
    cb_wait_front(cb::ones, 1);

    for (uint32_t item = 0; item < work_count; ++item) {
        cb_wait_front(cb::x, Gt);
        cb_wait_front(cb::gate, Gt);

        compute_inv_rms();

        for (uint32_t j = 0; j < Gt; ++j) {
            tile_regs_acquire();
            load_tile(cb::x, j, 0);
            load_tile(cb::gamma_b, j, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            load_tile(cb::inv, 0, 1);
            mul_binary_tile(0, 1, 0);
            load_tile(cb::gate, j, 1);
            silu_tile_init();
            silu_tile(1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            pack_and_push(0, cb::out);
        }

        cb_pop_front(cb::inv, 1);
        cb_pop_front(cb::x, Gt);
        cb_pop_front(cb::gate, Gt);
    }
}
