// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Grouped gated RMSNorm forward, one head group of Gt tiles per work item:
//   out = x * gamma * rsqrt(mean_V(x^2) + eps) * silu(gate)

#include "gated_rmsnorm_compute_common.hpp"

void kernel_main() {
    compute_kernel_hw_startup(cb::x, cb::gamma_b, cb::out);
    cb_wait_front(cb::gamma_b, Gt);
    cb_wait_front(cb::ones, onetile);

    for (uint32_t item = 0; item < work_count; ++item) {
        cb_wait_front(cb::x, Gt);
        cb_wait_front(cb::gate, Gt);

        compute_inv_rms();

        for (uint32_t j = 0; j < Gt; ++j) {
            tile_regs_acquire();
            load_tile(cb::x, j, 0U);
            load_tile(cb::gamma_b, j, 1U);
            mul_regs(0U, 1U, 0U);
            load_tile(cb::inv, 0U, 1U);
            mul_regs(0U, 1U, 0U);
            load_tile(cb::gate, j, 1U);
            silu_tile_init();
            silu_tile(1U);
            mul_regs(0U, 1U, 0U);
            tile_regs_commit();
            pack_and_push(0U, cb::out);
        }

        cb_pop_front(cb::inv, onetile);
        cb_pop_front(cb::x, Gt);
        cb_pop_front(cb::gate, Gt);
    }

    cb_pop_front(cb::gamma_b, Gt);
    cb_pop_front(cb::ones, onetile);
}
