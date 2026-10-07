// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// gated_rmsnorm backward: per (tile-row, group) work item, with
//   inv = rsqrt(mean_c(x^2) + eps),  u = x * gamma * inv,  du = dy * silu(gate)
//   t   = (sum_c u * du) * inv / group                      (per row, broadcast along columns)
//   dx     = inv * (gamma * du - t * x)
//   dgate  = dy * u * silu'(gate),  silu'(z) = s + z*s - z*s^2,  s = sigmoid(z)
//   dgamma_components = x * inv * du                        (COMPUTE_DGAMMA; reduced on host)

#include "gated_rmsnorm_compute_common.hpp"

// Pack DEST registers (r0, r1, r2) into fp32 CBs (c0, c1, c2) in one shot.
FORCE_INLINE void pack_three_f32(uint32_t r0, uint32_t c0, uint32_t r1, uint32_t c1, uint32_t r2, uint32_t c2) {
    cb_reserve_back(c0, 1);
    cb_reserve_back(c1, 1);
    cb_reserve_back(c2, 1);
    tile_regs_wait();
    pack_reconfig_data_format(c0);
    pack_tile(r0, c0);
    pack_tile(r1, c1);
    pack_tile(r2, c2);
    tile_regs_release();
    cb_push_back(c0, 1);
    cb_push_back(c1, 1);
    cb_push_back(c2, 1);
}

void kernel_main() {
    compute_kernel_hw_startup(cb::x, cb::gamma_b, cb::out);

    cb_wait_front(cb::gamma_b, Gt);
    cb_wait_front(cb::ones, 1);

    for (uint32_t item = 0; item < work_count; ++item) {
        cb_wait_front(cb::x, Gt);
        cb_wait_front(cb::gate, Gt);
        cb_wait_front(cb::dy, Gt);

        compute_inv_rms();

        // Pass 1: u_j, du_j and u_j * du_j.
        for (uint32_t j = 0; j < Gt; ++j) {
            tile_regs_acquire();
            load_tile(cb::x, j, 0);
            load_tile(cb::gamma_b, j, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            load_tile(cb::inv, 0, 1);
            mul_binary_tile(0, 1, 0);  // reg0 = u_j
            load_tile(cb::gate, j, 1);
            silu_tile_init();
            silu_tile(1);
            load_tile(cb::dy, j, 2);
            mul_binary_tile_init();
            mul_binary_tile(2, 1, 2);  // reg2 = du_j
            mul_binary_tile(0, 2, 1);  // reg1 = u_j * du_j
            tile_regs_commit();
            pack_three_f32(0, cb::u, 2, cb::du, 1, cb::prod);
        }

        // acc = sum_j prod_j
        cb_wait_front(cb::prod, Gt);
        tile_regs_acquire();
        add_binary_tile_init();
        load_tile(cb::prod, 0, 0);
        for (uint32_t j = 1; j < Gt; ++j) {
            load_tile(cb::prod, j, 1);
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        pack_and_push(0, cb::acc);
        cb_pop_front(cb::prod, Gt);

        // t = rowsum(acc) * inv / group, broadcast along columns via acc @ ones.
        cb_wait_front(cb::acc, 1);
        tile_regs_acquire();
        reconfig_data_format(cb::ones, cb::acc);
        matmul_init(cb::acc, cb::ones, 0);
        matmul_tiles(cb::acc, cb::ones, 0, 0, 0);
        load_tile(cb::inv, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_group_bits);
        tile_regs_commit();
        pack_and_push(0, cb::t);
        cb_pop_front(cb::acc, 1);

        cb_wait_front(cb::t, 1);
        cb_wait_front(cb::u, Gt);
        cb_wait_front(cb::du, Gt);

        // Pass 2: dx_j, dgate_j (and dgamma components).
        for (uint32_t j = 0; j < Gt; ++j) {
            // dx_j = inv * (gamma * du_j - t * x_j)
            tile_regs_acquire();
            load_tile(cb::du, j, 0);
            load_tile(cb::gamma_b, j, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            load_tile(cb::t, 0, 1);
            load_tile(cb::x, j, 2);
            mul_binary_tile(1, 2, 1);
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);
            load_tile(cb::inv, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            pack_and_push(0, cb::out);

            // dgate_j = dy_j * u_j * (s + z*s - z*s^2),  z = gate_j, s = sigmoid(z)
            tile_regs_acquire();
            load_tile(cb::gate, j, 0);
            load_tile(cb::gate, j, 1);
            sigmoid_tile_init();
            sigmoid_tile(1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 2);  // z*s
            mul_binary_tile(2, 1, 3);  // z*s^2
            add_binary_tile_init();
            add_binary_tile(1, 2, 1);  // s + z*s
            sub_binary_tile_init();
            sub_binary_tile(1, 3, 1);  // silu'(z)
            load_tile(cb::dy, j, 0);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            load_tile(cb::u, j, 1);
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            pack_and_push(0, cb::dgate);

#ifdef COMPUTE_DGAMMA
            // dgamma_components_j = x_j * inv * du_j
            tile_regs_acquire();
            load_tile(cb::x, j, 0);
            load_tile(cb::inv, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            load_tile(cb::du, j, 1);
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            pack_and_push(0, cb::dgamma);
#endif
        }

        cb_pop_front(cb::inv, 1);
        cb_pop_front(cb::t, 1);
        cb_pop_front(cb::u, Gt);
        cb_pop_front(cb::du, Gt);
        cb_pop_front(cb::x, Gt);
        cb_pop_front(cb::gate, Gt);
        cb_pop_front(cb::dy, Gt);
    }
}
