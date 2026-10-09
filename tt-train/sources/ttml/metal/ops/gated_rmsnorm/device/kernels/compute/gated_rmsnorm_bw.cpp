// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Grouped gated RMSNorm backward, one head group of Gt tiles per work item. With u = x*gamma*inv,
// du = dy*silu(gate), t = sum_c(u*du) * inv / V:
//   dx       = inv * (gamma*du - t*x)
//   dgate    = dy * u * silu'(gate),   silu'(z) = s * (1 + z - silu(z)),  s = sigmoid(z)
//   dgamma_c = x * inv * du            (COMPUTE_DGAMMA, unreduced; the host sums it)

#include "gated_rmsnorm_compute_common.hpp"

constexpr uint32_t kOneBits = 0x3F800000U;

inline void pack_and_push_three(
    const uint32_t reg_0,
    const uint32_t cb_0,
    const uint32_t reg_1,
    const uint32_t cb_1,
    const uint32_t reg_2,
    const uint32_t cb_2) {
    cb_reserve_back(cb_0, onetile);
    cb_reserve_back(cb_1, onetile);
    cb_reserve_back(cb_2, onetile);
    tile_regs_wait();
    pack_reconfig_data_format(cb_0);
    pack_tile(reg_0, cb_0);
    pack_reconfig_data_format(cb_1);
    pack_tile(reg_1, cb_1);
    pack_reconfig_data_format(cb_2);
    pack_tile(reg_2, cb_2);
    tile_regs_release();
    cb_push_back(cb_0, onetile);
    cb_push_back(cb_1, onetile);
    cb_push_back(cb_2, onetile);
}

// cb::u, cb::du, cb::prod for every tile of the group.
inline void compute_u_du_prod() {
    for (uint32_t j = 0; j < Gt; ++j) {
        tile_regs_acquire();
        load_tile(cb::x, j, 0U);
        load_tile(cb::gamma_b, j, 1U);
        mul_regs(0U, 1U, 0U);
        load_tile(cb::inv, 0U, 1U);
        mul_regs(0U, 1U, 0U);  // r0 = u
        load_tile(cb::gate, j, 1U);
        silu_tile_init();
        silu_tile(1U);
        load_tile(cb::dy, j, 2U);
        mul_regs(2U, 1U, 2U);  // r2 = du
        mul_regs(0U, 2U, 1U);  // r1 = u * du
        tile_regs_commit();
        pack_and_push_three(0U, cb::u, 2U, cb::du, 1U, cb::prod);
    }
}

// cb::t = rowsum(sum_j prod_j) * inv / V. Leaves cb::t waited.
inline void compute_t() {
    cb_wait_front(cb::prod, Gt);
    tile_regs_acquire();
    for (uint32_t j = 0; j < Gt; ++j) {
        const uint32_t reg = j == 0 ? 0U : 1U;
        load_tile(cb::prod, j, reg);
        if (j > 0) {
            add_regs(0U, 1U, 0U);
        }
    }
    tile_regs_commit();
    cb_pop_front(cb::prod, Gt);
    pack_and_push(0U, cb::acc);

    cb_wait_front(cb::acc, onetile);
    tile_regs_acquire();
    row_sum_to_reg(cb::acc, 0U);
    load_tile(cb::inv, 0U, 1U);
    mul_regs(0U, 1U, 0U);
    binop_with_scalar_tile_init();
    mul_unary_tile(0U, inv_group_bits);
    tile_regs_commit();
    cb_pop_front(cb::acc, onetile);
    pack_and_push(0U, cb::t);
    cb_wait_front(cb::t, onetile);
}

inline void compute_dx(const uint32_t j) {
    tile_regs_acquire();
    load_tile(cb::gamma_b, j, 0U);
    load_tile(cb::du, j, 1U);
    mul_regs(0U, 1U, 0U);  // r0 = gamma * du
    load_tile(cb::x, j, 1U);
    load_tile(cb::t, 0U, 2U);
    mul_regs(1U, 2U, 1U);  // r1 = t * x
    sub_regs(0U, 1U, 0U);
    load_tile(cb::inv, 0U, 1U);
    mul_regs(0U, 1U, 0U);
    tile_regs_commit();
    pack_and_push(0U, cb::out);
}

inline void compute_dgate(const uint32_t j) {
    tile_regs_acquire();
    load_tile(cb::gate, j, 0U);  // r0 = z
    load_tile(cb::gate, j, 1U);
    sigmoid_tile_init();
    sigmoid_tile(1U);      // r1 = s
    mul_regs(0U, 1U, 2U);  // r2 = silu(z) = z * s
    sub_regs(0U, 2U, 0U);  // r0 = z - silu(z)
    binop_with_scalar_tile_init();
    add_unary_tile(0U, kOneBits);
    mul_regs(0U, 1U, 0U);  // r0 = silu'(z)
    load_tile(cb::dy, j, 1U);
    mul_regs(0U, 1U, 0U);
    load_tile(cb::u, j, 1U);
    mul_regs(0U, 1U, 0U);
    tile_regs_commit();
    pack_and_push(0U, cb::dgate);
}

#ifdef COMPUTE_DGAMMA
inline void compute_dgamma(const uint32_t j) {
    tile_regs_acquire();
    load_tile(cb::x, j, 0U);
    load_tile(cb::inv, 0U, 1U);
    mul_regs(0U, 1U, 0U);
    load_tile(cb::du, j, 1U);
    mul_regs(0U, 1U, 0U);
    tile_regs_commit();
    pack_and_push(0U, cb::dgamma);
}
#endif

void kernel_main() {
    compute_kernel_hw_startup(cb::x, cb::gamma_b, cb::out);
    cb_wait_front(cb::gamma_b, Gt);
    cb_wait_front(cb::ones, onetile);

    for (uint32_t item = 0; item < work_count; ++item) {
        cb_wait_front(cb::x, Gt);
        cb_wait_front(cb::gate, Gt);
        cb_wait_front(cb::dy, Gt);

        compute_inv_rms();
        compute_u_du_prod();
        compute_t();

        cb_wait_front(cb::u, Gt);
        cb_wait_front(cb::du, Gt);
        for (uint32_t j = 0; j < Gt; ++j) {
            compute_dx(j);
            compute_dgate(j);
#ifdef COMPUTE_DGAMMA
            compute_dgamma(j);
#endif
        }

        cb_pop_front(cb::inv, onetile);
        cb_pop_front(cb::t, onetile);
        cb_pop_front(cb::u, Gt);
        cb_pop_front(cb::du, Gt);
        cb_pop_front(cb::x, Gt);
        cb_pop_front(cb::gate, Gt);
        cb_pop_front(cb::dy, Gt);
    }

    cb_pop_front(cb::gamma_b, Gt);
    cb_pop_front(cb::ones, onetile);
}
