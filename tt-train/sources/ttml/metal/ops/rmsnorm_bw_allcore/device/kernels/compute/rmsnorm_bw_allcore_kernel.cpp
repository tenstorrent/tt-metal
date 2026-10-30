// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// rmsnorm_bw phase B: per (tile-row, slice) work item
//   inv = 1 / rms                                   (row value broadcast along columns)
//   P   = rowsum(sum_s partial_s)                   (= sum_c a * gamma * dL_dout over the whole row)
//   t   = P * inv^2 / C
//   dL_da                = inv * (gamma * dL_dout - t * a)  [+ residual_grad with ADD_RESIDUAL_GRAD]
//   dL_dgamma_components = a * inv * dL_dout        (COMPUTE_DGAMMA; reduced over rows on host)

#include "rmsnorm_bw_allcore_compute_common.hpp"

void kernel_main() {
    compute_kernel_hw_startup(cb::dy, cb::gamma, cb::dx);
    cb_wait_front(cb::zero, 1);
    cb_wait_front(cb::ones, 1);
    cb_wait_front(cb::ones_row0, 1);

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work_start = get_arg_val<uint32_t>(0);
        const SliceGeometry sl = slice_for(work_start + item);

        cb_wait_front(cb::rms, 1);
        cb_wait_front(cb::partials, S);

        // inv = 1 / rms, broadcast along columns (rms @ ones_row0 picks column 0 of the rms tile).
        tile_regs_acquire();
        matmul_with_constant(cb::rms, cb::ones_row0, 0);
        recip_tile_init();
        recip_tile(0);
        tile_regs_commit();
        pack_and_push(0, cb::inv);

        // acc = sum_s partial_s
        tile_regs_acquire();
        load_tile(cb::partials, 0, 0);
        add_binary_tile_init();
        for (uint32_t s = 1; s < S; ++s) {
            load_tile(cb::partials, s, 1);
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        pack_and_push(0, cb::acc);

        // t = rowsum(acc) * inv * inv / C
        cb_wait_front(cb::acc, 1);
        cb_wait_front(cb::inv, 1);
        tile_regs_acquire();
        matmul_with_constant(cb::acc, cb::ones, 0);
        load_tile(cb::inv, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(0, 1, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_c_bits);
        tile_regs_commit();
        pack_and_push(0, cb::t);
        cb_pop_front(cb::acc, 1);
        cb_wait_front(cb::t, 1);

        for (uint32_t c = 0; c < sl.ncols; c += block) {
            const uint32_t n = (c + block <= sl.ncols) ? block : (sl.ncols - c);
            // Full-block CB traffic (see the reader): wait and pop `block`, use the first n tiles, and pad the
            // outputs of a shorter tail to a full block so the writer's pops never straddle the CB end either.
            cb_wait_front(cb::a, block);
            cb_wait_front(cb::gamma, block);
            cb_wait_front(cb::dy, block);
#ifdef ADD_RESIDUAL_GRAD
            cb_wait_front(cb::dres, block);
#endif
            for (uint32_t i = 0; i < n; ++i) {
                // dL_da = inv * (gamma * dy - t * a)
                tile_regs_acquire();
                gamma_times_dy(i, 0);
                load_tile(cb::t, 0, 1);
                load_tile(cb::a, i, 2);
                mul_binary_tile_init();
                mul_binary_tile(1, 2, 1);
                sub_binary_tile_init();
                sub_binary_tile(0, 1, 0);
                load_tile(cb::inv, 0, 1);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
#ifdef ADD_RESIDUAL_GRAD
                load_tile(cb::dres, i, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
#endif
                tile_regs_commit();
                pack_and_push(0, cb::dx);

#ifdef COMPUTE_DGAMMA
                // dL_dgamma_components = a * inv * dy
                tile_regs_acquire();
                load_tile(cb::a, i, 0);
                load_tile(cb::inv, 0, 1);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
                load_tile(cb::dy, i, 1);
                mul_binary_tile(0, 1, 0);
                tile_regs_commit();
                pack_and_push(0, cb::dgamma);
#endif
            }
            for (uint32_t i = n; i < block; ++i) {  // tail padding (unused tiles)
                tile_regs_acquire();
                tile_regs_commit();
                pack_and_push(0, cb::dx);
#ifdef COMPUTE_DGAMMA
                tile_regs_acquire();
                tile_regs_commit();
                pack_and_push(0, cb::dgamma);
#endif
            }
            cb_pop_front(cb::a, block);
            cb_pop_front(cb::gamma, block);
            cb_pop_front(cb::dy, block);
#ifdef ADD_RESIDUAL_GRAD
            cb_pop_front(cb::dres, block);
#endif
        }

        cb_pop_front(cb::rms, 1);
        cb_pop_front(cb::partials, S);
        cb_pop_front(cb::inv, 1);
        cb_pop_front(cb::t, 1);
    }
}
