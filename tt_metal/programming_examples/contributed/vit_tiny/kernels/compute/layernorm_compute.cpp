// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// LayerNorm across last dimension (row-wise) with gamma and beta.
// Processes Mt tile-rows, each Wt tiles wide.
//
// CB layout:
//   c_0 (cb_in):      input tiles, Wt per row
//   c_1 (cb_scaler):  tile with 1/N values for mean computation (persistent)
//   c_2 (cb_gamma):   gamma weights, Wt tiles (persistent, broadcast-row format)
//   c_3 (cb_beta):    beta weights, Wt tiles (persistent, broadcast-row format)
//   c_4 (cb_eps):     eps tile (persistent)
//   c_5 (cb_mean):    mean intermediate (1 tile)
//   c_6 (cb_xmm):     x - mean intermediate (Wt tiles)
//   c_7 (cb_xmm2):    (x-mean)^2 intermediate (Wt tiles)
//   c_8 (cb_var):     variance intermediate (1 tile)
//   c_9 (cb_norm):    normalized intermediate (Wt tiles)
//   c_16 (cb_out):    output tiles

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"

using namespace ckernel;

void kernel_main() {
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Wt = get_compile_time_arg_val(1);

    constexpr tt::CBIndex cb_in = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_scaler = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_gamma = tt::CBIndex::c_2;
    constexpr tt::CBIndex cb_beta = tt::CBIndex::c_3;
    constexpr tt::CBIndex cb_eps = tt::CBIndex::c_4;
    constexpr tt::CBIndex cb_mean = tt::CBIndex::c_5;
    constexpr tt::CBIndex cb_xmm = tt::CBIndex::c_6;
    constexpr tt::CBIndex cb_xmm2 = tt::CBIndex::c_7;
    constexpr tt::CBIndex cb_var = tt::CBIndex::c_8;
    constexpr tt::CBIndex cb_norm = tt::CBIndex::c_9;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    binary_op_init_common(cb_in, cb_scaler, cb_mean);

    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_gamma, Wt);
    cb_wait_front(cb_beta, Wt);
    cb_wait_front(cb_eps, 1);

    for (uint32_t mt = 0; mt < Mt; mt++) {
        cb_wait_front(cb_in, Wt);

        // 1. Mean: reduce_sum(x) * (1/N) via scaler
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_mean);
        tile_regs_acquire();
        for (uint32_t n = 0; n < Wt; n++) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, n, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_mean, 1);
        pack_tile(0, cb_mean);
        cb_push_back(cb_mean, 1);
        tile_regs_release();
        reduce_uninit();

        // 2. x - mean (broadcast column since mean is a column vector from reduce_row)
        cb_wait_front(cb_mean, 1);
        sub_bcast_cols_init_short(cb_in, cb_mean);
        for (uint32_t n = 0; n < Wt; n++) {
            tile_regs_acquire();
            sub_tiles_bcast_cols(cb_in, cb_mean, n, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_xmm, 1);
            pack_tile(0, cb_xmm);
            cb_push_back(cb_xmm, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_in, Wt);
        cb_pop_front(cb_mean, 1);

        // 3. (x - mean)^2: multiply each xmm tile by itself
        cb_wait_front(cb_xmm, Wt);
        mul_tiles_init(cb_xmm, cb_xmm);
        for (uint32_t n = 0; n < Wt; n++) {
            tile_regs_acquire();
            mul_tiles(cb_xmm, cb_xmm, n, n, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_xmm2, 1);
            pack_tile(0, cb_xmm2);
            cb_push_back(cb_xmm2, 1);
            tile_regs_release();
        }

        // 4. Variance: reduce_sum((x-mean)^2) * (1/N)
        cb_wait_front(cb_xmm2, Wt);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_xmm2, cb_scaler, cb_var);
        tile_regs_acquire();
        for (uint32_t n = 0; n < Wt; n++) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_xmm2, cb_scaler, n, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_var, 1);
        pack_tile(0, cb_var);
        cb_push_back(cb_var, 1);
        tile_regs_release();
        reduce_uninit();
        cb_pop_front(cb_xmm2, Wt);

        // 5. rsqrt(var + eps)
        cb_wait_front(cb_var, 1);
        add_tiles_init(cb_var, cb_eps);
        tile_regs_acquire();
        add_tiles(cb_var, cb_eps, 0, 0, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        cb_pop_front(cb_var, 1);
        cb_reserve_back(cb_var, 1);
        pack_tile(0, cb_var);
        cb_push_back(cb_var, 1);
        tile_regs_release();

        // 6. Normalize: (x - mean) * rsqrt(var + eps)
        cb_wait_front(cb_var, 1);
        mul_bcast_cols_init_short(cb_xmm, cb_var);
        for (uint32_t n = 0; n < Wt; n++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_xmm, cb_var, n, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_norm, 1);
            pack_tile(0, cb_norm);
            cb_push_back(cb_norm, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_xmm, Wt);
        cb_pop_front(cb_var, 1);

        // 7. Apply gamma (broadcast row): normalized * gamma -> cb_xmm (reused)
        cb_wait_front(cb_norm, Wt);
        mul_bcast_rows_init_short(cb_norm, cb_gamma);
        for (uint32_t n = 0; n < Wt; n++) {
            tile_regs_acquire();
            mul_tiles_bcast_rows(cb_norm, cb_gamma, n, n, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_xmm, 1);
            pack_tile(0, cb_xmm);
            cb_push_back(cb_xmm, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_norm, Wt);

        // 8. Apply beta (broadcast row): gamma_scaled + beta -> cb_out
        cb_wait_front(cb_xmm, Wt);
        add_bcast_rows_init_short(cb_xmm, cb_beta);
        for (uint32_t n = 0; n < Wt; n++) {
            tile_regs_acquire();
            add_tiles_bcast_rows(cb_xmm, cb_beta, n, n, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_xmm, Wt);
    }
}
