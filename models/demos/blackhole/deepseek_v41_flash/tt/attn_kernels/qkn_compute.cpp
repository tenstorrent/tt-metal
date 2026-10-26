// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// qk_norm (bit-exact to ttnn.rms_norm), compute: the LLK sequence of the reference op (layernorm.cpp, RMSNORM + gamma,
// one tile row):
//   sq = x * x (own tiles);  [reducer: ex = mean(sq) = reduce_tile SUM / REDUCE_ROW over ALL tiles of the row in order
//   (SQALL), * (1/N); rs = rsqrt(ex + eps)];  xr = x * rs (column broadcast);  out = xr * gamma (row broadcast) (own
//   tiles).
// bf16 intermediates, fp32_dest_acc off, approx mode on (the reference's actual configuration).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(2);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(3);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sqall = get_compile_time_arg_val(5);
    constexpr uint32_t cb_ex = get_compile_time_arg_val(6);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(7);
    constexpr uint32_t cb_xr = get_compile_time_arg_val(8);
    constexpr uint32_t cb_out = get_compile_time_arg_val(9);
    const uint32_t gpc = get_arg_val<uint32_t>(0);
    const uint32_t inv_n_bits = get_arg_val<uint32_t>(1);  // bits of fp32 1/N
    const uint32_t is_red = get_arg_val<uint32_t>(2);
    const uint32_t wt = get_arg_val<uint32_t>(3);  // tiles of the whole row

    compute_kernel_hw_startup(cb_x, cb_x, cb_sq);
    CircularBuffer x(cb_x), w(cb_w), sc(cb_sc), eps(cb_eps), sq(cb_sq), sqall(cb_sqall), ex(cb_ex), rs(cb_rs),
        xr(cb_xr), out(cb_out);
    eps.wait_front(1);
    sc.wait_front(1);
    x.wait_front(gpc);

    // sq = x^2
    reconfig_data_format(cb_x, cb_x);
    pack_reconfig_data_format(cb_sq);
    mul_init(cb_x, cb_x);
    for (uint32_t g = 0; g < gpc; ++g) {
        tile_regs_acquire();
        mul_tiles(cb_x, cb_x, g, g, 0);
        tile_regs_commit();
        tile_regs_wait();
        sq.reserve_back(1);
        pack_tile(0, cb_sq);
        sq.push_back(1);
        tile_regs_release();
    }

    if (is_red) {
        // ex = mean(sq)
        sqall.wait_front(wt);
        reconfig_data_format(cb_sqall, cb_sc);
        tile_regs_acquire();
        reconfig_data_format(cb_sc, cb_sqall);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sqall, cb_sc, cb_ex);
        for (uint32_t g = 0; g < wt; ++g) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sqall, cb_sc, g, 0, 0);
        }
        reduce_uninit();
        reconfig_data_format(cb_sqall, cb_sc);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_n_bits);
        tile_regs_commit();
        tile_regs_wait();
        ex.reserve_back(1);
        pack_reconfig_data_format(cb_ex);
        pack_tile(0, cb_ex);
        tile_regs_release();
        ex.push_back(1);
        ex.wait_front(1);

        // rs = rsqrt(ex + eps)
        reconfig_data_format(cb_ex, cb_eps);
        add_init(cb_ex, cb_eps);
        tile_regs_acquire();
        add_tiles(cb_ex, cb_eps, 0, 0, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        rs.reserve_back(1);
        pack_reconfig_data_format(cb_rs);
        pack_tile(0, cb_rs);
        tile_regs_release();
        rs.push_back(1);
        rs.wait_front(1);
    } else {
        rs.wait_front(1);
    }

    // xr = x * rs (column broadcast)
    reconfig_data_format(cb_x, cb_rs);
    pack_reconfig_data_format(cb_xr);
    mul_bcast_cols_init(cb_x, cb_rs);
    for (uint32_t g = 0; g < gpc; ++g) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(cb_x, cb_rs, g, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        xr.reserve_back(1);
        pack_tile(0, cb_xr);
        xr.push_back(1);
        tile_regs_release();
    }

    // out = xr * gamma (row broadcast)
    w.wait_front(gpc);
    reconfig_data_format(cb_xr, cb_w);
    pack_reconfig_data_format(cb_out);
    mul_bcast_rows_init(cb_xr, cb_w);
    for (uint32_t g = 0; g < gpc; ++g) {
        xr.wait_front(1);
        tile_regs_acquire();
        mul_tiles_bcast_rows(cb_xr, cb_w, 0, g, 0);
        tile_regs_commit();
        tile_regs_wait();
        out.reserve_back(1);
        pack_tile(0, cb_out);
        out.push_back(1);
        tile_regs_release();
        xr.pop_front(1);
    }
}
