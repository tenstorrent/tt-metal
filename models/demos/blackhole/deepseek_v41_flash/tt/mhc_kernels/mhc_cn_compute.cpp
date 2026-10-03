// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused mHC collapse + RMSNorm, compute.  h_g = bf16(sum_t A_t @ X_(t,g)); partial = sum_g h_g^2 @ ones; (exchange in
// the reader); rs = rsqrt(TOT + eps'); out_g = bf16(h_g * rs * w_g).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(2);
    constexpr uint32_t cb_hs = get_compile_time_arg_val(3);  // bf16 h
    constexpr uint32_t cb_sq = get_compile_time_arg_val(4);  // fp32 h^2
    constexpr uint32_t cb_p = get_compile_time_arg_val(5);   // fp32 partial
    constexpr uint32_t cb_tot = get_compile_time_arg_val(6);
    constexpr uint32_t cb_w = get_compile_time_arg_val(7);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(8);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out = get_compile_time_arg_val(10);
    constexpr uint32_t T = get_compile_time_arg_val(11);
    constexpr uint32_t G = get_compile_time_arg_val(12);
    constexpr uint32_t GPC = get_compile_time_arg_val(13);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(14);
    const uint32_t ng = GPC;

    compute_kernel_hw_startup(cb_a, cb_x, cb_hs);
    CircularBuffer a(cb_a), x(cb_x), ones(cb_ones), hs(cb_hs), sq(cb_sq), p(cb_p), tot(cb_tot), w(cb_w), rs(cb_rs),
        tmp(cb_tmp), out(cb_out);
    a.wait_front(G);
    ones.wait_front(1);

    x.wait_front(ng * G);
    reconfig_data_format(cb_a, cb_x);
    matmul_init(cb_a, cb_x);
    for (uint32_t g = 0; g < ng; ++g) {
        tile_regs_acquire();
        for (uint32_t q = 0; q < G; ++q) {
            matmul_tiles(cb_a, cb_x, q, g * G + q, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        hs.reserve_back(1);
        pack_tile(0, cb_hs);
        hs.push_back(1);
        tile_regs_release();
    }
    x.pop_front(ng * G);

    hs.wait_front(ng);
    pack_reconfig_data_format(cb_sq);
    for (uint32_t g = 0; g < ng; ++g) {
        reconfig_data_format(cb_hs, cb_hs);
        mul_tiles_init(cb_hs, cb_hs);
        tile_regs_acquire();
        mul_tiles(cb_hs, cb_hs, g, g, 0);
        tile_regs_commit();
        tile_regs_wait();
        sq.reserve_back(1);
        pack_tile(0, cb_sq);
        sq.push_back(1);
        tile_regs_release();
    }

    sq.wait_front(ng);
    reconfig_data_format(cb_sq, cb_ones);
    matmul_init(cb_sq, cb_ones);
    tile_regs_acquire();
    for (uint32_t g = 0; g < ng; ++g) {
        matmul_tiles(cb_sq, cb_ones, g, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    p.reserve_back(1);
    pack_tile(0, cb_p);
    p.push_back(1);
    tile_regs_release();

    // rs = rsqrt(R @ ones + eps')   (R[t, k] = partial sum of squares of core k, token t)
    tot.wait_front(1);
    reconfig_data_format(cb_tot, cb_ones);
    matmul_init(cb_tot, cb_ones);
    tile_regs_acquire();
    matmul_tiles(cb_tot, cb_ones, 0, 0, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    rs.reserve_back(1);
    pack_tile(0, cb_rs);
    rs.push_back(1);
    tile_regs_release();
    rs.wait_front(1);
    w.wait_front(ng);

    for (uint32_t g = 0; g < ng; ++g) {
        reconfig_data_format(cb_hs, cb_rs);
        mul_tiles_init(cb_hs, cb_rs);
        tile_regs_acquire();
        mul_tiles(cb_hs, cb_rs, g, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        tmp.reserve_back(1);
        pack_tile(0, cb_tmp);
        tmp.push_back(1);
        tile_regs_release();
    }
    tmp.wait_front(ng);
    pack_reconfig_data_format(cb_out);
    for (uint32_t g = 0; g < ng; ++g) {
        reconfig_data_format(cb_tmp, cb_w);
        mul_tiles_init(cb_tmp, cb_w);
        tile_regs_acquire();
        mul_tiles(cb_tmp, cb_w, g, g, 0);
        tile_regs_commit();
        tile_regs_wait();
        out.reserve_back(1);
        pack_tile(0, cb_out);
        out.push_back(1);
        tile_regs_release();
    }
}
