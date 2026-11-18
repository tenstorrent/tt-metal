// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B47c fused MTP RMSNorm, compute. xn = x (+ residual) in fp32 (SFPU add, one rounding); partial = sum_g xn_g^2 @ ones;
// (exchange in the reader); rs = rsqrt(R @ ones + eps'); out_g = bf16((xn_g * rs) * w_g) with fp32 SFPU multiplies (as
// the stock _rms6 chain: multiply, multiply(dtype=bf16)).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(3);
    constexpr uint32_t cb_xn = get_compile_time_arg_val(4);
    constexpr uint32_t cb_xo = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(6);
    constexpr uint32_t cb_p = get_compile_time_arg_val(7);
    constexpr uint32_t cb_tot = get_compile_time_arg_val(8);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out = get_compile_time_arg_val(10);
    constexpr uint32_t GPC = get_compile_time_arg_val(11);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(12);
    constexpr uint32_t HAS_R = get_compile_time_arg_val(13);
    constexpr uint32_t OUT_X = get_compile_time_arg_val(14);
    constexpr uint32_t cb_s = HAS_R ? cb_xn : cb_x;  // the (summed) x the rest reads

    compute_kernel_hw_startup(cb_x, cb_ones, cb_sq);
    CircularBuffer x(cb_x), r(cb_r), w(cb_w), ones(cb_ones), xn(cb_xn), xo(cb_xo), sq(cb_sq), p(cb_p), tot(cb_tot),
        rs(cb_rs), out(cb_out);
    x.wait_front(GPC);
    ones.wait_front(1);

    if constexpr (HAS_R) {
        r.wait_front(GPC);
        pack_reconfig_data_format(cb_xn);
        for (uint32_t g = 0; g < GPC; ++g) {
            tile_regs_acquire();
            reconfig_data_format_srca(cb_x);
            copy_init(cb_x);
            copy_tile(cb_x, g, 0);
            reconfig_data_format_srca(cb_r);
            copy_init(cb_r);
            copy_tile(cb_r, g, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            xn.reserve_back(1);
            pack_tile(0, cb_xn);
            xn.push_back(1);
            if constexpr (OUT_X) {
                xo.reserve_back(1);
                pack_tile(0, cb_xo);
                xo.push_back(1);
            }
            tile_regs_release();
        }
        xn.wait_front(GPC);
    }

    // squares (fp32, exact SFPU multiply)
    pack_reconfig_data_format(cb_sq);
    for (uint32_t g = 0; g < GPC; ++g) {
        tile_regs_acquire();
        reconfig_data_format_srca(cb_s);
        copy_init(cb_s);
        copy_tile(cb_s, g, 0);
        copy_tile(cb_s, g, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        sq.reserve_back(1);
        pack_tile(0, cb_sq);
        sq.push_back(1);
        tile_regs_release();
    }

    // partial row sums: sum_g sq_g @ ones (every column of the result = the row sum)
    sq.wait_front(GPC);
    reconfig_data_format(cb_sq, cb_ones);
    matmul_init(cb_sq, cb_ones);
    tile_regs_acquire();
    for (uint32_t g = 0; g < GPC; ++g) {
        matmul_tiles(cb_sq, cb_ones, g, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    p.reserve_back(1);
    pack_tile(0, cb_p);
    p.push_back(1);
    tile_regs_release();

    // rs = rsqrt(R @ ones + eps')
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
    w.wait_front(GPC);

    pack_reconfig_data_format(cb_out);
    for (uint32_t g = 0; g < GPC; ++g) {
        tile_regs_acquire();
        reconfig_data_format_srca(cb_s);
        copy_init(cb_s);
        copy_tile(cb_s, g, 0);
        reconfig_data_format_srca(cb_rs);
        copy_init(cb_rs);
        copy_tile(cb_rs, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        reconfig_data_format_srca(cb_w);
        copy_init(cb_w);
        copy_tile(cb_w, g, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        out.reserve_back(1);
        pack_tile(0, cb_out);
        out.push_back(1);
        tile_regs_release();
    }
}
