// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// out = h * rsqrt(sum_k R + eps') * w'   (R: gathered partial row sums; eps' = C*eps; w' = w*sqrt(C))

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
    constexpr uint32_t cb_r = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(1);
    constexpr uint32_t cb_h = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w = get_compile_time_arg_val(3);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(4);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(7);
    const uint32_t ng = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_r, cb_ones, cb_tmp);  // fp32 outputs first
    CircularBuffer r(cb_r), ones(cb_ones), h(cb_h), w(cb_w), rs(cb_rs), tmp(cb_tmp), out(cb_out);
    r.wait_front(1);
    ones.wait_front(1);
    h.wait_front(ng);
    w.wait_front(ng);

    // rs = rsqrt(R @ ones + eps')
    reconfig_data_format(cb_r, cb_ones);
    matmul_init(cb_r, cb_ones);
    tile_regs_acquire();
    matmul_tiles(cb_r, cb_ones, 0, 0, 0);
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

    // tmp_g = h_g * rs  (fp32)
    for (uint32_t g = 0; g < ng; ++g) {
        reconfig_data_format(cb_h, cb_rs);
        mul_tiles_init(cb_h, cb_rs);
        tile_regs_acquire();
        mul_tiles(cb_h, cb_rs, g, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        tmp.reserve_back(1);
        pack_tile(0, cb_tmp);
        tmp.push_back(1);
        tile_regs_release();
    }
    tmp.wait_front(ng);

    // out_g = tmp_g * w_g  (bf16)
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
