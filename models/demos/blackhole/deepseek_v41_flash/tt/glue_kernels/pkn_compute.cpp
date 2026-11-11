// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// T19 packed-mHC norm apply, compute.  rs = rsqrt(sum_p part_p * (1/C) + eps) (every column of a row holds the row's
// value); out_j = bf16((h_j * rs) * w_j), the weight tile broadcast down its rows (row 0).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
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
    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(1);
    constexpr uint32_t cb_h = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w = get_compile_time_arg_val(3);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(4);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t NRC = get_compile_time_arg_val(7);
    constexpr uint32_t INV_BITS = get_compile_time_arg_val(8);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(9);
    const uint32_t cn = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_part, cb_ones, cb_tmp);
    CircularBuffer part(cb_part), ones(cb_ones), h(cb_h), w(cb_w), rs(cb_rs), tmp(cb_tmp), out(cb_out);
    part.wait_front(NRC);
    ones.wait_front(1);

    reconfig_data_format(cb_part, cb_ones);
    matmul_init(cb_part, cb_ones);
    tile_regs_acquire();
    for (uint32_t p = 0; p < NRC; ++p) {
        matmul_tiles(cb_part, cb_ones, p, 0, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_BITS);
    add_unary_tile(0, EPS_BITS);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    rs.reserve_back(1);
    pack_tile(0, cb_rs);
    rs.push_back(1);
    tile_regs_release();
    rs.wait_front(1);

    h.wait_front(cn);
    w.wait_front(cn);
    for (uint32_t j = 0; j < cn; ++j) {
        reconfig_data_format(cb_h, cb_rs);
        mul_init(cb_h, cb_rs);
        tile_regs_acquire();
        mul_tiles(cb_h, cb_rs, j, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        tmp.reserve_back(1);
        pack_tile(0, cb_tmp);
        tmp.push_back(1);
        tile_regs_release();

        tmp.wait_front(1);
        pack_reconfig_data_format(cb_out);
        reconfig_data_format(cb_tmp, cb_w);
        mul_bcast_rows_init(cb_tmp, cb_w);
        tile_regs_acquire();
        mul_tiles_bcast_rows(cb_tmp, cb_w, 0, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        out.reserve_back(1);
        pack_tile(0, cb_out);
        out.push_back(1);
        tile_regs_release();
        tmp.pop_front(1);
        pack_reconfig_data_format(cb_tmp);
    }
}
