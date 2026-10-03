// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC expand + next projection, compute.
//   X'_j = A1 @ B1_j + A2 @ B2_j                       (row 4*tl + j: new stream j of token tl; handed to the writer
//   too) Z_i  = sum_j X'_j @ W_ij ;  Q = sum_j (X'_j^2) @ ONES_col ;  P = sum_i SEL_i @ Z_i + SEL_all @ Q    (see
//   mhc_proj2_compute.cpp)

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_b1 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b2 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_a = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w = get_compile_time_arg_val(3);
    constexpr uint32_t cb_xn = get_compile_time_arg_val(4);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(6);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(7);
    constexpr uint32_t cb_z = get_compile_time_arg_val(8);
    constexpr uint32_t cb_q = get_compile_time_arg_val(9);
    constexpr uint32_t cb_p = get_compile_time_arg_val(10);
    constexpr uint32_t TPJ = get_compile_time_arg_val(11);
    constexpr uint32_t cb_xw = get_compile_time_arg_val(12);  // copy of X' for the writer (the writer pops it)

    compute_kernel_hw_startup(cb_a, cb_b1, cb_xn);
    CircularBuffer b1(cb_b1), b2(cb_b2), a(cb_a), w(cb_w), xn(cb_xn), xw(cb_xw), ones(cb_ones), sel(cb_sel), sq(cb_sq),
        z(cb_z), q(cb_q), p(cb_p);
    a.wait_front(2);
    b1.wait_front(TPJ);
    b2.wait_front(TPJ);

    // expand: X'_j
    reconfig_data_format(cb_a, cb_b1);
    matmul_init(cb_a, cb_b1);
    for (uint32_t j = 0; j < TPJ; ++j) {
        tile_regs_acquire();
        matmul_tiles(cb_a, cb_b1, 0, j, 0);
        matmul_tiles(cb_a, cb_b2, 1, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        xn.reserve_back(1);
        xw.reserve_back(1);
        pack_tile(0, cb_xn);
        pack_tile(0, cb_xw);
        xn.push_back(1);
        xw.push_back(1);
        tile_regs_release();
    }
    xn.wait_front(TPJ);
    w.wait_front(4 * TPJ);

    for (uint32_t j = 0; j < TPJ; ++j) {
        reconfig_data_format(cb_xn, cb_xn);
        mul_tiles_init(cb_xn, cb_xn);
        tile_regs_acquire();
        mul_tiles(cb_xn, cb_xn, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        sq.reserve_back(1);
        pack_tile(0, cb_sq);
        sq.push_back(1);
        tile_regs_release();
    }
    for (uint32_t i = 0; i < 4; ++i) {
        reconfig_data_format(cb_xn, cb_w);
        matmul_init(cb_xn, cb_w);
        tile_regs_acquire();
        for (uint32_t j = 0; j < TPJ; ++j) {
            matmul_tiles(cb_xn, cb_w, j, i * TPJ + j, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        z.reserve_back(1);
        pack_tile(0, cb_z);
        z.push_back(1);
        tile_regs_release();
    }
    ones.wait_front(1);
    sq.wait_front(TPJ);
    reconfig_data_format(cb_sq, cb_ones);
    matmul_init(cb_sq, cb_ones);
    tile_regs_acquire();
    for (uint32_t j = 0; j < TPJ; ++j) {
        matmul_tiles(cb_sq, cb_ones, j, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    q.reserve_back(1);
    pack_tile(0, cb_q);
    q.push_back(1);
    tile_regs_release();
    sel.wait_front(5);
    z.wait_front(4);
    q.wait_front(1);
    reconfig_data_format(cb_sel, cb_z);
    matmul_init(cb_sel, cb_z);
    tile_regs_acquire();
    for (uint32_t i = 0; i < 4; ++i) {
        matmul_tiles(cb_sel, cb_z, i, i, 0);
    }
    reconfig_data_format(cb_sel, cb_q);
    matmul_init(cb_sel, cb_q);
    matmul_tiles(cb_sel, cb_q, 4, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    p.reserve_back(1);
    pack_tile(0, cb_p);
    p.push_back(1);
    tile_regs_release();
    xn.pop_front(TPJ);
}
