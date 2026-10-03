// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC projection v2, compute.  Xj (row 4*tl+i = token tl, stream i, 32 cols of column tile j), W_ij (fn^T chunk of
// stream i).
//   Z_i  = sum_j Xj @ W_ij                (row 4*tl+i of Z_i is the stream-i term of token tl; other rows unused)
//   Q    = sum_j (Xj*Xj) @ ONES_col       (column NCOL: per-row sum of squares)
//   P    = sum_i SEL_i @ Z_i + SEL_all @ Q  (row tl = token tl: mixes in columns 0..mix_hc-1, sum of squares in column
//   NCOL)

#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(3);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(4);
    constexpr uint32_t cb_z = get_compile_time_arg_val(5);
    constexpr uint32_t cb_q = get_compile_time_arg_val(6);
    constexpr uint32_t cb_p = get_compile_time_arg_val(7);
    constexpr uint32_t TPJ = get_compile_time_arg_val(8);

    compute_kernel_hw_startup(cb_x, cb_w, cb_p);
    CircularBuffer x(cb_x), w(cb_w), ones(cb_ones), sel(cb_sel), sq(cb_sq), z(cb_z), q(cb_q), p(cb_p);
    DeviceZoneScopedN("PC_ALL");
    x.wait_front(TPJ);
    w.wait_front(4 * TPJ);
    {
        DeviceZoneScopedN("PC_gotXW");
    }

    // squares
    for (uint32_t j = 0; j < TPJ; ++j) {
        reconfig_data_format(cb_x, cb_x);
        mul_tiles_init(cb_x, cb_x);
        tile_regs_acquire();
        mul_tiles(cb_x, cb_x, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        sq.reserve_back(1);
        pack_tile(0, cb_sq);
        sq.push_back(1);
        tile_regs_release();
    }
    {
        DeviceZoneScopedN("PC_sqdone");
    }
    // Z_i
    for (uint32_t i = 0; i < 4; ++i) {
        reconfig_data_format(cb_x, cb_w);
        matmul_init(cb_x, cb_w);
        tile_regs_acquire();
        for (uint32_t j = 0; j < TPJ; ++j) {
            matmul_tiles(cb_x, cb_w, j, i * TPJ + j, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        z.reserve_back(1);
        pack_tile(0, cb_z);
        z.push_back(1);
        tile_regs_release();
    }
    {
        DeviceZoneScopedN("PC_Zdone");
    }
    // Q
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
    {
        DeviceZoneScopedN("PC_Qdone");
    }
    // P
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
}
