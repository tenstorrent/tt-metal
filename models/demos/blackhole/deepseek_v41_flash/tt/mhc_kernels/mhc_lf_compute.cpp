// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC lane-parallel fused expand + projection (DSV41_MHC_LANEPAR), compute: the new x tiles come from the expand matmul
// (see mhc_lf_reader.cpp), then the projection of mhc_proj5_compute. (v3 text:) mHC projection v3 (NGRP token groups
// per core), compute.  Xj (row 4*tl+i = token tl, stream i, 32 cols of column tile j), W_ij (fn^T chunk of stream i).
//   Z_i  = sum_j Xj @ W_ij                (row 4*tl+i of Z_i is the stream-i term of token tl; other rows unused)
//   Q    = sum_j (Xj*Xj) @ ONES_col       (column NCOL: per-row sum of squares)
//   P    = sum_i SEL_i @ Z_i + SEL_all @ Q  (row tl = token tl: mixes in columns 0..mix_hc-1, sum of squares in column
//   NCOL)

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"
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
    constexpr uint32_t NGRP = get_compile_time_arg_val(9);
    constexpr uint32_t cb_a = get_compile_time_arg_val(10);
    constexpr uint32_t cb_b = get_compile_time_arg_val(11);

    compute_kernel_hw_startup(cb_a, cb_b, cb_x);
    CircularBuffer x(cb_x), w(cb_w), ones(cb_ones), sel(cb_sel), sq(cb_sq), z(cb_z), q(cb_q), p(cb_p), a(cb_a), b(cb_b);
    DeviceZoneScopedN("PC_ALL");
    // LANEPAR fused expand: x_new tile (g, j) = A_g @ B_(g, j) (the mHC expand, one fp32 tile matmul per group and
    // column tile, as mhc_expand4_compute), packed into cb_x = the X tile of the projection. All groups first: the
    // reader forwards / stores them independent of the weights.
    for (uint32_t g = 0; g < NGRP; ++g) {
        a.wait_front(g + 1);
        b.wait_front((g + 1) * TPJ);
        reconfig_data_format(cb_a, cb_b);
        matmul_init(cb_a, cb_b);
        for (uint32_t j = 0; j < TPJ; ++j) {
            tile_regs_acquire();
            matmul_tiles(cb_a, cb_b, g, g * TPJ + j, 0);
            tile_regs_commit();
            tile_regs_wait();
            x.reserve_back(1);
            pack_tile(0, cb_x);
            x.push_back(1);
            tile_regs_release();
        }
    }
    {
        DeviceZoneScopedN("PC_expdone");
    }

    for (uint32_t g = 0; g < NGRP; ++g) {
        const uint32_t xo = g * TPJ;
        x.wait_front(xo + TPJ);  // the x rows of group g have landed (the reader pushes them group by group)
        // squares
        for (uint32_t j = 0; j < TPJ; ++j) {
            reconfig_data_format(cb_x, cb_x);
            mul_init(cb_x, cb_x);
            tile_regs_acquire();
            mul_tiles(cb_x, cb_x, xo + j, xo + j, 0);
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
        if (g == 0) {
            w.wait_front(4 * TPJ);
            DeviceZoneScopedN("PC_gotXW");
        }
        // Z_i
        for (uint32_t i = 0; i < 4; ++i) {
            reconfig_data_format(cb_x, cb_w);
            matmul_init(cb_x, cb_w);
            tile_regs_acquire();
            for (uint32_t j = 0; j < TPJ; ++j) {
                matmul_tiles(cb_x, cb_w, xo + j, i * TPJ + j, 0);
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
        z.pop_front(4);
        q.pop_front(1);
        sq.pop_front(TPJ);
    }
}
