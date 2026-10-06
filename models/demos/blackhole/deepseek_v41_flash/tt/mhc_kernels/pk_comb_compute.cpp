// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Packed-layout mHC combine, compute.  D_q = I (.) bcast_col(CT_q) is the diagonal matrix of coefficient q (one value
// per token row); out_j(tile) = sum_i D[i,j] @ x_i(tile) [+ D_post_j @ y] [+ D_post_j @ y2]  -- row scaling through
// fp32 tile matmuls accumulated in DST.
//   NOUT == 4 (expand): D tile order 4*i+j for comb[i][j] (new stream j), then post_j at NCA+j.   NOUT == 1 (collapse):
//   D tile i = pre_i.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_i = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ct = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x = get_compile_time_arg_val(2);
    constexpr uint32_t cb_y = get_compile_time_arg_val(3);
    constexpr uint32_t cb_y2 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_d = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t NC = get_compile_time_arg_val(7);  // total coefficient tiles
    constexpr uint32_t NCA = get_compile_time_arg_val(8);
    constexpr uint32_t BLK = get_compile_time_arg_val(9);
    constexpr uint32_t NOUT = get_compile_time_arg_val(10);
    constexpr uint32_t HAS_Y = get_compile_time_arg_val(11);
    constexpr uint32_t HAS_Y2 = get_compile_time_arg_val(12);
    const uint32_t ncols = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_x, cb_d, cb_out);
    CircularBuffer ci_(cb_i), ct(cb_ct), x(cb_x), y(cb_y), y2(cb_y2), d(cb_d), out(cb_out);

    ci_.wait_front(1);
    ct.wait_front(NC);
    reconfig_data_format(cb_i, cb_ct);
    pack_reconfig_data_format(cb_d);
    mul_bcast_cols_init(cb_i, cb_ct);
    for (uint32_t q = 0; q < NC; ++q) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(cb_i, cb_ct, 0, q, 0);
        tile_regs_commit();
        tile_regs_wait();
        d.reserve_back(1);
        pack_tile(0, cb_d);
        d.push_back(1);
        tile_regs_release();
    }
    d.wait_front(NC);
    pack_reconfig_data_format(cb_out);

    for (uint32_t b = 0; b < ncols; b += BLK) {
        x.wait_front(4 * BLK);
        if constexpr (HAS_Y) {
            y.wait_front(BLK);
        }
        if constexpr (HAS_Y2) {
            y2.wait_front(BLK);
        }
        for (uint32_t ci = 0; ci < BLK; ++ci) {
            for (uint32_t j = 0; j < NOUT; ++j) {
                tile_regs_acquire();
                reconfig_data_format(cb_d, cb_x);
                matmul_init(cb_d, cb_x);
                for (uint32_t i = 0; i < 4; ++i) {
                    matmul_tiles(cb_d, cb_x, (NOUT == 4) ? (i * 4 + j) : i, ci * 4 + i, 0);
                }
                if constexpr (HAS_Y) {
                    reconfig_data_format(cb_d, cb_y);
                    matmul_init(cb_d, cb_y);
                    matmul_tiles(cb_d, cb_y, NCA + j, ci, 0);
                }
                if constexpr (HAS_Y2) {
                    reconfig_data_format(cb_d, cb_y2);
                    matmul_init(cb_d, cb_y2);
                    matmul_tiles(cb_d, cb_y2, NCA + j, ci, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                out.reserve_back(1);
                pack_tile(0, cb_out);
                out.push_back(1);
                tile_regs_release();
            }
        }
        x.pop_front(4 * BLK);
        if constexpr (HAS_Y) {
            y.pop_front(BLK);
        }
        if constexpr (HAS_Y2) {
            y2.pop_front(BLK);
        }
    }
}
