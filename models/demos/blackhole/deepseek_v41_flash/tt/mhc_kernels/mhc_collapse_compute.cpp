// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(2);
    constexpr uint32_t cb_hw = get_compile_time_arg_val(3);  // bf16 h, to the writer
    constexpr uint32_t cb_hs = get_compile_time_arg_val(4);  // bf16 h, for the squares
    constexpr uint32_t cb_sq = get_compile_time_arg_val(5);  // fp32 h^2
    constexpr uint32_t cb_p = get_compile_time_arg_val(6);   // fp32 partial row sums
    constexpr uint32_t T = get_compile_time_arg_val(7);
    constexpr uint32_t JB = get_compile_time_arg_val(8);
    const uint32_t ng = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_a, cb_x, cb_hw);
    CircularBuffer a(cb_a), x(cb_x), ones(cb_ones), hw(cb_hw), hs(cb_hs), sq(cb_sq), p(cb_p);
    a.wait_front(T);
    ones.wait_front(1);

    // h tiles -> bf16
    for (uint32_t g = 0; g < ng; ++g) {
        const uint32_t gl = g % JB;
        if (gl == 0) {
            x.wait_front(((ng - g < JB) ? (ng - g) : JB) * T);
        }
        reconfig_data_format(cb_a, cb_x);
        matmul_init(cb_a, cb_x);
        tile_regs_acquire();
        for (uint32_t t = 0; t < T; ++t) {
            matmul_tiles(cb_a, cb_x, t, gl * T + t, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        hw.reserve_back(1);
        hs.reserve_back(1);
        pack_tile(0, cb_hw);
        pack_tile(0, cb_hs);
        hw.push_back(1);
        hs.push_back(1);
        tile_regs_release();
        if (gl == JB - 1 || g == ng - 1) {
            x.pop_front(gl * T + T);
        }
    }

    // squares of the bf16-rounded h (exact in fp32)
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

    // partial row sums: sum_g (h^2)_g @ ones  (every column of the result = the row sum)
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
}
