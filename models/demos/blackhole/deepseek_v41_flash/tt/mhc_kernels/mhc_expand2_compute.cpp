// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t T = get_compile_time_arg_val(3);
    constexpr uint32_t G = get_compile_time_arg_val(4);
    constexpr uint32_t NG = get_compile_time_arg_val(5);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    CircularBuffer a(cb_a), b(cb_b), out(cb_out);
    a.wait_front(T);
    b.wait_front(G * NG);
    reconfig_data_format(cb_a, cb_b);
    matmul_init(cb_a, cb_b);
    for (uint32_t gi = 0; gi < NG; ++gi) {
        for (uint32_t t = 0; t < T; ++t) {
            tile_regs_acquire();
            matmul_tiles(cb_a, cb_b, t, gi * G + (t >> 2), 0);
            tile_regs_commit();
            tile_regs_wait();
            out.reserve_back(1);
            pack_tile(0, cb_out);
            out.push_back(1);
            tile_regs_release();
        }
    }
}
