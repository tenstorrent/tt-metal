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
    const uint32_t ng = get_arg_val<uint32_t>(0);
    const uint32_t nb = ng * T;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    CircularBuffer a(cb_a), b(cb_b), out(cb_out);
    a.wait_front(T);
    b.wait_front(nb);
    for (uint32_t n = 0; n < nb; ++n) {
        reconfig_data_format(cb_a, cb_b);
        matmul_init(cb_a, cb_b);
        tile_regs_acquire();
        matmul_tiles(cb_a, cb_b, n % T, n, 0);
        tile_regs_commit();
        tile_regs_wait();
        out.reserve_back(1);
        pack_tile(0, cb_out);
        out.push_back(1);
        tile_regs_release();
    }
}
