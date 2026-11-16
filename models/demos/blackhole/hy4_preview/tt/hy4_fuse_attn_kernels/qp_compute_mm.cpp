// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B17 fused q-prep, absorb cores: out tile n (n < 2) = sum_{k<6} x_k w_{n*6+k}.  CT: 0 cb_x, 1 cb_w, 2 cb_out
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    compute_kernel_hw_startup(cb_x, cb_w, cb_out);
    CircularBuffer x(cb_x), w(cb_w), o(cb_out);
    x.wait_front(6);
    w.wait_front(12);
    reconfig_data_format(cb_x, cb_w);
    matmul_init(cb_x, cb_w);
    tile_regs_acquire();
    for (uint32_t n = 0; n < 2; ++n) {
        for (uint32_t k = 0; k < 6; ++k) {
            matmul_tiles(cb_x, cb_w, k, n * 6 + k, n);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    o.reserve_back(2);
    pack_tile(0, cb_out, 0);
    pack_tile(1, cb_out, 1);
    o.push_back(2);
    tile_regs_release();
    x.pop_front(6);
    w.pop_front(12);
}
