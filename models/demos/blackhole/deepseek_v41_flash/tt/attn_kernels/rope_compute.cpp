// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    compute_kernel_hw_startup(cb_x, cb_r, cb_out);
    CircularBuffer x(cb_x), r(cb_r), out(cb_out);
    x.wait_front(1);
    r.wait_front(1);
    reconfig_data_format(cb_x, cb_r);
    matmul_init(cb_x, cb_r);
    tile_regs_acquire();
    matmul_tiles(cb_x, cb_r, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    out.reserve_back(1);
    pack_tile(0, cb_out);
    out.push_back(1);
    tile_regs_release();
}
