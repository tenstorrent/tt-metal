// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Packed residual add, compute (experts/stream.py: PackedResidualAdd): out = residual + cb_p tile by tile (cb_p is
// zero outside row 0), packed straight into the output residual shard (cb_out is backed by the output tensor).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_res = get_compile_time_arg_val(0);
    constexpr uint32_t cb_p = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t tiles = get_compile_time_arg_val(3);

    compute_kernel_hw_startup(cb_res, cb_p, cb_out);
    add_init(cb_res, cb_p);
    cb_wait_front(cb_res, tiles);
    cb_wait_front(cb_p, tiles);
    cb_reserve_back(cb_out, tiles);
    tile_regs_acquire();
    for (uint32_t t = 0; t < tiles; ++t) {
        add_tiles(cb_res, cb_p, t, t, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < tiles; ++t) {
        pack_tile(t, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, tiles);
    cb_pop_front(cb_p, tiles);
    cb_pop_front(cb_res, tiles);
}
