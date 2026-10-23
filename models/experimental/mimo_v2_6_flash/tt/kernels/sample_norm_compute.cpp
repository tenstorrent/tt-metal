// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode sampling head, step 1, compute (tt/device_sampler.py): the elementwise sum over this core's logit tiles of
// exp((x - m) * a), m and a the per-row shift and scale tiles of the reader (sample_scan.cpp). fp32 in DST
// throughout (the inputs unpack straight to DST), the SFPU's exact exp. One tile out (cb_s), whose rows the writer
// sums (sample_scan_writer.cpp).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_m = get_compile_time_arg_val(1);
    constexpr uint32_t cb_a = get_compile_time_arg_val(2);
    constexpr uint32_t cb_s = get_compile_time_arg_val(3);
    const uint32_t n = get_arg_val<uint32_t>(0);

    // DST: 0 the tile in work, 1 the shift, 2 the scale, 3 the running sum.
    init_sfpu(cb_x, cb_s);
    cb_wait_front(cb_m, 1);
    cb_wait_front(cb_a, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_m);
    copy_tile(cb_m, 0, 1);
    copy_tile_to_dst_init_short(cb_a);
    copy_tile(cb_a, 0, 2);
    fill_tile_init();
    fill_tile(3, 0.0f);
    cb_wait_front(cb_x, n);
    for (uint32_t i = 0; i < n; ++i) {
        copy_tile_to_dst_init_short(cb_x);
        copy_tile(cb_x, i, 0);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        mul_binary_tile_init();
        mul_binary_tile(0, 2, 0);
        exp_tile_init();
        exp_tile(0);
        add_binary_tile_init();
        add_binary_tile(3, 0, 3);
    }
    tile_regs_commit();
    cb_pop_front(cb_x, n);
    cb_pop_front(cb_m, 1);
    cb_pop_front(cb_a, 1);

    cb_reserve_back(cb_s, 1);
    tile_regs_wait();
    pack_tile(3, cb_s);
    tile_regs_release();
    cb_push_back(cb_s, 1);
}
