// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The activation stream and the output writer of the C++ down projection (see tt/cpp_down.py).
// Runs on the other RISC / NoC from the weight stream: every core reads the whole [MT x KT]
// activation one K block at a time, then writes its MT x PN output tiles one MT x CT subblock at a time, in the packed
// order.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t n0 = get_arg_val<uint32_t>(2);

    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    constexpr uint32_t PN = get_compile_time_arg_val(3);
    constexpr uint32_t KT = get_compile_time_arg_val(4);
    constexpr uint32_t NT = get_compile_time_arg_val(5);
    constexpr uint32_t CT = get_compile_time_arg_val(6);
    constexpr auto ax = TensorAccessorArgs<7>();
    constexpr auto ay = TensorAccessorArgs<ax.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(ax, x_addr);
    const auto sy = TensorAccessor(ay, y_addr);

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_y = 16;
    const uint32_t x_bytes = get_tile_size(cb_x);
    const uint32_t y_bytes = get_tile_size(cb_y);

    for (uint32_t b = 0; b < NB; ++b) {
        cb_reserve_back(cb_x, MT * KB);
        uint32_t l1 = get_write_ptr(cb_x);
        for (uint32_t r = 0; r < MT; ++r) {
            uint32_t t = r * KT + b * KB;
            for (uint32_t k = 0; k < KB; ++k) {
                noc_async_read_page(t + k, sx, l1);
                l1 += x_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, MT * KB);
    }

    for (uint32_t c0 = 0; c0 < PN; c0 += CT) {
        cb_wait_front(cb_y, MT * CT);
        uint32_t l1 = get_read_ptr(cb_y);
        for (uint32_t r = 0; r < MT; ++r) {
            for (uint32_t c = 0; c < CT; ++c) {
                noc_async_write_page(r * NT + n0 + c0 + c, sy, l1);
                l1 += y_bytes;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_y, MT * CT);
    }
}
